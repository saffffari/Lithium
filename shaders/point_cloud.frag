#version 410 core

uniform float u_brightness;       // -1.0 to 1.0, default 0.0
uniform float u_contrast;         // 0.0 to 3.0, default 1.0
uniform float u_saturation;       // 0.0 to 3.0, default 1.0
uniform vec3 u_rgb_gain;          // per-channel multiplier, default (1,1,1)
uniform sampler2D u_label_colors; // 256x1 RGBA label color LUT
uniform float u_label_blend;      // 0.0 = vertex color, 1.0 = label color
uniform sampler2D u_falloff_lut;  // 256x1 R float envelope sampled by point-coord distance
uniform int u_has_falloff = 0;    // 1 = sample u_falloff_lut, 0 = flat (no inner LUT)

in vec3 v_color;
flat in int v_label;
in float v_selected;
in float v_clipped;
in float v_depth_factor;
in float v_coc_factor;
in vec3 v_clip_proximity;
out vec4 frag_color;

void main() {
    // Discard points outside the clip box (set in the vertex shader)
    if (v_clipped > 0.5) {
        discard;
    }

    // Map gl_PointCoord from [0,1] to [-1,1]
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float dist_sq = dot(coord, coord);

    // Hard discard outside unit circle. This is the key to spatial
    // legibility on sparse clouds: every surviving fragment writes
    // alpha = 1 below, so near points OCCLUDE far points instead of
    // blending through them. Occlusion is the strongest monocular
    // depth cue in human vision — without this fix the soft splat
    // edges let background points "ghost" through foreground points
    // and the cloud reads as a transparent fog.
    if (dist_sq > 1.0) {
        discard;
    }

    float dist = sqrt(dist_sq);

    // Optional inner brightness modulation from the falloff LUT.
    // The LUT shapes how the splat *looks* (sharper centre vs flat
    // disc) without affecting alpha — alpha is forced to 1 below so
    // the splat occludes correctly. Floor at 0.5 so edges never go
    // fully dark and never produce a black halo against the bg.
    //
    // ``u_has_falloff`` defaults to 0 (declared above) so any render
    // path that forgets to bind a falloff texture gets flat lum=1
    // instead of sampling whatever junk is currently on texture unit
    // 1. See UNI-003 in REPORT.md.
    float lum = 1.0;
    if (u_has_falloff != 0) {
        float lut = texture(u_falloff_lut, vec2(dist, 0.5)).r;
        lum = mix(0.55, 1.0, lut);
    }

    // Color adjustments on vertex color
    vec3 color = v_color;
    color = color + u_brightness;
    color = (color - 0.5) * u_contrast + 0.5;
    color *= u_rgb_gain;
    float gray = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3(gray), color, u_saturation);
    // Clamp before LUT/label mixing so negative intermediates from
    // brightness/contrast don't produce incorrect blends.
    color = max(color, vec3(0.0));

    // Apply the LUT-driven inner brightness
    color *= lum;

    // Clip-plane proximity coloring — tint toward the dominant axis color.
    // Axis colors match the overlay renderer: X=coral, Y=green, Z=blue.
    const vec3 clip_colors[3] = vec3[3](
        vec3(0.95, 0.35, 0.35),   // X axis (R)
        vec3(0.35, 0.90, 0.45),   // Y axis (S)
        vec3(0.45, 0.65, 0.98)    // Z axis (A)
    );
    float max_prox = max(v_clip_proximity.x, max(v_clip_proximity.y, v_clip_proximity.z));
    if (max_prox > 0.0) {
        // Find dominant axis and blend toward its color
        int dom = (v_clip_proximity.x >= v_clip_proximity.y && v_clip_proximity.x >= v_clip_proximity.z) ? 0
                : (v_clip_proximity.y >= v_clip_proximity.z) ? 1 : 2;
        color = mix(color, clip_colors[dom], max_prox);
    }

    // Label color lookup — clamp to valid LUT range to avoid undefined
    // texelFetch behaviour with out-of-range label ids.
    int safe_label = clamp(v_label, 0, 255);
    vec4 label_color = texelFetch(u_label_colors, ivec2(safe_label, 0), 0);
    if (u_label_blend > 0.0 && label_color.a <= 0.001) {
        discard;
    }
    color = mix(color, label_color.rgb, u_label_blend * label_color.a);

    // Selection highlight: solid orange fill with a brighter outer ring.
    // Branchless to avoid nested warp divergence.
    vec3 sel_fill = vec3(1.00, 0.55, 0.10);
    vec3 sel_ring = vec3(1.0, 0.80, 0.35);
    vec3 sel_color = mix(sel_fill, sel_ring, step(0.78, dist));
    color = mix(color, sel_color, step(0.5, v_selected));

    // Depth-based brightness falloff (from vertex shader).
    color *= v_depth_factor;
    // Linear HDR output — do not clamp to [0,1]. The final tonemap
    // pass (post_tonemap.frag) handles display mapping. Guard against
    // negative values so ACES input stays in its domain.
    color = max(color, vec3(0.0));

    // DOF darkening: out-of-focus splats are inflated in the vertex
    // shader (coc_factor > 1) but kept fully opaque so they occlude
    // correctly — no transparent stacking. Instead we darken the
    // color proportionally: a splat inflated 4× becomes 4× dimmer,
    // producing solid darker circles that read as defocused without
    // ghosting or blending artifacts.
    float dof_dim = 1.0 / max(v_coc_factor, 1.0);
    color *= dof_dim;
    float alpha = 1.0;

    frag_color = vec4(color, alpha);
}
