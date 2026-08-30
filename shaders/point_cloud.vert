#version 410 core

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_point_size;
uniform float u_depth_falloff;  // 0 = no brightness falloff, >0 darkens distant points
uniform vec3 u_clip_min;  // world-space AABB clip minimum (pre-model)
uniform vec3 u_clip_max;  // world-space AABB clip maximum (pre-model)

// Depth of field — each splat's screen-space size grows with its
// distance from the focal plane, so out-of-focus points become
// larger soft blobs. u_dof_focus_dist auto-tracks camera.distance
// (the orbit pivot) so whatever you're looking at stays sharp.
// u_dof_strength = 0 disables the effect entirely.
uniform float u_dof_strength;
uniform float u_dof_focus_dist;

// Clip-plane proximity coloring: per-axis width of the tinting band.
// Points within this distance of an active clip face get tinted toward
// that axis's color. Zero = no tinting on that axis.
uniform vec3 u_clip_near_zone;

in vec3 in_position;
in vec3 in_color;
in int in_label;
in float in_selected;

out vec3 v_color;
flat out int v_label;
out float v_selected;
out float v_clipped;
out float v_depth_factor;  // per-point brightness multiplier from depth falloff
out float v_coc_factor;    // DOF size multiplier — fragment darkens by 1/k
out vec3 v_clip_proximity; // per-axis 0..1 nearness to clip face (1 = on face)

void main() {
    // Test against clip AABB in model-local space (pre-model-transform).
    // A point outside the box is flagged for the fragment shader to discard.
    bool outside = any(lessThan(in_position, u_clip_min))
                || any(greaterThan(in_position, u_clip_max));
    v_clipped = outside ? 1.0 : 0.0;

    // Clip-plane proximity: for each axis, compute how close this point
    // is to the nearest active clip face. The near-zone width is in
    // world units; a value of 0 means no clip on that axis.
    v_clip_proximity = vec3(0.0);
    for (int a = 0; a < 3; a++) {
        float zone = u_clip_near_zone[a];
        if (zone > 0.0) {
            float d_lo = in_position[a] - u_clip_min[a];
            float d_hi = u_clip_max[a] - in_position[a];
            float nearest = min(d_lo, d_hi);
            v_clip_proximity[a] = 1.0 - clamp(nearest / zone, 0.0, 1.0);
        }
    }

    vec4 view_pos = u_view * u_model * vec4(in_position, 1.0);
    gl_Position = u_projection * view_pos;

    // Standard perspective point size: roughly constant on-screen size at
    // moderate zoom via the (300/dist) attenuation rule.
    float dist = max(length(view_pos.xyz), 0.1);
    gl_PointSize = u_point_size * (300.0 / dist);

    // Circle of confusion — out-of-focus splats inflate linearly
    // with their absolute offset from the focal plane, normalised
    // by the focal distance so the effect is zoom-invariant.
    //
    // The fragment shader divides alpha by v_coc_factor² so total
    // emitted light is preserved: a splat that's 4× bigger becomes
    // 16× fainter per pixel, matching real optical defocus where
    // a point's energy spreads over its CoC area.
    float coc_factor = 1.0;
    if (u_dof_strength > 0.0) {
        float coc = abs(dist - u_dof_focus_dist) / max(u_dof_focus_dist, 1e-3);
        coc_factor = 1.0 + u_dof_strength * coc;
        gl_PointSize *= coc_factor;
    }
    v_coc_factor = coc_factor;

    // Depth-based brightness falloff: darker with distance.
    // u_depth_falloff = 0 -> factor is 1 everywhere (no cue).
    // Higher values pull far points towards black faster.
    v_depth_factor = 1.0 / (1.0 + u_depth_falloff * dist * 0.01);

    // Slightly enlarge selected points so they pop
    if (in_selected > 0.5) {
        gl_PointSize *= 1.4;
    }

    v_color = in_color;
    v_label = in_label;
    v_selected = in_selected;
}
