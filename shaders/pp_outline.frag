#version 430 core

// Selection silhouette outline.
//
// Reads the selection mask (1.0 = pixel belongs to a selected mesh,
// 0.0 = empty) and draws a 2-pixel orange stroke along the OUTER
// boundary of the mask region. Outer-only so the body of the mesh
// stays untouched and only the silhouette gets the accent — matches
// Blender's selection outline behaviour.
//
// Edge detection: a pixel is on the outer outline iff its own mask
// value is below threshold AND any neighbor within `radius` is above
// threshold. Sampling at radii 1 and 2 yields a 2-pixel-wide stroke.
// Output is RGBA with alpha = 1 on outline pixels and 0 elsewhere —
// the main loop blends this onto the tonemapped scene with standard
// alpha blending.

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_mask;
uniform vec2 u_texel_size;
uniform vec3 u_outline_color;   // RGB in linear space; default warm orange
uniform float u_outline_alpha;  // final composite alpha (1.0 = fully opaque)

void main() {
    float center = texture(u_mask, v_uv).r;

    // Only consider pixels OUTSIDE the mask as outline candidates —
    // gives an outer outline (stroke surrounds the mesh, doesn't tint
    // the mesh itself).
    if (center >= 0.5) {
        frag_color = vec4(0.0);
        return;
    }

    // Sample neighbors at distances 1 and 2 in screen space. If any
    // of them is inside the mask, this pixel is on the outer edge.
    vec2 t = u_texel_size;

    float n1 = 0.0;
    n1 = max(n1, texture(u_mask, v_uv + vec2(-t.x, 0.0)).r);
    n1 = max(n1, texture(u_mask, v_uv + vec2( t.x, 0.0)).r);
    n1 = max(n1, texture(u_mask, v_uv + vec2(0.0, -t.y)).r);
    n1 = max(n1, texture(u_mask, v_uv + vec2(0.0,  t.y)).r);

    float n2 = 0.0;
    vec2 t2 = t * 2.0;
    n2 = max(n2, texture(u_mask, v_uv + vec2(-t2.x, 0.0)).r);
    n2 = max(n2, texture(u_mask, v_uv + vec2( t2.x, 0.0)).r);
    n2 = max(n2, texture(u_mask, v_uv + vec2(0.0, -t2.y)).r);
    n2 = max(n2, texture(u_mask, v_uv + vec2(0.0,  t2.y)).r);
    // Diagonal taps at radius 2 — without these the outline thins at
    // 45° silhouette tangents where the cardinal axes miss the mask.
    n2 = max(n2, texture(u_mask, v_uv + vec2(-t2.x, -t2.y)).r);
    n2 = max(n2, texture(u_mask, v_uv + vec2( t2.x, -t2.y)).r);
    n2 = max(n2, texture(u_mask, v_uv + vec2(-t2.x,  t2.y)).r);
    n2 = max(n2, texture(u_mask, v_uv + vec2( t2.x,  t2.y)).r);

    float n = max(n1, n2);
    if (n < 0.5) {
        frag_color = vec4(0.0);
        return;
    }

    // Inside the outline band. Inner pixel (radius 1) at full
    // intensity, outer pixel (radius 2) at slightly reduced alpha for
    // a hint of anti-aliased softness on the outer edge.
    float alpha = (n1 >= 0.5) ? 1.0 : 0.55;
    frag_color = vec4(u_outline_color, alpha * u_outline_alpha);
}
