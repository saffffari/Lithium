#version 410 core

// Vertex shader for the surface mesh. Passes the view-space normal to
// the fragment shader for Lambert-style directional shading.
//
// We tried per-face flat shading via dFdx/dFdy of the view-space
// position in the fragment shader, but that produced uniformly-white
// results on multi-vertebra scenes (likely a precision issue when
// view-space coordinates have large magnitudes and triangles span only
// a few pixels each). The per-vertex normal path is older, slower (one
// dot product per fragment instead of per triangle), but rock-solid.

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_clip_min;
uniform vec3 u_clip_max;

in vec3 in_position;
in vec3 in_normal;
in int  in_label;

out vec3 v_view_normal;
flat out int v_label;
out float v_clipped;

void main() {
    // Clip AABB in model-local space matches the point shader exactly.
    bool outside = any(lessThan(in_position, u_clip_min))
                || any(greaterThan(in_position, u_clip_max));
    v_clipped = outside ? 1.0 : 0.0;

    vec4 view_pos = u_view * u_model * vec4(in_position, 1.0);

    // Transform normal into view space using the upper-left 3x3 of
    // (view * model). Assumes uniform scaling in the model transform
    // (true for our rigid + uniform-scale per-cloud transforms). If
    // non-uniform scale ever lands here the right fix is a normal
    // matrix uniform.
    mat3 vm3 = mat3(u_view * u_model);
    v_view_normal = normalize(vm3 * in_normal);

    gl_Position = u_projection * view_pos;
    v_label = in_label;
}
