#version 430 core

// Minimal vertex shader for the selection-mask pass.
//
// Renders selected meshes into a single-channel mask buffer that the
// outline post-process reads to find silhouette edges. We only need
// the position transform — the mask fragment shader writes a constant
// 1.0, so no normal / label / clip-box state is needed here.

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

in vec3 in_position;

void main() {
    gl_Position = u_projection * u_view * u_model * vec4(in_position, 1.0);
}
