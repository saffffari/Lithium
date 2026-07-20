#version 430 core

// Infinite adaptive grid vertex shader.
//
// Emits a large quad in the XZ plane, centered on the camera's
// ground-projection so the quad always extends past the visible
// horizon regardless of where the user pans. The actual grid lines
// are computed in the fragment shader from world-space XZ coords,
// so the spacing is resolution-independent and free of popping.

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_y_plane;     // Y of the ground plane in world space
uniform vec3 u_camera_pos;   // world-space eye position
uniform float u_extent;      // quad half-size in world units

in vec2 in_pos;              // unit quad vertex in [-1, 1]^2

out vec3 v_world;

void main() {
    vec3 world = vec3(
        u_camera_pos.x + in_pos.x * u_extent,
        u_y_plane,
        u_camera_pos.z + in_pos.y * u_extent
    );
    v_world = world;
    gl_Position = u_proj * u_view * vec4(world, 1.0);
}
