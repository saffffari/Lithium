#version 430 core

// Gallery cache blit. Passthrough sample of the cached gallery RT into
// scene_rt. Used once per frame to draw the whole Contact Sheets area
// from the persistent cache. Per-cell re-renders are handled upstream.

in vec2 v_uv;
out vec4 frag;

uniform sampler2D u_src;

void main() {
    frag = texture(u_src, v_uv);
}
