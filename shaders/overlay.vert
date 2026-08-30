#version 410 core

uniform mat4 u_mvp;

in vec3 in_position;
in vec4 in_color;

out vec4 v_color;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_color = in_color;
}
