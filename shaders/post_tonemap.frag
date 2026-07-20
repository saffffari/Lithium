#version 430 core

// Final display pass: HDR -> AO multiply -> ACES -> gamma 2.2.
//
// AO is applied PRE-tonemap so contact darkening reads through the
// ACES curve naturally rather than crushing already-tonemapped pixels
// (which would clip black). When SSAO is disabled the app binds a 1×1
// white texture to u_ao so the multiply is a no-op and the curve is
// unchanged.

in vec2 v_uv;
out vec4 frag;

uniform sampler2D u_hdr;
uniform sampler2D u_ao;
uniform float u_exposure;

// ACES filmic tonemap (Narkowicz 2015 approximation). Input is scene
// radiance in linear space; output is approximately sRGB after the
// gamma correction below.
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 c = texture(u_hdr, v_uv).rgb;
    float ao = texture(u_ao, v_uv).r;
    // Guard against NaN / negative inputs from anything upstream.
    c = max(c, vec3(0.0));
    c *= ao;
    c *= u_exposure;
    c = aces(c);
    c = pow(c, vec3(1.0 / 2.2));
    frag = vec4(c, 1.0);
}
