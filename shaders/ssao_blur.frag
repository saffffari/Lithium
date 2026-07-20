#version 430 core

// Bilateral blur for the raw SSAO buffer.
//
// The SSAO pass produces a low-sample-count noisy estimate per pixel
// (32 hemisphere samples per fragment, rotated by a 4×4 tiled noise
// texture). A small 5×5 blur with depth-aware sample rejection removes
// the noise pattern while preserving the AO contrast at depth
// discontinuities — without this, the AO reads as a fuzzy moiré
// pattern overlaid on the geometry instead of as anatomical contact
// darkening.
//
// Depth-aware rejection: any blur sample whose scene depth differs
// from the centre pixel's depth by more than u_depth_threshold is
// skipped. Prevents AO from bleeding across silhouette edges where
// one bone occludes another at a sharp boundary.

in vec2 v_uv;
out float frag_ao;

uniform sampler2D u_ssao;
uniform sampler2D u_depth;
uniform vec2 u_texel_size;
uniform float u_depth_threshold;

void main() {
    float center_depth = texture(u_depth, v_uv).r;
    // Far-plane pixels passed through SSAO as 1.0; no need to blur
    // them, just return the value.
    if (center_depth >= 0.9999) {
        frag_ao = texture(u_ssao, v_uv).r;
        return;
    }

    float ao_sum = 0.0;
    float weight_sum = 0.0;

    const int RADIUS = 2;  // 5×5 kernel
    for (int y = -RADIUS; y <= RADIUS; ++y) {
        for (int x = -RADIUS; x <= RADIUS; ++x) {
            vec2 offset = vec2(float(x), float(y)) * u_texel_size;
            vec2 sample_uv = v_uv + offset;
            float sample_depth = texture(u_depth, sample_uv).r;
            // Reject samples across a depth discontinuity so AO stays
            // crisp at bone edges.
            if (abs(sample_depth - center_depth) > u_depth_threshold) {
                continue;
            }
            ao_sum += texture(u_ssao, sample_uv).r;
            weight_sum += 1.0;
        }
    }

    frag_ao = (weight_sum > 0.0)
        ? ao_sum / weight_sum
        : texture(u_ssao, v_uv).r;
}
