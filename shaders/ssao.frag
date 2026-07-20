#version 430 core

// Screen-space ambient occlusion.
//
// Reads the scene depth buffer, reconstructs view-space position +
// normal per pixel (from depth derivatives — no G-buffer needed), then
// samples a hemisphere kernel rotated per-pixel by a small noise
// texture. Output is a single-channel occlusion factor in [0, 1] where
// 1 = fully lit, 0 = fully occluded. The tonemap pass multiplies the
// HDR scene by this before ACES so contact darkening reads through
// the curve naturally.
//
// Tuned for the clinical "scientific instrument readout" look: small
// radius (millimetres of view-space), gentle power, partial strength.
// The goal is shape readability — endplate edges, pedicle attachment
// crevices — not a heavy "rendered" look.

in vec2 v_uv;
out float frag_ao;

uniform sampler2D u_depth;
uniform sampler2D u_noise;

// Camera matrices — needed to reconstruct view-space position from
// NDC depth and to project sample points back to screen UV.
uniform mat4 u_projection;
uniform mat4 u_inv_projection;
// Viewport size in pixels / noise texture size (4×4) — tiles the noise
// across the screen so each pixel rotates the kernel uniquely.
uniform vec2 u_noise_scale;

// Hemisphere kernel — random unit vectors in the +Z half-space, magnitude
// weighted toward the origin so most samples are close to the fragment.
// Generated on CPU at init time (see _generate_ssao_kernel in main.py).
#define KERNEL_SIZE 32
uniform vec3 u_kernel[KERNEL_SIZE];

uniform float u_radius;      // sample radius in view-space units (mm in our case)
uniform float u_bias;        // small depth bias to prevent self-occlusion on flat surfaces
uniform float u_power;       // pow() applied to the AO factor — bigger = darker contact
uniform float u_strength;    // 0 = no AO, 1 = full AO (interpolates the final factor toward 1.0)

vec3 view_position_from_depth(vec2 uv, float depth) {
    // NDC z is [-1, 1] in OpenGL; raw depth texture sample is [0, 1].
    vec3 ndc = vec3(uv * 2.0 - 1.0, depth * 2.0 - 1.0);
    vec4 view = u_inv_projection * vec4(ndc, 1.0);
    return view.xyz / view.w;
}

void main() {
    float depth = texture(u_depth, v_uv).r;
    // Far-plane / cleared pixels (sidebar, padding, background) — no
    // surface to occlude. Return fully lit so the tonemap multiply is
    // a no-op there.
    if (depth >= 0.9999) {
        frag_ao = 1.0;
        return;
    }

    vec3 view_pos = view_position_from_depth(v_uv, depth);

    // Reconstruct view-space normal from screen-space derivatives of
    // the reconstructed position. ddx/ddy are the per-fragment partial
    // derivatives — cross(ddx, ddy) yields a normal perpendicular to
    // the local surface. The order (dpdy × dpdx) gives an outward
    // normal (toward +Z = toward camera) in our right-handed view.
    vec3 dpdx = dFdx(view_pos);
    vec3 dpdy = dFdy(view_pos);

    // Silhouette guard. At a depth discontinuity (bone edge against
    // background, or two bones meeting at an occlusion boundary) the
    // position derivatives jump by the full scene depth scale — the
    // reconstructed normal there is essentially random, and the
    // hemisphere kernel samples would land in arbitrary directions
    // producing a noisy halo around bone edges. Detect the
    // discontinuity by comparing the derivative magnitude against the
    // AO radius: any per-pixel position change larger than 2× the
    // sampling radius can't be a continuous surface, so skip AO.
    if (length(dpdx) > u_radius * 2.0
            || length(dpdy) > u_radius * 2.0) {
        frag_ao = 1.0;
        return;
    }

    vec3 normal = normalize(cross(dpdy, dpdx));

    // Random tangent vector for this pixel — tiles a 4×4 noise texture
    // across the screen so adjacent pixels rotate the kernel differently
    // and the noise can be cleaned up by the blur pass.
    vec3 random_vec = texture(u_noise, v_uv * u_noise_scale).xyz;

    // Build a tangent basis around the normal using Gram-Schmidt on the
    // random vector. tangent ⊥ normal, bitangent ⊥ both. The TBN matrix
    // rotates hemisphere samples into view space at the fragment's
    // orientation.
    vec3 tangent = normalize(random_vec - normal * dot(random_vec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 tbn = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        // Place sample in view space at this fragment.
        vec3 sample_view = view_pos + tbn * u_kernel[i] * u_radius;

        // Project the sample back to screen UV to read the actual
        // scene depth at that screen position.
        vec4 sample_clip = u_projection * vec4(sample_view, 1.0);
        vec3 sample_ndc = sample_clip.xyz / sample_clip.w;
        vec2 sample_uv = sample_ndc.xy * 0.5 + 0.5;

        // Skip samples that fall off-screen — no occluder info there.
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 ||
            sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        // Reconstruct the occluder's actual view-space z at that UV.
        float occluder_depth = texture(u_depth, sample_uv).r;
        vec3 occluder_view = view_position_from_depth(sample_uv, occluder_depth);

        // Range check: only count occluders within `radius` of the
        // fragment. Falls off smoothly past the radius so AO doesn't
        // produce hard halos at the edge of large objects.
        float depth_diff = abs(view_pos.z - occluder_view.z);
        float range = smoothstep(0.0, 1.0, u_radius / max(depth_diff, 0.001));

        // Occluder is "in front" of the sample (closer to camera) when
        // its view-space z is greater than the sample's z. View space
        // here is right-handed with camera looking down -Z, so closer
        // = larger z.
        if (occluder_view.z >= sample_view.z + u_bias) {
            occlusion += range;
        }
    }

    // 1 = fully lit, 0 = fully occluded. Pow shapes the falloff (bigger
    // = more concentrated darkening in tight crevices).
    occlusion = 1.0 - (occlusion / float(KERNEL_SIZE));
    occlusion = pow(clamp(occlusion, 0.0, 1.0), u_power);

    // Strength dial: 0 disables AO entirely, 1 applies the raw factor.
    // Keep this low for the clinical look — heavy AO reads as "rendered",
    // not "scientific".
    occlusion = mix(1.0, occlusion, u_strength);
    frag_ao = occlusion;
}
