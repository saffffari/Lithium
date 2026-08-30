#version 410 core

// Selection-mask fragment shader.
//
// Writes 1.0 to a single-channel mask buffer wherever a selected mesh
// is visible (depth-tested against the shared scene depth so occluded
// fragments are correctly excluded). The outline post-process samples
// this mask, detects outer boundary pixels, and draws the orange
// silhouette stroke on top of the tonemapped backbuffer.

out float frag_mask;

void main() {
    frag_mask = 1.0;
}
