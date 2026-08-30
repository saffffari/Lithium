#version 410 core

// Directional Lambert shading using the interpolated per-vertex normal.
// Camera-locked light keeps the lighting consistent regardless of how
// the user rotates the model.
//
// History — why this isn't matcap anymore:
// A matcap implementation lived here briefly. It compressed dynamic
// range relative to live Lambert and the bone-palette rim term in the
// procedural matcap read as a "glow halo" on bone silhouettes — the
// surface stopped reading as anatomy and started reading as a sticker.
// Live Lambert against a camera-locked light direction restores the
// crisp directional shading the diagrammatic look depends on.
//
// Selection: NOT handled in this shader. Selected meshes get an orange
// silhouette outline via a separate post-process pass (see
// ``shaders/pp_outline.frag``) that reads a selection mask buffer
// populated by ``draw_mesh_mask``. Keeping the selection logic in the
// outline pass means the BODY of a selected mesh keeps its full
// curvature shading — the orange only appears as a thin stroke at
// the silhouette, not as a body tint that washes out shape.

in vec3 v_view_normal;
flat in int v_label;
in float v_clipped;
out vec4 frag_color;

// Per-draw colour tint. Multiplies the neutral grey surface so the
// same mesh can render twice in dual-pose mode: once at supine pose
// in default grey, once at standing pose in a blue tint. Default
// (1, 1, 1) preserves the diagrammatic grey when not in dual-pose
// mode. Tint is applied to the SURFACE colour before Lambert
// shading, so curvature reads correctly in either pose.
uniform vec3 u_pose_tint = vec3(1.0, 1.0, 1.0);

void main() {
    if (v_clipped > 0.5) {
        discard;
    }

    // Flip back-facing normals so two-sided rendering reads correctly
    // — closed meshes shouldn't expose backs, but inside-out triangles
    // from a noisy Poisson reconstruction would otherwise produce
    // dark patches.
    vec3 n = v_view_normal;
    if (n.z < 0.0) n = -n;

    // Directional light fixed in VIEW space — upper-right-front. As
    // the user rotates the model the highlight stays in the same
    // viewer-relative position, so no surface goes pitch black just
    // because the user turned the spine sideways.
    //
    // Ambient kept low (0.12) so directional shading actually carries
    // visible information. With higher ambient (0.30+) the surface
    // washes out to near-uniform tone and the geometry stops reading
    // clearly. Dynamic range shadow → lit ≈ 8× makes vertebra
    // curvature unambiguous.
    vec3 light_dir = normalize(vec3(0.30, 0.50, 0.85));
    float lambert = max(dot(n, light_dir), 0.0);
    float ambient = 0.12;
    float diffuse = 1.0 - ambient;
    float intensity = ambient + diffuse * lambert;

    // Neutral diagrammatic grey, tinted per draw to distinguish pose
    // contexts (supine vs standing) in dual-pose mode. Anatomical
    // class is communicated by the datum overlays drawn on top of
    // the mesh, not by per-face colouring or material warmth.
    vec3 surface = vec3(0.55, 0.55, 0.55) * u_pose_tint;
    vec3 color = surface * intensity;

    // Subtle silhouette rim — brighten fragments whose normal is
    // perpendicular to the view direction (z near 0). Pulls bones
    // away from the black background. Kept low coefficient (0.15)
    // and neutral grey tint so it doesn't reintroduce the wash-out
    // we just dialled out of the ambient + base colour, and doesn't
    // read as a "glow" — that earlier matcap problem.
    float rim_t = 1.0 - abs(n.z);
    float rim = pow(clamp(rim_t, 0.0, 1.0), 4.0) * 0.15;
    color += vec3(0.32, 0.32, 0.32) * rim;

    color = max(color, vec3(0.0));
    frag_color = vec4(color, 1.0);
}
