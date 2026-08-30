#version 410 core

// Infinite adaptive grid fragment shader.
//
// Spacing is chosen from log10 of the camera's height above the plane,
// so the lines get finer as you zoom in and coarser as you zoom out.
// Both the current decade's minor/major lines and the next-coarser
// decade are drawn, cross-faded to hide the popping when a decade
// boundary is crossed. Lines are anti-aliased via fwidth().

uniform vec3 u_camera_pos;

in vec3 v_world;
out vec4 frag_color;

// Compute grid intensity in [0, 1] for a given world-space UV and
// spacing, using screen-space derivatives for anti-aliasing.
float grid_line(vec2 uv, float spacing) {
    vec2 c = uv / spacing;
    vec2 d = fwidth(c);
    vec2 a = abs(fract(c - 0.5) - 0.5) / max(d, vec2(1e-6));
    return 1.0 - clamp(min(a.x, a.y), 0.0, 1.0);
}

void main() {
    // Camera height above the plane drives the grid scale
    float cam_h = max(abs(u_camera_pos.y - v_world.y), 1e-3);

    // Pick the minor-line spacing so there are roughly 10 lines across
    // the viewport at the current zoom. Smoothly blend between decades.
    float log_h = log(cam_h) / log(10.0);
    float step_i = floor(log_h - 1.0);
    float blend = fract(log_h - 1.0);  // 0 = just switched to new decade

    float minor = pow(10.0, step_i);
    float major = minor * 10.0;
    float next  = major * 10.0;

    vec2 uv = v_world.xz;
    float g_minor = grid_line(uv, minor);
    float g_major = grid_line(uv, major);
    float g_next  = grid_line(uv, next);

    // Fade the finest (minor) lines in as you zoom in.
    // When blend is near 1 (just about to cross into finer), minor is
    // full strength. When blend is near 0 (just crossed), minor is weak.
    float minor_alpha = g_minor * (1.0 - blend) * 0.18;
    float major_alpha = g_major * 0.42;
    float next_alpha  = g_next  * 0.55;  // the "major" of the next decade

    float alpha = max(max(minor_alpha, major_alpha), next_alpha);

    // === Attenuation stack ===
    //
    // 1. Grazing-angle fade. At low viewing angles the grid lines pack
    //    into very few pixels and the AA'd intensity stacks into a
    //    bright band near the horizon. We use the ray direction's Y
    //    component as a cheap cos-of-angle proxy — vertical view (top
    //    down) is full strength, horizontal view (edge on) is zero.
    vec3 ray = v_world - u_camera_pos;
    float ray_len = max(length(ray), 1e-6);
    float cos_view = abs(ray.y / ray_len);
    float angle_fade = smoothstep(0.0, 0.35, cos_view);

    // 2. Distance fade. Much tighter than before — the grid softens
    //    within a few camera-heights rather than extending to the
    //    vanishing point. Exponential so there's no hard cutoff.
    float dist_ratio = ray_len / max(cam_h, 1e-3);
    float dist_fade = exp(-dist_ratio * 0.08);

    alpha *= angle_fade * dist_fade;

    if (alpha < 0.003) discard;

    frag_color = vec4(0.5, 0.5, 0.5, alpha);
}
