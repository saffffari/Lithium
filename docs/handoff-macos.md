# macOS handoff — status and scope (updated 2026-08-30: the GL wall is gone)

For the agent (or human) working in this repo cloned on the Mac.
Short version: **it runs.** The renderer needed nothing beyond OpenGL 4.1 — the
`#version 430` lines and the 4.3 context hint were the only blockers, and every
shader now compiles under a strict 4.1 core context (verified with Mesa's
`MESA_GL_VERSION_OVERRIDE=4.1FC`). See `docs/macos.md` for the recipe.

## Why there is no macOS build yet

The renderer is ModernGL on an OpenGL **4.3 core** context (all 18
shaders are `#version 430`; context hints in `src/main.py:init_window`).
macOS froze OpenGL at 4.1 and deprecated it — a 4.3 context cannot be
created on any Mac. This is a hard platform limit, not a packaging gap.

The macOS path is the in-progress WebGPU renderer
(`prototypes/wgpu/` → future `src/render2/`), which runs on **Metal**
natively via wgpu. When render2 replaces the GL renderer, macOS becomes
a first-class target and the `.app`/`.dmg` job gets added to
`.github/workflows/release.yml`.

## What IS worth doing on the Mac now

1. **Data layer + tests** — everything below the renderer is
   platform-neutral:
   ```bash
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install -r requirements-lock.txt
   python -m pytest tests -q     # expect 217 passed, 3 skipped
   ```
   Failures here are real cross-platform bugs — fix and commit.
2. **wgpu spike sanity** — `pip install wgpu rendercanvas`, then run
   `prototypes/wgpu/demo.py` on any packed block (it selects Metal
   automatically). This validates the entire render2 direction on
   Apple silicon and is genuinely useful signal.
3. **Packaging groundwork** (don't ship yet): confirm PyInstaller
   produces a `.app` from `lithium.spec` structurally; note
   codesigning/notarization needs an Apple Developer ID before any
   public artifact (Gatekeeper blocks unsigned apps).
4. **CLI headless** — `python -m src.main render ...` will fail on GL
   context creation; confirm the failure is a clean error message, not
   a hang. Improve the message if needed ("macOS requires the Metal
   renderer, coming in the render2 milestone").

## Explicitly out of scope

- Porting shaders to 4.1 / downgrading the GL feature set.
- MoltenVK / ANGLE shims. The wgpu renderer makes them dead ends.
- Training: needs CUDA; Macs are view/annotate targets only.
