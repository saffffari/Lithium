"""Capture the three tabs of the real app on public VerSe bones (Gold247).

Launches ~/Lithium/run.sh with the --project/--cloud/--tab startup flags,
waits for the view to populate, grabs the screen with KDE's spectacle, crops
the desktop panel off and writes paper/figures/screenshot_<tab>.png.
Needs a live Wayland/X session on HAL; run with the app closed.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
PROJECT = os.environ.get("LITHIUM_SHOT_PROJECT", "Deepfield · Gold247 · 6-class [LOCKED]")
LOCK = pathlib.Path(os.path.expanduser("~/.lithium/library/.lock"))
SHOTS = {
    "sheets": ["--project", PROJECT, "--tab", "sheets"],
    "light": ["--project", PROJECT, "--cloud", os.environ.get("LITHIUM_SHOT_CLOUD", "3"), "--tab", "light"],
    "train": ["--project", PROJECT, "--tab", "train"],
}
WAIT = float(os.environ.get("LITHIUM_SHOT_WAIT", "18"))
PANEL_PX = int(os.environ.get("LITHIUM_SHOT_PANEL", "30"))  # desktop top bar to crop

FIG.mkdir(parents=True, exist_ok=True)
for name, args in SHOTS.items():
    LOCK.unlink(missing_ok=True)
    proc = subprocess.Popen([str(ROOT / "run.sh"), *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(WAIT)
    raw = FIG / f"_raw_{name}.png"
    subprocess.run(["spectacle", "-b", "-n", "-f", "-o", str(raw)], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    LOCK.unlink(missing_ok=True)
    if not raw.exists():
        print(f"{name}: NO SCREENSHOT"); continue
    im = Image.open(raw).convert("RGB")
    w, h = im.size
    im.crop((0, PANEL_PX, w, h)).save(FIG / f"screenshot_{name}.png", optimize=True)
    raw.unlink()
    print(f"{name}: {w}x{h - PANEL_PX} -> figures/screenshot_{name}.png")
