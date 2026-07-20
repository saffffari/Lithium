#!/usr/bin/env bash
# Install a Linux desktop launcher for 3Photon (app menu + ~/Desktop).
# Paths are resolved relative to this repo, so it works wherever the repo lives.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS"

read -r -d '' ENTRY <<EOF || true
[Desktop Entry]
Type=Application
Version=1.0
Name=3Photon
GenericName=Point Cloud Workstation
Comment=Visualize, annotate, and train on point clouds
Exec=$REPO/run.sh %F
Icon=$REPO/assets/3photon.png
Terminal=false
Categories=Graphics;3DGraphics;
StartupNotify=true
StartupWMClass=3 P H O T O N
EOF

DEST="$APPS/3photon.desktop"
printf '%s\n' "$ENTRY" > "$DEST"
chmod +x "$DEST"
echo "installed: $DEST"

# Also place one on the Desktop, if there is one.
DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESK" ]; then
  cp "$DEST" "$DESK/3photon.desktop"
  chmod +x "$DESK/3photon.desktop"
  # KDE/GNOME: mark the desktop file as trusted so it launches on click.
  gio set "$DESK/3photon.desktop" metadata::trusted true 2>/dev/null || true
  echo "installed: $DESK/3photon.desktop"
fi

update-desktop-database "$APPS" 2>/dev/null || true
echo "done — 3Photon should appear in your application launcher."
