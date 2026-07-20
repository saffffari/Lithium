@echo off
REM 3Photon launcher (Windows) — uses this repo's .venv python, resolved
REM relative to the script so it works wherever the repo lives (incl. worktrees).
set "HERE=%~dp0"
if exist "%HERE%.venv\Scripts\python.exe" (
  "%HERE%.venv\Scripts\python.exe" -m src.main %*
) else (
  python -m src.main %*
)
