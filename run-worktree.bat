@echo off
REM Launcher for the overnight/audit-fixes worktree.
REM
REM Sets THREEPHOTON_LIBRARY_DIR so this build can't accidentally
REM read or write the user's installed ~/.3photon/library/ catalog.
REM See WORKTREE_SETUP.md for the rationale.

setlocal
set THREEPHOTON_LIBRARY_DIR=%USERPROFILE%\.3photon-overnight\library
echo [worktree] catalog -> %THREEPHOTON_LIBRARY_DIR%

REM Activate the parent install's venv (lighter than maintaining a
REM second one per worktree). Falls back to the system python if the
REM venv isn't reachable.
if exist "D:\3Photon\.venv\Scripts\python.exe" (
  D:\3Photon\.venv\Scripts\python.exe -m src.main %*
) else (
  python -m src.main %*
)
endlocal
