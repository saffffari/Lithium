@echo off
echo Building 3Photon...
call .venv\Scripts\activate
pyinstaller 3photon.spec --noconfirm
echo.
echo Build complete. Executable at: dist\3Photon.exe
pause
