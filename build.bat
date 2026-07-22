@echo off
echo Building Lithium...
call .venv\Scripts\activate
pyinstaller lithium.spec --noconfirm
echo.
echo Build complete. Executable at: dist\Lithium.exe
pause
