"""Create a desktop shortcut for 3Photon.exe."""

import os
import sys


def create_shortcut():
    exe_path = os.path.abspath(os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'dist', '3Photon.exe'
    ))
    icon_path = os.path.abspath(os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'assets', '3photon.ico'
    ))
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    shortcut_path = os.path.join(desktop, '3Photon.lnk')

    if not os.path.exists(exe_path):
        print(f"EXE not found: {exe_path}")
        print("Run the build first: python -m PyInstaller 3photon.spec --noconfirm")
        return

    try:
        # Use PowerShell to create the shortcut
        import subprocess
        ps_script = f'''
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut("{shortcut_path}")
$sc.TargetPath = "{exe_path}"
$sc.WorkingDirectory = "{os.path.dirname(exe_path)}"
$sc.IconLocation = "{icon_path}"
$sc.Description = "3Photon Point Cloud Visualizer"
$sc.Save()
'''
        subprocess.run(['powershell', '-Command', ps_script], check=True)
        print(f"Desktop shortcut created: {shortcut_path}")
    except Exception as e:
        print(f"Failed to create shortcut: {e}")


if __name__ == '__main__':
    create_shortcut()
