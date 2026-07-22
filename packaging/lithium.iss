; Inno Setup script for the Lithium Windows installer.
; Build: iscc packaging\lithium.iss   (after `pyinstaller lithium.spec`)
; Output: dist\LithiumSetup-<version>.exe

#define AppName "Lithium"
#define AppPublisher "2Photon Elements"
#define AppURL "https://2photon.io/elements/lithium"
#ifndef AppVersion
  #define AppVersion "1.1.0"
#endif

[Setup]
AppId={{7C3E1F52-2P0E-4L1T-H1UM-000000000003}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=LithiumSetup-{#AppVersion}
SetupIconFile=..\assets\lithium.ico
UninstallDisplayIcon={app}\Lithium.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; The app is 64-bit only (torch-free viz env, but numpy wheels are x64).
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\Lithium.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\Lithium.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\Lithium.exe"; \
    Tasks: desktopicon

[Run]
Filename: "{app}\Lithium.exe"; Description: "{cm:LaunchProgram,{#AppName}}"; \
    Flags: nowait postinstall skipifsilent
