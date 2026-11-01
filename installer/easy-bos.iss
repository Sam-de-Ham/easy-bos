; Inno Setup script for Easy BOS. Build with installer\build.ps1 (needs Inno Setup 6).
; The installer is small: it carries the application, uv.exe and the lock file, and installs
; the locked Python environment (incl. CUDA PyTorch) on the user's machine with install.ps1.

#define AppName "Easy BOS"
#ifndef AppVersion
  #define AppVersion "0.9.0-beta"
#endif

[Setup]
AppId={{6C3E0C55-0F7B-4C35-9C5B-0E0B8C1B2E11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Sam-de-Ham
DefaultDirName={localappdata}\EasyBOS
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=Output
OutputBaseFilename=EasyBOS-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#AppName}

[Files]
Source: "payload\*"; DestDir: "{app}\installer"; Flags: recursesubdirs ignoreversion

[Run]
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\install.ps1"" -InstallDir ""{app}"""; \
  StatusMsg: "Installing Python and libraries (about 3 GB on first install)..."; \
  Flags: waituntilterminated

[UninstallDelete]
Type: filesandordirs; Name: "{app}\env"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\cache"
Type: filesandordirs; Name: "{app}\app"
Type: files; Name: "{userprograms}\Easy BOS.lnk"
Type: files; Name: "{userdesktop}\Easy BOS.lnk"
