<#
.SYNOPSIS
    Installs Easy BOS for the current user (no administrator rights needed).

.DESCRIPTION
    Copies the application next to a private Python environment managed by uv, installs the
    locked dependencies (including CUDA-enabled PyTorch, ~3 GB) and creates Start menu and
    desktop shortcuts. Re-running it updates an existing installation in place; only changed
    packages are downloaded.

.PARAMETER InstallDir
    Where to install. Default: %LOCALAPPDATA%\EasyBOS
.PARAMETER CacheDir
    uv download cache. Default: <InstallDir>\cache (kept so updates are fast).
.PARAMETER NoShortcuts
    Skip creating shortcuts (used by automated tests).
#>
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "EasyBOS"),
    [string]$CacheDir = "",
    [switch]$NoShortcuts
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$payload = Join-Path $here "app"
if (-not (Test-Path (Join-Path $payload "pyproject.toml"))) {
    # Running from the source tree: the repository root is the payload.
    $payload = Split-Path -Parent $here
}

function Step($text) { Write-Host "  - $text" -ForegroundColor Gray }

Write-Host ""
Write-Host "Installing Easy BOS into $InstallDir" -ForegroundColor White
if ($InstallDir -match "OneDrive") {
    Write-Warning "The install folder is inside OneDrive. Syncing ~30,000 library files slows everything down; consider the default location."
}

# 1. uv (bundled next to this script, or already on PATH)
$uv = Join-Path $here "uv.exe"
if (-not (Test-Path $uv)) {
    $found = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -eq $found) {
        throw "uv.exe was not found next to the installer or on PATH. Use the release package, which includes it."
    }
    $uv = $found.Source
}
Step "Using uv: $uv"

# 2. Application files
$appDir = Join-Path $InstallDir "app"
New-Item -ItemType Directory -Force -Path $appDir | Out-Null
Step "Copying application"
foreach ($item in @("pyproject.toml", "uv.lock", ".python-version", "README.md", "LICENSE", "src")) {
    $src = Join-Path $payload $item
    if (Test-Path $src) {
        Copy-Item -Recurse -Force $src $appDir
    }
}

# 3. Python environment, isolated inside the install folder
if ($CacheDir -eq "") { $CacheDir = Join-Path $InstallDir "cache" }
$env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallDir "python"
$env:UV_CACHE_DIR = $CacheDir
$env:UV_LINK_MODE = "copy"
$env:UV_PROJECT_ENVIRONMENT = Join-Path $InstallDir "env"
Step "Installing Python and libraries (first install downloads about 3 GB)"
Push-Location $appDir
try {
    & $uv sync --frozen --no-dev --python-preference only-managed
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed (uv exit code $LASTEXITCODE)." }
} finally {
    Pop-Location
}

$exe = Join-Path $env:UV_PROJECT_ENVIRONMENT "Scripts\easy-bos.exe"
if (-not (Test-Path $exe)) { throw "Installation incomplete: $exe is missing." }

# 4. Quick self-test: import the processing core and report the compute device
$py = Join-Path $env:UV_PROJECT_ENVIRONMENT "Scripts\python.exe"
$device = & $py -c "from easy_bos.core.device import describe; print(describe().replace(chr(183), '-'))"
Step "Compute device: $device"

# 5. Shortcuts and uninstaller
$uninstall = Join-Path $InstallDir "Uninstall Easy BOS.cmd"
Set-Content -Path $uninstall -Encoding ASCII -Value @"
@echo off
echo Removing Easy BOS from $InstallDir
del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Easy BOS.lnk" 2>nul
del "%USERPROFILE%\Desktop\Easy BOS.lnk" 2>nul
cd /d "%TEMP%"
rmdir /s /q "$InstallDir"
"@

if (-not $NoShortcuts) {
    $shell = New-Object -ComObject WScript.Shell
    foreach ($dir in @([Environment]::GetFolderPath("Programs"), [Environment]::GetFolderPath("Desktop"))) {
        $lnk = $shell.CreateShortcut((Join-Path $dir "Easy BOS.lnk"))
        $lnk.TargetPath = $exe
        $lnk.WorkingDirectory = $InstallDir
        $lnk.Description = "Background-oriented schlieren"
        $lnk.Save()
    }
    Step "Shortcuts created in the Start menu and on the desktop"
}

Write-Host ""
Write-Host "Easy BOS is installed. Start it from the Start menu or the desktop shortcut." -ForegroundColor Green
