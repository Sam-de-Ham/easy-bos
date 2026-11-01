<#
.SYNOPSIS
    Builds the Easy BOS release package: dist\EasyBOS-<version>.zip (always) and
    installer\Output\EasyBOS-Setup-<version>.exe (if Inno Setup 6 is installed).
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$version = (Select-String -Path (Join-Path $root "src\easy_bos\__init__.py") -Pattern '__version__ = "(.+)"').Matches[0].Groups[1].Value
$payload = Join-Path $root "installer\payload"
Remove-Item -Recurse -Force $payload -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $payload "app") | Out-Null

foreach ($item in @("pyproject.toml", "uv.lock", ".python-version", "README.md", "LICENSE", "src")) {
    $src = Join-Path $root $item
    if (Test-Path $src) { Copy-Item -Recurse -Force $src (Join-Path $payload "app") }
}
Get-ChildItem -Recurse -Directory -Filter "__pycache__" (Join-Path $payload "app") | Remove-Item -Recurse -Force
Copy-Item (Join-Path $root "installer\install.ps1") $payload
Copy-Item (Join-Path $root "installer\Install Easy BOS.cmd") $payload
$uv = (Get-Command uv).Source
Copy-Item $uv (Join-Path $payload "uv.exe")

$dist = Join-Path $root "dist"
New-Item -ItemType Directory -Force -Path $dist | Out-Null
$zip = Join-Path $dist "EasyBOS-$version.zip"
Remove-Item -Force $zip -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $payload "*") -DestinationPath $zip
Write-Host "Package: $zip"

$iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if ($iscc) {
    & $iscc "/DAppVersion=$version" (Join-Path $root "installer\easy-bos.iss")
    Write-Host "Installer: installer\Output\EasyBOS-Setup-$version.exe"
} else {
    Write-Host "Inno Setup 6 not found: skipped the .exe installer (the zip works on its own)."
}
