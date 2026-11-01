@echo off
rem Double-click installer for Easy BOS (current user, no administrator rights needed).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if errorlevel 1 (
  echo.
  echo Installation failed. See the messages above.
)
echo.
pause
