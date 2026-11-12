# Cinegraph installer for Windows (PowerShell)
$ErrorActionPreference = "Stop"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  if (Get-Command winget -ErrorAction SilentlyContinue) { winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements }
  else { Write-Host "Install ffmpeg first: https://www.gyan.dev/ffmpeg/builds/ and add its bin folder to PATH"; exit 1 }
}
$venv = "$env:USERPROFILE\.cinegraph\venv"
py -3 -m venv $venv
& "$venv\Scripts\python.exe" -m pip install --upgrade pip
& "$venv\Scripts\python.exe" -m pip install -e "$PSScriptRoot\..[ai,full]"
Write-Host "`nDone. Run:  $venv\Scripts\cinegraph.exe info"
Write-Host "(Restart the terminal if ffmpeg was just installed.)"
