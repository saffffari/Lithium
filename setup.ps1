# One-command Lithium setup for annotators (Windows, PowerShell 5+).
#
#   .\setup.ps1                        install + launch
#   .\setup.ps1 -Kit C:\path\to\x.li   also import a project bundle first
#   .\setup.ps1 -NoLaunch              install only
#
# Installs uv if missing, creates .venv on Python 3.12, installs the
# annotator requirements, optionally imports a .li, then launches.
# Re-running is safe; it only updates what changed.
#
# If PowerShell refuses to run scripts:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
param(
    [string]$Kit = "",
    [switch]$NoLaunch
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Say($msg) { Write-Host "[lithium] $msg" -ForegroundColor Yellow }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "installing uv (Python package manager)..."
    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not on PATH after install. Open a new PowerShell window and run .\setup.ps1 again."
}

Say "python 3.12 venv"
uv venv .venv --python 3.12 --quiet
Say "packages"
uv pip install --python .venv\Scripts\python.exe -r requirements-annotator.txt --quiet

if ($Kit -ne "") {
    if (-not (Test-Path $Kit)) { throw "no such file: $Kit" }
    Say "importing $Kit"
    & .venv\Scripts\python.exe tools\bundle_cli.py import $Kit
    if ($LASTEXITCODE -ne 0) { throw "import failed" }
}

Say "ready. Launch any time with run.bat"
if (-not $NoLaunch) {
    & .\run.bat
}
