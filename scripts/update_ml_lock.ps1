<#
.SYNOPSIS
    Reproduce or regenerate the hash-pinned ML requirements lock.

.DESCRIPTION
    Builds a clean Python 3.12.13 venv with uv, installs the current
    ml/requirements.lock with --require-hashes, runs ml/tests, freezes the
    installed versions, downloads the matching Windows x64 wheels, and
    re-derives the hashes. By default it fails unless the generated lock
    matches the checked-in file (ignoring line endings). Use -Update to write
    the generated lock.

.PARAMETER Update
    Write the regenerated lock to ml/requirements.lock instead of only checking
    that it reproduces.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\update_ml_lock.ps1

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\update_ml_lock.ps1 -Update
#>
[CmdletBinding()]
param(
    [switch]$Update
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if ($env:OS -ne "Windows_NT") {
    throw "ml/requirements.lock targets Windows x64; run this script on Windows."
}

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    throw "uv is required: https://docs.astral.sh/uv/"
}

$lockPath = Join-Path $repoRoot "ml\requirements.lock"
$helper = Join-Path $repoRoot "scripts\hash_ml_lock.py"
$scratch = Join-Path $repoRoot ".lock-refresh"
$venv = Join-Path $scratch "venv"
$python = Join-Path $venv "Scripts\python.exe"
$freezePath = Join-Path $scratch "requirements.freeze.txt"
$wheelhouse = Join-Path $scratch "wheelhouse"
$generatedPath = Join-Path $scratch "requirements.lock.generated"

function Get-NormalizedText {
    param([string]$Path)

    $text = Get-Content -LiteralPath $Path -Raw
    return ($text -replace "`r`n", "`n") -replace "`r", "`n"
}

if (Test-Path -LiteralPath $scratch) {
    Remove-Item -LiteralPath $scratch -Recurse -Force
}
New-Item -ItemType Directory -Path $scratch | Out-Null

$pushed = $false
try {
    Push-Location -LiteralPath $repoRoot
    $pushed = $true

    Write-Host "==> Creating clean Python 3.12.13 venv" -ForegroundColor Cyan
    & $uv.Source python install 3.12.13
    if ($LASTEXITCODE -ne 0) { throw "uv python install failed." }
    & $uv.Source venv --python 3.12.13 --seed $venv
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed." }

    Write-Host "==> Installing the current hashed lock" -ForegroundColor Cyan
    & $uv.Source pip install --require-hashes --python $python -r $lockPath
    if ($LASTEXITCODE -ne 0) { throw "Hash-required install failed." }

    Write-Host "==> Running ML tests in the clean venv" -ForegroundColor Cyan
    & $python -m pytest "ml/tests" -q
    if ($LASTEXITCODE -ne 0) { throw "ML tests failed; the lock was not regenerated." }

    Write-Host "==> Freezing verified versions" -ForegroundColor Cyan
    $frozen = & $python -m pip freeze
    if ($LASTEXITCODE -ne 0) { throw "pip freeze failed." }
    $frozen | Set-Content -LiteralPath $freezePath -Encoding Ascii

    Write-Host "==> Downloading matching Windows wheels" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $wheelhouse | Out-Null
    & $python -m pip download --quiet --only-binary=:all: --no-deps --dest $wheelhouse -r $freezePath
    if ($LASTEXITCODE -ne 0) { throw "pip download failed." }

    Write-Host "==> Re-deriving hashes" -ForegroundColor Cyan
    & $python $helper --requirements $freezePath --wheelhouse $wheelhouse --output $generatedPath
    if ($LASTEXITCODE -ne 0) { throw "Lock generation failed." }

    $currentText = Get-NormalizedText -Path $lockPath
    $generatedText = Get-NormalizedText -Path $generatedPath

    if ($currentText -ceq $generatedText) {
        Write-Host "Lock reproduces from a clean Python 3.12.13 environment." -ForegroundColor Green
        return
    }

    if (-not $Update) {
        Write-Host "Generated lock differs from ml/requirements.lock:" -ForegroundColor Yellow
        Compare-Object ($currentText -split "`n") ($generatedText -split "`n") |
            Select-Object -First 20 |
            Format-Table -AutoSize |
            Out-String -Width 200 |
            Write-Host
        throw "Lock does not reproduce. Review the diff, then rerun with -Update if intended."
    }

    Copy-Item -LiteralPath $generatedPath -Destination $lockPath -Force
    Write-Host "Updated ml/requirements.lock from the clean environment." -ForegroundColor Green
}
finally {
    if ($pushed) { Pop-Location }
    Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
}
