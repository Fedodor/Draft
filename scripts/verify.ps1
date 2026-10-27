<#
.SYNOPSIS
    Deterministic quality gate for the Smart Draft Assistant.

.DESCRIPTION
    Runs the same checks locally and in CI: secret scan, Rust format/lint/tests,
    frontend lint + TypeScript typecheck + build + tests, and Python compile +
    tests. The ML feature-contract and model quality-gate tests run as their own
    visible step (they need no dataset download); the remaining Python suite
    runs after them. Any failure is reported at the end and makes the script
    exit non-zero.

.PARAMETER Python
    Path to the Python interpreter. Defaults to ml\venv\Scripts\python.exe
    when present, otherwise `python`.

.PARAMETER SkipRust
    Skip the Rust format, clippy, and test steps.

.PARAMETER SkipFrontend
    Skip the frontend install, lint, build, and test steps.

.PARAMETER SkipPython
    Skip the Python compile and test steps.

.PARAMETER ForceInstall
    Reinstall frontend dependencies even when node_modules already exists.

.PARAMETER SkipSecrets
    Skip the working-tree secret scan.

.EXAMPLE
    pwsh -File scripts/verify.ps1
#>
[CmdletBinding()]
param(
    [string]$Python = "",
    [switch]$SkipRust,
    [switch]$SkipFrontend,
    [switch]$SkipPython,
    [switch]$ForceInstall,
    [switch]$SkipSecrets
)

$ErrorActionPreference = "Stop"
$script:Failures = New-Object System.Collections.Generic.List[string]
$script:Results = New-Object System.Collections.Generic.List[object]

function Resolve-Python {
    param([string]$Explicit)
    if ($Explicit) { return $Explicit }
    $venv = Join-Path $repoRoot "ml\venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venv) { return $venv }
    return "python"
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action,
        [string]$WorkingDirectory = $repoRoot
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    Push-Location -LiteralPath $WorkingDirectory
    try {
        & $Action
    }
    catch {
        $script:Failures.Add($Name)
        $script:Results.Add([pscustomobject]@{ Step = $Name; Status = "FAIL"; Detail = $_.Exception.Message })
        Write-Host "    FAILED: $($_.Exception.Message)" -ForegroundColor Red
        Pop-Location
        return
    }
    Pop-Location

    if ($LASTEXITCODE -ne 0) {
        $script:Failures.Add($Name)
        $script:Results.Add([pscustomobject]@{ Step = $Name; Status = "FAIL"; Detail = "exit $LASTEXITCODE" })
        Write-Host "    FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        return
    }
    $script:Results.Add([pscustomobject]@{ Step = $Name; Status = "PASS"; Detail = "" })
    Write-Host "    OK" -ForegroundColor Green
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$desktopManifest = "apps/desktop/src-tauri/Cargo.toml"
$seedersManifest = "infrastructure/seeders/Cargo.toml"
$frontendDir = Join-Path $repoRoot "apps/desktop"
$pythonExe = Resolve-Python -Explicit $Python

Write-Host "Smart Draft Assistant quality gate" -ForegroundColor Yellow
Write-Host "Repository: $repoRoot"
Write-Host "Python:     $pythonExe"

if (-not $SkipSecrets) {
    Invoke-Step -Name "Secret scan" -Action {
        & $pythonExe "scripts/secret_scan.py"
    }
}

if (-not $SkipRust) {
    Invoke-Step -Name "Rust format (desktop)" -Action {
        cargo fmt --manifest-path $desktopManifest --all -- --check
    }
    Invoke-Step -Name "Rust format (seeders)" -Action {
        cargo fmt --manifest-path $seedersManifest --all -- --check
    }
    Invoke-Step -Name "Rust clippy (desktop)" -Action {
        cargo clippy --manifest-path $desktopManifest --all-targets -- -D warnings
    }
    Invoke-Step -Name "Rust clippy (seeders)" -Action {
        cargo clippy --manifest-path $seedersManifest --all-targets -- -D warnings
    }
    Invoke-Step -Name "Rust tests (desktop lib)" -Action {
        cargo test --manifest-path $desktopManifest --lib
    }
    Invoke-Step -Name "Rust tests (seeders)" -Action {
        cargo test --manifest-path $seedersManifest
    }
}

if (-not $SkipFrontend) {
    $nodeModules = Join-Path $frontendDir "node_modules"
    if ($ForceInstall -or -not (Test-Path -LiteralPath $nodeModules)) {
        $lock = Join-Path $frontendDir "package-lock.json"
        if (Test-Path -LiteralPath $lock) {
            Invoke-Step -Name "Frontend install (npm ci)" -WorkingDirectory $frontendDir -Action { npm ci }
        }
        else {
            Invoke-Step -Name "Frontend install (npm install)" -WorkingDirectory $frontendDir -Action { npm install }
        }
    }
    else {
        Write-Host ""
        Write-Host "==> Frontend install skipped (node_modules present; use -ForceInstall to refresh)" -ForegroundColor DarkGray
    }
    Invoke-Step -Name "Frontend lint" -WorkingDirectory $frontendDir -Action { npm run lint }
    Invoke-Step -Name "Frontend typecheck + build" -WorkingDirectory $frontendDir -Action { npm run build }
    Invoke-Step -Name "Frontend tests" -WorkingDirectory $frontendDir -Action { npm test }
}

if (-not $SkipPython) {
    Invoke-Step -Name "Python byte-compile" -Action {
        & $pythonExe -m compileall -q "ml/training" "ml/feature-engineering" "scripts"
    }
    Invoke-Step -Name "Python ML contract + quality-gate tests" -Action {
        & $pythonExe -m pytest "ml/tests/test_feature_schema.py" "ml/tests/test_quality_gate.py" -q
    }
    Invoke-Step -Name "Python tests" -Action {
        & $pythonExe -m pytest "ml/tests" -q --ignore="ml/tests/test_feature_schema.py" --ignore="ml/tests/test_quality_gate.py"
    }
}

Write-Host ""
Write-Host "==================== Summary ====================" -ForegroundColor Yellow
$script:Results | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

if ($script:Failures.Count -gt 0) {
    Write-Host "Quality gate FAILED: $($script:Failures.Count) step(s)." -ForegroundColor Red
    foreach ($failure in $script:Failures) { Write-Host " - $failure" -ForegroundColor Red }
    exit 1
}

Write-Host "Quality gate PASSED." -ForegroundColor Green
exit 0
