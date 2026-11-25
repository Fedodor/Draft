<#
.SYNOPSIS
    Deterministic quality gate for the Smart Draft Assistant.

.DESCRIPTION
    Runs the same checks locally and in CI: secret scan, Rust format/lint/tests,
    frontend lint + TypeScript typecheck + build + tests, and Python compile +
    tests. The ML feature-contract and model quality-gate tests run as their own
    visible step (they need no dataset download); the remaining Python suite
    runs after them. When the drafts export and a seeded runtime database are
    present, the scorer's top-1 regression gate replays the corpus through the
    engine as well; without them that step is reported as SKIP with the reason.
    Any failure is reported at the end and makes the script exit non-zero.

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

.PARAMETER SkipBacktest
    Skip the scorer top-1 gate (a release build plus a 20,000-draft replay).

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
    [switch]$SkipSecrets,
    [switch]$SkipBacktest
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

function Add-Skip {
    param([string]$Name, [string]$Detail)

    Write-Host ""
    Write-Host "==> $Name (skipped)" -ForegroundColor DarkGray
    Write-Host "    $Detail" -ForegroundColor DarkGray
    $script:Results.Add([pscustomobject]@{ Step = $Name; Status = "SKIP"; Detail = $Detail })
}

# The shipped ordering's regression gate: replay the exported corpus through the
# engine and fail when top-1 agreement falls below the 2.21% global popularity
# prior. Both inputs are local-only artifacts (a 124 MB export and the runtime
# database), so the step skips with the reason when either is missing, and also
# when the database exists but has no scoring data. The report goes to TEMP so
# the committed reference artifact is never rewritten by a verification run.
function Invoke-ScorerGate {
    $name = "Scorer top-1 gate (20,000 drafts)"
    $drafts = Join-Path $repoRoot "ml\backtest\data\drafts_cm.jsonl"
    $runtimeDb = Join-Path $env:APPDATA "com.smartdraftassistant.app\smart_draft.db"

    if (-not (Test-Path -LiteralPath $drafts)) {
        Add-Skip -Name $name -Detail "no drafts export at ml\backtest\data\drafts_cm.jsonl; run python ml/backtest/export_drafts.py to make it runnable"
        return
    }
    if (-not (Test-Path -LiteralPath $runtimeDb)) {
        Add-Skip -Name $name -Detail "no runtime database at $runtimeDb; seed it with infrastructure/seeders (seed_heroes, then seed_matchups)"
        return
    }

    Write-Host ""
    Write-Host "==> $name" -ForegroundColor Cyan
    $tempDir = [System.IO.Path]::GetTempPath()
    $report = Join-Path $tempDir "sda-backtest-gate.json"
    $stdoutLog = Join-Path $tempDir "sda-backtest-gate.out.log"
    $stderrLog = Join-Path $tempDir "sda-backtest-gate.err.log"
    # Start-Process rather than `& cargo ... *>`: PS 5.1 turns a native command's
    # redirected stderr into error records, which ErrorActionPreference=Stop would
    # turn into a terminating error and hide the gate's exit code.
    $arguments = @(
        "run", "--release", "--manifest-path", "`"$desktopManifest`"",
        "--example", "backtest_scorer", "--",
        "--drafts", "`"$drafts`"",
        "--db", "`"$runtimeDb`"",
        "--sample", "20000", "--min-top1", "0.0221",
        "--out", "`"$report`""
    )
    $process = Start-Process -FilePath "cargo" -ArgumentList $arguments -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
    $exitCode = $process.ExitCode

    $output = @(Get-Content -LiteralPath $stdoutLog -ErrorAction SilentlyContinue)
    $output += @(Get-Content -LiteralPath $stderrLog -ErrorAction SilentlyContinue)
    $output | Select-String -Pattern "top-1 gate:|^error" | ForEach-Object { Write-Host "    $_" }

    if ($exitCode -eq 0) {
        $script:Results.Add([pscustomobject]@{ Step = $name; Status = "PASS"; Detail = "report: $report" })
        Write-Host "    OK" -ForegroundColor Green
        return
    }

    if (($output -join "`n") -match "has no usable data") {
        # An unseeded database is a local setup gap, not an ordering regression.
        $detail = "runtime database is not seeded; seed it with infrastructure/seeders (seed_heroes, then seed_matchups)"
        $script:Results.Add([pscustomobject]@{ Step = $name; Status = "SKIP"; Detail = $detail })
        Write-Host "    SKIPPED: $detail" -ForegroundColor DarkGray
        return
    }

    # The gate verdict and any `error` line already printed above; the report at
    # $report holds the measurements behind the failure.
    $output | Select-String -Pattern "^error|panicked|assertion" | Select-Object -Last 5 |
        ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    $script:Failures.Add($name)
    $script:Results.Add([pscustomobject]@{ Step = $name; Status = "FAIL"; Detail = "exit $exitCode (report: $report)" })
    Write-Host "    FAILED (exit $exitCode): top-1 fell below the 2.21% global prior; see $report" -ForegroundColor Red
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

if (-not $SkipPython) {
    # The app activates the scoring seed's patch and reads bans from ban_seed;
    # a mismatch leaves the ban list empty while every health count passes.
    Invoke-Step -Name "Seed patch consistency" -Action {
        & $pythonExe "scripts/check_seed_patch.py"
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
        cargo clippy --locked --manifest-path $seedersManifest --all-targets -- -D warnings
    }
    Invoke-Step -Name "Rust tests (desktop lib)" -Action {
        cargo test --manifest-path $desktopManifest --lib
    }
    Invoke-Step -Name "Rust tests (seeders)" -Action {
        cargo test --locked --manifest-path $seedersManifest
    }
}

if (-not $SkipBacktest) {
    Invoke-ScorerGate
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
