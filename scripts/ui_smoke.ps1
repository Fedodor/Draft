# GUI smoke test for the Tauri app.
#
# Launches the real desktop binary against the Vite dev server, drives the
# WebView2 renderer over the Chrome DevTools Protocol, and fails on the
# regressions unit tests cannot see: blocked CSP assets, missing tabs, a broken
# overlay toggle, GSI ingestion not reaching the UI, and layout overflow.
#
# Windows only (WebView2 + Win32 window probes). CI runs it on windows-latest;
# run locally the same way:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\ui_smoke.ps1
#
# Parameters:
#   -ArtifactsDir  where screenshots/logs go (default: $env:RUNNER_TEMP\ui-smoke,
#                  else <repo>\.ui-smoke)
#   -SkipBuild     reuse apps/desktop/src-tauri/target/debug if it exists
#   -NoSeed        do not seed a missing database
#   -Bundle        build the release bundle (NSIS), install it into a temporary
#                  directory, and drive the installed app. Nothing is served by
#                  Vite in this mode: the UI must come from the packaged assets,
#                  and the ONNX model must load from the installed resources.

param(
    [string]$ArtifactsDir = "",
    [switch]$SkipBuild,
    [switch]$NoSeed,
    [switch]$Bundle
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$desktopDir = Join-Path $repoRoot "apps\desktop"
$tauriDir = Join-Path $desktopDir "src-tauri"
$seedersDir = Join-Path $repoRoot "infrastructure\seeders"
$dbPath = Join-Path $env:APPDATA "com.smartdraftassistant.app\smart_draft.db"
$bundleDir = Join-Path $tauriDir "target\release\bundle\nsis"
$npm = (Get-Command npm.cmd -ErrorAction Stop).Source

if ($Bundle) {
    # The installer is silent (`/S`) and NSIS takes the target directory from an
    # unquoted `/D=` that must be the last argument, so the path cannot contain
    # spaces. Fall back to the 8.3 short name when the temporary directory has
    # one (e.g. a user profile named "John Doe").
    $installDir = Join-Path ([IO.Path]::GetTempPath()) "sda-ui-smoke-install"
    New-Item -ItemType Directory -Force -Path $installDir | Out-Null
    if ($installDir -match '\s') {
        try {
            $shortPath = (New-Object -ComObject Scripting.FileSystemObject).GetFolder($installDir).ShortPath
        } catch {
            throw "Install path contains spaces and has no 8.3 short name: $installDir"
        }
        if ($shortPath -match '\s') { throw "Install path still contains spaces: $shortPath" }
        $installDir = $shortPath
    }
    # The installer keeps the Cargo binary name for the executable
    # (`smart-draft-assistant.exe`), not the display product name, so find it
    # after install instead of assuming a rename.
    $exePath = $null
} else {
    $installDir = $null
    $exePath = Join-Path $tauriDir "target\debug\smart-draft-assistant.exe"
}

$vitePort = 1420
$cdpPort = 9333
$gsiPort = 3000

# WebView2 runtime 150+ ignores WEBVIEW2_* variables (and HKCU policy) for an
# elevated host process, which is what GitHub-hosted Windows runners are. HKLM
# policy is the documented channel that survives elevation.
$isElevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
$policyKey = "HKLM:\Software\Policies\Microsoft\Edge\WebView2\AdditionalBrowserArguments"
# The policy is keyed by executable name; derive it from the binary actually
# launched (set in the launch step, once the bundle is installed).
$policyName = $null
$policySet = $false
$policyPrevious = $null

if (-not $ArtifactsDir) {
    $ArtifactsDir = if ($env:RUNNER_TEMP) {
        Join-Path $env:RUNNER_TEMP "ui-smoke"
    } else {
        Join-Path $repoRoot ".ui-smoke"
    }
}
New-Item -ItemType Directory -Force -Path $ArtifactsDir | Out-Null
$viteOut = Join-Path $ArtifactsDir "vite.out.log"
$viteErr = Join-Path $ArtifactsDir "vite.err.log"
$appOut = Join-Path $ArtifactsDir "app.out.log"
$appErr = Join-Path $ArtifactsDir "app.err.log"

$script:Failures = New-Object System.Collections.Generic.List[string]

function Check {
    param([string]$Name, [bool]$Ok, [string]$Detail = "")
    if ($Ok) {
        Write-Host "PASS  $Name" -ForegroundColor Green
    } else {
        Write-Host "FAIL  $Name :: $Detail" -ForegroundColor Red
        $script:Failures.Add($Name)
    }
}

function Test-PortInUse {
    param([int]$Port)
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return [bool]$listener
}

# --- CDP plumbing -----------------------------------------------------------

function Get-CdpTargets {
    param([int]$Port)
    (Invoke-RestMethod -Uri "http://127.0.0.1:$Port/json" -TimeoutSec 5) | ForEach-Object { $_ }
}

function Invoke-Cdp {
    param(
        [string]$WebSocketUrl,
        [string]$Method,
        [hashtable]$Params = @{}
    )
    $socket = New-Object System.Net.WebSockets.ClientWebSocket
    try {
        $socket.ConnectAsync([Uri]$WebSocketUrl, [Threading.CancellationToken]::None).Wait(10000) | Out-Null
        $payload = @{ id = 1; method = $Method; params = $Params } | ConvertTo-Json -Depth 20 -Compress
        $bytes = [Text.Encoding]::UTF8.GetBytes($payload)
        $segment = New-Object System.ArraySegment[byte] -ArgumentList @(, $bytes)
        $socket.SendAsync(
            $segment,
            [System.Net.WebSockets.WebSocketMessageType]::Text,
            $true,
            [Threading.CancellationToken]::None
        ).Wait(10000) | Out-Null

        while ($true) {
            $buffer = New-Object byte[] 65536
            $stream = New-Object System.IO.MemoryStream
            do {
                $recvSegment = New-Object System.ArraySegment[byte] -ArgumentList @(, $buffer)
                $result = $socket.ReceiveAsync($recvSegment, [Threading.CancellationToken]::None).Result
                $stream.Write($buffer, 0, $result.Count)
            } while (-not $result.EndOfMessage)
            $text = [Text.Encoding]::UTF8.GetString($stream.ToArray())
            $stream.Dispose()
            $parsed = $text | ConvertFrom-Json
            if ($parsed.id -eq 1) { return $parsed }
        }
    }
    finally {
        try { $socket.Dispose() } catch { }
    }
}

function Invoke-Eval {
    param([string]$WebSocketUrl, [string]$Expression)
    $response = Invoke-Cdp -WebSocketUrl $WebSocketUrl -Method "Runtime.evaluate" -Params @{
        expression = $Expression
        returnByValue = $true
        awaitPromise = $true
    }
    if ($response.result.exceptionDetails) {
        throw "JS exception: $($response.result.exceptionDetails.text)"
    }
    return $response.result.result.value
}

function Save-Screenshot {
    param([string]$WebSocketUrl, [string]$Path)
    $response = Invoke-Cdp -WebSocketUrl $WebSocketUrl -Method "Page.captureScreenshot" -Params @{
        format = "png"
        fromSurface = $true
    }
    [IO.File]::WriteAllBytes($Path, [Convert]::FromBase64String($response.result.data))
}

# --- Launch diagnostics -----------------------------------------------------

# Written when the app never opens a CDP port, so an environment failure (no
# WebView2 runtime, no interactive desktop, instant crash) is diagnosable from
# the uploaded artifacts instead of just "targets: 0".
function Write-AppDiagnostics {
    param($AppProcess, [string]$Reason)

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("reason: $Reason")
    $lines.Add("mode: $(if ($Bundle) { 'bundle' } else { 'dev' })")
    if ($Bundle) { $lines.Add("install dir: $installDir (exe: $exePath)") }
    if ($AppProcess) {
        $exited = $AppProcess.HasExited
        $lines.Add("app pid: $($AppProcess.Id), exited: $exited")
        if ($exited) { $lines.Add("app exit code: $($AppProcess.ExitCode)") }
        $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($AppProcess.Id)" -ErrorAction SilentlyContinue)
        $lines.Add("child processes: " + (($children | ForEach-Object { "$($_.Name)($($_.ProcessId))" }) -join ', '))
        $webviews = @(Get-Process -Name msedgewebview2 -ErrorAction SilentlyContinue)
        $lines.Add("msedgewebview2 processes: $($webviews.Count)")
        $browserProcs = @(Get-CimInstance Win32_Process -Filter "Name='msedgewebview2.exe'" -ErrorAction SilentlyContinue |
                Sort-Object CreationDate | Select-Object -First 2)
        foreach ($browserProc in $browserProcs) {
            $cmd = [string]$browserProc.CommandLine
            if ($cmd.Length -gt 700) { $cmd = $cmd.Substring(0, 700) + ' ...' }
            $lines.Add("msedgewebview2 cmdline: $cmd")
        }
        $hasPort = @($browserProcs | Where-Object { ([string]$_.CommandLine) -like "*remote-debugging-port*" }).Count -gt 0
        $lines.Add("any webview cmdline has remote-debugging-port: $hasPort")
    }
    $lines.Add("host elevated: $isElevated")
    $policyValue = (Get-ItemProperty -Path $policyKey -Name $policyName -ErrorAction SilentlyContinue).$policyName
    $lines.Add("WebView2 HKLM policy: $(if ($policyValue) { $policyValue } else { 'not set' })")
    $lines.Add("vite port ${vitePort} listening: $(Test-PortInUse -Port $vitePort)")
    $lines.Add("cdp port ${cdpPort} listening: $(Test-PortInUse -Port $cdpPort)")
    $lines.Add("gsi port ${gsiPort} listening: $(Test-PortInUse -Port $gsiPort)")
    $profileDir = Join-Path $env:LOCALAPPDATA "com.smartdraftassistant.app\EBWebView"
    $lines.Add("webview user data dir exists: $(Test-Path -LiteralPath $profileDir) ($profileDir)")

    $runtime = $null
    foreach ($key in @(
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        "HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    )) {
        try {
            $pv = (Get-ItemProperty -Path $key -Name pv -ErrorAction Stop).pv
            if ($pv) { $runtime = $pv; break }
        } catch { }
    }
    $lines.Add("WebView2 runtime version: $(if ($runtime) { $runtime } else { 'not found in registry' })")
    $lines.Add("user interactive: $([Environment]::UserInteractive)")

    if ($AppProcess) {
        try {
            $windows = @(Get-AppWindows -ProcessId $AppProcess.Id)
            $lines.Add("app windows: " + (($windows | ForEach-Object { "'$($_.Title)'(visible=$($_.Visible))" }) -join ', '))
        } catch { $lines.Add("app windows: enumeration failed: $($_.Exception.Message)") }
    }

    try {
        $crashes = @(Get-WinEvent -FilterHashtable @{ LogName = 'Application'; StartTime = (Get-Date).AddMinutes(-20) } `
                -ErrorAction Stop | Where-Object {
                $_.LevelDisplayName -eq 'Error' -and ($_.Message -like '*smart-draft-assistant*' -or $_.ProviderName -like '*Application Error*')
            } | Select-Object -First 3)
        if ($crashes.Count -eq 0) {
            $lines.Add("recent app crash events: none")
        } else {
            foreach ($crash in $crashes) {
                $firstLine = ($crash.Message -split "`n" | Select-Object -First 2) -join ' / '
                $lines.Add("crash event $($crash.Id) $($crash.ProviderName): $firstLine")
            }
        }
    } catch { $lines.Add("recent app crash events: query failed: $($_.Exception.Message)") }

    $debugLog = Join-Path $profileDir "chrome_debug.log"
    if (Test-Path -LiteralPath $debugLog) {
        $lines.Add("--- chrome_debug.log (last 20 lines) ---")
        $lines.AddRange([string[]](Get-Content -LiteralPath $debugLog -Tail 20))
    } else {
        $lines.Add("chrome_debug.log: not present")
    }

    $appLog = Join-Path $env:LOCALAPPDATA "com.smartdraftassistant.app\logs\app-diagnostics.log"
    if (Test-Path -LiteralPath $appLog) {
        $lines.Add("--- app-diagnostics.log (last 30 lines) ---")
        $lines.AddRange([string[]](Get-Content -LiteralPath $appLog -Tail 30))
    } else {
        $lines.Add("app-diagnostics.log: not present")
    }

    $text = $lines -join [Environment]::NewLine
    Set-Content -LiteralPath (Join-Path $ArtifactsDir "app-diagnostics.txt") -Value $text -Encoding UTF8
    Write-Host $text
}

# --- Win32 window probe -----------------------------------------------------

Add-Type -TypeDefinition @"
using System;
using System.Text;
using System.Runtime.InteropServices;
public class UiSmokeWindows {
    public delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr lParam);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowTextW(IntPtr hWnd, StringBuilder text, int count);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
}
"@

function Get-AppWindows {
    param([int]$ProcessId)
    $found = New-Object System.Collections.Generic.List[object]
    $callback = {
        param($hWnd, $lParam)
        $windowPid = 0
        [UiSmokeWindows]::GetWindowThreadProcessId($hWnd, [ref]$windowPid) | Out-Null
        if ($windowPid -eq $ProcessId) {
            $builder = New-Object System.Text.StringBuilder 256
            [UiSmokeWindows]::GetWindowTextW($hWnd, $builder, 256) | Out-Null
            $title = $builder.ToString()
            if ($title) {
                $found.Add([pscustomobject]@{
                        Title   = $title
                        Visible = [UiSmokeWindows]::IsWindowVisible($hWnd)
                    })
            }
        }
        return $true
    }
    [UiSmokeWindows]::EnumWindows($callback, [IntPtr]::Zero) | Out-Null
    return $found
}

function Test-AppWindowVisible {
    param([int]$ProcessId, [string]$Title)
    $match = Get-AppWindows -ProcessId $ProcessId | Where-Object { $_.Title -like "*$Title*" }
    return [bool]($match | Where-Object { $_.Visible })
}

# --- Assertion payloads -----------------------------------------------------

$jsLayout = @'
(() => {
  const overlaps = (a, b) => a.left < b.right - 0.5 && b.left < a.right - 0.5 && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5;
  const cards = [...document.querySelectorAll('.recommendation-card')];
  const bad = [];
  cards.forEach((card, index) => {
    const rects = [...card.querySelectorAll('*')]
      .filter(e => e.children.length === 0 && e.textContent.trim())
      .map(e => ({ text: e.textContent.trim(), rect: e.getBoundingClientRect() }));
    const score = rects.find(r => /^\d+%$/.test(r.text));
    const confidence = rects.find(r => r.text.startsWith('Confidence:'));
    const badges = rects.filter(r => ['TOP', 'MEDIUM', 'LOW', 'HIGH'].includes(r.text));
    const cardRect = card.getBoundingClientRect();
    const overlap = [...badges, ...(confidence ? [confidence] : [])]
      .filter(x => score && overlaps(x.rect, score.rect))
      .map(x => x.text);
    const outside = [...badges, ...(confidence ? [confidence] : [])]
      .filter(x => x.rect.right > cardRect.right + 0.5 || x.rect.left < cardRect.left - 0.5)
      .map(x => x.text);
    if (overlap.length || outside.length) bad.push({ index, overlap, outside });
  });
  return {
    cards: cards.length,
    bad,
    pageOverflowX: document.documentElement.scrollWidth > innerWidth,
    width: innerWidth,
    height: innerHeight
  };
})()
'@

$jsHeader = @'
(() => {
  const header = document.querySelector('header');
  if (!header) return { present: false };
  const labels = [...header.querySelectorAll('span,div')]
    .filter(e => e.children.length === 0 && /^(GSI (Live|Off)|Patch )/.test(e.textContent.trim()))
    .map(e => ({ text: e.textContent.trim(), height: Math.round(e.getBoundingClientRect().height) }));
  return {
    present: true,
    clientWidth: header.clientWidth,
    scrollWidth: header.scrollWidth,
    labels,
    wrapped: labels.some(l => l.height > 20)
  };
})()
'@

$jsAssets = @'
(() => {
  const images = [...document.querySelectorAll('img')];
  return {
    cards: document.querySelectorAll('.recommendation-card').length,
    text: document.body.innerText.slice(0, 120).replace(/\s+/g, ' '),
    images: images.length,
    loaded: images.filter(i => i.complete && i.naturalWidth > 0).length,
    hosts: [...new Set(images.map(i => { try { return new URL(i.src).host; } catch (e) { return 'inline'; } }))],
    inter: document.fonts.check('12px Inter'),
    mono: document.fonts.check('12px "JetBrains Mono"'),
    bodyBackground: getComputedStyle(document.body).backgroundColor
  };
})()
'@

$jsPicks = @'
(() => {
  const text = document.body.innerText;
  return {
    ally: (text.match(/ALLY\n(\d)\/5/) || [])[1] || null,
    enemy: (text.match(/ENEMY\n(\d)\/5/) || [])[1] || null,
    turn: (text.match(/Pick #\d+ - \w+'s turn/) || [])[0] || null
  };
})()
'@

$jsClickTab = @'
((label) => {
  const tab = [...document.querySelectorAll('[role=tab]')].find(t => t.textContent.trim() === label);
  if (!tab) return 'missing';
  tab.click();
  return 'clicked';
})('%TAB%')
'@

# --- Setup ------------------------------------------------------------------

$appProcess = $null
$viteProcess = $null

try {
    Write-Host "==> GUI smoke test" -ForegroundColor Cyan
    Write-Host "    repo:      $repoRoot"
    Write-Host "    artifacts: $ArtifactsDir"
    Write-Host "    mode:      $(if ($Bundle) { 'packaged release bundle' } else { 'dev build' })"

    foreach ($port in @($vitePort, $cdpPort, $gsiPort)) {
        if (Test-PortInUse -Port $port) {
            throw "Port $port is already in use. Stop the running app or dev server first."
        }
    }

    if (-not $NoSeed -and -not (Test-Path -LiteralPath $dbPath)) {
        Write-Host "==> Seeding an empty database"
        cargo build --release --manifest-path (Join-Path $seedersDir "Cargo.toml") | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "seeder build failed" }
        $seedHeroes = Join-Path $seedersDir "target\release\seed_heroes.exe"
        $seedMatchups = Join-Path $seedersDir "target\release\seed_matchups.exe"
        & $seedHeroes | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "seed_heroes failed" }
        & $seedMatchups --synthetic | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "seed_matchups --synthetic failed" }
    }

    if ($Bundle) {
        $installers = @(Get-ChildItem -Path $bundleDir -Filter "*setup.exe" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending)
        if (-not $SkipBuild -or $installers.Count -eq 0) {
            Write-Host "==> Building the release bundle (NSIS)"
            Push-Location $desktopDir
            try {
                & $npm run tauri build -- --bundles nsis | Out-Host
            } finally {
                Pop-Location
            }
            if ($LASTEXITCODE -ne 0) { throw "tauri build failed" }
            $installers = @(Get-ChildItem -Path $bundleDir -Filter "*setup.exe" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending)
        }
        if ($installers.Count -eq 0) { throw "no NSIS installer found in $bundleDir" }
        $installerPath = $installers[0].FullName
        Write-Host "    installer: $installerPath"

        # A previous run can leave the bundle behind (or registered): clear it so
        # the installer always runs from scratch.
        if (@(Get-ChildItem -LiteralPath $installDir -ErrorAction SilentlyContinue).Count -gt 0) {
            $staleUninstaller = Join-Path $installDir "uninstall.exe"
            if (Test-Path -LiteralPath $staleUninstaller) {
                Start-Process -FilePath $staleUninstaller -ArgumentList @("/S") -Wait -ErrorAction SilentlyContinue
            }
            Remove-Item -LiteralPath $installDir -Recurse -Force -ErrorAction SilentlyContinue
            New-Item -ItemType Directory -Force -Path $installDir | Out-Null
        }
        Write-Host "==> Installing the bundle silently into $installDir"
        $installer = Start-Process -FilePath $installerPath -ArgumentList @("/S", "/D=$installDir") -Wait -PassThru
        Check "NSIS installer exited successfully" ($installer.ExitCode -eq 0) "exit=$($installer.ExitCode)"

        $installedExes = @(Get-ChildItem -LiteralPath $installDir -Filter "*.exe" -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -ne "uninstall.exe" } | Sort-Object Length -Descending)
        Check "installed bundle contains the app binary" ($installedExes.Count -ge 1) "exes=$($installedExes.Name -join ',')"
        if ($installedExes.Count -eq 0) {
            Write-AppDiagnostics -AppProcess $null -Reason "installed binary missing"
            throw "installed binary missing"
        }
        $exePath = $installedExes[0].FullName
        Write-Host "    installed exe: $exePath"

        $installedFiles = @(Get-ChildItem -LiteralPath $installDir -Recurse -File -ErrorAction SilentlyContinue)
        $wantedResources = @(
            "draft_model_xgboost_full.onnx",
            "draft_model_xgboost_full_features.json",
            "draft_model_xgboost_full_schema.json",
            "onnxruntime.dll"
        )
        $missingResources = @($wantedResources | Where-Object { $installedFiles.Name -notcontains $_ })
        Check "installed bundle ships the model, sidecars and ONNX runtime" ($missingResources.Count -eq 0) `
            "missing=$($missingResources -join ',') files=$($installedFiles.Count)"
    } else {
        if (-not $SkipBuild -or -not (Test-Path -LiteralPath $exePath)) {
            Write-Host "==> Building the desktop binary (dev profile)"
            cargo build --manifest-path (Join-Path $tauriDir "Cargo.toml") | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
        }
    }

    if (-not $Bundle) {
        Write-Host "==> Starting Vite"
        $viteProcess = Start-Process -FilePath $npm -ArgumentList @("run", "dev") -WorkingDirectory $desktopDir `
            -PassThru -RedirectStandardOutput $viteOut -RedirectStandardError $viteErr -WindowStyle Hidden

        $viteReady = $false
        for ($i = 0; $i -lt 120; $i++) {
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:$vitePort/" -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -eq 200) { $viteReady = $true; break }
            } catch { }
            Start-Sleep -Milliseconds 500
        }
        Check "vite dev server answers on :$vitePort" $viteReady
        if (-not $viteReady) { throw "Vite did not start" }
    }

    Write-Host "==> Launching the app"
    $policyName = Split-Path -Leaf $exePath
    $modelCopies = @(Get-ChildItem -LiteralPath (Split-Path -Parent $exePath) -Recurse -Filter "draft_model_xgboost_full.onnx" -ErrorAction SilentlyContinue |
            Select-Object -First 5 | ForEach-Object { $_.FullName })
    Write-Host "    model copies beside the binary: $(if ($modelCopies) { $modelCopies -join ', ' } else { 'none' })"

    # Set the HKLM policy when we have the rights, and restore it afterwards.
    if ($isElevated) {
        try {
            if (-not (Test-Path -LiteralPath $policyKey)) { New-Item -Path $policyKey -Force | Out-Null }
            $existing = Get-ItemProperty -Path $policyKey -Name $policyName -ErrorAction SilentlyContinue
            if ($existing) { $policyPrevious = $existing.$policyName }
            Set-ItemProperty -Path $policyKey -Name $policyName -Value "--remote-debugging-port=$cdpPort" -Type String
            $policySet = $true
            Write-Host "    elevated host: WebView2 HKLM policy set for $policyName"
        } catch {
            Write-Host "    elevated host: HKLM policy write failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "    standard user: relying on WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"
    }

    $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = "--remote-debugging-port=$cdpPort"
    try {
        $appProcess = Start-Process -FilePath $exePath -WorkingDirectory $tauriDir `
            -PassThru -RedirectStandardOutput $appOut -RedirectStandardError $appErr
    }
    finally {
        Remove-Item Env:\WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS -ErrorAction SilentlyContinue
    }

    $targets = @()
    for ($i = 0; $i -lt 120; $i++) {
        try {
            $targets = @(Get-CdpTargets -Port $cdpPort | Where-Object { $_.type -eq "page" })
            if ($targets.Count -ge 2) { break }
        } catch { }
        # A process that already exited will never open a port; stop waiting.
        if ($appProcess.HasExited) { break }
        Start-Sleep -Milliseconds 500
    }
    Check "WebView2 exposes both windows over CDP" ($targets.Count -ge 2) "targets: $($targets.Count)"
    if ($targets.Count -lt 2) {
        if ($appProcess.HasExited) {
            Check "app process stayed alive" $false "exit code $($appProcess.ExitCode)"
        }
        Write-AppDiagnostics -AppProcess $appProcess -Reason "CDP targets missing"
        throw "CDP targets missing"
    }

    $mainTarget = $null
    $overlayTarget = $null
    for ($i = 0; $i -lt 40; $i++) {
        foreach ($target in $targets) {
            try {
                $isOverlay = Invoke-Eval -WebSocketUrl $target.webSocketDebuggerUrl -Expression "document.body.classList.contains('overlay-mode')"
            } catch {
                # A renderer that is still starting rejects the evaluation;
                # retry on the next poll instead of treating it as "main".
                continue
            }
            if ($isOverlay) { $overlayTarget = $target } else { $mainTarget = $target }
        }
        if ($mainTarget -and $overlayTarget) { break }
        Start-Sleep -Milliseconds 500
        $targets = @(Get-CdpTargets -Port $cdpPort | Where-Object { $_.type -eq "page" })
    }
    Check "main dashboard and overlay targets were identified" ([bool]($mainTarget -and $overlayTarget))
    if (-not ($mainTarget -and $overlayTarget)) { throw "could not classify targets" }

    $main = $mainTarget.webSocketDebuggerUrl
    $overlay = $overlayTarget.webSocketDebuggerUrl

    # Give React one render pass with data.
    Start-Sleep -Seconds 2

    if ($Bundle) {
        Write-Host "==> Packaged bundle"
        $shell = Invoke-Eval -WebSocketUrl $main -Expression @'
(() => ({
  origin: location.origin,
  href: location.href,
  scripts: [...document.querySelectorAll('script[src]')].map(s => new URL(s.src).origin),
  styles: [...document.querySelectorAll('link[rel=stylesheet]')].map(l => new URL(l.href).origin),
  usesViteClient: [...document.scripts].some(s => s.src.includes('/@vite/client')),
  bodyText: document.body.innerText.length
}))()
'@
        Check "packaged UI does not use the Vite dev client" (-not $shell.usesViteClient) "href=$($shell.href)"
        Check "UI is served from the app origin, not the dev server" `
            ($shell.origin -notlike "*:$vitePort" -and $shell.href -notlike "*:$vitePort*") "origin=$($shell.origin) href=$($shell.href)"
        # Everything the bundle ships must come from the app origin; the only
        # remote stylesheet is the Google Fonts one the CSP allows.
        $foreignAssets = @(@($shell.scripts) + @($shell.styles) |
                Where-Object { $_ -ne $shell.origin -and $_ -ne "https://fonts.googleapis.com" })
        Check "embedded script and style assets load from the app origin" `
            ((@($shell.scripts).Count + @($shell.styles).Count) -gt 0 -and $foreignAssets.Count -eq 0) `
            "scripts=$($shell.scripts -join ',') styles=$($shell.styles -join ',') foreign=$($foreignAssets -join ',')"
        Check "dev server stays down while the bundle runs" (-not (Test-PortInUse -Port $vitePort)) "port $vitePort is listening"
    }

    # Resource resolution, checked in both modes: the installed bundle ships the
    # model under its own `resources/` directory, while the dev build copies it
    # into target/debug. A path that only works in one of them is a packaging bug.
    $ml = Invoke-Eval -WebSocketUrl $main -Expression `
        "(async () => await window.__TAURI_INTERNALS__.invoke('get_ml_status'))()"
    Check "ONNX model loaded from the app's resources" ([bool]$ml.loaded) "loaded=$($ml.loaded) path=$($ml.model_path)"
    Check "model features and schema sidecars loaded" ([int]$ml.feature_count -gt 0) "features=$($ml.feature_count)"
    if ($Bundle) {
        Check "model was loaded from the installed resources" ($ml.model_path -like "*$installDir*") `
            "path=$($ml.model_path) installDir=$installDir"
    }

    Write-Host "==> Tabs"
    $tabLabels = Invoke-Eval -WebSocketUrl $main -Expression "[...document.querySelectorAll('[role=tab]')].map(t => t.textContent.trim())"
    Check "four tabs are present" (@($tabLabels).Count -eq 4) "tabs=$($tabLabels -join ',')"

    $tabChecks = @(
        @{ Label = "Picks"; Marker = "TOP PICKS" },
        @{ Label = "Draft"; Marker = "MANUAL DRAFT CONTROL" },
        @{ Label = "Analysis"; Marker = "Team Composition" },
        @{ Label = "Settings"; Marker = "Game State Integration" }
    )
    foreach ($tabCheck in $tabChecks) {
        $expression = $jsClickTab.Replace("%TAB%", $tabCheck.Label)
        $clicked = Invoke-Eval -WebSocketUrl $main -Expression $expression
        Start-Sleep -Milliseconds 700
        $state = Invoke-Eval -WebSocketUrl $main -Expression @"
(() => ({
  selected: ([...document.querySelectorAll('[role=tab]')].find(t => t.getAttribute('aria-selected') === 'true') || {}).textContent || '',
  hasMarker: document.body.innerText.includes('$($tabCheck.Marker)')
}))()
"@
        Check "tab '$($tabCheck.Label)' renders its panel" `
            ($clicked -eq "clicked" -and $state.selected.Trim() -eq $tabCheck.Label -and $state.hasMarker) `
            "clicked=$clicked selected='$($state.selected)' marker=$($state.hasMarker)"
    }

    # Back to the Picks tab for the layout and GSI assertions.
    Invoke-Eval -WebSocketUrl $main -Expression $jsClickTab.Replace("%TAB%", "Picks") | Out-Null
    Start-Sleep -Milliseconds 500

    Write-Host "==> GSI ingestion reaches the UI"
    $payload = @{
        provider = @{ name = "Dota 2"; appid = 570 }
        map      = @{ game_state = "DOTA_GAMERULES_STATE_HERO_SELECTION"; game_time = 0; clock_time = 0 }
        player   = @{ team_name = "radiant" }
        draft    = @{
            activeteam = 2
            pick       = $true
            team2      = @{ team_id = 2; pick0_id = 1; pick1_id = 2; pick2_id = 3 }
            team3      = @{ team_id = 3; pick0_id = 4; pick1_id = 5; pick2_id = 6; ban0_id = 7; ban1_id = 8 }
        }
    } | ConvertTo-Json -Depth 8 -Compress
    $gsiResponse = $null
    try {
        $gsiResponse = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$gsiPort/gsi" -Body $payload -ContentType "application/json" -TimeoutSec 10
    } catch {
        $gsiResponse = "error: $($_.Exception.Message)"
    }
    Check "GSI listener accepted the payload" ("$gsiResponse" -eq "OK") "response=$gsiResponse"
    Start-Sleep -Seconds 2
    $picks = Invoke-Eval -WebSocketUrl $main -Expression $jsPicks
    Check "draft reached the UI" ($picks.ally -eq "3" -and $picks.enemy -eq "3") "ally=$($picks.ally) enemy=$($picks.enemy)"
    Check "turn indicator followed the payload" ($picks.turn -like "Pick #7*") "turn=$($picks.turn)"

    # Portraits only exist once recommendation cards render, so this runs after
    # the draft arrives. It still fails on a CSP that blocks the Steam CDN, on
    # fonts the CSP rejects, or on styles that never apply.
    Write-Host "==> Assets and CSP"
    $assets = Invoke-Eval -WebSocketUrl $main -Expression $jsAssets
    for ($i = 0; $i -lt 20 -and ($assets.images -lt 5 -or $assets.loaded -ne $assets.images); $i++) {
        Start-Sleep -Milliseconds 500
        $assets = Invoke-Eval -WebSocketUrl $main -Expression $jsAssets
    }
    Check "hero portraits render (no CSP block)" `
        ($assets.images -ge 5 -and $assets.loaded -eq $assets.images) `
        "cards=$($assets.cards) images=$($assets.images) loaded=$($assets.loaded) hosts=$($assets.hosts -join ',') text='$($assets.text)'"
    Check "portraits come from the Steam CDN" ($assets.hosts -contains "cdn.cloudflare.steamstatic.com") "hosts=$($assets.hosts -join ',')"
    Check "web fonts loaded" ($assets.inter -and $assets.mono) "inter=$($assets.inter) mono=$($assets.mono)"
    Check "styles applied" ($assets.bodyBackground -ne "rgba(0, 0, 0, 0)") "background=$($assets.bodyBackground)"

    Write-Host "==> Layout at 900x700"
    Start-Sleep -Seconds 1
    $layout900 = Invoke-Eval -WebSocketUrl $main -Expression $jsLayout
    Check "recommendation cards rendered" ($layout900.cards -ge 1) "cards=$($layout900.cards)"
    Check "no card header overlap" (@($layout900.bad).Count -eq 0) "bad=$($layout900.bad | ConvertTo-Json -Compress)"
    Check "no horizontal overflow at 900" (-not $layout900.pageOverflowX) "scrollWidth > innerWidth"
    Save-Screenshot -WebSocketUrl $main -Path (Join-Path $ArtifactsDir "main-900.png")

    Write-Host "==> Layout at 800x700 (window minimum width)"
    $resized = Invoke-Eval -WebSocketUrl $main -Expression @'
(async () => {
  await window.__TAURI_INTERNALS__.invoke('plugin:window|set_size', { label: 'main', value: { Logical: { width: 800, height: 700 } } });
  return true;
})()
'@
    Start-Sleep -Seconds 2
    $layout800 = Invoke-Eval -WebSocketUrl $main -Expression $jsLayout
    Check "window resized to the 800px minimum" ($resized -and $layout800.width -eq 800) "width=$($layout800.width)"
    Check "no card header overlap at 800" (@($layout800.bad).Count -eq 0) "bad=$($layout800.bad | ConvertTo-Json -Compress)"
    Check "no horizontal overflow at 800" (-not $layout800.pageOverflowX) "scrollWidth > innerWidth"

    $header = Invoke-Eval -WebSocketUrl $main -Expression $jsHeader
    Check "header status labels stay on one line" `
        ($header.present -and -not $header.wrapped -and $header.scrollWidth -le $header.clientWidth) `
        "labels=$($header.labels | ConvertTo-Json -Compress)"
    Save-Screenshot -WebSocketUrl $main -Path (Join-Path $ArtifactsDir "main-800.png")

    $mlStatusPath = Join-Path $ArtifactsDir "ml-status.json"
    $ml | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $mlStatusPath -Encoding UTF8

    Write-Host "==> Overlay window"
    $overlayText = Invoke-Eval -WebSocketUrl $overlay -Expression "document.body.innerText"
    Check "overlay renders draft content" ($overlayText -like "*DRAFT OVERLAY*") "overlay body missing heading"

    $toggleOn = Invoke-Eval -WebSocketUrl $main -Expression @'
(() => {
  const button = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Overlay');
  if (!button) return false;
  button.click();
  return true;
})()
'@
    Start-Sleep -Seconds 2
    Check "overlay toggle shows the overlay window" `
        ($toggleOn -and (Test-AppWindowVisible -ProcessId $appProcess.Id -Title "Draft Overlay")) `
        "toggleClicked=$toggleOn"

    $dashboardSurvived = Invoke-Eval -WebSocketUrl $main -Expression "[...document.querySelectorAll('[role=tab]')].length"
    Check "main dashboard survives the overlay toggle" ($dashboardSurvived -eq 4) "tabs=$dashboardSurvived"
    Save-Screenshot -WebSocketUrl $overlay -Path (Join-Path $ArtifactsDir "overlay.png")

    $toggleOff = Invoke-Eval -WebSocketUrl $main -Expression @'
(() => {
  const button = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Overlay');
  if (!button) return false;
  button.click();
  return true;
})()
'@
    Start-Sleep -Seconds 2
    Check "overlay toggle hides the overlay window" `
        ($toggleOff -and -not (Test-AppWindowVisible -ProcessId $appProcess.Id -Title "Draft Overlay")) `
        "toggleClicked=$toggleOff"
}
catch {
    Write-Host ""
    Write-Host "GUI smoke test aborted: $($_.Exception.Message)" -ForegroundColor Red
    $script:Failures.Add("harness: $($_.Exception.Message)")
    try { Write-AppDiagnostics -AppProcess $appProcess -Reason "aborted: $($_.Exception.Message)" } catch { }
}
finally {
    if ($appProcess -and -not $appProcess.HasExited) {
        # Kill the tree: WebView2 children keep the install directory locked, so
        # the uninstaller would fail on a lingering host process.
        & taskkill /PID $appProcess.Id /T /F 2>&1 | Out-Null
    }
    if ($viteProcess -and -not $viteProcess.HasExited) {
        & taskkill /PID $viteProcess.Id /T /F 2>&1 | Out-Null
    }
    if ($policySet) {
        try {
            if ($policyPrevious) {
                Set-ItemProperty -Path $policyKey -Name $policyName -Value $policyPrevious -Type String
            } else {
                Remove-ItemProperty -Path $policyKey -Name $policyName -ErrorAction SilentlyContinue
            }
        } catch { }
    }
    # Always archive the app's own log: model/DLL resolution failures are
    # non-fatal, so they never reach app-diagnostics.txt and cannot be
    # diagnosed from a red check alone.
    try {
        $appLog = Join-Path $env:LOCALAPPDATA "com.smartdraftassistant.app\logs\app-diagnostics.log"
        if (Test-Path -LiteralPath $appLog) {
            Copy-Item -LiteralPath $appLog -Destination (Join-Path $ArtifactsDir "app-log.txt") -Force -ErrorAction Stop
        }
    } catch { }
    if ($Bundle -and $installDir -and (Test-Path -LiteralPath $installDir)) {
        # Registered per-user by NSIS, so run the uninstaller before deleting the
        # directory; otherwise every run leaves an entry in Apps & features.
        $uninstaller = Join-Path $installDir "uninstall.exe"
        if (Test-Path -LiteralPath $uninstaller) {
            try { Start-Process -FilePath $uninstaller -ArgumentList @("/S") -Wait -ErrorAction Stop } catch { }
        }
        Remove-Item -LiteralPath $installDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ""
if ($script:Failures.Count -gt 0) {
    Write-Host "GUI SMOKE FAILED ($($script:Failures.Count) check(s))" -ForegroundColor Red
    foreach ($failure in $script:Failures) { Write-Host " - $failure" -ForegroundColor Red }
    Write-Host "Artifacts: $ArtifactsDir"
    exit 1
}

Write-Host "GUI SMOKE PASSED" -ForegroundColor Green
Write-Host "Artifacts: $ArtifactsDir"
exit 0
