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

param(
    [string]$ArtifactsDir = "",
    [switch]$SkipBuild,
    [switch]$NoSeed
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$desktopDir = Join-Path $repoRoot "apps\desktop"
$tauriDir = Join-Path $desktopDir "src-tauri"
$exePath = Join-Path $tauriDir "target\debug\smart-draft-assistant.exe"
$seedersDir = Join-Path $repoRoot "infrastructure\seeders"
$dbPath = Join-Path $env:APPDATA "com.smartdraftassistant.app\smart_draft.db"

$vitePort = 1420
$cdpPort = 9333
$gsiPort = 3000

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

    if (-not $SkipBuild -or -not (Test-Path -LiteralPath $exePath)) {
        Write-Host "==> Building the desktop binary (dev profile)"
        cargo build --manifest-path (Join-Path $tauriDir "Cargo.toml") | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
    }

    Write-Host "==> Starting Vite"
    $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
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

    Write-Host "==> Launching the app"
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
        Start-Sleep -Milliseconds 500
    }
    Check "WebView2 exposes both windows over CDP" ($targets.Count -ge 2) "targets: $($targets.Count)"
    if ($targets.Count -lt 2) { throw "CDP targets missing" }

    $mainTarget = $null
    $overlayTarget = $null
    for ($i = 0; $i -lt 40; $i++) {
        foreach ($target in $targets) {
            $isOverlay = Invoke-Eval -WebSocketUrl $target.webSocketDebuggerUrl -Expression "document.body.classList.contains('overlay-mode')"
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

    Write-Host "==> Assets and CSP"
    $assets = Invoke-Eval -WebSocketUrl $main -Expression $jsAssets
    Check "hero portraits render (no CSP block)" `
        ($assets.images -ge 5 -and $assets.loaded -eq $assets.images) `
        "images=$($assets.images) loaded=$($assets.loaded) hosts=$($assets.hosts -join ',')"
    Check "portraits come from the Steam CDN" ($assets.hosts -contains "cdn.cloudflare.steamstatic.com") "hosts=$($assets.hosts -join ',')"
    Check "web fonts loaded" ($assets.inter -and $assets.mono) "inter=$($assets.inter) mono=$($assets.mono)"
    Check "styles applied" ($assets.bodyBackground -ne "rgba(0, 0, 0, 0)") "background=$($assets.bodyBackground)"

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
}
finally {
    if ($appProcess -and -not $appProcess.HasExited) {
        Stop-Process -Id $appProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($viteProcess -and -not $viteProcess.HasExited) {
        & taskkill /PID $viteProcess.Id /T /F 2>&1 | Out-Null
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
