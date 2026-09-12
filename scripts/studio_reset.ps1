param(
    [ValidateSet("start", "refresh", "hot", "cold")]
    [string]$Mode = "start"
)

$ErrorActionPreference = "Continue"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$env:GRADIO_ANALYTICS_ENABLED = "False"
$env:GRADIO_VERSION_CHECK = "False"
$env:PYTHONUNBUFFERED = "1"

$LogDir = Join-Path $Root "logs\studio"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$StudioLog = Join-Path $LogDir "latest-ai-studio.log"
$EngineLog = Join-Path $LogDir "latest-fooocus-engine.log"

$PythonCmd = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $PythonCmd = ".venv\Scripts\python.exe"
}

# Stable local engine profile for 6GB-class NVIDIA cards.
# This avoids the native 0xC0000005 crash seen after VAE/attention startup by keeping VAE on CPU,
# using split attention, disabling xformers, and forcing low VRAM mode.
$EngineSafeArgs = "--disable-analytics --disable-in-browser --attention-split --vae-in-cpu --always-low-vram --disable-xformers"

function Test-PortListening {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $conn
}

function Show-LogTail {
    param(
        [string]$Name,
        [string]$LogPath,
        [int]$TailLines = 40
    )

    if (-not (Test-Path $LogPath)) {
        Write-Warning "$Name log does not exist yet: $LogPath"
        return
    }

    Write-Host ""
    Write-Host "Last $TailLines lines from $Name log: $LogPath"
    Write-Host "------------------------------------------------------------"
    Get-Content -Path $LogPath -Tail $TailLines -ErrorAction SilentlyContinue
    Write-Host "------------------------------------------------------------"
}

function Test-LogHasFailureClue {
    param([string]$LogPath)
    if (-not (Test-Path $LogPath)) {
        return $false
    }

    $tail = Get-Content -Path $LogPath -Tail 80 -ErrorAction SilentlyContinue
    return ($tail -match "Traceback|ModuleNotFoundError|ImportError|RuntimeError|ERROR:|Exception|exited with code [1-9]|exited with code -1073741819|Fooocus watchdog: child exited").Count -gt 0
}

function Wait-PortReady {
    param(
        [string]$Name,
        [int]$Port,
        [int]$TimeoutSeconds,
        [string]$LogPath,
        [int]$LogEverySeconds = 20
    )

    $start = Get-Date
    $deadline = $start.AddSeconds($TimeoutSeconds)
    $lastNotice = $start.AddSeconds(-10)
    $lastLogTail = $start.AddSeconds(-1 * $LogEverySeconds)

    while ((Get-Date) -lt $deadline) {
        if (Test-PortListening $Port) {
            $elapsed = [int]((Get-Date) - $start).TotalSeconds
            Write-Host "$Name is ready on http://127.0.0.1:${Port} after ${elapsed}s"
            return $true
        }

        $now = Get-Date
        $elapsed = [int]($now - $start).TotalSeconds
        $remaining = [Math]::Max(0, [int]($deadline - $now).TotalSeconds)

        if (($now - $lastNotice).TotalSeconds -ge 5) {
            Write-Host "Waiting for $Name on port ${Port}... ${elapsed}s elapsed, ${remaining}s remaining"
            $lastNotice = $now
        }

        if (($now - $lastLogTail).TotalSeconds -ge $LogEverySeconds) {
            if (Test-LogHasFailureClue $LogPath) {
                Write-Warning "$Name log contains an error clue while waiting. Showing recent lines now."
                Show-LogTail $Name $LogPath 60
            } elseif (Test-Path $LogPath) {
                Write-Host "Recent $Name startup log clue:"
                Get-Content -Path $LogPath -Tail 8 -ErrorAction SilentlyContinue
            } else {
                Write-Host "$Name log has not been created yet: $LogPath"
            }
            $lastLogTail = $now
        }

        Start-Sleep -Milliseconds 500
    }

    Write-Warning "$Name did not become ready on port ${Port} within $TimeoutSeconds seconds."
    Show-LogTail $Name $LogPath 100
    return $false
}

function Stop-PortProcess {
    param([int]$Port)
    $pids = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique

    foreach ($processId in $pids) {
        try {
            Write-Host "Stopping process on port ${Port}: PID $processId"
            Stop-Process -Id $processId -Force -ErrorAction Stop
        } catch {
            Write-Warning "Could not stop PID $processId on port ${Port}: $($_.Exception.Message)"
        }
    }
}

function ConvertTo-ProcessArgument {
    param([string]$Value)

    if ($null -eq $Value) {
        return '""'
    }

    $escaped = $Value.Replace('`', '``').Replace('"', '`"')
    return '"' + $escaped + '"'
}

function Start-LoggedWindow {
    param(
        [string]$Title,
        [string]$Command,
        [string]$LogName
    )

    $scriptPath = Join-Path $Root "scripts\run_logged_process.ps1"
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (ConvertTo-ProcessArgument $scriptPath),
        "-Title", (ConvertTo-ProcessArgument $Title),
        "-Command", (ConvertTo-ProcessArgument $Command),
        "-LogName", (ConvertTo-ProcessArgument $LogName)
    ) -join " "

    Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $Root
}

function Start-FooocusEngine {
    if (Test-PortListening 7865) {
        Write-Host "Fooocus engine already appears to be running on http://127.0.0.1:7865"
        return
    }

    if (-not (Test-Path "$env:TEMP\fooocus")) {
        New-Item -ItemType Directory -Path "$env:TEMP\fooocus" | Out-Null
    }

    Write-Host "Using stable Fooocus engine profile: $EngineSafeArgs"
    Start-LoggedWindow "Fooocus Engine" "$PythonCmd -u scripts\run_fooocus_engine_watchdog.py $EngineSafeArgs" "fooocus-engine"
    Write-Host "Starting Fooocus engine watchdog on http://127.0.0.1:7865"
}

function Start-AIStudio {
    if (Test-PortListening 7872) {
        Write-Host "AI Studio already appears to be running on http://127.0.0.1:7872"
        return
    }

    Start-LoggedWindow "AI Studio" "$PythonCmd -u scripts\run_ai_studio_app.py" "ai-studio"
    Write-Host "Starting AI Studio on http://127.0.0.1:7872"
}

function Open-StudioWhenReady {
    param([int]$TimeoutSeconds = 90)

    if (Wait-PortReady "AI Studio" 7872 $TimeoutSeconds $StudioLog 15) {
        Write-Host "Opening AI Studio in browser."
        Start-Process "http://127.0.0.1:7872"
        return $true
    }

    Write-Warning "AI Studio did not open. Review $StudioLog"
    return $false
}

function Start-StudioAndEngine {
    Write-Host "Starting missing services. Studio will open as soon as it is ready. Fooocus may continue warming in the background."
    Start-FooocusEngine
    Start-AIStudio

    $studioReady = Open-StudioWhenReady 90
    $engineReady = Wait-PortReady "Fooocus Engine" 7865 300 $EngineLog 20

    if ($studioReady -and $engineReady) {
        Write-Host "Both AI Studio and Fooocus Engine are ready. Watchdog remains active for Fooocus restarts."
    } elseif ($studioReady -and -not $engineReady) {
        Write-Warning "AI Studio is open, but Fooocus Engine did not become ready. Hidden engine autofill/generation will not work until Fooocus starts."
        Write-Host "Try option 4 Cold reset, or open the engine log: notepad logs\studio\latest-fooocus-engine.log"
    } elseif (-not $studioReady -and $engineReady) {
        Write-Warning "Fooocus Engine is ready, but AI Studio did not become ready."
        Write-Host "Open the Studio log: notepad logs\studio\latest-ai-studio.log"
    } else {
        Write-Warning "Neither service became ready. Review both logs in logs\studio."
    }
}

switch ($Mode) {
    "refresh" {
        Write-Host "Refreshing browser only. No processes stopped."
        Start-Process "http://127.0.0.1:7872"
    }
    "hot" {
        Write-Host "Hot reset: restarting AI Studio only. Fooocus engine stays warm."
        Stop-PortProcess 7872
        Start-Sleep -Seconds 2
        Start-AIStudio
        Open-StudioWhenReady 90 | Out-Null
    }
    "cold" {
        Write-Host "Cold reset: stopping Studio and Fooocus, clearing temp session folder, then starting clean."
        Stop-PortProcess 7872
        Stop-PortProcess 7865
        Remove-Item "$env:TEMP\fooocus" -Recurse -Force -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Path "$env:TEMP\fooocus" -ErrorAction SilentlyContinue | Out-Null
        Start-Sleep -Seconds 2
        Start-StudioAndEngine
    }
    default {
        Start-StudioAndEngine
    }
}

Write-Host ""
Write-Host "AI Studio:      http://127.0.0.1:7872"
Write-Host "Fooocus Engine: http://127.0.0.1:7865"
Write-Host "Logs:           $LogDir"
Write-Host "Latest Studio:  logs\studio\latest-ai-studio.log"
Write-Host "Latest Engine:  logs\studio\latest-fooocus-engine.log"