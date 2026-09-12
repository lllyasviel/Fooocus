param(
    [Parameter(Mandatory = $true)]
    [string]$Title,

    [Parameter(Mandatory = $true)]
    [string]$Command,

    [Parameter(Mandatory = $true)]
    [string]$LogName
)

$ErrorActionPreference = "Continue"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$LogDir = Join-Path $Root "logs\studio"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$SafeLogName = $LogName -replace '[^a-zA-Z0-9_.-]', '_'
$LogPath = Join-Path $LogDir "$Stamp-$SafeLogName.log"
$LatestPath = Join-Path $LogDir "latest-$SafeLogName.log"

$host.UI.RawUI.WindowTitle = $Title

function Write-LogLine {
    param([string]$Message)
    $line = "[$(Get-Date -Format o)] $Message"
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
    Add-Content -Path $LatestPath -Value $line -Encoding UTF8
}

Remove-Item $LatestPath -Force -ErrorAction SilentlyContinue
Write-LogLine "Starting $Title"
Write-LogLine "Repository: $Root"
Write-LogLine "Command: $Command"
Write-LogLine "Environment: PYTHONUNBUFFERED=$env:PYTHONUNBUFFERED PYTHONIOENCODING=$env:PYTHONIOENCODING"
Write-LogLine "Log file: $LogPath"
Write-LogLine "Latest log: $LatestPath"
Write-LogLine "------------------------------------------------------------"

try {
    cmd.exe /d /s /c "$Command" 2>&1 | ForEach-Object {
        $text = [string]$_
        Write-Host $text
        Add-Content -Path $LogPath -Value $text -Encoding UTF8
        Add-Content -Path $LatestPath -Value $text -Encoding UTF8
    }

    $exitCode = $LASTEXITCODE
    Write-LogLine "------------------------------------------------------------"
    Write-LogLine "$Title exited with code $exitCode"

    if ($exitCode -ne 0) {
        Write-LogLine "ERROR: $Title crashed or exited with a non-zero code. Review the stack trace above."
    }
} catch {
    Write-LogLine "FATAL: $Title launcher wrapper failed."
    Write-LogLine $_.Exception.ToString()
    $exitCode = 1
}

Write-Host ""
Write-Host "Exact log saved to: $LogPath"
Write-Host "Latest log shortcut: $LatestPath"
Write-Host "Press any key to close this window."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
exit $exitCode
