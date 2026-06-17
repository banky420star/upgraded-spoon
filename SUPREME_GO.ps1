#Requires -RunAsAdministrator
<#
.SYNOPSIS
    SupremeChainsaw - SINGLE CLICK FULL STACK BOT LAUNCHER

.DESCRIPTION
    Launches everything:
    1. API Server (port 5051)
    2. Trading Bot / Server_AGI (live trading)
    3. React Dashboard (port 4180)

    All with proper environment variables for demo account trading.

.EXAMPLE
    .\SUPREME_GO.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

# Banner
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SUPREME CHAINSAW - BOT LAUNCHER" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Set location
$RepoRoot = "C:\Users\Administrator\Desktop\SupremeChainsaw_Clean"
Set-Location $RepoRoot

# ============================================
# SET ENVIRONMENT VARIABLES
# ============================================
Write-Host "[Config] Setting environment variables..." -ForegroundColor Yellow

$env:CHAIN_GAMBLER_EXECUTION_MODE = "demo"
$env:AGI_LIVE_ENABLED              = "true"
$env:CHAIN_GAMBLER_ALLOW_LIVE      = "1"
$env:MT5_LOGIN                     = "435656990"
$env:MT5_PASSWORD                  = "Fuckyou2/"
$env:MT5_SERVER                    = "Exness-MT5Trial9"
$env:AGI_API_PORT                  = "5051"
$env:AGI_API_HOST                  = "0.0.0.0"
$env:AGI_CONTROL_TOKEN             = "supreme_control_token_12345"
$env:TELEGRAM_TOKEN                = "dummy_token_for_demo"
$env:TELEGRAM_CHAT_ID              = "0"

Write-Host "         Mode: DEMO (Live Demo Trading)" -ForegroundColor Green
Write-Host "         Account: 435656990" -ForegroundColor Green
Write-Host "         Server: Exness-MT5Trial9" -ForegroundColor Green
Write-Host ""

# ============================================
# KILL EXISTING PROCESSES
# ============================================
Write-Host "[1/5] Stopping any existing processes..." -ForegroundColor Yellow

$processes = @("python", "node")
foreach ($proc in $processes) {
    Get-Process -Name $proc -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -match "API Server|Trading Bot|React UI|api_server|Server_AGI" } |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2
Write-Host "      Done!" -ForegroundColor Green
Write-Host ""

# ============================================
# START API SERVER
# ============================================
Write-Host "[2/5] Starting API Server on port 5051..." -ForegroundColor Yellow

$apiJob = Start-Job -ScriptBlock {
    param($path)
    Set-Location $path\02_Core_Python
    ..\venv312\Scripts\python.exe -m Python.api_server
} -ArgumentList $RepoRoot

Start-Sleep -Seconds 5
Write-Host "      API Server started!" -ForegroundColor Green
Write-Host ""

# ============================================
# START TRADING BOT
# ============================================
Write-Host "[3/5] Starting Trading Bot (Server_AGI)..." -ForegroundColor Yellow

$botJob = Start-Job -ScriptBlock {
    param($path)
    Set-Location $path\02_Core_Python
    ..\venv312\Scripts\python.exe -m Python.Server_AGI
} -ArgumentList $RepoRoot

Start-Sleep -Seconds 10
Write-Host "      Trading Bot started!" -ForegroundColor Green
Write-Host ""

# ============================================
# START REACT UI
# ============================================
Write-Host "[4/5] Starting React Dashboard on port 4180..." -ForegroundColor Yellow

$uiJob = Start-Job -ScriptBlock {
    param($path)
    Set-Location (Join-Path $path "ui_lab_app")
    & "C:\Users\Administrator\Downloads\node-v24.16.0-win-x64\node-v24.16.0-win-x64\node.exe" "node_modules\vite\bin\vite.js" dev --port 4180 --host 0.0.0.0
} -ArgumentList $RepoRoot

Start-Sleep -Seconds 8
Write-Host "      React Dashboard started!" -ForegroundColor Green
Write-Host ""

# ============================================
# VERIFY CONNECTIONS
# ============================================
Write-Host "[5/5] Verifying connections..." -ForegroundColor Yellow

# Test API
$apiOk = $false
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:5051/api/status" -UseBasicParsing -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        $apiOk = $true
        Write-Host "      [OK] API Server responding" -ForegroundColor Green
    }
} catch {
    Write-Host "      [WAIT] API Server still starting..." -ForegroundColor Yellow
}

# Test UI
$uiOk = $false
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:4180" -UseBasicParsing -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        $uiOk = $true
        Write-Host "      [OK] React UI responding" -ForegroundColor Green
    }
} catch {
    Write-Host "      [WAIT] React UI still starting..." -ForegroundColor Yellow
}

Write-Host ""

# ============================================
# DISPLAY STATUS
# ============================================
Write-Host "============================================" -ForegroundColor Green
Write-Host "  ALL SYSTEMS LAUNCHED!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard:    http://localhost:4180/" -ForegroundColor Cyan
Write-Host "  API Status:   http://localhost:5051/api/status" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Account:      435656990" -ForegroundColor White
Write-Host "  Server:       Exness-MT5Trial9" -ForegroundColor White
Write-Host "  Mode:         DEMO (Live Trading)" -ForegroundColor White
Write-Host ""
Write-Host "  Symbols:      BTCUSDm, XAUUSDm, EURUSDm, GBPUSDm" -ForegroundColor White
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  BOT IS NOW RUNNING!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

# Log to file
$logFile = "$RepoRoot\SUPREME_GO_LAUNCH_LOG.txt"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
@"
================================================================================
SUPREME CHAINSAW - BOT LAUNCH LOG
Timestamp: $timestamp
================================================================================

Environment Variables Set:
- CHAIN_GAMBLER_EXECUTION_MODE: demo
- AGI_LIVE_ENABLED: true
- MT5_LOGIN: 435656990
- MT5_SERVER: Exness-MT5Trial9
- AGI_API_PORT: 5051

Processes Started:
- API Server: Job ID $($apiJob.Id)
- Trading Bot: Job ID $($botJob.Id)
- React UI: Job ID $($uiJob.Id)

Connection Status:
- API Server: $(if($apiOk){"ONLINE"}else{"STARTING"})
- React UI: $(if($uiOk){"ONLINE"}else{"STARTING"})

URLs:
- Dashboard: http://localhost:4180/
- API Status: http://localhost:5051/api/status

================================================================================
END OF LOG
================================================================================
"@ | Out-File -FilePath $logFile -Encoding UTF8

Write-Host "Log saved to: $logFile" -ForegroundColor DarkGray
Write-Host ""

# Open dashboard
$openBrowser = Read-Host "Open dashboard in browser? (Y/n)"
if ($openBrowser -ne "n") {
    Start-Process "http://localhost:4180/"
}

Write-Host ""
Write-Host "Press Ctrl+C to stop all services or close this window." -ForegroundColor Magenta
Write-Host ""

# Keep running
while ($true) {
    Start-Sleep -Seconds 30

    # Show heartbeat
    try {
        $status = Invoke-WebRequest -Uri "http://127.0.0.1:5051/api/status" -UseBasicParsing -TimeoutSec 5
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Heartbeat: OK" -ForegroundColor DarkGreen
    } catch {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Heartbeat: CHECK" -ForegroundColor Yellow
    }
}
