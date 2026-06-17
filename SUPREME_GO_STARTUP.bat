@echo off
title Supreme Chainsaw Startup Service

REM =======================================================================
REM SUPREME_GO_STARTUP.bat
REM Headless startup wrapper with boot logging.
REM
REM Launches SUPREME_GO.bat in a minimized window with stdin piped
REM from nul so that any "pause" commands are auto-dismissed at boot.
REM Boot log is written to startup_boot.log in the project root.
REM Designed to be placed in the Windows Startup folder.
REM =======================================================================

set LOG_FILE=%~dp0startup_boot.log

REM ---- Header ----
echo [%DATE% %TIME%] ======================================== >> "%LOG_FILE%"
echo [%DATE% %TIME%] Supreme Chainsaw Startup Service started >> "%LOG_FILE%"
echo [%DATE% %TIME%] ======================================== >> "%LOG_FILE%"
echo [%DATE% %TIME%] Script: %~f0 >> "%LOG_FILE%"
echo [%DATE% %TIME%] Project: %~dp0 >> "%LOG_FILE%"
echo Startup sequence initiated -- see startup_boot.log for details

REM ---- Step 1: Validate SUPREME_GO.bat ----
if not exist "%~dp0SUPREME_GO.bat" (
    echo [%DATE% %TIME%] FATAL: SUPREME_GO.bat not found at %~dp0SUPREME_GO.bat >> "%LOG_FILE%"
    echo ERROR: SUPREME_GO.bat not found -- see startup_boot.log for details
    echo [%DATE% %TIME%] Startup aborted -- exit code 1 >> "%LOG_FILE%"
    exit /b 1
)
echo [%DATE% %TIME%] SUPREME_GO.bat found >> "%LOG_FILE%"

REM ---- Step 2: Wait for network/MT5 initialization ----
echo [%DATE% %TIME%] Waiting 15s for network / MT5 init... >> "%LOG_FILE%"
timeout /t 15 /nobreak >nul
echo [%DATE% %TIME%] Network/MT5 init delay complete >> "%LOG_FILE%"

REM ---- Step 3: Launch SUPREME_GO.bat headlessly ----
echo [%DATE% %TIME%] Launching SUPREME_GO.bat (minimized, stdin=nul)... >> "%LOG_FILE%"
type nul | start /min "" cmd /c ""%~dp0SUPREME_GO.bat""

REM ---- Done ----
echo [%DATE% %TIME%] Launch command issued >> "%LOG_FILE%"
echo [%DATE% %TIME%] ======================================== >> "%LOG_FILE%"
echo [%DATE% %TIME%] Startup sequence complete >> "%LOG_FILE%"
echo [%DATE% %TIME%] ======================================== >> "%LOG_FILE%"
echo.>> "%LOG_FILE%"

exit /b 0
