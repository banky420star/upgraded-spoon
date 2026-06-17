@echo off
echo SUPREME_GO_NONINTERACTIVE.bat -- UNATTENDED MODE
echo All `pause` commands replaced with `ping 127.0.0.1 -n 4 ^>nul`
echo Script runs from start to finish without keyboard input.
echo.
echo SUPREME_GO.bat initiated -- see below for step-by-step progress
echo =======================================================================
echo                              ___                              
echo                             / __)                _           
echo    ___ ___  _   _ _ __ ___ \__ \   ___ ___  ___^| ^|_  ___  
echo   / __/ _ \^| ^| ^| ^| '_ ` _ \ ^|__) ^| / __/ _ \/ __^| __^|/ _ \ 
echo  ^| (_^| (_) ^| ^|_^| ^| ^| ^| ^| ^| ^|  __/ ^| (_^|  __/\__ \ ^|_^|  __/ 
echo   \___\___/ \__,_^|_^| ^|_^| ^|_^|^|_^|    \___\___^|^|___/\__^|\___^|  
echo.                                                             
echo =======================================================================
echo.
echo   [ INITIALIZING SUPREME CHAINSAW TRADING MATRIX ]
echo.
echo   > Loading Rainforest Neural Ensemble...
echo   > Calibrating LSTM Market Models...
echo   > Priming PPO Deep Reinforcement Learning...
echo   > Syncing MT5 Gateway (Exness-MT5Trial9)...
echo.
echo =======================================================================
echo.
echo This will start:
echo   1. Train Rainforest ML models (all symbols)
echo   2. API Server (port 5051)
echo   3. Trading Bot (Server_AGI)
echo   4. React Dashboard (port 4180)
echo.
echo Press any key to continue or CTRL+C to cancel...
ping 127.0.0.1 /t 3 /nobreak >nul

REM ============================================
REM SET ENVIRONMENT VARIABLES FOR LIVE TRADING
REM ============================================
set CHAIN_GAMBLER_EXECUTION_MODE=demo
set AGI_LIVE_ENABLED=true
set CHAIN_GAMBLER_ALLOW_LIVE=1
set MT5_LOGIN=435656990
set MT5_PASSWORD=Fuckyou2/
set MT5_SERVER=Exness-MT5Trial9
set AGI_API_PORT=5051
set AGI_API_HOST=0.0.0.0
set AGI_CONTROL_TOKEN=supreme_control_token_12345
set TELEGRAM_TOKEN=dummy_token_for_demo
set TELEGRAM_CHAT_ID=0

cd /d "C:\Users\Administrator\Desktop\SupremeChainsaw_Clean"

REM ============================================
REM CREATE LOG DIRECTORY FOR SUB-WINDOW OUTPUT
REM ============================================
if not exist "C:\Users\Administrator\Desktop\SupremeChainsaw_Clean\runtime\logs" mkdir "C:\Users\Administrator\Desktop\SupremeChainsaw_Clean\runtime\logs"
echo.
echo [LOG] Sub-window output redirected to runtime\logs\:
echo        api_server.log, trading_bot.log, react_ui.log
echo.

REM ============================================
REM KILL ANY EXISTING PROCESSES
REM ============================================
echo.
echo [1/5] Stopping any existing processes...
taskkill /F /FI "WINDOWTITLE eq API Server" 2>nul
taskkill /F /FI "WINDOWTITLE eq Trading Bot" 2>nul
taskkill /F /FI "WINDOWTITLE eq React UI" 2>nul
ping 127.0.0.1 /t 2 /nobreak >nul
echo      Done!

REM ============================================
REM TRAIN RAINFOREST MODELS
REM ============================================
echo.
echo [2/5] Training Rainforest ML models (4 symbols)...

cd /d "02_Core_Python"

echo      Training BTCUSDm...
..\.venv312\Scripts\python.exe -m Python.training.train_rainforest --symbol BTCUSDm --n_estimators 200 --timesteps 5000
..\.venv312\Scripts\python.exe -m Python.training.sync_rainforest_model --all 2>&1

echo      Training XAUUSDm...
..\.venv312\Scripts\python.exe -m Python.training.train_rainforest --symbol XAUUSDm --n_estimators 200 --timesteps 5000

echo      Training EURUSDm...
..\.venv312\Scripts\python.exe -m Python.training.train_rainforest --symbol EURUSDm --n_estimators 200 --timesteps 5000

echo      Training GBPUSDm...
..\.venv312\Scripts\python.exe -m Python.training.train_rainforest --symbol GBPUSDm --n_estimators 200 --timesteps 5000

echo      All Rainforest models trained!
cd /d "C:\Users\Administrator\Desktop\SupremeChainsaw_Clean"


REM ============================================
REM START API SERVER (in new window)
REM ============================================
echo.
echo [3/5] Starting API Server on port 5051...
start "API Server" cmd /c "cd /d %~dp002_Core_Python && %~dp0.venv312\Scripts\python.exe -m Python.api_server >>%~dp0runtime\logs\api_server.log 2>&1 && ping 127.0.0.1 -n 4 >nul"
ping 127.0.0.1 -n 6 >nul
echo      API Server started!

REM ============================================
REM START TRADING BOT / SERVER_AGI (in new window)
REM ============================================
echo.
echo [4/5] Starting Trading Bot (Server_AGI)...
start "Trading Bot" cmd /c "cd /d %~dp002_Core_Python && %~dp0.venv312\Scripts\python.exe -m Python.Server_AGI >>%~dp0runtime\logs\trading_bot.log 2>&1 && ping 127.0.0.1 -n 4 >nul"
ping 127.0.0.1 -n 11 >nul
echo      Trading Bot started!

REM ============================================
REM START REACT UI (in new window)
REM ============================================
echo.
echo [5/5] Starting React Dashboard on port 4180...
start "React UI" cmd /c "cd /d C:\supreme-chainsaw\ui_lab_app && C:\supreme-chainsaw\ui_lab_app\tools\node\node.exe node_modules\vite\bin\vite.js dev --port 4180 --host 0.0.0.0 >>%~dp0runtime\logs\react_ui.log 2>&1 && ping 127.0.0.1 -n 4 >nul"
ping 127.0.0.1 -n 6 >nul
echo      React UI started!

REM ============================================
REM WAIT AND VERIFY
REM ============================================
echo.
echo ============================================
echo  WAITING FOR SERVICES TO START...
echo ============================================
ping 127.0.0.1 -n 9 >nul

echo.
echo Testing connections...
curl -s http://127.0.0.1:5051/api/status >nul 2>&1
if %errorlevel% equ 0 (
    echo      [OK] API Server responding
) else (
    echo      [WAIT] API Server still starting...
)

curl -s http://127.0.0.1:4180 >nul 2>&1
if %errorlevel% equ 0 (
    echo      [OK] React UI responding
) else (
    echo      [WAIT] React UI still starting...
)

REM ============================================
REM DISPLAY STATUS
REM ============================================
echo.
echo ============================================
echo  ALL SYSTEMS LAUNCHED!
echo ============================================
echo.
echo  Dashboard:    http://localhost:4180/
echo  API Status:   http://localhost:5051/api/status
echo.
echo  Account:      435656990
echo  Server:       Exness-MT5Trial9
echo  Mode:         DEMO (Live Trading)
echo.
echo  MT5 Symbols:  BTCUSDm, XAUUSDm, EURUSDm, GBPUSDm
echo.
echo ============================================
echo  BOT IS NOW RUNNING!
echo ============================================
echo.
echo Check the open command windows for logs.
echo.
echo Press any key to open dashboard in browser...
ping 127.0.0.1 /t 3 /nobreak >nul

REM Open dashboard
start http://localhost:4180/

echo.
echo Done! SupremeChainsaw is now trading.
echo.
ping 127.0.0.1 /t 3 /nobreak >nul
