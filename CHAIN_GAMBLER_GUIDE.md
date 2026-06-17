# Chain Gambler - Complete System Guide

## What is Chain Gambler?

Chain Gambler is an **autonomous AI trading system** that uses multiple machine learning models working together to make trading decisions. Think of it as a team of AI "brains" that analyze market data and vote on whether to buy, sell, or hold.

---

## How It Works (The Big Picture)

```
Market Data (MT5)
       ↓
Feature Pipeline (150 features extracted)
       ↓
Hybrid Brain (combines model signals)
       ↓
Risk Engine (safety checks)
       ↓
Trade Execution (MT5 orders)
```

### The Three AI Brains

| Brain | What It Does | Analogy |
|-------|--------------|---------|
| **LSTM** | Analyzes price sequences and patterns | Pattern recognition expert |
| **PPO** | Learns optimal trading policies through trial and error | Strategy optimizer |
| **Dreamer** | Predicts future scenarios | Future simulator |

Each brain outputs a "target exposure" (-1 to +1):
- **+1** = Strong buy signal
- **0** = Hold/do nothing  
- **-1** = Strong sell signal

### The Hybrid Brain

The Hybrid Brain combines signals from all three models:

```
Final Signal = (LSTM × weight) + (PPO × weight) + (Dreamer × weight)
```

If the final signal is strong enough, a trade is executed.

---

## Champion/Canary System (Model Evolution)

Think of this like **A/B testing for AI models**:

1. **Champion** - The current best model making live trades
2. **Canary** - A new model being tested (shadow mode)
3. **Candidates** - Models waiting to be evaluated

```
Training → Evaluation → Canary (shadow) → Promotion → Champion (live)
```

A canary only becomes champion if it performs better after live testing.

---

## Trading Modes

| Mode | Description | Risk Level |
|------|-------------|------------|
| **Paper/Dry Run** | Simulated trading, no real money | None |
| **Demo** | Real broker demo account | Fake money |
| **Real Live** | Real money trading (LOCKED by default) | Real money |

**Current Mode**: DEMO (Login: 435656990, Server: Exness-MT5Trial9)

---

## Risk Controls (Safety Features)

The system has multiple safety layers:

### Hard Limits (Cannot Override)
- Max daily loss: $1,000
- Max drawdown: 8%
- Max open positions: 8
- Max positions per symbol: 2

### Soft Checks (Per Trade)
- Spread must be < 25 bps
- Cooldown between trades: 45 seconds
- Symbol exposure < 35% of equity

If risk limits are hit, trading **automatically halts**.

---

## The Dashboard Explained

### Main Tabs

#### 1. **Trades** - Current Activity
- Live positions and PnL
- Equity curve graph
- Recent trade history
- Economic calendar (high-impact events)

#### 2. **Model Brains** - AI Status
- Which models are loaded
- Champion vs Canary comparison
- Model confidence scores

#### 3. **Pipeline** - Training Queue
- What symbols are being trained
- Training progress bars
- Queue status

#### 4. **Training** - Model Development
- Active training cycles
- LSTM/PPO/Dreamer progress
- Training controls (Start/Stop/Force Ingest)

#### 5. **Registry** - Model Library
- All saved models
- Per-symbol model status
- Bundle history

#### 6. **Signal Lanes** - Live Decisions
- Per-symbol trading signals
- Confidence levels
- Current exposure

### Color Coding

| Color | Meaning |
|-------|---------|
| **Green** | Good/Active/Win |
| **Red** | Bad/Error/Loss/Halt |
| **Amber/Yellow** | Warning/Idle/Pending |
| **Cyan** | Info/Neutral |

### Status Indicators

- **LIVE** - System active and trading
- **HALTED** - Trading paused (check risk status)
- **LOCKED** - Real money disabled
- **WS/POLL** - WebSocket connected or polling

---

## Common Scenarios

### "All Symbols Show HOLD"
This is normal! The bot only trades when:
1. Confidence > threshold (usually ~60%)
2. Risk checks pass
3. No economic events blocking

### "Training Not Running"
Models are already trained. Training runs when:
- New symbols added
- Scheduled retraining
- Manual cycle started

### "No Champion Model"
The system works without a champion using "fresh weights." Run training to create champion models.

### "Real Money Locked"
This is a safety feature. To unlock (DANGER):
```yaml
# In config.yaml
system:
  real_money_locked: false
```

---

## Starting the System

### Method 1: Double-Click (Recommended)
Double-click `START_DEMO_BOT.bat`

### Method 2: PowerShell
```powershell
.\GO.ps1
```

### Method 3: Manual
```powershell
# Terminal 1: API Server
cd 02_Core_Python
..\.venv312\Scripts\python.exe -m Python.api_server

# Terminal 2: Trading Bot
cd 02_Core_Python
$env:CHAIN_GAMBLER_EXECUTION_MODE="demo"
$env:AGI_LIVE_ENABLED="true"
$env:MT5_LOGIN="435656990"
$env:MT5_PASSWORD="Fuckyou2/"
$env:MT5_SERVER="Exness-MT5Trial9"
..\.venv312\Scripts\python.exe -m Python.Server_AGI

# Terminal 3: UI
cd 03_UI_Monitoring/frontend
npm run dev -- --port 4180
```

---

## Understanding Log Messages

### Normal Messages
```
DECISION BTCUSDm | regime=HOLD conf=0.36  → Analyzing, no trade
RainforestDetector trained on 5000 bars   → Pattern detection ready
HybridBrain.live_trade executing          → Signal processing
```

### Warning Messages
```
No PPO model found for live inference     → Using fallback
No trained default model found            → Using fresh weights
Waiting for demo trades                   → No trades yet (normal)
```

### Error Messages
```
MT5 connection failed                     → Check MT5 is running
Risk engine halted                        → Check /api/emergency_status
Execution loop error                      → Model/feature mismatch
```

---

## File Structure

```
SupremeChainsaw_Clean/
├── 01_Launchers/          # PowerShell/Batch launchers
├── 02_Core_Python/        # Main Python code
│   ├── Python/
│   │   ├── Server_AGI.py      # Main trading engine
│   │   ├── api_server.py       # REST API
│   │   ├── hybrid_brain.py     # Signal blending
│   │   ├── agi_brain.py        # LSTM model
│   │   ├── mt5_executor.py     # Order execution
│   │   └── feature_pipeline.py # 150-feature extraction
│   ├── alerts/              # Telegram notifications
│   └── models/              # Trained model storage
├── 03_UI_Monitoring/        # React dashboard
│   └── frontend/
│       ├── src/
│       │   ├── App.tsx          # Main app
│       │   ├── components/      # UI panels
│       │   └── services/api.ts  # API client
├── 04_MQL5/                 # MT5 integration
├── 05_Documentation/        # Documentation
└── logs/                    # Runtime logs
```

---

## Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| Bot not starting | Check alerts module exists |
| MT5 not connected | Start MT5, verify credentials |
| Port 5051 in use | Kill existing Python processes |
| UI won't load | Run `npm install` in frontend folder |
| No trades happening | Normal - wait for confidence threshold |
| Training stuck | Restart Server_AGI |
| Feature size error | Model mismatch - retrain or restart |

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/status` | Full system status |
| `GET /api/trades` | Trade history |
| `GET /api/equity_curve` | Account equity over time |
| `POST /api/control` | Send commands (start/stop/etc) |
| `GET /api/emergency_status` | Risk halt status |

---

## Safety Reminders

⚠️ **NEVER** disable `real_money_locked` without:
- Extensive backtesting
- Demo account validation
- Understanding the risks

⚠️ **ALWAYS** monitor:
- Drawdown levels
- Daily trade counts
- Risk halt status

⚠️ **REMEMBER**:
- Past performance ≠ future results
- AI models can fail
- Markets are unpredictable

---

## Support & Logs

- **UI**: http://localhost:4180
- **API**: http://localhost:5051/api/status
- **Logs**: `02_Core_Python/logs/`
- **Config**: `02_Core_Python/config.yaml`

---

*Chain Gambler v1.0 - Trade at your own risk*
