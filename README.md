# 🚀 Crypto Trading Bot - Simple & Powerful

## 🎯 Overview

A sophisticated cryptocurrency trading bot that combines 8 different analysis modules to generate high-probability trading signals:

- **📊 Options Analysis** - Options chain data
- **🧠 Smart Money Detection** - Institutional vs retail flows  
- **🌐 Multi-Exchange Analysis** - Cross-exchange validation
- **📈 Technical Indicators** - RSI, MACD, EMA, Bollinger
- **📰 News Sentiment** - Social media and news analysis
- **🕐 Session Context** - Market timing adjustments
- **💎 Gemini Integration** - Additional market validation
- **⚡ Orderbook Analysis** - Depth and liquidity analysis

## ⚡ Super Simple Usage

Just run it and let it trade:
```bash
python main.py
```

That's it! The bot will continuously scan for high-probability trades until you stop it.

## 🔧 Quick Setup

### 1. Install Dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Trading Parameters
Edit `config.json`:
```json
{
  "trading": {
    "mode": "demo",           // "demo" or "live"
    "trade_amount": 20,       // Dollar amount per trade
    "leverage": 1,            // Trading leverage
    "min_confidence_threshold": 65.0  // Minimum signal confidence
  },
  "exchanges": {
    "demo": {
      "api_key": "your_testnet_key",
      "api_secret": "your_testnet_secret"
    },
    "live": {
      "api_key": "your_binance_key", 
      "api_secret": "your_binance_secret"
    }
  }
}
```

### 3. Set Optional API Keys (in config.json)
```json
"optional_apis": {
  // Gemini integration removed - unnecessary
}
```

## 🚀 Usage

### Main Bot (Continuous Trading)
```bash
# Start the bot (runs until you stop it)
python main.py
```

### Testing/Debugging (Optional)
```bash
# Test analysis engine directly
python main_engine.py BTC

# Test specific analysis modules  
python -c "
import asyncio
from main_engine import test_technical_signal
result = asyncio.run(test_technical_signal('BTC'))
print(result)
"
```

## 🎯 Trading Modes

### Demo Mode (Safe Testing)
- **Exchange**: Binance Testnet
- **Money**: Fake/Paper trading
- **Purpose**: Test strategies risk-free
- **API**: Requires Binance Testnet credentials

### Live Mode (Real Trading)
- **Exchange**: Binance
- **Money**: Real money at risk
- **Purpose**: Production trading
- **API**: Requires Binance Live credentials

**Switch modes in `config.json`:**
```json
"trading": {
  "mode": "demo"  // or "live"
}
```

## 📊 Configuration Options

### Trading Parameters
```json
"trading": {
  "trade_amount": 20,              // $ per trade
  "leverage": 1,                   // 1x to 10x
  "take_profit_percent": 2.0,      // 2% profit target
  "stop_loss_percent": 1.0,        // 1% stop loss
  "min_confidence_threshold": 65.0, // 65% minimum signal
  "max_concurrent_trades": 3       // Max open positions
}
```

### Risk Management
```json
"risk_management": {
  "max_daily_trades": 10,          // Daily trade limit
  "max_daily_loss_percent": 5.0,   // 5% daily loss limit
  "position_size_percent": 2.0,    // 2% of account per trade
  "required_signal_strength": "high"
}
```

### Supported Coins
```json
"analysis": {
  "supported_coins": ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOT"],
  "analysis_interval_seconds": 300  // 5 minutes between cycles
}
```

## 📈 Example Trading Session

```bash
# 1. Configure your settings in config.json
# 2. Start the bot
python main.py

# 3. Monitor in real-time
tail -f logs/trading_bot.log

# 4. Stop when needed (Ctrl+C)
```

## 📁 File Structure

```
prediction_bot/
├── main.py              # 🚀 MAIN BOT - Just run this!
├── config.json          # Configuration 
├── main_engine.py       # Analysis engine
├── modules/             # Analysis modules
│   ├── options_analyzer.py
│   ├── smart_money_detector.py
│   ├── multi_exchange_analyzer.py
│   ├── indicator_engine.py
│   ├── news_analyzer.py
│   ├── session_analyzer.py
│   # gemini_analyzer.py (removed)
│   └── utils.py
├── logs/                # Trading logs
└── requirements.txt     # Dependencies
```

## 🔑 API Requirements

### Required (for trading):
- **Binance Testnet** (demo mode) - Free
- **Binance Live** (live mode) - Real account

### Optional (for enhanced analysis):
- **Gemini Token** - Enhanced market data
- All other data sources use free public APIs

## ⚠️ Important Notes

1. **Start with Demo Mode** - Always test strategies first
2. **Free Data Sources** - Uses public APIs, no paid subscriptions needed
3. **Risk Management** - Built-in position sizing and stop losses
4. **High Probability Trades** - Only trades when confidence > 65%
5. **Simple Configuration** - Everything controlled via `config.json`

## 🎯 High-Probability Trading Strategy

The bot only executes trades when:
- ✅ **Multiple signals align** (8 different analysis modules)
- ✅ **Confidence > 65%** (configurable threshold)
- ✅ **Risk limits respected** (daily limits, position sizing)
- ✅ **Market conditions favorable** (session timing, volatility)

This conservative approach focuses on **quality over quantity** for sustainable profits.

---

**⚡ Simple. Powerful. Profitable.**# trading_bot0
