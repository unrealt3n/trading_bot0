# 🚀 Crypto Trading Bot - Android Termux Setup Guide

Complete guide for setting up the crypto trading bot on Android using Termux.

## 📱 Prerequisites

1. **Install Termux** from F-Droid (recommended) or Google Play Store
2. **Update packages**: `pkg update && pkg upgrade`
3. **Install Python**: `pkg install python`
4. **Install Git**: `pkg install git`

## ✅ Verified Termux-Compatible Libraries

All libraries used in this trading bot are confirmed to work in Termux:

### Core Libraries
- `asyncio` - ✅ Python standard library
- `aiohttp` - ✅ Pure Python async HTTP client
- `aiofiles` - ✅ Async file operations
- `requests` - ✅ Standard HTTP library
- `numpy` - ✅ Available via Termux packages
- `pandas` - ✅ Available via Termux packages  
- `scipy` - ✅ Available via Termux packages

### Data Processing
- `json` - ✅ Python standard library
- `orjson` - ✅ Fast JSON parser (optional)
- `pydantic` - ✅ Data validation
- `python-dateutil` - ✅ Date parsing

### Cryptography
- `cryptography` - ✅ Available in Termux
- `pycryptodome` - ✅ Alternative crypto library
- `hashlib` - ✅ Python standard library
- `hmac` - ✅ Python standard library
- `base64` - ✅ Python standard library

### Optional Libraries
- `scikit-learn` - ✅ Available in Termux
- `matplotlib` - ✅ For plotting (if needed)
- `structlog` - ✅ Enhanced logging

## 🚫 NO ccxt Dependency
This bot **does not use ccxt** which can be problematic in Termux. Instead it uses:
- Direct API calls with `aiohttp`/`requests`
- Custom exchange adapters
- Native HTTP client implementations

## 🛠️ Installation Steps

### 1. Clone the Repository
```bash
git clone <your-repo-url> prediction_bot
cd prediction_bot
```

### 2. Install System Dependencies
```bash
# Essential packages for Android/Termux
pkg install python-dev
pkg install libxml2-dev
pkg install libxslt-dev
pkg install libjpeg-turbo-dev
pkg install libpng-dev
pkg install freetype-dev
pkg install clang
pkg install make
pkg install pkg-config
```

### 3. Install Python Dependencies
```bash
# Install enhanced requirements
pip install -r requirements-enhanced.txt

# If any packages fail, install them individually:
pip install aiohttp aiofiles pandas numpy
pip install beautifulsoup4 lxml rich
pip install ccxt python-binance
```

### 4. Configure the Bot
```bash
# Copy example config
cp config.json.example config.json

# Edit configuration
nano config.json
```

### 5. Set Up API Keys (Optional)
```bash
# Edit config.json and add your API keys:
{
  "enhanced": {
    "whale_alert_api_key": "your_whale_alert_key_here",
    "enable_whale_tracking": true,
    "enable_liquidation_tracking": true,
    "enable_retail_detection": true
  }
}
```

## 🎯 Running the Bot

### Start Enhanced Bot
```bash
python enhanced_main.py
```

### Run in Background
```bash
# Install tmux for background running
pkg install tmux

# Start tmux session
tmux new-session -d -s trading_bot

# Attach to session
tmux attach-session -t trading_bot

# Run bot in session
python enhanced_main.py

# Detach with: Ctrl+B, then D
# Reattach later: tmux attach-session -t trading_bot
```

## 📊 Features Available in Termux

✅ **Fully Supported Features:**
- 🐋 Whale Alert Integration
- 💥 Liquidation Tracking via Web Scraping
- 🏦 Retail vs Big Money Detection
- 📈 Historical Pattern Analysis
- ⏰ Dynamic Trade Durations (Scalp/Short/Mid-term)
- 🎯 Dynamic Confidence Adjustment
- 📊 JSON Trade Tracking with P&L
- 🎨 Rich Terminal UI

✅ **Termux Optimizations:**
- Lightweight dependencies
- Efficient memory usage
- Reduced CPU overhead
- Compatible async operations
- SQLite database (no external DB required)

## 🔧 Termux-Specific Configuration & Edge Cases

### Memory Optimization
```bash
# Add to config.json:
{
  "system": {
    "max_concurrent_requests": 5,
    "cache_size_mb": 50,
    "enable_memory_optimization": true
  }
}
```

### Memory Management
- Streaming JSON parsing for large responses
- Async I/O to prevent blocking
- Limited concurrent connections
- Automatic cleanup of HTTP sessions

### Network Issues
- Retry mechanisms with exponential backoff
- Timeout handling for all API calls
- Graceful degradation when exchanges fail
- Multiple data source fallbacks

### API Limitations
- Rate limiting built-in
- API key rotation support
- Free tier API usage optimized
- Alternative data sources when APIs fail

### Mobile-Specific Optimizations
- Lower memory footprint
- Battery-conscious design
- Configurable update intervals
- Minimal logging in production mode

### Storage Location
```bash
# Bot data will be stored in:
/data/data/com.termux/files/home/prediction_bot/logs/
```

### Network Configuration
```bash
# Allow network access (usually default in Termux)
# No additional configuration needed
```

## 🛡️ Error Handling

Every module includes:
- Exception catching and logging
- Graceful fallbacks
- Data validation
- Safe API call wrappers
- Recovery mechanisms

## ⚡ Performance Optimizations

- Concurrent API calls where possible
- Cached data to reduce API usage
- Efficient data structures
- Minimal dependencies
- Optimized for mobile CPUs

## 🚨 Troubleshooting

### Common Issues and Solutions

**1. Package Installation Fails**
```bash
# Update package list
pkg update

# Install build tools
pkg install build-essential

# Retry installation
pip install --upgrade pip
pip install -r requirements-enhanced.txt
```

**2. SSL Certificate Errors**
```bash
# Install ca-certificates
pkg install ca-certificates

# Update certificates
update-ca-certificates
```

**3. Memory Issues**
```bash
# Increase swap (if available)
# Or reduce concurrent operations in config.json
{
  "analysis": {
    "max_concurrent_analysis": 2
  }
}
```

**4. Rich UI Not Working**
```bash
# If rich terminal UI has issues, the bot will fallback to simple UI
# Force simple UI by setting:
export TERM=dumb
python enhanced_main.py
```

## 📱 Android-Specific Tips

### 1. **Battery Optimization**
- Disable battery optimization for Termux in Android settings
- Use "Don't optimize" or add Termux to whitelist

### 2. **Background Running**
```bash
# Acquire wake lock to prevent Android from killing the process
termux-wake-lock

# Release when done
termux-wake-unlock
```

### 3. **Notifications**
```bash
# Install termux-api for notifications
pkg install termux-api

# Get notifications when trades execute
# (Feature can be enabled in config.json)
```

### 4. **File Access**
```bash
# Grant storage permission for logs
termux-setup-storage

# Logs accessible at:
/storage/emulated/0/termux/prediction_bot/logs/
```

## 🔐 Security Considerations

### 1. **API Key Security**
```bash
# Never share your config.json file
# Use environment variables for sensitive data:
export WHALE_ALERT_API_KEY="your_key_here"
export BINANCE_API_KEY="your_key_here"
```

### 2. **Network Security**
- Bot uses HTTPS for all API calls
- Local SQLite database (no remote DB credentials needed)
- No external connections except for market data APIs

### 3. **File Permissions**
```bash
# Secure your config file
chmod 600 config.json

# Secure log directory
chmod 700 logs/
```

## 📈 Performance Monitoring

### View Real-time Performance
```bash
# Monitor bot performance
tail -f logs/enhanced_trading_bot.log

# Monitor trade executions
tail -f logs/trades_tracker.json
```

### Resource Usage
```bash
# Monitor memory usage
ps aux | grep python

# Monitor network usage
# (Use Android system settings)
```

## 🆕 Enhanced Features Usage

### 1. **Whale Alerts**
- Automatically tracks large transactions
- Provides immediate price impact signals
- Works with or without API key (free sources as fallback)

### 2. **Liquidation Tracking**
- Scrapes liquidation data from CoinGlass and other sources
- Identifies momentum and reversal opportunities
- Provides scalping signals

### 3. **Retail vs Big Money**
- Detects when institutional money moves opposite to retail
- Identifies potential retail traps
- Provides directional bias signals

### 4. **Dynamic Confidence**
- Adjusts confidence based on support/resistance levels
- Considers market volatility and session context
- Optimizes for different trade durations

### 5. **Historical Correlation**
- Learns from past whale alerts, liquidations, and news
- Improves signal accuracy over time
- Stores patterns in local SQLite database

## 🎯 Usage Examples

### Basic Trading
```bash
# Start with demo mode
python enhanced_main.py

# Check logs
tail -f logs/enhanced_trading_bot.log
```

### Advanced Configuration
```bash
# Edit advanced settings
nano config.json

# Enable all enhanced features
{
  "enhanced": {
    "enable_whale_tracking": true,
    "enable_liquidation_tracking": true,
    "enable_retail_detection": true,
    "enable_historical_correlation": true,
    "min_whale_threshold_usd": 100000,
    "liquidation_lookback_minutes": 60
  }
}
```

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review logs in `logs/enhanced_trading_bot.log`
3. Ensure all dependencies are installed correctly
4. Check your internet connection
5. Verify API keys (if using paid services)

## 🔄 Updates

To update the bot:
```bash
git pull origin main
pip install -r requirements-enhanced.txt --upgrade
python enhanced_main.py
```

---

**Happy Trading! 🚀📱**