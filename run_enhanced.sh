#!/bin/bash

# 🚀 Enhanced Trading Bot Launcher Script
# Android Termux Compatible

echo "🚀 Starting Enhanced Crypto Trading Bot..."
echo "=================================================="

# Check if we're running in Termux
if [[ "$PREFIX" == *"com.termux"* ]]; then
    echo "📱 Detected Android Termux environment"
    export TERMUX_ENV=true
else
    echo "💻 Detected standard Linux environment"
    export TERMUX_ENV=false
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p logs/backups
mkdir -p data

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
echo "Python version: $python_version"

if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "❌ Python 3.8+ required. Please upgrade Python."
    exit 1
fi

# Check if enhanced requirements are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import aiohttp, pandas, beautifulsoup4" 2>/dev/null; then
    echo "📥 Installing enhanced requirements..."
    pip install -r requirements-enhanced.txt
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install requirements. Trying individual packages..."
        pip install aiohttp aiofiles pandas numpy beautifulsoup4 lxml rich
    fi
fi

# Check if config exists
if [ ! -f "config.json" ]; then
    echo "⚙️ Creating default config from enhanced template..."
    cp config_enhanced.json config.json
    echo "✅ Default config created. Please edit config.json with your settings."
fi

# Set up environment variables for Termux
if [ "$TERMUX_ENV" = true ]; then
    echo "🔧 Setting up Termux environment..."
    
    # Acquire wake lock to prevent Android from killing the process
    if command -v termux-wake-lock >/dev/null 2>&1; then
        echo "🔒 Acquiring wake lock..."
        termux-wake-lock
    fi
    
    # Set optimal environment variables for Android
    export PYTHONUNBUFFERED=1
    export PYTHONIOENCODING=utf-8
    export TERM=xterm-256color
    
    # Optimize for mobile
    export OMP_NUM_THREADS=2
    export NUMEXPR_MAX_THREADS=2
fi

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down Enhanced Trading Bot..."
    
    if [ "$TERMUX_ENV" = true ] && command -v termux-wake-unlock >/dev/null 2>&1; then
        echo "🔓 Releasing wake lock..."
        termux-wake-unlock
    fi
    
    echo "✅ Cleanup complete. Goodbye!"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check for API keys and warn if missing
echo "🔑 Checking API configuration..."
if grep -q '"whale_alert_api_key": ""' config.json; then
    echo "⚠️  Whale Alert API key not configured - using free sources only"
fi

if grep -q '"api_key": ""' config.json; then
    echo "⚠️  Exchange API keys not configured - demo mode only"
fi

# Display startup information
echo ""
echo "🎯 Enhanced Features Available:"
echo "  🐋 Whale Alert Integration"
echo "  💥 Liquidation Tracking" 
echo "  🏦 Retail vs Big Money Detection"
echo "  📈 Historical Pattern Analysis"
echo "  ⏰ Dynamic Trade Durations"
echo "  🎯 Dynamic Confidence Adjustment"
echo "  📊 JSON Trade Tracking"
echo "  🎨 Rich Terminal UI"
echo ""

# Check system resources
if [ "$TERMUX_ENV" = true ]; then
    echo "📱 Android System Info:"
    echo "  Memory: $(cat /proc/meminfo | grep MemAvailable | awk '{print $2/1024 "MB"}')"
    echo "  Storage: $(df -h . | tail -1 | awk '{print $4}') free"
    echo ""
fi

# Start the enhanced bot
echo "🚀 Launching Enhanced Trading Bot..."
echo "=================================================="
echo ""

# Run with error handling
python3 enhanced_main.py

# Capture exit code
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "❌ Bot exited with error code: $exit_code"
    echo "📋 Check logs/enhanced_trading_bot.log for details"
    
    # Show last few lines of log for debugging
    if [ -f "logs/enhanced_trading_bot.log" ]; then
        echo ""
        echo "📋 Last 10 log entries:"
        echo "========================"
        tail -10 logs/enhanced_trading_bot.log
    fi
fi

# Run cleanup
cleanup