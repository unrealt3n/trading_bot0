"""
🚀 Crypto Trading Bot Configuration
Centralized settings for signal confidence system
"""

import os
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class APIConfig:
    """API configuration settings"""
    # Deribit API
    deribit_base_url: str = "https://www.deribit.com/api/v2"
    deribit_test_url: str = "https://test.deribit.com/api/v2"
    deribit_api_key: str = os.getenv("DERIBIT_API_KEY", "")
    deribit_api_secret: str = os.getenv("DERIBIT_API_SECRET", "")
    
    # CoinGlass API
    coinglass_base_url: str = "https://open-api-v4.coinglass.com"
    coinglass_api_key: str = os.getenv("COINGLASS_API_KEY", "")
    
    # Gemini API - REMOVED
    # gemini_base_url: str = "https://api.gemini.com"
    # gemini_sandbox_url: str = "https://api.sandbox.gemini.com"
    # gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    # gemini_api_secret: str = os.getenv("GEMINI_API_SECRET", "")
    
    # Exchange APIs (public endpoints)
    binance_base_url: str = "https://api.binance.com"
    okx_base_url: str = "https://www.okx.com"
    bybit_base_url: str = "https://api.bybit.com"
    coinbase_base_url: str = "https://api.exchange.coinbase.com"
    kraken_base_url: str = "https://api.kraken.com"
    bitget_base_url: str = "https://api.bitget.com"

@dataclass
class WeightConfig:
    """Enhanced signal weighting configuration with smart money detection"""
    # Core signals (Enhanced)
    options_flow: float = 0.25          # 📊 Options chain analysis (reduced to accommodate new signals)
    smart_money_flow: float = 0.25      # 🧠 NEW: Retail vs institutional detection
    multi_exchange: float = 0.20        # 🌐 NEW: Enhanced multi-exchange analysis
    technical_indicators: float = 0.12  # 📈 RSI, MACD, EMA, Bollinger (reduced)
    session_context: float = 0.08       # 🕐 Market session adjustments (reduced)
    news_sentiment: float = 0.05        # 📰 Reddit/Twitter sentiment (reduced due to API limits)
    # gemini_bonus: float = 0.05          # 🧠 Gemini API data bonus - REMOVED
    
    # Future additions (Phase 2+)
    funding_rates: float = 0.0          # 💰 Perpetual funding rates
    trend_alignment: float = 0.0        # 🎯 Multi-timeframe trends
    volatility_regime: float = 0.0      # 📊 Volatility regime detection
    
    def normalize_weights(self) -> Dict[str, float]:
        """Ensure weights sum to 1.0"""
        total = sum([
            self.options_flow, self.smart_money_flow, self.multi_exchange,
            self.technical_indicators, self.news_sentiment,
            self.session_context
        ])
        return {
            'options_flow': self.options_flow / total,
            'smart_money_flow': self.smart_money_flow / total,
            'multi_exchange': self.multi_exchange / total,
            'technical_indicators': self.technical_indicators / total,
            'news_sentiment': self.news_sentiment / total,
            'session_context': self.session_context / total
        }

@dataclass
class TradingConfig:
    """Trading and risk management settings"""
    # Supported coins
    supported_coins: List[str] = None
    
    # Confidence thresholds
    min_confidence_to_trade: float = 0.65  # Don't trade below 65% confidence
    high_confidence_threshold: float = 0.85  # High confidence trades
    
    # Risk management
    max_concurrent_trades: int = 3
    position_size_pct: float = 0.02  # 2% of portfolio per trade
    stop_loss_pct: float = 0.03      # 3% stop loss
    take_profit_pct: float = 0.06    # 6% take profit (2:1 RR)
    
    # Session trading hours (UTC)
    asia_session: tuple = (0, 8)     # 00:00 - 08:00 UTC
    europe_session: tuple = (7, 16)  # 07:00 - 16:00 UTC  
    us_session: tuple = (13, 22)     # 13:00 - 22:00 UTC
    
    def __post_init__(self):
        if self.supported_coins is None:
            self.supported_coins = ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOT"]

@dataclass  
class SystemConfig:
    """System and logging configuration"""
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/trading_bot.log"
    trade_log_file: str = "logs/trade_attempts.jsonl"
    
    # HTTP settings
    request_timeout: int = 30        # seconds
    max_retries: int = 3
    retry_delay: float = 1.0         # seconds
    rate_limit_delay: float = 0.1    # seconds between requests
    
    # Performance
    max_concurrent_requests: int = 10
    cache_ttl: int = 60              # seconds
    
    # Error handling
    max_consecutive_errors: int = 5
    error_cooldown: int = 300        # 5 minutes

# Global configuration instances
api_config = APIConfig()
weight_config = WeightConfig()
trading_config = TradingConfig()
system_config = SystemConfig()

# Constants
SUPPORTED_EXCHANGES = [
    "binance", "okx", "bybit", 
    "coinbase", "kraken"
    # "bitget"  # Temporarily disabled due to API parameter issues
]

SESSION_VOLATILITY_MULTIPLIERS = {
    "asia": 0.8,     # Lower volatility typically
    "europe": 1.0,   # Normal volatility
    "us": 1.2,       # Higher volatility during US hours
    "overlap": 1.1   # Session overlaps
}

COIN_OPTION_AVAILABILITY = {
    "BTC": True,   # Full options data on Deribit
    "ETH": True,   # Full options data on Deribit  
    "SOL": False,  # Use funding rates instead
    "XRP": False,  # Use funding rates instead
    "BNB": False,  # Use funding rates instead
    "ADA": False,  # Use funding rates instead
    "DOT": False   # Use funding rates instead
}