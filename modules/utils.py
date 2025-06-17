"""
🛠️ Utility functions for the crypto trading bot
HTTP client, error handling, and common functions
"""

import asyncio
import aiohttp
import logging
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import hmac
import hashlib
import base64
from config.settings import system_config

# Configure logging
logging.basicConfig(
    level=getattr(logging, system_config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(system_config.log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom exception for API-related errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(self.message)

class HTTPClient:
    """Async HTTP client with error handling and rate limiting"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0
        self.consecutive_errors = 0
        
    async def __aenter__(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=system_config.request_timeout)
            connector = aiohttp.TCPConnector(limit=system_config.max_concurrent_requests)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'CryptoTradingBot/1.0'}
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _rate_limit(self):
        """Implement rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < system_config.rate_limit_delay:
            await asyncio.sleep(system_config.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request with error handling and retries"""
        return await self._request('GET', url, params=params, headers=headers)
    
    async def post(self, url: str, data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make POST request with error handling and retries"""
        return await self._request('POST', url, json=data, headers=headers)
    
    async def _request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Internal request method with retry logic"""
        
        # Check consecutive errors
        if self.consecutive_errors >= system_config.max_consecutive_errors:
            logger.error(f"🚨 Too many consecutive errors ({self.consecutive_errors}). Cooling down...")
            await asyncio.sleep(system_config.error_cooldown)
            self.consecutive_errors = 0
        
        await self._rate_limit()
        
        for attempt in range(system_config.max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    
                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"⏳ Rate limited. Waiting {retry_after}s before retry...")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    # Handle successful responses
                    if response.status == 200:
                        self.consecutive_errors = 0  # Reset error counter
                        try:
                            return await response.json()
                        except json.JSONDecodeError:
                            text = await response.text()
                            logger.error(f"❌ Invalid JSON response from {url}: {text[:200]}")
                            raise APIError(f"Invalid JSON response from {url}")
                    
                    # Handle error responses
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ HTTP {response.status} from {url}: {error_text[:200]}")
                        
                        if response.status >= 500:  # Server errors - retry
                            if attempt < system_config.max_retries - 1:
                                await asyncio.sleep(system_config.retry_delay * (2 ** attempt))
                                continue
                        
                        # Client errors - don't retry
                        raise APIError(
                            f"HTTP {response.status} from {url}",
                            status_code=response.status,
                            response_data={"error": error_text}
                        )
                        
            except aiohttp.ClientError as e:
                self.consecutive_errors += 1
                logger.error(f"❌ Network error (attempt {attempt + 1}): {str(e)}")
                
                if attempt < system_config.max_retries - 1:
                    await asyncio.sleep(system_config.retry_delay * (2 ** attempt))
                else:
                    raise APIError(f"Network error after {system_config.max_retries} attempts: {str(e)}")
        
        raise APIError(f"Max retries exceeded for {url}")

def get_current_session() -> str:
    """Determine current market session based on UTC time"""
    from config.settings import trading_config
    
    current_hour = datetime.now(timezone.utc).hour
    
    # Check for session overlaps first
    if (trading_config.europe_session[0] <= current_hour <= trading_config.europe_session[1] and
        trading_config.us_session[0] <= current_hour <= trading_config.us_session[1]):
        return "overlap"
    
    # Individual sessions
    if trading_config.asia_session[0] <= current_hour <= trading_config.asia_session[1]:
        return "asia"
    elif trading_config.europe_session[0] <= current_hour <= trading_config.europe_session[1]:
        return "europe"
    elif trading_config.us_session[0] <= current_hour <= trading_config.us_session[1]:
        return "us"
    else:
        return "asia"  # Default to Asia for off-hours

def calculate_volatility_multiplier(session: str) -> float:
    """Get volatility multiplier for current session"""
    from config.settings import SESSION_VOLATILITY_MULTIPLIERS
    return SESSION_VOLATILITY_MULTIPLIERS.get(session, 1.0)

def normalize_signal(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Normalize signal to 0-1 range"""
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))

def calculate_signal_strength(signals: List[float]) -> float:
    """Calculate overall signal strength from multiple indicators"""
    if not signals:
        return 0.0
    
    # Remove outliers (outside 2 standard deviations)
    import statistics
    if len(signals) > 2:
        mean = statistics.mean(signals)
        stdev = statistics.stdev(signals)
        filtered_signals = [s for s in signals if abs(s - mean) <= 2 * stdev]
        if filtered_signals:
            signals = filtered_signals
    
    return statistics.mean(signals)

def create_hmac_signature(secret: str, message: str) -> str:
    """Create HMAC signature for authenticated API requests"""
    return hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def create_base64_signature(secret: str, message: str) -> str:
    """Create base64 encoded signature for Gemini API"""
    signature = hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha384
    ).hexdigest()
    return base64.b64encode(signature.encode()).decode()

async def safe_api_call(api_func, default_return: Any = None, error_message: str = "API call failed"):
    """Safely call API function with error handling"""
    try:
        return await api_func()
    except Exception as e:
        logger.error(f"❌ {error_message}: {str(e)}")
        return default_return

def log_trade_attempt(trade_data: Dict[str, Any]):
    """Log trade attempt to JSONL file"""
    try:
        with open(system_config.trade_log_file, 'a') as f:
            f.write(json.dumps(trade_data) + '\n')
    except Exception as e:
        logger.error(f"❌ Failed to log trade attempt: {str(e)}")

def format_number(number: float, decimals: int = 2) -> str:
    """Format number with appropriate decimal places"""
    if abs(number) >= 1e9:
        return f"{number/1e9:.{decimals}f}B"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.{decimals}f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.{decimals}f}K"
    else:
        return f"{number:.{decimals}f}"

def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat()

# Emoji helpers for Windows terminal compatibility
def get_emoji(name: str) -> str:
    """Get emoji with fallback for Windows terminals"""
    emoji_map = {
        'bull': '🔼',
        'bear': '🔽', 
        'neutral': '➖',
        'high_confidence': '🎯',
        'low_confidence': '❓',
        'options': '📊',
        'orderbook': '⚡',
        'indicators': '📈',
        'news': '📰',
        'session': '🕐',
        # 'gemini': '🧠', # Removed
        'success': '✅',
        'warning': '⚠️',
        'error': '❌',
        'info': 'ℹ️',
        'rocket': '🚀',
        'money': '💰'
    }
    return emoji_map.get(name, '•')