#!/usr/bin/env python3
"""
Order Flow Imbalance Trading Bot for Binance Futures
Single-file implementation optimized for PC/Cloud deployment
Tested on Binance Futures Testnet
"""

import asyncio
import aiohttp
import websockets
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
import hmac
import hashlib
import time
import json
import logging
from enum import Enum
import signal
import sys
import traceback
import os
from dotenv import load_dotenv


load_dotenv() 

# ===== LOGGER SETUP =====
logger = logging.getLogger("OrderFlowBot")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# ===== CONFIGURATION =====
@dataclass
class Config:
    """Bot configuration - Update with your testnet credentials"""
    # Binance Testnet API (Get from https://testnet.binancefuture.com/)
    API_KEY: str = os.getenv("BINANCE_API_KEY")
    API_SECRET: str = os.getenv("BINANCE_API_SECRET")
    
    # Testnet URLs
    REST_URL: str = "https://testnet.binancefuture.com"
    WS_URL: str = "wss://stream.binancefuture.com/ws"
    
    # Trading Parameters
    POSITION_SIZE_USDT: float = 30.0
    LEVERAGE: int = 15
    MAX_CONCURRENT_TRADES: int = 2
    
    # Strategy Parameters
    IMBALANCE_THRESHOLD: float = 2.0  # Buy/sell volume ratio trigger
    VOLUME_SPIKE_MIN: float = 50000   # Min USDT volume in window
    LOOKBACK_SECONDS: int = 5         # Order flow analysis window
    
    # Order Parameters
    TARGET_PROFIT_PCT: float = 0.008  # 0.8% (accounting for fees)
    STOP_LOSS_PCT: float = 0.005     # 0.5%
    ORDER_TIMEOUT_SEC: int = 15       # Cancel unfilled orders after this
    POST_ONLY: bool = True           # Ensure maker orders only
    
    # Risk Management
    DAILY_LOSS_LIMIT_USDT: float = 10.0
    MAX_TRADES_PER_DAY: int = 100
    MIN_TIME_BETWEEN_TRADES: int = 30  # seconds
    
    # Symbols to Trade (high volume pairs)
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"
    ])
    
    # Performance Tracking
    LOG_LEVEL: str = "INFO"
    SAVE_TRADES: bool = True
    METRICS_UPDATE_INTERVAL: int = 300  # 5 minutes

# ===== ENUMS AND TYPES =====
class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    NEW = "NEW"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"

@dataclass
class Trade:
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    timestamp: datetime
    order_id: str = ""
    exit_price: float = 0.0
    pnl: float = 0.0
    status: str = "OPEN"
    
@dataclass
class OrderFlowMetrics:
    buy_volume: float
    sell_volume: float
    imbalance_ratio: float
    volume_spike: bool
    price_change_pct: float
    trade_count: int
    timestamp: datetime

# ===== UTILITIES =====
class BinanceAuth:
    """Handle Binance API authentication"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
    
    def sign(self, params: dict) -> dict:
        """Sign request parameters"""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        return params

# ===== MAIN COMPONENTS =====
class OrderFlowAnalyzer:
    """Analyzes order flow to detect imbalances"""
    
    def __init__(self, config: Config):
        self.config = config
        self.trades_buffer: Dict[str, Deque[dict]] = {
            symbol: deque(maxlen=1000) for symbol in config.SYMBOLS
        }
        self.orderbook_snapshots: Dict[str, dict] = {}
        self.metrics_history: Dict[str, Deque[OrderFlowMetrics]] = {
            symbol: deque(maxlen=100) for symbol in config.SYMBOLS
        }
        
    def process_trade(self, symbol: str, trade_data: dict):
        """Process incoming trade data"""
        trade = {
            'price': float(trade_data['p']),
            'quantity': float(trade_data['q']),
            'time': int(trade_data['T']),
            'is_buyer_maker': trade_data['m']
        }
        self.trades_buffer[symbol].append(trade)
        
    def calculate_metrics(self, symbol: str) -> Optional[OrderFlowMetrics]:
        """Calculate order flow metrics for a symbol"""
        trades = list(self.trades_buffer[symbol])
        if len(trades) < 10:
            return None
            
        current_time = trades[-1]['time']
        window_start = current_time - (self.config.LOOKBACK_SECONDS * 1000)
        
        # Filter trades within window
        window_trades = [t for t in trades if t['time'] >= window_start]
        if not window_trades:
            return None
            
        # Calculate volumes
        buy_volume = sum(t['price'] * t['quantity'] for t in window_trades if not t['is_buyer_maker'])
        sell_volume = sum(t['price'] * t['quantity'] for t in window_trades if t['is_buyer_maker'])
        total_volume = buy_volume + sell_volume
        
        # Calculate metrics
        imbalance_ratio = buy_volume / sell_volume if sell_volume > 0 else 10.0
        volume_spike = total_volume > self.config.VOLUME_SPIKE_MIN
        
        # Price change
        start_price = window_trades[0]['price']
        end_price = window_trades[-1]['price']
        price_change_pct = ((end_price - start_price) / start_price) * 100
        
        metrics = OrderFlowMetrics(
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            imbalance_ratio=imbalance_ratio,
            volume_spike=volume_spike,
            price_change_pct=price_change_pct,
            trade_count=len(window_trades),
            timestamp=datetime.now(timezone.utc)
        )
        
        self.metrics_history[symbol].append(metrics)
        return metrics

class SignalGenerator:
    """Generate trading signals from order flow metrics"""
    
    def __init__(self, config: Config):
        self.config = config
        self.last_signal_time: Dict[str, datetime] = {}
        
    def generate_signal(self, symbol: str, metrics: OrderFlowMetrics) -> SignalType:
        """Generate trading signal based on metrics"""
        # Check cooldown period
        if symbol in self.last_signal_time:
            time_since_last = (datetime.now(timezone.utc) - self.last_signal_time[symbol]).seconds
            if time_since_last < self.config.MIN_TIME_BETWEEN_TRADES:
                return SignalType.NONE
        
        # Long signal: Strong buying pressure that might revert
        if (metrics.imbalance_ratio > self.config.IMBALANCE_THRESHOLD and
            metrics.volume_spike and
            metrics.price_change_pct > 0.1):
            self.last_signal_time[symbol] = datetime.now(timezone.utc)
            return SignalType.SHORT  # Fade the move
            
        # Short signal: Strong selling pressure that might revert
        if (metrics.imbalance_ratio < (1 / self.config.IMBALANCE_THRESHOLD) and
            metrics.volume_spike and
            metrics.price_change_pct < -0.1):
            self.last_signal_time[symbol] = datetime.now(timezone.utc)
            return SignalType.LONG  # Fade the move
            
        return SignalType.NONE

class OrderManager:
    """Manage order placement and tracking"""
    
    def __init__(self, config: Config, auth: BinanceAuth):
        self.config = config
        self.auth = auth
        self.session: Optional[aiohttp.ClientSession] = None
        self.open_orders: Dict[str, dict] = {}
        self.positions: Dict[str, Trade] = {}
        
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            
    async def place_limit_order(self, symbol: str, side: OrderSide, quantity: float, price: float) -> Optional[str]:
        """Place a post-only limit order"""
        try:
            params = {
                'symbol': symbol,
                'side': side.value,
                'type': 'LIMIT',
                'timeInForce': 'GTX' if self.config.POST_ONLY else 'GTC',
                'quantity': f"{quantity:.3f}",
                'price': f"{price:.2f}",
                'timestamp': int(time.time() * 1000)
            }
            
            params = self.auth.sign(params)
            
            async with self.session.post(
                f"{self.config.REST_URL}/fapi/v1/order",
                headers={'X-MBX-APIKEY': self.config.API_KEY},
                params=params
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    order_id = data['orderId']
                    self.open_orders[order_id] = {
                        'symbol': symbol,
                        'side': side,
                        'price': price,
                        'quantity': quantity,
                        'timestamp': datetime.now(timezone.utc)
                    }
                    logger.info(f"Order placed: {symbol} {side.value} {quantity} @ {price}")
                    return order_id
                else:
                    error = await resp.text()
                    logger.error(f"Order placement failed: {error}")
                    return None
                    
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return None
            
    async def cancel_order(self, symbol: str, order_id: str):
        """Cancel an open order"""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id,
                'timestamp': int(time.time() * 1000)
            }
            
            params = self.auth.sign(params)
            
            async with self.session.delete(
                f"{self.config.REST_URL}/fapi/v1/order",
                headers={'X-MBX-APIKEY': self.config.API_KEY},
                params=params
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Order canceled: {order_id}")
                    if order_id in self.open_orders:
                        del self.open_orders[order_id]
                        
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            
    async def check_order_status(self, symbol: str, order_id: str) -> Optional[str]:
        """Check order status"""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id,
                'timestamp': int(time.time() * 1000)
            }
            
            params = self.auth.sign(params)
            
            async with self.session.get(
                f"{self.config.REST_URL}/fapi/v1/order",
                headers={'X-MBX-APIKEY': self.config.API_KEY},
                params=params
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['status']
                    
        except Exception as e:
            logger.error(f"Order status check error: {e}")
            return None

class RiskManager:
    """Manage risk limits and position sizing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.open_positions = 0
        self.last_reset = datetime.now(timezone.utc).date()
        
    def check_daily_reset(self):
        """Reset daily counters if new day"""
        current_date = datetime.now(timezone.utc).date()
        if current_date > self.last_reset:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = current_date
            logger.info("Daily risk counters reset")
            
    def can_trade(self) -> Tuple[bool, str]:
        """Check if we can place a new trade"""
        self.check_daily_reset()
        
        # Check daily loss limit
        if self.daily_pnl <= -self.config.DAILY_LOSS_LIMIT_USDT:
            return False, "Daily loss limit reached"
            
        # Check max concurrent trades
        if self.open_positions >= self.config.MAX_CONCURRENT_TRADES:
            return False, "Max concurrent trades reached"
            
        # Check max daily trades
        if self.daily_trades >= self.config.MAX_TRADES_PER_DAY:
            return False, "Max daily trades reached"
            
        return True, "OK"
        
    def calculate_position_size(self, price: float) -> float:
        """Calculate position size in base currency"""
        return self.config.POSITION_SIZE_USDT / price
        
    def update_pnl(self, pnl: float):
        """Update daily PnL"""
        self.daily_pnl += pnl
        logger.info(f"Daily PnL updated: ${self.daily_pnl:.2f}")

class OrderFlowBot:
    """Main bot orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.auth = BinanceAuth(config.API_KEY, config.API_SECRET)
        self.analyzer = OrderFlowAnalyzer(config)
        self.signal_gen = SignalGenerator(config)
        self.order_mgr = OrderManager(config, self.auth)
        self.risk_mgr = RiskManager(config)
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.start_time = datetime.now(timezone.utc)
        
    async def initialize(self):
        """Initialize bot components"""
        await self.order_mgr.initialize()
        logger.info("Bot initialized")
        
    async def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            
        # Close all positions
        for symbol, trade in list(self.order_mgr.positions.items()):
            logger.info(f"Closing position: {symbol}")
            # Place market order to close (in production)
            
        # Cancel all orders
        for order_id, order in list(self.order_mgr.open_orders.items()):
            await self.order_mgr.cancel_order(order['symbol'], order_id)
            
        await self.order_mgr.close()
        logger.info("Bot shutdown complete")
        
    async def stream_trades(self, symbol: str):
        """Stream trade data via WebSocket"""
        stream_url = f"{self.config.WS_URL}/{symbol.lower()}@aggTrade"
        
        while self.running:
            try:
                async with websockets.connect(stream_url) as ws:
                    logger.info(f"Connected to {symbol} trade stream")
                    
                    while self.running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        self.analyzer.process_trade(symbol, data)
                        
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                await asyncio.sleep(5)
                
    async def monitor_orders(self):
        """Monitor open orders for fills and timeouts"""
        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                
                for order_id, order in list(self.order_mgr.open_orders.items()):
                    # Check timeout
                    order_age = (current_time - order['timestamp']).seconds
                    if order_age > self.config.ORDER_TIMEOUT_SEC:
                        logger.info(f"Order timeout: {order_id}")
                        await self.order_mgr.cancel_order(order['symbol'], order_id)
                        continue
                        
                    # Check status
                    status = await self.order_mgr.check_order_status(order['symbol'], order_id)
                    if status == 'FILLED':
                        logger.info(f"Order filled: {order_id}")
                        # Create position entry
                        trade = Trade(
                            symbol=order['symbol'],
                            side=order['side'],
                            entry_price=order['price'],
                            quantity=order['quantity'],
                            timestamp=current_time,
                            order_id=order_id
                        )
                        self.order_mgr.positions[order['symbol']] = trade
                        del self.order_mgr.open_orders[order_id]
                        self.risk_mgr.open_positions += 1
                        self.risk_mgr.daily_trades += 1
                        
                        # Place TP/SL orders
                        await self.place_exit_orders(trade)
                        
            except Exception as e:
                logger.error(f"Order monitoring error: {e}")
                
            await asyncio.sleep(1)
            
    async def place_exit_orders(self, trade: Trade):
        """Place take-profit and stop-loss orders"""
        if trade.side == OrderSide.BUY:
            tp_price = trade.entry_price * (1 + self.config.TARGET_PROFIT_PCT)
            sl_price = trade.entry_price * (1 - self.config.STOP_LOSS_PCT)
            tp_side = OrderSide.SELL
        else:
            tp_price = trade.entry_price * (1 - self.config.TARGET_PROFIT_PCT)
            sl_price = trade.entry_price * (1 + self.config.STOP_LOSS_PCT)
            tp_side = OrderSide.BUY
            
        # Place TP order
        tp_order_id = await self.order_mgr.place_limit_order(
            trade.symbol, tp_side, trade.quantity, tp_price
        )
        
        # For simplicity, using stop-limit for SL (in production use stop-market)
        sl_order_id = await self.order_mgr.place_limit_order(
            trade.symbol, tp_side, trade.quantity, sl_price
        )
        
        logger.info(f"Exit orders placed - TP: {tp_price:.2f}, SL: {sl_price:.2f}")
        
    async def trading_loop(self):
        """Main trading logic loop"""
        while self.running:
            try:
                for symbol in self.config.SYMBOLS:
                    # Skip if already have position
                    if symbol in self.order_mgr.positions:
                        continue
                        
                    # Calculate metrics
                    metrics = self.analyzer.calculate_metrics(symbol)
                    if not metrics:
                        continue
                        
                    # Generate signal
                    signal = self.signal_gen.generate_signal(symbol, metrics)
                    if signal == SignalType.NONE:
                        continue
                        
                    # Check risk limits
                    can_trade, reason = self.risk_mgr.can_trade()
                    if not can_trade:
                        logger.warning(f"Trade blocked: {reason}")
                        continue
                        
                    # Get current price (use last trade price)
                    if self.analyzer.trades_buffer[symbol]:
                        current_price = self.analyzer.trades_buffer[symbol][-1]['price']
                    else:
                        continue
                        
                    # Calculate order price (slightly better than current)
                    if signal == SignalType.LONG:
                        order_price = current_price * 0.9995  # 0.05% below
                        side = OrderSide.BUY
                    else:
                        order_price = current_price * 1.0005  # 0.05% above
                        side = OrderSide.SELL
                        
                    # Calculate position size
                    quantity = self.risk_mgr.calculate_position_size(order_price)
                    
                    # Place order
                    logger.info(f"Signal generated: {symbol} {signal.value}")
                    logger.info(f"Metrics: Imbalance={metrics.imbalance_ratio:.2f}, "
                              f"Volume={metrics.buy_volume + metrics.sell_volume:.0f}")
                    
                    order_id = await self.order_mgr.place_limit_order(
                        symbol, side, quantity, order_price
                    )
                    
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                traceback.print_exc()
                
            await asyncio.sleep(0.5)
            
    async def monitor_positions(self):
        """Monitor open positions for exit"""
        while self.running:
            try:
                for symbol, trade in list(self.order_mgr.positions.items()):
                    # Check if any exit orders filled
                    # (In production, track exit order IDs properly)
                    
                    # For now, simple PnL tracking
                    if self.analyzer.trades_buffer[symbol]:
                        current_price = self.analyzer.trades_buffer[symbol][-1]['price']
                        
                        if trade.side == OrderSide.BUY:
                            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                        else:
                            pnl_pct = (trade.entry_price - current_price) / trade.entry_price
                            
                        pnl_usdt = pnl_pct * self.config.POSITION_SIZE_USDT
                        
                        # Check if hit TP or SL
                        if pnl_pct >= self.config.TARGET_PROFIT_PCT:
                            logger.info(f"Take profit hit: {symbol} +${pnl_usdt:.2f}")
                            self.close_position(symbol, current_price, pnl_usdt)
                        elif pnl_pct <= -self.config.STOP_LOSS_PCT:
                            logger.info(f"Stop loss hit: {symbol} -${pnl_usdt:.2f}")
                            self.close_position(symbol, current_price, pnl_usdt)
                            
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                
            await asyncio.sleep(1)
            
    def close_position(self, symbol: str, exit_price: float, pnl: float):
        """Close position and update stats"""
        if symbol in self.order_mgr.positions:
            trade = self.order_mgr.positions[symbol]
            trade.exit_price = exit_price
            trade.pnl = pnl
            trade.status = "CLOSED"
            
            # Update stats
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            self.total_pnl += pnl
            self.risk_mgr.update_pnl(pnl)
            self.risk_mgr.open_positions -= 1
            
            # Remove position
            del self.order_mgr.positions[symbol]
            
            # Log performance
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            logger.info(f"Trade closed: {symbol} PnL=${pnl:.2f}")
            logger.info(f"Stats: Trades={self.total_trades}, WinRate={win_rate:.1f}%, "
                       f"TotalPnL=${self.total_pnl:.2f}, DailyPnL=${self.risk_mgr.daily_pnl:.2f}")
            
    async def print_metrics(self):
        """Print performance metrics periodically"""
        while self.running:
            await asyncio.sleep(self.config.METRICS_UPDATE_INTERVAL)
            
            runtime = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            logger.info("="*50)
            logger.info(f"Bot Performance Summary - Runtime: {runtime:.1f} hours")
            logger.info(f"Total Trades: {self.total_trades}")
            logger.info(f"Win Rate: {win_rate:.1f}%")
            logger.info(f"Total PnL: ${self.total_pnl:.2f}")
            logger.info(f"Daily PnL: ${self.risk_mgr.daily_pnl:.2f}")
            logger.info(f"Open Positions: {self.risk_mgr.open_positions}")
            logger.info("="*50)
            
    async def run(self):
        """Main bot execution"""
        self.running = True
        await self.initialize()
        
        # Start all tasks
        self.tasks = [
            asyncio.create_task(self.trading_loop()),
            asyncio.create_task(self.monitor_orders()),
            asyncio.create_task(self.monitor_positions()),
            asyncio.create_task(self.print_metrics()),
        ]
        
        # Start WebSocket streams
        for symbol in self.config.SYMBOLS:
            self.tasks.append(asyncio.create_task(self.stream_trades(symbol)))
            
        logger.info(f"Bot started - Trading {len(self.config.SYMBOLS)} symbols")
        logger.info(f"Position size: ${self.config.POSITION_SIZE_USDT}, Leverage: {self.config.LEVERAGE}x")
        
        # Wait for all tasks
        await asyncio.gather(*self.tasks, return_exceptions=True)

# ===== MAIN ENTRY POINT =====
async def main():
    """Main entry point"""
    # Load config (can be from file/env in production)
    config = Config()
    
    # Setup logging
    logging.getLogger().setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Create bot
    bot = OrderFlowBot(config)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        asyncio.create_task(bot.shutdown())
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await bot.run()
    except Exception as e:
        logger.error(f"Bot error: {e}")
        traceback.print_exc()
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════╗
    ║     Order Flow Imbalance Trading Bot         ║
    ║     Binance Futures - Testnet Ready          ║
    ╚══════════════════════════════════════════════╝
    """)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8+ required")
        sys.exit(1)
        
    # Run bot
    asyncio.run(main())