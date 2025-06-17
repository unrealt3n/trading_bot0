"""
🚀 Binance Testnet Trade Executor
Executes real trades on Binance Testnet with proper TP/SL management
"""

import asyncio
import logging
import hmac
import hashlib
import time
import json
from typing import Dict, Any, Optional, List
from decimal import Decimal, ROUND_DOWN
import aiohttp
from datetime import datetime

from .utils import HTTPClient
from .trade_tracker import TradeTracker, Trade, TradeType, TradeStatus

logger = logging.getLogger(__name__)

class BinanceTestnetExecutor:
    """
    Executes trades on Binance Testnet with TP/SL management
    """
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://testnet.binance.vision"  # Binance Testnet
        
        # Trading configuration
        self.trade_amount_usd = 20.0  # $20 per trade in demo mode
        self.min_notional = 5.0       # Minimum $5 trade
        
        # Rate limiting
        self.last_request_time = 0
        self.request_delay = 0.1      # 100ms between requests
        
        logger.info(f"🚀 Binance Testnet Executor initialized - Trade amount: ${self.trade_amount_usd}")
    
    def _create_signature(self, query_string: str) -> str:
        """Create HMAC SHA256 signature for Binance API"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _make_signed_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make signed request to Binance API"""
        
        if params is None:
            params = {}
        
        # Add timestamp
        params['timestamp'] = int(time.time() * 1000)
        
        # Create query string
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        
        # Create signature
        signature = self._create_signature(query_string)
        query_string += f"&signature={signature}"
        
        # Headers
        headers = {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        # Rate limiting
        now = time.time()
        if now - self.last_request_time < self.request_delay:
            await asyncio.sleep(self.request_delay - (now - self.last_request_time))
        self.last_request_time = time.time()
        
        # Make request
        url = f"{self.base_url}{endpoint}?{query_string}"
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers) as response:
                result = await response.json()
                
                if response.status != 200:
                    logger.error(f"❌ Binance API error: {result}")
                    raise Exception(f"Binance API error: {result}")
                
                return result
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balances"""
        try:
            result = await self._make_signed_request('GET', '/api/v3/account')
            
            balances = {}
            for balance in result['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                if free > 0:
                    balances[asset] = free
            
            return balances
            
        except Exception as e:
            logger.error(f"❌ Error getting account balance: {e}")
            return {}
    
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get trading symbol information"""
        try:
            # Use public endpoint (no signature needed)
            url = f"{self.base_url}/api/v3/exchangeInfo"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    result = await response.json()
            
            for sym in result['symbols']:
                if sym['symbol'] == symbol and sym['status'] == 'TRADING':
                    return sym
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error getting symbol info for {symbol}: {e}")
            return {}
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {'symbol': symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    result = await response.json()
                    return float(result['price'])
                    
        except Exception as e:
            logger.error(f"❌ Error getting price for {symbol}: {e}")
            return 0.0
    
    def _calculate_quantity(self, symbol_info: Dict, price: float, usd_amount: float) -> str:
        """Calculate proper quantity based on symbol filters"""
        
        # Get lot size filter
        lot_size_filter = None
        min_notional_filter = None
        
        for filter_item in symbol_info.get('filters', []):
            if filter_item['filterType'] == 'LOT_SIZE':
                lot_size_filter = filter_item
            elif filter_item['filterType'] == 'MIN_NOTIONAL':
                min_notional_filter = filter_item
        
        if not lot_size_filter:
            logger.error(f"❌ No LOT_SIZE filter found for {symbol_info.get('symbol')}")
            return "0"
        
        # Calculate base quantity
        base_quantity = usd_amount / price
        
        # Apply lot size constraints
        min_qty = float(lot_size_filter['minQty'])
        max_qty = float(lot_size_filter['maxQty'])
        step_size = float(lot_size_filter['stepSize'])
        
        # Ensure minimum quantity
        if base_quantity < min_qty:
            base_quantity = min_qty
        
        # Ensure maximum quantity
        if base_quantity > max_qty:
            base_quantity = max_qty
        
        # Round down to step size
        decimal_places = len(lot_size_filter['stepSize'].rstrip('0').split('.')[-1])
        quantity = Decimal(str(base_quantity)).quantize(
            Decimal(lot_size_filter['stepSize']), 
            rounding=ROUND_DOWN
        )
        
        return str(quantity)
    
    async def place_limit_order(self, symbol: str, side: str, quantity: str, price: str) -> Dict[str, Any]:
        """Place limit order for lower fees"""
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'LIMIT',
            'quantity': quantity,
            'price': price,
            'timeInForce': 'GTC'  # Good Till Canceled
        }
        
        try:
            result = await self._make_signed_request('POST', '/api/v3/order', params)
            logger.info(f"✅ Limit order placed: {side} {quantity} {symbol} @ ${price}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error placing limit order: {e}")
            return {}
    
    def _calculate_limit_price(self, current_price: float, side: str, aggressive: bool = True) -> float:
        """Calculate optimal limit price for immediate fill with lower fees"""
        
        # For aggressive fills (immediate execution), place slightly inside spread
        if aggressive:
            if side == "BUY":
                # Buy slightly above current price (but below market price)
                return current_price * 1.0005  # 0.05% above
            else:  # SELL
                # Sell slightly below current price (but above market price)
                return current_price * 0.9995  # 0.05% below
        else:
            # For passive fills (better price), place at current price
            return current_price
    
    async def place_oco_order(self, symbol: str, side: str, quantity: str, 
                             stop_price: str, stop_limit_price: str, price: str) -> Dict[str, Any]:
        """Place OCO (One-Cancels-Other) order for TP/SL with limit orders for lower fees"""
        
        params = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,  # Take profit limit price
            'stopPrice': stop_price,  # Stop loss trigger
            'stopLimitPrice': stop_limit_price,  # Stop loss limit price
            'stopLimitTimeInForce': 'GTC',  # Good Till Canceled
            'timeInForce': 'GTC'  # Take profit limit order timeInForce
        }
        
        try:
            result = await self._make_signed_request('POST', '/api/v3/order/oco', params)
            logger.info(f"✅ OCO limit orders placed: {side} {quantity} {symbol}")
            logger.info(f"   📈 Take Profit (LIMIT): ${price}")
            logger.info(f"   🛑 Stop Loss (STOP_LIMIT): ${stop_price} → ${stop_limit_price}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error placing OCO order: {e}")
            return {}
    
    async def execute_trade(self, analysis_result: Dict[str, Any], 
                          trade_tracker: TradeTracker) -> Optional[Trade]:
        """
        Execute trade based on analysis result
        
        Args:
            analysis_result: Result from enhanced_main_engine with trade decision
            trade_tracker: TradeTracker instance for recording trades
        
        Returns:
            Trade object if successful, None if failed
        """
        
        # Check if we should take the trade
        if not analysis_result.get('take_trade', False):
            return None
        
        coin = analysis_result.get('coin', 'BTC')
        bias = analysis_result.get('bias', 'bullish')
        confidence = analysis_result.get('confidence', 0.0)
        duration_data = analysis_result.get('duration_data', {})
        
        # Convert coin to Binance symbol
        symbol = f"{coin}USDT"
        
        logger.info(f"🚀 Executing {bias} trade for {symbol} - Confidence: {confidence:.1%}")
        
        try:
            # Get symbol info and current price
            symbol_info = await self.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"❌ Symbol {symbol} not found or not trading")
                return None
            
            current_price = await self.get_current_price(symbol)
            if current_price <= 0:
                logger.error(f"❌ Could not get current price for {symbol}")
                return None
            
            # Calculate quantity
            quantity = self._calculate_quantity(symbol_info, current_price, self.trade_amount_usd)
            if quantity == "0":
                logger.error(f"❌ Invalid quantity calculated for {symbol}")
                return None
            
            # Determine trade side
            side = "BUY" if bias == "bullish" else "SELL"
            
            # Calculate limit price for immediate fill with lower fees
            limit_price = self._calculate_limit_price(current_price, side, aggressive=True)
            
            # Format limit price for Binance (match price precision)
            price_precision = len(str(current_price).split('.')[-1])
            limit_price_str = f"{limit_price:.{price_precision}f}"
            
            # Place limit order to enter position (lower fees than market order)
            order_result = await self.place_limit_order(symbol, side, quantity, limit_price_str)
            if not order_result:
                logger.error(f"❌ Failed to place entry limit order for {symbol}")
                return None
            
            # For limit orders, use the limit price as fill price initially
            # Will be updated when order actually fills
            fill_price = limit_price
            
            logger.info(f"📝 Limit order placed for entry - waiting for fill...")
            
            # Wait a moment for order to potentially fill
            await asyncio.sleep(2)
            
            # Check if order filled
            order_id = order_result.get('orderId')
            if order_id:
                try:
                    order_status = await self._make_signed_request('GET', '/api/v3/order', {
                        'symbol': symbol,
                        'orderId': order_id
                    })
                    
                    if order_status.get('status') == 'FILLED':
                        # Get actual fill price from filled order
                        fill_price = float(order_status.get('price', limit_price))
                        logger.info(f"✅ Entry limit order filled @ ${fill_price:.4f}")
                    else:
                        logger.info(f"📝 Entry limit order pending - using limit price ${limit_price:.4f}")
                        fill_price = limit_price
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not check order status: {e}")
                    fill_price = limit_price
            
            # Create trade record
            trade = await trade_tracker.create_trade(
                coin=coin,
                analysis_data=analysis_result,
                duration_data=duration_data,
                entry_price=fill_price,
                amount=self.trade_amount_usd
            )
            
            # Calculate TP/SL prices from trade object
            tp_price = trade.take_profit_price
            sl_price = trade.stop_loss_price
            
            # Format prices for Binance (match price precision)
            price_precision = len(str(current_price).split('.')[-1])
            tp_price_str = f"{tp_price:.{price_precision}f}"
            sl_price_str = f"{sl_price:.{price_precision}f}"
            
            # Place OCO order for TP/SL
            exit_side = "SELL" if side == "BUY" else "BUY"
            
            oco_result = await self.place_oco_order(
                symbol=symbol,
                side=exit_side,
                quantity=quantity,
                stop_price=sl_price_str,
                stop_limit_price=sl_price_str,
                price=tp_price_str
            )
            
            if oco_result:
                logger.info(f"✅ Trade executed successfully for {symbol}")
                logger.info(f"   Entry: ${fill_price:.4f} | TP: ${tp_price:.4f} | SL: ${sl_price:.4f}")
                logger.info(f"   Duration: {duration_data.get('duration', 'short')} ({duration_data.get('duration_minutes', 240)}min)")
                
                # Store OCO order ID and entry order ID in trade for monitoring
                trade.oco_order_id = oco_result.get('orderListId')
                trade.entry_order_id = order_result.get('orderId')
                await trade_tracker.save_trades()
                
                return trade
            else:
                logger.warning(f"⚠️ Entry order placed but TP/SL failed for {symbol}")
                return trade
            
        except Exception as e:
            logger.error(f"❌ Error executing trade for {symbol}: {e}")
            return None
    
    async def monitor_orders(self, trade_tracker: TradeTracker):
        """Monitor both entry limit orders and OCO orders"""
        
        try:
            active_trades = await trade_tracker.get_active_trades()
            
            for trade in active_trades:
                symbol = f"{trade.coin}USDT"
                
                # Check entry order status (if it's a limit order that might still be pending)
                if hasattr(trade, 'entry_order_id') and trade.entry_order_id:
                    await self._check_entry_order(trade, symbol, trade_tracker)
                
                # Check OCO order status (for TP/SL completion)
                if hasattr(trade, 'oco_order_id') and trade.oco_order_id:
                    await self._check_oco_order(trade, symbol, trade_tracker)
                    
        except Exception as e:
            logger.error(f"❌ Error monitoring orders: {e}")
    
    async def _check_entry_order(self, trade, symbol: str, trade_tracker: TradeTracker):
        """Check if entry limit order has filled"""
        
        try:
            order_status = await self._make_signed_request('GET', '/api/v3/order', {
                'symbol': symbol,
                'orderId': trade.entry_order_id
            })
            
            if order_status.get('status') == 'FILLED':
                # Update trade with actual fill price
                actual_fill_price = float(order_status.get('price', trade.entry_price))
                if abs(actual_fill_price - trade.entry_price) > 0.0001:  # Price changed
                    logger.info(f"📝 Entry order filled for {trade.id}: ${actual_fill_price:.4f} (was ${trade.entry_price:.4f})")
                    # Update trade entry price
                    trade.entry_price = actual_fill_price
                    await trade_tracker.save_trades()
                
                # Clear entry order ID since it's filled
                trade.entry_order_id = None
                
        except Exception as e:
            logger.warning(f"⚠️ Error checking entry order for {trade.id}: {e}")
    
    async def _check_oco_order(self, trade, symbol: str, trade_tracker: TradeTracker):
        """Check if OCO order (TP/SL) has completed"""
        
        try:
            result = await self._make_signed_request('GET', '/api/v3/orderList', {
                'orderListId': trade.oco_order_id
            })
            
            if result.get('listStatusType') == 'ALL_DONE':
                # OCO completed - one of TP or SL was hit
                executed_orders = [order for order in result.get('orders', []) if order.get('status') == 'FILLED']
                
                if executed_orders:
                    executed_order = executed_orders[0]
                    exit_price = float(executed_order.get('price', 0))
                    
                    # Determine if TP or SL was hit
                    if abs(exit_price - trade.take_profit_price) < abs(exit_price - trade.stop_loss_price):
                        status = TradeStatus.COMPLETED_TP
                        logger.info(f"🎯 Take Profit (LIMIT) hit for {trade.id}: ${exit_price:.4f}")
                    else:
                        status = TradeStatus.COMPLETED_SL
                        logger.info(f"🛑 Stop Loss (STOP_LIMIT) hit for {trade.id}: ${exit_price:.4f}")
                    
                    # Close trade in tracker
                    await trade_tracker.close_trade(trade.id, exit_price, status)
            
        except Exception as e:
            logger.warning(f"⚠️ Error checking OCO order for {trade.id}: {e}")
    
    # Keep the old method name for compatibility
    async def monitor_oco_orders(self, trade_tracker: TradeTracker):
        """Compatibility method - calls monitor_orders"""
        await self.monitor_orders(trade_tracker)

# Add order IDs to Trade dataclass
import sys
import importlib
trade_tracker_module = importlib.import_module('modules.trade_tracker')
if not hasattr(trade_tracker_module.Trade, 'oco_order_id'):
    # Monkey patch to add order tracking fields
    original_init = trade_tracker_module.Trade.__init__
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, 'oco_order_id'):
            self.oco_order_id = None
        if not hasattr(self, 'entry_order_id'):
            self.entry_order_id = None
    trade_tracker_module.Trade.__init__ = new_init