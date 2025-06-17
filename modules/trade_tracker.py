"""
📊 Trade Tracker
JSON-based trade execution and profitability tracking with TP/SL monitoring
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import aiofiles
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class TradeStatus(Enum):
    ACTIVE = "active"
    COMPLETED_TP = "completed_tp"  # Hit take profit
    COMPLETED_SL = "completed_sl"  # Hit stop loss
    EXPIRED = "expired"            # Max duration reached
    CANCELLED = "cancelled"

class TradeType(Enum):
    LONG = "long"
    SHORT = "short"

@dataclass
class Trade:
    """Trade data structure"""
    id: str
    coin: str
    trade_type: TradeType
    duration: str  # scalp/short/mid
    entry_price: float
    entry_time: datetime
    amount: float
    take_profit_price: float
    stop_loss_price: float
    max_duration_minutes: int
    status: TradeStatus
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl_percent: Optional[float] = None
    pnl_usd: Optional[float] = None
    fees_usd: float = 0.0
    confidence: float = 0.0
    primary_signals: List[str] = None
    partial_exits: List[Dict] = None
    
    def __post_init__(self):
        if self.primary_signals is None:
            self.primary_signals = []
        if self.partial_exits is None:
            self.partial_exits = []

class TradeTracker:
    """Tracks all trades and their performance in JSON format"""
    
    def __init__(self, trades_file: str = "logs/trades_tracker.json"):
        self.trades_file = Path(trades_file)
        self.trades_file.parent.mkdir(exist_ok=True)
        
        self.active_trades: Dict[str, Trade] = {}
        self.completed_trades: List[Trade] = []
        
        # Performance tracking
        self.daily_stats = {
            'date': datetime.now().date().isoformat(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl_percent': 0.0,
            'total_pnl_usd': 0.0,
            'win_rate': 0.0,
            'avg_win_percent': 0.0,
            'avg_loss_percent': 0.0,
            'best_trade_percent': 0.0,
            'worst_trade_percent': 0.0,
            'scalp_trades': 0,
            'short_trades': 0,
            'mid_trades': 0
        }
        
        # Load existing trades (will be done lazily when first accessed)
        self._trades_loaded = False
        
        logger.info(f"📊 Trade Tracker initialized - File: {self.trades_file}")
    
    async def _ensure_trades_loaded(self):
        """Ensure trades are loaded (called lazily)"""
        if not self._trades_loaded:
            await self.load_trades()
            self._trades_loaded = True
    
    async def load_trades(self):
        """Load existing trades from JSON file"""
        try:
            if self.trades_file.exists():
                async with aiofiles.open(self.trades_file, 'r') as f:
                    data = json.loads(await f.read())
                
                # Load active trades
                for trade_data in data.get('active_trades', []):
                    trade = self._dict_to_trade(trade_data)
                    self.active_trades[trade.id] = trade
                
                # Load completed trades
                for trade_data in data.get('completed_trades', []):
                    trade = self._dict_to_trade(trade_data)
                    self.completed_trades.append(trade)
                
                # Load daily stats
                self.daily_stats = data.get('daily_stats', self.daily_stats)
                
                logger.info(f"📊 Loaded {len(self.active_trades)} active and {len(self.completed_trades)} completed trades")
            
        except Exception as e:
            logger.error(f"❌ Error loading trades: {e}")
    
    async def save_trades(self):
        """Save all trades to JSON file"""
        try:
            data = {
                'active_trades': [self._trade_to_dict(trade) for trade in self.active_trades.values()],
                'completed_trades': [self._trade_to_dict(trade) for trade in self.completed_trades],
                'daily_stats': self.daily_stats,
                'last_updated': datetime.now().isoformat()
            }
            
            async with aiofiles.open(self.trades_file, 'w') as f:
                await f.write(json.dumps(data, indent=2, default=str))
            
        except Exception as e:
            logger.error(f"❌ Error saving trades: {e}")
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade object to dictionary for JSON serialization"""
        trade_dict = asdict(trade)
        trade_dict['trade_type'] = trade.trade_type.value
        trade_dict['status'] = trade.status.value
        trade_dict['entry_time'] = trade.entry_time.isoformat()
        if trade.exit_time:
            trade_dict['exit_time'] = trade.exit_time.isoformat()
        return trade_dict
    
    def _dict_to_trade(self, trade_dict: Dict) -> Trade:
        """Convert dictionary to Trade object"""
        trade_dict['trade_type'] = TradeType(trade_dict['trade_type'])
        trade_dict['status'] = TradeStatus(trade_dict['status'])
        trade_dict['entry_time'] = datetime.fromisoformat(trade_dict['entry_time'])
        if trade_dict.get('exit_time'):
            trade_dict['exit_time'] = datetime.fromisoformat(trade_dict['exit_time'])
        return Trade(**trade_dict)
    
    async def create_trade(self, coin: str, analysis_data: Dict[str, Any], duration_data: Dict[str, Any], 
                          entry_price: float, amount: float) -> Trade:
        """Create new trade from analysis data"""
        
        await self._ensure_trades_loaded()
        
        trade_id = f"{coin}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine trade type from bias
        bias = analysis_data.get('bias', 'bullish')
        trade_type = TradeType.LONG if bias == 'bullish' else TradeType.SHORT
        
        # Calculate TP/SL prices
        exit_strategy = duration_data.get('exit_strategy', {})
        tp_percent = exit_strategy.get('take_profit_percent', 2.0) / 100
        sl_percent = exit_strategy.get('stop_loss_percent', 1.0) / 100
        
        if trade_type == TradeType.LONG:
            take_profit_price = entry_price * (1 + tp_percent)
            stop_loss_price = entry_price * (1 - sl_percent)
        else:
            take_profit_price = entry_price * (1 - tp_percent)
            stop_loss_price = entry_price * (1 + sl_percent)
        
        # Create trade
        trade = Trade(
            id=trade_id,
            coin=coin,
            trade_type=trade_type,
            duration=duration_data.get('duration', 'short'),
            entry_price=entry_price,
            entry_time=datetime.now(),
            amount=amount,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            max_duration_minutes=duration_data.get('duration_minutes', 240),
            status=TradeStatus.ACTIVE,
            confidence=analysis_data.get('confidence', 0.0) * 100,
            primary_signals=duration_data.get('primary_signals', []),
            partial_exits=exit_strategy.get('partial_exits', [])
        )
        
        # Add to active trades
        self.active_trades[trade_id] = trade
        
        # Update daily stats
        self.daily_stats['total_trades'] += 1
        duration_key = f"{trade.duration}_trades"
        if duration_key in self.daily_stats:
            self.daily_stats[duration_key] += 1
        
        await self.save_trades()
        
        logger.info(f"📊 Created {trade.trade_type.value.upper()} trade: {trade_id} - {coin} @ ${entry_price:.4f}")
        
        return trade
    
    async def update_trade_price(self, trade_id: str, current_price: float) -> Optional[Trade]:
        """Update trade with current price and check for TP/SL"""
        
        if trade_id not in self.active_trades:
            return None
        
        trade = self.active_trades[trade_id]
        
        # Check if trade expired
        if datetime.now() - trade.entry_time > timedelta(minutes=trade.max_duration_minutes):
            return await self.close_trade(trade_id, current_price, TradeStatus.EXPIRED)
        
        # Check TP/SL conditions
        if trade.trade_type == TradeType.LONG:
            if current_price >= trade.take_profit_price:
                return await self.close_trade(trade_id, current_price, TradeStatus.COMPLETED_TP)
            elif current_price <= trade.stop_loss_price:
                return await self.close_trade(trade_id, current_price, TradeStatus.COMPLETED_SL)
        else:  # SHORT
            if current_price <= trade.take_profit_price:
                return await self.close_trade(trade_id, current_price, TradeStatus.COMPLETED_TP)
            elif current_price >= trade.stop_loss_price:
                return await self.close_trade(trade_id, current_price, TradeStatus.COMPLETED_SL)
        
        # Check partial exits
        await self._check_partial_exits(trade, current_price)
        
        return trade
    
    async def _check_partial_exits(self, trade: Trade, current_price: float):
        """Check and execute partial exits"""
        
        if not trade.partial_exits:
            return
        
        current_pnl_percent = self._calculate_pnl_percent(trade, current_price)
        
        for partial_exit in trade.partial_exits:
            target_profit = partial_exit['at_profit']
            exit_percent = partial_exit['exit_percent']
            
            # Check if we haven't already executed this partial exit
            if not partial_exit.get('executed', False) and current_pnl_percent >= target_profit:
                # Execute partial exit
                partial_exit['executed'] = True
                partial_exit['execution_price'] = current_price
                partial_exit['execution_time'] = datetime.now().isoformat()
                
                logger.info(f"📊 Partial exit executed for {trade.id}: {exit_percent}% at {target_profit}% profit")
    
    async def close_trade(self, trade_id: str, exit_price: float, status: TradeStatus) -> Trade:
        """Close active trade and calculate final P&L"""
        
        if trade_id not in self.active_trades:
            raise ValueError(f"Trade {trade_id} not found in active trades")
        
        trade = self.active_trades[trade_id]
        
        # Update trade
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.status = status
        trade.pnl_percent = self._calculate_pnl_percent(trade, exit_price)
        trade.pnl_usd = self._calculate_pnl_usd(trade, exit_price)
        
        # Move to completed trades
        self.completed_trades.append(trade)
        del self.active_trades[trade_id]
        
        # Update daily stats
        self._update_daily_stats(trade)
        
        await self.save_trades()
        
        status_emoji = "✅" if status == TradeStatus.COMPLETED_TP else "❌" if status == TradeStatus.COMPLETED_SL else "⏰"
        logger.info(f"{status_emoji} Trade closed: {trade_id} - {trade.pnl_percent:+.2f}% (${trade.pnl_usd:+.2f})")
        
        return trade
    
    def _calculate_pnl_percent(self, trade: Trade, current_price: float) -> float:
        """Calculate P&L percentage"""
        if trade.trade_type == TradeType.LONG:
            return ((current_price - trade.entry_price) / trade.entry_price) * 100
        else:
            return ((trade.entry_price - current_price) / trade.entry_price) * 100
    
    def _calculate_pnl_usd(self, trade: Trade, current_price: float) -> float:
        """Calculate P&L in USD"""
        pnl_percent = self._calculate_pnl_percent(trade, current_price) / 100
        return trade.amount * pnl_percent - trade.fees_usd
    
    def _update_daily_stats(self, trade: Trade):
        """Update daily statistics with completed trade"""
        
        if trade.pnl_percent > 0:
            self.daily_stats['winning_trades'] += 1
            if trade.pnl_percent > self.daily_stats['best_trade_percent']:
                self.daily_stats['best_trade_percent'] = trade.pnl_percent
        else:
            self.daily_stats['losing_trades'] += 1
            if trade.pnl_percent < self.daily_stats['worst_trade_percent']:
                self.daily_stats['worst_trade_percent'] = trade.pnl_percent
        
        self.daily_stats['total_pnl_percent'] += trade.pnl_percent
        self.daily_stats['total_pnl_usd'] += trade.pnl_usd
        
        # Calculate win rate
        total_closed = self.daily_stats['winning_trades'] + self.daily_stats['losing_trades']
        if total_closed > 0:
            self.daily_stats['win_rate'] = (self.daily_stats['winning_trades'] / total_closed) * 100
        
        # Calculate average win/loss
        if self.daily_stats['winning_trades'] > 0:
            winning_trades = [t for t in self.completed_trades if t.pnl_percent > 0]
            self.daily_stats['avg_win_percent'] = sum(t.pnl_percent for t in winning_trades) / len(winning_trades)
        
        if self.daily_stats['losing_trades'] > 0:
            losing_trades = [t for t in self.completed_trades if t.pnl_percent <= 0]
            self.daily_stats['avg_loss_percent'] = sum(t.pnl_percent for t in losing_trades) / len(losing_trades)
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get detailed performance summary"""
        
        # Check if new day
        current_date = datetime.now().date().isoformat()
        if current_date != self.daily_stats['date']:
            await self._reset_daily_stats()
        
        # Calculate current P&L for active trades
        active_pnl = 0.0
        for trade in self.active_trades.values():
            # This would need current market price - placeholder for now
            # active_pnl += self._calculate_pnl_usd(trade, current_market_price)
            pass
        
        return {
            'daily_stats': self.daily_stats,
            'active_trades_count': len(self.active_trades),
            'completed_trades_count': len(self.completed_trades),
            'active_trades_pnl': active_pnl,
            'trade_breakdown': {
                'scalp': {'total': self.daily_stats['scalp_trades'], 'success_rate': 0},
                'short': {'total': self.daily_stats['short_trades'], 'success_rate': 0},
                'mid': {'total': self.daily_stats['mid_trades'], 'success_rate': 0}
            }
        }
    
    async def _reset_daily_stats(self):
        """Reset daily stats for new day"""
        self.daily_stats = {
            'date': datetime.now().date().isoformat(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl_percent': 0.0,
            'total_pnl_usd': 0.0,
            'win_rate': 0.0,
            'avg_win_percent': 0.0,
            'avg_loss_percent': 0.0,
            'best_trade_percent': 0.0,
            'worst_trade_percent': 0.0,
            'scalp_trades': 0,
            'short_trades': 0,
            'mid_trades': 0
        }
        await self.save_trades()
        logger.info("📊 Daily stats reset for new trading day")
    
    async def get_active_trades(self) -> List[Trade]:
        """Get list of active trades"""
        await self._ensure_trades_loaded()
        return list(self.active_trades.values())
    
    def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """Get specific trade by ID"""
        return self.active_trades.get(trade_id) or next(
            (t for t in self.completed_trades if t.id == trade_id), None
        )