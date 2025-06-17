#!/usr/bin/env python3
"""
🚀 Enhanced Crypto Trading Bot - Main Entry Point
Features: Whale Alerts, Liquidation Tracking, Retail vs Big Money, Dynamic Confidence, Trade Tracking
Android Termux Compatible
"""

import asyncio
import json
import logging
import signal
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Enhanced imports
from enhanced_main_engine import compute_trade_confidence
from modules.trade_tracker import TradeTracker
from modules.historical_correlation_engine import HistoricalCorrelationEngine
from modules.binance_executor import BinanceTestnetExecutor

# Terminal UI imports
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Installing rich for better terminal UI...")
    os.system("pip install rich")

# Set up logging with colors
class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_trading_bot.log'),
        logging.StreamHandler()
    ]
)

# Apply colored formatter to console handler
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
        handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)

class EnhancedTradingBot:
    """Enhanced trading bot with all new features"""
    
    def __init__(self):
        self.config = self.load_config()
        self.running = True
        
        # Initialize enhanced components
        self.trade_tracker = TradeTracker()
        self.historical_engine = HistoricalCorrelationEngine()
        
        # Initialize Binance Testnet Executor
        binance_testnet_api_key = os.getenv('BINANCE_TESTNET_API_KEY')
        binance_testnet_secret = os.getenv('BINANCE_TESTNET_SECRET_KEY')
        
        if binance_testnet_api_key and binance_testnet_secret:
            self.binance_executor = BinanceTestnetExecutor(binance_testnet_api_key, binance_testnet_secret)
            logger.info("🚀 Binance Testnet Executor initialized - Real trades enabled!")
        else:
            self.binance_executor = None
            logger.warning("⚠️ Binance Testnet keys not found - Analysis only mode")
        
        # Performance tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'signals_analyzed': 0,
            'trades_executed': 0,
            'whale_alerts_detected': 0,
            'liquidations_tracked': 0,
            'retail_divergences': 0,
            'avg_confidence': 0.0,
            'best_signal': {'coin': 'N/A', 'confidence': 0.0}
        }
        
        # Console for rich UI
        if RICH_AVAILABLE:
            self.console = Console()
            self.setup_rich_ui()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"🚀 Enhanced Trading Bot started in {self.config['trading']['mode'].upper()} mode")
        self.print_enhanced_startup_info()
    
    def setup_rich_ui(self):
        """Setup rich terminal UI components"""
        if not RICH_AVAILABLE:
            return
        
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=8),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=4)
        )
        
        self.layout["main"].split_row(
            Layout(name="signals", ratio=2),
            Layout(name="trades", ratio=1)
        )
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration with enhanced defaults"""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            # Add enhanced configuration defaults
            if 'enhanced' not in config:
                config['enhanced'] = {
                    'whale_alert_api_key': '',
                    'enable_whale_tracking': True,
                    'enable_liquidation_tracking': True,
                    'enable_retail_detection': True,
                    'enable_historical_correlation': True,
                    'min_whale_threshold_usd': 100000,
                    'liquidation_lookback_minutes': 60,
                    'retail_divergence_threshold': 0.5
                }
            
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            sys.exit(1)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("⏹️ Shutdown signal received, stopping enhanced bot...")
        self.running = False
    
    def print_enhanced_startup_info(self):
        """Print enhanced bot startup information"""
        if RICH_AVAILABLE:
            self.print_rich_startup()
        else:
            self.print_simple_startup()
    
    def print_rich_startup(self):
        """Print startup info using rich"""
        
        # Create startup table
        table = Table(title="🚀 Enhanced Crypto Trading Bot", title_style="bold blue")
        table.add_column("Feature", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        # Add feature rows
        features = [
            ("Trading Mode", self.config['trading']['mode'].upper(), f"${self.config['trading']['trade_amount']} per trade"),
            ("Confidence Threshold", f"{self.config['trading']['min_confidence_threshold']:.1f}%", "Minimum for trade execution"),
            ("Supported Coins", str(len(self.config['analysis']['supported_coins'])), ", ".join(self.config['analysis']['supported_coins'])),
            ("🐋 Whale Alerts", "✅ ENABLED" if self.config['enhanced']['enable_whale_tracking'] else "❌ DISABLED", f"${self.config['enhanced']['min_whale_threshold_usd']:,}+ threshold"),
            ("💥 Liquidation Tracking", "✅ ENABLED" if self.config['enhanced']['enable_liquidation_tracking'] else "❌ DISABLED", f"{self.config['enhanced']['liquidation_lookback_minutes']}min lookback"),
            ("🏦 Retail vs Big Money", "✅ ENABLED" if self.config['enhanced']['enable_retail_detection'] else "❌ DISABLED", "Smart money detection"),
            ("📈 Historical Correlation", "✅ ENABLED" if self.config['enhanced']['enable_historical_correlation'] else "❌ DISABLED", "Pattern learning"),
            ("📊 Trade Tracking", "✅ ENABLED", "JSON-based P&L tracking"),
            ("⏰ Trade Durations", "✅ ENABLED", "Scalp/Short/Mid-term"),
            ("🎯 Dynamic Confidence", "✅ ENABLED", "S/R and volatility adjustments")
        ]
        
        for feature, status, details in features:
            table.add_row(feature, status, details)
        
        # Create startup panel
        startup_panel = Panel(
            table,
            title="🚀 Enhanced Trading Bot Initialization",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(startup_panel)
        
        # Print controls
        controls = Panel(
            "🎯 [bold green]Bot will continuously scan for high-probability trades[/bold green]\n"
            "⏹️  [yellow]Press Ctrl+C to stop[/yellow]\n"
            "📊 [cyan]Live dashboard will update during operation[/cyan]",
            title="Controls",
            border_style="green"
        )
        
        self.console.print(controls)
    
    def print_simple_startup(self):
        """Print startup info for terminals without rich"""
        
        mode = self.config['trading']['mode'].upper()
        amount = self.config['trading']['trade_amount']
        threshold = self.config['trading']['min_confidence_threshold']
        coins = self.config['analysis']['supported_coins']
        interval = self.config['analysis']['analysis_interval_seconds']
        
        print(f"""
🚀 ================================================
   ENHANCED CRYPTO TRADING BOT STARTED
🚀 ================================================

📊 CORE SETTINGS:
   Mode: {mode}
   Trade Amount: ${amount}
   Min Confidence: {threshold}%
   Coins: {', '.join(coins)}
   Check Interval: {interval}s ({interval//60}min)

🆕 ENHANCED FEATURES:
   🐋 Whale Alerts: {'✅' if self.config['enhanced']['enable_whale_tracking'] else '❌'}
   💥 Liquidation Tracking: {'✅' if self.config['enhanced']['enable_liquidation_tracking'] else '❌'}
   🏦 Retail vs Big Money: {'✅' if self.config['enhanced']['enable_retail_detection'] else '❌'}
   📈 Historical Correlation: {'✅' if self.config['enhanced']['enable_historical_correlation'] else '❌'}
   📊 Trade Tracking: ✅
   ⏰ Trade Durations: ✅ (Scalp/Short/Mid-term)
   🎯 Dynamic Confidence: ✅

🎯 Bot will scan for high-probability trades
⏹️  Press Ctrl+C to stop

================================================
        """)
    
    async def analyze_coin_enhanced(self, coin: str) -> Dict[str, Any]:
        """Enhanced coin analysis with all new features"""
        
        try:
            logger.info(f"🔍 Enhanced analysis for {coin}...")
            self.session_stats['signals_analyzed'] += 1
            
            # Get enhanced analysis
            analysis = await compute_trade_confidence(coin)
            
            # Record event for historical correlation
            if self.config['enhanced']['enable_historical_correlation']:
                await self.historical_engine.record_event(
                    coin, 'analysis', analysis, analysis.get('current_price', 0)
                )
            
            # Update session stats
            confidence = analysis.get('confidence', 0) * 100
            self.session_stats['avg_confidence'] = (
                (self.session_stats['avg_confidence'] * (self.session_stats['signals_analyzed'] - 1) + confidence) 
                / self.session_stats['signals_analyzed']
            )
            
            if confidence > self.session_stats['best_signal']['confidence']:
                self.session_stats['best_signal'] = {'coin': coin, 'confidence': confidence}
            
            # Enhanced logging
            self.log_enhanced_analysis(coin, analysis)
            
            # Execute trade if conditions met
            if self.should_take_enhanced_trade(analysis):
                logger.info(f"✅ Enhanced trading conditions met for {coin}")
                success = await self.execute_enhanced_trade(coin, analysis)
                analysis['trade_executed'] = success
                if success:
                    self.session_stats['trades_executed'] += 1
            else:
                analysis['trade_executed'] = False
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced analysis for {coin}: {e}")
            return {'trade_executed': False, 'error': str(e)}
    
    def log_enhanced_analysis(self, coin: str, analysis: Dict[str, Any]):
        """Enhanced logging with detailed breakdown"""
        
        confidence = analysis.get('confidence', 0) * 100
        bias = analysis.get('bias', 'neutral')
        duration = analysis.get('duration', 'unknown')
        take_trade = analysis.get('take_trade', False)
        
        # Main result with enhanced emoji
        action_emoji = "🚀" if take_trade else "📊"
        logger.info(f"{action_emoji} {coin}: {bias.upper()} ({confidence:.1f}%) - {duration.upper()} - {'TRADE' if take_trade else 'ANALYZE'}")
        
        # Enhanced signal breakdown
        enhanced_analysis = analysis.get('enhanced_analysis', {})
        signal_scores = enhanced_analysis.get('signal_scores', {})
        
        if signal_scores:
            logger.info(f"🎯 Enhanced Signal Breakdown for {coin}:")
            
            # Sort by weighted contribution
            sorted_signals = sorted(
                signal_scores.items(),
                key=lambda x: abs(x[1].get('weighted_contribution', 0)),
                reverse=True
            )
            
            for signal_name, score_data in sorted_signals:
                confidence_pct = score_data.get('confidence', 0) * 100
                contribution_pct = score_data.get('weighted_contribution', 0) * 100
                weight_pct = score_data.get('weight', 0) * 100
                signal_bias = score_data.get('bias', 'neutral')
                
                # Enhanced emoji for each signal type
                emoji_map = {
                    'whale_alerts': '🐋',
                    'liquidations': '💥',
                    'retail_vs_bigmoney': '🏦',
                    'smart_money': '💰',
                    'technical_analysis': '📈',
                    'news_sentiment': '📰'
                }
                
                emoji = emoji_map.get(signal_name, '📊')
                logger.info(f"   {emoji} {signal_name}: {signal_bias.upper()} ({confidence_pct:.1f}%) "
                           f"| Weight: {weight_pct:.1f}% | Impact: {contribution_pct:+.1f}%")
        
        # Duration and exit strategy
        duration_data = analysis.get('duration_data', {})
        if duration_data:
            exit_strategy = duration_data.get('exit_strategy', {})
            tp_pct = exit_strategy.get('take_profit_percent', 0)
            sl_pct = exit_strategy.get('stop_loss_percent', 0)
            duration_minutes = duration_data.get('duration_minutes', 0)
            
            logger.info(f"⏰ Duration: {duration.upper()} ({duration_minutes}min) | TP: {tp_pct:.2f}% | SL: {sl_pct:.2f}%")
        
        # Confidence adjustments
        if 'confidence_adjustments' in enhanced_analysis:
            adj_data = enhanced_analysis['confidence_adjustments']
            base_conf = adj_data.get('base_confidence', 0)
            final_conf = adj_data.get('adjusted_confidence', 0)
            change = final_conf - base_conf
            
            if abs(change) > 0.01:
                logger.info(f"🎯 Confidence: {base_conf:.3f} → {final_conf:.3f} ({change:+.3f})")
    
    def should_take_enhanced_trade(self, analysis: Dict[str, Any]) -> bool:
        """Enhanced trade decision logic"""
        
        # Basic checks
        if not analysis.get('take_trade', False):
            return False
        
        confidence = analysis.get('confidence', 0) * 100
        min_confidence = self.config['trading']['min_confidence_threshold']
        
        if confidence < min_confidence:
            logger.info(f"📊 Confidence {confidence:.1f}% < {min_confidence}% threshold")
            return False
        
        # Enhanced checks
        duration = analysis.get('duration', 'short')
        enhanced_analysis = analysis.get('enhanced_analysis', {})
        
        # Duration-specific requirements
        if duration == 'scalp':
            # Scalping requires whale alerts OR liquidations
            signal_scores = enhanced_analysis.get('signal_scores', {})
            whale_strength = signal_scores.get('whale_alerts', {}).get('confidence', 0)
            liq_strength = signal_scores.get('liquidations', {}).get('confidence', 0)
            
            if whale_strength < 0.6 and liq_strength < 0.6:
                logger.info(f"📊 Scalp trade requires strong whale/liquidation signals")
                return False
        
        # Check daily limits
        max_daily = self.config['risk_management']['max_daily_trades']
        daily_stats = self.trade_tracker.daily_stats
        if daily_stats['total_trades'] >= max_daily:
            logger.info(f"⚠️ Daily limit reached ({daily_stats['total_trades']}/{max_daily})")
            return False
        
        # Check concurrent trades
        max_concurrent = self.config['trading']['max_concurrent_trades']
        active_trades = len(self.trade_tracker.get_active_trades())
        if active_trades >= max_concurrent:
            logger.info(f"⚠️ Max concurrent trades ({active_trades}/{max_concurrent})")
            return False
        
        return True
    
    async def execute_enhanced_trade(self, coin: str, analysis: Dict[str, Any]) -> bool:
        """Execute enhanced trade with tracking"""
        
        try:
            mode = self.config['trading']['mode']
            amount = self.config['trading']['trade_amount']
            bias = analysis['bias']
            confidence = analysis['confidence'] * 100
            duration = analysis.get('duration', 'short')
            
            logger.info(f"🎯 EXECUTING {bias.upper()} {duration.upper()} TRADE for {coin}")
            logger.info(f"   Amount: ${amount} | Confidence: {confidence:.1f}%")
            
            # Execute trade with Binance Testnet
            if self.binance_executor:
                logger.info(f"🚀 LIVE BINANCE TESTNET TRADE - Real orders with TP/SL!")
                trade = await self.binance_executor.execute_trade(analysis, self.trade_tracker)
                success = trade is not None
                
                if success:
                    current_price = trade.entry_price
                    logger.info(f"✅ Enhanced trade executed: {trade.id}")
                    logger.info(f"   Coin: {coin} | Type: {bias.upper()} | Duration: {duration.upper()}")
                    logger.info(f"   Entry: ${current_price:.4f} | Amount: ${amount}")
                    logger.info(f"   TP: ${trade.take_profit_price:.4f} | SL: ${trade.stop_loss_price:.4f}")
                else:
                    logger.error(f"❌ Failed to execute Binance trade for {coin}")
            else:
                # Demo mode fallback
                logger.info(f"📝 DEMO TRADE - Enhanced simulation (no Binance keys)")
                current_price = 50000.0  # Placeholder
                await asyncio.sleep(1)
                
                # Create trade in tracker for demo
                duration_data = analysis.get('duration_data', {})
                trade = await self.trade_tracker.create_trade(
                    coin, analysis, duration_data, current_price, amount
                )
                success = True
            
            if success and trade:
                
                # Record successful trade event
                await self.historical_engine.record_event(
                    coin, 'trade_execution', {
                        'trade_id': trade.id,
                        'bias': bias,
                        'duration': duration,
                        'confidence': confidence
                    }, current_price
                )
                
                return True
            else:
                logger.error(f"❌ Enhanced trade failed: {coin}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing enhanced trade: {e}")
            return False
    
    async def place_enhanced_order(self, coin: str, side: str, amount: float, analysis: Dict[str, Any]) -> bool:
        """Place enhanced order with additional validation"""
        
        try:
            # Enhanced order placement with risk management
            duration_data = analysis.get('duration_data', {})
            exit_strategy = duration_data.get('exit_strategy', {})
            
            logger.info(f"🔄 Placing enhanced {side} order for {coin}: ${amount}")
            logger.info(f"   Exit Strategy: TP {exit_strategy.get('take_profit_percent', 0):.2f}% | "
                       f"SL {exit_strategy.get('stop_loss_percent', 0):.2f}%")
            
            # Simulate order placement
            await asyncio.sleep(1)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error placing enhanced order: {e}")
            return False
    
    async def run_enhanced_trading_cycle(self):
        """Run enhanced trading cycle with all features"""
        
        coins = self.config['analysis']['supported_coins']
        logger.info(f"🔄 Starting enhanced trading cycle for {len(coins)} coins...")
        
        cycle_start = datetime.now()
        trades_executed = 0
        
        # Update active trades first
        await self.update_active_trades()
        
        # Monitor OCO orders on Binance
        if self.binance_executor:
            await self.binance_executor.monitor_oco_orders(self.trade_tracker)
        
        for coin in coins:
            if not self.running:
                break
                
            try:
                result = await self.analyze_coin_enhanced(coin)
                
                if result.get('trade_executed'):
                    trades_executed += 1
                    logger.info(f"💰 Enhanced trade #{trades_executed} executed for {coin}")
                
                # Small delay between coins
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error processing {coin}: {e}")
                continue
        
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        logger.info(f"✅ Enhanced cycle complete: {trades_executed} trades in {cycle_time:.1f}s")
        
        # Print enhanced summary
        await self.print_enhanced_summary()
        
        return trades_executed
    
    async def update_active_trades(self):
        """Update active trades with current market prices"""
        
        try:
            active_trades = await self.trade_tracker.get_active_trades()
            
            for trade in active_trades:
                # Get current price (placeholder)
                current_price = trade.entry_price * 1.001  # Simulate slight price movement
                
                # Update trade
                updated_trade = await self.trade_tracker.update_trade_price(trade.id, current_price)
                
                if updated_trade and updated_trade.status.value != 'active':
                    logger.info(f"🔄 Trade {trade.id} status updated: {updated_trade.status.value}")
                    
        except Exception as e:
            logger.error(f"❌ Error updating active trades: {e}")
    
    async def print_enhanced_summary(self):
        """Print enhanced trading summary"""
        
        if RICH_AVAILABLE:
            await self.print_rich_summary()
        else:
            await self.print_simple_summary()
    
    async def print_rich_summary(self):
        """Print summary using rich"""
        
        # Get performance data
        performance = await self.trade_tracker.get_performance_summary()
        daily_stats = performance['daily_stats']
        active_count = performance['active_trades_count']
        
        # Create summary table
        summary_table = Table(title="📊 Enhanced Trading Summary", title_style="bold green")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")
        summary_table.add_column("Details", style="white")
        
        # Session stats
        runtime = datetime.now() - self.session_stats['start_time']
        runtime_str = f"{runtime.total_seconds()/3600:.1f}h"
        
        summary_data = [
            ("Session Runtime", runtime_str, f"Started {self.session_stats['start_time'].strftime('%H:%M')}"),
            ("Signals Analyzed", str(self.session_stats['signals_analyzed']), f"Avg confidence: {self.session_stats['avg_confidence']:.1f}%"),
            ("Trades Today", str(daily_stats['total_trades']), f"Active: {active_count}"),
            ("Win Rate", f"{daily_stats['win_rate']:.1f}%", f"W: {daily_stats['winning_trades']} | L: {daily_stats['losing_trades']}"),
            ("Daily P&L", f"{daily_stats['total_pnl_percent']:+.2f}%", f"${daily_stats['total_pnl_usd']:+.2f}"),
            ("Best Signal", self.session_stats['best_signal']['coin'], f"{self.session_stats['best_signal']['confidence']:.1f}% confidence"),
        ]
        
        for metric, value, details in summary_data:
            summary_table.add_row(metric, value, details)
        
        summary_panel = Panel(summary_table, border_style="green")
        self.console.print(summary_panel)
    
    async def print_simple_summary(self):
        """Print simple summary for basic terminals"""
        
        performance = await self.trade_tracker.get_performance_summary()
        daily_stats = performance['daily_stats']
        active_count = performance['active_trades_count']
        
        runtime = datetime.now() - self.session_stats['start_time']
        
        print(f"""
📊 =====================================
   ENHANCED TRADING SUMMARY
📊 =====================================

⏱️  Runtime: {runtime.total_seconds()/3600:.1f}h
🔍 Signals Analyzed: {self.session_stats['signals_analyzed']}
💰 Trades Today: {daily_stats['total_trades']} (Active: {active_count})
📈 Win Rate: {daily_stats['win_rate']:.1f}%
💵 Daily P&L: {daily_stats['total_pnl_percent']:+.2f}% (${daily_stats['total_pnl_usd']:+.2f})
🎯 Best Signal: {self.session_stats['best_signal']['coin']} ({self.session_stats['best_signal']['confidence']:.1f}%)

=====================================
        """)
    
    async def run(self):
        """Main enhanced bot loop"""
        
        logger.info("🚀 Starting enhanced continuous trading...")
        
        interval = self.config['analysis']['analysis_interval_seconds']
        
        while self.running:
            try:
                # Run enhanced trading cycle
                trades = await self.run_enhanced_trading_cycle()
                
                if not self.running:
                    break
                
                # Wait until next cycle
                logger.info(f"⏳ Waiting {interval}s until next enhanced cycle...")
                
                # Wait with periodic checks for shutdown
                wait_start = datetime.now()
                while self.running and (datetime.now() - wait_start).total_seconds() < interval:
                    await asyncio.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("⏹️ Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"❌ Error in enhanced main loop: {e}")
                await asyncio.sleep(30)
        
        logger.info("⏹️ Enhanced trading bot stopped")
        await self.print_enhanced_summary()

async def main():
    """Enhanced entry point"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    try:
        # Create and run enhanced bot
        bot = EnhancedTradingBot()
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Enhanced bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in enhanced bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting Enhanced Crypto Trading Bot...")
    asyncio.run(main())