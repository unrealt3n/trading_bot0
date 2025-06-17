#!/usr/bin/env python3
"""
🚀 Crypto Trading Bot - Main Entry Point
Simple: Just run it and let it trade continuously
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import enhanced analysis engine
from enhanced_main_engine import compute_trade_confidence

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    """Simple continuous trading bot"""
    
    def __init__(self):
        self.config = self.load_config()
        self.running = True
        self.active_trades = []
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'date': datetime.now().date()
        }
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"🚀 Trading Bot started in {self.config['trading']['mode'].upper()} mode")
        self.print_startup_info()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            sys.exit(1)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("⏹️ Shutdown signal received, stopping bot...")
        self.running = False
    
    def print_startup_info(self):
        """Print bot startup information"""
        mode = self.config['trading']['mode'].upper()
        amount = self.config['trading']['trade_amount']
        threshold = self.config['trading']['min_confidence_threshold']
        coins = self.config['analysis']['supported_coins']
        interval = self.config['analysis']['analysis_interval_seconds']
        
        print(f"""
🚀 ===============================
   CRYPTO TRADING BOT STARTED
🚀 ===============================

Mode: {mode}
Trade Amount: ${amount}
Min Confidence: {threshold}%
Coins: {', '.join(coins)}
Check Interval: {interval}s ({interval//60}min)

🎯 Bot will continuously scan for trades
⏹️  Press Ctrl+C to stop

===============================
        """)
    
    def should_trade(self, analysis: Dict[str, Any]) -> bool:
        """Check if conditions are met for trading"""
        
        # Basic checks
        if not analysis.get('take_trade', False):
            return False
            
        confidence = analysis.get('confidence', 0) * 100
        min_confidence = self.config['trading']['min_confidence_threshold']
        
        if confidence < min_confidence:
            logger.info(f"📊 Confidence {confidence:.1f}% < {min_confidence}% threshold")
            return False
        
        # Check daily limits
        max_daily = self.config['risk_management']['max_daily_trades']
        if self.daily_stats['trades'] >= max_daily:
            logger.info(f"⚠️ Daily limit reached ({self.daily_stats['trades']}/{max_daily})")
            return False
        
        # Check concurrent trades
        max_concurrent = self.config['trading']['max_concurrent_trades']
        if len(self.active_trades) >= max_concurrent:
            logger.info(f"⚠️ Max concurrent trades ({len(self.active_trades)}/{max_concurrent})")
            return False
        
        return True
    
    async def execute_trade(self, coin: str, analysis: Dict[str, Any]) -> bool:
        """Execute trade (demo simulation for now)"""
        
        mode = self.config['trading']['mode']
        amount = self.config['trading']['trade_amount']
        bias = analysis['bias']
        confidence = analysis['confidence'] * 100
        
        logger.info(f"🎯 EXECUTING {bias.upper()} TRADE for {coin}")
        logger.info(f"   Amount: ${amount} | Confidence: {confidence:.1f}%")
        
        if mode == 'demo':
            logger.info(f"📝 DEMO TRADE - Simulated execution")
            # Simulate trade execution
            await asyncio.sleep(1)
            success = True
        else:
            logger.info(f"💰 LIVE TRADE - Real money!")
            # TODO: Implement actual Binance trading here
            success = await self.place_binance_order(coin, bias, amount)
        
        if success:
            # Record trade
            trade = {
                'coin': coin,
                'bias': bias,
                'amount': amount,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'mode': mode
            }
            
            self.active_trades.append(trade)
            self.daily_stats['trades'] += 1
            
            # Log trade
            self.log_trade(trade, analysis)
            
            logger.info(f"✅ Trade executed: {coin} {bias.upper()} ${amount}")
            return True
        else:
            logger.error(f"❌ Trade failed: {coin}")
            return False
    
    async def place_binance_order(self, coin: str, side: str, amount: float) -> bool:
        """Place actual Binance order (placeholder)"""
        # TODO: Implement actual Binance API integration
        logger.info(f"🔄 Placing {side} order for {coin}: ${amount}")
        await asyncio.sleep(1)
        return True  # Simulate success
    
    def log_trade(self, trade: Dict[str, Any], analysis: Dict[str, Any]):
        """Log trade to file"""
        log_entry = {
            'timestamp': trade['timestamp'].isoformat(),
            'coin': trade['coin'],
            'bias': trade['bias'],
            'amount': trade['amount'],
            'confidence': trade['confidence'],
            'mode': trade['mode'],
            'reasons': analysis['reasons'][:3]
        }
        
        try:
            with open('logs/trades.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"❌ Failed to log trade: {e}")
    
    def reset_daily_stats(self):
        """Reset daily stats if new day"""
        current_date = datetime.now().date()
        if current_date != self.daily_stats['date']:
            logger.info(f"📅 New day - resetting stats")
            self.daily_stats = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0.0,
                'date': current_date
            }
    
    def print_daily_summary(self):
        """Print daily trading summary"""
        stats = self.daily_stats
        print(f"""
📊 Daily Summary ({stats['date']}):
   Trades: {stats['trades']}
   Active: {len(self.active_trades)}
   P&L: {stats['pnl']:.2f}%
        """)
    
    async def analyze_coin(self, coin: str) -> Dict[str, Any]:
        """Analyze single coin for trading opportunity"""
        
        try:
            logger.info(f"🔍 Analyzing {coin}...")
            
            # Get full analysis
            analysis = await compute_trade_confidence(coin)
            
            # Log results with detailed breakdown
            bias = analysis['bias']
            confidence = analysis['confidence'] * 100
            take_trade = analysis['take_trade']
            reasons = analysis.get('reasons', [])
            
            # Create signal contribution breakdown
            weighted_analysis = analysis.get('weighted_analysis', {})
            signal_contributions = weighted_analysis.get('signal_contributions', {})
            
            # Log main result
            logger.info(f"📊 {coin}: {bias.upper()} ({confidence:.1f}%) - {'TRADE' if take_trade else 'HOLD'}")
            
            # Log signal breakdown
            if signal_contributions:
                logger.info(f"📈 Signal Breakdown for {coin}:")
                for signal_name, contrib in sorted(signal_contributions.items(), key=lambda x: x[1]['weighted_contribution'], reverse=True):
                    weight_pct = contrib['weight'] * 100
                    contrib_pct = contrib['weighted_contribution'] * 100
                    signal_bias = contrib['bias']
                    signal_conf = contrib['confidence'] * 100
                    logger.info(f"   • {signal_name}: {signal_bias.upper()} ({signal_conf:.1f}%) | Weight: {weight_pct:.1f}% | Contribution: {contrib_pct:+.1f}%")
            
            # Log top reasons
            if reasons:
                logger.info(f"🔍 Key Factors for {coin}: {' | '.join(reasons[:3])}")
            
            # Execute trade if conditions met
            if self.should_trade(analysis):
                logger.info(f"✅ Trading conditions met for {coin}")
                success = await self.execute_trade(coin, analysis)
                analysis['trade_executed'] = success
            else:
                analysis['trade_executed'] = False
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {coin}: {e}")
            return {'trade_executed': False, 'error': str(e)}
    
    async def run_trading_cycle(self):
        """Run one complete trading cycle"""
        
        self.reset_daily_stats()
        
        coins = self.config['analysis']['supported_coins']
        logger.info(f"🔄 Starting trading cycle for {len(coins)} coins...")
        
        cycle_start = datetime.now()
        trades_executed = 0
        
        for coin in coins:
            if not self.running:
                break
                
            try:
                result = await self.analyze_coin(coin)
                
                if result.get('trade_executed'):
                    trades_executed += 1
                    logger.info(f"💰 Trade #{trades_executed} executed for {coin}")
                
                # Small delay between coins
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error processing {coin}: {e}")
                continue
        
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        logger.info(f"✅ Cycle complete: {trades_executed} trades in {cycle_time:.1f}s")
        
        return trades_executed
    
    async def run(self):
        """Main bot loop"""
        
        logger.info("🚀 Starting continuous trading...")
        
        interval = self.config['analysis']['analysis_interval_seconds']
        
        while self.running:
            try:
                # Run trading cycle
                trades = await self.run_trading_cycle()
                
                # Print summary
                if trades > 0:
                    self.print_daily_summary()
                
                if not self.running:
                    break
                
                # Wait until next cycle
                logger.info(f"⏳ Waiting {interval}s until next cycle...")
                
                # Wait with periodic checks for shutdown
                wait_start = datetime.now()
                while self.running and (datetime.now() - wait_start).total_seconds() < interval:
                    await asyncio.sleep(10)  # Check every 10s
                
            except KeyboardInterrupt:
                logger.info("⏹️ Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(30)  # Wait 30s before retry
        
        logger.info("⏹️ Trading bot stopped")
        self.print_daily_summary()

async def main():
    """Entry point"""
    
    # Create logs directory
    import os
    os.makedirs('logs', exist_ok=True)
    
    try:
        # Create and run bot
        bot = TradingBot()
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting Crypto Trading Bot...")
    asyncio.run(main())