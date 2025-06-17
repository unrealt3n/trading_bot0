"""
🚀 Main Trading Signal Engine
Orchestrates all analysis modules to compute final trade confidence
Implements weighted scoring and trade logging system
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime, timezone

# Import all analysis modules
from modules.options_analyzer import OptionsAnalyzer
from modules.orderbook_analyzer import OrderbookAnalyzer
from modules.indicator_engine import IndicatorEngine
from modules.news_analyzer import NewsAnalyzer
from modules.session_analyzer import SessionAnalyzer
# Gemini analyzer removed - unnecessary
from modules.smart_money_detector import SmartMoneyDetector
from modules.multi_exchange_analyzer import MultiExchangeAnalyzer

# Import utilities and config
from modules.utils import get_emoji, get_utc_timestamp, log_trade_attempt
from config.settings import weight_config, trading_config, system_config

logger = logging.getLogger(__name__)

class TradingSignalEngine:
    """Main engine that orchestrates all analysis modules"""
    
    def __init__(self):
        # Initialize all analyzers
        self.options_analyzer = OptionsAnalyzer()
        self.orderbook_analyzer = OrderbookAnalyzer()
        self.indicator_engine = IndicatorEngine()
        self.news_analyzer = NewsAnalyzer()
        self.session_analyzer = SessionAnalyzer()
        # Gemini analyzer removed
        
        # Initialize new enhanced analyzers
        self.smart_money_detector = SmartMoneyDetector()
        self.multi_exchange_analyzer = MultiExchangeAnalyzer()
        
        # Get normalized weights
        self.weights = weight_config.normalize_weights()
        
        logger.info(f"🚀 Enhanced Trading Signal Engine initialized with weights: {self.weights}")
    
    async def compute_trade_confidence(self, coin: str) -> Dict[str, Any]:
        """
        Core logic that evaluates whether to take a trade.
        Aggregates data from options, orderbook, indicators, news, and session context.
        Uses weighted scoring to produce a final signal and logs all trades.

        Returns:
        {
            "take_trade": True,              # or False
            "confidence": 0.81,              # float from 0 to 1
            "bias": "bullish",               # "bullish", "bearish", or "neutral"
            "reasons": [...],                # key reasons for signal
            "raw_inputs": {...},             # unprocessed results from each module
            "timestamp": "2025-06-15T15:43Z" # UTC time
        }
        """
        
        logger.info(f"🎯 Computing trade confidence for {coin}")
        start_time = datetime.now()
        
        try:
            # Validate coin
            if coin.upper() not in trading_config.supported_coins:
                logger.warning(f"⚠️ {coin} not in supported coins list")
                return await self._create_error_response(
                    coin, f"{coin} not supported", start_time
                )
            
            # 🔥 PHASE 1: Gather enhanced multi-exchange data first
            logger.info(f"🌐 Gathering enhanced multi-exchange data for {coin}")
            multi_exchange_data = await self.multi_exchange_analyzer.enhanced_multi_exchange_analysis(coin)
            
            # Extract orderbook data for smart money detection from multi-exchange analysis
            exchange_orderbook_data = {}
            if 'raw_data' in multi_exchange_data and isinstance(multi_exchange_data['raw_data'], dict):
                # Check different possible locations for exchange data
                raw_data = multi_exchange_data['raw_data']
                
                # Method 1: Direct exchange data in validation results
                validation_results = raw_data.get('validation_results', {})
                if 'exchange_metrics' in validation_results:
                    # This contains basic metrics but we need full orderbook data
                    pass
                
                # Method 2: From volume flow analysis
                volume_analysis = raw_data.get('volume_flow_analysis', {})
                if 'exchange_volume_analysis' in volume_analysis:
                    # This also doesn't have full orderbook data
                    pass
                
                # Method 3: We need to extract from the raw multi-exchange analyzer
                # For now, let's gather fresh orderbook data specifically for smart money detection
                try:
                    # Get fresh orderbook data for smart money analysis
                    from modules.orderbook_analyzer import OrderbookAnalyzer
                    orderbook_analyzer = OrderbookAnalyzer()
                    
                    # Get orderbook data from the internal method
                    exchange_tasks = {}
                    for exchange in ['binance', 'okx', 'bybit', 'coinbase', 'kraken']:
                        # Check coin availability before adding task
                        config = orderbook_analyzer.exchange_configs.get(exchange, {})
                        supported_coins = config.get('supported_coins', [])
                        if supported_coins and coin not in supported_coins:
                            logger.info(f"ℹ️ Skipping {exchange} orderbook in smart money analysis - {coin} not supported")
                            continue
                        exchange_tasks[exchange] = orderbook_analyzer._get_exchange_orderbook(exchange, coin)
                    
                    orderbook_results = await asyncio.gather(*exchange_tasks.values(), return_exceptions=True)
                    
                    for i, (exchange, task) in enumerate(exchange_tasks.items()):
                        result = orderbook_results[i]
                        if not isinstance(result, Exception) and result:
                            exchange_orderbook_data[exchange] = result
                            
                except Exception as e:
                    logger.warning(f"⚠️ Could not gather fresh orderbook data for smart money analysis: {str(e)}")
                    # Continue with empty data - smart money detector will handle gracefully
            
            # 🔥 PHASE 2: Gather all signals concurrently (including new enhanced signals)
            logger.info(f"📊 Gathering signals from all modules for {coin}")
            
            signal_tasks = {
                'options': self.options_analyzer.analyze_options_signal(coin),
                'indicators': self.indicator_engine.analyze_technical_signal(coin),
                'news': self.news_analyzer.analyze_news_sentiment(coin),
                'session': self.session_analyzer.analyze_session_context(coin),
                # 'gemini': removed
                'smart_money': self.smart_money_detector.detect_smart_money_activity(coin, exchange_orderbook_data),
                'multi_exchange': asyncio.create_task(asyncio.sleep(0.01, result=multi_exchange_data))  # Already computed
            }
            
            # Execute all analysis tasks concurrently
            raw_signals = {}
            results = await asyncio.gather(*signal_tasks.values(), return_exceptions=True)
            
            # Process results and handle exceptions
            for i, (signal_name, task) in enumerate(signal_tasks.items()):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"❌ {signal_name} analysis failed: {str(result)}")
                    raw_signals[signal_name] = {
                        'bias': 'neutral',
                        'confidence': 0.0,
                        'reasons': [f'{signal_name} analysis failed'],
                        'raw_data': {'error': str(result)}
                    }
                else:
                    raw_signals[signal_name] = result
            
            # Map signal names to match weight configuration
            mapped_signals = {
                'options_flow': raw_signals.get('options', {}),
                'smart_money_flow': raw_signals.get('smart_money', {}),
                'multi_exchange': raw_signals.get('multi_exchange', {}),
                'technical_indicators': raw_signals.get('indicators', {}),
                'news_sentiment': raw_signals.get('news', {}),
                'session_context': raw_signals.get('session', {}),
                # 'gemini_bonus': removed
            }
            
            # 🔥 PHASE 3: Apply weighted scoring
            logger.info(f"⚖️ Applying weighted scoring for {coin}")
            
            weighted_analysis = await self._apply_weighted_scoring(mapped_signals)
            
            # 🔥 PHASE 4: Make final trade decision
            logger.info(f"🎯 Making final trade decision for {coin}")
            
            final_decision = await self._make_trade_decision(
                weighted_analysis, mapped_signals, coin, start_time
            )
            
            # 🔥 PHASE 5: Log trade attempt
            await self._log_trade_attempt(final_decision, mapped_signals, coin)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Trade confidence computed for {coin} in {execution_time:.2f}s - Decision: {final_decision['take_trade']}")
            
            return final_decision
            
        except Exception as e:
            logger.error(f"❌ Critical error computing trade confidence for {coin}: {str(e)}")
            return await self._create_error_response(coin, str(e), start_time)
    
    async def _apply_weighted_scoring(self, raw_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Apply weighted scoring to combine all signals"""
        
        try:
            # Initialize scoring components
            weighted_scores = {
                'bullish': 0.0,
                'bearish': 0.0,
                'neutral': 0.0
            }
            
            total_weight = 0.0
            signal_contributions = {}
            
            # Process each signal with its weight
            for signal_name, signal_data in raw_signals.items():
                if signal_name not in self.weights:
                    continue
                
                weight = self.weights[signal_name]
                bias = signal_data.get('bias', 'neutral')
                confidence = signal_data.get('confidence', 0.0)
                
                # Calculate weighted contribution
                weighted_contribution = confidence * weight
                total_weight += weight
                
                # Add to appropriate bias bucket
                if bias == 'bullish':
                    weighted_scores['bullish'] += weighted_contribution
                elif bias == 'bearish':
                    weighted_scores['bearish'] += weighted_contribution
                else:
                    weighted_scores['neutral'] += weighted_contribution
                
                # Track individual contributions
                signal_contributions[signal_name] = {
                    'bias': bias,
                    'confidence': confidence,
                    'weight': weight,
                    'weighted_contribution': weighted_contribution
                }
            
            # Gemini bonus removed
            
            # Normalize scores
            if total_weight > 0:
                for bias_type in weighted_scores:
                    weighted_scores[bias_type] = max(0, weighted_scores[bias_type] / total_weight)
            
            # Determine final bias and confidence
            max_score = max(weighted_scores.values())
            final_bias = max(weighted_scores.keys(), key=lambda k: weighted_scores[k])
            
            # Special case: if scores are very close, default to neutral
            sorted_scores = sorted(weighted_scores.values(), reverse=True)
            if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < 0.1:
                final_bias = 'neutral'
                final_confidence = 0.3
            else:
                final_confidence = max_score
            
            return {
                'final_bias': final_bias,
                'final_confidence': final_confidence,
                'weighted_scores': weighted_scores,
                'signal_contributions': signal_contributions,
                'total_weight': total_weight
            }
            
        except Exception as e:
            logger.error(f"❌ Error in weighted scoring: {str(e)}")
            return {
                'final_bias': 'neutral',
                'final_confidence': 0.0,
                'weighted_scores': {'bullish': 0, 'bearish': 0, 'neutral': 0},
                'signal_contributions': {},
                'error': str(e)
            }
    
    async def _make_trade_decision(self, weighted_analysis: Dict, raw_signals: Dict, 
                                 coin: str, start_time: datetime) -> Dict[str, Any]:
        """Make final trade decision based on weighted analysis"""
        
        try:
            final_bias = weighted_analysis['final_bias']
            final_confidence = weighted_analysis['final_confidence']
            
            # Enhanced reasoning with retail vs institutional evidence
            reasons = []
            smart_money_insights = []
            exchange_insights = []
            signal_contributions = weighted_analysis.get('signal_contributions', {})
            
            # Sort signals by weighted contribution
            sorted_contributions = sorted(
                signal_contributions.items(),
                key=lambda x: x[1]['weighted_contribution'],
                reverse=True
            )
            
            # Collect enhanced reasons with smart money analysis
            for signal_name, contribution in sorted_contributions[:4]:
                if contribution['weighted_contribution'] > 0.03:  # Lower threshold for more insights
                    signal_data = raw_signals[signal_name]
                    signal_reasons = signal_data.get('reasons', [])
                    
                    # Add smart money specific insights
                    if signal_name == 'smart_money_flow' and signal_reasons:
                        institutional_activity = signal_data.get('institutional_activity_level', 0)
                        institutional_signal = signal_data.get('institutional_signal', 'neutral')
                        retail_signal = signal_data.get('retail_signal', 'neutral')
                        
                        if institutional_activity > 0.6:
                            smart_money_insights.append(f"🏦 HIGH institutional activity ({institutional_activity:.1%})")
                            if institutional_signal != 'neutral':
                                smart_money_insights.append(f"🏦 Institutions: {institutional_signal.upper()}")
                        
                        if retail_signal != 'neutral':
                            smart_money_insights.append(f"👥 Retail traders: {retail_signal.upper()}")
                        
                        reasons.extend([f"🧠 {reason}" for reason in signal_reasons[:2]])
                    
                    # Add multi-exchange insights
                    elif signal_name == 'multi_exchange' and signal_reasons:
                        exchange_insights_data = signal_data.get('exchange_insights', [])
                        if exchange_insights_data:
                            exchange_insights.extend(exchange_insights_data[:2])
                        reasons.extend([f"🌐 {reason}" for reason in signal_reasons[:2]])
                    
                    # Add other signal reasons with emojis
                    elif signal_reasons:
                        emoji_map = {
                            'options_flow': '📊',
                            'technical_indicators': '📈',
                            'news_sentiment': '📰',
                            'session_context': '🕐',
                            # 'gemini_bonus': '🔷' # removed
                        }
                        emoji = emoji_map.get(signal_name, '•')
                        reasons.extend([f"{emoji} {reason}" for reason in signal_reasons[:2]])
            
            # Add smart money insights to main reasons
            if smart_money_insights:
                reasons = smart_money_insights[:2] + reasons
            
            # Add exchange insights if significant
            if exchange_insights and len(exchange_insights) > 0:
                reasons.append(f"🌐 {exchange_insights[0]}")
            
            # If no specific reasons, create enhanced generic ones
            if not reasons:
                reasons = [f'{final_bias.title()} consensus across {len(signal_contributions)} enhanced signals']
            
            # Limit to top 6 reasons for enhanced system
            reasons = reasons[:6]
            
            # 🎯 TRADE DECISION LOGIC
            take_trade = False
            
            # Rule 1: Minimum confidence threshold
            if final_confidence >= trading_config.min_confidence_to_trade:
                # Rule 2: Bias must not be neutral
                if final_bias != 'neutral':
                    take_trade = True
                    reasons.insert(0, f'High confidence {final_bias} signal ({final_confidence:.1%})')
                else:
                    reasons.append(f'Neutral bias despite {final_confidence:.1%} confidence')
            else:
                reasons.append(f'Confidence below threshold ({final_confidence:.1%} < {trading_config.min_confidence_to_trade:.1%})')
            
            # Rule 3: Override for very high confidence
            if final_confidence >= trading_config.high_confidence_threshold:
                take_trade = True
                reasons.insert(0, f'Very high confidence override ({final_confidence:.1%})')
            
            # Rule 4: Session-based adjustments
            session_signal = raw_signals.get('session', {})
            if session_signal.get('bias') == 'neutral' and 'weekend' in str(session_signal.get('raw_data', {})):
                final_confidence *= 0.8  # Reduce confidence on weekends
                reasons.append('Weekend adjustment applied')
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'take_trade': take_trade,
                'confidence': round(final_confidence, 3),
                'bias': final_bias,
                'reasons': reasons,
                'raw_inputs': raw_signals,
                'timestamp': get_utc_timestamp(),
                'execution_time_seconds': round(execution_time, 2),
                'weighted_analysis': weighted_analysis,
                'coin': coin.upper()
            }
            
        except Exception as e:
            logger.error(f"❌ Error making trade decision: {str(e)}")
            return await self._create_error_response(coin, str(e), start_time)
    
    async def _create_error_response(self, coin: str, error_msg: str, start_time: datetime) -> Dict[str, Any]:
        """Create standardized error response"""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        error_response = {
            'take_trade': False,
            'confidence': 0.0,
            'bias': 'neutral',
            'reasons': [f'Error: {error_msg}'],
            'raw_inputs': {'error': error_msg},
            'timestamp': get_utc_timestamp(),
            'execution_time_seconds': round(execution_time, 2),
            'coin': coin.upper(),
            'error': True
        }
        
        # Log error attempt
        await self._log_trade_attempt(error_response, {}, coin)
        
        return error_response
    
    async def _log_trade_attempt(self, decision: Dict, raw_signals: Dict, coin: str):
        """Log trade attempt to JSONL file"""
        
        try:
            # Create simplified log entry
            log_entry = {
                'timestamp': decision.get('timestamp'),
                'coin': coin.upper(),
                'bias': decision.get('bias'),
                'confidence': decision.get('confidence'),
                'take_trade': decision.get('take_trade'),
                'reasons': decision.get('reasons', [])[:3],  # Top 3 reasons only
                'execution_time': decision.get('execution_time_seconds'),
                'signal_summary': {}
            }
            
            # Add simplified signal summary
            for signal_name, signal_data in raw_signals.items():
                if isinstance(signal_data, dict):
                    log_entry['signal_summary'][signal_name] = {
                        'bias': signal_data.get('bias', 'neutral'),
                        'confidence': signal_data.get('confidence', 0.0)
                    }
            
            # Add weighted analysis summary if available
            weighted_analysis = decision.get('weighted_analysis', {})
            if weighted_analysis:
                log_entry['weighted_scores'] = weighted_analysis.get('weighted_scores', {})
            
            # Write to log file
            log_trade_attempt(log_entry)
            
            logger.info(f"📝 Trade attempt logged for {coin}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log trade attempt for {coin}: {str(e)}")

# Global engine instance
_engine_instance = None

async def compute_trade_confidence(coin: str) -> Dict[str, Any]:
    """
    Global function to compute trade confidence
    This is the main entry point for the trading system
    """
    global _engine_instance
    
    if _engine_instance is None:
        _engine_instance = TradingSignalEngine()
    
    return await _engine_instance.compute_trade_confidence(coin)

# Convenience functions for testing individual modules
async def test_options_signal(coin: str) -> Dict[str, Any]:
    """Test options analyzer independently"""
    analyzer = OptionsAnalyzer()
    return await analyzer.analyze_options_signal(coin)

async def test_orderbook_signal(coin: str) -> Dict[str, Any]:
    """Test orderbook analyzer independently"""
    analyzer = OrderbookAnalyzer()
    return await analyzer.analyze_orderbook_signal(coin)

async def test_technical_signal(coin: str) -> Dict[str, Any]:
    """Test technical indicators independently"""
    analyzer = IndicatorEngine()
    return await analyzer.analyze_technical_signal(coin)

async def test_news_signal(coin: str) -> Dict[str, Any]:
    """Test news sentiment independently"""
    analyzer = NewsAnalyzer()
    return await analyzer.analyze_news_sentiment(coin)

async def test_session_signal(coin: str) -> Dict[str, Any]:
    """Test session context independently"""
    analyzer = SessionAnalyzer()
    return await analyzer.analyze_session_context(coin)

# Gemini test function removed

async def test_smart_money_signal(coin: str) -> Dict[str, Any]:
    """Test smart money detector independently"""
    # First get multi-exchange data
    multi_exchange_analyzer = MultiExchangeAnalyzer()
    multi_exchange_data = await multi_exchange_analyzer.enhanced_multi_exchange_analysis(coin)
    
    # Extract orderbook data
    exchange_orderbook_data = {}
    if 'raw_data' in multi_exchange_data and isinstance(multi_exchange_data['raw_data'], dict):
        for analysis_type, analysis_data in multi_exchange_data['raw_data'].items():
            if isinstance(analysis_data, dict) and 'exchange_data' in analysis_data:
                for exchange, data in analysis_data.get('exchange_data', {}).items():
                    if 'orderbook' in data:
                        exchange_orderbook_data[exchange] = data['orderbook']
    
    # Run smart money detection
    smart_money_detector = SmartMoneyDetector()
    result = await smart_money_detector.detect_smart_money_activity(coin, exchange_orderbook_data)
    
    # Normalize to expected structure for testing
    normalized_result = {
        'bias': 'neutral',
        'confidence': result.get('smart_money_confidence', 0.0),
        'reasons': result.get('reasons', []),
        'raw_data': result.get('raw_data', {}),
        'institutional_signal': result.get('institutional_signal', 'neutral'),
        'retail_signal': result.get('retail_signal', 'neutral'),
        'institutional_activity_level': result.get('institutional_activity_level', 0.0)
    }
    
    # Determine overall bias from institutional and retail signals
    institutional_signal = result.get('institutional_signal', 'neutral')
    retail_signal = result.get('retail_signal', 'neutral')
    institutional_activity = result.get('institutional_activity_level', 0.0)
    
    # If institutional activity is high, prioritize institutional signal
    if institutional_activity > 0.5:
        normalized_result['bias'] = institutional_signal
    elif institutional_signal != 'neutral':
        normalized_result['bias'] = institutional_signal
    elif retail_signal != 'neutral':
        normalized_result['bias'] = retail_signal
    
    # Add smart money specific reasons if none exist
    if not normalized_result['reasons']:
        normalized_result['reasons'] = [
            f"Institutional signal: {institutional_signal}",
            f"Retail signal: {retail_signal}",
            f"Institutional activity: {institutional_activity:.1%}"
        ]
    
    return normalized_result

async def test_multi_exchange_signal(coin: str) -> Dict[str, Any]:
    """Test multi-exchange analyzer independently"""
    analyzer = MultiExchangeAnalyzer()
    return await analyzer.enhanced_multi_exchange_analysis(coin)

if __name__ == "__main__":
    # Simple test runner
    import sys
    
    async def main():
        if len(sys.argv) < 2:
            print("Usage: python main_engine.py <COIN> [test_module]")
            print("Examples:")
            print("  python main_engine.py BTC")
            print("  python main_engine.py BTC options")
            print("  python main_engine.py ETH orderbook")
            return
        
        coin = sys.argv[1].upper()
        test_module = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"🚀 Testing {coin} analysis...")
        
        if test_module == 'options':
            result = await test_options_signal(coin)
        elif test_module == 'orderbook':
            result = await test_orderbook_signal(coin)
        elif test_module == 'technical':
            result = await test_technical_signal(coin)
        elif test_module == 'news':
            result = await test_news_signal(coin)
        elif test_module == 'session':
            result = await test_session_signal(coin)
        # gemini test removed
        else:
            result = await compute_trade_confidence(coin)
        
        print(f"\n📊 Result for {coin}:")
        print(json.dumps(result, indent=2))
    
    asyncio.run(main())