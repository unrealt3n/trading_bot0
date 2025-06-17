"""
🚀 Enhanced Main Trading Signal Engine
Integrates whale alerts, liquidations, retail vs big money, and dynamic confidence adjustment
Provides high-probability trading signals with duration-specific strategies
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

# Import existing analysis modules
from modules.options_analyzer import OptionsAnalyzer
from modules.orderbook_analyzer import OrderbookAnalyzer
from modules.indicator_engine import IndicatorEngine
from modules.news_analyzer import NewsAnalyzer
from modules.session_analyzer import SessionAnalyzer
# Gemini analyzer removed - unnecessary
from modules.smart_money_detector import SmartMoneyDetector
from modules.multi_exchange_analyzer import MultiExchangeAnalyzer

# Import new enhanced modules
from modules.whale_alert_analyzer import WhaleAlertAnalyzer
from modules.liquidation_analyzer import LiquidationAnalyzer
from modules.retail_vs_bigmoney_detector import RetailVsBigMoneyDetector
from modules.trade_duration_engine import TradeDurationEngine
from modules.dynamic_confidence_adjuster import DynamicConfidenceAdjuster
from modules.trade_tracker import TradeTracker

# Import utilities and config
from modules.utils import get_emoji, get_utc_timestamp, log_trade_attempt
from config.settings import weight_config, trading_config, system_config

logger = logging.getLogger(__name__)

class EnhancedTradingSignalEngine:
    """Enhanced main engine with whale alerts, liquidations, and dynamic features"""
    
    def __init__(self, whale_alert_api_key: Optional[str] = None):
        # Initialize existing analyzers
        self.options_analyzer = OptionsAnalyzer()
        self.orderbook_analyzer = OrderbookAnalyzer()
        self.indicator_engine = IndicatorEngine()
        self.news_analyzer = NewsAnalyzer()
        self.session_analyzer = SessionAnalyzer()
# Gemini analyzer removed
        self.smart_money_detector = SmartMoneyDetector()
        self.multi_exchange_analyzer = MultiExchangeAnalyzer()
        
        # Initialize new enhanced analyzers
        self.whale_analyzer = WhaleAlertAnalyzer(api_key=whale_alert_api_key)
        self.liquidation_analyzer = LiquidationAnalyzer()
        self.retail_detector = RetailVsBigMoneyDetector()
        self.duration_engine = TradeDurationEngine()
        self.confidence_adjuster = DynamicConfidenceAdjuster()
        self.trade_tracker = TradeTracker()
        
        # Enhanced weight configuration
        self.enhanced_weights = {
            # High-impact signals
            'whale_alerts': 0.20,           # Immediate price impact
            'liquidations': 0.18,           # Momentum signals
            'retail_vs_bigmoney': 0.15,     # Directional bias
            
            # Medium-impact signals  
            'smart_money': 0.12,            # Institutional flows
            'technical_analysis': 0.10,     # Support/resistance
            'news_sentiment': 0.08,         # Fundamental context
            
            # Supporting signals
            'options_flow': 0.07,           # Options positioning
            'orderbook': 0.05,              # Microstructure
            'session_analysis': 0.03,       # Session context
            'multi_exchange': 0.02          # Cross-exchange validation
        }
        
        # Normalize weights
        total_weight = sum(self.enhanced_weights.values())
        self.enhanced_weights = {k: v/total_weight for k, v in self.enhanced_weights.items()}
        
        logger.info(f"🚀 Enhanced Trading Signal Engine initialized")
        logger.info(f"📊 Enhanced weights: {self.enhanced_weights}")
    
    async def compute_enhanced_trade_confidence(self, coin: str) -> Dict[str, Any]:
        """
        Enhanced trade confidence computation with all new features
        
        Returns:
        {
            'take_trade': bool,
            'confidence': float,
            'bias': str,
            'duration': str,
            'duration_data': {...},
            'enhanced_analysis': {...},
            'trade_tracker_data': {...},
            'reasons': [...],
            'timestamp': str
        }
        """
        
        logger.info(f"🎯 Computing enhanced trade confidence for {coin}")
        start_time = datetime.now()
        
        try:
            # Phase 1: Gather all signal data concurrently
            logger.info(f"📊 Phase 1: Gathering enhanced signal data for {coin}")
            
            # High-priority signals (run concurrently)
            whale_task = self.whale_analyzer.analyze_whale_activity(coin)
            liquidation_task = self._analyze_liquidations_safely(coin)
            retail_task = self._analyze_retail_safely(coin)
            
            # Existing signals
            technical_task = self.indicator_engine.analyze_technical_signal(coin)
            news_task = self.news_analyzer.analyze_news_sentiment(coin)
            smart_money_task = self.smart_money_detector.detect_smart_money_activity(coin, {})
            
            # Wait for all high-priority signals
            whale_data, liquidation_data, retail_data, technical_data, news_data, smart_money_data = await asyncio.gather(
                whale_task, liquidation_task, retail_task, technical_task, news_task, smart_money_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            whale_data = self._handle_analysis_exception(whale_data, "whale_alerts")
            liquidation_data = self._handle_analysis_exception(liquidation_data, "liquidations")
            retail_data = self._handle_analysis_exception(retail_data, "retail_vs_bigmoney")
            technical_data = self._handle_analysis_exception(technical_data, "technical_analysis")
            news_data = self._handle_analysis_exception(news_data, "news_sentiment")
            smart_money_data = self._handle_analysis_exception(smart_money_data, "smart_money")
            
            # Phase 2: Calculate weighted signal scores
            logger.info(f"⚖️ Phase 2: Calculating weighted signals for {coin}")
            
            signal_scores = self._calculate_enhanced_signal_scores({
                'whale_alerts': whale_data,
                'liquidations': liquidation_data,
                'retail_vs_bigmoney': retail_data,
                'technical_analysis': technical_data,
                'news_sentiment': news_data,
                'smart_money': smart_money_data
            })
            
            # Phase 3: Determine base confidence and bias
            base_confidence, base_bias = self._calculate_base_confidence_and_bias(signal_scores)
            
            # Phase 4: Determine optimal trade duration
            logger.info(f"⏰ Phase 4: Determining trade duration for {coin}")
            
            market_data = {
                'raw_inputs': {
                    'whale_data': whale_data,
                    'liquidation_data': liquidation_data,
                    'retail_data': retail_data,
                    'technical_data': technical_data,
                    'news_data': news_data,
                    'smart_money_data': smart_money_data
                },
                'weighted_analysis': {
                    'signal_contributions': signal_scores
                },
                'confidence': base_confidence,
                'bias': base_bias
            }
            
            duration_data = await self.duration_engine.determine_trade_duration(coin, market_data)
            
            # Phase 5: Apply dynamic confidence adjustments
            logger.info(f"🎯 Phase 5: Applying dynamic confidence adjustments for {coin}")
            
            adjustment_data = await self.confidence_adjuster.adjust_confidence(
                coin, base_confidence, base_bias, duration_data['duration'], market_data
            )
            
            final_confidence = adjustment_data['adjusted_confidence']
            
            # Phase 6: Make final trade decision
            take_trade = self._should_take_enhanced_trade(
                coin, final_confidence, base_bias, duration_data, signal_scores
            )
            
            # Phase 7: Generate comprehensive analysis
            enhanced_analysis = {
                'signal_scores': signal_scores,
                'base_confidence': base_confidence,
                'final_confidence': final_confidence,
                'confidence_adjustments': adjustment_data,
                'duration_analysis': duration_data,
                'whale_analysis': whale_data,
                'liquidation_analysis': liquidation_data,
                'retail_analysis': retail_data,
                'execution_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
            
            # Generate reasons
            reasons = self._generate_enhanced_reasons(
                base_bias, signal_scores, duration_data, adjustment_data
            )
            
            # Final result
            result = {
                'take_trade': take_trade,
                'confidence': final_confidence,
                'bias': base_bias,
                'duration': duration_data['duration'],
                'duration_data': duration_data,
                'enhanced_analysis': enhanced_analysis,
                'reasons': reasons,
                'timestamp': get_utc_timestamp(),
                'coin': coin,
                'signal_strength': final_confidence * 100
            }
            
            # Log detailed results
            self._log_enhanced_results(coin, result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced trade confidence computation for {coin}: {e}")
            return await self._create_error_response(coin, str(e), start_time)
    
    async def _analyze_liquidations_safely(self, coin: str) -> Dict[str, Any]:
        """Safely analyze liquidations with context manager"""
        try:
            async with self.liquidation_analyzer:
                return await self.liquidation_analyzer.analyze_liquidations(coin)
        except Exception as e:
            logger.error(f"❌ Liquidation analysis error for {coin}: {e}")
            return {'signal_strength': 0, 'bias': 'neutral', 'confidence': 0}
    
    async def _analyze_retail_safely(self, coin: str) -> Dict[str, Any]:
        """Safely analyze retail vs big money with context manager"""
        try:
            async with self.retail_detector:
                return await self.retail_detector.analyze_retail_vs_bigmoney(coin)
        except Exception as e:
            logger.error(f"❌ Retail analysis error for {coin}: {e}")
            return {'signal_strength': 0, 'bias': 'neutral', 'confidence': 0}
    
    def _handle_analysis_exception(self, result: Any, analyzer_name: str) -> Dict[str, Any]:
        """Handle exceptions from analysis tasks"""
        if isinstance(result, Exception):
            logger.error(f"❌ {analyzer_name} analysis failed: {result}")
            return {
                'signal_strength': 0,
                'bias': 'neutral',
                'confidence': 0,
                'error': str(result)
            }
        return result if isinstance(result, dict) else {}
    
    def _calculate_enhanced_signal_scores(self, signal_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate weighted signal scores for all analyzers"""
        
        signal_scores = {}
        
        for signal_name, data in signal_data.items():
            try:
                if not data or data.get('error'):
                    continue
                
                # Extract signal metrics
                confidence = data.get('confidence', 0)
                signal_strength = data.get('signal_strength', 0) / 100.0 if data.get('signal_strength', 0) > 1 else data.get('signal_strength', 0)
                bias = data.get('bias', 'neutral')
                
                # Normalize confidence
                if confidence > 1:
                    confidence = confidence / 100.0
                
                # Calculate weighted contribution
                weight = self.enhanced_weights.get(signal_name, 0)
                weighted_contribution = confidence * weight
                
                # Apply directional bias
                if bias == 'bearish':
                    weighted_contribution *= -1
                elif bias == 'neutral':
                    weighted_contribution *= 0.5
                
                signal_scores[signal_name] = {
                    'confidence': confidence,
                    'bias': bias,
                    'weight': weight,
                    'weighted_contribution': weighted_contribution,
                    'signal_strength': signal_strength,
                    'raw_data': data
                }
                
            except Exception as e:
                logger.error(f"❌ Error processing {signal_name} signals: {e}")
                continue
        
        return signal_scores
    
    def _calculate_base_confidence_and_bias(self, signal_scores: Dict[str, Dict[str, Any]]) -> tuple[float, str]:
        """Calculate base confidence and bias from weighted signals"""
        
        total_weighted_score = 0.0
        bullish_weight = 0.0
        bearish_weight = 0.0
        
        for signal_name, score_data in signal_scores.items():
            contribution = score_data['weighted_contribution']
            total_weighted_score += abs(contribution)
            
            if contribution > 0:
                bullish_weight += contribution
            elif contribution < 0:
                bearish_weight += abs(contribution)
        
        # Calculate base confidence
        base_confidence = min(total_weighted_score, 1.0)
        
        # Determine bias
        if bullish_weight > bearish_weight * 1.2:  # 20% threshold
            bias = 'bullish'
        elif bearish_weight > bullish_weight * 1.2:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        return base_confidence, bias
    
    def _should_take_enhanced_trade(self, coin: str, confidence: float, bias: str, 
                                   duration_data: Dict[str, Any], signal_scores: Dict[str, Dict[str, Any]]) -> bool:
        """Enhanced trade decision logic"""
        
        # Basic confidence threshold
        min_confidence = 0.65  # 65%
        if confidence < min_confidence:
            logger.info(f"📊 {coin}: Confidence {confidence:.1%} < {min_confidence:.1%} threshold")
            return False
        
        # Must have clear directional bias
        if bias == 'neutral':
            logger.info(f"📊 {coin}: Neutral bias - no trade")
            return False
        
        # Duration-specific requirements
        duration = duration_data['duration']
        duration_confidence = duration_data['confidence']
        
        if duration == 'scalp':
            # Scalping requires high confidence and strong signals
            if confidence < 0.75 or duration_confidence < 70:
                logger.info(f"📊 {coin}: Scalp requirements not met (conf: {confidence:.1%}, dur_conf: {duration_confidence:.1f})")
                return False
            
            # Scalping requires whale alerts OR liquidations
            whale_signal = signal_scores.get('whale_alerts', {}).get('confidence', 0)
            liq_signal = signal_scores.get('liquidations', {}).get('confidence', 0)
            
            if whale_signal < 0.6 and liq_signal < 0.6:
                logger.info(f"📊 {coin}: Scalp requires strong whale/liquidation signals")
                return False
        
        elif duration == 'short':
            # Short-term requires good confidence
            if confidence < 0.70:
                logger.info(f"📊 {coin}: Short-term confidence {confidence:.1%} too low")
                return False
        
        elif duration == 'mid':
            # Mid-term can work with lower confidence but needs institutional signals
            if confidence < 0.60:
                logger.info(f"📊 {coin}: Mid-term confidence {confidence:.1%} too low")
                return False
            
            # Mid-term benefits from retail vs big money divergence
            retail_signal = signal_scores.get('retail_vs_bigmoney', {}).get('confidence', 0)
            if retail_signal < 0.5:
                logger.info(f"📊 {coin}: Mid-term benefits from retail divergence signals")
        
        # Check for signal confluence (multiple strong signals)
        strong_signals = sum(1 for score in signal_scores.values() if score.get('confidence', 0) > 0.7)
        if strong_signals < 2:
            logger.info(f"📊 {coin}: Need at least 2 strong signals (have {strong_signals})")
            return False
        
        logger.info(f"✅ {coin}: All enhanced trade conditions met")
        return True
    
    def _generate_enhanced_reasons(self, bias: str, signal_scores: Dict[str, Dict[str, Any]], 
                                  duration_data: Dict[str, Any], adjustment_data: Dict[str, Any]) -> List[str]:
        """Generate comprehensive reasons for the trading decision"""
        
        reasons = []
        
        # Top contributing signals
        sorted_signals = sorted(
            signal_scores.items(),
            key=lambda x: abs(x[1].get('weighted_contribution', 0)),
            reverse=True
        )
        
        for signal_name, score_data in sorted_signals[:3]:
            if abs(score_data.get('weighted_contribution', 0)) > 0.05:  # Significant contribution
                confidence = score_data.get('confidence', 0) * 100
                signal_bias = score_data.get('bias', 'neutral')
                readable_name = signal_name.replace('_', ' ').title()
                reasons.append(f"{readable_name}: {signal_bias} ({confidence:.1f}%)")
        
        # Duration reasoning
        duration = duration_data.get('duration', 'unknown')
        primary_signals = duration_data.get('primary_signals', [])
        if primary_signals:
            signal_list = ', '.join(primary_signals)
            reasons.append(f"{duration.title()} trade based on {signal_list}")
        
        # Confidence adjustments
        adjustment_reasons = adjustment_data.get('reasons', [])
        if adjustment_reasons:
            reasons.extend(adjustment_reasons[:2])  # Top 2 adjustment reasons
        
        return reasons[:5]  # Return top 5 reasons
    
    def _log_enhanced_results(self, coin: str, result: Dict[str, Any]):
        """Log enhanced analysis results"""
        
        confidence = result['confidence']
        bias = result['bias']
        duration = result['duration']
        take_trade = result['take_trade']
        
        # Main result
        action = "✅ TAKE TRADE" if take_trade else "⏸️ HOLD"
        logger.info(f"🎯 {coin} ENHANCED RESULT: {action}")
        logger.info(f"   Confidence: {confidence:.1%} | Bias: {bias.upper()} | Duration: {duration.upper()}")
        
        # Signal breakdown
        signal_scores = result['enhanced_analysis']['signal_scores']
        logger.info(f"📊 Signal Breakdown for {coin}:")
        
        for signal_name, score_data in sorted(signal_scores.items(), 
                                            key=lambda x: abs(x[1].get('weighted_contribution', 0)), 
                                            reverse=True):
            confidence_pct = score_data.get('confidence', 0) * 100
            contribution_pct = score_data.get('weighted_contribution', 0) * 100
            weight_pct = score_data.get('weight', 0) * 100
            signal_bias = score_data.get('bias', 'neutral')
            
            logger.info(f"   • {signal_name}: {signal_bias.upper()} ({confidence_pct:.1f}%) "
                       f"| Weight: {weight_pct:.1f}% | Contribution: {contribution_pct:+.1f}%")
        
        # Duration details
        duration_data = result['duration_data']
        duration_minutes = duration_data.get('duration_minutes', 0)
        exit_strategy = duration_data.get('exit_strategy', {})
        
        logger.info(f"⏰ Duration Details: {duration.upper()} ({duration_minutes}min)")
        if exit_strategy:
            tp_pct = exit_strategy.get('take_profit_percent', 0)
            sl_pct = exit_strategy.get('stop_loss_percent', 0)
            logger.info(f"   TP: {tp_pct:.2f}% | SL: {sl_pct:.2f}%")
        
        # Confidence adjustments
        adjustment_data = result['enhanced_analysis']['confidence_adjustments']
        base_conf = adjustment_data.get('base_confidence', 0)
        final_conf = adjustment_data.get('adjusted_confidence', 0)
        change = final_conf - base_conf
        
        if abs(change) > 0.01:  # Significant adjustment
            logger.info(f"🎯 Confidence Adjusted: {base_conf:.3f} → {final_conf:.3f} ({change:+.3f})")
            adjustments = adjustment_data.get('adjustments', {})
            for factor, adj in adjustments.items():
                if abs(adj) > 0.01:
                    direction = "↑" if adj > 0 else "↓"
                    logger.info(f"   {factor}: {direction} {abs(adj)*100:.1f}%")
    
    async def _create_error_response(self, coin: str, error: str, start_time: datetime) -> Dict[str, Any]:
        """Create error response for failed analysis"""
        
        return {
            'take_trade': False,
            'confidence': 0.0,
            'bias': 'neutral',
            'duration': 'short',
            'duration_data': {},
            'enhanced_analysis': {
                'error': error,
                'execution_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            },
            'reasons': [f"Analysis error: {error}"],
            'timestamp': get_utc_timestamp(),
            'coin': coin,
            'signal_strength': 0.0
        }

# Create global instance for backward compatibility
enhanced_engine = EnhancedTradingSignalEngine()

async def compute_trade_confidence(coin: str) -> Dict[str, Any]:
    """Global function for backward compatibility"""
    return await enhanced_engine.compute_enhanced_trade_confidence(coin)