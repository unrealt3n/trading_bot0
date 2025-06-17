"""
🚀 Trade Duration Engine
Determines optimal trade duration (scalping/short-term/mid-term) based on signal types and market conditions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class TradeDuration(Enum):
    SCALP = "scalp"      # 1-15 minutes
    SHORT = "short"      # 1-8 hours  
    MID = "mid"          # 1-7 days

class TradeDurationEngine:
    """Determines optimal trade duration based on market signals"""
    
    def __init__(self):
        # Duration thresholds and weights
        self.duration_weights = {
            'whale_alerts': 0.30,
            'liquidations': 0.25,
            'smart_money': 0.20,
            'technical': 0.15,
            'news': 0.10
        }
        
        # Signal strength thresholds for each duration
        self.duration_thresholds = {
            TradeDuration.SCALP: {
                'min_whale_strength': 70,
                'min_liquidation_strength': 60,
                'max_duration_minutes': 15,
                'required_signals': ['whale_alerts', 'liquidations']
            },
            TradeDuration.SHORT: {
                'min_smart_money_divergence': 65,
                'min_technical_confluence': 60,
                'max_duration_hours': 8,
                'required_signals': ['smart_money', 'technical']
            },
            TradeDuration.MID: {
                'min_news_impact': 70,
                'min_institutional_signal': 65,
                'max_duration_days': 7,
                'required_signals': ['news', 'smart_money']
            }
        }
        
        logger.info("🕒 Trade Duration Engine initialized")
    
    async def determine_trade_duration(self, coin: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine optimal trade duration based on signal analysis
        
        Returns:
        {
            'duration': TradeDuration.SCALP/SHORT/MID,
            'duration_minutes': int,
            'confidence': float,
            'primary_signals': [...],
            'exit_strategy': {...}
        }
        """
        
        try:
            logger.info(f"🕒 Determining trade duration for {coin}")
            
            # Extract signal strengths
            signals = self._extract_signal_strengths(analysis_data)
            
            # Score each duration type
            duration_scores = {}
            for duration in TradeDuration:
                score = await self._score_duration(duration, signals)
                duration_scores[duration] = score
            
            # Select best duration
            best_duration = max(duration_scores.items(), key=lambda x: x[1]['score'])
            selected_duration, duration_data = best_duration
            
            # Generate exit strategy
            exit_strategy = self._generate_exit_strategy(selected_duration, signals, analysis_data)
            
            result = {
                'duration': selected_duration.value,
                'duration_minutes': self._get_duration_minutes(selected_duration),
                'confidence': duration_data['score'],
                'primary_signals': duration_data['primary_signals'],
                'exit_strategy': exit_strategy,
                'signal_breakdown': signals,
                'all_scores': {d.value: s['score'] for d, s in duration_scores.items()}
            }
            
            logger.info(f"✅ {coin} duration: {selected_duration.value.upper()} ({result['duration_minutes']}min) - {result['confidence']:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error determining trade duration for {coin}: {e}")
            return self._get_default_duration()
    
    def _extract_signal_strengths(self, analysis_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract signal strengths from analysis data"""
        
        signals = {
            'whale_alerts': 0.0,
            'liquidations': 0.0,
            'smart_money': 0.0,
            'technical': 0.0,
            'news': 0.0
        }
        
        try:
            # Extract from weighted analysis if available
            weighted_analysis = analysis_data.get('weighted_analysis', {})
            signal_contributions = weighted_analysis.get('signal_contributions', {})
            
            # Map signal contributions to our duration signals
            for signal_name, contrib in signal_contributions.items():
                confidence = contrib.get('confidence', 0) * 100
                
                # Map analyzer names to duration signals
                if 'whale' in signal_name.lower() or 'smart_money' in signal_name.lower():
                    signals['whale_alerts'] = max(signals['whale_alerts'], confidence)
                    signals['smart_money'] = max(signals['smart_money'], confidence)
                elif 'liquidation' in signal_name.lower():
                    signals['liquidations'] = max(signals['liquidations'], confidence)
                elif 'technical' in signal_name.lower() or 'indicator' in signal_name.lower():
                    signals['technical'] = max(signals['technical'], confidence)
                elif 'news' in signal_name.lower():
                    signals['news'] = max(signals['news'], confidence)
            
            # Also check raw inputs for additional signals
            raw_inputs = analysis_data.get('raw_inputs', {})
            
            # Check for specific whale/liquidation data
            if 'whale_data' in raw_inputs:
                whale_strength = raw_inputs['whale_data'].get('signal_strength', 0)
                signals['whale_alerts'] = max(signals['whale_alerts'], whale_strength)
            
            if 'liquidation_data' in raw_inputs:
                liq_strength = raw_inputs['liquidation_data'].get('signal_strength', 0)
                signals['liquidations'] = max(signals['liquidations'], liq_strength)
            
        except Exception as e:
            logger.error(f"❌ Error extracting signal strengths: {e}")
        
        return signals
    
    async def _score_duration(self, duration: TradeDuration, signals: Dict[str, float]) -> Dict[str, Any]:
        """Score a specific duration based on signal requirements"""
        
        thresholds = self.duration_thresholds[duration]
        score = 0.0
        primary_signals = []
        
        try:
            if duration == TradeDuration.SCALP:
                # Scalping requires strong whale + liquidation signals
                whale_score = min(signals['whale_alerts'] / thresholds['min_whale_strength'], 1.0)
                liq_score = min(signals['liquidations'] / thresholds['min_liquidation_strength'], 1.0)
                
                if whale_score >= 0.7:
                    primary_signals.append('whale_alerts')
                if liq_score >= 0.6:
                    primary_signals.append('liquidations')
                
                # Both signals needed for scalping
                if len(primary_signals) >= 2:
                    score = (whale_score * 0.6 + liq_score * 0.4) * 100
                else:
                    score = max(whale_score, liq_score) * 50  # Penalty for missing required signal
            
            elif duration == TradeDuration.SHORT:
                # Short-term requires smart money + technical confluence
                smart_score = min(signals['smart_money'] / thresholds['min_smart_money_divergence'], 1.0)
                tech_score = min(signals['technical'] / thresholds['min_technical_confluence'], 1.0)
                
                if smart_score >= 0.65:
                    primary_signals.append('smart_money')
                if tech_score >= 0.6:
                    primary_signals.append('technical')
                
                # Both signals preferred for short-term
                if len(primary_signals) >= 2:
                    score = (smart_score * 0.6 + tech_score * 0.4) * 100
                else:
                    score = max(smart_score, tech_score) * 70  # Less penalty than scalping
            
            elif duration == TradeDuration.MID:
                # Mid-term requires news + institutional signals
                news_score = min(signals['news'] / thresholds['min_news_impact'], 1.0)
                inst_score = min(signals['smart_money'] / thresholds['min_institutional_signal'], 1.0)
                
                if news_score >= 0.7:
                    primary_signals.append('news')
                if inst_score >= 0.65:
                    primary_signals.append('smart_money')
                
                # Can work with one strong signal for mid-term
                if len(primary_signals) >= 1:
                    score = (news_score * 0.5 + inst_score * 0.5) * 100
                else:
                    score = max(news_score, inst_score) * 80
        
        except Exception as e:
            logger.error(f"❌ Error scoring duration {duration.value}: {e}")
        
        return {
            'score': score,
            'primary_signals': primary_signals,
            'signal_scores': {
                'whale_alerts': signals['whale_alerts'],
                'liquidations': signals['liquidations'],
                'smart_money': signals['smart_money'],
                'technical': signals['technical'],
                'news': signals['news']
            }
        }
    
    def _generate_exit_strategy(self, duration: TradeDuration, signals: Dict[str, float], analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate exit strategy based on duration and signals"""
        
        base_confidence = analysis_data.get('confidence', 0.5) * 100
        
        if duration == TradeDuration.SCALP:
            return {
                'take_profit_percent': 0.5 + (base_confidence - 50) * 0.01,  # 0.5-1.0%
                'stop_loss_percent': 0.3 + (100 - base_confidence) * 0.005,  # 0.3-0.55%
                'max_duration_minutes': 15,
                'trail_stop': True,
                'partial_exits': [
                    {'at_profit': 0.3, 'exit_percent': 50},  # Take 50% at 0.3% profit
                    {'at_profit': 0.6, 'exit_percent': 30}   # Take 30% at 0.6% profit
                ]
            }
        
        elif duration == TradeDuration.SHORT:
            return {
                'take_profit_percent': 1.5 + (base_confidence - 50) * 0.02,  # 1.5-2.5%
                'stop_loss_percent': 0.8 + (100 - base_confidence) * 0.01,   # 0.8-1.3%
                'max_duration_hours': 8,
                'trail_stop': True,
                'partial_exits': [
                    {'at_profit': 1.0, 'exit_percent': 40},  # Take 40% at 1% profit
                    {'at_profit': 2.0, 'exit_percent': 30}   # Take 30% at 2% profit
                ]
            }
        
        else:  # MID-term
            return {
                'take_profit_percent': 3.0 + (base_confidence - 50) * 0.05,  # 3.0-5.5%
                'stop_loss_percent': 1.5 + (100 - base_confidence) * 0.02,   # 1.5-2.5%
                'max_duration_days': 7,
                'trail_stop': False,  # Less aggressive for longer trades
                'partial_exits': [
                    {'at_profit': 2.0, 'exit_percent': 25},  # Take 25% at 2% profit
                    {'at_profit': 4.0, 'exit_percent': 25}   # Take 25% at 4% profit
                ]
            }
    
    def _get_duration_minutes(self, duration: TradeDuration) -> int:
        """Get duration in minutes for tracking"""
        if duration == TradeDuration.SCALP:
            return 15
        elif duration == TradeDuration.SHORT:
            return 480  # 8 hours
        else:  # MID
            return 10080  # 7 days
    
    def _get_default_duration(self) -> Dict[str, Any]:
        """Return default duration when analysis fails"""
        return {
            'duration': TradeDuration.SHORT.value,
            'duration_minutes': 240,  # 4 hours
            'confidence': 50.0,
            'primary_signals': ['technical'],
            'exit_strategy': self._generate_exit_strategy(TradeDuration.SHORT, {}, {'confidence': 0.5}),
            'signal_breakdown': {},
            'all_scores': {}
        }