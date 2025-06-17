"""
🕐 Market Session Context Analyzer
Detects current trading session and adjusts signal weights accordingly
Weight: 10% in signal confidence system
"""

import logging
from typing import Dict, Any
from datetime import datetime, timezone
from .utils import get_current_session, calculate_volatility_multiplier, get_emoji
from config.settings import trading_config, SESSION_VOLATILITY_MULTIPLIERS

logger = logging.getLogger(__name__)

class SessionAnalyzer:
    """Analyzes current market session context and its impact on trading signals"""
    
    def __init__(self):
        self.session_characteristics = {
            'asia': {
                'volatility': 'low',
                'volume': 'medium',
                'key_markets': ['Japan', 'South Korea', 'Singapore', 'Hong Kong'],
                'typical_behavior': 'consolidation',
                'bias_tendency': 'neutral'
            },
            'europe': {
                'volatility': 'medium',
                'volume': 'high',
                'key_markets': ['London', 'Frankfurt', 'Zurich'],
                'typical_behavior': 'trend_following',
                'bias_tendency': 'continuation'
            },
            'us': {
                'volatility': 'high',
                'volume': 'very_high',
                'key_markets': ['New York', 'Chicago'],
                'typical_behavior': 'breakouts',
                'bias_tendency': 'momentum'
            },
            'overlap': {
                'volatility': 'very_high',
                'volume': 'maximum',
                'key_markets': ['Multi-region'],
                'typical_behavior': 'major_moves',
                'bias_tendency': 'amplified'
            }
        }
    
    async def analyze_session_context(self, coin: str) -> Dict[str, Any]:
        """
        Analyze current market session and its impact on signal reliability
        
        Returns:
        {
            'bias': 'bullish'|'bearish'|'neutral',
            'confidence': 0.0-1.0,
            'reasons': ['list of reasons'],
            'raw_data': {...}
        }
        """
        logger.info(f"{get_emoji('session')} Analyzing market session context for {coin}")
        
        try:
            current_time = datetime.now(timezone.utc)
            current_session = get_current_session()
            volatility_multiplier = calculate_volatility_multiplier(current_session)
            
            session_info = self.session_characteristics.get(current_session, {})
            
            # Analyze session-specific behavior patterns
            session_analysis = self._analyze_session_patterns(current_session, current_time, coin)
            
            # Calculate session impact on signal reliability
            reliability_impact = self._calculate_reliability_impact(current_session, session_info)
            
            # Determine if current session favors any particular bias
            session_bias = self._determine_session_bias(current_session, session_info, current_time)
            
            # Combine analysis
            final_analysis = self._combine_session_analysis(
                current_session, session_analysis, reliability_impact, session_bias, volatility_multiplier
            )
            
            logger.info(f"✅ Session analysis complete for {coin} - {current_session} session")
            return final_analysis
            
        except Exception as e:
            logger.error(f"❌ Session analysis failed for {coin}: {str(e)}")
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'reasons': [f'Session analysis error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }
    
    def _analyze_session_patterns(self, session: str, current_time: datetime, coin: str) -> Dict[str, Any]:
        """Analyze historical patterns for current session"""
        
        try:
            hour_utc = current_time.hour
            day_of_week = current_time.weekday()  # 0 = Monday, 6 = Sunday
            
            patterns = {
                'session': session,
                'hour_utc': hour_utc,
                'day_of_week': day_of_week,
                'is_weekend': day_of_week >= 5,
                'session_start': self._is_session_start(session, hour_utc),
                'session_end': self._is_session_end(session, hour_utc)
            }
            
            # Weekend behavior (crypto markets are 24/7 but institutional activity is lower)
            if patterns['is_weekend']:
                patterns['institutional_activity'] = 'low'
                patterns['retail_dominance'] = 'high'
                patterns['volatility_expectation'] = 'lower'
            else:
                patterns['institutional_activity'] = 'normal'
                patterns['retail_dominance'] = 'normal'
                patterns['volatility_expectation'] = 'normal'
            
            # Session start/end effects
            if patterns['session_start']:
                patterns['breakout_probability'] = 'high'
                patterns['direction_change_probability'] = 'medium'
            elif patterns['session_end']:
                patterns['profit_taking_probability'] = 'high'
                patterns['consolidation_probability'] = 'high'
            else:
                patterns['trend_continuation_probability'] = 'high'
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Error analyzing session patterns: {str(e)}")
            return {'session': session, 'error': str(e)}
    
    def _is_session_start(self, session: str, hour_utc: int) -> bool:
        """Check if we're at the start of a trading session"""
        session_starts = {
            'asia': 0,      # 00:00 UTC
            'europe': 7,    # 07:00 UTC
            'us': 13        # 13:00 UTC (1 PM)
        }
        
        start_hour = session_starts.get(session)
        if start_hour is None:
            return False
        
        # Consider first 2 hours as "session start"
        return start_hour <= hour_utc <= start_hour + 2
    
    def _is_session_end(self, session: str, hour_utc: int) -> bool:
        """Check if we're at the end of a trading session"""
        session_ends = {
            'asia': [6, 7, 8],      # 06:00-08:00 UTC
            'europe': [15, 16],     # 15:00-16:00 UTC  
            'us': [21, 22]          # 21:00-22:00 UTC
        }
        
        end_hours = session_ends.get(session, [])
        return hour_utc in end_hours
    
    def _calculate_reliability_impact(self, session: str, session_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate how current session affects signal reliability"""
        
        try:
            # Base reliability multipliers for different signal types
            reliability_multipliers = {
                'options_signals': 1.0,
                'orderbook_signals': 1.0,
                'technical_signals': 1.0,
                'news_signals': 1.0
            }
            
            # Adjust based on session characteristics
            volatility = session_info.get('volatility', 'medium')
            volume = session_info.get('volume', 'medium')
            
            # High volatility sessions make technical signals more reliable
            if volatility in ['high', 'very_high']:
                reliability_multipliers['technical_signals'] *= 1.2
                reliability_multipliers['orderbook_signals'] *= 1.3
            elif volatility == 'low':
                reliability_multipliers['technical_signals'] *= 0.8
                reliability_multipliers['orderbook_signals'] *= 0.7
            
            # High volume sessions make orderbook signals more reliable
            if volume in ['high', 'very_high', 'maximum']:
                reliability_multipliers['orderbook_signals'] *= 1.2
                reliability_multipliers['options_signals'] *= 1.1
            
            # News impact varies by session
            if session == 'us':
                # US session often has more market-moving news
                reliability_multipliers['news_signals'] *= 1.3
            elif session == 'asia':
                # Asia session typically less news-driven
                reliability_multipliers['news_signals'] *= 0.7
            
            return reliability_multipliers
            
        except Exception as e:
            logger.error(f"❌ Error calculating reliability impact: {str(e)}")
            return {'options_signals': 1.0, 'orderbook_signals': 1.0, 'technical_signals': 1.0, 'news_signals': 1.0}
    
    def _determine_session_bias(self, session: str, session_info: Dict[str, Any], current_time: datetime) -> Dict[str, Any]:
        """Determine if current session has any inherent directional bias"""
        
        try:
            bias_analysis = {
                'directional_bias': 'neutral',
                'bias_strength': 0.0,
                'reasons': []
            }
            
            hour_utc = current_time.hour
            typical_behavior = session_info.get('typical_behavior', 'neutral')
            
            # Session-specific bias patterns
            if session == 'asia':
                # Asia session often sees consolidation, slight bearish bias at end
                if 6 <= hour_utc <= 8:  # End of Asia session
                    bias_analysis['directional_bias'] = 'bearish'
                    bias_analysis['bias_strength'] = 0.2
                    bias_analysis['reasons'].append('Asia session end - typical profit taking')
                
            elif session == 'europe':
                # Europe session often continues trends from Asia
                if 7 <= hour_utc <= 9:  # Start of Europe session
                    bias_analysis['directional_bias'] = 'neutral'  # Wait for direction
                    bias_analysis['bias_strength'] = 0.0
                    bias_analysis['reasons'].append('Europe session start - direction setting phase')
                
            elif session == 'us':
                # US session often has stronger directional moves
                if 13 <= hour_utc <= 15:  # Start of US session
                    bias_analysis['directional_bias'] = 'neutral'  # Could go either way strongly
                    bias_analysis['bias_strength'] = 0.0
                    bias_analysis['reasons'].append('US session start - potential for strong moves')
                elif 20 <= hour_utc <= 22:  # End of US session
                    bias_analysis['directional_bias'] = 'bearish'
                    bias_analysis['bias_strength'] = 0.15
                    bias_analysis['reasons'].append('US session end - profit taking before Asia')
                
            elif session == 'overlap':
                # Overlap periods often see amplified moves
                bias_analysis['directional_bias'] = 'neutral'
                bias_analysis['bias_strength'] = 0.0
                bias_analysis['reasons'].append('Session overlap - signals may be amplified')
            
            # Weekend adjustments (lower institutional activity)
            if current_time.weekday() >= 5:  # Weekend
                bias_analysis['bias_strength'] *= 0.5  # Reduce bias strength
                bias_analysis['reasons'].append('Weekend - reduced institutional activity')
            
            return bias_analysis
            
        except Exception as e:
            logger.error(f"❌ Error determining session bias: {str(e)}")
            return {'directional_bias': 'neutral', 'bias_strength': 0.0, 'reasons': []}
    
    def _combine_session_analysis(self, session: str, patterns: Dict, reliability: Dict, 
                                bias_info: Dict, volatility_multiplier: float) -> Dict[str, Any]:
        """Combine all session analysis components into final result"""
        
        try:
            # Extract session bias
            session_bias = bias_info.get('directional_bias', 'neutral')
            bias_strength = bias_info.get('bias_strength', 0.0)
            
            # Calculate confidence based on session characteristics
            base_confidence = 0.3  # Base confidence for session analysis
            
            # Adjust confidence based on volatility and patterns
            if patterns.get('is_weekend', False):
                base_confidence *= 0.7  # Lower confidence on weekends
            
            if patterns.get('session_start', False) or patterns.get('session_end', False):
                base_confidence *= 1.2  # Higher confidence at session transitions
            
            # Apply volatility multiplier to confidence
            final_confidence = base_confidence * min(1.5, volatility_multiplier)
            
            # Collect reasons
            reasons = []
            reasons.extend(bias_info.get('reasons', []))
            
            # Add session-specific insights
            if session == 'us':
                reasons.append('US session - higher volatility expected')
            elif session == 'asia':
                reasons.append('Asia session - typically lower volatility')
            elif session == 'europe':
                reasons.append('Europe session - trend continuation likely')
            elif session == 'overlap':
                reasons.append('Session overlap - amplified signal reliability')
            
            # Add pattern-based reasons
            if patterns.get('is_weekend', False):
                reasons.append('Weekend - reduced institutional participation')
            
            return {
                'bias': session_bias,
                'confidence': min(1.0, final_confidence),
                'reasons': reasons[:3],  # Limit to top 3 reasons
                'raw_data': {
                    'current_session': session,
                    'volatility_multiplier': volatility_multiplier,
                    'session_patterns': patterns,
                    'reliability_multipliers': reliability,
                    'bias_strength': bias_strength,
                    'weekend_adjustment': patterns.get('is_weekend', False)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error combining session analysis: {str(e)}")
            return {
                'bias': 'neutral',
                'confidence': 0.1,
                'reasons': [f'Session analysis combination error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }