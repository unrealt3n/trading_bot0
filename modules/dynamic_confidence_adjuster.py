"""
🎯 Dynamic Confidence Adjuster
Adjusts trade confidence based on support/resistance levels, market conditions, and trade duration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class DynamicConfidenceAdjuster:
    """Dynamically adjusts trade confidence based on market context"""
    
    def __init__(self):
        # Adjustment weights for different factors
        self.adjustment_factors = {
            'support_resistance': 0.30,  # Major S/R levels
            'market_structure': 0.25,    # Trend/range conditions  
            'volatility': 0.20,          # Market volatility impact
            'session_context': 0.15,     # Trading session context
            'volume_profile': 0.10       # Volume-based adjustments
        }
        
        # Support/Resistance adjustment rules
        self.sr_adjustments = {
            'approaching_resistance_long': -0.25,    # Reduce confidence 25% for longs near resistance
            'approaching_support_short': -0.25,      # Reduce confidence 25% for shorts near support
            'breaking_resistance_long': +0.20,       # Increase confidence 20% for resistance breaks
            'breaking_support_short': +0.20,         # Increase confidence 20% for support breaks
            'in_range': -0.10,                       # Slightly reduce in ranging markets
            'clear_direction': +0.15                 # Increase when clear trend direction
        }
        
        # Market structure adjustments
        self.structure_adjustments = {
            'strong_trend': +0.15,      # Trending markets
            'weak_trend': +0.05,        # Weak trends
            'ranging': -0.15,           # Ranging markets
            'consolidation': -0.10,     # Consolidation phases
            'breakout': +0.25,          # Fresh breakouts
            'reversal': +0.20           # Clear reversals
        }
        
        # Volatility adjustments by trade duration
        self.volatility_adjustments = {
            'scalp': {
                'low_vol': -0.20,      # Scalping needs volatility
                'normal_vol': 0.0,
                'high_vol': +0.15
            },
            'short': {
                'low_vol': +0.05,      # Short-term can work in normal vol
                'normal_vol': +0.10,
                'high_vol': +0.05
            },
            'mid': {
                'low_vol': +0.15,      # Mid-term prefers lower vol
                'normal_vol': +0.10,
                'high_vol': -0.15
            }
        }
        
        logger.info("🎯 Dynamic Confidence Adjuster initialized")
    
    async def adjust_confidence(self, coin: str, base_confidence: float, trade_bias: str, 
                               duration: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust base confidence based on market conditions and trade parameters
        
        Args:
            coin: Trading pair
            base_confidence: Initial confidence (0.0-1.0)
            trade_bias: 'bullish' or 'bearish'
            duration: 'scalp', 'short', or 'mid'
            market_data: Current market analysis data
            
        Returns:
            {
                'adjusted_confidence': float,
                'confidence_change': float,
                'adjustments': {...},
                'reasons': [...]
            }
        """
        
        try:
            logger.info(f"🎯 Adjusting confidence for {coin} {trade_bias} {duration} trade")
            
            total_adjustment = 0.0
            adjustments = {}
            reasons = []
            
            # 1. Support/Resistance adjustments
            sr_adjustment, sr_reason = await self._calculate_sr_adjustment(
                coin, trade_bias, market_data
            )
            total_adjustment += sr_adjustment * self.adjustment_factors['support_resistance']
            adjustments['support_resistance'] = sr_adjustment
            if sr_reason:
                reasons.append(sr_reason)
            
            # 2. Market structure adjustments
            structure_adjustment, structure_reason = await self._calculate_structure_adjustment(
                coin, trade_bias, market_data
            )
            total_adjustment += structure_adjustment * self.adjustment_factors['market_structure']
            adjustments['market_structure'] = structure_adjustment
            if structure_reason:
                reasons.append(structure_reason)
            
            # 3. Volatility adjustments
            vol_adjustment, vol_reason = await self._calculate_volatility_adjustment(
                coin, duration, market_data
            )
            total_adjustment += vol_adjustment * self.adjustment_factors['volatility']
            adjustments['volatility'] = vol_adjustment
            if vol_reason:
                reasons.append(vol_reason)
            
            # 4. Session context adjustments
            session_adjustment, session_reason = await self._calculate_session_adjustment(
                coin, duration, market_data
            )
            total_adjustment += session_adjustment * self.adjustment_factors['session_context']
            adjustments['session_context'] = session_adjustment
            if session_reason:
                reasons.append(session_reason)
            
            # 5. Volume profile adjustments
            volume_adjustment, volume_reason = await self._calculate_volume_adjustment(
                coin, trade_bias, market_data
            )
            total_adjustment += volume_adjustment * self.adjustment_factors['volume_profile']
            adjustments['volume_profile'] = volume_adjustment
            if volume_reason:
                reasons.append(volume_reason)
            
            # Apply total adjustment
            adjusted_confidence = max(0.0, min(1.0, base_confidence + total_adjustment))
            confidence_change = adjusted_confidence - base_confidence
            
            logger.info(f"🎯 {coin} confidence: {base_confidence:.3f} → {adjusted_confidence:.3f} "
                       f"({confidence_change:+.3f})")
            
            return {
                'adjusted_confidence': adjusted_confidence,
                'confidence_change': confidence_change,
                'total_adjustment': total_adjustment,
                'adjustments': adjustments,
                'reasons': reasons[:5],  # Top 5 reasons
                'base_confidence': base_confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Error adjusting confidence for {coin}: {e}")
            return {
                'adjusted_confidence': base_confidence,
                'confidence_change': 0.0,
                'total_adjustment': 0.0,
                'adjustments': {},
                'reasons': [f"Adjustment error: {str(e)}"],
                'base_confidence': base_confidence
            }
    
    async def _calculate_sr_adjustment(self, coin: str, trade_bias: str, 
                                      market_data: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Calculate support/resistance based confidence adjustment"""
        
        try:
            # Extract current price and S/R levels from market data
            current_price = market_data.get('current_price', 0)
            if not current_price:
                return 0.0, None
            
            # Look for S/R data in technical analysis
            technical_data = market_data.get('raw_inputs', {}).get('indicator_analysis', {})
            sr_levels = technical_data.get('support_resistance', {})
            
            if not sr_levels:
                return 0.0, None
            
            resistance_levels = sr_levels.get('resistance', [])
            support_levels = sr_levels.get('support', [])
            
            # Calculate distance to nearest S/R levels
            nearest_resistance = self._find_nearest_level(current_price, resistance_levels, 'above')
            nearest_support = self._find_nearest_level(current_price, support_levels, 'below')
            
            adjustment = 0.0
            reason = None
            
            if trade_bias == 'bullish':
                # Long trade - check resistance levels
                if nearest_resistance:
                    distance_pct = (nearest_resistance - current_price) / current_price * 100
                    
                    if distance_pct < 1.0:  # Very close to resistance (<1%)
                        adjustment = self.sr_adjustments['approaching_resistance_long']
                        reason = f"Approaching resistance at ${nearest_resistance:.4f} ({distance_pct:.1f}%)"
                    elif distance_pct < 0:  # Breaking resistance
                        adjustment = self.sr_adjustments['breaking_resistance_long']
                        reason = f"Breaking resistance at ${nearest_resistance:.4f}"
                    elif distance_pct > 5.0:  # Clear path to resistance
                        adjustment = self.sr_adjustments['clear_direction'] * 0.5
                        reason = f"Clear path to resistance ({distance_pct:.1f}%)"
            
            elif trade_bias == 'bearish':
                # Short trade - check support levels
                if nearest_support:
                    distance_pct = (current_price - nearest_support) / current_price * 100
                    
                    if distance_pct < 1.0:  # Very close to support (<1%)
                        adjustment = self.sr_adjustments['approaching_support_short']
                        reason = f"Approaching support at ${nearest_support:.4f} ({distance_pct:.1f}%)"
                    elif distance_pct < 0:  # Breaking support
                        adjustment = self.sr_adjustments['breaking_support_short']
                        reason = f"Breaking support at ${nearest_support:.4f}"
                    elif distance_pct > 5.0:  # Clear path to support
                        adjustment = self.sr_adjustments['clear_direction'] * 0.5
                        reason = f"Clear path to support ({distance_pct:.1f}%)"
            
            # Check if in range-bound market
            if nearest_resistance and nearest_support:
                range_size = (nearest_resistance - nearest_support) / current_price * 100
                if range_size < 3.0:  # Tight range
                    adjustment += self.sr_adjustments['in_range']
                    if not reason:
                        reason = f"Tight range ({range_size:.1f}%)"
            
            return adjustment, reason
            
        except Exception as e:
            logger.error(f"❌ Error calculating S/R adjustment: {e}")
            return 0.0, None
    
    def _find_nearest_level(self, current_price: float, levels: List[float], 
                           direction: str) -> Optional[float]:
        """Find nearest support/resistance level"""
        
        if not levels:
            return None
        
        if direction == 'above':
            # Find nearest resistance above current price
            above_levels = [level for level in levels if level > current_price]
            return min(above_levels) if above_levels else None
        else:
            # Find nearest support below current price
            below_levels = [level for level in levels if level < current_price]
            return max(below_levels) if below_levels else None
    
    async def _calculate_structure_adjustment(self, coin: str, trade_bias: str, 
                                            market_data: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Calculate market structure based adjustment"""
        
        try:
            # Extract trend and structure data
            technical_data = market_data.get('raw_inputs', {}).get('indicator_analysis', {})
            trend_data = technical_data.get('trend_analysis', {})
            
            if not trend_data:
                return 0.0, None
            
            trend_strength = trend_data.get('trend_strength', 'neutral')
            trend_direction = trend_data.get('trend_direction', 'neutral')
            market_phase = trend_data.get('market_phase', 'ranging')
            
            adjustment = 0.0
            reason = None
            
            # Trend alignment bonus
            if trend_direction == trade_bias and trend_strength == 'strong':
                adjustment += self.structure_adjustments['strong_trend']
                reason = f"Strong {trend_direction} trend alignment"
            elif trend_direction == trade_bias and trend_strength == 'weak':
                adjustment += self.structure_adjustments['weak_trend']
                reason = f"Weak {trend_direction} trend alignment"
            elif trend_direction != trade_bias and trend_direction != 'neutral':
                adjustment -= 0.10  # Counter-trend penalty
                reason = f"Counter-trend trade ({trend_direction} trend)"
            
            # Market phase adjustments
            if market_phase == 'breakout':
                adjustment += self.structure_adjustments['breakout']
                if not reason:
                    reason = "Fresh breakout pattern"
            elif market_phase == 'reversal':
                adjustment += self.structure_adjustments['reversal']
                if not reason:
                    reason = "Clear reversal pattern"
            elif market_phase == 'ranging':
                adjustment += self.structure_adjustments['ranging']
                if not reason:
                    reason = "Range-bound market"
            elif market_phase == 'consolidation':
                adjustment += self.structure_adjustments['consolidation']
                if not reason:
                    reason = "Market consolidation"
            
            return adjustment, reason
            
        except Exception as e:
            logger.error(f"❌ Error calculating structure adjustment: {e}")
            return 0.0, None
    
    async def _calculate_volatility_adjustment(self, coin: str, duration: str, 
                                             market_data: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Calculate volatility based adjustment"""
        
        try:
            # Extract volatility data
            technical_data = market_data.get('raw_inputs', {}).get('indicator_analysis', {})
            volatility_data = technical_data.get('volatility', {})
            
            if not volatility_data:
                return 0.0, None
            
            current_vol = volatility_data.get('current_volatility', 0)
            avg_vol = volatility_data.get('average_volatility', 0)
            vol_percentile = volatility_data.get('volatility_percentile', 50)
            
            # Determine volatility regime
            if vol_percentile < 30:
                vol_regime = 'low_vol'
            elif vol_percentile > 70:
                vol_regime = 'high_vol'
            else:
                vol_regime = 'normal_vol'
            
            # Get duration-specific adjustment
            duration_adjustments = self.volatility_adjustments.get(duration, {})
            adjustment = duration_adjustments.get(vol_regime, 0.0)
            
            reason = None
            if adjustment != 0:
                reason = f"{vol_regime.replace('_', ' ').title()} for {duration} trade"
            
            return adjustment, reason
            
        except Exception as e:
            logger.error(f"❌ Error calculating volatility adjustment: {e}")
            return 0.0, None
    
    async def _calculate_session_adjustment(self, coin: str, duration: str, 
                                          market_data: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Calculate trading session based adjustment"""
        
        try:
            # Extract session data
            session_data = market_data.get('raw_inputs', {}).get('session_analysis', {})
            
            if not session_data:
                return 0.0, None
            
            current_session = session_data.get('current_session', 'unknown')
            session_overlap = session_data.get('session_overlap', False)
            volume_profile = session_data.get('volume_profile', 'normal')
            
            adjustment = 0.0
            reason = None
            
            # Session-based adjustments
            if duration == 'scalp':
                # Scalping prefers high-volume sessions
                if current_session in ['london', 'newyork'] or session_overlap:
                    adjustment += 0.10
                    reason = f"High-volume {current_session} session"
                elif current_session == 'asian':
                    adjustment -= 0.05
                    reason = "Lower volume Asian session"
            
            elif duration == 'short':
                # Short-term trades can work in most sessions
                if session_overlap:
                    adjustment += 0.05
                    reason = "Session overlap period"
            
            elif duration == 'mid':
                # Mid-term less affected by session
                adjustment += 0.02  # Slight positive bias
                reason = "Session-neutral for mid-term"
            
            # Volume profile adjustment
            if volume_profile == 'high':
                adjustment += 0.05
            elif volume_profile == 'low':
                adjustment -= 0.05
            
            return adjustment, reason
            
        except Exception as e:
            logger.error(f"❌ Error calculating session adjustment: {e}")
            return 0.0, None
    
    async def _calculate_volume_adjustment(self, coin: str, trade_bias: str, 
                                         market_data: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Calculate volume profile based adjustment"""
        
        try:
            # Extract volume data
            technical_data = market_data.get('raw_inputs', {}).get('indicator_analysis', {})
            volume_data = technical_data.get('volume_analysis', {})
            
            if not volume_data:
                return 0.0, None
            
            volume_trend = volume_data.get('volume_trend', 'neutral')
            volume_confirmation = volume_data.get('volume_confirmation', False)
            relative_volume = volume_data.get('relative_volume', 1.0)
            
            adjustment = 0.0
            reason = None
            
            # Volume confirmation
            if volume_confirmation:
                adjustment += 0.10
                reason = "Volume confirms price action"
            elif volume_trend == 'decreasing':
                adjustment -= 0.05
                reason = "Decreasing volume divergence"
            
            # Relative volume
            if relative_volume > 1.5:  # High volume
                adjustment += 0.05
                if not reason:
                    reason = f"High relative volume ({relative_volume:.1f}x)"
            elif relative_volume < 0.5:  # Low volume
                adjustment -= 0.05
                if not reason:
                    reason = f"Low relative volume ({relative_volume:.1f}x)"
            
            return adjustment, reason
            
        except Exception as e:
            logger.error(f"❌ Error calculating volume adjustment: {e}")
            return 0.0, None
    
    def get_adjustment_summary(self, adjustments: Dict[str, float]) -> str:
        """Generate human-readable adjustment summary"""
        
        summary_parts = []
        
        for factor, adjustment in adjustments.items():
            if adjustment != 0:
                direction = "increased" if adjustment > 0 else "decreased"
                percentage = abs(adjustment) * 100
                factor_name = factor.replace('_', ' ').title()
                summary_parts.append(f"{factor_name}: {direction} by {percentage:.1f}%")
        
        return " | ".join(summary_parts) if summary_parts else "No adjustments applied"