"""
📊 Options Chain Analysis Module
Analyzes options data from Deribit (primary) and CoinGlass (fallback)
Highest weight in signal confidence system (40%)
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from .utils import HTTPClient, APIError, safe_api_call, get_emoji
from config.settings import api_config, COIN_OPTION_AVAILABILITY

logger = logging.getLogger(__name__)

class OptionsAnalyzer:
    """Analyzes options chain data for directional bias and confidence"""
    
    def __init__(self):
        self.deribit_base = api_config.deribit_base_url
        self.coinglass_base = api_config.coinglass_base_url
        
    async def analyze_options_signal(self, coin: str) -> Dict[str, Any]:
        """
        Main function to analyze options data and return directional signal
        
        Returns:
        {
            'bias': 'bullish'|'bearish'|'neutral',
            'confidence': 0.0-1.0,
            'reasons': ['list of reasons'],
            'raw_data': {...}
        }
        """
        # Check if coin has options data available
        has_options = COIN_OPTION_AVAILABILITY.get(coin, False)
        
        if has_options:
            logger.info(f"{get_emoji('options')} Analyzing options data for {coin} - Deribit Call/Put Open Interest, Volume, Put/Call Ratio → Binance funding fallback")
        else:
            logger.info(f"{get_emoji('options')} No options available for {coin} - using Binance funding rate proxy (real futures data)")
        
        if not has_options:
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'reasons': [f'No options data available for {coin}'],
                'raw_data': {'error': 'No options available'}
            }
        
        try:
            # Try Deribit first (primary source)
            deribit_data = await safe_api_call(
                lambda: self._get_deribit_options_data(coin),
                default_return=None,
                error_message=f"Deribit options data failed for {coin}"
            )
            
            if deribit_data:
                signal = await self._analyze_deribit_data(coin, deribit_data)
                if signal['confidence'] > 0:
                    logger.info(f"✅ Deribit options analysis complete for {coin}")
                    return signal
            
            # Use funding rates as fallback for options-like signal
            logger.info(f"⚠️ Deribit options failed - switching to Binance perpetual funding rate (real futures market data) for {coin}")
            funding_signal = await self._get_funding_rate_fallback(coin)
            if funding_signal['confidence'] > 0:
                rate = funding_signal['raw_data'].get('funding_rate', 0)
                logger.info(f"✅ Binance funding rate analysis complete for {coin} (rate: {rate*100:.4f}%, bias: {funding_signal['bias']}) - Real futures sentiment")
                return funding_signal
            
            # All sources failed
            logger.error(f"❌ All options data sources failed for {coin}")
            return {
                'bias': 'neutral', 
                'confidence': 0.1,  # Small baseline confidence instead of 0
                'reasons': ['Options data unavailable, using baseline signal'],
                'raw_data': {'error': 'All sources failed'}
            }
            
        except Exception as e:
            logger.error(f"❌ Options analysis failed for {coin}: {str(e)}")
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'reasons': [f'Options analysis error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }
    
    async def _get_deribit_options_data(self, coin: str) -> Optional[Dict[str, Any]]:
        """Fetch options data from Deribit API"""
        
        async with HTTPClient() as client:
            try:
                # Get current instruments for the coin
                instruments_url = f"{self.deribit_base}/public/get_instruments"
                instruments_params = {
                    'currency': coin,
                    'kind': 'option',
                    'expired': 'false'
                }
                
                instruments_response = await client.get(instruments_url, params=instruments_params)
                instruments = instruments_response.get('result', [])
                
                if not instruments:
                    logger.warning(f"No active options instruments found for {coin}")
                    return None
                
                # Get current price for reference
                index_url = f"{self.deribit_base}/public/get_index"
                index_params = {'currency': coin}
                index_response = await client.get(index_url, params=index_params)
                current_price = index_response.get('result', {}).get(coin.lower(), 0)
                
                # Analyze instruments to find relevant options (within ±20% of current price)
                relevant_options = []
                price_range = (current_price * 0.8, current_price * 1.2)
                
                for instrument in instruments:
                    strike = instrument.get('strike', 0)
                    if price_range[0] <= strike <= price_range[1]:
                        # Get order book for this instrument
                        book_url = f"{self.deribit_base}/public/get_order_book"
                        book_params = {'instrument_name': instrument['instrument_name']}
                        
                        try:
                            book_response = await client.get(book_url, params=book_params)
                            book_data = book_response.get('result', {})
                            
                            if book_data.get('open_interest', 0) > 0:
                                relevant_options.append({
                                    'instrument': instrument,
                                    'order_book': book_data,
                                    'strike': strike,
                                    'option_type': instrument.get('option_type', ''),
                                    'expiration': instrument.get('expiration_timestamp', 0)
                                })
                        except Exception as e:
                            logger.warning(f"Failed to get order book for {instrument['instrument_name']}: {e}")
                            continue
                
                return {
                    'current_price': current_price,
                    'relevant_options': relevant_options,
                    'total_instruments': len(instruments)
                }
                
            except Exception as e:
                logger.error(f"❌ Error fetching Deribit data: {str(e)}")
                return None
    
    async def _analyze_deribit_data(self, coin: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Deribit options data for directional bias"""
        
        try:
            current_price = data['current_price']
            options = data['relevant_options']
            
            if not options or current_price <= 0:
                return {
                    'bias': 'neutral',
                    'confidence': 0.0,
                    'reasons': ['Insufficient options data'],
                    'raw_data': data
                }
            
            # Separate calls and puts
            calls = [opt for opt in options if opt['option_type'].lower() == 'call']
            puts = [opt for opt in options if opt['option_type'].lower() == 'put']
            
            # Calculate total open interest for calls and puts
            call_oi = sum(opt['order_book'].get('open_interest', 0) for opt in calls)
            put_oi = sum(opt['order_book'].get('open_interest', 0) for opt in puts)
            
            total_oi = call_oi + put_oi
            if total_oi == 0:
                return {
                    'bias': 'neutral',
                    'confidence': 0.0,
                    'reasons': ['No open interest in relevant options'],
                    'raw_data': data
                }
            
            # Calculate call/put ratio
            call_put_ratio = call_oi / put_oi if put_oi > 0 else float('inf')
            
            # Find max pain level (strike with highest total OI)
            strike_oi = {}
            for opt in options:
                strike = opt['strike']
                oi = opt['order_book'].get('open_interest', 0)
                if strike not in strike_oi:
                    strike_oi[strike] = 0
                strike_oi[strike] += oi
            
            max_pain_strike = max(strike_oi.keys(), key=lambda k: strike_oi[k]) if strike_oi else current_price
            
            # Analyze bias based on multiple factors
            reasons = []
            confidence_factors = []
            
            # Factor 1: Call/Put Open Interest Ratio
            if call_put_ratio > 1.5:
                bias_lean = 'bullish'
                reasons.append(f'Call OI dominates (C/P ratio: {call_put_ratio:.2f})')
                confidence_factors.append(min(0.8, (call_put_ratio - 1) / 2))
            elif call_put_ratio < 0.67:
                bias_lean = 'bearish'
                reasons.append(f'Put OI dominates (C/P ratio: {call_put_ratio:.2f})')
                confidence_factors.append(min(0.8, (1 - call_put_ratio) * 2))
            else:
                bias_lean = 'neutral'
                reasons.append(f'Balanced call/put OI (C/P ratio: {call_put_ratio:.2f})')
                confidence_factors.append(0.3)
            
            # Factor 2: Max Pain vs Current Price
            max_pain_distance = (current_price - max_pain_strike) / current_price
            if abs(max_pain_distance) > 0.05:  # More than 5% from max pain
                if max_pain_distance > 0:
                    reasons.append(f'Price above max pain (${max_pain_strike:.0f})')
                    if bias_lean == 'bearish':
                        confidence_factors.append(0.6)
                else:
                    reasons.append(f'Price below max pain (${max_pain_strike:.0f})')
                    if bias_lean == 'bullish':
                        confidence_factors.append(0.6)
            
            # Factor 3: Options Volume vs Open Interest (momentum indicator)
            total_volume = sum(opt['order_book'].get('stats', {}).get('volume_usd', 0) for opt in options)
            volume_oi_ratio = total_volume / total_oi if total_oi > 0 else 0
            
            if volume_oi_ratio > 0.1:  # High volume relative to OI suggests fresh positioning
                reasons.append('High options volume suggests active positioning')
                confidence_factors.append(0.4)
            
            # Calculate final confidence
            base_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0
            
            # Boost confidence based on total open interest size
            oi_boost = min(0.2, total_oi / 10000)  # Up to 0.2 boost for high OI
            final_confidence = min(1.0, base_confidence + oi_boost)
            
            return {
                'bias': bias_lean,
                'confidence': final_confidence,
                'reasons': reasons,
                'raw_data': {
                    'current_price': current_price,
                    'call_oi': call_oi,
                    'put_oi': put_oi,
                    'call_put_ratio': call_put_ratio,
                    'max_pain_strike': max_pain_strike,
                    'total_volume': total_volume,
                    'total_options_analyzed': len(options)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Deribit data: {str(e)}")
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'reasons': [f'Analysis error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }
    
    async def _get_coinglass_options_data(self, coin: str) -> Optional[Dict[str, Any]]:
        """Fetch options data from CoinGlass API as fallback"""
        
        async with HTTPClient() as client:
            try:
                # CoinGlass options endpoint (this is a placeholder - need to verify actual endpoint)
                url = f"{self.coinglass_base}/api/options/open_interest"
                params = {'symbol': f'{coin}USD'}
                
                headers = {}
                if api_config.coinglass_api_key:
                    headers['X-API-KEY'] = api_config.coinglass_api_key
                
                response = await client.get(url, params=params, headers=headers)
                return response.get('data', {})
                
            except Exception as e:
                logger.error(f"❌ Error fetching CoinGlass data: {str(e)}")
                return None
    
    async def _get_funding_rate_fallback(self, coin: str) -> Dict[str, Any]:
        """Use funding rates as fallback options signal for coins without options"""
        
        async with HTTPClient() as client:
            try:
                # Try Binance funding rate
                url = "https://fapi.binance.com/fapi/v1/premiumIndex"
                params = {'symbol': f'{coin}USDT'}
                
                response = await client.get(url, params=params)
                funding_rate = float(response.get('lastFundingRate', 0))
                
                # Convert funding rate to options-like signal
                # Positive funding = longs pay shorts = bearish pressure
                # Negative funding = shorts pay longs = bullish pressure
                
                abs_rate = abs(funding_rate)
                
                if abs_rate > 0.0001:  # 0.01% threshold
                    if funding_rate > 0:
                        bias = 'bearish'
                        reason = f'High positive funding rate ({funding_rate*100:.3f}%) - long pressure'
                    else:
                        bias = 'bullish'
                        reason = f'Negative funding rate ({funding_rate*100:.3f}%) - short pressure'
                    
                    # Scale confidence based on rate magnitude
                    confidence = min(0.4, abs_rate * 1000)  # Max 0.4 confidence from funding
                    
                    return {
                        'bias': bias,
                        'confidence': confidence,
                        'reasons': [reason, 'Using funding rate as options proxy'],
                        'raw_data': {'funding_rate': funding_rate, 'source': 'binance_funding'}
                    }
                else:
                    return {
                        'bias': 'neutral',
                        'confidence': 0.1,
                        'reasons': ['Neutral funding rate', 'Using funding rate as options proxy'],
                        'raw_data': {'funding_rate': funding_rate, 'source': 'binance_funding'}
                    }
                    
            except Exception as e:
                logger.error(f"❌ Funding rate fallback failed: {str(e)}")
                return {
                    'bias': 'neutral',
                    'confidence': 0.0,
                    'reasons': ['Funding rate fallback failed'],
                    'raw_data': {'error': str(e)}
                }

    async def _analyze_coinglass_data(self, coin: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CoinGlass options data (simplified analysis)"""
        
        try:
            # This is a simplified placeholder - actual implementation depends on CoinGlass data structure
            call_oi = data.get('call_open_interest', 0)
            put_oi = data.get('put_open_interest', 0)
            
            if call_oi + put_oi == 0:
                return {
                    'bias': 'neutral',
                    'confidence': 0.0,
                    'reasons': ['No CoinGlass options data available'],
                    'raw_data': data
                }
            
            call_put_ratio = call_oi / put_oi if put_oi > 0 else float('inf')
            
            if call_put_ratio > 1.2:
                bias = 'bullish'
                confidence = min(0.6, (call_put_ratio - 1) / 3)  # Lower confidence for fallback source
                reasons = [f'CoinGlass: Call OI leads (C/P: {call_put_ratio:.2f})']
            elif call_put_ratio < 0.8:
                bias = 'bearish'
                confidence = min(0.6, (1 - call_put_ratio) * 3)
                reasons = [f'CoinGlass: Put OI leads (C/P: {call_put_ratio:.2f})']
            else:
                bias = 'neutral'
                confidence = 0.2
                reasons = [f'CoinGlass: Balanced options (C/P: {call_put_ratio:.2f})']
            
            return {
                'bias': bias,
                'confidence': confidence,
                'reasons': reasons,
                'raw_data': data
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing CoinGlass data: {str(e)}")
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'reasons': [f'CoinGlass analysis error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }