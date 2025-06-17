"""
📈 Technical Indicators Engine
Analyzes RSI, MACD, EMA crossovers, Bollinger Bands with dynamic thresholds
Weight: 15% in signal confidence system
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .utils import HTTPClient, APIError, safe_api_call, get_emoji, normalize_signal
from config.settings import api_config

logger = logging.getLogger(__name__)

class IndicatorEngine:
    """Technical indicator analysis with dynamic thresholds per coin"""
    
    def __init__(self):
        # Use Binance for OHLCV data (most reliable and comprehensive)
        self.data_source = api_config.binance_base_url
        
        # Dynamic thresholds based on coin volatility
        self.default_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_strong_oversold': 20,
            'rsi_strong_overbought': 80,
            'bb_squeeze_threshold': 0.02,  # 2% bandwidth for squeeze
            'volume_spike_threshold': 2.0   # 2x average volume
        }
    
    async def analyze_technical_signal(self, coin: str) -> Dict[str, Any]:
        """
        Main function to analyze technical indicators and return directional signal
        
        Returns:
        {
            'bias': 'bullish'|'bearish'|'neutral',
            'confidence': 0.0-1.0,
            'reasons': ['list of reasons'],
            'raw_data': {...}
        }
        """
        indicators_list = ['RSI', 'MACD', 'EMA (20/50)', 'Bollinger Bands', 'Volume Analysis']
        timeframes = ['1h', '4h', '1d']
        logger.info(f"{get_emoji('indicators')} Analyzing technical indicators for {coin} - Indicators: {', '.join(indicators_list)} | Timeframes: {', '.join(timeframes)}")
        
        try:
            # Fetch OHLCV data for multiple timeframes
            timeframes = ['1h', '4h', '1d']  # Multiple timeframe analysis
            ohlcv_data = {}
            
            for tf in timeframes:
                data = await safe_api_call(
                    lambda timeframe=tf: self._get_ohlcv_data(coin, timeframe),
                    default_return=None,
                    error_message=f"OHLCV data failed for {coin} {tf}"
                )
                if data:
                    ohlcv_data[tf] = data
            
            if not ohlcv_data:
                logger.error(f"❌ No OHLCV data available for {coin}")
                return {
                    'bias': 'neutral',
                    'confidence': 0.0,
                    'reasons': ['No price data available for technical analysis'],
                    'raw_data': {'error': 'No OHLCV data'}
                }
            
            # Calculate indicators for each timeframe
            indicator_results = {}
            for tf, data in ohlcv_data.items():
                indicators = await self._calculate_all_indicators(coin, data, tf)
                if indicators:
                    indicator_results[tf] = indicators
            
            if not indicator_results:
                return {
                    'bias': 'neutral',
                    'confidence': 0.0,
                    'reasons': ['Failed to calculate technical indicators'],
                    'raw_data': {'error': 'Indicator calculation failed'}
                }
            
            # Analyze multi-timeframe consensus
            signal = await self._analyze_multi_timeframe_consensus(coin, indicator_results)
            
            # Create detailed completion log
            active_timeframes = list(indicator_results.keys())
            bias = signal.get('bias', 'neutral')
            confidence = signal.get('confidence', 0) * 100
            
            logger.info(f"✅ Technical analysis complete for {coin} - Timeframes: {', '.join(active_timeframes)} | Result: {bias.upper()} ({confidence:.1f}%)")
            return signal
            
        except Exception as e:
            logger.error(f"❌ Technical analysis failed for {coin}: {str(e)}")
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'reasons': [f'Technical analysis error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }
    
    async def _get_ohlcv_data(self, coin: str, timeframe: str, limit: int = 100) -> Optional[List[List[float]]]:
        """Fetch OHLCV data from Binance"""
        
        # Convert timeframe format
        binance_timeframe = self._convert_timeframe(timeframe)
        symbol = f"{coin}USDT"
        
        async with HTTPClient() as client:
            try:
                url = f"{self.data_source}/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': binance_timeframe,
                    'limit': limit
                }
                
                response = await client.get(url, params=params)
                
                # Convert to OHLCV format: [timestamp, open, high, low, close, volume]
                ohlcv_data = []
                for candle in response:
                    ohlcv_data.append([
                        int(candle[0]),      # timestamp
                        float(candle[1]),    # open
                        float(candle[2]),    # high
                        float(candle[3]),    # low
                        float(candle[4]),    # close
                        float(candle[5])     # volume
                    ])
                
                return ohlcv_data
                
            except Exception as e:
                logger.error(f"❌ Error fetching OHLCV data: {str(e)}")
                return None
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Binance format"""
        tf_map = {
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '15m': '15m',
            '1w': '1w'
        }
        return tf_map.get(timeframe, '1h')
    
    async def _calculate_all_indicators(self, coin: str, ohlcv_data: List[List[float]], timeframe: str) -> Optional[Dict[str, Any]]:
        """Calculate all technical indicators for given OHLCV data"""
        
        if len(ohlcv_data) < 50:  # Need minimum data for indicators
            logger.warning(f"⚠️ Insufficient data for {coin} {timeframe}: {len(ohlcv_data)} candles")
            return None
        
        try:
            # Extract price arrays
            timestamps = [candle[0] for candle in ohlcv_data]
            opens = np.array([candle[1] for candle in ohlcv_data])
            highs = np.array([candle[2] for candle in ohlcv_data])
            lows = np.array([candle[3] for candle in ohlcv_data])
            closes = np.array([candle[4] for candle in ohlcv_data])
            volumes = np.array([candle[5] for candle in ohlcv_data])
            
            # Calculate all indicators
            indicators = {
                'timeframe': timeframe,
                'current_price': closes[-1],
                'price_change_pct': ((closes[-1] - closes[-2]) / closes[-2]) * 100,
            }
            
            # RSI (14 period)
            rsi = self._calculate_rsi(closes, period=14)
            indicators['rsi'] = rsi[-1] if len(rsi) > 0 else 50
            
            # MACD (12, 26, 9)
            macd_line, signal_line, histogram = self._calculate_macd(closes)
            indicators['macd'] = {
                'macd': macd_line[-1] if len(macd_line) > 0 else 0,
                'signal': signal_line[-1] if len(signal_line) > 0 else 0,
                'histogram': histogram[-1] if len(histogram) > 0 else 0
            }
            
            # EMA crossovers (12, 26)
            ema_12 = self._calculate_ema(closes, period=12)
            ema_26 = self._calculate_ema(closes, period=26)
            indicators['ema'] = {
                'ema_12': ema_12[-1] if len(ema_12) > 0 else closes[-1],
                'ema_26': ema_26[-1] if len(ema_26) > 0 else closes[-1],
                'crossover': (ema_12[-1] - ema_26[-1]) / closes[-1] * 100 if len(ema_12) > 0 and len(ema_26) > 0 else 0
            }
            
            # Bollinger Bands (20, 2)
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes, period=20, std_dev=2)
            if len(bb_upper) > 0:
                bb_position = (closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                bb_bandwidth = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
                indicators['bollinger'] = {
                    'upper': bb_upper[-1],
                    'middle': bb_middle[-1],
                    'lower': bb_lower[-1],
                    'position': bb_position,  # 0 = at lower band, 1 = at upper band
                    'bandwidth': bb_bandwidth,
                    'squeeze': bb_bandwidth < self.default_thresholds['bb_squeeze_threshold']
                }
            
            # Volume analysis
            avg_volume = np.mean(volumes[-20:])  # 20-period average
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            indicators['volume'] = {
                'current': current_volume,
                'average': avg_volume,
                'ratio': volume_ratio,
                'spike': volume_ratio > self.default_thresholds['volume_spike_threshold']
            }
            
            # Trend strength (ADX approximation using price momentum)
            price_momentum = self._calculate_price_momentum(closes, period=14)
            indicators['trend_strength'] = price_momentum
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators for {coin} {timeframe}: {str(e)}")
            return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return np.array([])
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros_like(gains)
        avg_losses = np.zeros_like(losses)
        
        # Initial averages
        avg_gains[period-1] = np.mean(gains[:period])
        avg_losses[period-1] = np.mean(losses[:period])
        
        # Smoothed averages
        for i in range(period, len(gains)):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
        
        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi[period-1:]
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD line, signal line, and histogram"""
        if len(prices) < slow + signal:
            return np.array([]), np.array([]), np.array([])
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        # Align arrays
        min_len = min(len(ema_fast), len(ema_slow))
        macd_line = ema_fast[-min_len:] - ema_slow[-min_len:]
        
        # Signal line (EMA of MACD)
        signal_line = self._calculate_ema(macd_line, signal)
        
        # Histogram (MACD - Signal)
        min_len = min(len(macd_line), len(signal_line))
        histogram = macd_line[-min_len:] - signal_line[-min_len:]
        
        return macd_line, signal_line, histogram
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.array([])
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return np.array([]), np.array([]), np.array([])
        
        # Simple moving average
        sma = np.convolve(prices, np.ones(period)/period, mode='valid')
        
        # Standard deviation
        std = np.array([np.std(prices[i:i+period]) for i in range(len(prices)-period+1)])
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def _calculate_price_momentum(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate price momentum/trend strength"""
        if len(prices) < period:
            return 0.0
        
        # Calculate rate of change over period
        roc = (prices[-1] - prices[-period]) / prices[-period] * 100
        
        # Normalize to 0-1 range (assuming ±20% is extreme)
        normalized_momentum = max(-1, min(1, roc / 20))
        
        return abs(normalized_momentum)  # Return absolute strength
    
    async def _analyze_multi_timeframe_consensus(self, coin: str, indicator_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze consensus across multiple timeframes"""
        
        try:
            # Weight timeframes by importance
            timeframe_weights = {
                '1h': 0.2,   # Short-term
                '4h': 0.5,   # Medium-term (most important)
                '1d': 0.3    # Long-term
            }
            
            # Analyze each indicator type across timeframes
            rsi_signals = []
            macd_signals = []
            ema_signals = []
            bb_signals = []
            volume_signals = []
            
            reasons = []
            
            for tf, indicators in indicator_results.items():
                weight = timeframe_weights.get(tf, 0.3)
                
                # RSI analysis
                rsi_signal = self._analyze_rsi_signal(indicators, tf)
                rsi_signals.append((rsi_signal, weight))
                
                # MACD analysis
                macd_signal = self._analyze_macd_signal(indicators, tf)
                macd_signals.append((macd_signal, weight))
                
                # EMA crossover analysis
                ema_signal = self._analyze_ema_signal(indicators, tf)
                ema_signals.append((ema_signal, weight))
                
                # Bollinger Bands analysis
                bb_signal = self._analyze_bollinger_signal(indicators, tf)
                bb_signals.append((bb_signal, weight))
                
                # Volume analysis
                volume_signal = self._analyze_volume_signal(indicators, tf)
                volume_signals.append((volume_signal, weight))
            
            # Calculate weighted consensus for each indicator
            rsi_consensus = self._calculate_weighted_consensus(rsi_signals)
            macd_consensus = self._calculate_weighted_consensus(macd_signals)
            ema_consensus = self._calculate_weighted_consensus(ema_signals)
            bb_consensus = self._calculate_weighted_consensus(bb_signals)
            volume_consensus = self._calculate_weighted_consensus(volume_signals)
            
            # Combine all signals with equal weight
            all_signals = [rsi_consensus, macd_consensus, ema_consensus, bb_consensus, volume_consensus]
            valid_signals = [s for s in all_signals if s['confidence'] > 0]
            
            if not valid_signals:
                return {
                    'bias': 'neutral',
                    'confidence': 0.0,
                    'reasons': ['No valid technical signals generated'],
                    'raw_data': {'error': 'No valid signals'}
                }
            
            # Calculate final consensus
            bullish_votes = sum(1 for s in valid_signals if s['bias'] == 'bullish')
            bearish_votes = sum(1 for s in valid_signals if s['bias'] == 'bearish')
            neutral_votes = sum(1 for s in valid_signals if s['bias'] == 'neutral')
            
            total_votes = len(valid_signals)
            avg_confidence = sum(s['confidence'] for s in valid_signals) / total_votes
            
            # Determine final bias
            if bullish_votes > bearish_votes and bullish_votes > neutral_votes:
                final_bias = 'bullish'
                consensus_strength = bullish_votes / total_votes
            elif bearish_votes > bullish_votes and bearish_votes > neutral_votes:
                final_bias = 'bearish'
                consensus_strength = bearish_votes / total_votes
            else:
                final_bias = 'neutral'
                consensus_strength = 0.5
            
            # Adjust confidence based on consensus strength
            final_confidence = avg_confidence * consensus_strength
            
            # Collect reasons from individual signals
            for signal in valid_signals:
                if signal['bias'] == final_bias:
                    reasons.extend(signal.get('reasons', []))
            
            if not reasons:
                reasons = [f'{bullish_votes}B/{bearish_votes}B/{neutral_votes}N technical consensus']
            
            return {
                'bias': final_bias,
                'confidence': min(1.0, final_confidence),
                'reasons': reasons[:3],  # Top 3 reasons
                'raw_data': {
                    'rsi_consensus': rsi_consensus,
                    'macd_consensus': macd_consensus,
                    'ema_consensus': ema_consensus,
                    'bollinger_consensus': bb_consensus,
                    'volume_consensus': volume_consensus,
                    'votes': {'bullish': bullish_votes, 'bearish': bearish_votes, 'neutral': neutral_votes},
                    'timeframes_analyzed': list(indicator_results.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing multi-timeframe consensus: {str(e)}")
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'reasons': [f'Consensus analysis error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }
    
    def _analyze_rsi_signal(self, indicators: Dict, timeframe: str) -> Dict[str, Any]:
        """Analyze RSI signal for single timeframe"""
        rsi = indicators.get('rsi', 50)
        
        if rsi <= self.default_thresholds['rsi_strong_oversold']:
            return {'bias': 'bullish', 'confidence': 0.8, 'reasons': [f'RSI oversold ({rsi:.1f}) on {timeframe}']}
        elif rsi <= self.default_thresholds['rsi_oversold']:
            return {'bias': 'bullish', 'confidence': 0.5, 'reasons': [f'RSI approaching oversold ({rsi:.1f}) on {timeframe}']}
        elif rsi >= self.default_thresholds['rsi_strong_overbought']:
            return {'bias': 'bearish', 'confidence': 0.8, 'reasons': [f'RSI overbought ({rsi:.1f}) on {timeframe}']}
        elif rsi >= self.default_thresholds['rsi_overbought']:
            return {'bias': 'bearish', 'confidence': 0.5, 'reasons': [f'RSI approaching overbought ({rsi:.1f}) on {timeframe}']}
        else:
            return {'bias': 'neutral', 'confidence': 0.2, 'reasons': [f'RSI neutral ({rsi:.1f}) on {timeframe}']}
    
    def _analyze_macd_signal(self, indicators: Dict, timeframe: str) -> Dict[str, Any]:
        """Analyze MACD signal for single timeframe"""
        macd_data = indicators.get('macd', {})
        macd = macd_data.get('macd', 0)
        signal = macd_data.get('signal', 0)
        histogram = macd_data.get('histogram', 0)
        
        # MACD crossover signals
        if macd > signal and histogram > 0:
            confidence = min(0.7, abs(histogram) * 1000)  # Scale histogram
            return {'bias': 'bullish', 'confidence': confidence, 'reasons': [f'MACD bullish crossover on {timeframe}']}
        elif macd < signal and histogram < 0:
            confidence = min(0.7, abs(histogram) * 1000)
            return {'bias': 'bearish', 'confidence': confidence, 'reasons': [f'MACD bearish crossover on {timeframe}']}
        else:
            return {'bias': 'neutral', 'confidence': 0.1, 'reasons': [f'MACD neutral on {timeframe}']}
    
    def _analyze_ema_signal(self, indicators: Dict, timeframe: str) -> Dict[str, Any]:
        """Analyze EMA crossover signal for single timeframe"""
        ema_data = indicators.get('ema', {})
        crossover = ema_data.get('crossover', 0)
        
        if crossover > 0.5:  # EMA12 > EMA26 by more than 0.5%
            return {'bias': 'bullish', 'confidence': min(0.6, crossover / 2), 'reasons': [f'EMA bullish crossover on {timeframe}']}
        elif crossover < -0.5:  # EMA12 < EMA26 by more than 0.5%
            return {'bias': 'bearish', 'confidence': min(0.6, abs(crossover) / 2), 'reasons': [f'EMA bearish crossover on {timeframe}']}
        else:
            return {'bias': 'neutral', 'confidence': 0.1, 'reasons': [f'EMA neutral on {timeframe}']}
    
    def _analyze_bollinger_signal(self, indicators: Dict, timeframe: str) -> Dict[str, Any]:
        """Analyze Bollinger Bands signal for single timeframe"""
        bb_data = indicators.get('bollinger', {})
        if not bb_data:
            return {'bias': 'neutral', 'confidence': 0.0, 'reasons': []}
        
        position = bb_data.get('position', 0.5)
        squeeze = bb_data.get('squeeze', False)
        
        if squeeze:
            return {'bias': 'neutral', 'confidence': 0.1, 'reasons': [f'BB squeeze on {timeframe} - low volatility']}
        elif position <= 0.1:  # Near lower band
            return {'bias': 'bullish', 'confidence': 0.6, 'reasons': [f'Price at BB lower band on {timeframe}']}
        elif position >= 0.9:  # Near upper band
            return {'bias': 'bearish', 'confidence': 0.6, 'reasons': [f'Price at BB upper band on {timeframe}']}
        else:
            return {'bias': 'neutral', 'confidence': 0.1, 'reasons': [f'Price in BB middle on {timeframe}']}
    
    def _analyze_volume_signal(self, indicators: Dict, timeframe: str) -> Dict[str, Any]:
        """Analyze volume signal for single timeframe"""
        volume_data = indicators.get('volume', {})
        ratio = volume_data.get('ratio', 1.0)
        spike = volume_data.get('spike', False)
        price_change = indicators.get('price_change_pct', 0)
        
        if spike and abs(price_change) > 1:  # Volume spike with significant price movement
            if price_change > 0:
                return {'bias': 'bullish', 'confidence': 0.5, 'reasons': [f'Volume spike with price rise on {timeframe}']}
            else:
                return {'bias': 'bearish', 'confidence': 0.5, 'reasons': [f'Volume spike with price drop on {timeframe}']}
        elif spike:
            return {'bias': 'neutral', 'confidence': 0.2, 'reasons': [f'Volume spike on {timeframe}']}
        else:
            return {'bias': 'neutral', 'confidence': 0.1, 'reasons': [f'Normal volume on {timeframe}']}
    
    def _calculate_weighted_consensus(self, signals: List[Tuple[Dict, float]]) -> Dict[str, Any]:
        """Calculate weighted consensus from multiple signals"""
        if not signals:
            return {'bias': 'neutral', 'confidence': 0.0, 'reasons': []}
        
        weighted_bull = 0
        weighted_bear = 0
        weighted_neutral = 0
        total_weight = 0
        all_reasons = []
        
        for signal, weight in signals:
            bias = signal.get('bias', 'neutral')
            confidence = signal.get('confidence', 0)
            reasons = signal.get('reasons', [])
            
            weighted_confidence = confidence * weight
            total_weight += weight
            
            if bias == 'bullish':
                weighted_bull += weighted_confidence
            elif bias == 'bearish':
                weighted_bear += weighted_confidence
            else:
                weighted_neutral += weighted_confidence
            
            all_reasons.extend(reasons)
        
        if total_weight == 0:
            return {'bias': 'neutral', 'confidence': 0.0, 'reasons': []}
        
        # Normalize
        weighted_bull /= total_weight
        weighted_bear /= total_weight
        weighted_neutral /= total_weight
        
        # Determine final bias
        if weighted_bull > weighted_bear and weighted_bull > weighted_neutral:
            return {'bias': 'bullish', 'confidence': weighted_bull, 'reasons': all_reasons}
        elif weighted_bear > weighted_bull and weighted_bear > weighted_neutral:
            return {'bias': 'bearish', 'confidence': weighted_bear, 'reasons': all_reasons}
        else:
            return {'bias': 'neutral', 'confidence': max(weighted_neutral, 0.1), 'reasons': all_reasons}