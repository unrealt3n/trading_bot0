"""
🏦 Retail vs Big Money Detector
Identifies when big money is moving opposite to retail sentiment
Detects potential retail traps and institutional accumulation/distribution
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class MarketFlow:
    """Market flow data structure"""
    timestamp: datetime
    retail_sentiment: float  # -1 to 1
    institutional_flow: float  # -1 to 1
    volume_profile: Dict[str, float]
    order_flow: Dict[str, float]
    funding_rates: Dict[str, float]

class RetailVsBigMoneyDetector:
    """Detects retail vs institutional money flow patterns"""
    
    def __init__(self):
        self.session = None
        
        # Data sources for retail sentiment
        self.retail_sentiment_sources = {
            'binance_futures': 'https://fapi.binance.com/fapi/v1/takeLongShortRatio',
            'bybit_sentiment': 'https://api.bybit.com/v2/public/account-ratio',
            'coinglass_sentiment': 'https://open-api.coinglass.com/public/v2/long_short_ratio',
            'feargreed_index': 'https://api.alternative.me/fng/',
        }
        
        # Data sources for institutional flow
        self.institutional_sources = {
            'cme_futures': 'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future',
            'grayscale_flows': 'https://grayscale.com/wp-content/uploads/2023/01/Grayscale_Holdings.json',
            'coinbase_premium': 'https://api.coinbase.com/v2/exchange-rates',
            'otc_flows': 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
        }
        
        # Detection thresholds
        self.divergence_thresholds = {
            'strong_divergence': 0.7,    # 70% divergence
            'moderate_divergence': 0.5,  # 50% divergence
            'weak_divergence': 0.3       # 30% divergence
        }
        
        # Pattern recognition weights
        self.pattern_weights = {
            'retail_sentiment': 0.25,
            'institutional_flow': 0.35,
            'volume_profile': 0.20,
            'funding_rates': 0.20
        }
        
        logger.info("🏦 Retail vs Big Money Detector initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def analyze_retail_vs_bigmoney(self, coin: str) -> Dict[str, Any]:
        """
        Analyze retail vs big money flow patterns
        
        Returns:
        {
            'signal_strength': 0-100,
            'bias': 'bullish'/'bearish'/'neutral',
            'confidence': 0.0-1.0,
            'divergence_type': 'strong'/'moderate'/'weak',
            'retail_sentiment': {...},
            'institutional_flow': {...},
            'trap_probability': 0.0-1.0
        }
        """
        
        try:
            logger.info(f"🏦 Analyzing retail vs big money for {coin}")
            
            if not self.session:
                async with self:
                    return await self.analyze_retail_vs_bigmoney(coin)
            
            # Gather data from multiple sources
            retail_data = await self._get_retail_sentiment_data(coin)
            institutional_data = await self._get_institutional_flow_data(coin)
            
            # Analyze divergences
            divergence_analysis = self._detect_divergences(retail_data, institutional_data)
            
            # Generate signal
            signal = self._generate_divergence_signal(coin, divergence_analysis, retail_data, institutional_data)
            
            logger.info(f"🏦 {coin} divergence signal: {signal['bias'].upper()} "
                       f"({signal['signal_strength']:.1f}/100) - {signal['divergence_type']} divergence")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing retail vs big money for {coin}: {e}")
            return self._create_neutral_signal(f"Analysis error: {str(e)}")
    
    async def _get_retail_sentiment_data(self, coin: str) -> Dict[str, Any]:
        """Gather retail sentiment data from multiple sources"""
        
        retail_data = {
            'long_short_ratio': 0.5,
            'fear_greed_index': 50,
            'social_sentiment': 0.0,
            'funding_rates': {},
            'retail_volume_ratio': 0.0,
            'sources': []
        }
        
        try:
            # Get Binance Long/Short Ratio
            binance_data = await self._fetch_binance_long_short_ratio(coin)
            if binance_data:
                retail_data['long_short_ratio'] = binance_data['long_short_ratio']
                retail_data['sources'].append('binance')
            
            # Get Fear & Greed Index
            fng_data = await self._fetch_fear_greed_index()
            if fng_data:
                retail_data['fear_greed_index'] = fng_data['value']
                retail_data['sources'].append('fear_greed')
            
            # Get funding rates (retail behavior indicator)
            funding_data = await self._fetch_funding_rates(coin)
            if funding_data:
                retail_data['funding_rates'] = funding_data
                retail_data['sources'].append('funding_rates')
            
            # Get social sentiment (retail proxy)
            social_data = await self._fetch_social_sentiment(coin)
            if social_data:
                retail_data['social_sentiment'] = social_data['sentiment_score']
                retail_data['sources'].append('social')
        
        except Exception as e:
            logger.error(f"❌ Error fetching retail sentiment data: {e}")
        
        return retail_data
    
    async def _get_institutional_flow_data(self, coin: str) -> Dict[str, Any]:
        """Gather institutional flow data"""
        
        institutional_data = {
            'cme_futures_oi': 0.0,
            'grayscale_flows': 0.0,
            'coinbase_premium': 0.0,
            'otc_volume': 0.0,
            'whale_movements': 0.0,
            'institutional_sentiment': 0.0,
            'sources': []
        }
        
        try:
            # Get CME futures data (institutional proxy)
            cme_data = await self._fetch_cme_futures_data(coin)
            if cme_data:
                institutional_data['cme_futures_oi'] = cme_data['open_interest_change']
                institutional_data['sources'].append('cme')
            
            # Get Coinbase premium (institutional buying)
            coinbase_data = await self._fetch_coinbase_premium(coin)
            if coinbase_data:
                institutional_data['coinbase_premium'] = coinbase_data['premium']
                institutional_data['sources'].append('coinbase')
            
            # Get OTC volume estimates
            otc_data = await self._estimate_otc_volume(coin)
            if otc_data:
                institutional_data['otc_volume'] = otc_data['volume_ratio']
                institutional_data['sources'].append('otc')
            
            # Get whale movement data (from our whale analyzer)
            whale_data = await self._get_whale_movement_summary(coin)
            if whale_data:
                institutional_data['whale_movements'] = whale_data['net_flow']
                institutional_data['sources'].append('whale')
        
        except Exception as e:
            logger.error(f"❌ Error fetching institutional flow data: {e}")
        
        return institutional_data
    
    async def _fetch_binance_long_short_ratio(self, coin: str) -> Optional[Dict[str, Any]]:
        """Fetch Binance long/short ratio"""
        
        try:
            url = "https://fapi.binance.com/fapi/v1/takeLongShortRatio"
            params = {
                'symbol': f"{coin}USDT",
                'period': '5m',
                'limit': 1
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        latest = data[0]
                        long_ratio = float(latest['takeLongShortRatio'])
                        return {
                            'long_short_ratio': long_ratio,
                            'timestamp': datetime.fromtimestamp(int(latest['timestamp']) / 1000)
                        }
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch Binance long/short ratio: {e}")
        
        return None
    
    async def _fetch_fear_greed_index(self) -> Optional[Dict[str, Any]]:
        """Fetch Fear & Greed Index"""
        
        try:
            url = "https://api.alternative.me/fng/"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and 'data' in data:
                        latest = data['data'][0]
                        return {
                            'value': int(latest['value']),
                            'classification': latest['value_classification']
                        }
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch Fear & Greed Index: {e}")
        
        return None
    
    async def _fetch_funding_rates(self, coin: str) -> Optional[Dict[str, float]]:
        """Fetch funding rates from multiple exchanges"""
        
        funding_rates = {}
        
        try:
            # Binance funding rate
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            params = {'symbol': f"{coin}USDT"}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list) and data:
                        funding_rates['binance'] = float(data[0]['lastFundingRate'])
                    elif isinstance(data, dict):
                        funding_rates['binance'] = float(data['lastFundingRate'])
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch funding rates: {e}")
        
        return funding_rates if funding_rates else None
    
    async def _fetch_social_sentiment(self, coin: str) -> Optional[Dict[str, Any]]:
        """Fetch social sentiment data"""
        
        try:
            # This could integrate with social sentiment APIs
            # For now, return placeholder data
            return {
                'sentiment_score': 0.0,  # -1 to 1
                'mention_count': 0,
                'trending_score': 0.0
            }
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch social sentiment: {e}")
        
        return None
    
    async def _fetch_cme_futures_data(self, coin: str) -> Optional[Dict[str, Any]]:
        """Fetch CME futures data"""
        
        try:
            # CME futures data (requires parsing from CME website or API)
            # This is a placeholder implementation
            return {
                'open_interest_change': 0.0,
                'volume_change': 0.0,
                'net_positions': 0.0
            }
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch CME futures data: {e}")
        
        return None
    
    async def _fetch_coinbase_premium(self, coin: str) -> Optional[Dict[str, Any]]:
        """Fetch Coinbase premium (institutional buying indicator)"""
        
        try:
            # Get Coinbase and Binance prices to calculate premium
            coinbase_url = "https://api.coinbase.com/v2/exchange-rates"
            
            async with self.session.get(coinbase_url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data and 'rates' in data['data']:
                        # Calculate premium vs other exchanges
                        # This is simplified - real implementation would compare multiple exchanges
                        return {
                            'premium': 0.0,  # Percentage premium
                            'institutional_buying': False
                        }
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch Coinbase premium: {e}")
        
        return None
    
    async def _estimate_otc_volume(self, coin: str) -> Optional[Dict[str, Any]]:
        """Estimate OTC trading volume (institutional indicator)"""
        
        try:
            # OTC volume estimation based on market data analysis
            # This is complex and would require multiple data sources
            return {
                'volume_ratio': 0.0,  # OTC volume as ratio of total volume
                'estimated_otc_volume': 0.0
            }
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to estimate OTC volume: {e}")
        
        return None
    
    async def _get_whale_movement_summary(self, coin: str) -> Optional[Dict[str, Any]]:
        """Get summary of whale movements from whale analyzer"""
        
        try:
            # This would integrate with our whale analyzer
            # For now, return placeholder data
            return {
                'net_flow': 0.0,  # Net flow in/out of exchanges
                'whale_activity_score': 0.0
            }
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to get whale movement summary: {e}")
        
        return None
    
    def _detect_divergences(self, retail_data: Dict, institutional_data: Dict) -> Dict[str, Any]:
        """Detect divergences between retail and institutional behavior"""
        
        # Calculate retail sentiment score (-1 to 1)
        retail_score = self._calculate_retail_sentiment_score(retail_data)
        
        # Calculate institutional flow score (-1 to 1)
        institutional_score = self._calculate_institutional_flow_score(institutional_data)
        
        # Calculate divergence
        divergence = abs(retail_score - institutional_score)
        
        # Determine divergence type
        if divergence >= self.divergence_thresholds['strong_divergence']:
            divergence_type = 'strong'
        elif divergence >= self.divergence_thresholds['moderate_divergence']:
            divergence_type = 'moderate'
        elif divergence >= self.divergence_thresholds['weak_divergence']:
            divergence_type = 'weak'
        else:
            divergence_type = 'none'
        
        # Determine trap probability
        trap_probability = self._calculate_trap_probability(retail_score, institutional_score, divergence)
        
        return {
            'retail_sentiment_score': retail_score,
            'institutional_flow_score': institutional_score,
            'divergence': divergence,
            'divergence_type': divergence_type,
            'trap_probability': trap_probability,
            'retail_bullish': retail_score > 0,
            'institutional_bullish': institutional_score > 0,
            'follow_smart_money': institutional_score > retail_score
        }
    
    def _calculate_retail_sentiment_score(self, retail_data: Dict) -> float:
        """Calculate overall retail sentiment score"""
        
        score = 0.0
        weight_sum = 0.0
        
        # Long/Short ratio contribution
        if 'long_short_ratio' in retail_data:
            ls_ratio = retail_data['long_short_ratio']
            # Convert ratio to sentiment (-1 to 1)
            ls_sentiment = (ls_ratio - 0.5) * 2  # 0.5 = neutral, >0.5 = bullish, <0.5 = bearish
            score += ls_sentiment * 0.4
            weight_sum += 0.4
        
        # Fear & Greed Index contribution
        if 'fear_greed_index' in retail_data:
            fng = retail_data['fear_greed_index']
            # Convert to -1 to 1 scale
            fng_sentiment = (fng - 50) / 50  # 50 = neutral
            score += fng_sentiment * 0.3
            weight_sum += 0.3
        
        # Funding rates contribution (high funding = retail greed)
        if 'funding_rates' in retail_data and retail_data['funding_rates']:
            avg_funding = sum(retail_data['funding_rates'].values()) / len(retail_data['funding_rates'])
            # Normalize funding rate to sentiment
            funding_sentiment = max(-1, min(1, avg_funding * 1000))  # Scale funding rate
            score += funding_sentiment * 0.3
            weight_sum += 0.3
        
        # Normalize by weight sum
        return score / max(weight_sum, 0.1)
    
    def _calculate_institutional_flow_score(self, institutional_data: Dict) -> float:
        """Calculate institutional flow score"""
        
        score = 0.0
        weight_sum = 0.0
        
        # CME futures OI change
        if 'cme_futures_oi' in institutional_data:
            cme_score = max(-1, min(1, institutional_data['cme_futures_oi'] / 100))
            score += cme_score * 0.3
            weight_sum += 0.3
        
        # Coinbase premium
        if 'coinbase_premium' in institutional_data:
            premium_score = max(-1, min(1, institutional_data['coinbase_premium'] / 5))
            score += premium_score * 0.25
            weight_sum += 0.25
        
        # Whale movements
        if 'whale_movements' in institutional_data:
            whale_score = max(-1, min(1, institutional_data['whale_movements'] / 1000000))
            score += whale_score * 0.25
            weight_sum += 0.25
        
        # OTC volume
        if 'otc_volume' in institutional_data:
            otc_score = max(-1, min(1, institutional_data['otc_volume'] - 0.5))
            score += otc_score * 0.2
            weight_sum += 0.2
        
        # Normalize by weight sum
        return score / max(weight_sum, 0.1)
    
    def _calculate_trap_probability(self, retail_score: float, institutional_score: float, divergence: float) -> float:
        """Calculate probability of retail trap"""
        
        # High trap probability when:
        # 1. Strong divergence
        # 2. Retail very bullish/bearish but institutions opposite
        # 3. Extreme retail sentiment
        
        trap_probability = 0.0
        
        # Divergence component
        trap_probability += divergence * 0.4
        
        # Retail extremes component
        retail_extreme = abs(retail_score)
        trap_probability += retail_extreme * 0.3
        
        # Opposite direction component
        if (retail_score > 0 and institutional_score < 0) or (retail_score < 0 and institutional_score > 0):
            trap_probability += 0.3
        
        return min(trap_probability, 1.0)
    
    def _generate_divergence_signal(self, coin: str, divergence_analysis: Dict, 
                                   retail_data: Dict, institutional_data: Dict) -> Dict[str, Any]:
        """Generate trading signal from divergence analysis"""
        
        # Base signal strength from divergence
        divergence_strength = divergence_analysis['divergence'] * 100
        
        # Direction: follow smart money (institutions)
        institutional_bullish = divergence_analysis['institutional_bullish']
        if institutional_bullish:
            bias = 'bullish'
        else:
            bias = 'bearish'
        
        # Adjust signal strength based on divergence type
        if divergence_analysis['divergence_type'] == 'strong':
            signal_strength = min(divergence_strength * 1.2, 100)
        elif divergence_analysis['divergence_type'] == 'moderate':
            signal_strength = divergence_strength
        else:
            signal_strength = divergence_strength * 0.8
        
        # Calculate confidence
        confidence = signal_strength / 100.0
        
        # Adjust for trap probability
        trap_probability = divergence_analysis['trap_probability']
        if trap_probability > 0.7:
            confidence *= 1.3  # High confidence when retail likely trapped
        
        # Generate summary
        summary = {
            'retail_sentiment': 'bullish' if divergence_analysis['retail_bullish'] else 'bearish',
            'institutional_flow': 'bullish' if divergence_analysis['institutional_bullish'] else 'bearish',
            'divergence_strength': divergence_analysis['divergence'],
            'trap_probability': trap_probability,
            'follow_smart_money': divergence_analysis['follow_smart_money'],
            'retail_data_sources': retail_data.get('sources', []),
            'institutional_data_sources': institutional_data.get('sources', [])
        }
        
        return {
            'signal_strength': signal_strength,
            'bias': bias,
            'confidence': confidence,
            'divergence_type': divergence_analysis['divergence_type'],
            'retail_sentiment': retail_data,
            'institutional_flow': institutional_data,
            'divergence_analysis': divergence_analysis,
            'trap_probability': trap_probability,
            'summary': summary,
            'reasons': self._generate_divergence_reasons(divergence_analysis, summary)
        }
    
    def _generate_divergence_reasons(self, divergence_analysis: Dict, summary: Dict) -> List[str]:
        """Generate human-readable reasons for divergence signal"""
        
        reasons = []
        
        # Divergence strength
        divergence_type = divergence_analysis['divergence_type']
        if divergence_type != 'none':
            reasons.append(f"{divergence_type.capitalize()} retail-institutional divergence")
        
        # Direction conflict
        if summary['retail_sentiment'] != summary['institutional_flow']:
            reasons.append(f"Retail {summary['retail_sentiment']} vs institutions {summary['institutional_flow']}")
        
        # Trap probability
        if summary['trap_probability'] > 0.6:
            reasons.append(f"High retail trap probability ({summary['trap_probability']:.1%})")
        
        # Smart money direction
        if summary['follow_smart_money']:
            reasons.append("Following smart money direction")
        
        return reasons[:3]
    
    def _create_neutral_signal(self, reason: str) -> Dict[str, Any]:
        """Create neutral signal when analysis fails"""
        
        return {
            'signal_strength': 0.0,
            'bias': 'neutral',
            'confidence': 0.0,
            'divergence_type': 'none',
            'retail_sentiment': {},
            'institutional_flow': {},
            'trap_probability': 0.0,
            'summary': {'reason': reason},
            'reasons': [reason]
        }