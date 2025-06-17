"""
💥 Liquidation Analyzer  
Tracks crypto liquidations across major exchanges using CoinGlass and other sources
Provides momentum and reversal signals based on liquidation patterns
"""

import aiohttp
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

@dataclass
class LiquidationEvent:
    """Liquidation event data structure"""
    timestamp: datetime
    symbol: str
    exchange: str
    side: str  # 'long' or 'short'
    amount_usd: float
    price: float
    leverage: Optional[int] = None

class LiquidationAnalyzer:
    """Analyzes liquidation data for trading signals"""
    
    def __init__(self):
        self.session = None
        
        # CoinGlass API endpoints (some may be free)
        self.coinglass_endpoints = {
            'liquidations': 'https://open-api.coinglass.com/public/v2/liquidation',
            'heatmap': 'https://www.coinglass.com/LiquidationData',
            'stats': 'https://open-api.coinglass.com/public/v2/liquidation_chart'
        }
        
        # Alternative data sources
        self.alternative_sources = [
            'https://www.bybt.com/LiquidationData',
            'https://liquidations.xyz/',
            'https://debank.com/stream'
        ]
        
        # Liquidation thresholds for signal generation
        self.liquidation_thresholds = {
            'small': 100_000,      # $100K
            'medium': 1_000_000,   # $1M  
            'large': 10_000_000,   # $10M
            'massive': 50_000_000  # $50M
        }
        
        # Signal strength multipliers
        self.signal_multipliers = {
            'long_liquidations': -0.8,   # Bearish continuation
            'short_liquidations': 0.8,   # Bullish continuation  
            'balanced': 0.2,             # Neutral
            'cascade': 1.5               # Amplifier for cascading liquidations
        }
        
        # Cache for recent data
        self.liquidation_cache: Dict[str, List[LiquidationEvent]] = {}
        self.cache_duration = timedelta(minutes=5)
        
        logger.info("💥 Liquidation Analyzer initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def analyze_liquidations(self, coin: str, lookback_minutes: int = 60) -> Dict[str, Any]:
        """
        Analyze recent liquidations for trading signals
        
        Returns:
        {
            'signal_strength': 0-100,
            'bias': 'bullish'/'bearish'/'neutral',
            'confidence': 0.0-1.0,
            'liquidations': [...],
            'summary': {...}
        }
        """
        
        try:
            logger.info(f"💥 Analyzing liquidations for {coin} (last {lookback_minutes}min)")
            
            if not self.session:
                async with self:
                    return await self.analyze_liquidations(coin, lookback_minutes)
            
            # Get recent liquidation data
            liquidations = await self.get_recent_liquidations(coin, lookback_minutes)
            
            if not liquidations:
                return self._create_neutral_signal("No recent liquidations")
            
            # Calculate liquidation metrics
            metrics = self._calculate_liquidation_metrics(liquidations)
            
            # Generate trading signal
            signal = self._generate_liquidation_signal(coin, metrics, liquidations)
            
            logger.info(f"💥 {coin} liquidation signal: {signal['bias'].upper()} "
                       f"({signal['signal_strength']:.1f}/100)")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing liquidations for {coin}: {e}")
            return self._create_neutral_signal(f"Analysis error: {str(e)}")
    
    async def get_recent_liquidations(self, coin: str, lookback_minutes: int) -> List[LiquidationEvent]:
        """Get recent liquidation data from multiple sources"""
        
        # Check cache first
        cache_key = f"{coin}_{lookback_minutes}"
        if cache_key in self.liquidation_cache:
            cached_data = self.liquidation_cache[cache_key]
            if cached_data and datetime.now() - cached_data[0].timestamp < self.cache_duration:
                return cached_data
        
        liquidations = []
        
        # Try multiple data sources
        try:
            # Try CoinGlass API first
            coinglass_data = await self._fetch_coinglass_data(coin, lookback_minutes)
            liquidations.extend(coinglass_data)
            
            # Try alternative sources if needed
            if len(liquidations) < 5:  # If we don't have enough data
                alt_data = await self._fetch_alternative_sources(coin, lookback_minutes)
                liquidations.extend(alt_data)
            
        except Exception as e:
            logger.error(f"❌ Error fetching liquidation data: {e}")
        
        # Cache results
        if liquidations:
            self.liquidation_cache[cache_key] = liquidations
        
        return liquidations
    
    async def _fetch_coinglass_data(self, coin: str, lookback_minutes: int) -> List[LiquidationEvent]:
        """Fetch liquidation data from CoinGlass"""
        
        liquidations = []
        
        try:
            # Try public API endpoints
            api_url = "https://open-api.coinglass.com/public/v2/liquidation"
            params = {
                'symbol': f"{coin}USDT",
                'time_type': '5m',  # 5-minute intervals
                'limit': 50
            }
            
            async with self.session.get(api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    liquidations.extend(self._parse_coinglass_api_response(data, coin))
                else:
                    logger.warning(f"⚠️ CoinGlass API returned {response.status}")
        
        except Exception as e:
            logger.warning(f"⚠️ CoinGlass API failed: {e}")
        
        # If API fails, try scraping the website
        if not liquidations:
            try:
                liquidations.extend(await self._scrape_coinglass_website(coin))
            except Exception as e:
                logger.warning(f"⚠️ CoinGlass scraping failed: {e}")
        
        return liquidations
    
    async def _scrape_coinglass_website(self, coin: str) -> List[LiquidationEvent]:
        """Scrape liquidation data from CoinGlass website"""
        
        liquidations = []
        
        try:
            # CoinGlass liquidation page
            url = f"https://www.coinglass.com/LiquidationData"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for liquidation data in the page
                    # This would need to be adapted based on CoinGlass's actual HTML structure
                    liquidation_elements = soup.find_all('div', class_=re.compile(r'liquidation|liq'))
                    
                    for element in liquidation_elements:
                        try:
                            # Extract liquidation data from HTML
                            # This is a simplified example - real implementation would need
                            # to match CoinGlass's actual HTML structure
                            liquidation = self._parse_liquidation_html_element(element, coin)
                            if liquidation:
                                liquidations.append(liquidation)
                        except Exception as e:
                            continue
        
        except Exception as e:
            logger.error(f"❌ Error scraping CoinGlass: {e}")
        
        return liquidations
    
    async def _fetch_alternative_sources(self, coin: str, lookback_minutes: int) -> List[LiquidationEvent]:
        """Fetch liquidation data from alternative sources"""
        
        liquidations = []
        
        for source_url in self.alternative_sources:
            try:
                async with self.session.get(source_url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Parse liquidation data from different sources
                        if 'bybt.com' in source_url:
                            liquidations.extend(self._parse_bybt_data(html, coin))
                        elif 'liquidations.xyz' in source_url:
                            liquidations.extend(self._parse_liquidations_xyz_data(html, coin))
                        elif 'debank.com' in source_url:  
                            liquidations.extend(self._parse_debank_data(html, coin))
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch from {source_url}: {e}")
                continue
        
        return liquidations
    
    def _parse_coinglass_api_response(self, data: Dict, coin: str) -> List[LiquidationEvent]:
        """Parse CoinGlass API response"""
        
        liquidations = []
        
        try:
            # Parse based on actual CoinGlass API structure
            if 'data' in data:
                for item in data['data']:
                    liquidation = LiquidationEvent(
                        timestamp=datetime.fromtimestamp(item.get('time', 0) / 1000),
                        symbol=coin,
                        exchange=item.get('exchange', 'unknown'),
                        side=item.get('side', 'unknown').lower(),
                        amount_usd=float(item.get('usd', 0)),
                        price=float(item.get('price', 0))
                    )
                    liquidations.append(liquidation)
        
        except Exception as e:
            logger.error(f"❌ Error parsing CoinGlass API response: {e}")
        
        return liquidations
    
    def _parse_liquidation_html_element(self, element, coin: str) -> Optional[LiquidationEvent]:
        """Parse individual liquidation HTML element"""
        
        try:
            # This would need to be customized based on actual HTML structure
            # Looking for patterns like:
            # - "$1.2M LONG liquidated"
            # - "BTC/USDT Short $500K"
            # - Timestamps, exchanges, etc.
            
            text = element.get_text(strip=True)
            
            # Look for liquidation amounts
            amount_match = re.search(r'\$?([\d.]+)([KMB]?)', text)
            if amount_match:
                amount = float(amount_match.group(1))
                unit = amount_match.group(2)
                
                # Convert to USD
                multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
                amount_usd = amount * multipliers.get(unit, 1)
                
                # Determine side
                side = 'long' if 'long' in text.lower() else 'short' if 'short' in text.lower() else 'unknown'
                
                return LiquidationEvent(
                    timestamp=datetime.now(),  # Would need to parse actual timestamp
                    symbol=coin,
                    exchange='coinglass',
                    side=side,
                    amount_usd=amount_usd,
                    price=0.0  # Would need to extract if available
                )
        
        except Exception as e:
            pass
        
        return None
    
    def _parse_bybt_data(self, html: str, coin: str) -> List[LiquidationEvent]:
        """Parse BYBT liquidation data"""
        
        liquidations = []
        
        try:
            # Parse BYBT HTML structure
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for liquidation tables or data containers
            # This would need to match BYBT's actual structure
            liquidation_rows = soup.find_all('tr', class_=re.compile(r'liquidation|liq'))
            
            for row in liquidation_rows:
                try:
                    cells = row.find_all('td')
                    if len(cells) >= 4:  # Assuming minimum required columns
                        liquidation = LiquidationEvent(
                            timestamp=datetime.now(),  # Parse from HTML
                            symbol=coin,
                            exchange='bybt',
                            side='unknown',
                            amount_usd=0.0,  # Parse from HTML
                            price=0.0
                        )
                        liquidations.append(liquidation)
                except Exception:
                    continue
        
        except Exception as e:
            logger.error(f"❌ Error parsing BYBT data: {e}")
        
        return liquidations
    
    def _parse_liquidations_xyz_data(self, html: str, coin: str) -> List[LiquidationEvent]:
        """Parse liquidations.xyz data"""
        
        # Similar implementation to BYBT
        return []
    
    def _parse_debank_data(self, html: str, coin: str) -> List[LiquidationEvent]:
        """Parse DeBank liquidation data"""
        
        # Similar implementation to BYBT
        return []
    
    def _calculate_liquidation_metrics(self, liquidations: List[LiquidationEvent]) -> Dict[str, Any]:
        """Calculate liquidation metrics for signal generation"""
        
        total_long_liq = 0.0
        total_short_liq = 0.0
        total_volume = 0.0
        exchange_distribution = {}
        
        # Process each liquidation
        for liq in liquidations:
            total_volume += liq.amount_usd
            
            if liq.side == 'long':
                total_long_liq += liq.amount_usd
            elif liq.side == 'short':
                total_short_liq += liq.amount_usd
            
            # Track exchange distribution
            exchange = liq.exchange
            if exchange not in exchange_distribution:
                exchange_distribution[exchange] = 0.0
            exchange_distribution[exchange] += liq.amount_usd
        
        # Calculate ratios
        if total_volume > 0:
            long_liq_ratio = total_long_liq / total_volume
            short_liq_ratio = total_short_liq / total_volume
        else:
            long_liq_ratio = 0.0
            short_liq_ratio = 0.0
        
        # Calculate liquidation momentum (recent vs older)
        recent_threshold = datetime.now() - timedelta(minutes=15)
        recent_liquidations = [liq for liq in liquidations if liq.timestamp > recent_threshold]
        recent_volume = sum(liq.amount_usd for liq in recent_liquidations)
        
        momentum = recent_volume / max(total_volume, 1)
        
        return {
            'total_long_liquidations': total_long_liq,
            'total_short_liquidations': total_short_liq,
            'total_volume': total_volume,
            'long_liq_ratio': long_liq_ratio,
            'short_liq_ratio': short_liq_ratio,
            'liquidation_count': len(liquidations),
            'momentum': momentum,
            'exchange_distribution': exchange_distribution,
            'avg_liquidation_size': total_volume / max(len(liquidations), 1)
        }
    
    def _generate_liquidation_signal(self, coin: str, metrics: Dict, liquidations: List[LiquidationEvent]) -> Dict[str, Any]:
        """Generate trading signal from liquidation metrics"""
        
        # Determine signal direction
        long_liq_ratio = metrics['long_liq_ratio']
        short_liq_ratio = metrics['short_liq_ratio']
        
        # More long liquidations = bearish continuation
        # More short liquidations = bullish continuation
        if long_liq_ratio > 0.7:
            bias = 'bearish'
            direction_score = long_liq_ratio * 60
        elif short_liq_ratio > 0.7:
            bias = 'bullish'  
            direction_score = short_liq_ratio * 60
        else:
            bias = 'neutral'
            direction_score = 20
        
        # Volume score (higher volume = stronger signal)
        volume_score = min(metrics['total_volume'] / 5_000_000, 40)  # Cap at $5M = 40 points
        
        # Momentum score (recent activity)
        momentum_score = metrics['momentum'] * 20
        
        # Calculate final signal strength
        signal_strength = min(direction_score + volume_score + momentum_score, 100)
        
        # Calculate confidence
        confidence = signal_strength / 100.0
        
        # Adjust for cascading liquidations (rapid succession)
        if self._detect_cascade_pattern(liquidations):
            signal_strength = min(signal_strength * 1.3, 100)
            confidence = min(confidence * 1.2, 1.0)
        
        # Generate summary
        summary = {
            'dominant_side': 'long' if long_liq_ratio > short_liq_ratio else 'short',
            'largest_liquidation': max([liq.amount_usd for liq in liquidations], default=0),
            'liquidation_rate': len(liquidations) / 60,  # per minute
            'cascade_detected': self._detect_cascade_pattern(liquidations),
            'primary_exchange': max(metrics['exchange_distribution'].items(), key=lambda x: x[1], default=('unknown', 0))[0]
        }
        
        return {
            'signal_strength': signal_strength,
            'bias': bias,
            'confidence': confidence,
            'liquidations': liquidations,
            'metrics': metrics,
            'summary': summary,
            'reasons': self._generate_liquidation_reasons(bias, metrics, summary)
        }
    
    def _detect_cascade_pattern(self, liquidations: List[LiquidationEvent]) -> bool:
        """Detect cascading liquidation pattern"""
        
        if len(liquidations) < 3:
            return False
        
        # Sort by timestamp
        sorted_liq = sorted(liquidations, key=lambda x: x.timestamp)
        
        # Look for rapid succession of liquidations
        cascade_count = 0
        for i in range(1, len(sorted_liq)):
            time_diff = (sorted_liq[i].timestamp - sorted_liq[i-1].timestamp).total_seconds()
            if time_diff < 60:  # Less than 1 minute apart
                cascade_count += 1
        
        # Cascade if more than 50% of liquidations are in rapid succession
        return cascade_count > len(liquidations) * 0.5
    
    def _generate_liquidation_reasons(self, bias: str, metrics: Dict, summary: Dict) -> List[str]:
        """Generate human-readable reasons for liquidation signal"""
        
        reasons = []
        
        # Volume reasons
        if metrics['total_volume'] > 10_000_000:
            reasons.append(f"High liquidation volume: ${metrics['total_volume']:,.0f}")
        
        # Direction reasons
        if bias == 'bearish':
            reasons.append(f"Dominant long liquidations: {metrics['long_liq_ratio']:.1%}")
        elif bias == 'bullish':
            reasons.append(f"Dominant short liquidations: {metrics['short_liq_ratio']:.1%}")
        
        # Pattern reasons
        if summary['cascade_detected']:
            reasons.append("Cascading liquidations detected")
        
        # Momentum reasons
        if metrics['momentum'] > 0.6:
            reasons.append("High recent liquidation momentum")
        
        return reasons[:3]
    
    def _create_neutral_signal(self, reason: str) -> Dict[str, Any]:
        """Create neutral signal when no liquidations detected"""
        
        return {
            'signal_strength': 0.0,
            'bias': 'neutral',
            'confidence': 0.0,
            'liquidations': [],
            'metrics': {},
            'summary': {'reason': reason},
            'reasons': [reason]
        }
    
    async def get_liquidation_heatmap(self, coin: str) -> Dict[str, Any]:
        """Get liquidation heatmap data for price levels"""
        
        try:
            if not self.session:
                async with self:
                    return await self.get_liquidation_heatmap(coin)
            
            # Try to get liquidation heatmap from CoinGlass
            url = f"https://www.coinglass.com/LiquidationData"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    # Parse heatmap data - this would need custom implementation
                    # based on CoinGlass's actual heatmap structure
                    return {'heatmap': 'placeholder'}
        
        except Exception as e:
            logger.error(f"❌ Error fetching liquidation heatmap: {e}")
        
        return {'heatmap': None}