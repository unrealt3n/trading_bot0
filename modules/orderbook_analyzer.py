"""
⚡ Orderbook Analysis Module
Analyzes L2 orderbook data from top 6 exchanges
Detects volume imbalance, iceberg orders, dynamic support/resistance
Weight: 20% in signal confidence system
"""

import asyncio
import logging
import statistics
from typing import Dict, Any, Optional, List, Tuple
from .utils import HTTPClient, APIError, safe_api_call, get_emoji, normalize_signal
from config.settings import api_config, SUPPORTED_EXCHANGES

logger = logging.getLogger(__name__)

class OrderbookAnalyzer:
    """Analyzes orderbook data from multiple exchanges for market structure insights"""
    
    def __init__(self):
        self.exchange_configs = {
            'binance': {
                'base_url': api_config.binance_base_url,
                'depth_endpoint': '/api/v3/depth',
                'symbol_format': lambda coin: f"{coin}USDT"
            },
            'okx': {
                'base_url': api_config.okx_base_url,
                'depth_endpoint': '/api/v5/market/books',
                'symbol_format': lambda coin: f"{coin}-USDT"
            },
            'bybit': {
                'base_url': api_config.bybit_base_url,
                'depth_endpoint': '/v5/market/orderbook',
                'symbol_format': lambda coin: f"{coin}USDT"
            },
            'coinbase': {
                'base_url': api_config.coinbase_base_url,
                'depth_endpoint': '/products/{symbol}/book',
                'symbol_format': lambda coin: f"{coin}-USD",
                'supported_coins': ['BTC', 'ETH', 'SOL', 'XRP']  # BNB not available
            },
            'kraken': {
                'base_url': api_config.kraken_base_url,
                'depth_endpoint': '/0/public/Depth',
                'symbol_format': lambda coin: f"{coin}USD"
            },
            'bitget': {
                'base_url': api_config.bitget_base_url,
                'depth_endpoint': '/api/spot/v1/market/depth',
                'symbol_format': lambda coin: f"{coin}USDT"
            }
        }
    
    async def analyze_orderbook_signal(self, coin: str) -> Dict[str, Any]:
        """
        Main function to analyze orderbook data from multiple exchanges
        
        Returns:
        {
            'bias': 'bullish'|'bearish'|'neutral',
            'confidence': 0.0-1.0,
            'reasons': ['list of reasons'],
            'raw_data': {...}
        }
        """
        logger.info(f"{get_emoji('orderbook')} Analyzing orderbook data for {coin}")
        
        try:
            # Fetch orderbook data from all exchanges concurrently
            tasks = []
            valid_exchanges = []
            for exchange in SUPPORTED_EXCHANGES:
                # Check if coin is supported on this exchange
                config = self.exchange_configs.get(exchange, {})
                supported_coins = config.get('supported_coins', [])
                if supported_coins and coin not in supported_coins:
                    logger.info(f"ℹ️ Skipping {exchange} orderbook - {coin} not supported")
                    continue
                
                task = safe_api_call(
                    lambda ex=exchange: self._get_exchange_orderbook(ex, coin),
                    default_return=None,
                    error_message=f"{exchange} orderbook failed for {coin}"
                )
                tasks.append(task)
                valid_exchanges.append(exchange)
            
            exchange_data = await asyncio.gather(*tasks)
            
            # Filter successful responses
            valid_data = {}
            for i, data in enumerate(exchange_data):
                if data:
                    exchange_name = valid_exchanges[i]
                    valid_data[exchange_name] = data
            
            if not valid_data:
                logger.error(f"❌ No orderbook data available for {coin}")
                return {
                    'bias': 'neutral',
                    'confidence': 0.0,
                    'reasons': ['No orderbook data from any exchange'],
                    'raw_data': {'error': 'All exchanges failed'}
                }
            
            # Analyze combined orderbook data
            signal = await self._analyze_combined_orderbooks(coin, valid_data)
            logger.info(f"✅ Orderbook analysis complete for {coin} ({len(valid_data)} exchanges)")
            return signal
            
        except Exception as e:
            logger.error(f"❌ Orderbook analysis failed for {coin}: {str(e)}")
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'reasons': [f'Orderbook analysis error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }
    
    async def _get_exchange_orderbook(self, exchange: str, coin: str) -> Optional[Dict[str, Any]]:
        """Fetch orderbook data from a specific exchange"""
        
        if exchange not in self.exchange_configs:
            logger.warning(f"⚠️ Unknown exchange: {exchange}")
            return None
        
        config = self.exchange_configs[exchange]
        symbol = config['symbol_format'](coin)
        
        async with HTTPClient() as client:
            try:
                if exchange == 'coinbase':
                    # Coinbase uses symbol in URL path
                    url = f"{config['base_url']}{config['depth_endpoint'].format(symbol=symbol)}"
                    params = {'level': 2}
                else:
                    # Other exchanges use query parameters
                    url = f"{config['base_url']}{config['depth_endpoint']}"
                    params = self._get_exchange_params(exchange, symbol)
                
                response = await client.get(url, params=params)
                
                # Normalize response format across exchanges
                normalized_data = self._normalize_orderbook_data(exchange, response)
                normalized_data['exchange'] = exchange
                normalized_data['symbol'] = symbol
                
                return normalized_data
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch {exchange} orderbook for {coin}: {str(e)}")
                return None
    
    def _get_exchange_params(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """Get API parameters for each exchange"""
        params_map = {
            'binance': {'symbol': symbol, 'limit': 100},
            'okx': {'instId': symbol, 'sz': 100},
            'bybit': {'category': 'spot', 'symbol': symbol, 'limit': 50},
            'kraken': {'pair': symbol, 'count': 100},
            'bitget': {'symbol': symbol, 'limit': 100, 'type': 'step0'}
        }
        return params_map.get(exchange, {})
    
    def _normalize_orderbook_data(self, exchange: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize orderbook data format across exchanges"""
        
        try:
            if exchange == 'binance':
                return {
                    'bids': [[float(price), float(qty)] for price, qty in response.get('bids', [])],
                    'asks': [[float(price), float(qty)] for price, qty in response.get('asks', [])]
                }
            
            elif exchange == 'okx':
                data = response.get('data', [{}])[0] if response.get('data') else {}
                return {
                    'bids': [[float(bid[0]), float(bid[1])] for bid in data.get('bids', [])],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in data.get('asks', [])]
                }
            
            elif exchange == 'bybit':
                result = response.get('result', {})
                return {
                    'bids': [[float(bid[0]), float(bid[1])] for bid in result.get('b', [])],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in result.get('a', [])]
                }
            
            elif exchange == 'coinbase':
                return {
                    'bids': [[float(bid[0]), float(bid[1])] for bid in response.get('bids', [])],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in response.get('asks', [])]
                }
            
            elif exchange == 'kraken':
                # Kraken response format is different
                pair_data = list(response.get('result', {}).values())[0] if response.get('result') else {}
                return {
                    'bids': [[float(bid[0]), float(bid[1])] for bid in pair_data.get('bids', [])],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in pair_data.get('asks', [])]
                }
            
            elif exchange == 'bitget':
                data = response.get('data', {})
                return {
                    'bids': [[float(bid[0]), float(bid[1])] for bid in data.get('bids', [])],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in data.get('asks', [])]
                }
            
            else:
                return {'bids': [], 'asks': []}
                
        except Exception as e:
            logger.error(f"❌ Error normalizing {exchange} orderbook data: {str(e)}")
            return {'bids': [], 'asks': []}
    
    async def _analyze_combined_orderbooks(self, coin: str, exchange_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze combined orderbook data from multiple exchanges"""
        
        try:
            # Extract key metrics from each exchange
            exchange_metrics = {}
            all_bids = []
            all_asks = []
            
            for exchange, data in exchange_data.items():
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                
                if not bids or not asks:
                    continue
                
                # Calculate exchange-specific metrics
                metrics = self._calculate_orderbook_metrics(bids, asks)
                exchange_metrics[exchange] = metrics
                
                # Collect all price levels for overall analysis
                all_bids.extend(bids[:20])  # Top 20 levels only
                all_asks.extend(asks[:20])
            
            if not exchange_metrics:
                return {
                    'bias': 'neutral',
                    'confidence': 0.0,
                    'reasons': ['No valid orderbook data to analyze'],
                    'raw_data': {'error': 'No valid data'}
                }
            
            # Analyze overall market structure
            overall_analysis = self._analyze_market_structure(all_bids, all_asks, exchange_metrics)
            
            # Detect potential iceberg orders
            iceberg_signals = self._detect_iceberg_patterns(exchange_metrics)
            
            # Calculate final signal
            signal = self._calculate_final_orderbook_signal(overall_analysis, iceberg_signals, exchange_metrics)
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing combined orderbooks: {str(e)}")
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'reasons': [f'Analysis error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }
    
    def _calculate_orderbook_metrics(self, bids: List[List[float]], asks: List[List[float]]) -> Dict[str, float]:
        """Calculate key orderbook metrics for a single exchange"""
        
        if not bids or not asks:
            return {}
        
        try:
            # Best bid/ask
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            
            # Volume imbalance (top 10 levels)
            bid_volume = sum(qty for price, qty in bids[:10])
            ask_volume = sum(qty for price, qty in asks[:10])
            total_volume = bid_volume + ask_volume
            
            volume_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Weighted mid price (volume-weighted average of top levels)
            bid_weighted = sum(price * qty for price, qty in bids[:5])
            ask_weighted = sum(price * qty for price, qty in asks[:5])
            bid_weight = sum(qty for price, qty in bids[:5])
            ask_weight = sum(qty for price, qty in asks[:5])
            
            if bid_weight > 0 and ask_weight > 0:
                weighted_mid = ((bid_weighted / bid_weight) + (ask_weighted / ask_weight)) / 2
            else:
                weighted_mid = (best_bid + best_ask) / 2
            
            # Support/Resistance levels (high volume areas)
            support_levels = self._find_support_resistance(bids, is_support=True)
            resistance_levels = self._find_support_resistance(asks, is_support=False)
            
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_pct': spread_pct,
                'volume_imbalance': volume_imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'weighted_mid': weighted_mid,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating orderbook metrics: {str(e)}")
            return {}
    
    def _find_support_resistance(self, orders: List[List[float]], is_support: bool) -> List[float]:
        """Find significant support/resistance levels based on volume clustering"""
        
        if len(orders) < 5:
            return []
        
        try:
            # Group orders by price ranges and sum volumes
            price_ranges = {}
            for price, qty in orders[:20]:  # Top 20 levels
                # Round to nearest significant level (0.1% ranges)
                range_key = round(price * 1000) / 1000
                if range_key not in price_ranges:
                    price_ranges[range_key] = 0
                price_ranges[range_key] += qty
            
            # Find top 3 volume clusters
            sorted_ranges = sorted(price_ranges.items(), key=lambda x: x[1], reverse=True)
            significant_levels = [price for price, volume in sorted_ranges[:3]]
            
            return significant_levels
            
        except Exception:
            return []
    
    def _analyze_market_structure(self, all_bids: List[List[float]], all_asks: List[List[float]], 
                                 exchange_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze overall market structure from combined orderbook data"""
        
        try:
            # Calculate aggregate volume imbalance
            total_bid_volume = sum(metrics.get('bid_volume', 0) for metrics in exchange_metrics.values())
            total_ask_volume = sum(metrics.get('ask_volume', 0) for metrics in exchange_metrics.values())
            total_volume = total_bid_volume + total_ask_volume
            
            aggregate_imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0
            
            # Calculate average spread across exchanges
            spreads = [metrics.get('spread_pct', 0) for metrics in exchange_metrics.values() if 'spread_pct' in metrics]
            avg_spread = statistics.mean(spreads) if spreads else 0
            
            # Identify consensus support/resistance levels (appearing on multiple exchanges)
            all_support = []
            all_resistance = []
            for metrics in exchange_metrics.values():
                all_support.extend(metrics.get('support_levels', []))
                all_resistance.extend(metrics.get('resistance_levels', []))
            
            # Find levels that appear on multiple exchanges (within 0.5% tolerance)
            consensus_support = self._find_consensus_levels(all_support)
            consensus_resistance = self._find_consensus_levels(all_resistance)
            
            return {
                'aggregate_imbalance': aggregate_imbalance,
                'avg_spread': avg_spread,
                'consensus_support': consensus_support,
                'consensus_resistance': consensus_resistance,
                'exchange_count': len(exchange_metrics)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing market structure: {str(e)}")
            return {}
    
    def _find_consensus_levels(self, levels: List[float], tolerance: float = 0.005) -> List[float]:
        """Find support/resistance levels that appear across multiple exchanges"""
        
        if len(levels) < 2:
            return levels
        
        try:
            consensus_levels = []
            sorted_levels = sorted(levels)
            
            i = 0
            while i < len(sorted_levels):
                current_level = sorted_levels[i]
                cluster = [current_level]
                
                # Find all levels within tolerance
                j = i + 1
                while j < len(sorted_levels):
                    if abs(sorted_levels[j] - current_level) / current_level <= tolerance:
                        cluster.append(sorted_levels[j])
                        j += 1
                    else:
                        break
                
                # If cluster has multiple levels, it's a consensus level
                if len(cluster) >= 2:
                    consensus_levels.append(statistics.mean(cluster))
                
                i = j if j > i else i + 1
            
            return consensus_levels[:5]  # Return top 5 consensus levels
            
        except Exception:
            return []
    
    def _detect_iceberg_patterns(self, exchange_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Detect potential iceberg order patterns"""
        
        try:
            iceberg_signals = {
                'detected': False,
                'side': 'neutral',
                'confidence': 0.0,
                'reasons': []
            }
            
            # Look for signs of iceberg orders:
            # 1. Unusual spread patterns
            # 2. Volume concentration at specific levels
            # 3. Imbalance persistence across exchanges
            
            spreads = [metrics.get('spread_pct', 0) for metrics in exchange_metrics.values() if 'spread_pct' in metrics]
            imbalances = [metrics.get('volume_imbalance', 0) for metrics in exchange_metrics.values() if 'volume_imbalance' in metrics]
            
            if not spreads or not imbalances:
                return iceberg_signals
            
            # Check for spread compression (potential iceberg absorption)
            avg_spread = statistics.mean(spreads)
            if avg_spread < 0.05:  # Very tight spreads might indicate hidden liquidity
                iceberg_signals['reasons'].append('Tight spreads suggest hidden liquidity')
                iceberg_signals['confidence'] += 0.2
            
            # Check for consistent imbalance across exchanges
            imbalance_consistency = statistics.stdev(imbalances) if len(imbalances) > 1 else 1.0
            avg_imbalance = statistics.mean(imbalances)
            
            if imbalance_consistency < 0.1 and abs(avg_imbalance) > 0.1:
                # Consistent imbalance across exchanges suggests coordinated flow
                iceberg_signals['detected'] = True
                iceberg_signals['side'] = 'bullish' if avg_imbalance > 0 else 'bearish'
                iceberg_signals['confidence'] += min(0.4, abs(avg_imbalance) * 2)
                iceberg_signals['reasons'].append(f'Consistent {iceberg_signals["side"]} imbalance across exchanges')
            
            return iceberg_signals
            
        except Exception as e:
            logger.error(f"❌ Error detecting iceberg patterns: {str(e)}")
            return {'detected': False, 'side': 'neutral', 'confidence': 0.0, 'reasons': []}
    
    def _calculate_final_orderbook_signal(self, structure_analysis: Dict, iceberg_signals: Dict, 
                                        exchange_metrics: Dict) -> Dict[str, Any]:
        """Calculate final orderbook signal from all analysis components"""
        
        try:
            reasons = []
            confidence_factors = []
            
            # Factor 1: Aggregate volume imbalance
            aggregate_imbalance = structure_analysis.get('aggregate_imbalance', 0)
            if abs(aggregate_imbalance) > 0.1:
                if aggregate_imbalance > 0:
                    bias_lean = 'bullish'
                    reasons.append(f'Strong bid volume dominance ({aggregate_imbalance:.1%})')
                else:
                    bias_lean = 'bearish'
                    reasons.append(f'Strong ask volume dominance ({abs(aggregate_imbalance):.1%})')
                confidence_factors.append(min(0.7, abs(aggregate_imbalance) * 3))
            else:
                bias_lean = 'neutral'
                reasons.append(f'Balanced orderbook volumes ({aggregate_imbalance:.1%})')
                confidence_factors.append(0.2)
            
            # Factor 2: Iceberg order detection
            if iceberg_signals.get('detected', False):
                iceberg_bias = iceberg_signals.get('side', 'neutral')
                iceberg_confidence = iceberg_signals.get('confidence', 0)
                
                if iceberg_bias == bias_lean:
                    # Iceberg signals align with volume imbalance
                    confidence_factors.append(iceberg_confidence)
                    reasons.extend(iceberg_signals.get('reasons', []))
                elif iceberg_bias != 'neutral':
                    # Conflicting signals
                    bias_lean = 'neutral'
                    reasons.append('Conflicting iceberg vs volume signals')
                    confidence_factors = [0.1]
            
            # Factor 3: Exchange consensus
            exchange_count = structure_analysis.get('exchange_count', 0)
            if exchange_count >= 4:
                reasons.append(f'Strong consensus across {exchange_count} exchanges')
                confidence_factors.append(0.3)
            elif exchange_count >= 2:
                reasons.append(f'Moderate consensus across {exchange_count} exchanges')
                confidence_factors.append(0.1)
            
            # Factor 4: Support/Resistance levels
            support_levels = structure_analysis.get('consensus_support', [])
            resistance_levels = structure_analysis.get('consensus_resistance', [])
            
            if len(support_levels) > len(resistance_levels):
                reasons.append('Strong support levels identified')
                if bias_lean == 'bullish':
                    confidence_factors.append(0.2)
            elif len(resistance_levels) > len(support_levels):
                reasons.append('Strong resistance levels identified')
                if bias_lean == 'bearish':
                    confidence_factors.append(0.2)
            
            # Calculate final confidence
            final_confidence = statistics.mean(confidence_factors) if confidence_factors else 0.0
            
            return {
                'bias': bias_lean,
                'confidence': min(1.0, final_confidence),
                'reasons': reasons,
                'raw_data': {
                    'aggregate_imbalance': aggregate_imbalance,
                    'exchange_count': exchange_count,
                    'avg_spread': structure_analysis.get('avg_spread', 0),
                    'iceberg_detected': iceberg_signals.get('detected', False),
                    'consensus_support_levels': len(support_levels),
                    'consensus_resistance_levels': len(resistance_levels)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating final orderbook signal: {str(e)}")
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'reasons': [f'Signal calculation error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }