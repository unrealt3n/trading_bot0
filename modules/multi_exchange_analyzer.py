"""
🌐 Enhanced Multi-Exchange Analysis System
Advanced cross-exchange validation, arbitrage detection, and exchange reliability scoring
Maximizes accuracy through sophisticated inter-exchange analysis
"""

import asyncio
import logging
import numpy as np
import statistics
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from .utils import HTTPClient, APIError, safe_api_call, get_emoji
from config.settings import SUPPORTED_EXCHANGES

logger = logging.getLogger(__name__)

class MultiExchangeAnalyzer:
    """
    Advanced multi-exchange analysis with cross-validation and arbitrage detection
    Provides exchange reliability scoring and sophisticated flow analysis
    """
    
    def __init__(self):
        # Enhanced exchange configurations with priority and reliability data
        self.exchange_configs = {
            'binance': {
                'base_url': 'https://api.binance.com',
                'depth_endpoint': '/api/v3/depth',
                'trades_endpoint': '/api/v3/aggTrades',
                'ticker_endpoint': '/api/v3/ticker/24hr',
                'symbol_format': lambda coin: f"{coin}USDT",
                'priority': 1,  # Highest priority (most reliable)
                'typical_spread': 0.001,  # 0.1% typical spread
                'volume_rank': 1,
                'api_reliability': 0.99,
                'rate_limit_weight': 1
            },
            'okx': {
                'base_url': 'https://www.okx.com',
                'depth_endpoint': '/api/v5/market/books',
                'trades_endpoint': '/api/v5/market/trades',
                'ticker_endpoint': '/api/v5/market/ticker',
                'symbol_format': lambda coin: f"{coin}-USDT",
                'priority': 2,
                'typical_spread': 0.0015,
                'volume_rank': 2,
                'api_reliability': 0.98,
                'rate_limit_weight': 1
            },
            'bybit': {
                'base_url': 'https://api.bybit.com',
                'depth_endpoint': '/v5/market/orderbook',
                'trades_endpoint': '/v5/market/recent-trade',
                'ticker_endpoint': '/v5/market/tickers',
                'symbol_format': lambda coin: f"{coin}USDT",
                'priority': 3,
                'typical_spread': 0.002,
                'volume_rank': 3,
                'api_reliability': 0.97,
                'rate_limit_weight': 1
            },
            'coinbase': {
                'base_url': 'https://api.exchange.coinbase.com',
                'depth_endpoint': '/products/{symbol}/book',
                'trades_endpoint': '/products/{symbol}/trades',
                'ticker_endpoint': '/products/{symbol}/ticker',
                'symbol_format': lambda coin: f"{coin}-USD",
                'priority': 4,
                'typical_spread': 0.0025,
                'volume_rank': 4,
                'api_reliability': 0.96,
                'rate_limit_weight': 1,
                'supported_coins': ['BTC', 'ETH', 'SOL', 'XRP']  # BNB not available on Coinbase
            },
            'kraken': {
                'base_url': 'https://api.kraken.com',
                'depth_endpoint': '/0/public/Depth',
                'trades_endpoint': '/0/public/Trades',
                'ticker_endpoint': '/0/public/Ticker',
                'symbol_format': lambda coin: f"{coin}USD",
                'priority': 5,
                'typical_spread': 0.003,
                'volume_rank': 5,
                'api_reliability': 0.95,
                'rate_limit_weight': 2  # Kraken has stricter limits
            },
            'bitget': {
                'base_url': 'https://api.bitget.com',
                'depth_endpoint': '/api/spot/v1/market/depth',
                'trades_endpoint': '/api/spot/v1/market/fills',
                'ticker_endpoint': '/api/spot/v1/market/ticker',
                'symbol_format': lambda coin: f"{coin}USDT",
                'priority': 6,
                'typical_spread': 0.004,
                'volume_rank': 6,
                'api_reliability': 0.93,
                'rate_limit_weight': 1
            }
        }
        
        # Dynamic exchange performance tracking
        self.exchange_performance = {
            exchange: {
                'success_rate': config['api_reliability'],
                'avg_response_time': 0.0,
                'last_success': datetime.now(),
                'consecutive_failures': 0,
                'price_accuracy_score': 1.0,
                'volume_accuracy_score': 1.0
            }
            for exchange, config in self.exchange_configs.items()
        }
        
        # Arbitrage thresholds
        self.arbitrage_thresholds = {
            'BTC': 0.002,   # 0.2% spread for arbitrage opportunity
            'ETH': 0.003,   # 0.3% spread
            'SOL': 0.005,   # 0.5% spread
            'default': 0.004 # 0.4% default spread
        }
    
    async def enhanced_multi_exchange_analysis(self, coin: str) -> Dict[str, Any]:
        """
        Main function for enhanced multi-exchange analysis
        
        Returns comprehensive analysis including:
        - Cross-exchange validation
        - Arbitrage opportunities
        - Exchange reliability scores
        - Price leadership detection
        - Volume flow analysis
        """
        
        logger.info(f"{get_emoji('rocket')} Enhanced multi-exchange analysis for {coin} (targeting: {', '.join(SUPPORTED_EXCHANGES)})")
        
        try:
            # Phase 1: Concurrent data gathering from all exchanges
            exchange_data = await self._gather_multi_exchange_data(coin)
            
            if not exchange_data:
                return self._create_error_response("No exchange data available")
            
            # Phase 2: Data quality assessment and filtering
            quality_filtered_data = await self._assess_data_quality(exchange_data, coin)
            
            # Phase 3: Cross-exchange validation
            validation_results = await self._cross_validate_exchange_data(quality_filtered_data, coin)
            
            # Phase 4: Arbitrage opportunity detection
            arbitrage_analysis = await self._detect_arbitrage_opportunities(quality_filtered_data, coin)
            
            # Phase 5: Price leadership analysis
            leadership_analysis = await self._analyze_price_leadership(quality_filtered_data, coin)
            
            # Phase 6: Volume flow analysis
            volume_flow_analysis = await self._analyze_volume_flows(quality_filtered_data, coin)
            
            # Phase 7: Exchange reliability scoring
            reliability_scores = await self._calculate_exchange_reliability_scores(quality_filtered_data)
            
            # Phase 8: Synthesize enhanced signal
            enhanced_signal = await self._synthesize_enhanced_signal(
                coin, validation_results, arbitrage_analysis, leadership_analysis,
                volume_flow_analysis, reliability_scores, quality_filtered_data
            )
            
            # Count successful exchanges
            successful_exchanges = []
            if 'raw_data' in enhanced_signal:
                exchanges_analyzed = enhanced_signal['raw_data'].get('exchanges_analyzed', 0)
                if 'validation_results' in enhanced_signal['raw_data']:
                    successful_exchanges = list(enhanced_signal['raw_data']['validation_results'].get('exchange_metrics', {}).keys())
            
            logger.info(f"✅ Enhanced multi-exchange analysis complete for {coin} - Used: {', '.join(successful_exchanges) if successful_exchanges else 'No exchanges'} ({len(successful_exchanges)}/{len(SUPPORTED_EXCHANGES)} available)")
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"❌ Enhanced multi-exchange analysis failed for {coin}: {str(e)}")
            return self._create_error_response(str(e))
    
    async def _gather_multi_exchange_data(self, coin: str) -> Dict[str, Any]:
        """Enhanced data gathering with parallel execution and error handling"""
        
        # Create tasks for each exchange with different data types
        tasks = {}
        attempted_exchanges = []
        skipped_exchanges = []
        
        for exchange in SUPPORTED_EXCHANGES:
            if exchange in self.exchange_configs:
                config = self.exchange_configs[exchange]
                
                # Skip if exchange has too many consecutive failures
                if self.exchange_performance[exchange]['consecutive_failures'] >= 3:
                    skipped_exchanges.append(f"{exchange} (failures)")
                    continue
                
                # Skip if coin is not supported on this exchange
                supported_coins = config.get('supported_coins', [])
                if supported_coins and coin not in supported_coins:
                    skipped_exchanges.append(f"{exchange} (unsupported)")
                    continue
                
                # Create tasks for different data types
                attempted_exchanges.append(exchange)
                tasks[f"{exchange}_orderbook"] = self._fetch_enhanced_orderbook(exchange, coin)
                tasks[f"{exchange}_trades"] = self._fetch_recent_trades(exchange, coin)
                tasks[f"{exchange}_ticker"] = self._fetch_ticker_data(exchange, coin)
        
        if not tasks:
            logger.warning(f"⚠️ No exchanges available for {coin} - All skipped: {', '.join(skipped_exchanges) if skipped_exchanges else 'None'}")
            return {}
        
        # Log what we're attempting
        logger.info(f"🔄 Fetching data from {len(attempted_exchanges)} exchanges: {', '.join(attempted_exchanges)}")
        if skipped_exchanges:
            logger.info(f"⏸️ Skipped: {', '.join(skipped_exchanges)}")
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Process results and update exchange performance
        exchange_data = {}
        for i, (task_name, task) in enumerate(tasks.items()):
            result = results[i]
            exchange = task_name.split('_')[0]
            data_type = '_'.join(task_name.split('_')[1:])
            
            if isinstance(result, Exception):
                logger.error(f"❌ {task_name} failed: {str(result)}")
                self._update_exchange_performance(exchange, success=False)
            elif result:
                if exchange not in exchange_data:
                    exchange_data[exchange] = {}
                exchange_data[exchange][data_type] = result
                self._update_exchange_performance(exchange, success=True)
        
        return exchange_data
    
    async def _fetch_enhanced_orderbook(self, exchange: str, coin: str) -> Optional[Dict[str, Any]]:
        """Fetch enhanced orderbook data with metadata"""
        
        config = self.exchange_configs[exchange]
        symbol = config['symbol_format'](coin)
        
        async with HTTPClient() as client:
            try:
                start_time = datetime.now()
                
                if exchange == 'coinbase':
                    url = f"{config['base_url']}{config['depth_endpoint'].format(symbol=symbol)}"
                    params = {'level': 2}
                else:
                    url = f"{config['base_url']}{config['depth_endpoint']}"
                    params = self._get_orderbook_params(exchange, symbol)
                
                response = await client.get(url, params=params)
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Normalize orderbook data
                normalized_data = self._normalize_orderbook_response(exchange, response)
                
                # Add metadata
                normalized_data.update({
                    'exchange': exchange,
                    'symbol': symbol,
                    'fetch_time': start_time.isoformat(),
                    'response_time': response_time,
                    'data_quality': self._assess_orderbook_quality(normalized_data)
                })
                
                return normalized_data
                
            except Exception as e:
                logger.error(f"❌ {exchange} orderbook fetch failed: {str(e)}")
                return None
    
    async def _fetch_recent_trades(self, exchange: str, coin: str) -> Optional[Dict[str, Any]]:
        """Fetch recent trade data for flow analysis"""
        
        config = self.exchange_configs[exchange]
        symbol = config['symbol_format'](coin)
        
        async with HTTPClient() as client:
            try:
                if exchange == 'coinbase':
                    url = f"{config['base_url']}{config['trades_endpoint'].format(symbol=symbol)}"
                    params = {'limit': 100}
                else:
                    url = f"{config['base_url']}{config['trades_endpoint']}"
                    params = self._get_trades_params(exchange, symbol)
                
                response = await client.get(url, params=params)
                
                # Normalize trade data
                normalized_trades = self._normalize_trades_response(exchange, response)
                
                return {
                    'trades': normalized_trades,
                    'exchange': exchange,
                    'symbol': symbol,
                    'trade_count': len(normalized_trades)
                }
                
            except Exception as e:
                logger.error(f"❌ {exchange} trades fetch failed: {str(e)}")
                return None
    
    async def _fetch_ticker_data(self, exchange: str, coin: str) -> Optional[Dict[str, Any]]:
        """Fetch 24h ticker data for validation"""
        
        config = self.exchange_configs[exchange]
        symbol = config['symbol_format'](coin)
        
        async with HTTPClient() as client:
            try:
                if exchange == 'coinbase':
                    url = f"{config['base_url']}{config['ticker_endpoint'].format(symbol=symbol)}"
                    params = {}
                else:
                    url = f"{config['base_url']}{config['ticker_endpoint']}"
                    params = self._get_ticker_params(exchange, symbol)
                
                response = await client.get(url, params=params)
                
                # Normalize ticker data
                normalized_ticker = self._normalize_ticker_response(exchange, response)
                
                return {
                    **normalized_ticker,
                    'exchange': exchange,
                    'symbol': symbol
                }
                
            except Exception as e:
                logger.error(f"❌ {exchange} ticker fetch failed: {str(e)}")
                return None
    
    def _get_orderbook_params(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """Get exchange-specific orderbook parameters"""
        params_map = {
            'binance': {'symbol': symbol, 'limit': 100},
            'okx': {'instId': symbol, 'sz': 100},
            'bybit': {'category': 'spot', 'symbol': symbol, 'limit': 50},
            'kraken': {'pair': symbol, 'count': 100},
            'bitget': {'symbol': symbol, 'limit': 100, 'type': 'step0'}
        }
        return params_map.get(exchange, {})
    
    def _get_trades_params(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """Get exchange-specific trade parameters"""
        params_map = {
            'binance': {'symbol': symbol, 'limit': 100},
            'okx': {'instId': symbol, 'limit': 100},
            'bybit': {'category': 'spot', 'symbol': symbol, 'limit': 60},
            'kraken': {'pair': symbol, 'count': 100},
            'bitget': {'symbol': symbol, 'limit': 100}
        }
        return params_map.get(exchange, {})
    
    def _get_ticker_params(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """Get exchange-specific ticker parameters"""
        params_map = {
            'binance': {'symbol': symbol},
            'okx': {'instId': symbol},
            'bybit': {'category': 'spot', 'symbol': symbol},
            'kraken': {'pair': symbol},
            'bitget': {'symbol': symbol}
        }
        return params_map.get(exchange, {})
    
    def _normalize_orderbook_response(self, exchange: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced orderbook normalization with quality metrics"""
        
        try:
            if exchange == 'binance':
                bids = [[float(price), float(qty)] for price, qty in response.get('bids', [])]
                asks = [[float(price), float(qty)] for price, qty in response.get('asks', [])]
            
            elif exchange == 'okx':
                data = response.get('data', [{}])[0] if response.get('data') else {}
                bids = [[float(bid[0]), float(bid[1])] for bid in data.get('bids', [])]
                asks = [[float(ask[0]), float(ask[1])] for ask in data.get('asks', [])]
            
            elif exchange == 'bybit':
                result = response.get('result', {})
                bids = [[float(bid[0]), float(bid[1])] for bid in result.get('b', [])]
                asks = [[float(ask[0]), float(ask[1])] for ask in result.get('a', [])]
            
            elif exchange == 'coinbase':
                bids = [[float(bid[0]), float(bid[1])] for bid in response.get('bids', [])]
                asks = [[float(ask[0]), float(ask[1])] for ask in response.get('asks', [])]
            
            elif exchange == 'kraken':
                pair_data = list(response.get('result', {}).values())[0] if response.get('result') else {}
                bids = [[float(bid[0]), float(bid[1])] for bid in pair_data.get('bids', [])]
                asks = [[float(ask[0]), float(ask[1])] for ask in pair_data.get('asks', [])]
            
            elif exchange == 'bitget':
                data = response.get('data', {})
                bids = [[float(bid[0]), float(bid[1])] for bid in data.get('bids', [])]
                asks = [[float(ask[0]), float(ask[1])] for ask in data.get('asks', [])]
            
            else:
                bids, asks = [], []
            
            return {'bids': bids, 'asks': asks}
            
        except Exception as e:
            logger.error(f"❌ Error normalizing {exchange} orderbook: {str(e)}")
            return {'bids': [], 'asks': []}
    
    def _normalize_trades_response(self, exchange: str, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize trade data across exchanges"""
        
        try:
            trades = []
            
            if exchange == 'binance':
                for trade in response:
                    trades.append({
                        'price': float(trade.get('p', 0)),
                        'size': float(trade.get('q', 0)),
                        'timestamp': int(trade.get('T', 0)) / 1000,
                        'side': 'buy' if trade.get('m', False) else 'sell'
                    })
            
            elif exchange == 'okx':
                data = response.get('data', [])
                for trade in data:
                    trades.append({
                        'price': float(trade.get('px', 0)),
                        'size': float(trade.get('sz', 0)),
                        'timestamp': int(trade.get('ts', 0)) / 1000,
                        'side': trade.get('side', 'unknown')
                    })
            
            elif exchange == 'bybit':
                result = response.get('result', {}).get('list', [])
                for trade in result:
                    trades.append({
                        'price': float(trade.get('price', 0)),
                        'size': float(trade.get('size', 0)),
                        'timestamp': int(trade.get('time', 0)) / 1000,
                        'side': trade.get('side', 'unknown').lower()
                    })
            
            # Add similar logic for other exchanges...
            
            return trades[:100]  # Limit to last 100 trades
            
        except Exception as e:
            logger.error(f"❌ Error normalizing {exchange} trades: {str(e)}")
            return []
    
    def _normalize_ticker_response(self, exchange: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize 24h ticker data"""
        
        try:
            if exchange == 'binance':
                return {
                    'last_price': float(response.get('lastPrice', 0)),
                    'volume_24h': float(response.get('volume', 0)),
                    'price_change_24h': float(response.get('priceChangePercent', 0)),
                    'high_24h': float(response.get('highPrice', 0)),
                    'low_24h': float(response.get('lowPrice', 0))
                }
            
            elif exchange == 'okx':
                data = response.get('data', [{}])[0]
                return {
                    'last_price': float(data.get('last', 0)),
                    'volume_24h': float(data.get('vol24h', 0)),
                    'price_change_24h': float(data.get('sodUtc8', 0)),
                    'high_24h': float(data.get('high24h', 0)),
                    'low_24h': float(data.get('low24h', 0))
                }
            
            # Add similar logic for other exchanges...
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error normalizing {exchange} ticker: {str(e)}")
            return {}
    
    def _assess_orderbook_quality(self, orderbook_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of orderbook data"""
        
        bids = orderbook_data.get('bids', [])
        asks = orderbook_data.get('asks', [])
        
        if not bids or not asks:
            return {'overall_quality': 0.0}
        
        try:
            # Quality metrics
            depth_quality = min(1.0, len(bids) / 20)  # Prefer 20+ levels
            spread_quality = self._calculate_spread_quality(bids, asks)
            volume_quality = self._calculate_volume_quality(bids, asks)
            price_continuity = self._calculate_price_continuity(bids, asks)
            
            overall_quality = statistics.mean([
                depth_quality, spread_quality, volume_quality, price_continuity
            ])
            
            return {
                'overall_quality': overall_quality,
                'depth_quality': depth_quality,
                'spread_quality': spread_quality,
                'volume_quality': volume_quality,
                'price_continuity': price_continuity
            }
            
        except Exception as e:
            logger.error(f"❌ Error assessing orderbook quality: {str(e)}")
            return {'overall_quality': 0.0}
    
    def _calculate_spread_quality(self, bids: List[List[float]], asks: List[List[float]]) -> float:
        """Calculate spread quality (tighter = better)"""
        
        if not bids or not asks:
            return 0.0
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        
        if best_bid <= 0 or best_ask <= 0:
            return 0.0
        
        spread_pct = (best_ask - best_bid) / best_bid
        
        # Quality decreases as spread increases
        # Good spread < 0.1%, poor spread > 1%
        if spread_pct <= 0.001:
            return 1.0
        elif spread_pct <= 0.005:
            return 0.8
        elif spread_pct <= 0.01:
            return 0.6
        else:
            return max(0.1, 1.0 - spread_pct * 100)
    
    def _calculate_volume_quality(self, bids: List[List[float]], asks: List[List[float]]) -> float:
        """Calculate volume quality (more volume = better)"""
        
        if not bids or not asks:
            return 0.0
        
        total_bid_volume = sum(size for price, size in bids[:10])
        total_ask_volume = sum(size for price, size in asks[:10])
        total_volume = total_bid_volume + total_ask_volume
        
        # Normalize based on typical volumes (this would be calibrated per coin)
        # For now, use a simple threshold
        if total_volume > 100:
            return 1.0
        elif total_volume > 50:
            return 0.8
        elif total_volume > 10:
            return 0.6
        else:
            return 0.3
    
    def _calculate_price_continuity(self, bids: List[List[float]], asks: List[List[float]]) -> float:
        """Calculate price level continuity (no gaps = better)"""
        
        if len(bids) < 5 or len(asks) < 5:
            return 0.5
        
        # Check for reasonable price progression
        bid_gaps = []
        for i in range(min(4, len(bids) - 1)):
            price_diff = abs(bids[i][0] - bids[i + 1][0]) / bids[i][0]
            bid_gaps.append(price_diff)
        
        ask_gaps = []
        for i in range(min(4, len(asks) - 1)):
            price_diff = abs(asks[i + 1][0] - asks[i][0]) / asks[i][0]
            ask_gaps.append(price_diff)
        
        all_gaps = bid_gaps + ask_gaps
        if not all_gaps:
            return 0.5
        
        avg_gap = statistics.mean(all_gaps)
        
        # Good continuity has small, consistent gaps
        if avg_gap <= 0.001:
            return 1.0
        elif avg_gap <= 0.005:
            return 0.8
        elif avg_gap <= 0.01:
            return 0.6
        else:
            return 0.3
    
    def _update_exchange_performance(self, exchange: str, success: bool):
        """Update exchange performance metrics"""
        
        if exchange not in self.exchange_performance:
            return
        
        perf = self.exchange_performance[exchange]
        
        if success:
            perf['consecutive_failures'] = 0
            perf['last_success'] = datetime.now()
            # Update success rate with exponential moving average
            perf['success_rate'] = perf['success_rate'] * 0.95 + 0.05
        else:
            perf['consecutive_failures'] += 1
            # Decrease success rate
            perf['success_rate'] = perf['success_rate'] * 0.95
    
    async def _assess_data_quality(self, exchange_data: Dict[str, Any], coin: str) -> Dict[str, Any]:
        """Assess and filter data based on quality metrics"""
        
        quality_filtered = {}
        quality_scores = {}
        
        for exchange, data in exchange_data.items():
            if 'orderbook' not in data:
                continue
            
            orderbook = data['orderbook']
            quality_score = orderbook.get('data_quality', {}).get('overall_quality', 0.0)
            
            # Only include exchanges with reasonable quality
            if quality_score >= 0.3:  # Minimum 30% quality threshold
                quality_filtered[exchange] = data
                quality_scores[exchange] = quality_score
            else:
                logger.warning(f"⚠️ Excluding {exchange} due to low data quality: {quality_score:.2f}")
        
        # Add quality ranking
        for exchange in quality_filtered:
            quality_filtered[exchange]['quality_rank'] = len(quality_scores) - sorted(quality_scores.values()).index(quality_scores[exchange])
        
        return quality_filtered
    
    async def _cross_validate_exchange_data(self, exchange_data: Dict[str, Any], coin: str) -> Dict[str, Any]:
        """Cross-validate data across exchanges for anomaly detection"""
        
        if len(exchange_data) < 2:
            return {'validation_status': 'insufficient_data', 'anomalies': []}
        
        # Extract key metrics for validation
        exchange_metrics = {}
        for exchange, data in exchange_data.items():
            orderbook = data.get('orderbook', {})
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                continue
            
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            spread = (best_ask - best_bid) / mid_price
            
            # Calculate top-of-book volume
            top_bid_volume = bids[0][1]
            top_ask_volume = asks[0][1]
            
            exchange_metrics[exchange] = {
                'mid_price': mid_price,
                'spread': spread,
                'top_bid_volume': top_bid_volume,
                'top_ask_volume': top_ask_volume,
                'best_bid': best_bid,
                'best_ask': best_ask
            }
        
        if len(exchange_metrics) < 2:
            return {'validation_status': 'insufficient_metrics', 'anomalies': []}
        
        # Detect anomalies
        anomalies = []
        price_values = [metrics['mid_price'] for metrics in exchange_metrics.values()]
        price_median = statistics.median(price_values)
        price_std = statistics.stdev(price_values) if len(price_values) > 1 else 0
        
        # Price anomaly detection
        for exchange, metrics in exchange_metrics.items():
            price_deviation = abs(metrics['mid_price'] - price_median) / price_median
            
            if price_deviation > 0.01:  # 1% deviation threshold
                anomalies.append({
                    'exchange': exchange,
                    'type': 'price_anomaly',
                    'deviation': price_deviation,
                    'value': metrics['mid_price'],
                    'median': price_median
                })
        
        # Spread anomaly detection
        spread_values = [metrics['spread'] for metrics in exchange_metrics.values()]
        spread_median = statistics.median(spread_values)
        
        for exchange, metrics in exchange_metrics.items():
            if metrics['spread'] > spread_median * 3:  # 3x median spread
                anomalies.append({
                    'exchange': exchange,
                    'type': 'spread_anomaly',
                    'value': metrics['spread'],
                    'median': spread_median
                })
        
        # Calculate consensus metrics
        consensus_price = price_median
        consensus_spread = spread_median
        price_consensus_strength = 1.0 - (price_std / price_median if price_median > 0 else 1.0)
        
        return {
            'validation_status': 'success',
            'anomalies': anomalies,
            'consensus_price': consensus_price,
            'consensus_spread': consensus_spread,
            'price_consensus_strength': price_consensus_strength,
            'exchanges_validated': len(exchange_metrics),
            'exchange_metrics': exchange_metrics
        }
    
    async def _detect_arbitrage_opportunities(self, exchange_data: Dict[str, Any], coin: str) -> Dict[str, Any]:
        """Detect arbitrage opportunities between exchanges"""
        
        if len(exchange_data) < 2:
            return {'arbitrage_opportunities': [], 'max_spread': 0.0}
        
        # Extract prices from all exchanges
        exchange_prices = {}
        for exchange, data in exchange_data.items():
            orderbook = data.get('orderbook', {})
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if bids and asks:
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                exchange_prices[exchange] = {'bid': best_bid, 'ask': best_ask}
        
        if len(exchange_prices) < 2:
            return {'arbitrage_opportunities': [], 'max_spread': 0.0}
        
        # Find arbitrage opportunities
        opportunities = []
        max_spread = 0.0
        
        exchanges = list(exchange_prices.keys())
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                exchange_a = exchanges[i]
                exchange_b = exchanges[j]
                
                price_a = exchange_prices[exchange_a]
                price_b = exchange_prices[exchange_b]
                
                # Check both directions
                # Direction 1: Buy on A, sell on B
                spread_ab = (price_b['bid'] - price_a['ask']) / price_a['ask']
                
                # Direction 2: Buy on B, sell on A
                spread_ba = (price_a['bid'] - price_b['ask']) / price_b['ask']
                
                threshold = self.arbitrage_thresholds.get(coin, self.arbitrage_thresholds['default'])
                
                if spread_ab > threshold:
                    opportunities.append({
                        'buy_exchange': exchange_a,
                        'sell_exchange': exchange_b,
                        'buy_price': price_a['ask'],
                        'sell_price': price_b['bid'],
                        'spread_percentage': spread_ab * 100,
                        'direction': f"Buy {exchange_a} -> Sell {exchange_b}"
                    })
                    max_spread = max(max_spread, spread_ab)
                
                if spread_ba > threshold:
                    opportunities.append({
                        'buy_exchange': exchange_b,
                        'sell_exchange': exchange_a,
                        'buy_price': price_b['ask'],
                        'sell_price': price_a['bid'],
                        'spread_percentage': spread_ba * 100,
                        'direction': f"Buy {exchange_b} -> Sell {exchange_a}"
                    })
                    max_spread = max(max_spread, spread_ba)
        
        # Sort opportunities by profitability
        opportunities.sort(key=lambda x: x['spread_percentage'], reverse=True)
        
        return {
            'arbitrage_opportunities': opportunities[:5],  # Top 5 opportunities
            'max_spread': max_spread * 100,  # Convert to percentage
            'opportunity_count': len(opportunities),
            'exchanges_analyzed': len(exchange_prices)
        }
    
    async def _analyze_price_leadership(self, exchange_data: Dict[str, Any], coin: str) -> Dict[str, Any]:
        """Analyze which exchanges lead price movements"""
        
        # This is a simplified version - full implementation would require historical price data
        
        exchange_priorities = {}
        volume_weights = {}
        
        for exchange, data in exchange_data.items():
            config = self.exchange_configs.get(exchange, {})
            priority = config.get('priority', 99)
            
            # Calculate volume weight
            orderbook = data.get('orderbook', {})
            bids = orderbook.get('bids', [])[:5]  # Top 5 levels
            asks = orderbook.get('asks', [])[:5]
            
            total_volume = sum(size for price, size in bids + asks)
            volume_weights[exchange] = total_volume
            
            exchange_priorities[exchange] = {
                'priority_score': 1.0 / priority,  # Lower priority number = higher score
                'volume_weight': total_volume,
                'reliability': self.exchange_performance[exchange]['success_rate']
            }
        
        # Determine price leader
        if exchange_priorities:
            # Combine priority, volume, and reliability
            leadership_scores = {}
            for exchange, metrics in exchange_priorities.items():
                score = (metrics['priority_score'] * 0.4 + 
                        metrics['volume_weight'] / max(volume_weights.values() or [1]) * 0.4 +
                        metrics['reliability'] * 0.2)
                leadership_scores[exchange] = score
            
            price_leader = max(leadership_scores.keys(), key=lambda k: leadership_scores[k])
            
            return {
                'price_leader': price_leader,
                'leadership_scores': leadership_scores,
                'leadership_strength': leadership_scores[price_leader],
                'volume_distribution': volume_weights
            }
        
        return {'price_leader': None, 'leadership_scores': {}}
    
    async def _analyze_volume_flows(self, exchange_data: Dict[str, Any], coin: str) -> Dict[str, Any]:
        """Analyze volume flows and patterns across exchanges"""
        
        volume_analysis = {}
        total_volume = 0.0
        
        for exchange, data in exchange_data.items():
            orderbook = data.get('orderbook', {})
            bids = orderbook.get('bids', [])[:10]
            asks = orderbook.get('asks', [])[:10]
            
            bid_volume = sum(size for price, size in bids)
            ask_volume = sum(size for price, size in asks)
            exchange_volume = bid_volume + ask_volume
            
            volume_imbalance = (bid_volume - ask_volume) / exchange_volume if exchange_volume > 0 else 0
            
            volume_analysis[exchange] = {
                'total_volume': exchange_volume,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'volume_imbalance': volume_imbalance
            }
            
            total_volume += exchange_volume
        
        # Calculate volume distribution
        volume_distribution = {}
        dominant_exchange = None
        max_volume_share = 0.0
        
        for exchange, metrics in volume_analysis.items():
            volume_share = metrics['total_volume'] / total_volume if total_volume > 0 else 0
            volume_distribution[exchange] = volume_share
            
            if volume_share > max_volume_share:
                max_volume_share = volume_share
                dominant_exchange = exchange
        
        # Analyze volume concentration
        volume_concentration = max_volume_share if max_volume_share else 0.0
        
        # Check for volume imbalance consensus
        imbalances = [metrics['volume_imbalance'] for metrics in volume_analysis.values()]
        avg_imbalance = statistics.mean(imbalances) if imbalances else 0.0
        imbalance_consensus = 1.0 - statistics.stdev(imbalances) if len(imbalances) > 1 else 0.0
        
        return {
            'volume_distribution': volume_distribution,
            'dominant_exchange': dominant_exchange,
            'volume_concentration': volume_concentration,
            'avg_volume_imbalance': avg_imbalance,
            'imbalance_consensus': imbalance_consensus,
            'total_volume_analyzed': total_volume,
            'exchange_volume_analysis': volume_analysis
        }
    
    async def _calculate_exchange_reliability_scores(self, exchange_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate dynamic reliability scores for each exchange"""
        
        reliability_scores = {}
        
        for exchange, data in exchange_data.items():
            # Base reliability from configuration
            base_reliability = self.exchange_configs.get(exchange, {}).get('api_reliability', 0.5)
            
            # Current performance
            current_performance = self.exchange_performance.get(exchange, {})
            
            # Data quality
            orderbook = data.get('orderbook', {})
            data_quality = orderbook.get('data_quality', {}).get('overall_quality', 0.5)
            
            # Response time factor
            response_time = orderbook.get('response_time', 5.0)
            response_factor = max(0.1, min(1.0, 2.0 / response_time))  # Faster = better
            
            # Calculate composite reliability score
            reliability_score = (
                base_reliability * 0.4 +
                current_performance.get('success_rate', 0.5) * 0.3 +
                data_quality * 0.2 +
                response_factor * 0.1
            )
            
            reliability_scores[exchange] = {
                'overall_score': reliability_score,
                'base_reliability': base_reliability,
                'current_performance': current_performance.get('success_rate', 0.5),
                'data_quality': data_quality,
                'response_factor': response_factor
            }
        
        return reliability_scores
    
    async def _synthesize_enhanced_signal(self, coin: str, validation_results: Dict,
                                        arbitrage_analysis: Dict, leadership_analysis: Dict,
                                        volume_flow_analysis: Dict, reliability_scores: Dict,
                                        exchange_data: Dict) -> Dict[str, Any]:
        """Synthesize all analyses into enhanced trading signal"""
        
        try:
            # Enhanced signal components
            signal_strength = 0.0
            confidence_factors = []
            reasons = []
            bias = 'neutral'
            
            # Factor 1: Cross-exchange consensus (30% weight)
            consensus_strength = validation_results.get('price_consensus_strength', 0.0)
            signal_strength += consensus_strength * 0.30
            confidence_factors.append(consensus_strength)
            
            if consensus_strength > 0.8:
                price_deviation = (1.0 - consensus_strength) * 100
                reasons.append(f"Price deviation <{price_deviation:.1f}% across exchanges")
            
            # Factor 2: Volume flow analysis (25% weight)
            volume_imbalance = volume_flow_analysis.get('avg_volume_imbalance', 0.0)
            imbalance_consensus = volume_flow_analysis.get('imbalance_consensus', 0.0)
            
            volume_factor = abs(volume_imbalance) * imbalance_consensus
            signal_strength += volume_factor * 0.25
            confidence_factors.append(volume_factor)
            
            if abs(volume_imbalance) > 0.1 and imbalance_consensus > 0.6:
                direction = 'bullish' if volume_imbalance > 0 else 'bearish'
                bias = direction
                imbalance_pct = abs(volume_imbalance) * 100
                reasons.append(f"{direction.title()} volume imbalance: {imbalance_pct:.1f}% across exchanges")
            
            # Factor 3: Price leadership confirmation (20% weight)
            price_leader = leadership_analysis.get('price_leader')
            leadership_strength = leadership_analysis.get('leadership_strength', 0.0)
            
            if price_leader and leadership_strength > 0.7:
                signal_strength += leadership_strength * 0.20
                confidence_factors.append(leadership_strength)
                leadership_pct = leadership_strength * 100
                reasons.append(f"{price_leader} leading price discovery ({leadership_pct:.1f}%)")
            
            # Factor 4: Arbitrage opportunities (15% weight)
            max_spread = arbitrage_analysis.get('max_spread', 0.0)
            opportunity_count = arbitrage_analysis.get('opportunity_count', 0)
            
            if opportunity_count > 0:
                arbitrage_factor = min(1.0, max_spread / 100)  # Normalize spread
                signal_strength += arbitrage_factor * 0.15
                confidence_factors.append(arbitrage_factor)
                reasons.append(f"Arbitrage opportunities detected: {max_spread:.2f}% max spread")
            
            # Factor 5: Data quality and reliability (10% weight)
            avg_reliability = statistics.mean([
                scores['overall_score'] for scores in reliability_scores.values()
            ]) if reliability_scores else 0.5
            
            signal_strength += avg_reliability * 0.10
            confidence_factors.append(avg_reliability)
            
            if avg_reliability > 0.9:
                reasons.append(f"High data quality across {len(reliability_scores)} exchanges")
            
            # Anomaly penalty
            anomaly_count = len(validation_results.get('anomalies', []))
            if anomaly_count > 0:
                anomaly_penalty = min(0.3, anomaly_count * 0.1)
                signal_strength -= anomaly_penalty
                reasons.append(f"Data anomalies detected: {anomaly_count} exchanges affected")
            
            # Final confidence calculation
            final_confidence = max(0.0, min(1.0, signal_strength))
            
            # Exchange-specific insights
            exchange_insights = []
            dominant_exchange = volume_flow_analysis.get('dominant_exchange')
            if dominant_exchange:
                volume_share = volume_flow_analysis.get('volume_distribution', {}).get(dominant_exchange, 0)
                # Only report if significantly high volume share (>40%)
                if volume_share > 0.4:
                    exchange_insights.append(f"Volume concentrated: {dominant_exchange} {volume_share:.1%}")
            
            return {
                'bias': bias,
                'confidence': final_confidence,
                'reasons': reasons[:5],  # Top 5 reasons
                'exchange_insights': exchange_insights,
                'signal_components': {
                    'consensus_strength': consensus_strength,
                    'volume_factor': volume_factor,
                    'leadership_strength': leadership_strength,
                    'arbitrage_factor': arbitrage_analysis.get('max_spread', 0) / 100,
                    'reliability_factor': avg_reliability
                },
                'raw_data': {
                    'validation_results': validation_results,
                    'arbitrage_analysis': arbitrage_analysis,
                    'leadership_analysis': leadership_analysis,
                    'volume_flow_analysis': volume_flow_analysis,
                    'reliability_scores': reliability_scores,
                    'exchanges_analyzed': len(exchange_data)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error synthesizing enhanced signal: {str(e)}")
            return self._create_error_response(str(e))
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'bias': 'neutral',
            'confidence': 0.0,
            'reasons': [f'Multi-exchange analysis error: {error_message}'],
            'exchange_insights': [],
            'signal_components': {},
            'raw_data': {'error': error_message}
        }