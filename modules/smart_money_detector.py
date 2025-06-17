"""
🧠 Smart Money Detection Module
Advanced retail vs institutional flow analysis using market microstructure
Detects whale activity, block trades, coordinated flows, and institutional patterns
"""

import asyncio
import logging
import numpy as np
import statistics
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from .utils import get_emoji, normalize_signal

logger = logging.getLogger(__name__)

class SmartMoneyDetector:
    """
    Advanced detector for institutional vs retail money flows
    Uses market microstructure analysis to identify smart money activity
    """
    
    def __init__(self):
        # Thresholds for institutional activity detection (dynamic per coin)
        self.institutional_thresholds = {
            'BTC': {
                'large_order_size': 5.0,      # 5+ BTC orders considered large
                'whale_order_size': 25.0,     # 25+ BTC orders considered whale
                'volume_percentile': 95,       # Top 5% of orders by size
                'time_clustering_window': 300, # 5 minutes for clustering
                'coordination_threshold': 0.7   # Cross-exchange coordination
            },
            'ETH': {
                'large_order_size': 50.0,     # 50+ ETH orders
                'whale_order_size': 250.0,    # 250+ ETH orders
                'volume_percentile': 95,
                'time_clustering_window': 300,
                'coordination_threshold': 0.7
            },
            'SOL': {
                'large_order_size': 500.0,    # 500+ SOL orders
                'whale_order_size': 2500.0,   # 2500+ SOL orders
                'volume_percentile': 95,
                'time_clustering_window': 300,
                'coordination_threshold': 0.7
            },
            'default': {
                'large_order_size_usd': 50000,   # $50k+ orders
                'whale_order_size_usd': 250000,  # $250k+ orders
                'volume_percentile': 95,
                'time_clustering_window': 300,
                'coordination_threshold': 0.7
            }
        }
        
        # Institutional behavior patterns
        self.institutional_patterns = {
            'time_preferences': {
                'asia_session': 0.2,      # Lower institutional activity
                'europe_session': 0.6,    # Moderate institutional activity  
                'us_session': 0.8,        # High institutional activity
                'overlap_sessions': 0.9    # Peak institutional activity
            },
            'round_number_preference': 0.8,  # Institutions love round numbers
            'iceberg_probability': 0.7,      # Probability of using iceberg orders
            'cross_exchange_coordination': 0.6  # Coordination across exchanges
        }
    
    async def detect_smart_money_activity(self, coin: str, multi_exchange_data: Dict[str, Any],
                                        trade_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Main function to detect institutional vs retail money flows
        
        Args:
            coin: Cryptocurrency symbol
            multi_exchange_data: Orderbook data from multiple exchanges
            trade_history: Optional recent trade history for flow analysis
        
        Returns:
        {
            'institutional_signal': 'bullish'|'bearish'|'neutral',
            'retail_signal': 'bullish'|'bearish'|'neutral', 
            'smart_money_confidence': 0.0-1.0,
            'institutional_activity_level': 0.0-1.0,
            'flow_analysis': {...},
            'reasons': ['specific evidence for institutional/retail activity'],
            'raw_data': {...}
        }
        """
        
        sources = ['Multi-exchange orderbook depth', 'Large order clustering', 'Cross-exchange coordination', 'Round number preferences', 'Iceberg order detection']
        logger.info(f"{get_emoji('money')} Detecting smart money activity for {coin} - Sources: {', '.join(sources[:3])} + 2 more")
        
        try:
            # Get coin-specific thresholds
            thresholds = self.institutional_thresholds.get(coin, self.institutional_thresholds['default'])
            
            # Phase 1: Analyze order size distribution
            order_size_analysis = await self._analyze_order_size_distribution(
                multi_exchange_data, thresholds, coin
            )
            
            # Phase 2: Detect temporal clustering patterns
            temporal_analysis = await self._analyze_temporal_patterns(
                multi_exchange_data, trade_history, thresholds
            )
            
            # Phase 3: Cross-exchange coordination detection
            coordination_analysis = await self._detect_cross_exchange_coordination(
                multi_exchange_data, thresholds
            )
            
            # Phase 4: Price level analysis (round numbers, support/resistance)
            price_level_analysis = await self._analyze_price_level_preferences(
                multi_exchange_data, coin
            )
            
            # Phase 5: Iceberg order detection enhancement
            iceberg_analysis = await self._advanced_iceberg_detection(
                multi_exchange_data, thresholds
            )
            
            # Phase 6: Market impact analysis
            market_impact_analysis = await self._analyze_market_impact_patterns(
                multi_exchange_data, trade_history
            )
            
            # Combine all analyses for final smart money assessment
            final_assessment = await self._synthesize_smart_money_signals(
                coin, order_size_analysis, temporal_analysis, coordination_analysis,
                price_level_analysis, iceberg_analysis, market_impact_analysis
            )
            
            # Log detailed results
            inst_signal = final_assessment.get('institutional_signal', 'neutral')
            retail_signal = final_assessment.get('retail_signal', 'neutral')
            inst_activity = final_assessment.get('institutional_activity_level', 0) * 100
            confidence = final_assessment.get('smart_money_confidence', 0) * 100
            
            logger.info(f"✅ Smart money detection complete for {coin} - Institutional: {inst_signal.upper()} ({inst_activity:.1f}%), Retail: {retail_signal.upper()} | Confidence: {confidence:.1f}%")
            return final_assessment
            
        except Exception as e:
            logger.error(f"❌ Smart money detection failed for {coin}: {str(e)}")
            return {
                'institutional_signal': 'neutral',
                'retail_signal': 'neutral',
                'smart_money_confidence': 0.0,
                'institutional_activity_level': 0.0,
                'flow_analysis': {},
                'reasons': [f'Smart money detection error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }
    
    async def _analyze_order_size_distribution(self, multi_exchange_data: Dict[str, Any], 
                                             thresholds: Dict, coin: str) -> Dict[str, Any]:
        """Analyze order size distribution to identify institutional vs retail patterns"""
        
        try:
            all_orders = []
            exchange_patterns = {}
            
            # Collect all orders from all exchanges
            for exchange, data in multi_exchange_data.items():
                if not data or 'bids' not in data or 'asks' not in data:
                    continue
                
                exchange_orders = []
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                
                # Process bids and asks
                for price, size in bids[:20]:  # Top 20 levels
                    exchange_orders.append({
                        'price': float(price),
                        'size': float(size),
                        'side': 'bid',
                        'exchange': exchange
                    })
                
                for price, size in asks[:20]:
                    exchange_orders.append({
                        'price': float(price),
                        'size': float(size),
                        'side': 'ask',
                        'exchange': exchange
                    })
                
                all_orders.extend(exchange_orders)
                exchange_patterns[exchange] = self._analyze_single_exchange_patterns(exchange_orders, thresholds)
            
            if not all_orders:
                return {'error': 'No order data available'}
            
            # Calculate size distribution metrics
            sizes = [order['size'] for order in all_orders]
            
            # Statistical analysis
            size_percentiles = {
                '50th': np.percentile(sizes, 50),
                '75th': np.percentile(sizes, 75), 
                '90th': np.percentile(sizes, 90),
                '95th': np.percentile(sizes, 95),
                '99th': np.percentile(sizes, 99)
            }
            
            # Institutional order identification
            large_order_threshold = thresholds.get('large_order_size', thresholds.get('large_order_size_usd', 50000))
            whale_order_threshold = thresholds.get('whale_order_size', thresholds.get('whale_order_size_usd', 250000))
            
            large_orders = [o for o in all_orders if o['size'] >= large_order_threshold]
            whale_orders = [o for o in all_orders if o['size'] >= whale_order_threshold]
            
            # Calculate institutional activity metrics
            total_orders = len(all_orders)
            large_order_ratio = len(large_orders) / total_orders if total_orders > 0 else 0
            whale_order_ratio = len(whale_orders) / total_orders if total_orders > 0 else 0
            
            # Volume concentration analysis
            total_volume = sum(sizes)
            large_order_volume = sum(o['size'] for o in large_orders)
            whale_order_volume = sum(o['size'] for o in whale_orders)
            
            large_volume_concentration = large_order_volume / total_volume if total_volume > 0 else 0
            whale_volume_concentration = whale_order_volume / total_volume if total_volume > 0 else 0
            
            # Side bias analysis for institutional orders
            large_bids = [o for o in large_orders if o['side'] == 'bid']
            large_asks = [o for o in large_orders if o['side'] == 'ask']
            
            institutional_side_bias = 'neutral'
            if len(large_bids) > len(large_asks) * 1.5:
                institutional_side_bias = 'bullish'
            elif len(large_asks) > len(large_bids) * 1.5:
                institutional_side_bias = 'bearish'
            
            return {
                'size_percentiles': size_percentiles,
                'large_order_ratio': large_order_ratio,
                'whale_order_ratio': whale_order_ratio,
                'large_volume_concentration': large_volume_concentration,
                'whale_volume_concentration': whale_volume_concentration,
                'institutional_side_bias': institutional_side_bias,
                'total_orders_analyzed': total_orders,
                'large_orders_count': len(large_orders),
                'whale_orders_count': len(whale_orders),
                'exchange_patterns': exchange_patterns
            }
            
        except Exception as e:
            logger.error(f"❌ Error in order size analysis: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_single_exchange_patterns(self, orders: List[Dict], thresholds: Dict) -> Dict[str, Any]:
        """Analyze patterns for a single exchange"""
        
        if not orders:
            return {}
        
        sizes = [o['size'] for o in orders]
        
        # Gini coefficient for size inequality (higher = more concentrated)
        gini = self._calculate_gini_coefficient(sizes)
        
        # Order size clustering detection
        size_clusters = self._detect_size_clustering(sizes)
        
        return {
            'gini_coefficient': gini,
            'size_clustering_score': size_clusters,
            'avg_order_size': statistics.mean(sizes),
            'order_count': len(orders)
        }
    
    def _calculate_gini_coefficient(self, sizes: List[float]) -> float:
        """Calculate Gini coefficient for order size inequality"""
        
        if not sizes:
            return 0.0
        
        # Sort sizes
        sorted_sizes = sorted(sizes)
        n = len(sorted_sizes)
        
        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_sizes)) / (n * np.sum(sorted_sizes)) - (n + 1) / n
    
    def _detect_size_clustering(self, sizes: List[float]) -> float:
        """Detect if order sizes cluster around specific values (institutional behavior)"""
        
        if len(sizes) < 5:
            return 0.0
        
        # Look for clustering around round numbers and similar sizes
        rounded_sizes = [round(size, -int(np.floor(np.log10(abs(size)))) + 1) for size in sizes if size > 0]
        
        if not rounded_sizes:
            return 0.0
        
        # Count frequency of each rounded size
        size_counts = {}
        for size in rounded_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        
        # Calculate clustering score (higher = more clustering)
        max_count = max(size_counts.values())
        total_count = len(rounded_sizes)
        
        return max_count / total_count if total_count > 0 else 0.0
    
    async def _analyze_temporal_patterns(self, multi_exchange_data: Dict[str, Any],
                                       trade_history: Optional[List[Dict]], 
                                       thresholds: Dict) -> Dict[str, Any]:
        """Analyze temporal patterns to identify institutional trading times"""
        
        try:
            current_time = datetime.now()
            current_hour = current_time.hour
            
            # Determine current session
            session = self._get_current_session(current_hour)
            institutional_time_preference = self.institutional_patterns['time_preferences'].get(session, 0.5)
            
            # Analyze trade timing if available
            temporal_clustering = 0.0
            if trade_history:
                temporal_clustering = self._analyze_trade_timing_clusters(trade_history, thresholds)
            
            # Weekend vs weekday analysis
            is_weekend = current_time.weekday() >= 5
            weekend_adjustment = 0.3 if is_weekend else 1.0  # Lower institutional activity on weekends
            
            return {
                'current_session': session,
                'institutional_time_preference': institutional_time_preference,
                'temporal_clustering_score': temporal_clustering,
                'is_weekend': is_weekend,
                'weekend_adjustment': weekend_adjustment,
                'current_hour_utc': current_hour
            }
            
        except Exception as e:
            logger.error(f"❌ Error in temporal analysis: {str(e)}")
            return {'error': str(e)}
    
    def _get_current_session(self, hour_utc: int) -> str:
        """Determine current trading session"""
        if 0 <= hour_utc <= 8:
            return 'asia_session'
        elif 7 <= hour_utc <= 16:
            if 7 <= hour_utc <= 8:
                return 'overlap_sessions'  # Asia-Europe overlap
            return 'europe_session'
        elif 13 <= hour_utc <= 22:
            if 13 <= hour_utc <= 16:
                return 'overlap_sessions'  # Europe-US overlap
            return 'us_session'
        else:
            return 'asia_session'
    
    def _analyze_trade_timing_clusters(self, trade_history: List[Dict], thresholds: Dict) -> float:
        """Analyze if large trades are clustered in time (institutional behavior)"""
        
        if not trade_history:
            return 0.0
        
        try:
            # Extract timestamps and sizes
            trades_with_time = []
            for trade in trade_history:
                if 'timestamp' in trade and 'size' in trade:
                    trades_with_time.append({
                        'timestamp': trade['timestamp'],
                        'size': float(trade['size'])
                    })
            
            if len(trades_with_time) < 10:
                return 0.0
            
            # Identify large trades
            sizes = [t['size'] for t in trades_with_time]
            large_threshold = np.percentile(sizes, 90)  # Top 10% by size
            large_trades = [t for t in trades_with_time if t['size'] >= large_threshold]
            
            if len(large_trades) < 3:
                return 0.0
            
            # Calculate time differences between large trades
            timestamps = sorted([t['timestamp'] for t in large_trades])
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            
            # Look for clustering (small time differences)
            clustering_window = thresholds.get('time_clustering_window', 300)  # 5 minutes
            clustered_trades = sum(1 for diff in time_diffs if diff <= clustering_window)
            
            return clustered_trades / len(time_diffs) if time_diffs else 0.0
            
        except Exception as e:
            logger.error(f"❌ Error analyzing trade timing: {str(e)}")
            return 0.0
    
    async def _detect_cross_exchange_coordination(self, multi_exchange_data: Dict[str, Any],
                                                thresholds: Dict) -> Dict[str, Any]:
        """Detect coordinated activity across multiple exchanges"""
        
        try:
            if len(multi_exchange_data) < 2:
                return {'coordination_score': 0.0, 'coordinated_exchanges': []}
            
            # Calculate volume imbalances for each exchange
            exchange_imbalances = {}
            exchange_spreads = {}
            
            for exchange, data in multi_exchange_data.items():
                if not data or 'bids' not in data or 'asks' not in data:
                    continue
                
                bids = data.get('bids', [])[:10]  # Top 10 levels
                asks = data.get('asks', [])[:10]
                
                if not bids or not asks:
                    continue
                
                # Calculate volume imbalance
                bid_volume = sum(float(size) for price, size in bids)
                ask_volume = sum(float(size) for price, size in asks)
                total_volume = bid_volume + ask_volume
                
                if total_volume > 0:
                    imbalance = (bid_volume - ask_volume) / total_volume
                    exchange_imbalances[exchange] = imbalance
                
                # Calculate spread
                best_bid = float(bids[0][0]) if bids else 0
                best_ask = float(asks[0][0]) if asks else 0
                if best_bid > 0 and best_ask > 0:
                    spread = (best_ask - best_bid) / best_bid
                    exchange_spreads[exchange] = spread
            
            if len(exchange_imbalances) < 2:
                return {'coordination_score': 0.0, 'coordinated_exchanges': []}
            
            # Calculate coordination metrics
            imbalance_values = list(exchange_imbalances.values())
            spread_values = list(exchange_spreads.values())
            
            # Coordination = low variance in imbalances (similar across exchanges)
            imbalance_std = np.std(imbalance_values) if len(imbalance_values) > 1 else 1.0
            spread_std = np.std(spread_values) if len(spread_values) > 1 else 1.0
            
            # Lower standard deviation = higher coordination
            coordination_score = 1.0 / (1.0 + imbalance_std + spread_std)
            
            # Identify most coordinated direction
            avg_imbalance = np.mean(imbalance_values)
            coordinated_direction = 'neutral'
            if avg_imbalance > 0.1:
                coordinated_direction = 'bullish'
            elif avg_imbalance < -0.1:
                coordinated_direction = 'bearish'
            
            # Find exchanges with similar imbalances
            coordinated_exchanges = []
            for exchange, imbalance in exchange_imbalances.items():
                if abs(imbalance - avg_imbalance) < 0.1:  # Within 10% of average
                    coordinated_exchanges.append(exchange)
            
            return {
                'coordination_score': coordination_score,
                'coordinated_direction': coordinated_direction,
                'coordinated_exchanges': coordinated_exchanges,
                'avg_imbalance': avg_imbalance,
                'imbalance_std': imbalance_std,
                'spread_std': spread_std,
                'exchange_imbalances': exchange_imbalances
            }
            
        except Exception as e:
            logger.error(f"❌ Error in coordination detection: {str(e)}")
            return {'error': str(e)}
    
    async def _analyze_price_level_preferences(self, multi_exchange_data: Dict[str, Any], 
                                             coin: str) -> Dict[str, Any]:
        """Analyze preference for round numbers and psychological levels"""
        
        try:
            all_price_levels = []
            
            # Collect all price levels from all exchanges
            for exchange, data in multi_exchange_data.items():
                if not data or 'bids' not in data or 'asks' not in data:
                    continue
                
                bids = data.get('bids', [])[:10]
                asks = data.get('asks', [])[:10]
                
                for price, size in bids + asks:
                    all_price_levels.append({
                        'price': float(price),
                        'size': float(size),
                        'exchange': exchange
                    })
            
            if not all_price_levels:
                return {'error': 'No price data available'}
            
            # Analyze round number preference
            round_number_score = self._calculate_round_number_preference(all_price_levels)
            
            # Analyze psychological level clustering
            psychological_clustering = self._analyze_psychological_levels(all_price_levels, coin)
            
            return {
                'round_number_preference_score': round_number_score,
                'psychological_clustering': psychological_clustering,
                'total_levels_analyzed': len(all_price_levels)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in price level analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_round_number_preference(self, price_levels: List[Dict]) -> float:
        """Calculate preference for round numbers (institutional behavior)"""
        
        if not price_levels:
            return 0.0
        
        round_number_volume = 0.0
        total_volume = 0.0
        
        for level in price_levels:
            price = level['price']
            size = level['size']
            total_volume += size
            
            # Check if price is a round number
            if self._is_round_number(price):
                round_number_volume += size
        
        return round_number_volume / total_volume if total_volume > 0 else 0.0
    
    def _is_round_number(self, price: float) -> bool:
        """Check if a price is a round number"""
        
        # For high-priced assets (>$1000), check for round hundreds
        if price >= 1000:
            return price % 100 == 0 or price % 50 == 0
        
        # For medium-priced assets ($10-$1000), check for round tens
        elif price >= 10:
            return price % 10 == 0 or price % 5 == 0
        
        # For low-priced assets (<$10), check for round dollars or 50 cents
        else:
            return price % 1 == 0 or price % 0.5 == 0
    
    def _analyze_psychological_levels(self, price_levels: List[Dict], coin: str) -> Dict[str, Any]:
        """Analyze clustering around psychological levels (support/resistance)"""
        
        if not price_levels:
            return {}
        
        prices = [level['price'] for level in price_levels]
        current_price = statistics.median(prices)  # Approximate current price
        
        # Define psychological levels based on current price
        psychological_levels = []
        
        # Add round number levels around current price
        for multiplier in [0.9, 0.95, 1.0, 1.05, 1.1]:
            level = current_price * multiplier
            
            # Round to appropriate psychological level
            if level >= 1000:
                rounded_level = round(level / 100) * 100
            elif level >= 10:
                rounded_level = round(level / 10) * 10
            else:
                rounded_level = round(level)
            
            psychological_levels.append(rounded_level)
        
        # Calculate clustering around these levels
        clustering_scores = {}
        for psych_level in psychological_levels:
            nearby_volume = 0.0
            total_volume = 0.0
            
            for level in price_levels:
                price = level['price']
                size = level['size']
                total_volume += size
                
                # Check if price is within 1% of psychological level
                if abs(price - psych_level) / psych_level <= 0.01:
                    nearby_volume += size
            
            clustering_scores[psych_level] = nearby_volume / total_volume if total_volume > 0 else 0.0
        
        # Find strongest clustering
        max_clustering = max(clustering_scores.values()) if clustering_scores else 0.0
        strongest_level = max(clustering_scores.keys(), key=lambda k: clustering_scores[k]) if clustering_scores else 0.0
        
        return {
            'max_clustering_score': max_clustering,
            'strongest_psychological_level': strongest_level,
            'clustering_scores': clustering_scores,
            'current_price_estimate': current_price
        }
    
    async def _advanced_iceberg_detection(self, multi_exchange_data: Dict[str, Any],
                                        thresholds: Dict) -> Dict[str, Any]:
        """Enhanced iceberg order detection using multiple signals"""
        
        try:
            iceberg_signals = []
            
            for exchange, data in multi_exchange_data.items():
                if not data or 'bids' not in data or 'asks' not in data:
                    continue
                
                exchange_iceberg = self._detect_exchange_iceberg_patterns(data, exchange, thresholds)
                if exchange_iceberg:
                    iceberg_signals.append(exchange_iceberg)
            
            if not iceberg_signals:
                return {'iceberg_detected': False, 'confidence': 0.0}
            
            # Aggregate iceberg signals
            avg_confidence = statistics.mean([signal['confidence'] for signal in iceberg_signals])
            
            # Count directional consensus
            bullish_count = sum(1 for signal in iceberg_signals if signal['direction'] == 'bullish')
            bearish_count = sum(1 for signal in iceberg_signals if signal['direction'] == 'bearish')
            
            final_direction = 'neutral'
            if bullish_count > bearish_count:
                final_direction = 'bullish'
            elif bearish_count > bullish_count:
                final_direction = 'bearish'
            
            return {
                'iceberg_detected': avg_confidence > 0.3,
                'confidence': avg_confidence,
                'direction': final_direction,
                'exchange_signals': iceberg_signals,
                'consensus_strength': max(bullish_count, bearish_count) / len(iceberg_signals)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in iceberg detection: {str(e)}")
            return {'error': str(e)}
    
    def _detect_exchange_iceberg_patterns(self, data: Dict, exchange: str, thresholds: Dict) -> Optional[Dict]:
        """Detect iceberg patterns on a single exchange"""
        
        try:
            bids = data.get('bids', [])[:20]
            asks = data.get('asks', [])[:20]
            
            if not bids or not asks:
                return None
            
            # Pattern 1: Unusual size distribution (large orders at multiple levels)
            bid_sizes = [float(size) for price, size in bids]
            ask_sizes = [float(size) for price, size in asks]
            
            # Pattern 2: Regular size patterns (iceberg orders often have consistent sizes)
            bid_regularity = self._calculate_size_regularity(bid_sizes)
            ask_regularity = self._calculate_size_regularity(ask_sizes)
            
            # Pattern 3: Deep book concentration
            total_bid_volume = sum(bid_sizes)
            total_ask_volume = sum(ask_sizes)
            
            # Look for concentration in deeper levels (iceberg hiding)
            deep_bid_volume = sum(bid_sizes[5:15])  # Levels 6-15
            deep_ask_volume = sum(ask_sizes[5:15])
            
            deep_bid_ratio = deep_bid_volume / total_bid_volume if total_bid_volume > 0 else 0
            deep_ask_ratio = deep_ask_volume / total_ask_volume if total_ask_volume > 0 else 0
            
            # Iceberg scoring
            iceberg_score = 0.0
            direction = 'neutral'
            
            # High regularity + deep book concentration suggests iceberg
            if bid_regularity > 0.6 and deep_bid_ratio > 0.3:
                iceberg_score += 0.4
                direction = 'bullish'
            
            if ask_regularity > 0.6 and deep_ask_ratio > 0.3:
                if direction == 'bullish':
                    direction = 'neutral'  # Conflicting signals
                else:
                    iceberg_score += 0.4
                    direction = 'bearish'
            
            # Additional scoring based on size patterns
            if max(bid_regularity, ask_regularity) > 0.7:
                iceberg_score += 0.3
            
            return {
                'exchange': exchange,
                'confidence': min(1.0, iceberg_score),
                'direction': direction,
                'bid_regularity': bid_regularity,
                'ask_regularity': ask_regularity,
                'deep_bid_ratio': deep_bid_ratio,
                'deep_ask_ratio': deep_ask_ratio
            }
            
        except Exception as e:
            logger.error(f"❌ Error detecting iceberg on {exchange}: {str(e)}")
            return None
    
    def _calculate_size_regularity(self, sizes: List[float]) -> float:
        """Calculate how regular/consistent order sizes are (iceberg pattern)"""
        
        if len(sizes) < 3:
            return 0.0
        
        # Calculate coefficient of variation (lower = more regular)
        mean_size = statistics.mean(sizes)
        if mean_size == 0:
            return 0.0
        
        std_size = statistics.stdev(sizes)
        cv = std_size / mean_size
        
        # Convert to regularity score (higher = more regular)
        regularity = 1.0 / (1.0 + cv)
        
        return regularity
    
    async def _analyze_market_impact_patterns(self, multi_exchange_data: Dict[str, Any],
                                            trade_history: Optional[List[Dict]]) -> Dict[str, Any]:
        """Analyze market impact patterns to identify institutional trading"""
        
        try:
            if not trade_history:
                return {'market_impact_score': 0.0, 'impact_analysis': 'No trade history available'}
            
            # Analyze large trades and their market impact
            large_trades = self._identify_large_trades(trade_history)
            
            if not large_trades:
                return {'market_impact_score': 0.0, 'impact_analysis': 'No large trades identified'}
            
            # Calculate market impact metrics
            impact_scores = []
            for trade in large_trades:
                impact = self._calculate_trade_impact(trade, trade_history)
                if impact is not None:
                    impact_scores.append(impact)
            
            if not impact_scores:
                return {'market_impact_score': 0.0, 'impact_analysis': 'Could not calculate impact'}
            
            avg_impact = statistics.mean(impact_scores)
            
            # Low impact despite large size = sophisticated execution (institutional)
            institutional_sophistication = 1.0 - avg_impact  # Inverse relationship
            
            return {
                'market_impact_score': institutional_sophistication,
                'avg_large_trade_impact': avg_impact,
                'large_trades_count': len(large_trades),
                'impact_scores': impact_scores[:5]  # Sample for debugging
            }
            
        except Exception as e:
            logger.error(f"❌ Error in market impact analysis: {str(e)}")
            return {'error': str(e)}
    
    def _identify_large_trades(self, trade_history: List[Dict]) -> List[Dict]:
        """Identify large trades from trade history"""
        
        if not trade_history:
            return []
        
        # Extract trade sizes
        trade_sizes = []
        for trade in trade_history:
            if 'size' in trade:
                try:
                    size = float(trade['size'])
                    trade_sizes.append(size)
                except (ValueError, TypeError):
                    continue
        
        if len(trade_sizes) < 10:
            return []
        
        # Define large trade threshold (top 10%)
        large_threshold = np.percentile(trade_sizes, 90)
        
        # Filter large trades
        large_trades = []
        for trade in trade_history:
            if 'size' in trade:
                try:
                    size = float(trade['size'])
                    if size >= large_threshold:
                        large_trades.append({**trade, 'size': size})
                except (ValueError, TypeError):
                    continue
        
        return large_trades
    
    def _calculate_trade_impact(self, large_trade: Dict, all_trades: List[Dict]) -> Optional[float]:
        """Calculate the market impact of a large trade"""
        
        try:
            # This is a simplified impact calculation
            # In production, you'd want more sophisticated TWAP/VWAP analysis
            
            trade_timestamp = large_trade.get('timestamp', 0)
            trade_price = float(large_trade.get('price', 0))
            
            if trade_price == 0 or trade_timestamp == 0:
                return None
            
            # Find trades before and after
            pre_trades = [t for t in all_trades 
                         if t.get('timestamp', 0) < trade_timestamp and 
                         abs(t.get('timestamp', 0) - trade_timestamp) <= 300]  # 5 minutes before
            
            post_trades = [t for t in all_trades 
                          if t.get('timestamp', 0) > trade_timestamp and 
                          abs(t.get('timestamp', 0) - trade_timestamp) <= 300]  # 5 minutes after
            
            if not pre_trades or not post_trades:
                return None
            
            # Calculate average prices before and after
            pre_prices = [float(t['price']) for t in pre_trades if 'price' in t]
            post_prices = [float(t['price']) for t in post_trades if 'price' in t]
            
            if not pre_prices or not post_prices:
                return None
            
            avg_pre_price = statistics.mean(pre_prices)
            avg_post_price = statistics.mean(post_prices)
            
            # Calculate impact as price change relative to trade price
            impact = abs(avg_post_price - avg_pre_price) / trade_price
            
            return min(1.0, impact)  # Cap at 100% impact
            
        except Exception as e:
            logger.error(f"❌ Error calculating trade impact: {str(e)}")
            return None
    
    async def _synthesize_smart_money_signals(self, coin: str, order_size_analysis: Dict,
                                            temporal_analysis: Dict, coordination_analysis: Dict,
                                            price_level_analysis: Dict, iceberg_analysis: Dict,
                                            market_impact_analysis: Dict) -> Dict[str, Any]:
        """Synthesize all analyses into final smart money assessment"""
        
        try:
            # Initialize scoring
            institutional_score = 0.0
            retail_score = 0.0
            confidence_factors = []
            reasons = []
            
            # Factor 1: Order Size Distribution (Weight: 30%)
            if 'error' not in order_size_analysis:
                large_volume_concentration = order_size_analysis.get('large_volume_concentration', 0)
                whale_volume_concentration = order_size_analysis.get('whale_volume_concentration', 0)
                institutional_side_bias = order_size_analysis.get('institutional_side_bias', 'neutral')
                
                # High concentration of large orders = institutional activity
                institutional_contribution = (large_volume_concentration + whale_volume_concentration) / 2
                institutional_score += institutional_contribution * 0.3
                confidence_factors.append(institutional_contribution)
                
                if large_volume_concentration > 0.3:
                    reasons.append(f"Large order concentration: {large_volume_concentration:.1%} of volume")
                
                if whale_volume_concentration > 0.1:
                    reasons.append(f"Whale activity detected: {whale_volume_concentration:.1%} of volume")
                
                if institutional_side_bias != 'neutral':
                    reasons.append(f"Institutional bias: {institutional_side_bias} from large orders")
            
            # Factor 2: Temporal Patterns (Weight: 15%)
            if 'error' not in temporal_analysis:
                institutional_time_pref = temporal_analysis.get('institutional_time_preference', 0.5)
                weekend_adjustment = temporal_analysis.get('weekend_adjustment', 1.0)
                
                temporal_contribution = institutional_time_pref * weekend_adjustment
                institutional_score += temporal_contribution * 0.15
                confidence_factors.append(temporal_contribution)
                
                if institutional_time_pref > 0.7:
                    session = temporal_analysis.get('current_session', 'unknown')
                    reasons.append(f"High institutional time preference: {session}")
            
            # Factor 3: Cross-Exchange Coordination (Weight: 25%)
            if 'error' not in coordination_analysis:
                coordination_score = coordination_analysis.get('coordination_score', 0)
                coordinated_direction = coordination_analysis.get('coordinated_direction', 'neutral')
                
                institutional_score += coordination_score * 0.25
                confidence_factors.append(coordination_score)
                
                if coordination_score > 0.6:
                    coordinated_exchanges = coordination_analysis.get('coordinated_exchanges', [])
                    reasons.append(f"Cross-exchange coordination detected: {len(coordinated_exchanges)} exchanges")
                    
                    if coordinated_direction != 'neutral':
                        reasons.append(f"Coordinated {coordinated_direction} flow across exchanges")
            
            # Factor 4: Price Level Preferences (Weight: 10%)
            if 'error' not in price_level_analysis:
                round_number_pref = price_level_analysis.get('round_number_preference_score', 0)
                
                institutional_score += round_number_pref * 0.1
                confidence_factors.append(round_number_pref)
                
                if round_number_pref > 0.5:
                    reasons.append(f"Strong round number preference: {round_number_pref:.1%}")
            
            # Factor 5: Iceberg Detection (Weight: 10%)
            if 'error' not in iceberg_analysis:
                iceberg_detected = iceberg_analysis.get('iceberg_detected', False)
                iceberg_confidence = iceberg_analysis.get('confidence', 0)
                iceberg_direction = iceberg_analysis.get('direction', 'neutral')
                
                if iceberg_detected:
                    institutional_score += iceberg_confidence * 0.1
                    confidence_factors.append(iceberg_confidence)
                    reasons.append(f"Iceberg orders detected: {iceberg_direction} bias")
            
            # Factor 6: Market Impact Analysis (Weight: 10%)
            if 'error' not in market_impact_analysis:
                impact_score = market_impact_analysis.get('market_impact_score', 0)
                
                institutional_score += impact_score * 0.1
                confidence_factors.append(impact_score)
                
                if impact_score > 0.6:
                    reasons.append("Sophisticated execution detected (low market impact)")
            
            # Calculate retail score (inverse of institutional for some factors)
            retail_score = max(0, 1.0 - institutional_score)
            
            # Determine dominant signal
            institutional_signal = 'neutral'
            retail_signal = 'neutral'
            
            if institutional_score > 0.6:
                # Determine institutional direction
                if coordination_analysis.get('coordinated_direction') == 'bullish':
                    institutional_signal = 'bullish'
                elif coordination_analysis.get('coordinated_direction') == 'bearish':
                    institutional_signal = 'bearish'
                elif order_size_analysis.get('institutional_side_bias') != 'neutral':
                    institutional_signal = order_size_analysis.get('institutional_side_bias')
            
            if retail_score > 0.6:
                # Retail typically follows trends or sentiment
                retail_signal = 'bullish' if institutional_signal == 'bullish' else 'bearish'
            
            # Calculate overall confidence
            smart_money_confidence = statistics.mean(confidence_factors) if confidence_factors else 0.0
            
            # Institutional activity level
            institutional_activity_level = institutional_score
            
            return {
                'institutional_signal': institutional_signal,
                'retail_signal': retail_signal,
                'smart_money_confidence': smart_money_confidence,
                'institutional_activity_level': institutional_activity_level,
                'flow_analysis': {
                    'institutional_score': institutional_score,
                    'retail_score': retail_score,
                    'confidence_factors': confidence_factors
                },
                'reasons': reasons[:5],  # Top 5 reasons
                'raw_data': {
                    'order_size_analysis': order_size_analysis,
                    'temporal_analysis': temporal_analysis,
                    'coordination_analysis': coordination_analysis,
                    'price_level_analysis': price_level_analysis,
                    'iceberg_analysis': iceberg_analysis,
                    'market_impact_analysis': market_impact_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error synthesizing smart money signals: {str(e)}")
            return {
                'institutional_signal': 'neutral',
                'retail_signal': 'neutral',
                'smart_money_confidence': 0.0,
                'institutional_activity_level': 0.0,
                'flow_analysis': {},
                'reasons': [f'Signal synthesis error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }