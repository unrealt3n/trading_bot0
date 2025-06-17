"""
📈 Historical Correlation Engine
Analyzes historical patterns between news, whale alerts, and liquidations to improve signal accuracy
"""

import asyncio
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import aiofiles
from pathlib import Path
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class HistoricalEvent:
    """Historical market event data structure"""
    timestamp: datetime
    coin: str
    event_type: str  # 'whale', 'liquidation', 'news'
    event_data: Dict[str, Any]
    price_before: float
    price_after_1h: float
    price_after_4h: float
    price_after_24h: float
    volume_impact: float
    
class HistoricalCorrelationEngine:
    """Analyzes historical patterns to improve prediction accuracy"""
    
    def __init__(self, db_path: str = "logs/historical_events.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Correlation patterns storage
        self.correlation_patterns = {
            'whale_price_impact': {},      # Whale alert → price movement patterns
            'liquidation_momentum': {},    # Liquidation → momentum patterns
            'news_sentiment_accuracy': {}, # News → actual price movement accuracy
            'combined_signals': {}         # Multi-signal correlation patterns
        }
        
        # Pattern recognition parameters
        self.lookback_days = 30
        self.min_events_for_pattern = 5
        self.correlation_threshold = 0.6
        
        # Initialize database
        asyncio.create_task(self._init_database())
        
        logger.info("📈 Historical Correlation Engine initialized")
    
    async def _init_database(self):
        """Initialize SQLite database for historical events"""
        try:
            async with aiofiles.open(self.db_path, 'w') as f:
                pass  # Create file if not exists
            
            # Create tables
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    price_before REAL NOT NULL,
                    price_after_1h REAL,
                    price_after_4h REAL,
                    price_after_24h REAL,
                    volume_impact REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS correlation_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    accuracy_score REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_coin_time 
                ON historical_events(coin, timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_patterns_coin_type 
                ON correlation_patterns(coin, pattern_type)
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("📈 Historical events database initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing database: {e}")
    
    async def record_event(self, coin: str, event_type: str, event_data: Dict[str, Any], 
                          current_price: float) -> bool:
        """Record a new market event for historical analysis"""
        
        try:
            event = HistoricalEvent(
                timestamp=datetime.now(),
                coin=coin,
                event_type=event_type,
                event_data=event_data,
                price_before=current_price,
                price_after_1h=0.0,  # Will be updated later
                price_after_4h=0.0,  # Will be updated later
                price_after_24h=0.0, # Will be updated later
                volume_impact=event_data.get('volume_impact', 0.0)
            )
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO historical_events 
                (timestamp, coin, event_type, event_data, price_before, volume_impact)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp.isoformat(),
                event.coin,
                event.event_type,
                json.dumps(event.event_data),
                event.price_before,
                event.volume_impact
            ))
            
            conn.commit()
            conn.close()
            
            # Schedule price updates
            asyncio.create_task(self._schedule_price_updates(event))
            
            logger.info(f"📈 Recorded {event_type} event for {coin} at ${current_price:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error recording event: {e}")
            return False
    
    async def _schedule_price_updates(self, event: HistoricalEvent):
        """Schedule price updates for historical event tracking"""
        
        try:
            # Update prices at 1h, 4h, and 24h intervals
            update_times = [
                (timedelta(hours=1), 'price_after_1h'),
                (timedelta(hours=4), 'price_after_4h'),
                (timedelta(hours=24), 'price_after_24h')
            ]
            
            for delay, column in update_times:
                asyncio.create_task(self._update_price_after_delay(event, delay, column))
                
        except Exception as e:
            logger.error(f"❌ Error scheduling price updates: {e}")
    
    async def _update_price_after_delay(self, event: HistoricalEvent, delay: timedelta, column: str):
        """Update price after specified delay"""
        
        try:
            # Wait for the specified delay
            await asyncio.sleep(delay.total_seconds())
            
            # Get current price (this would need actual price fetching)
            # For now, simulate with the initial price
            current_price = event.price_before  # Placeholder
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
                UPDATE historical_events 
                SET {column} = ? 
                WHERE coin = ? AND timestamp = ? AND event_type = ?
            ''', (
                current_price,
                event.coin,
                event.timestamp.isoformat(),
                event.event_type
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"📈 Updated {column} for {event.coin} {event.event_type} event")
            
        except Exception as e:
            logger.error(f"❌ Error updating price after delay: {e}")
    
    async def analyze_historical_patterns(self, coin: str) -> Dict[str, Any]:
        """Analyze historical patterns for a specific coin"""
        
        try:
            logger.info(f"📈 Analyzing historical patterns for {coin}")
            
            # Get recent historical events
            events = await self._get_recent_events(coin, self.lookback_days)
            
            if len(events) < self.min_events_for_pattern:
                return self._create_no_pattern_response(f"Insufficient data ({len(events)} events)")
            
            # Analyze different pattern types
            patterns = {}
            
            # 1. Whale alert patterns
            whale_patterns = await self._analyze_whale_patterns(events)
            patterns['whale_alerts'] = whale_patterns
            
            # 2. Liquidation patterns
            liquidation_patterns = await self._analyze_liquidation_patterns(events)
            patterns['liquidations'] = liquidation_patterns
            
            # 3. News patterns
            news_patterns = await self._analyze_news_patterns(events)
            patterns['news'] = news_patterns
            
            # 4. Combined signal patterns
            combined_patterns = await self._analyze_combined_patterns(events)
            patterns['combined_signals'] = combined_patterns
            
            # Calculate overall pattern strength
            pattern_strength = self._calculate_pattern_strength(patterns)
            
            # Generate recommendations
            recommendations = self._generate_pattern_recommendations(patterns, pattern_strength)
            
            result = {
                'coin': coin,
                'pattern_strength': pattern_strength,
                'patterns': patterns,
                'recommendations': recommendations,
                'events_analyzed': len(events),
                'analysis_period_days': self.lookback_days,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache patterns for future use
            await self._cache_patterns(coin, result)
            
            logger.info(f"📈 {coin} pattern analysis complete: {pattern_strength:.1%} strength")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing historical patterns for {coin}: {e}")
            return self._create_no_pattern_response(f"Analysis error: {str(e)}")
    
    async def _get_recent_events(self, coin: str, days: int) -> List[HistoricalEvent]:
        """Get recent historical events from database"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, coin, event_type, event_data, price_before, 
                       price_after_1h, price_after_4h, price_after_24h, volume_impact
                FROM historical_events
                WHERE coin = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (coin, cutoff_date.isoformat()))
            
            rows = cursor.fetchall()
            conn.close()
            
            events = []
            for row in rows:
                event = HistoricalEvent(
                    timestamp=datetime.fromisoformat(row[0]),
                    coin=row[1],
                    event_type=row[2],
                    event_data=json.loads(row[3]),
                    price_before=row[4],
                    price_after_1h=row[5] or 0.0,
                    price_after_4h=row[6] or 0.0,
                    price_after_24h=row[7] or 0.0,
                    volume_impact=row[8] or 0.0
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"❌ Error getting recent events: {e}")
            return []
    
    async def _analyze_whale_patterns(self, events: List[HistoricalEvent]) -> Dict[str, Any]:
        """Analyze whale alert patterns"""
        
        whale_events = [e for e in events if e.event_type == 'whale']
        
        if len(whale_events) < 3:
            return {'accuracy': 0.0, 'confidence': 0.0, 'patterns': []}
        
        try:
            patterns = []
            
            # Analyze price impact by whale transaction size
            size_impact_correlation = self._calculate_size_impact_correlation(whale_events)
            patterns.append({
                'type': 'size_impact',
                'correlation': size_impact_correlation,
                'description': f"Whale size vs price impact correlation: {size_impact_correlation:.2f}"
            })
            
            # Analyze time-based patterns
            time_patterns = self._analyze_time_patterns(whale_events)
            patterns.extend(time_patterns)
            
            # Calculate overall accuracy
            accuracy = self._calculate_prediction_accuracy(whale_events)
            confidence = min(len(whale_events) / 10.0, 1.0)  # More events = higher confidence
            
            return {
                'accuracy': accuracy,
                'confidence': confidence,
                'patterns': patterns,
                'total_events': len(whale_events)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing whale patterns: {e}")
            return {'accuracy': 0.0, 'confidence': 0.0, 'patterns': []}
    
    async def _analyze_liquidation_patterns(self, events: List[HistoricalEvent]) -> Dict[str, Any]:
        """Analyze liquidation patterns"""
        
        liquidation_events = [e for e in events if e.event_type == 'liquidation']
        
        if len(liquidation_events) < 3:
            return {'accuracy': 0.0, 'confidence': 0.0, 'patterns': []}
        
        try:
            patterns = []
            
            # Analyze liquidation cascade patterns
            cascade_patterns = self._analyze_cascade_patterns(liquidation_events)
            patterns.extend(cascade_patterns)
            
            # Analyze directional momentum
            momentum_patterns = self._analyze_momentum_patterns(liquidation_events)
            patterns.extend(momentum_patterns)
            
            accuracy = self._calculate_prediction_accuracy(liquidation_events)
            confidence = min(len(liquidation_events) / 8.0, 1.0)
            
            return {
                'accuracy': accuracy,
                'confidence': confidence,
                'patterns': patterns,
                'total_events': len(liquidation_events)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing liquidation patterns: {e}")
            return {'accuracy': 0.0, 'confidence': 0.0, 'patterns': []}
    
    async def _analyze_news_patterns(self, events: List[HistoricalEvent]) -> Dict[str, Any]:
        """Analyze news sentiment patterns"""
        
        news_events = [e for e in events if e.event_type == 'news']
        
        if len(news_events) < 3:
            return {'accuracy': 0.0, 'confidence': 0.0, 'patterns': []}
        
        try:
            patterns = []
            
            # Analyze sentiment vs actual price movement
            sentiment_accuracy = self._analyze_sentiment_accuracy(news_events)
            patterns.append({
                'type': 'sentiment_accuracy',
                'accuracy': sentiment_accuracy,
                'description': f"News sentiment prediction accuracy: {sentiment_accuracy:.1%}"
            })
            
            accuracy = sentiment_accuracy
            confidence = min(len(news_events) / 6.0, 1.0)
            
            return {
                'accuracy': accuracy,
                'confidence': confidence,
                'patterns': patterns,
                'total_events': len(news_events)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing news patterns: {e}")
            return {'accuracy': 0.0, 'confidence': 0.0, 'patterns': []}
    
    async def _analyze_combined_patterns(self, events: List[HistoricalEvent]) -> Dict[str, Any]:
        """Analyze patterns when multiple signals occur together"""
        
        try:
            # Group events by time windows to find coinciding signals
            time_windows = self._group_events_by_time_windows(events, window_minutes=60)
            
            combined_events = [window for window in time_windows if len(window) > 1]
            
            if len(combined_events) < 2:
                return {'accuracy': 0.0, 'confidence': 0.0, 'patterns': []}
            
            patterns = []
            
            # Analyze multi-signal accuracy
            multi_signal_accuracy = self._calculate_multi_signal_accuracy(combined_events)
            patterns.append({
                'type': 'multi_signal',
                'accuracy': multi_signal_accuracy,
                'description': f"Multi-signal accuracy: {multi_signal_accuracy:.1%}"
            })
            
            # Analyze signal combination types
            combination_patterns = self._analyze_signal_combinations(combined_events)
            patterns.extend(combination_patterns)
            
            return {
                'accuracy': multi_signal_accuracy,
                'confidence': min(len(combined_events) / 5.0, 1.0),
                'patterns': patterns,
                'total_events': len(combined_events)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing combined patterns: {e}")
            return {'accuracy': 0.0, 'confidence': 0.0, 'patterns': []}
    
    def _calculate_size_impact_correlation(self, whale_events: List[HistoricalEvent]) -> float:
        """Calculate correlation between whale transaction size and price impact"""
        
        try:
            if len(whale_events) < 3:
                return 0.0
            
            sizes = []
            impacts = []
            
            for event in whale_events:
                size = event.event_data.get('amount_usd', 0)
                if size > 0 and event.price_after_1h > 0:
                    price_impact = abs(event.price_after_1h - event.price_before) / event.price_before
                    sizes.append(size)
                    impacts.append(price_impact)
            
            if len(sizes) < 3:
                return 0.0
            
            # Calculate Pearson correlation
            correlation = np.corrcoef(sizes, impacts)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_prediction_accuracy(self, events: List[HistoricalEvent]) -> float:
        """Calculate prediction accuracy for events"""
        
        try:
            if not events:
                return 0.0
            
            correct_predictions = 0
            total_predictions = 0
            
            for event in events:
                if event.price_after_1h <= 0:
                    continue
                
                predicted_direction = event.event_data.get('predicted_direction', 'neutral')
                actual_change = (event.price_after_1h - event.price_before) / event.price_before
                
                if predicted_direction == 'bullish' and actual_change > 0.001:  # >0.1% move
                    correct_predictions += 1
                elif predicted_direction == 'bearish' and actual_change < -0.001:  # <-0.1% move
                    correct_predictions += 1
                elif predicted_direction == 'neutral' and abs(actual_change) < 0.001:
                    correct_predictions += 1
                
                total_predictions += 1
            
            return correct_predictions / max(total_predictions, 1)
            
        except Exception:
            return 0.0
    
    def _create_no_pattern_response(self, reason: str) -> Dict[str, Any]:
        """Create response when no patterns found"""
        
        return {
            'pattern_strength': 0.0,
            'patterns': {},
            'recommendations': [],
            'events_analyzed': 0,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
    
    # Placeholder implementations for remaining methods
    def _analyze_time_patterns(self, events: List[HistoricalEvent]) -> List[Dict[str, Any]]:
        """Analyze time-based patterns"""
        return []
    
    def _analyze_cascade_patterns(self, events: List[HistoricalEvent]) -> List[Dict[str, Any]]:
        """Analyze liquidation cascade patterns"""
        return []
    
    def _analyze_momentum_patterns(self, events: List[HistoricalEvent]) -> List[Dict[str, Any]]:
        """Analyze momentum patterns"""
        return []
    
    def _analyze_sentiment_accuracy(self, events: List[HistoricalEvent]) -> float:
        """Analyze news sentiment accuracy"""
        return self._calculate_prediction_accuracy(events)
    
    def _group_events_by_time_windows(self, events: List[HistoricalEvent], window_minutes: int) -> List[List[HistoricalEvent]]:
        """Group events by time windows"""
        # Simplified implementation
        return [[event] for event in events]
    
    def _calculate_multi_signal_accuracy(self, combined_events: List[List[HistoricalEvent]]) -> float:
        """Calculate accuracy when multiple signals occur"""
        return 0.8  # Placeholder
    
    def _analyze_signal_combinations(self, combined_events: List[List[HistoricalEvent]]) -> List[Dict[str, Any]]:
        """Analyze different signal combination patterns"""
        return []
    
    def _calculate_pattern_strength(self, patterns: Dict[str, Any]) -> float:
        """Calculate overall pattern strength"""
        
        total_weight = 0.0
        weighted_accuracy = 0.0
        
        weights = {
            'whale_alerts': 0.3,
            'liquidations': 0.3,
            'news': 0.2,
            'combined_signals': 0.2
        }
        
        for pattern_type, pattern_data in patterns.items():
            if pattern_type in weights:
                accuracy = pattern_data.get('accuracy', 0.0)
                confidence = pattern_data.get('confidence', 0.0)
                weight = weights[pattern_type]
                
                weighted_accuracy += accuracy * confidence * weight
                total_weight += weight * confidence
        
        return weighted_accuracy / max(total_weight, 0.1)
    
    def _generate_pattern_recommendations(self, patterns: Dict[str, Any], pattern_strength: float) -> List[str]:
        """Generate recommendations based on patterns"""
        
        recommendations = []
        
        if pattern_strength > 0.7:
            recommendations.append("Strong historical patterns detected - high confidence signals")
        elif pattern_strength > 0.5:
            recommendations.append("Moderate historical patterns - medium confidence signals")
        else:
            recommendations.append("Weak historical patterns - use additional confirmation")
        
        # Add specific pattern recommendations
        for pattern_type, pattern_data in patterns.items():
            accuracy = pattern_data.get('accuracy', 0.0)
            if accuracy > 0.8:
                recommendations.append(f"{pattern_type.replace('_', ' ').title()} shows high accuracy ({accuracy:.1%})")
        
        return recommendations[:3]  # Top 3 recommendations
    
    async def _cache_patterns(self, coin: str, result: Dict[str, Any]):
        """Cache pattern analysis results"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Remove old patterns for this coin
            cursor.execute('DELETE FROM correlation_patterns WHERE coin = ?', (coin,))
            
            # Insert new patterns
            cursor.execute('''
                INSERT INTO correlation_patterns 
                (pattern_type, coin, pattern_data, accuracy_score, confidence_level)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                'comprehensive',
                coin,
                json.dumps(result),
                result.get('pattern_strength', 0.0),
                1.0
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error caching patterns: {e}")
    
    async def get_pattern_boost(self, coin: str, signal_type: str, signal_data: Dict[str, Any]) -> float:
        """Get pattern-based confidence boost for a signal"""
        
        try:
            # Get cached patterns
            patterns = await self.analyze_historical_patterns(coin)
            
            pattern_data = patterns.get('patterns', {}).get(signal_type, {})
            accuracy = pattern_data.get('accuracy', 0.0)
            confidence = pattern_data.get('confidence', 0.0)
            
            # Calculate boost based on historical accuracy
            if accuracy > 0.8 and confidence > 0.7:
                return 0.15  # 15% boost for very reliable patterns
            elif accuracy > 0.6 and confidence > 0.5:
                return 0.10  # 10% boost for reliable patterns
            elif accuracy > 0.4:
                return 0.05  # 5% boost for somewhat reliable patterns
            else:
                return 0.0   # No boost for unreliable patterns
                
        except Exception as e:
            logger.error(f"❌ Error getting pattern boost: {e}")
            return 0.0