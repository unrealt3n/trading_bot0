"""
🐋 Whale Alert Analyzer
Tracks large cryptocurrency transactions using WhaleAlert API
Provides immediate high-impact trading signals based on whale movements
"""

import aiohttp
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WhaleTransaction:
    """Whale transaction data structure"""
    transaction_id: str
    blockchain: str
    symbol: str
    amount: float
    amount_usd: float
    from_address: str
    to_address: str
    timestamp: datetime
    transaction_type: str  # 'transfer', 'exchange_deposit', 'exchange_withdrawal'
    exchange_name: Optional[str] = None
    
class WhaleAlertAnalyzer:
    """Analyzes whale movements for trading signals"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.whale-alert.io/v1"
        
        # Thresholds (in USD)
        self.whale_thresholds = {
            'BTC': 1_000_000,    # $1M+
            'ETH': 500_000,      # $500K+
            'SOL': 100_000,      # $100K+
            'XRP': 100_000,      # $100K+
            'BNB': 100_000,      # $100K+
            'default': 50_000    # $50K+ for other coins
        }
        
        # Signal strength multipliers
        self.signal_multipliers = {
            'exchange_deposit': -1.0,    # Bearish (selling pressure)
            'exchange_withdrawal': 1.0,   # Bullish (accumulation)
            'whale_to_whale': 0.5,       # Neutral to slightly bullish
            'unknown': 0.3               # Low impact
        }
        
        # Cache for recent transactions
        self.recent_transactions: Dict[str, List[WhaleTransaction]] = {}
        self.cache_duration = timedelta(minutes=15)
        
        logger.info("🐋 Whale Alert Analyzer initialized")
    
    async def analyze_whale_activity(self, coin: str, lookback_minutes: int = 60) -> Dict[str, Any]:
        """
        Analyze recent whale activity for a specific coin
        
        Returns:
        {
            'signal_strength': 0-100,
            'bias': 'bullish'/'bearish'/'neutral',
            'confidence': 0.0-1.0,
            'transactions': [...],
            'summary': {...}
        }
        """
        
        try:
            logger.info(f"🐋 Analyzing whale activity for {coin} (last {lookback_minutes}min)")
            
            # Get recent transactions
            transactions = await self.get_recent_transactions(coin, lookback_minutes)
            
            if not transactions:
                return self._create_neutral_signal("No recent whale activity")
            
            # Calculate signal strength
            signal_data = self._calculate_whale_signals(coin, transactions)
            
            # Generate trading signal
            trading_signal = self._generate_trading_signal(coin, signal_data, transactions)
            
            logger.info(f"🐋 {coin} whale signal: {trading_signal['bias'].upper()} "
                       f"({trading_signal['signal_strength']:.1f}/100)")
            
            return trading_signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing whale activity for {coin}: {e}")
            return self._create_neutral_signal(f"Analysis error: {str(e)}")
    
    async def get_recent_transactions(self, coin: str, lookback_minutes: int) -> List[WhaleTransaction]:
        """Get recent whale transactions for a coin"""
        
        # Check cache first
        cache_key = f"{coin}_{lookback_minutes}"
        if cache_key in self.recent_transactions:
            cached_time = self.recent_transactions[cache_key][0].timestamp if self.recent_transactions[cache_key] else datetime.now()
            if datetime.now() - cached_time < self.cache_duration:
                return self.recent_transactions[cache_key]
        
        transactions = []
        
        if self.api_key:
            # Use real WhaleAlert API
            transactions = await self._fetch_from_whale_alert_api(coin, lookback_minutes)
        else:
            # Use alternative free sources
            transactions = await self._fetch_from_alternative_sources(coin, lookback_minutes)
        
        # Cache results
        self.recent_transactions[cache_key] = transactions
        
        return transactions
    
    async def _fetch_from_whale_alert_api(self, coin: str, lookback_minutes: int) -> List[WhaleTransaction]:
        """Fetch from official WhaleAlert API (requires API key)"""
        
        try:
            # Calculate time range
            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(minutes=lookback_minutes)).timestamp())
            
            url = f"{self.base_url}/transactions"
            params = {
                'api_key': self.api_key,
                'start': start_time,
                'end': end_time,
                'symbol': coin,
                'min_value': self.whale_thresholds.get(coin, self.whale_thresholds['default'])
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_whale_alert_response(data)
                    else:
                        logger.warning(f"⚠️ WhaleAlert API error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"❌ Error fetching from WhaleAlert API: {e}")
            return []
    
    async def _fetch_from_alternative_sources(self, coin: str, lookback_minutes: int) -> List[WhaleTransaction]:
        """Fetch from alternative free sources when no API key"""
        
        transactions = []
        
        try:
            # Try to scrape from public whale trackers
            transactions.extend(await self._scrape_whale_tracker_sites(coin))
            
            # Try blockchain explorers for large transactions
            transactions.extend(await self._check_blockchain_explorers(coin, lookback_minutes))
            
        except Exception as e:
            logger.error(f"❌ Error fetching from alternative sources: {e}")
        
        return transactions
    
    async def _scrape_whale_tracker_sites(self, coin: str) -> List[WhaleTransaction]:
        """Scrape public whale tracking websites"""
        
        transactions = []
        
        try:
            # Example: Scrape from public whale alert websites
            # This is a simplified implementation - real scraping would be more complex
            
            urls = [
                f"https://whale-alert.io/transaction/bitcoin/recent",
                f"https://clankapp.com/transactions",
                # Add more public whale tracking sites
            ]
            
            async with aiohttp.ClientSession() as session:
                for url in urls:
                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                # Parse HTML/JSON response
                                # This would need proper HTML parsing
                                text = await response.text()
                                # Simplified parsing - would need BeautifulSoup in real implementation
                                parsed_transactions = self._parse_scraped_data(text, coin)
                                transactions.extend(parsed_transactions)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to scrape {url}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"❌ Error scraping whale tracker sites: {e}")
        
        return transactions
    
    async def _check_blockchain_explorers(self, coin: str, lookback_minutes: int) -> List[WhaleTransaction]:
        """Check blockchain explorers for large transactions"""
        
        transactions = []
        
        try:
            # Blockchain explorer APIs (free tier)
            explorers = {
                'BTC': 'https://blockstream.info/api/address/{}/txs/mempool',
                'ETH': 'https://api.etherscan.io/api',
                'SOL': 'https://api.solscan.io/transaction',
                # Add more as needed
            }
            
            if coin in explorers:
                # This is a simplified implementation
                # Real implementation would need to:
                # 1. Get recent blocks
                # 2. Parse transactions
                # 3. Filter by value threshold
                # 4. Identify exchange addresses
                pass
        
        except Exception as e:
            logger.error(f"❌ Error checking blockchain explorers: {e}")
        
        return transactions
    
    def _parse_scraped_data(self, html_content: str, coin: str) -> List[WhaleTransaction]:
        """Parse scraped HTML data into whale transactions"""
        
        # This is a placeholder - real implementation would use BeautifulSoup
        # to parse HTML and extract transaction data
        
        # For demo purposes, create some sample transactions
        if "large transaction" in html_content.lower():
            return [
                WhaleTransaction(
                    transaction_id="demo_tx_001",
                    blockchain=coin.lower(),
                    symbol=coin,
                    amount=1000.0,
                    amount_usd=50_000.0,
                    from_address="unknown",
                    to_address="binance",
                    timestamp=datetime.now() - timedelta(minutes=30),
                    transaction_type="exchange_deposit"
                )
            ]
        
        return []
    
    def _parse_whale_alert_response(self, data: Dict) -> List[WhaleTransaction]:
        """Parse WhaleAlert API response"""
        
        transactions = []
        
        try:
            for tx_data in data.get('transactions', []):
                transaction = WhaleTransaction(
                    transaction_id=tx_data.get('hash', ''),
                    blockchain=tx_data.get('blockchain', ''),
                    symbol=tx_data.get('symbol', ''),
                    amount=float(tx_data.get('amount', 0)),
                    amount_usd=float(tx_data.get('amount_usd', 0)),
                    from_address=tx_data.get('from', {}).get('address', ''),
                    to_address=tx_data.get('to', {}).get('address', ''),
                    timestamp=datetime.fromtimestamp(tx_data.get('timestamp', 0)),
                    transaction_type=tx_data.get('transaction_type', 'unknown'),
                    exchange_name=tx_data.get('from', {}).get('owner', '') or tx_data.get('to', {}).get('owner', '')
                )
                transactions.append(transaction)
        
        except Exception as e:
            logger.error(f"❌ Error parsing WhaleAlert response: {e}")
        
        return transactions
    
    def _calculate_whale_signals(self, coin: str, transactions: List[WhaleTransaction]) -> Dict[str, Any]:
        """Calculate trading signals from whale transactions"""
        
        total_inflow = 0.0  # Money flowing into exchanges (bearish)
        total_outflow = 0.0  # Money flowing out of exchanges (bullish)
        total_volume = 0.0
        transaction_count = len(transactions)
        
        # Analyze each transaction
        for tx in transactions:
            volume_usd = tx.amount_usd
            total_volume += volume_usd
            
            if tx.transaction_type == 'exchange_deposit':
                total_inflow += volume_usd
            elif tx.transaction_type == 'exchange_withdrawal':
                total_outflow += volume_usd
        
        # Calculate net flow
        net_flow = total_outflow - total_inflow
        
        # Calculate signal metrics
        if total_volume > 0:
            flow_ratio = net_flow / total_volume  # -1 to 1
            inflow_ratio = total_inflow / total_volume
            outflow_ratio = total_outflow / total_volume
        else:
            flow_ratio = 0.0
            inflow_ratio = 0.0
            outflow_ratio = 0.0
        
        return {
            'total_inflow_usd': total_inflow,
            'total_outflow_usd': total_outflow,
            'net_flow_usd': net_flow,
            'total_volume_usd': total_volume,
            'flow_ratio': flow_ratio,
            'inflow_ratio': inflow_ratio,
            'outflow_ratio': outflow_ratio,
            'transaction_count': transaction_count,
            'avg_transaction_size': total_volume / max(transaction_count, 1)
        }
    
    def _generate_trading_signal(self, coin: str, signal_data: Dict, transactions: List[WhaleTransaction]) -> Dict[str, Any]:
        """Generate final trading signal from whale analysis"""
        
        # Base signal strength from volume
        volume_score = min(signal_data['total_volume_usd'] / 1_000_000, 100)  # Cap at $1M = 100 points
        
        # Flow direction score
        flow_ratio = signal_data['flow_ratio']
        if flow_ratio > 0.3:  # Strong outflow (bullish)
            bias = 'bullish'
            flow_score = min(flow_ratio * 100, 50)
        elif flow_ratio < -0.3:  # Strong inflow (bearish)
            bias = 'bearish'
            flow_score = min(abs(flow_ratio) * 100, 50)
        else:  # Neutral
            bias = 'neutral'
            flow_score = 20
        
        # Transaction count bonus
        count_bonus = min(signal_data['transaction_count'] * 5, 20)
        
        # Calculate final signal strength
        signal_strength = min(volume_score + flow_score + count_bonus, 100)
        
        # Calculate confidence
        confidence = signal_strength / 100.0
        
        # Adjust for recency (more recent = higher confidence)
        if transactions:
            latest_tx = max(transactions, key=lambda x: x.timestamp)
            minutes_ago = (datetime.now() - latest_tx.timestamp).total_seconds() / 60
            recency_multiplier = max(0.5, 1.0 - (minutes_ago / 60))  # Decay over 1 hour
            confidence *= recency_multiplier
        
        # Create summary
        summary = {
            'largest_transaction_usd': max([tx.amount_usd for tx in transactions], default=0),
            'most_recent_transaction': max([tx.timestamp for tx in transactions], default=datetime.now()),
            'dominant_flow': 'inflow' if signal_data['inflow_ratio'] > signal_data['outflow_ratio'] else 'outflow',
            'exchange_deposits': len([tx for tx in transactions if tx.transaction_type == 'exchange_deposit']),
            'exchange_withdrawals': len([tx for tx in transactions if tx.transaction_type == 'exchange_withdrawal'])
        }
        
        return {
            'signal_strength': signal_strength,
            'bias': bias,
            'confidence': confidence,
            'transactions': transactions,
            'signal_data': signal_data,
            'summary': summary,
            'reasons': self._generate_reasons(bias, signal_data, summary)
        }
    
    def _generate_reasons(self, bias: str, signal_data: Dict, summary: Dict) -> List[str]:
        """Generate human-readable reasons for the signal"""
        
        reasons = []
        
        # Volume reasons
        if signal_data['total_volume_usd'] > 1_000_000:
            reasons.append(f"Large whale volume: ${signal_data['total_volume_usd']:,.0f}")
        
        # Flow reasons
        if bias == 'bullish':
            reasons.append(f"Net outflow from exchanges: ${signal_data['net_flow_usd']:,.0f}")
            if summary['exchange_withdrawals'] > summary['exchange_deposits']:
                reasons.append(f"More withdrawals ({summary['exchange_withdrawals']}) than deposits ({summary['exchange_deposits']})")
        elif bias == 'bearish':
            reasons.append(f"Net inflow to exchanges: ${abs(signal_data['net_flow_usd']):,.0f}")
            if summary['exchange_deposits'] > summary['exchange_withdrawals']:
                reasons.append(f"More deposits ({summary['exchange_deposits']}) than withdrawals ({summary['exchange_withdrawals']})")
        
        # Recency reasons
        latest_tx = summary.get('most_recent_transaction')
        if latest_tx and isinstance(latest_tx, datetime):
            minutes_ago = (datetime.now() - latest_tx).total_seconds() / 60
            if minutes_ago < 30:
                reasons.append(f"Recent whale activity ({minutes_ago:.0f}min ago)")
        
        return reasons[:3]  # Return top 3 reasons
    
    def _create_neutral_signal(self, reason: str) -> Dict[str, Any]:
        """Create neutral signal when no whale activity detected"""
        
        return {
            'signal_strength': 0.0,
            'bias': 'neutral',
            'confidence': 0.0,
            'transactions': [],
            'signal_data': {},
            'summary': {'reason': reason},
            'reasons': [reason]
        }
    
    async def get_whale_alert_summary(self, coins: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get whale alert summary for multiple coins"""
        
        results = {}
        
        # Analyze all coins concurrently
        tasks = [self.analyze_whale_activity(coin) for coin in coins]
        analyses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for coin, analysis in zip(coins, analyses):
            if isinstance(analysis, Exception):
                results[coin] = self._create_neutral_signal(f"Error: {str(analysis)}")
            else:
                results[coin] = analysis
        
        return results