"""
📰 News Sentiment Analysis Module
Analyzes sentiment from CryptoPanic, Reddit and other sources
Weight: 10% in signal confidence system
"""

import asyncio
import logging
import re
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .utils import HTTPClient, APIError, safe_api_call, get_emoji
from config.settings import api_config

logger = logging.getLogger(__name__)

# Crypto-specific sentiment keywords
CRYPTO_BULLISH_KEYWORDS = [
    'moon', 'mooning', 'pump', 'pumping', 'bull', 'bullish', 'green', 'rocket', 'diamond hands',
    'hodl', 'buy the dip', 'accumulate', 'breakout', 'rally', 'surge', 'gains', 'profit',
    'adoption', 'institutional', 'partnership', 'upgrade', 'innovation', 'mass adoption'
]

CRYPTO_BEARISH_KEYWORDS = [
    'dump', 'dumping', 'bear', 'bearish', 'red', 'crash', 'collapse', 'panic', 'fear',
    'sell off', 'liquidation', 'correction', 'drop', 'fall', 'decline', 'regulation',
    'ban', 'hack', 'scam', 'rugpull', 'dead cat bounce', 'bubble', 'overvalued'
]

CRYPTO_NEUTRAL_KEYWORDS = [
    'sideways', 'consolidation', 'range', 'stable', 'waiting', 'watching', 'uncertain',
    'analysis', 'technical', 'support', 'resistance', 'volume', 'chart'
]

class NewsAnalyzer:
    """Analyzes news sentiment with focus on crypto-specific language"""
    
    def __init__(self):
        # Load API keys from environment
        self.crypto_panic_token = os.getenv('CRYPTO_PANIC_TOKEN', '')
        
        # API endpoints
        self.crypto_panic_base = "https://cryptopanic.com/api/v1"
        self.reddit_base = "https://www.reddit.com"
        
        # Use CryptoPanic if token available, otherwise fallback to scraping
        self.use_crypto_panic = bool(self.crypto_panic_token)
        self.use_web_scraping = True  # Always enable as fallback
        
        logger.info(f"📰 News Analyzer initialized - CryptoPanic: {'✅' if self.use_crypto_panic else '❌'}")
        
    async def analyze_news_sentiment(self, coin: str) -> Dict[str, Any]:
        """
        Main function to analyze news sentiment for a coin
        
        Returns:
        {
            'bias': 'bullish'|'bearish'|'neutral',
            'confidence': 0.0-1.0,
            'reasons': ['list of reasons'],
            'raw_data': {...}
        }
        """
        sources = ['CryptoPanic API', 'CoinGecko social', 'CryptoCompare news']
        logger.info(f"{get_emoji('news')} Analyzing news sentiment for {coin} - Sources: {', '.join(sources)} (NO MOCK DATA)")
        
        try:
            # Due to API limitations, we'll use multiple approaches
            sentiment_sources = []
            
            # 1. CryptoPanic API (free, real news data)
            cryptopanic_sentiment = await safe_api_call(
                lambda: self._analyze_cryptopanic_sentiment(coin),
                default_return=None,
                error_message=f"CryptoPanic sentiment analysis failed for {coin}"
            )
            if cryptopanic_sentiment:
                sentiment_sources.append(('cryptopanic', cryptopanic_sentiment))
            
            # 2. Alternative: Use CoinGecko social sentiment (if available)
            coingecko_sentiment = await safe_api_call(
                lambda: self._get_coingecko_sentiment(coin),
                default_return=None,
                error_message=f"CoinGecko sentiment failed for {coin}"
            )
            if coingecko_sentiment:
                sentiment_sources.append(('coingecko', coingecko_sentiment))
            
            # 3. News aggregation from crypto news APIs
            news_sentiment = await safe_api_call(
                lambda: self._analyze_crypto_news(coin),
                default_return=None,
                error_message=f"Crypto news analysis failed for {coin}"
            )
            if news_sentiment:
                sentiment_sources.append(('news', news_sentiment))
            
            if not sentiment_sources:
                logger.warning(f"⚠️ No sentiment data available for {coin}")
                return {
                    'bias': 'neutral',
                    'confidence': 0.0,
                    'reasons': ['No sentiment data sources available'],
                    'raw_data': {'error': 'All sentiment sources failed', 'api_limitations': True}
                }
            
            # Combine sentiment from multiple sources
            combined_sentiment = self._combine_sentiment_sources(sentiment_sources)
            
            # Log successful sources
            successful_sources = [source[0] for source in sentiment_sources]
            bias = combined_sentiment.get('bias', 'neutral')
            confidence = combined_sentiment.get('confidence', 0) * 100
            
            logger.info(f"✅ News sentiment analysis complete for {coin} - Sources used: {', '.join(successful_sources)} ({len(sentiment_sources)}/3) | Result: {bias.upper()} ({confidence:.1f}%)")
            return combined_sentiment
            
        except Exception as e:
            logger.error(f"❌ News sentiment analysis failed for {coin}: {str(e)}")
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'reasons': [f'Sentiment analysis error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }
    
    async def _analyze_cryptopanic_sentiment(self, coin: str) -> Optional[Dict[str, Any]]:
        """
        Analyze sentiment from CryptoPanic API (requires token)
        """
        if not self.crypto_panic_token:
            logger.warning("⚠️ CryptoPanic token not configured - skipping CryptoPanic analysis")
            return None
            
        try:
            async with HTTPClient() as client:
                # CryptoPanic API endpoint
                url = f"{self.crypto_panic_base}/posts/"
                params = {
                    'auth_token': self.crypto_panic_token,
                    'currencies': coin.upper(),
                    'filter': 'hot',  # Hot news only
                    'public': 'true'
                }
                
                logger.info(f"🔄 Fetching real CryptoPanic news for {coin}")
                response = await client.get(url, params=params)
                
                if 'results' not in response:
                    logger.warning(f"⚠️ No CryptoPanic results for {coin}")
                    return None
                
                posts = response.get('results', [])
                if not posts:
                    return None
                
                # Analyze sentiment from real news posts
                positive_count = 0
                negative_count = 0
                neutral_count = 0
                
                for post in posts[:20]:  # Analyze top 20 posts
                    title = post.get('title', '').lower()
                    
                    # Check for positive sentiment keywords
                    positive_keywords = ['bull', 'pump', 'moon', 'green', 'gain', 'rally', 'surge', 'breakout', 'adoption', 'partnership']
                    negative_keywords = ['bear', 'dump', 'crash', 'red', 'fall', 'drop', 'decline', 'sell', 'liquidation', 'hack']
                    
                    pos_score = sum(1 for keyword in positive_keywords if keyword in title)
                    neg_score = sum(1 for keyword in negative_keywords if keyword in title)
                    
                    if pos_score > neg_score:
                        positive_count += 1
                    elif neg_score > pos_score:
                        negative_count += 1
                    else:
                        neutral_count += 1
                
                total_posts = positive_count + negative_count + neutral_count
                if total_posts == 0:
                    return None
                
                # Calculate sentiment score
                sentiment_score = (positive_count - negative_count) / total_posts
                
                # Determine bias and confidence
                if sentiment_score > 0.2:
                    bias = 'bullish'
                    confidence = min(0.7, abs(sentiment_score) + 0.3)
                elif sentiment_score < -0.2:
                    bias = 'bearish'
                    confidence = min(0.7, abs(sentiment_score) + 0.3)
                else:
                    bias = 'neutral'
                    confidence = 0.4
                
                logger.info(f"✅ CryptoPanic: {total_posts} posts analyzed for {coin}")
                
                return {
                    'bias': bias,
                    'confidence': confidence,
                    'sentiment_score': sentiment_score,
                    'source_data': {
                        'posts_analyzed': total_posts,
                        'positive_posts': positive_count,
                        'negative_posts': negative_count,
                        'neutral_posts': neutral_count,
                        'source': 'cryptopanic_api'
                    },
                    'reasons': [f'CryptoPanic: {positive_count} positive, {negative_count} negative posts']
                }
                
        except Exception as e:
            logger.error(f"❌ CryptoPanic sentiment analysis error: {str(e)}")
            return None
    
    async def _get_coingecko_sentiment(self, coin: str) -> Optional[Dict[str, Any]]:
        """Get sentiment data from CoinGecko API (free tier available)"""
        
        # CoinGecko coin ID mapping
        coin_id_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'SOL': 'solana',
            'XRP': 'ripple',
            'BNB': 'binancecoin',
            'ADA': 'cardano',
            'DOT': 'polkadot'
        }
        
        coin_id = coin_id_map.get(coin.upper())
        if not coin_id:
            logger.warning(f"⚠️ No CoinGecko mapping for {coin}")
            return None
        
        async with HTTPClient() as client:
            try:
                # Get social stats from CoinGecko
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                params = {
                    'localization': 'false',
                    'tickers': 'false',
                    'market_data': 'false',
                    'community_data': 'true',
                    'developer_data': 'false',
                    'sparkline': 'false'
                }
                
                response = await client.get(url, params=params)
                community_data = response.get('community_data', {})
                
                if not community_data:
                    return None
                
                # Analyze social metrics
                twitter_followers = community_data.get('twitter_followers', 0)
                reddit_subscribers = community_data.get('reddit_subscribers', 0)
                reddit_active_users = community_data.get('reddit_accounts_active_48h', 0)
                
                # Simple sentiment scoring based on social activity
                social_score = 0.0
                reasons = []
                
                if reddit_active_users > 0 and reddit_subscribers > 0:
                    activity_ratio = reddit_active_users / reddit_subscribers
                    if activity_ratio > 0.01:  # High activity ratio
                        social_score += 0.3
                        reasons.append(f'High Reddit activity ({reddit_active_users} active users)')
                    elif activity_ratio < 0.005:  # Low activity
                        social_score -= 0.2
                        reasons.append(f'Low Reddit activity ({reddit_active_users} active users)')
                
                # Determine bias based on social score
                if social_score > 0.1:
                    bias = 'bullish'
                elif social_score < -0.1:
                    bias = 'bearish'
                else:
                    bias = 'neutral'
                
                confidence = min(0.4, abs(social_score) + 0.1)  # Low to moderate confidence
                
                return {
                    'bias': bias,
                    'confidence': confidence,
                    'sentiment_score': social_score,
                    'source_data': {
                        'twitter_followers': twitter_followers,
                        'reddit_subscribers': reddit_subscribers,
                        'reddit_active_users': reddit_active_users,
                        'activity_ratio': activity_ratio if reddit_subscribers > 0 else 0
                    },
                    'reasons': reasons if reasons else [f'CoinGecko social data for {coin}']
                }
                
            except Exception as e:
                logger.error(f"❌ CoinGecko sentiment error: {str(e)}")
                return None
    
    async def _analyze_crypto_news(self, coin: str) -> Optional[Dict[str, Any]]:
        """Analyze crypto news sentiment using free news APIs"""
        
        try:
            # Use CryptoCompare free news API
            async with HTTPClient() as client:
                url = "https://min-api.cryptocompare.com/data/v2/news/"
                params = {
                    'lang': 'EN',
                    'sortOrder': 'latest',
                    'categories': f'{coin},Blockchain',
                    'excludeCategories': 'Sponsored'
                }
                
                response = await client.get(url, params=params)
                news_data = response.get('Data', [])
                
                if not news_data:
                    return None
                
                # Analyze sentiment of recent news (last 24 hours)
                recent_news = []
                current_time = datetime.now().timestamp()
                day_ago = current_time - (24 * 60 * 60)
                
                for article in news_data[:20]:  # Analyze up to 20 recent articles
                    published_on = article.get('published_on', 0)
                    if published_on >= day_ago:
                        recent_news.append(article)
                
                if not recent_news:
                    return None
                
                # Analyze sentiment using keyword matching
                sentiment_scores = []
                article_analysis = []
                
                for article in recent_news:
                    title = article.get('title', '').lower()
                    body = article.get('body', '')[:500].lower()  # First 500 chars
                    text = f"{title} {body}"
                    
                    # Calculate sentiment score
                    sentiment_score = self._calculate_text_sentiment(text, coin.lower())
                    sentiment_scores.append(sentiment_score)
                    
                    article_analysis.append({
                        'title': article.get('title', ''),
                        'sentiment_score': sentiment_score,
                        'published': article.get('published_on', 0)
                    })
                
                if not sentiment_scores:
                    return None
                
                # Calculate overall sentiment
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                
                # Determine bias and confidence
                if avg_sentiment > 0.1:
                    bias = 'bullish'
                    confidence = min(0.6, avg_sentiment + 0.2)
                elif avg_sentiment < -0.1:
                    bias = 'bearish'
                    confidence = min(0.6, abs(avg_sentiment) + 0.2)
                else:
                    bias = 'neutral'
                    confidence = 0.2
                
                reasons = [f'News sentiment analysis: {len(recent_news)} articles']
                if avg_sentiment > 0.2:
                    reasons.append('Predominantly positive crypto news')
                elif avg_sentiment < -0.2:
                    reasons.append('Predominantly negative crypto news')
                
                return {
                    'bias': bias,
                    'confidence': confidence,
                    'sentiment_score': avg_sentiment,
                    'source_data': {
                        'articles_analyzed': len(recent_news),
                        'avg_sentiment': avg_sentiment,
                        'sample_articles': article_analysis[:5]  # Top 5 for debugging
                    },
                    'reasons': reasons
                }
                
        except Exception as e:
            logger.error(f"❌ Crypto news analysis error: {str(e)}")
            return None
    
    def _calculate_text_sentiment(self, text: str, coin: str) -> float:
        """
        Calculate sentiment score for text using crypto-specific keywords
        Returns score between -1 (very bearish) and +1 (very bullish)
        """
        try:
            # Clean and tokenize text
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Calculate keyword scores
            bullish_score = 0
            bearish_score = 0
            
            for word in words:
                # Check bullish keywords
                for keyword in CRYPTO_BULLISH_KEYWORDS:
                    if keyword in text:
                        bullish_score += 1
                        # Extra weight if keyword is directly related to the coin
                        if coin in text and word in keyword:
                            bullish_score += 0.5
                
                # Check bearish keywords
                for keyword in CRYPTO_BEARISH_KEYWORDS:
                    if keyword in text:
                        bearish_score += 1
                        # Extra weight if keyword is directly related to the coin
                        if coin in text and word in keyword:
                            bearish_score += 0.5
            
            # Normalize scores
            total_keywords = bullish_score + bearish_score
            if total_keywords == 0:
                return 0.0
            
            # Return normalized sentiment score
            sentiment_score = (bullish_score - bearish_score) / max(1, total_keywords)
            return max(-1.0, min(1.0, sentiment_score))
            
        except Exception as e:
            logger.error(f"❌ Text sentiment calculation error: {str(e)}")
            return 0.0
    
    def _combine_sentiment_sources(self, sentiment_sources: List[tuple]) -> Dict[str, Any]:
        """Combine sentiment from multiple sources with appropriate weighting"""
        
        try:
            # Weight different sources
            source_weights = {
                'reddit': 0.4,      # High weight if working
                'coingecko': 0.3,   # Medium weight for social data
                'news': 0.3         # Medium weight for news sentiment  
            }
            
            weighted_scores = []
            all_reasons = []
            source_data = {}
            
            for source_name, sentiment_data in sentiment_sources:
                weight = source_weights.get(source_name, 0.2)
                bias = sentiment_data.get('bias', 'neutral')
                confidence = sentiment_data.get('confidence', 0.0)
                sentiment_score = sentiment_data.get('sentiment_score', 0.0)
                reasons = sentiment_data.get('reasons', [])
                
                # Convert bias to numeric score
                if bias == 'bullish':
                    numeric_score = abs(sentiment_score) if sentiment_score != 0 else 0.5
                elif bias == 'bearish':
                    numeric_score = -abs(sentiment_score) if sentiment_score != 0 else -0.5
                else:
                    numeric_score = 0.0
                
                # Weight the score by confidence and source weight
                weighted_score = numeric_score * confidence * weight
                weighted_scores.append(weighted_score)
                
                # Collect reasons and source data
                all_reasons.extend(reasons)
                source_data[source_name] = sentiment_data.get('source_data', {})
            
            if not weighted_scores:
                return {
                    'bias': 'neutral',
                    'confidence': 0.0,
                    'reasons': ['No valid sentiment sources'],
                    'raw_data': {'error': 'No valid sources'}
                }
            
            # Calculate final sentiment
            final_score = sum(weighted_scores) / len(weighted_scores)
            
            # Determine final bias and confidence
            if final_score > 0.1:
                final_bias = 'bullish'
                final_confidence = min(0.5, final_score + 0.1)  # Cap at moderate confidence
            elif final_score < -0.1:
                final_bias = 'bearish'
                final_confidence = min(0.5, abs(final_score) + 0.1)
            else:
                final_bias = 'neutral'
                final_confidence = 0.1
            
            # Limit sentiment confidence due to API restrictions
            final_confidence = min(0.4, final_confidence)  # Max 40% confidence for sentiment
            
            return {
                'bias': final_bias,
                'confidence': final_confidence,
                'reasons': all_reasons[:3] if all_reasons else ['Combined sentiment analysis'],
                'raw_data': {
                    'final_score': final_score,
                    'sources_used': len(sentiment_sources),
                    'source_data': source_data,
                    'api_limitations': 'Reddit/Twitter APIs have restricted access'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error combining sentiment sources: {str(e)}")
            return {
                'bias': 'neutral',
                'confidence': 0.0,
                'reasons': [f'Sentiment combination error: {str(e)}'],
                'raw_data': {'error': str(e)}
            }