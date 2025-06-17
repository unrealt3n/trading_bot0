#!/usr/bin/env python3
"""
🧪 CryptoPanic API Test Script
Test your CryptoPanic token and see news sentiment analysis in action
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add modules to path
sys.path.append('.')

from modules.news_analyzer import NewsAnalyzer

async def test_crypto_panic():
    """Test CryptoPanic integration"""
    
    print("🧪 Testing CryptoPanic Integration")
    print("=" * 50)
    
    # Check if token is configured
    token = os.getenv('CRYPTO_PANIC_TOKEN', '')
    
    if not token or token == 'your_crypto_panic_token_here':
        print("❌ CryptoPanic token not configured!")
        print("📝 Please add your token to .env file:")
        print("   CRYPTO_PANIC_TOKEN=your_actual_token_here")
        print("")
        print("🔗 Get your free token at: https://cryptopanic.com/developers/api/")
        return
    
    print(f"✅ CryptoPanic token configured: {token[:8]}...")
    print("")
    
    # Initialize news analyzer
    analyzer = NewsAnalyzer()
    
    # Test coins
    test_coins = ['BTC', 'ETH', 'SOL']
    
    for coin in test_coins:
        print(f"📰 Testing {coin} news sentiment...")
        
        try:
            result = await analyzer.analyze_news_sentiment(coin)
            
            print(f"🎯 {coin} Results:")
            print(f"   Bias: {result['bias'].upper()}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Reasons: {', '.join(result['reasons'][:2])}")
            
            # Show CryptoPanic specific data if available
            raw_data = result.get('raw_data', {})
            for source_name, source_data in raw_data.items():
                if 'cryptopanic' in source_name.lower():
                    source_info = source_data.get('source_data', {})
                    if source_info:
                        posts = source_info.get('posts_analyzed', 0)
                        positive = source_info.get('positive_posts', 0)
                        negative = source_info.get('negative_posts', 0)
                        print(f"   📊 CryptoPanic: {posts} posts ({positive} positive, {negative} negative)")
            
            print("")
            
        except Exception as e:
            print(f"❌ Error testing {coin}: {e}")
            print("")
    
    print("✅ CryptoPanic test complete!")

async def test_individual_cryptopanic():
    """Test just the CryptoPanic method directly"""
    
    print("🧪 Testing CryptoPanic API Directly")
    print("=" * 50)
    
    analyzer = NewsAnalyzer()
    
    # Test BTC
    print("📰 Testing BTC CryptoPanic API...")
    
    try:
        result = await analyzer._analyze_cryptopanic_sentiment('BTC')
        
        if result:
            print("✅ CryptoPanic API working!")
            print(f"   Bias: {result['bias']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Posts analyzed: {result['source_data']['posts_analyzed']}")
            print(f"   Positive posts: {result['source_data']['positive_posts']}")
            print(f"   Negative posts: {result['source_data']['negative_posts']}")
        else:
            print("❌ No data returned from CryptoPanic API")
            
    except Exception as e:
        print(f"❌ CryptoPanic API error: {e}")

def main():
    """Main function"""
    
    print("🚀 CryptoPanic Integration Test")
    print("=" * 50)
    print("")
    
    # Check if we have the token
    token = os.getenv('CRYPTO_PANIC_TOKEN', '')
    
    if not token:
        print("❌ No CryptoPanic token found in .env")
        print("📝 Please add CRYPTO_PANIC_TOKEN=your_token_here to .env file")
        return
    
    print("Choose test:")
    print("1. Full news sentiment analysis (recommended)")
    print("2. CryptoPanic API only")
    print("")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(test_crypto_panic())
    elif choice == "2":
        asyncio.run(test_individual_cryptopanic())
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()