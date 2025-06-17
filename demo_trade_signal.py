#!/usr/bin/env python3
"""
🎯 Trade Signal Demo
Shows exactly what a trade signal looks like when it triggers
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add modules to path
sys.path.append('.')

# Import rich for better display
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich for better display...")
    os.system("pip install rich")

def create_demo_whale_signal() -> Dict[str, Any]:
    """Create a realistic whale alert signal"""
    return {
        'signal_strength': 85.0,
        'bias': 'bullish',
        'confidence': 0.78,
        'transactions': [
            {
                'amount_usd': 15_000_000,
                'transaction_type': 'exchange_withdrawal',
                'exchange_name': 'binance',
                'timestamp': datetime.now()
            },
            {
                'amount_usd': 8_500_000,
                'transaction_type': 'exchange_withdrawal', 
                'exchange_name': 'coinbase',
                'timestamp': datetime.now()
            }
        ],
        'summary': {
            'net_flow_usd': 23_500_000,
            'exchange_withdrawals': 2,
            'exchange_deposits': 0,
            'dominant_flow': 'outflow'
        },
        'reasons': [
            'Large whale volume: $23,500,000',
            'Net outflow from exchanges: $23,500,000',
            'More withdrawals (2) than deposits (0)'
        ]
    }

def create_demo_liquidation_signal() -> Dict[str, Any]:
    """Create a realistic liquidation signal"""
    return {
        'signal_strength': 72.0,
        'bias': 'bullish',
        'confidence': 0.68,
        'liquidations': [
            {'side': 'short', 'amount_usd': 5_200_000, 'exchange': 'binance'},
            {'side': 'short', 'amount_usd': 3_800_000, 'exchange': 'bybit'},
            {'side': 'long', 'amount_usd': 1_200_000, 'exchange': 'okx'}
        ],
        'metrics': {
            'total_short_liquidations': 9_000_000,
            'total_long_liquidations': 1_200_000,
            'short_liq_ratio': 0.88,
            'liquidation_count': 3
        },
        'summary': {
            'dominant_side': 'short',
            'cascade_detected': True,
            'liquidation_rate': 0.05
        },
        'reasons': [
            'Dominant short liquidations: 88.0%',
            'Cascading liquidations detected',
            'High liquidation volume: $10,200,000'
        ]
    }

def create_demo_retail_signal() -> Dict[str, Any]:
    """Create a realistic retail vs big money signal"""
    return {
        'signal_strength': 65.0,
        'bias': 'bullish',
        'confidence': 0.61,
        'divergence_type': 'moderate',
        'divergence_analysis': {
            'retail_sentiment_score': -0.3,  # Retail bearish
            'institutional_flow_score': 0.5,  # Institutions bullish
            'divergence': 0.8,
            'trap_probability': 0.75
        },
        'summary': {
            'retail_sentiment': 'bearish',
            'institutional_flow': 'bullish',
            'follow_smart_money': True,
            'trap_probability': 0.75
        },
        'reasons': [
            'Moderate retail-institutional divergence',
            'Retail bearish vs institutions bullish',
            'High retail trap probability (75.0%)',
            'Following smart money direction'
        ]
    }

def create_demo_enhanced_analysis() -> Dict[str, Any]:
    """Create a complete enhanced trade analysis"""
    
    # Individual signals
    whale_data = create_demo_whale_signal()
    liquidation_data = create_demo_liquidation_signal()
    retail_data = create_demo_retail_signal()
    
    # Signal scores with weights
    signal_scores = {
        'whale_alerts': {
            'confidence': 0.78,
            'bias': 'bullish',
            'weight': 0.20,
            'weighted_contribution': 0.78 * 0.20,  # 0.156
            'signal_strength': 85.0
        },
        'liquidations': {
            'confidence': 0.68,
            'bias': 'bullish',
            'weight': 0.18,
            'weighted_contribution': 0.68 * 0.18,  # 0.122
            'signal_strength': 72.0
        },
        'retail_vs_bigmoney': {
            'confidence': 0.61,
            'bias': 'bullish',
            'weight': 0.15,
            'weighted_contribution': 0.61 * 0.15,  # 0.092
            'signal_strength': 65.0
        },
        'technical_analysis': {
            'confidence': 0.55,
            'bias': 'bullish',
            'weight': 0.10,
            'weighted_contribution': 0.55 * 0.10,  # 0.055
            'signal_strength': 55.0
        },
        'news_sentiment': {
            'confidence': 0.45,
            'bias': 'neutral',
            'weight': 0.08,
            'weighted_contribution': 0.45 * 0.08 * 0.5,  # 0.018 (neutral = 50%)
            'signal_strength': 45.0
        }
    }
    
    # Calculate total confidence
    total_weighted = sum(score['weighted_contribution'] for score in signal_scores.values())
    base_confidence = min(total_weighted, 1.0)  # 0.443
    
    # Duration analysis
    duration_data = {
        'duration': 'short',
        'duration_minutes': 240,  # 4 hours
        'confidence': 78.5,
        'primary_signals': ['whale_alerts', 'liquidations'],
        'exit_strategy': {
            'take_profit_percent': 2.2,
            'stop_loss_percent': 1.1,
            'max_duration_hours': 4,
            'trail_stop': True,
            'partial_exits': [
                {'at_profit': 1.0, 'exit_percent': 40},
                {'at_profit': 2.0, 'exit_percent': 30}
            ]
        }
    }
    
    # Confidence adjustments
    adjustment_data = {
        'base_confidence': base_confidence,
        'adjusted_confidence': base_confidence + 0.157,  # 0.60
        'confidence_change': 0.157,
        'adjustments': {
            'support_resistance': 0.10,    # No major resistance ahead
            'market_structure': 0.05,      # Trending market
            'volatility': 0.007,           # Good volatility for short-term
            'session_context': 0.0,        # Neutral session
            'volume_profile': 0.0          # Normal volume
        },
        'reasons': [
            'Clear path to resistance (5.2%)',
            'Strong trend alignment',
            'Good volatility for short-term trade'
        ]
    }
    
    final_confidence = adjustment_data['adjusted_confidence']
    
    return {
        'take_trade': True,
        'confidence': final_confidence,
        'bias': 'bullish',
        'duration': duration_data['duration'],
        'duration_data': duration_data,
        'enhanced_analysis': {
            'signal_scores': signal_scores,
            'base_confidence': base_confidence,
            'final_confidence': final_confidence,
            'confidence_adjustments': adjustment_data,
            'whale_analysis': whale_data,
            'liquidation_analysis': liquidation_data,
            'retail_analysis': retail_data
        },
        'reasons': [
            'Whale Alerts: bullish (78%)',
            'Liquidations: bullish (68%)',
            'Retail vs Big Money: bullish (61%)',
            'Short trade based on whale_alerts, liquidations',
            'Clear path to resistance (5.2%)'
        ],
        'timestamp': datetime.now().isoformat(),
        'coin': 'BTC',
        'signal_strength': final_confidence * 100
    }

def print_rich_demo():
    """Print demo using rich formatting"""
    
    console = Console()
    
    # Create demo analysis
    analysis = create_demo_enhanced_analysis()
    
    # Header
    header = Panel(
        "[bold blue]🚀 ENHANCED TRADE SIGNAL DETECTED! 🚀[/bold blue]",
        style="blue",
        padding=(1, 2)
    )
    console.print(header)
    console.print()
    
    # Main trade info
    trade_table = Table(title="🎯 Trade Decision", title_style="bold green")
    trade_table.add_column("Parameter", style="cyan", no_wrap=True)
    trade_table.add_column("Value", style="bold green")
    trade_table.add_column("Details", style="yellow")
    
    confidence_pct = analysis['confidence'] * 100
    signal_strength = analysis['signal_strength']
    
    trade_table.add_row("🎯 Action", "✅ TAKE TRADE", "All conditions met")
    trade_table.add_row("💰 Coin", analysis['coin'], "Bitcoin")
    trade_table.add_row("📈 Direction", analysis['bias'].upper(), "Long position")
    trade_table.add_row("🎲 Confidence", f"{confidence_pct:.1f}%", f"Signal strength: {signal_strength:.1f}")
    trade_table.add_row("⏰ Duration", analysis['duration'].upper(), f"{analysis['duration_data']['duration_minutes']}min")
    trade_table.add_row("🎯 Take Profit", f"{analysis['duration_data']['exit_strategy']['take_profit_percent']:.1f}%", "Partial exits enabled")
    trade_table.add_row("🛡️ Stop Loss", f"{analysis['duration_data']['exit_strategy']['stop_loss_percent']:.1f}%", "Risk management")
    
    console.print(trade_table)
    console.print()
    
    # Signal breakdown
    signal_table = Table(title="📊 Enhanced Signal Breakdown", title_style="bold blue")
    signal_table.add_column("Signal Source", style="cyan")
    signal_table.add_column("Bias", style="white")
    signal_table.add_column("Confidence", style="green")
    signal_table.add_column("Weight", style="yellow")
    signal_table.add_column("Contribution", style="magenta")
    
    signal_scores = analysis['enhanced_analysis']['signal_scores']
    
    # Sort by contribution
    sorted_signals = sorted(signal_scores.items(), 
                           key=lambda x: abs(x[1]['weighted_contribution']), 
                           reverse=True)
    
    for signal_name, data in sorted_signals:
        emoji_map = {
            'whale_alerts': '🐋',
            'liquidations': '💥', 
            'retail_vs_bigmoney': '🏦',
            'technical_analysis': '📈',
            'news_sentiment': '📰'
        }
        
        emoji = emoji_map.get(signal_name, '📊')
        name = f"{emoji} {signal_name.replace('_', ' ').title()}"
        bias = data['bias'].upper()
        confidence = f"{data['confidence']*100:.1f}%"
        weight = f"{data['weight']*100:.1f}%"
        contribution = f"{data['weighted_contribution']*100:+.1f}%"
        
        # Color code bias
        if bias == "BULLISH":
            bias = f"[green]{bias}[/green]"
        elif bias == "BEARISH":
            bias = f"[red]{bias}[/red]"
        else:
            bias = f"[yellow]{bias}[/yellow]"
        
        signal_table.add_row(name, bias, confidence, weight, contribution)
    
    console.print(signal_table)
    console.print()
    
    # Confidence adjustments
    adj_table = Table(title="🎯 Dynamic Confidence Adjustments", title_style="bold cyan")
    adj_table.add_column("Factor", style="cyan")
    adj_table.add_column("Adjustment", style="white")
    adj_table.add_column("Reason", style="yellow")
    
    base_conf = analysis['enhanced_analysis']['base_confidence']
    final_conf = analysis['enhanced_analysis']['final_confidence']
    adjustments = analysis['enhanced_analysis']['confidence_adjustments']['adjustments']
    
    adj_table.add_row("📊 Base Confidence", f"{base_conf:.1%}", "From weighted signals")
    
    for factor, adjustment in adjustments.items():
        if abs(adjustment) > 0.001:  # Only show significant adjustments
            factor_name = factor.replace('_', ' ').title()
            adj_str = f"{adjustment:+.1%}"
            
            if factor == 'support_resistance':
                reason = "No major resistance ahead"
            elif factor == 'market_structure':
                reason = "Trending market structure"
            elif factor == 'volatility':
                reason = "Optimal volatility level"
            else:
                reason = "Market condition factor"
            
            adj_table.add_row(f"⚙️ {factor_name}", adj_str, reason)
    
    adj_table.add_row("🎯 Final Confidence", f"{final_conf:.1%}", f"Adjusted by +{(final_conf-base_conf):.1%}")
    
    console.print(adj_table)
    console.print()
    
    # Key insights
    insights = Panel(
        "[bold yellow]🔍 Key Insights:[/bold yellow]\n\n"
        "🐋 [cyan]Large whale outflows ($23.5M) suggest institutional accumulation[/cyan]\n"
        "💥 [green]Short liquidations dominating (88%) creating bullish momentum[/green]\n"
        "🏦 [blue]Retail bearish while institutions bullish - classic setup[/blue]\n"
        "📈 [white]Technical analysis confirms breakout above resistance[/white]\n"
        "⏰ [yellow]Short-term duration optimal for current volatility[/yellow]",
        title="📋 Analysis Summary",
        border_style="green"
    )
    console.print(insights)
    console.print()
    
    # Trade execution preview
    execution = Panel(
        "[bold green]💰 Trade Execution Preview:[/bold green]\n\n"
        f"📈 Direction: LONG {analysis['coin']}\n"
        f"💵 Entry: $50,247.83 (current price)\n"
        f"🎯 Take Profit: $51,353.29 (+2.2%)\n"
        f"🛡️ Stop Loss: $49,695.51 (-1.1%)\n"
        f"⏰ Max Duration: 4 hours\n"
        f"📊 Position Size: $20 (demo mode)\n"
        f"🎲 Risk/Reward: 1:2 ratio",
        title="🚀 Ready to Execute",
        border_style="blue"
    )
    console.print(execution)

def print_simple_demo():
    """Print demo for terminals without rich"""
    
    analysis = create_demo_enhanced_analysis()
    
    print("🚀" + "="*60)
    print("   ENHANCED TRADE SIGNAL DETECTED!")
    print("🚀" + "="*60)
    print()
    
    # Main trade info
    confidence_pct = analysis['confidence'] * 100
    duration_mins = analysis['duration_data']['duration_minutes']
    tp_pct = analysis['duration_data']['exit_strategy']['take_profit_percent']
    sl_pct = analysis['duration_data']['exit_strategy']['stop_loss_percent']
    
    print("🎯 TRADE DECISION:")
    print(f"   ✅ Action: TAKE TRADE")
    print(f"   💰 Coin: {analysis['coin']}")
    print(f"   📈 Direction: {analysis['bias'].upper()}")
    print(f"   🎲 Confidence: {confidence_pct:.1f}%")
    print(f"   ⏰ Duration: {analysis['duration'].upper()} ({duration_mins}min)")
    print(f"   🎯 Take Profit: {tp_pct:.1f}%")
    print(f"   🛡️ Stop Loss: {sl_pct:.1f}%")
    print()
    
    # Signal breakdown
    print("📊 ENHANCED SIGNAL BREAKDOWN:")
    signal_scores = analysis['enhanced_analysis']['signal_scores']
    
    sorted_signals = sorted(signal_scores.items(), 
                           key=lambda x: abs(x[1]['weighted_contribution']), 
                           reverse=True)
    
    for signal_name, data in sorted_signals:
        emoji_map = {
            'whale_alerts': '🐋',
            'liquidations': '💥',
            'retail_vs_bigmoney': '🏦', 
            'technical_analysis': '📈',
            'news_sentiment': '📰'
        }
        
        emoji = emoji_map.get(signal_name, '📊')
        name = signal_name.replace('_', ' ').title()
        bias = data['bias'].upper()
        confidence = data['confidence'] * 100
        weight = data['weight'] * 100
        contribution = data['weighted_contribution'] * 100
        
        print(f"   {emoji} {name}: {bias} ({confidence:.1f}%) | "
              f"Weight: {weight:.1f}% | Contribution: {contribution:+.1f}%")
    
    print()
    
    # Confidence adjustments
    print("🎯 CONFIDENCE ADJUSTMENTS:")
    base_conf = analysis['enhanced_analysis']['base_confidence']
    final_conf = analysis['enhanced_analysis']['final_confidence']
    change = final_conf - base_conf
    
    print(f"   📊 Base Confidence: {base_conf:.1%}")
    print(f"   ⚙️ Adjustments: {change:+.1%}")
    print(f"   🎯 Final Confidence: {final_conf:.1%}")
    print()
    
    # Key insights
    print("🔍 KEY INSIGHTS:")
    print("   🐋 Large whale outflows ($23.5M) suggest accumulation")
    print("   💥 Short liquidations dominating (88%) = bullish momentum")
    print("   🏦 Retail bearish vs institutions bullish = classic setup")
    print("   📈 Technical confirms breakout above resistance")
    print("   ⏰ Short-term duration optimal for volatility")
    print()
    
    # Trade execution
    print("💰 TRADE EXECUTION:")
    print("   📈 Direction: LONG BTC")
    print("   💵 Entry: $50,247.83")
    print("   🎯 Take Profit: $51,353.29 (+2.2%)")
    print("   🛡️ Stop Loss: $49,695.51 (-1.1%)")
    print("   ⏰ Max Duration: 4 hours")
    print("   📊 Position Size: $20 (demo)")
    print("   🎲 Risk/Reward: 1:2 ratio")
    print()
    print("="*62)

def main():
    """Main demo function"""
    
    print("🎯 Enhanced Trading Bot - Trade Signal Demo")
    print("=" * 50)
    print()
    print("This shows exactly what happens when a high-probability trade signal is detected...")
    print()
    
    print("Starting demo in 2 seconds...")
    print()
    
    if RICH_AVAILABLE:
        print_rich_demo()
    else:
        print_simple_demo()
    
    print()
    print("🚀 This is what you'll see when the bot finds a trade!")
    print("📊 The bot analyzes all signals and combines them intelligently")
    print("⚡ Real signals will vary based on current market conditions")

if __name__ == "__main__":
    main()