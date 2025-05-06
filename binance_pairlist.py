#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance USDC/USDT Pair List Generator für Freqtrade
Optimiert für ausgewogene Strategie mit kurz- bis mittelfristigem Gewinnpotential
Ideal für kleineres Kapital (50-100 USDC/USDT)
"""

import ccxt
import json
import argparse
from typing import List, Dict, Any, Tuple
import sys
import time
import numpy as np
from datetime import datetime, timedelta


def fetch_binance_pairs(
    quote_currency: str = "USDC",
    auto_mode: bool = True,
    include_standard_coins: bool = True,
    filter_standard_coins: bool = True,
    max_assets: int = 8,
    max_standard_assets: int = 4,  # Erhöht für mehr etablierte Coins
    capital_size: str = "small"  # "small", "medium", or "large"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch trading pairs from Binance optimized for balanced profit potential.
    
    Args:
        quote_currency: Quote currency (default: USDC)
        auto_mode: Whether to automatically determine optimal parameters
        include_standard_coins: Whether to include standard reliable coins
        filter_standard_coins: Whether to filter standard coins based on quality
        max_assets: Maximum number of assets to include in total
        max_standard_assets: Maximum number of standard assets to include
        capital_size: Size of trading capital ("small", "medium", "large")
        
    Returns:
        List of filtered trading pairs with their details and market stats
    """
    print(f"Fetching {quote_currency} trading pairs from Binance...")
    print(f"Optimizing for {capital_size} capital size with balanced profit potential...")
    print(f"Max standard assets: {max_standard_assets}, Total assets: {max_assets}")
    
    # Define standard reliable coins that should always be included if available
    # Ordered by priority (first ones more likely to be included)
    standard_coins = [
        "BTC", "ETH", "XRP", "ADA", "SOL", "DOT", 
        "AVAX", "MATIC", "LINK", "LTC", "BNB", "UNI", 
        "AAVE", "DOT", "ATOM", "ALGO", "XLM", "ETC"
    ]
    
    # For small capital, we may want to prioritize some mid-caps with more movement
    if capital_size == "small":
        # Reorder to prioritize some mid-caps with better short-term movements
        standard_coins = [
            "ETH", "SOL",  "MATIC", "AVAX", "BTC", "ADA", 
            "DOT", "LINK", "XRP", "UNI", "AAVE", "LTC", "DOGE"
        ]
    
    standard_pairs = [f"{coin}/{quote_currency}" for coin in standard_coins]
    
    try:
        # Initialize Binance exchange
        binance = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        
        # Load markets
        markets = binance.load_markets()
        
        # Filter for quote currency pairs
        quote_pairs = []
        standard_pairs_available = []
        
        for symbol, market in markets.items():
            if market['quote'] == quote_currency and not market.get('linear') and market['active']:
                quote_pairs.append(market)
                # Check if this is one of our standard pairs
                if include_standard_coins and symbol in standard_pairs:
                    standard_pairs_available.append(symbol)
        
        print(f"Found {len(quote_pairs)} {quote_currency} pairs total")
        if include_standard_coins:
            print(f"Found {len(standard_pairs_available)} standard reliable coins available for trading")
        
        # Get all tickers at once to avoid rate limits
        tickers = binance.fetch_tickers([market['symbol'] for market in quote_pairs])
        
        # Try to get OHLCV data for a few pairs to analyze patterns
        # This helps identify coins with regular movement patterns
        pattern_data = {}
        sample_pairs = standard_pairs_available[:3]  # Use a few standard pairs as samples
        
        for symbol in sample_pairs:
            try:
                # Get daily candles for the last 14 days
                ohlcv = binance.fetch_ohlcv(symbol, '1d', limit=14)
                
                # Calculate average daily range as percentage
                daily_ranges = []
                for candle in ohlcv:
                    if candle[2] > 0:  # high
                        daily_range = (candle[2] - candle[3]) / candle[4] * 100  # (high-low)/close * 100
                        daily_ranges.append(daily_range)
                
                if daily_ranges:
                    avg_range = sum(daily_ranges) / len(daily_ranges)
                    pattern_data[symbol] = {
                        'avg_daily_range': avg_range,
                        'consistency': np.std(daily_ranges)  # Lower std = more consistent moves
                    }
            except Exception as e:
                # Skip if we can't get data for this pair
                continue
        
        # Collect market statistics to determine optimal parameters
        volumes = []
        prices = []
        
        for market in quote_pairs:
            symbol = market['symbol']
            if symbol in tickers:
                ticker = tickers[symbol]
                if ticker['quoteVolume'] and ticker['last']:
                    volumes.append(ticker['quoteVolume'])
                    prices.append(ticker['last'])
        
        # Market statistics for auto-mode
        market_stats = {}
        if volumes:
            market_stats = {
                'volume_median': np.median(volumes),
                'volume_25th': np.percentile(volumes, 25),
                'volume_75th': np.percentile(volumes, 75),
                'price_median': np.median(prices),
                'total_pairs': len(volumes)
            }
        
        # Set parameters based on capital size
        if capital_size == "small":  # 50-100 USDC
            # For small capital, favor mid-volume coins with consistent movement
            min_volume = max(30000, market_stats.get('volume_25th', 30000) * 0.4)
            ideal_volatility = (3, 15)  # Höhere Volatilität für Buy-Low-Sell-High
            max_volatility = 25  # Cap to avoid excessive risk
            # With small capital, favor coins with price < $10 to get meaningful positions
            price_preference = "low_to_mid"
        elif capital_size == "medium":  # 100-1000 USDC
            min_volume = max(80000, market_stats.get('volume_25th', 80000) * 0.6)
            ideal_volatility = (2, 12)  # 2-10% daily range
            max_volatility = 20
            price_preference = "balanced"
        else:  # large capital
            min_volume = max(150000, market_stats.get('volume_median', 150000) * 0.7)
            ideal_volatility = (2, 10)  # 2-8% daily range 
            max_volatility = 18
            price_preference = "balanced"
            
        # Set minimum price to avoid dust issues
        min_price = 0.000001
        
        # For auto mode, determine optimal parameters based on market conditions
        if auto_mode and volumes:
            # Dynamic count based on capital size and market
            dynamic_pairs_count = max(0, max_assets - max_standard_assets)
            
            print(f"\nAuto-determined parameters based on current market conditions:")
            print(f"Minimum volume: ${min_volume:.2f}")
            print(f"Ideal volatility range: {ideal_volatility[0]}-{ideal_volatility[1]}%")
            print(f"Price preference: {price_preference}")
            print(f"Dynamic pairs to select: {dynamic_pairs_count}")
        else:
            # Default parameters if not in auto mode or if market data is insufficient
            dynamic_pairs_count = max(0, max_assets - max_standard_assets)
        
        # Add market data for each pair
        enriched_pairs = []
        
        for market in quote_pairs:
            symbol = market['symbol']
            if symbol in tickers:
                ticker = tickers[symbol]
                volume_quote = ticker['quoteVolume'] if ticker['quoteVolume'] else 0
                last_price = ticker['last'] if ticker['last'] else 0
                
                # Check if it's a standard pair
                is_standard = symbol in standard_pairs_available
                
                # Filter standard pairs with basic quality checks
                standard_quality_check = True
                if is_standard and filter_standard_coins:
                    # Minimum volume for standard coins (lower than regular, but still need some activity)
                    min_standard_volume = min_volume * 0.3  # 30% of the regular minimum volume
                    
                    # Check for extremely negative trend (avoid trading in strong downtrends)
                    if ticker.get('percentage', 0) < -10:  # More than 10% down in 24h
                        standard_quality_check = False
                        
                    # Check for minimum volume (even standards need some liquidity)
                    if volume_quote < min_standard_volume:
                        standard_quality_check = False
                        
                    # Check for too low volatility (not worth trading)
                    if ticker['high'] and ticker['low'] and ticker['last']:
                        volatility = (ticker['high'] - ticker['low']) / ticker['last'] * 100
                        if volatility < 1.0:  # Less than 1% daily range is too flat
                            standard_quality_check = False
                
                # Include pair if it passes filters
                if (is_standard and standard_quality_check) or (volume_quote >= min_volume and last_price >= min_price):
                    # Calculate price change
                    change_24h = ticker['percentage'] if 'percentage' in ticker else ticker.get('change', 0)
                    
                    # Calculate volatility (high-low range as percentage of price)
                    if ticker['high'] and ticker['low'] and ticker['last']:
                        volatility = (ticker['high'] - ticker['low']) / ticker['last'] * 100
                    else:
                        volatility = 0
                    
                    # Get bid-ask spread if available (tighter is better for short-term trading)
                    # Use estimation if not available
                    spread = 0.1  # Default estimation 0.1%
                    if 'bid' in ticker and 'ask' in ticker and ticker['bid'] and ticker['ask']:
                        spread = (ticker['ask'] - ticker['bid']) / ticker['ask'] * 100
                        
                    # Add pattern data if we have it
                    avg_daily_range = None
                    range_consistency = None
                    if symbol in pattern_data:
                        avg_daily_range = pattern_data[symbol]['avg_daily_range']
                        range_consistency = pattern_data[symbol]['consistency']
                    
                    enriched_pairs.append({
                        'symbol': symbol,
                        'volume': volume_quote,
                        'price': last_price,
                        'change_24h': change_24h,
                        'volatility': volatility,
                        'spread': spread,
                        'avg_daily_range': avg_daily_range,
                        'range_consistency': range_consistency,
                        'is_standard': is_standard
                    })
        
        # Score each pair based on multiple factors optimized for profit potential
        for pair in enriched_pairs:
            # Standard pairs get a high base score
            if pair['is_standard']:
                # Give priority based on order in standard_coins list
                standard_index = 0
                for i, std_pair in enumerate(standard_pairs):
                    if pair['symbol'] == std_pair:
                        standard_index = i
                        break
                
                # Base score for standard coins
                pair['score'] = 800 - (standard_index * 10)  # Higher priority standards get higher scores
                
                # Adjust standard scores based on volatility for profit potential
                volatility = pair['volatility']
                # For standard coins, ideal volatility is within our range, but we're more lenient
                if ideal_volatility[0] <= volatility <= ideal_volatility[1]:
                    # Perfect range - boost score
                    pair['score'] += 200
                elif volatility > ideal_volatility[1] and volatility <= max_volatility:
                    # Above ideal but still acceptable - smaller boost
                    pair['score'] += 100
                elif volatility < ideal_volatility[0] and volatility >= 1.5:
                    # Below ideal but still tradeable
                    pair['score'] += 50
                
                continue
                
            # Score components for dynamic pairs focused on profit potential:
            
            # 1. Volume score (moderate volume is ideal for small capital)
            if volume_quote < min_volume:
                volume_score = 0
            elif volume_quote < min_volume * 2:
                volume_score = 50 + (volume_quote - min_volume) / (min_volume) * 50
            elif volume_quote < market_stats['volume_75th']:
                volume_score = 100  # Ideal volume range
            else:
                # Penalize extremely high volume for small capital
                excess_factor = (volume_quote / market_stats['volume_75th'])
                volume_score = max(40, 100 - (excess_factor - 1) * 30)
            
            # 2. Volatility score (optimized for balanced profit potential)
            volatility = pair['volatility']
            if ideal_volatility[0] <= volatility <= ideal_volatility[1]:
                # Perfect volatility range
                volatility_score = 100
            elif volatility > ideal_volatility[1] and volatility <= max_volatility:
                # Higher than ideal but still usable
                volatility_score = 90 - ((volatility - ideal_volatility[1]) / (max_volatility - ideal_volatility[1])) * 30
            elif volatility < ideal_volatility[0] and volatility >= 1.5:
                # Lower than ideal
                volatility_score = 70 * (volatility / ideal_volatility[0])
            else:
                # Too flat, not enough movement
                volatility_score = max(0, 60 - abs(volatility - (ideal_volatility[0] + 0.5)) * 15)
                
            # 3. Price preference score
            if price_preference == "low_to_mid":
                if last_price < 0.1:
                    # Very low priced coins often have issues
                    price_score = 60
                elif 0.1 <= last_price <= 10:
                    # Good price range for small capital
                    price_score = 100
                elif 10 < last_price <= 50:
                    # Still ok but less ideal for small positions
                    price_score = 80
                elif 50 < last_price <= 200:
                    # Higher priced coins less ideal for small capital
                    price_score = 60
                else:
                    # Very high priced coins means tiny positions with small capital
                    price_score = 40
            else:  # balanced preference
                price_score = 80  # Neutral score
                
            # 4. Recent performance score (prefer gentle uptrends, avoid strong downtrends)
            change = pair['change_24h']
            if change > 5 and change < 15:
                # Healthy uptrend, not too extreme
                change_score = 100
            elif change >= 15:
                # Strong uptrend, might be due for correction
                change_score = max(30, 100 - (change - 15) * 5)
            elif change > 0 and change <= 5:
                # Gentle uptrend, very good
                change_score = 90
            elif change > -5 and change <= 0:
                # Flat to slightly negative, still ok
                change_score = 70
            elif change > -15 and change <= -5:
                # Downtrend, caution
                change_score = max(0, 70 - abs(change + 5) * 7)
            else:
                # Strong downtrend, avoid
                change_score = max(0, 30 - abs(change + 15) * 2)
                
            # 5. Spread score (tighter spread is better for short-term trading)
            spread = pair.get('spread', 0.15)
            if spread < 0.05:
                # Excellent spread
                spread_score = 100
            elif spread < 0.1:
                # Very good spread
                spread_score = 90
            elif spread < 0.2:
                # Good spread
                spread_score = 80
            elif spread < 0.5:
                # Average spread
                spread_score = 60
            else:
                # Wide spread, not ideal for short-term
                spread_score = max(0, 60 - (spread - 0.5) * 30)
                
            # 6. Pattern consistency score (more consistent patterns are better for predictable trading)
            if pair['range_consistency'] is not None:
                consistency = pair['range_consistency']
                if consistency < 2:
                    # Very consistent daily ranges
                    consistency_score = 100
                elif consistency < 4:
                    # Good consistency
                    consistency_score = 80
                else:
                    # Less predictable
                    consistency_score = max(0, 80 - (consistency - 4) * 10)
            else:
                # No pattern data available
                consistency_score = 50
                
            # Combined score with weights tailored for short-term profit potential
            pair['score'] = (
                (volume_score * 0.2) +       # 20% weight on volume
                (volatility_score * 0.25) +   # 25% weight on volatility
                (price_score * 0.1) +         # 10% weight on price
                (change_score * 0.2) +        # 20% weight on recent performance
                (spread_score * 0.15) +       # 15% weight on spread
                (consistency_score * 0.1)     # 10% weight on pattern consistency
            )
        
        # Sort by score (descending)
        enriched_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        # Zwei-Phasen-Auswahl:
        # 1. Garantiere zunächst maximal max_standard_assets Standard-Coins
        standard_selected = [p for p in enriched_pairs if p['is_standard']][:max_standard_assets]
        
        # 2. Wähle dann die besten übrigen Coins, unabhängig davon ob Standard oder nicht
        # Entferne die bereits ausgewählten Standard-Coins aus der Pool
        remaining_pairs = [p for p in enriched_pairs if p not in standard_selected]
        
        # Wähle die besten übrigen Coins bis zum Erreichen von max_assets
        remaining_slots = max_assets - len(standard_selected)
        best_remaining = remaining_pairs[:remaining_slots]
        
        # Kombiniere beide Listen
        selected_pairs = standard_selected + best_remaining
        
        # Sortiere die finale Auswahl nochmal nach Score für die Ausgabe
        selected_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        return selected_pairs, market_stats
        
    except ccxt.NetworkError as e:
        print(f"Network error: {e}")
        sys.exit(1)
    except ccxt.ExchangeError as e:
        print(f"Exchange error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def generate_freqtrade_config(pairs: List[Dict[str, Any]], output_file: str = None) -> str:
    """
    Generate a Freqtrade-compatible static pair list configuration.
    
    Args:
        pairs: List of trading pairs with their details
        output_file: Optional file to write the configuration to
        
    Returns:
        Freqtrade configuration string in JSON format
    """
    # Extract only the pair symbols
    pair_symbols = [pair['symbol'] for pair in pairs]
    
    # Create config
    config_section = {
        "exchange": {
            "pair_whitelist": pair_symbols,
            "pair_blacklist": [
                ".*DOWN/.*", ".*UP/.*",
                ".*BEAR/.*", ".*BULL/.*",
                "BNB/.*"
            ]
        },
        "pairlists": [
            {"method": "StaticPairList"}
        ]
    }
    
    # Format JSON with indentation
    config_str = json.dumps(config_section, indent=4)
    
    # Print config
    print("\nFreqtrade Static Pair List Configuration:")
    print("-----------------------------------------")
    print(config_str)
    
    # Write to file if specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(config_str)
            print(f"\nConfiguration written to {output_file}")
        except Exception as e:
            print(f"Error writing to file: {e}")
    
    return config_str


def display_pair_info(pairs: List[Dict[str, Any]], capital_size: str = "small") -> None:
    """
    Display detailed information about the selected pairs.
    
    Args:
        pairs: List of trading pairs with their details
        capital_size: Size of capital, affects recommendations
    """
    # Separate standard and dynamic pairs for display
    standard_pairs = [p for p in pairs if p.get('is_standard', False)]
    dynamic_pairs = [p for p in pairs if not p.get('is_standard', False)]
    
    print("\nAusgewählte Trading-Paare:")
    print("---------------------")
    
    # Print header
    print(f"{'Symbol':<14} {'Price':<12} {'24h Change':<12} {'Volatility':<12} {'Volume (USD)':<15} {'Score':<8} {'Notes':<20}")
    print("-" * 95)
    
    # Calculate recommended position size
    if capital_size == "small":
        # For 50-100 USDC capital
        capital_est = 75
    elif capital_size == "medium":
        capital_est = 500
    else:
        capital_est = 2000
        
    # Calculate total score for distribution weighting
    total_score = sum(p['score'] for p in pairs)
    
    # Print all pairs sorted by score
    for pair in pairs:
        symbol = pair['symbol']
        price = pair['price']
        change = pair.get('change_24h', 0)
        volatility = pair.get('volatility', 0)
        volume = pair['volume']
        score = pair.get('score', 0)
        is_standard = pair.get('is_standard', False)
        
        # Calculate recommended position size based on score weighting
        weight = score / total_score
        position_size = round(capital_est * weight, 2)
        
        change_str = f"{change:+.2f}%" if change else "N/A"
        volatility_str = f"{volatility:.2f}%" if volatility else "N/A"
        
        # Add quality indicators and type
        notes = ""
        if is_standard:
            type_indicator = "🔷 STANDARD"
        else:
            type_indicator = "💠 ALTCOIN"
            
        if change < -5:
            notes += "⚠️ DOWNTREND "
        if volatility < 2:
            notes += "⚠️ LOW VOL "
        elif volatility > 15:
            notes += "⚠️ HIGH VOL "
        if volume < 100000:
            notes += "⚠️ LOW LIQUIDITY "
                
        if not notes:
            notes = f"✅ ~${position_size}"
        else:
            notes += f"(~${position_size})"
            
        # Add profit potential indicator
        if change > 3 and volatility >= 3 and volatility <= 15:
            notes = "🔥 HIGH POTENTIAL " + notes
        elif change > 0 and volatility >= 2 and volatility <= 12:
            notes = "✅ GOOD POTENTIAL " + notes
        
        print(f"{symbol:<14} {price:<12.6f} {change_str:<12} {volatility_str:<12} {volume:<15,.2f} {score:<8.1f} {type_indicator} {notes}")
    
    print(f"\nTotal pairs: {len(pairs)} ({len(standard_pairs)} standard + {len(dynamic_pairs)} alternative)")
    print(f"Hinweis: Die Coins sind nach Score sortiert (höher = besser), unabhängig vom Typ.")


def get_user_input():
    """
    Interactive console interface to get user preferences
    
    Returns:
        Dictionary of user preferences
    """
    print("\n==== Binance Pair List Generator für langfristigen Handel mit etablierten Coins ====")
    print("Dieses Tool hilft dir, optimale Trading-Paare für langfristigen Handel zu finden")
    
    # Get capital size
    print("\n1. Wie groß ist dein Handelskapital?")
    print("   1: Klein (50-100 USDC/USDT)")
    print("   2: Mittel (100-1000 USDC/USDT)")
    print("   3: Groß (>1000 USDC/USDT)")
    
    capital_choice = input("Wähle 1, 2 oder 3 [1]: ").strip() or "1"
    
    if capital_choice == "1":
        capital_size = "small"
    elif capital_choice == "2":
        capital_size = "medium"
    else:
        capital_size = "large"
    
    # Get quote currency
    print("\n2. Wähle deine Quote-Währung:")
    print("   Übliche Optionen: USDT, USDC, BUSD, BTC")
    quote_currency = input("Quote-Währung eingeben [USDT]: ").strip().upper() or "USDT"
    
    # Get max assets
    if capital_size == "small":
        default_assets = 6
        recommended_range = "4-6"
    elif capital_size == "medium":
        default_assets = 8
        recommended_range = "6-10"
    else:
        default_assets = 10
        recommended_range = "8-12"
        
    while True:
        try:
            max_assets = int(input(f"\n3. Wie viele Assets möchtest du insgesamt handeln? [empfohlen: {recommended_range}]: ") or default_assets)
            if max_assets <= 0:
                print("Bitte gib eine positive Zahl ein.")
                continue
            break
        except ValueError:
            print("Bitte gib eine gültige Zahl ein.")
    
    # Für langfristigen Handel mit etablierten Coins erhöhen wir die Anzahl der Standard-Coins
    suggested_std = min(max_assets // 2, max_assets - 1)  # Etwa die Hälfte als etablierte Coins
    while True:
        try:
            max_standard = int(input(f"\n4. Maximale Anzahl an etablierten Coins (BTC, ETH, etc.) [0-{max_assets}]: ") or suggested_std)
            if max_standard < 0:
                print("Bitte gib eine nicht-negative Zahl ein.")
                continue
            if max_standard > max_assets:
                print(f"Kann die Gesamtzahl an Assets ({max_assets}) nicht überschreiten.")
                continue
            break
        except ValueError:
            print("Bitte gib eine gültige Zahl ein.")
    
    # Get quality filtering preference
    filter_standard = input("\n5. Qualitätsfilterung für etablierte Coins anwenden? (j/n) [j]: ").strip().lower() != "n"
    
    # Get output file path
    output_file = input(f"\n6. Ausgabedateipfad [freqtrade_{quote_currency.lower()}_longterm.json]: ").strip() 
    if not output_file:
        output_file = f"freqtrade_{quote_currency.lower()}_longterm.json"
    
    # Standardmäßig auf langfristige Strategie setzen
    profit_strategy = "long_term"
    
    # Auto mode is always on for interactive use
    return {
        "quote_currency": quote_currency,
        "max_assets": max_assets,
        "max_standard_assets": max_standard,
        "filter_standard_coins": filter_standard,
        "output_file": output_file,
        "auto_mode": True,
        "capital_size": capital_size,
        "profit_strategy": profit_strategy
    }


def main():
    parser = argparse.ArgumentParser(description='Generate Freqtrade static pair list from Binance')
    parser.add_argument('--non-interactive', action='store_true',
                         help='Run in non-interactive mode (use command line arguments)')
    parser.add_argument('--manual', action='store_true',
                        help='Disable auto parameter determination (use defaults)')
    parser.add_argument('--no-standard', action='store_true',
                        help='Disable inclusion of standard reliable coins')
    parser.add_argument('--all-standard', action='store_true',
                        help='Include all standard coins without quality filtering')
    parser.add_argument('--output', type=str, default='freqtrade_pairlist.json', 
                        help='Output file for the configuration')
    parser.add_argument('--quote', type=str, default='USDT',
                        help='Quote currency to use (default: USDT)')
    parser.add_argument('--max-assets', type=int, default=8,
                        help='Maximum number of assets to include (default: 8)')
    parser.add_argument('--max-standard', type=int, default=2,
                        help='Maximum number of standard coins to include (default: 2)')
    parser.add_argument('--capital', type=str, choices=['small', 'medium', 'large'], default='small',
                        help='Capital size (small: 50-100, medium: 100-1000, large: 1000+)')
    
    args = parser.parse_args()
    
    # Determine if we should run in interactive mode
    if not args.non_interactive:
        # Get user preferences interactively
        preferences = get_user_input()
        
        # Extract values from preferences
        quote_currency = preferences["quote_currency"]
        max_assets = preferences["max_assets"]
        max_standard_assets = preferences["max_standard_assets"]  # Fixed: changed from max_standard_assets
        filter_standard_coins = preferences["filter_standard_coins"]
        output_file = preferences["output_file"]
        auto_mode = preferences["auto_mode"]
        capital_size = preferences["capital_size"]
    else:
        # Use command line arguments
        quote_currency = args.quote
        max_assets = args.max_assets
        max_standard_assets = args.max_standard
        filter_standard_coins = not args.all_standard
        output_file = args.output
        auto_mode = not args.manual
        capital_size = args.capital
    
    print(f"\nAusführung mit folgenden Einstellungen:")
    print(f"- Quote-Währung: {quote_currency}")
    print(f"- Kapital: {capital_size}")
    print(f"- Max Assets gesamt: {max_assets}")
    print(f"- Max Standard Assets: {max_standard_assets}")
    print(f"- Qualitätsfilterung: {'Aktiviert' if filter_standard_coins else 'Deaktiviert'}")
    print(f"- Ausgabedatei: {output_file}")
    print(f"- Auto-Parameter Modus: {'Aktiviert' if auto_mode else 'Deaktiviert'}")
    
    # Fetch pairs
    pairs, market_stats = fetch_binance_pairs(
        quote_currency=quote_currency,
        auto_mode=auto_mode,
        include_standard_coins=True,  # Always include standard coins in interactive mode
        filter_standard_coins=filter_standard_coins,
        max_assets=max_assets,
        max_standard_assets=max_standard_assets,  # Fixed: changed from max_standard_assets
        capital_size=capital_size
    )
    
    if not pairs:
        print("Keine geeigneten Paare gefunden. Versuche, die Parameter anzupassen oder überprüfe deine Verbindung.")
        sys.exit(1)
    
    # Display pair info
    display_pair_info(pairs, capital_size)
    
    # Generate config
    generate_freqtrade_config(pairs, output_file)
    
    print("\nTool erfolgreich ausgeführt!")
    print("\nEmpfohlene Freqtrade-Einstellungen für deine Konfiguration:")
    print("--------------------------------------------------------")
    
    # Tailor recommendations based on capital size
    if capital_size == "small":
        max_trades = min(max_assets, max_assets-1)
        percent_per_trade = round(100 / max_trades, 1)
        print(f"1. max_open_trades: {max_trades}")
        print(f"2. stake_amount: \"percentage\"")
        print(f"   stake_amount_percentage: {percent_per_trade}  # Etwa {percent_per_trade}% deines Kapitals pro Trade")
        print("   # Alternativ für feste Beträge: stake_amount = 10")
        print("3. tradable_balance_ratio: 0.95")
        print("4. Empfohlene minimal_roi:")
        print("   minimal_roi = {")
        print("       '0': 0.04,")
        print("       '30': 0.02,")
        print("       '60': 0.01")
        print("   }")
        print("5. Empfohlener stoploss: -0.06")
        print("6. Empfohlener trailing_stop: True")
        print("7. trailing_stop_positive: 0.01")
        print("8. trailing_stop_positive_offset: 0.02")
    elif capital_size == "medium":
        max_trades = min(max_assets, max_assets-2)
        percent_per_trade = round(100 / max_trades, 1)
        print(f"1. max_open_trades: {max_trades}")
        print(f"2. stake_amount: \"percentage\"")
        print(f"   stake_amount_percentage: {percent_per_trade}  # Etwa {percent_per_trade}% deines Kapitals pro Trade")
        print("   # Alternativ für feste Beträge: stake_amount = 50")
        print("3. tradable_balance_ratio: 0.95")
        print("4. Empfohlene minimal_roi:")
        print("   minimal_roi = {")
        print("       '0': 0.05,")
        print("       '40': 0.02,")
        print("       '90': 0.01")
        print("   }")
        print("5. Empfohlener stoploss: -0.07")
        print("6. Empfohlener trailing_stop: True")
        print("7. trailing_stop_positive: 0.01")
        print("8. trailing_stop_positive_offset: 0.025")
    else:  # large capital
        max_trades = min(max_assets, max_assets-3)
        percent_per_trade = round(100 / max_trades, 1)
        print(f"1. max_open_trades: {max_trades}")
        print(f"2. stake_amount: \"percentage\"")
        print(f"   stake_amount_percentage: {percent_per_trade}  # Etwa {percent_per_trade}% deines Kapitals pro Trade")
        print("   # Alternativ: stake_amount = 100 oder höher")
        print("3. tradable_balance_ratio: 0.95")
        print("4. Empfohlene minimal_roi:")
        print("   minimal_roi = {")
        print("       '0': 0.06,")
        print("       '60': 0.03,")
        print("       '120': 0.01")
        print("   }")
        print("5. Empfohlener stoploss: -0.08")
        print("6. Empfohlener trailing_stop: True")
        print("7. trailing_stop_positive: 0.02")
        print("8. trailing_stop_positive_offset: 0.03")
    
    # Angepasste Empfehlungen für langfristigen Handel
    print("\nEmpfohlene Freqtrade-Einstellungen für langfristigen Handel:")
    print("--------------------------------------------------------")
    
    if capital_size == "small":
        max_trades = min(max_assets, max_assets-1)
        percent_per_trade = round(100 / max_trades, 1)
        print(f"1. max_open_trades: {max_trades}")
        print(f"2. stake_amount: \"percentage\"")
        print(f"   stake_amount_percentage: {percent_per_trade}")
        print("3. tradable_balance_ratio: 0.95")
        print("4. Empfohlene minimal_roi für langfristigen Handel:")
        print("   minimal_roi = {")
        print("       '0': 0.08,      # Höhere Gewinnziele für langfristigen Handel")
        print("       '60': 0.05,")
        print("       '180': 0.03,")
        print("       '720': 0.02")
        print("   }")
        print("5. Empfohlener stoploss: -0.10  # Weiterer Stoploss für langfristigen Handel")
        print("6. Empfohlener trailing_stop: True")
        print("7. trailing_stop_positive: 0.02")
        print("8. trailing_stop_positive_offset: 0.04")


if __name__ == "__main__":
    main()
