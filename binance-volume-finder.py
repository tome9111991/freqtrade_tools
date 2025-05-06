#!/usr/bin/env python3
"""
Binance Volume Finder for Freqtrade
-----------------------------------
This script finds the highest volume trading pairs on Binance for a specific quote currency
and outputs them in a format ready to use in Freqtrade's static pair list.
"""

import ccxt
import json
from datetime import datetime
import questionary
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

def get_high_volume_pairs(quote_currency='USDC', num_pairs=20, min_volume=0, min_volume_percent=0, 
                        exchange_id='binance', use_blacklist=True, 
                        apply_fiat_filter=True, apply_stablecoin_filter=True,
                        apply_tokenized_stocks_filter=True, apply_high_supply_filter=True,
                        apply_low_mcap_filter=False):
    """
    Get the highest volume trading pairs from Binance for a specific quote currency.
    
    Args:
        quote_currency (str): Quote currency to filter for (e.g., 'USDC', 'USDT')
        num_pairs (int): Number of pairs to return
        min_volume (float): Minimum 24h volume in quote currency (absolute value)
        min_volume_percent (float): Minimum volume as percentage of highest volume pair (0-100)
        exchange_id (str): Exchange ID to use (default: 'binance')
        use_blacklist (bool): Whether to apply standard blacklist filters
        apply_fiat_filter (bool): Filter out FIAT pairs
        apply_stablecoin_filter (bool): Filter out stablecoin vs stablecoin pairs
        apply_tokenized_stocks_filter (bool): Filter out tokenized stocks
        apply_high_supply_filter (bool): Filter out high supply tokens
        apply_low_mcap_filter (bool): Filter out low market cap coins
        
    Returns:
        list: List of trading pairs sorted by volume
    """
    
    # Default blacklist patterns
    default_blacklist = [
        # Leveraged tokens
        r".*DOWN/.*",     # Leveraged DOWN tokens
        r".*UP/.*",       # Leveraged UP tokens
        r".*BEAR/.*",     # Bear tokens
        r".*BULL/.*",     # Bull tokens
        r".*HEDGE/.*",    # Hedged tokens
        
        # Base pairs
        r"BNB/.*",        # BNB base pairs
        
        # Special tokens
        r".*_PREMIUM/.*", # Premium index
        r".*SUSD/.*",     # Synthetic USD
        r".*BVOL/.*",     # Bitcoin Volatility tokens
        r".*1000SHIB/.*", # 1000SHIB and similar high-unit tokens
        r".*1000XEC/.*",
        r".*BTCST/.*",    # Bitcoin Standard Hashrate Token
        
        # FIAT pairs
        r".*/EUR",        # Euro pairs
        r".*/GBP",        # British Pound pairs
        r".*/AUD",        # Australian Dollar pairs
        r".*/TRY",        # Turkish Lira pairs
        r".*/BRL",        # Brazilian Real pairs
        r".*/RUB",        # Russian Ruble pairs
        r".*/JPY",        # Japanese Yen pairs
        r".*/KRW",        # Korean Won pairs
        
        # Stablecoin pairs
        r".*BKRW/.*",     # Korean won-pegged stablecoin pairs
        r".*BUSD/.*",     # Old BUSD pairs
        r"USDT/.*",       # Inverted USDT pairs
        r"USDC/.*",       # Inverted USDC pairs
        r"DAI/.*",        # Inverted DAI pairs
        
        # Other problematic patterns
        r".*TUSD/.*",     # TrueUSD pairs
        r".*PAXG/.*",     # PAX Gold
        r".*VAI/.*",      # VAI stablecoin
        r".*HOOKS/.*",    # Hooks token
        r".*DYDX/.*",     # dYdX token
        r".*DUSD/.*",     # DUSD stablecoin
        r".*USDS/.*"      # Stably USD
    ]
    # Create a console for rich output
    console = Console()
    
    try:
        # Initialize the exchange
        console.print(f"[bold cyan]Connecting to {exchange_id.capitalize()}...[/bold cyan]")
        exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })
        
        # Load markets
        console.print("[bold cyan]Loading markets...[/bold cyan]")
        markets = exchange.load_markets()
        
        # Filter for quote currency
        quote_pairs = [symbol for symbol in markets.keys() 
                      if symbol.endswith('/' + quote_currency)]
        
        # Apply blacklist if enabled
        if use_blacklist:
            import re
            filtered_pairs = []
            for pair in quote_pairs:
                # Check if pair matches any blacklist pattern
                blacklisted = False
                
                # Core patterns that are always blocked if blacklist is enabled
                core_patterns = [
                    r".*DOWN/.*", r".*UP/.*", r".*BEAR/.*", r".*BULL/.*", r".*HEDGE/.*",
                    r"BNB/.*", r".*_PREMIUM/.*", r".*BVOL/.*"
                ]
                
                # Optional filter patterns
                fiat_patterns = [
                    # Filter FIAT as quote currency
                    r".*/EUR", r".*/GBP", r".*/AUD", r".*/TRY", r".*/BRL", 
                    r".*/RUB", r".*/JPY", r".*/KRW",
                    # Filter FIAT as base currency
                    r"EUR/.*", r"GBP/.*", r"AUD/.*", r"TRY/.*", r"BRL/.*",
                    r"RUB/.*", r"JPY/.*", r"KRW/.*"
                ] if apply_fiat_filter else []
                
                stablecoin_patterns = [
                    r".*BUSD/.*", r"USDT/.*", r"USDC/.*", r"DAI/.*", 
                    r".*TUSD/.*", r".*SUSD/.*", r".*DUSD/.*", r".*USDS/.*", r".*VAI/.*"
                ] if apply_stablecoin_filter else []
                
                tokenized_stock_patterns = [
                    r".*STOCK/.*", r".*MICRO/.*", r".*TESLA/.*", r".*BABA/.*", r".*COIN/.*"
                ] if apply_tokenized_stocks_filter else []
                
                high_supply_patterns = [
                    r".*1000SHIB/.*", r".*1000XEC/.*", r".*BTT/.*", r".*SAFEMOON/.*"
                ] if apply_high_supply_filter else []
                
                # Combine all relevant patterns
                active_patterns = core_patterns + fiat_patterns + stablecoin_patterns + tokenized_stock_patterns + high_supply_patterns
                
                # Check if pair matches any active pattern
                for pattern in active_patterns:
                    if re.match(pattern, pair):
                        blacklisted = True
                        break
                
                if not blacklisted:
                    filtered_pairs.append(pair)
                    
            blacklisted_count = len(quote_pairs) - len(filtered_pairs)
            console.print(f"[green]Found {len(quote_pairs)} {quote_currency} pairs[/green]")
            if blacklisted_count > 0:
                console.print(f"[yellow]Filtered out {blacklisted_count} blacklisted pairs[/yellow]")
            quote_pairs = filtered_pairs
        
        # Get 24h ticker data for all pairs
        console.print(f"[bold cyan]Fetching volume data for {len(quote_pairs)} pairs...[/bold cyan]")
        
        pair_volumes = []
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Fetching ticker data...", total=len(quote_pairs))
            
            for pair in quote_pairs:
                try:
                    ticker = exchange.fetch_ticker(pair)
                    volume_usd = ticker['quoteVolume']  # Volume in quote currency
                    
                    if volume_usd >= min_volume:
                        pair_volumes.append({
                            'pair': pair,
                            'volume': volume_usd,
                            'price': ticker['last'],
                            'change': ticker['percentage']
                        })
                    
                except Exception as e:
                    console.print(f"[yellow]Error fetching data for {pair}: {str(e)}[/yellow]")
                
                progress.update(task, advance=1)
        
        # Sort by volume
        pair_volumes.sort(key=lambda x: x['volume'], reverse=True)
        
        # If we have pairs and a percentage filter is set
        if pair_volumes and min_volume_percent > 0:
            # Get the volume of the highest volume pair
            max_volume = pair_volumes[0]['volume']
            # Filter pairs that have at least the min percentage volume
            min_volume_value = max_volume * (min_volume_percent / 100)
            pair_volumes = [p for p in pair_volumes if p['volume'] >= min_volume_value]
            console.print(f"[cyan]Filtered to pairs with at least {min_volume_percent}% of top volume ({min_volume_value:.2f} {quote_currency})[/cyan]")
        
        # Get top pairs
        top_pairs = pair_volumes[:num_pairs]
        
        return top_pairs
    
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return []

def display_pairs(pairs, quote_currency):
    """
    Display the pairs in a nice table format
    """
    console = Console()
    
    table = Table(title=f"Top {len(pairs)} {quote_currency} Pairs by Volume")
    
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Pair", style="green")
    table.add_column("Volume (24h)", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Change (24h)", justify="right")
    
    for i, pair_data in enumerate(pairs, 1):
        # Format the volume with commas and 2 decimal places
        volume_formatted = f"{pair_data['volume']:,.2f}"
        
        # Format the price (use scientific notation for very small numbers)
        if pair_data['price'] < 0.001:
            price_formatted = f"{pair_data['price']:.8f}"
        else:
            price_formatted = f"{pair_data['price']:.4f}"
        
        # Format the percentage change and color accordingly
        percentage = pair_data['change']
        if percentage > 0:
            change_formatted = f"[green]+{percentage:.2f}%[/green]"
        elif percentage < 0:
            change_formatted = f"[red]{percentage:.2f}%[/red]"
        else:
            change_formatted = f"{percentage:.2f}%"
        
        table.add_row(
            str(i),
            pair_data['pair'],
            volume_formatted,
            price_formatted,
            change_formatted
        )
    
    console.print(table)

def format_for_freqtrade(pairs):
    """
    Format the pairs for use in Freqtrade's static pair list
    """
    # Convert from "BTC/USDC" format to "BTC/USDC" format (already in the right format)
    formatted_pairs = [pair['pair'] for pair in pairs]
    
    # Create the static_pair_list entry
    static_list = json.dumps(formatted_pairs, indent=4)
    
    return static_list

def main():
    """
    Main function to run the script interactively
    """
    console = Console()
    console.print(f"\n[bold]Binance Volume Finder for Freqtrade[/bold]")
    console.print(f"Running at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Get user inputs interactively using questionary
    quote_currency = questionary.text(
        "Enter quote currency:",
        default="USDC"
    ).ask()
    
    num_pairs = questionary.text(
        "How many pairs do you want to find?",
        default="20",
        validate=lambda text: text.isdigit() and int(text) > 0
    ).ask()
    num_pairs = int(num_pairs)
    
    # Ask for volume filter type
    volume_filter_type = questionary.select(
        "How do you want to filter by volume?",
        choices=[
            "Absolute value (specific amount in quote currency)",
            "Percentage (% of highest volume pair)"
        ]
    ).ask()
    
    min_volume = 0
    min_volume_percent = 0
    
    if "Absolute" in volume_filter_type:
        min_volume = questionary.text(
            "Minimum 24h volume (in quote currency):",
            default="0",
            validate=lambda text: text.replace('.', '', 1).isdigit()
        ).ask()
        min_volume = float(min_volume)
    else:
        min_volume_percent = questionary.text(
            "Minimum volume (% of highest volume pair):",
            default="1",
            validate=lambda text: text.replace('.', '', 1).isdigit() and float(text) >= 0 and float(text) <= 100
        ).ask()
        min_volume_percent = float(min_volume_percent)
    
    # Ask about using blacklist
    use_blacklist = questionary.confirm(
        "Do you want to use the standard blacklist to filter problematic pairs?",
        default=True
    ).ask()
    
    # Use simple confirm questions instead of checkbox
    apply_fiat_filter = False
    apply_stablecoin_filter = False
    apply_tokenized_stocks_filter = False
    apply_high_supply_filter = False
    apply_low_mcap_filter = False
    
    if use_blacklist:
        # Individual yes/no questions for each filter type
        apply_fiat_filter = questionary.confirm(
            "Filter out FIAT pairs (EUR, GBP, etc.)?",
            default=True
        ).ask()
        
        apply_stablecoin_filter = questionary.confirm(
            "Filter out stablecoin vs stablecoin pairs?",
            default=True
        ).ask()
        
        apply_tokenized_stocks_filter = questionary.confirm(
            "Filter out tokenized stocks?",
            default=True
        ).ask()
        
        apply_high_supply_filter = questionary.confirm(
            "Filter out high supply tokens (like 1000SHIB)?",
            default=True
        ).ask()
        
        apply_low_mcap_filter = questionary.confirm(
            "Filter out low market cap coins (outside top 200)?",
            default=False
        ).ask()
    
    exchange_options = ['binance', 'kucoin', 'huobi', 'okex', 'gate']
    exchange_id = questionary.select(
        "Select exchange:",
        choices=exchange_options,
        default='binance'
    ).ask()
    
    # Get the pairs
    pairs = get_high_volume_pairs(
        quote_currency=quote_currency,
        num_pairs=num_pairs,
        min_volume=min_volume,
        min_volume_percent=min_volume_percent,
        exchange_id=exchange_id,
        use_blacklist=use_blacklist,
        apply_fiat_filter=apply_fiat_filter,
        apply_stablecoin_filter=apply_stablecoin_filter,
        apply_tokenized_stocks_filter=apply_tokenized_stocks_filter,
        apply_high_supply_filter=apply_high_supply_filter,
        apply_low_mcap_filter=apply_low_mcap_filter
    )
    
    if not pairs:
        console.print("[bold red]No pairs found matching the criteria.[/bold red]")
        return
    
    # Display the pairs
    display_pairs(pairs, quote_currency)
    
    # Format for Freqtrade
    freqtrade_format = format_for_freqtrade(pairs)
    
    console.print("\n[bold]Freqtrade Static Pair List Format:[/bold]")
    console.print(f"[yellow]\"exchange\": {{[/yellow]")
    console.print(f"[yellow]    \"pair_whitelist\": {freqtrade_format},[/yellow]")
    
    # Print standard blacklist recommendation
    console.print(f"[yellow]    \"pair_blacklist\": [")
    console.print(f"[yellow]        \".*DOWN/.*\",")
    console.print(f"[yellow]        \".*UP/.*\",")
    console.print(f"[yellow]        \".*BEAR/.*\",")
    console.print(f"[yellow]        \".*BULL/.*\",")
    console.print(f"[yellow]        \"BNB/.*\",")
    console.print(f"[yellow]        \".*/EUR\",")
    console.print(f"[yellow]        \".*/GBP\",")
    console.print(f"[yellow]        \".*/AUD\",")
    console.print(f"[yellow]        \".*1000SHIB/.*\",")
    console.print(f"[yellow]        \".*TUSD/.*\"")
    console.print(f"[yellow]    ][/yellow]")
    console.print(f"[yellow]}}[/yellow]")
    
    # Ask if user wants to save to a file
    save_to_file = questionary.confirm("Do you want to save this list to a file?").ask()
    
    if save_to_file:
        filename = questionary.text(
            "Enter filename to save (e.g., whitelist.json):",
            default="whitelist.json"
        ).ask()
        
        try:
            with open(filename, 'w') as f:
                json.dump({"pair_whitelist": [pair['pair'] for pair in pairs]}, f, indent=4)
            console.print(f"[bold green]✓ Saved to {filename}![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error saving file: {str(e)}[/bold red]")
    
    console.print("\n[bold green]✓ Done![/bold green]")

if __name__ == "__main__":
    main()
