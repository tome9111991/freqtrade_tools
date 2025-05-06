# --- START OF FILE findSimilarCoins_CorrAndVol_v1.5_inclRefWhitelist.py ---

# Installation notwendig:
# pip install requests python-binance pycoingecko pandas tqdm numpy scikit-learn rich

import time
import pandas as pd
from binance.client import Client
from pycoingecko import CoinGeckoAPI
import requests
import threading
from queue import Queue, Empty
from datetime import datetime, timedelta, timezone
import numpy as np
from tqdm import tqdm
import warnings # Um die FutureWarning gezielt zu ignorieren

# --- Rich für schönere Ausgabe ---
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress
from rich.rule import Rule

# --- Pandas FutureWarning ignorieren (optional, falls sie stören) ---
# warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Konfiguration ---
REFERENCE_COIN_SYMBOL = "BTC" # Referenz-Coin
TARGET_QUOTE_ASSET = "USDC"   # Ziel-Quote-Asset
LOOKBACK_DAYS = 90            # Tage zurück für Analyse

# --- Filter für Ähnlichkeit ---
MIN_CORRELATION = 0.70        # Mindest-Korrelation der täglichen Returns (Richtung)
MAX_VOLATILITY_RATIO_DIFFERENCE = 0.4 # Max. Abweichung der Vola-Ratio (z.B. 0.4 => 0.6x bis 1.4x)

# --- Filter für Kandidaten-Qualität ---
MIN_MARKET_CAP_USD = 500_000_000
MIN_TOTAL_VOLUME_USD = 3_000_000 # CoinGecko Gesamtvolumen
MIN_BINANCE_USDC_VOLUME_USD = 300_000 # Heutiges Volumen auf Binance

# --- Technische Einstellungen ---
TOP_N_COINS_TO_CHECK = 300
NUM_THREADS = 15

# --- Globale Variablen ---
global_start_date = None
ref_avg_daily_range = None

# --- Initialisierung ---
console = Console()
cg = CoinGeckoAPI()
try:
    binance_client = Client(requests_params={"timeout": 20})
    binance_client.ping()
except Exception as e:
    console.print(f"[bold red]Fehler bei der Initialisierung des Binance Clients:[/bold red] {e}")
    exit()

# --- Thread-sichere Speicherung ---
results_list = []
results_lock = threading.Lock()
processed_counter = 0
skipped_counter = 0

# --- Hilfsfunktionen ---
def safe_api_call(func, *args, **kwargs):
    max_retries = 4
    delay = 5
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException, requests.exceptions.ReadTimeout) as e:
            time.sleep(delay)
            delay *= 1.5
        except Exception as e:
            return None
    return None

def get_daily_data(symbol, start_str, end_str, lookback_days):
    klines = safe_api_call(binance_client.get_historical_klines,
                           symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY,
                           start_str=start_str, end_str=end_str, limit=1000)

    if not klines: return None, None

    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    df = pd.DataFrame(klines, columns=columns)
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms', utc=True)
    for col in ['High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['High', 'Low', 'Close'])
    df = df.set_index('Close time')
    df = df[['High', 'Low', 'Close']]

    if global_start_date is None:
         console.print("[bold red]FEHLER: global_start_date nicht gesetzt![/]")
         return None, None
    df = df.loc[df.index >= global_start_date]

    if len(df) < lookback_days * 0.7:
        return None, None

    df['Daily_Range_Perc'] = 0.0
    valid_close = df['Close'] > 1e-9
    df.loc[valid_close, 'Daily_Range_Perc'] = ((df['High'] - df['Low']) / df['Close']) * 100
    df = df[df['Daily_Range_Perc'] >= 0]
    df = df[df['Daily_Range_Perc'] < 200]

    df['Return'] = df['Close'].pct_change()
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Return', 'Daily_Range_Perc'])

    if df_clean.empty:
        return None, None

    return df_clean['Return'], df_clean['Daily_Range_Perc']

# --- Worker-Funktion ---
def process_coin(coin_queue, binance_tradable_symbols, ref_returns_series, start_str, end_str, lookback_days, pbar):
    global processed_counter, skipped_counter
    while not coin_queue.empty():
        try:
            coin = coin_queue.get(block=False)
        except Empty:
            break

        candidate_symbol_base = coin['symbol']
        candidate_symbol_binance = f"{candidate_symbol_base}{TARGET_QUOTE_ASSET}"
        full_pair_string = f"{candidate_symbol_base}/{TARGET_QUOTE_ASSET}"

        if candidate_symbol_binance not in binance_tradable_symbols:
             with results_lock: skipped_counter += 1;
             coin_queue.task_done(); # pbar.update(1) # Wird im finally gemacht
             continue # Direkt weiter, da nicht handelbar

        try:
            # 1. Heutiges Volumen prüfen
            ticker_data = safe_api_call(binance_client.get_ticker, symbol=candidate_symbol_binance)
            if not ticker_data:
                with results_lock: skipped_counter += 1
                continue
            try: pair_volume_quote_today = float(ticker_data.get('quoteVolume', 0))
            except (ValueError, TypeError): pair_volume_quote_today = 0

            if pair_volume_quote_today < MIN_BINANCE_USDC_VOLUME_USD:
                 if candidate_symbol_base != REFERENCE_COIN_SYMBOL:
                     with results_lock: skipped_counter += 1
                     continue

            # 2. Historische Daten holen (Returns UND Daily Range)
            candidate_returns, candidate_daily_range = get_daily_data(candidate_symbol_binance, start_str, end_str, lookback_days)

            if candidate_returns is None or candidate_returns.empty or candidate_daily_range is None or candidate_daily_range.empty:
                with results_lock: skipped_counter += 1
                continue

            # 3. Korrelation berechnen
            if ref_returns_series is None or ref_returns_series.empty: continue

            common_index = ref_returns_series.index.intersection(candidate_returns.index)
            required_overlap = max(10, int(lookback_days * 0.6))
            if len(common_index) < required_overlap:
                with results_lock: skipped_counter += 1
                continue

            ref_aligned = ref_returns_series.loc[common_index]
            candidate_aligned = candidate_returns.loc[common_index]

            if ref_aligned.empty or candidate_aligned.empty:
                 with results_lock: skipped_counter += 1
                 continue

            try:
                if ref_aligned.std() < 1e-10 or candidate_aligned.std() < 1e-10: correlation = 0.0
                else: correlation = ref_aligned.corr(candidate_aligned, method='pearson')
                if pd.isna(correlation): correlation = 0.0
            except Exception: correlation = 0.0

            # 4. Volatilitäts-Vergleich
            candidate_avg_daily_range = candidate_daily_range.mean()
            volatility_ratio = 0.0
            if ref_avg_daily_range is not None and ref_avg_daily_range > 1e-9:
                volatility_ratio = candidate_avg_daily_range / ref_avg_daily_range

            # 5. BEIDE Filter anwenden
            is_correlation_ok = correlation >= MIN_CORRELATION
            is_volatility_similar = abs(1.0 - volatility_ratio) <= MAX_VOLATILITY_RATIO_DIFFERENCE

            if is_correlation_ok and is_volatility_similar:
                with results_lock:
                    results_list.append({
                        'Pair': full_pair_string,
                        'Symbol': candidate_symbol_base,
                        'Name': coin['name'],
                        'Market_Cap_Raw': coin.get('market_cap_usd', 0),
                        'Vol_Today_Raw': pair_volume_quote_today,
                        'Correlation': correlation,
                        'Avg_Daily_Range_Perc': candidate_avg_daily_range,
                        'Volatility_Ratio': volatility_ratio,
                        'Compared_Days': len(common_index)
                    })
            else:
                 # Zählt als übersprungen, da Filter nicht passten
                 with results_lock: skipped_counter += 1

        except Exception as e:
            # Fehler im Try-Block zählt auch als skipped
            with results_lock: skipped_counter += 1
            pass
        finally:
            # Wird immer ausgeführt, zählt jeden Versuch als 'processed'
            coin_queue.task_done()
            with results_lock:
                processed_counter += 1
                pbar.update(1)

# ======== Hauptprogrammablauf ========

# --- Header ---
header_text = Text(f"Krypto Coin Ähnlichkeits-Finder v1.5 (Korr+Vola, inkl. Ref)", style="bold white", justify="center")
sub_text = Text(f"Sucht ähnliche Coins zu {REFERENCE_COIN_SYMBOL}/{TARGET_QUOTE_ASSET}", style="cyan", justify="center")
console.print(Panel(Text.assemble(header_text, "\n", sub_text), border_style="blue", padding=(1, 2)))

# --- Schritt 0: Zeitrahmen ---
console.print(Rule("[bold yellow]Schritt 0: Setup[/]"))
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=LOOKBACK_DAYS)
global_start_date = start_date
start_str = str(int(start_date.timestamp() * 1000))
end_str = str(int(end_date.timestamp() * 1000))
console.print(f"Analysezeitraum: [cyan]{start_date.strftime('%Y-%m-%d')}[/] bis [cyan]{end_date.strftime('%Y-%m-%d')}[/] ({LOOKBACK_DAYS} Tage)")
console.print(f"[green]Binance API erreichbar.[/]")

# --- Schritt 1: Referenz-Coin Daten & Volatilität ---
console.print(Rule("[bold yellow]Schritt 1: Referenz-Coin Daten & Volatilität[/]"))
console.print(f"Lade historische Daten für Referenz-Coin [bold cyan]{REFERENCE_COIN_SYMBOL}/{TARGET_QUOTE_ASSET}[/]...")
ref_symbol_binance = f"{REFERENCE_COIN_SYMBOL}{TARGET_QUOTE_ASSET}"
ref_returns, ref_daily_range = get_daily_data(ref_symbol_binance, start_str, end_str, LOOKBACK_DAYS)

if ref_returns is None or ref_returns.empty or ref_daily_range is None or ref_daily_range.empty:
    console.print(f"\n[bold red]FEHLER:[/bold red] Konnte nicht genügend historische Daten (Returns/Range) für [bold cyan]{ref_symbol_binance}[/] laden.")
    exit()

ref_avg_daily_range = ref_daily_range.mean()
console.print(f"[green]Daten für {REFERENCE_COIN_SYMBOL} geladen.[/]")
console.print(f"  -> Avg. Daily Returns: {len(ref_returns)} Tage")
console.print(f"  -> [bold]Avg. Daily Range: {ref_avg_daily_range:.2f}%[/] ({len(ref_daily_range)} Tage)")

# --- Schritt 2: CoinGecko Kandidaten ---
console.print(Rule("[bold yellow]Schritt 2: CoinGecko Kandidaten[/]"))
console.print(f"Abrufen der Top {TOP_N_COINS_TO_CHECK} Coins von CoinGecko...")
all_cg_coins_raw = []
page = 1
fetched_count = 0
max_per_page = 250
try:
    while fetched_count < TOP_N_COINS_TO_CHECK:
        fetch_limit = min(max_per_page, TOP_N_COINS_TO_CHECK - fetched_count)
        if fetch_limit <= 0: break
        current_page_data = safe_api_call(cg.get_coins_markets, vs_currency='usd', order='market_cap_desc', per_page=fetch_limit, page=page, sparkline=False)
        if not current_page_data:
            # console.print(f"[yellow]Warnung: Konnte Seite {page} von CoinGecko nicht abrufen oder Seite war leer.[/]")
            break
        all_cg_coins_raw.extend(current_page_data)
        fetched_count += len(current_page_data)
        page += 1
        if len(current_page_data) < fetch_limit: break
        if fetched_count < TOP_N_COINS_TO_CHECK: time.sleep(1.0)
except Exception as e:
    console.print(f"[yellow]Warnung: Fehler beim Holen weiterer CoinGecko Seiten: {e}[/]")

if not all_cg_coins_raw:
    console.print("[bold red]Fehler: Konnte keine Marktdaten von CoinGecko abrufen.[/]"); exit()

unique_coins_dict = {}
for coin in all_cg_coins_raw:
    coin_id = coin.get('id')
    if coin_id and coin_id not in unique_coins_dict:
        unique_coins_dict[coin_id] = coin
coins_market_data = list(unique_coins_dict.values())
console.print(f"[green]{len(all_cg_coins_raw)} Roh-Coins von CoinGecko erhalten, {len(coins_market_data)} unique IDs nach Filterung.[/]")

# --- Schritt 3: Vorbereitung der Kandidaten ---
console.print(Rule("[bold yellow]Schritt 3: Vorbereitung der Kandidaten[/]"))
console.print(f"Filtere unique Kandidaten (MarketCap >= [cyan]${MIN_MARKET_CAP_USD:,.0f}[/], GesamtVol >= [cyan]${MIN_TOTAL_VOLUME_USD:,.0f}[/])...")
initial_filtered_coins = []
for coin in coins_market_data:
    market_cap = coin.get('market_cap')
    total_volume = coin.get('total_volume')
    symbol = coin.get('symbol', '').upper()
    coin_id = coin.get('id')
    coin_name = coin.get('name')
    if market_cap and total_volume and symbol and coin_name:
        if market_cap >= MIN_MARKET_CAP_USD and total_volume >= MIN_TOTAL_VOLUME_USD:
            initial_filtered_coins.append({
                'id': coin_id, 'symbol': symbol, 'name': coin_name, 'market_cap_usd': market_cap
            })

console.print(f"[green]{len(initial_filtered_coins)} Kandidaten nach Markt-/Volumen-Filter übrig.[/]")
if not initial_filtered_coins:
    console.print("[yellow]Keine Kandidaten übrig. Programmende.[/]"); exit()

console.print(f"Prüfe auf Binance Handelbarkeit für Quote Asset [cyan]{TARGET_QUOTE_ASSET}[/]...")
all_binance_symbols_info = safe_api_call(binance_client.get_exchange_info)
binance_tradable_symbols = set()
if all_binance_symbols_info:
    try:
        binance_tradable_symbols = {s['symbol'] for s in all_binance_symbols_info.get('symbols', [])
                                    if s['status'] == 'TRADING' and TARGET_QUOTE_ASSET in s['quoteAsset']}
        console.print(f"[green]{len(binance_tradable_symbols)} handelbare {TARGET_QUOTE_ASSET}-Paare auf Binance gefunden.[/]")
    except Exception as e:
        console.print(f"[bold red]Fehler beim Verarbeiten der Binance Symbolinformationen:[/bold red] {e}")
        binance_tradable_symbols = set()
else:
    console.print("[bold red]Fehler: Konnte keine Symbole von Binance abrufen. Breche ab.[/]"); exit()

if not binance_tradable_symbols:
     console.print(f"[bold yellow]WARNUNG:[/bold yellow] Keine handelbaren Symbole für Quote Asset '[cyan]{TARGET_QUOTE_ASSET}[/]' auf Binance gefunden.")

coin_queue = Queue()
valid_candidates_for_queue = 0
skipped_in_prep = 0
for coin in initial_filtered_coins:
    binance_pair_name = f"{coin['symbol']}{TARGET_QUOTE_ASSET}"
    if binance_pair_name in binance_tradable_symbols :
         coin_queue.put(coin)
         valid_candidates_for_queue += 1
    else:
        skipped_in_prep += 1

total_to_process = valid_candidates_for_queue
console.print(f"[green]{total_to_process} Kandidaten zur Analyse vorbereitet.[/] ([yellow]{skipped_in_prep}[/] in Vorbereitung übersprungen: nicht auf Binance handelbar).")

if total_to_process == 0:
    console.print(f"\n[yellow]Keine gültigen Kandidaten zum Vergleichen übrig.[/]")
    exit()

# --- Schritt 4: Bestätigung ---
console.print(Rule("[bold yellow]Schritt 4: Bestätigung der Analyse[/]"))
summary_text = Text.assemble(
    ("Referenz-Coin:", "bold white"), f" {REFERENCE_COIN_SYMBOL}/{TARGET_QUOTE_ASSET} (Avg Range: {ref_avg_daily_range:.2f}%)\n",
    ("Quote Asset:", "bold white"), f" {TARGET_QUOTE_ASSET}\n",
    ("Analysezeitraum:", "bold white"), f" {LOOKBACK_DAYS} Tage\n",
    ("Min. Korrelation:", "bold white"), f" {MIN_CORRELATION:.2f}\n",
    ("Max. Volatilitäts-Ratio-Abw.:", "bold white"), f" {MAX_VOLATILITY_RATIO_DIFFERENCE:.2f} (Ratio muss zw. {1-MAX_VOLATILITY_RATIO_DIFFERENCE:.2f} und {1+MAX_VOLATILITY_RATIO_DIFFERENCE:.2f} sein)\n",
    ("Min. Binance-Volumen (Heute):", "bold white"), f" ${MIN_BINANCE_USDC_VOLUME_USD:,.0f} (Ref-Coin ausgenommen)\n",
    ("Zu prüfende Kandidaten:", "bold white"), f" {total_to_process}\n",
    ("Threads:", "bold white"), f" {NUM_THREADS}",
)
console.print(Panel(summary_text, title="Analyse-Parameter", border_style="yellow", padding=(1, 2)))

try:
    confirmation = console.input("\n[bold]Analyse starten? ([green]yes[/]/[red]no[/]): [/]").lower().strip()
    if confirmation not in ['yes', 'y']:
        console.print("\n[yellow]Vorgang abgebrochen.[/]"); exit()
    else:
        console.print("[green]Bestätigt. Starte parallele Analyse...[/]")
except EOFError:
    console.print("\n[yellow]Keine Eingabe. Abbruch.[/]"); exit()

# --- Schritt 5: Parallele Verarbeitung ---
console.print(Rule("[bold yellow]Schritt 5: Parallele Verarbeitung[/]"))
threads = []
pbar = tqdm(total=total_to_process, desc=f"Vergleiche mit {REFERENCE_COIN_SYMBOL}", unit=" coin", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

for i in range(NUM_THREADS):
    thread = threading.Thread(target=process_coin, args=(coin_queue, binance_tradable_symbols, ref_returns, start_str, end_str, LOOKBACK_DAYS, pbar), daemon=True)
    thread.start()
    threads.append(thread)

coin_queue.join()
pbar.close()
# `processed_counter` zählt alle Durchläufe im Worker, `results_list` enthält die erfolgreichen.
# `processed_counter - len(results_list)` ist die Anzahl derer, die übersprungen wurden oder die Filter nicht bestanden.
console.print(f"\n[green]Vergleich abgeschlossen.[/] [cyan]{processed_counter}[/] Coins verarbeitet, [yellow]{processed_counter - len(results_list)}[/] davon passten nicht zu den Kriterien oder wurden übersprungen.")

# --- Schritt 6: Ergebnisse Anzeigen ---
console.print(Rule(f"[bold yellow]Schritt 6: Ergebnisse für {REFERENCE_COIN_SYMBOL}/{TARGET_QUOTE_ASSET}[/]"))

if results_list:
    df = pd.DataFrame(results_list)

    df.loc[df['Symbol'] == REFERENCE_COIN_SYMBOL, 'Correlation'] = 1.0
    df.loc[df['Symbol'] == REFERENCE_COIN_SYMBOL, 'Volatility_Ratio'] = 1.0
    if ref_avg_daily_range is not None:
        df.loc[df['Symbol'] == REFERENCE_COIN_SYMBOL, 'Avg_Daily_Range_Perc'] = ref_avg_daily_range

    df['Combined_Score'] = df['Correlation'] * (1 - abs(1 - df['Volatility_Ratio']))
    df = df.sort_values(by=['Combined_Score', 'Market_Cap_Raw'], ascending=[False, False])

    console.print(f"\n[bold green]{len(df)} Coins gefunden[/] mit Korrelation >= [magenta]{MIN_CORRELATION:.2f}[/] UND Volatilitäts-Ratio zw. [magenta]{1-MAX_VOLATILITY_RATIO_DIFFERENCE:.2f}[/] und [magenta]{1+MAX_VOLATILITY_RATIO_DIFFERENCE:.2f}[/]:")

    table = Table(title=f"Ähnliche Coins zu {REFERENCE_COIN_SYMBOL} (Korrelation & Volatilität)",
                  show_header=True, header_style="bold magenta", border_style="dim blue")

    table.add_column("Rank", style="dim", width=4, justify="right")
    table.add_column("Pair", style="cyan", no_wrap=True)
    table.add_column("Name", style="white", min_width=10)
    table.add_column(f"Market Cap ($)", style="green", justify="right")
    table.add_column(f"Vol ({TARGET_QUOTE_ASSET} Today) ($)", style="blue", justify="right")
    table.add_column("Korrelation", style="yellow", justify="center")
    table.add_column("Avg Range %", style="magenta", justify="right")
    table.add_column("Vola Ratio", style="magenta", justify="center")

    for index, row in enumerate(df.itertuples()):
        market_cap_str = f"{row.Market_Cap_Raw:,.0f}"
        volume_str = f"{row.Vol_Today_Raw:,.0f}"
        correlation_val = 1.0 if row.Symbol == REFERENCE_COIN_SYMBOL else row.Correlation
        correlation_str = f"{correlation_val:.3f}"
        avg_range_str = f"{row.Avg_Daily_Range_Perc:.2f}%"
        vol_ratio_str = f"{row.Volatility_Ratio:.2f}"
        row_style = "on grey19" if row.Symbol == REFERENCE_COIN_SYMBOL else ""

        table.add_row(
            str(index + 1), row.Pair, row.Name, market_cap_str, volume_str,
            correlation_str, avg_range_str, vol_ratio_str, style=row_style
        )
    console.print(table)
    console.print(f"\n[dim]Referenz-Coin ({REFERENCE_COIN_SYMBOL}) Avg Daily Range: {ref_avg_daily_range:.2f}%[/]")

    # --- Optional: Whitelist generieren ---
    try:
        console.print("\n")
        num_assets_str = console.input(f"Wie viele der Top [bold cyan]{len(df)}[/] Paare (inkl. Ref-Coin falls Top N) sollen in die Whitelist? ([cyan]Zahl[/] eingeben, [cyan]0[/] für keine): ")
        num_assets = int(num_assets_str)

        if 0 < num_assets <= len(df):
            selected_pairs = df['Pair'].head(num_assets).tolist()
            whitelist_panel_title = f"Freqtrade `pair_whitelist` für Top {num_assets} Paare"
            whitelist_panel_subtitle = f"(Ähnlichste zu {REFERENCE_COIN_SYMBOL} nach Korr. & Vola., inkl. Ref-Coin)"

            # --- Erzeuge den Whitelist-String ---
            whitelist_content = '        "pair_whitelist": [\n'
            for i, pair in enumerate(selected_pairs):
                 whitelist_content += f'            "{pair}"'
                 if i < len(selected_pairs) - 1: whitelist_content += ',\n'
                 else: whitelist_content += '\n'
            whitelist_content += '        ],'
            # --- Ende Whitelist-String Erzeugung ---

            # --- 1. Ausgabe mit Panel (visuell) ---
            console.print(Panel(Text(whitelist_content, style="green"),
                                title=whitelist_panel_title,
                                subtitle=whitelist_panel_subtitle,
                                border_style="green",
                                padding=(1, 4)))
            console.print("(Visuelle Anzeige oben)")

            # --- 2. Ausgabe als reiner Text (zum Kopieren) ---
            console.print("\n" + "-"*40)
            console.print(Text("↓↓↓ FÜR CONFIG KOPIEREN ↓↓↓", style="bold yellow"))
            console.print(whitelist_content) # <--- Hier wird der reine String ausgegeben
            console.print(Text("↑↑↑ FÜR CONFIG KOPIEREN ↑↑↑", style="bold yellow"))
            console.print("-"*40 + "\n")


        elif num_assets == 0: console.print("\n[yellow]Keine Whitelist generiert.[/]")
        else:
             console.print(f"[red]Ungültige Anzahl.[/] Es wurden {len(df)} Paare insgesamt gefunden.")
    except ValueError: console.print("[red]Ungültige Eingabe.[/] Es wurde keine Whitelist generiert.")
    except EOFError: console.print("\n[yellow]Keine Eingabe erhalten. Whitelist übersprungen.[/]")
    except Exception as e: console.print(f"[red]Ein Fehler ist aufgetreten:[/red] {e}")

else:
    console.print("\n[bold yellow]Keine Coins gefunden, die ALLEN Kriterien (Korrelation UND Volatilitäts-Ähnlichkeit) entsprechen.[/]")
    console.print(f"- Versuche, [cyan]MIN_CORRELATION[/] ({MIN_CORRELATION:.2f}) zu senken.")
    console.print(f"- Versuche, [cyan]MAX_VOLATILITY_RATIO_DIFFERENCE[/] ({MAX_VOLATILITY_RATIO_DIFFERENCE:.2f}) zu erhöhen.")
    console.print(f"- Überprüfe die anderen Filter (Market Cap, Volumen etc.).")
    console.print(f"- Prüfe den Referenz-Coin ({ref_symbol_binance}) und dessen Daten.")


# --- Schlussbemerkung ---
console.print(Rule("[bold blue]Wichtiger Hinweis[/]"))
console.print("[italic]Diese Analyse kombiniert Richtungs- (Korrelation) und Stärke-Ähnlichkeit (Volatilität).[/italic]")
console.print("[italic]Dennoch ist es keine Garantie für zukünftiges Verhalten. Backtesting ist unerlässlich![/italic]")
console.print("[bold orange1]Führe IMMER deine eigene Recherche (DYOR) durch und teste Strategien gründlich![/bold orange1]")


# --- END OF FILE findSimilarCoins_CorrAndVol_v1.5_inclRefWhitelist.py ---
