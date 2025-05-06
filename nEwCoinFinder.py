# Installation notwendig:
# pip install requests python-binance pycoingecko pandas tqdm numpy

import time
import pandas as pd
from binance.client import Client
from pycoingecko import CoinGeckoAPI
import requests
import threading
from queue import Queue
from datetime import datetime, timedelta, timezone
import numpy as np
from tqdm import tqdm

# --- Konfigurierbare Filter ---
MIN_MARKET_CAP_USD = 800_000_000 # Behalte diesen Wert moderat, um eine Auswahl zu haben
MIN_TOTAL_VOLUME_USD = 5_000_000
MIN_BINANCE_USDC_VOLUME_USD = 500_000 # Mindest-Volumen HEUTE für das Paar

# --- Filter für historische Konsistenz ---
LOOKBACK_DAYS = 60
MIN_AVG_DAILY_RANGE_PERCENT = 1.5 # Grundfilter: Durchschnittliche Range muss mind. so hoch sein

TARGET_QUOTE_ASSET = "USDC"
TOP_N_COINS_TO_CHECK = 250
NUM_THREADS = 15

# --- Initialisierung der APIs ---
cg = CoinGeckoAPI()
binance_client = Client()

# --- Thread-sichere Speicherung der Ergebnisse ---
results_list = []
results_lock = threading.Lock()
processed_counter = 0

# --- Hilfsfunktion für sichere API-Aufrufe ---
def safe_api_call(func, *args, **kwargs):
    max_retries = 3
    delay = 5
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            time.sleep(delay)
        except Exception as e:
            return None
    return None

# --- Worker-Funktion für Threads ---
def process_coin(coin_queue, binance_tradable_symbols, pbar):
    global processed_counter
    while not coin_queue.empty():
        try:
            coin = coin_queue.get(block=False)
        except Queue.Empty:
            break

        symbol_on_binance = f"{coin['symbol']}{TARGET_QUOTE_ASSET}"
        full_pair_string = f"{coin['symbol']}/{TARGET_QUOTE_ASSET}"

        try:
            if symbol_on_binance in binance_tradable_symbols:
                # 1. Heutiges Volumen prüfen
                ticker_data = safe_api_call(binance_client.get_ticker, symbol=symbol_on_binance)
                if not ticker_data: continue

                try: pair_volume_quote_today = float(ticker_data.get('quoteVolume', 0))
                except (ValueError, TypeError): pair_volume_quote_today = 0

                if pair_volume_quote_today < MIN_BINANCE_USDC_VOLUME_USD: continue

                # 2. Historische Daten holen
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=LOOKBACK_DAYS)
                start_str = str(int(start_date.timestamp() * 1000))
                end_str = str(int(end_date.timestamp() * 1000))

                klines = safe_api_call(binance_client.get_historical_klines,
                                       symbol=symbol_on_binance, interval=Client.KLINE_INTERVAL_1DAY,
                                       start_str=start_str, end_str=end_str)

                if not klines or len(klines) < LOOKBACK_DAYS * 0.8: continue

                # 3. Daten verarbeiten
                columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
                df = pd.DataFrame(klines, columns=columns)
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['High', 'Low', 'Close'])
                df = df[df['Close'] > 0]

                if df.empty: continue

                df['Daily Range %'] = ((df['High'] - df['Low']) / df['Close']) * 100
                df = df.dropna(subset=['Daily Range %'])

                if df.empty: continue

                # 4. Metriken berechnen
                avg_daily_range = df['Daily Range %'].mean()
                median_daily_range = df['Daily Range %'].median()
                p25_daily_range = df['Daily Range %'].quantile(0.25)

                # 5. Grundfilter anwenden und Ergebnis speichern (inkl. Market Cap)
                if avg_daily_range >= MIN_AVG_DAILY_RANGE_PERCENT:
                    with results_lock:
                        results_list.append({
                            'Pair': full_pair_string,
                            'Symbol': coin['symbol'],
                            'Name': coin['name'],
                            'Market Cap': coin.get('market_cap_usd', 0), # Market Cap hinzugefügt
                            f'Vol (Today)': pair_volume_quote_today,
                            f'Avg Range %': avg_daily_range,
                            f'Median Range %': median_daily_range,
                            f'P25 Range %': p25_daily_range
                        })
        except Exception as e:
            # Optional: print(f"\nFehler bei Verarbeitung von {full_pair_string}: {e}")
            pass
        finally:
            coin_queue.task_done()
            with results_lock:
                processed_counter += 1
                pbar.update(1)

# ======== Hauptprogrammablauf ========

print("="*30)
print("   Krypto Coin Screener v2.2  ")
print(" (Mit erweiterter Sortieroption)")
print("="*30)

# --- Schritt 1: CoinGecko Daten holen ---
print("\nSchritt 1: Abrufen der Top Coins von CoinGecko...")
try:
    ping_result = cg.ping()
    if not ping_result.get('gecko_says'): print("Fehler: CoinGecko API nicht erreichbar."); exit()
    print("CoinGecko API erreichbar.")
except Exception as e: print(f"Fehler beim Ping der CoinGecko API: {e}"); exit()
coins_market_data = safe_api_call(cg.get_coins_markets, vs_currency='usd', order='market_cap_desc', per_page=TOP_N_COINS_TO_CHECK, page=1, sparkline=False)
if not coins_market_data: print("Fehler: Konnte keine Marktdaten von CoinGecko abrufen."); exit()
print(f"{len(coins_market_data)} Coins von CoinGecko erhalten.")

# --- Schritt 2: Initiales Filtern (Market Cap wird mitgenommen) ---
print("\nSchritt 2: Filtern nach Marktkapitalisierung und Gesamtvolumen...")
initial_filtered_coins = []
for coin in coins_market_data:
    market_cap = coin.get('market_cap')
    total_volume = coin.get('total_volume')
    symbol = coin.get('symbol', '').upper()
    if market_cap and total_volume and symbol:
        if market_cap >= MIN_MARKET_CAP_USD and total_volume >= MIN_TOTAL_VOLUME_USD:
            initial_filtered_coins.append({'id': coin.get('id'), 'symbol': symbol, 'name': coin.get('name'), 'market_cap_usd': market_cap}) # Market Cap hier speichern
print(f"{len(initial_filtered_coins)} Coins nach initialen Filtern übrig.")
if not initial_filtered_coins: print("Keine Coins nach initialen Filtern übrig. Programmende."); exit()

# --- Schritt 3: Parallele Binance Prüfung ---
print(f"\nSchritt 3: Paralleles Prüfen auf Binance ({TARGET_QUOTE_ASSET}-Paar, Volumen & Historie)...")
print(f"(Letzte {LOOKBACK_DAYS} Tage: Mindestens {MIN_AVG_DAILY_RANGE_PERCENT:.2f}% avg. Range erforderlich)")
print("Rufe alle Binance-Symbole ab...")
all_binance_symbols_info = safe_api_call(binance_client.get_exchange_info)
binance_tradable_symbols = set()
if all_binance_symbols_info:
    binance_tradable_symbols = {s['symbol'] for s in all_binance_symbols_info.get('symbols', []) if s['status'] == 'TRADING'}
    print(f"{len(binance_tradable_symbols)} handelbare Symbole auf Binance gefunden.")
else: print("Fehler: Konnte keine Symbole von Binance abrufen. Breche ab."); exit()

# --- Start der parallelen Verarbeitung ---
coin_queue = Queue()
for coin in initial_filtered_coins: coin_queue.put(coin)
total_to_process = len(initial_filtered_coins)
threads = []
pbar = tqdm(total=total_to_process, desc="Prüfe Coins auf Binance", unit="coin", ncols=100)
print("Starte Worker-Threads...")
for i in range(NUM_THREADS):
    thread = threading.Thread(target=process_coin, args=(coin_queue, binance_tradable_symbols, pbar), daemon=True)
    thread.start()
    threads.append(thread)
coin_queue.join()
pbar.close()
print("\nAlle Threads haben ihre Arbeit beendet.")

# --- Schritt 4: Ergebnisse Anzeigen und Whitelist generieren ---
print("\n\n" + "="*50)
print("  Zusammenfassung der gefundenen Kandidaten ")
print("="*50)

if results_list:
    df = pd.DataFrame(results_list)
    # Temporäre Spalten für Sortierung erstellen (reine Zahlen)
    df['sort_market_cap'] = df['Market Cap']
    df['sort_vol_today'] = df['Vol (Today)']
    df['sort_avg_range'] = df['Avg Range %']
    df['sort_median_range'] = df['Median Range %']
    df['sort_p25_range'] = df['P25 Range %']

    # Formatierung für die Anzeige
    pd.options.display.float_format = '{:,.0f}'.format # Standard für große Zahlen
    df['Market Cap'] = df['Market Cap'].map('{:,.0f}'.format)
    df['Vol (Today)'] = df['Vol (Today)'].map('{:,.0f}'.format)
    for col in ['Avg Range %', 'Median Range %', 'P25 Range %']:
        if col in df.columns: df[col] = df[col].map('{:,.2f}%'.format)

    # Standardmäßig nach Market Cap sortieren für die erste Anzeige
    df = df.sort_values(by='sort_market_cap', ascending=False)

    print(f"\n{len(df)} Coins entsprechen den Kriterien (Avg Range >= {MIN_AVG_DAILY_RANGE_PERCENT}%):")
    pd.set_option('display.max_rows', None); pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1200); pd.set_option('display.colheader_justify', 'center')

    # Angezeigte Spalten (ohne die temporären Sortierspalten)
    display_columns = ['Pair', 'Name', 'Market Cap', 'Vol (Today)', 'Avg Range %', 'Median Range %', 'P25 Range %']
    print(df[display_columns].to_string(index=False))

    # --- Abfrage für Whitelist Größe und Sortierung ---
    selected_pairs = []
    num_assets = -1
    sort_column_name = default_sort_column = 'Market Cap' # Trackt den Namen der Sortierspalte

    while True:
        try:
            print("\n" + "-"*40); print(" Sortierung für die Top-Paar Auswahl:"); print("-"*40)
            print(" Wähle, nach welcher Metrik die Liste für die Whitelist sortiert werden soll:\n")
            print("  << Fokus auf Etablierung / Liquidität >>")
            print("  [1] Nach 'Market Cap':          (Größte Coins oben)")
            print("  [2] Nach 'Vol (Today)':         (Meiste heutige Aktivität oben)\n")
            print("  << Fokus auf Volatilitäts-Konsistenz >>")
            print("  [3] Nach 'Median Range %':      (Höchste 'typische' Tagesbewegung oben)")
            print("  [4] Nach 'P25 Range %':         (Höchste 'garantierte Mindestbewegung' oben)\n")
            print("  << Fokus auf Max. Durchschnitts-Volatilität >>")
            print("  [5] Nach 'Avg Range %':         (Höchste Durchschnittsbewegung oben)\n")

            sort_choice = input("Deine Wahl für die Sortierung (1-5): ")
            sort_column = None # Temporäre Sortierspalte
            sort_column_name = "Unknown" # Angezeigter Name

            if sort_choice == '1': sort_column = 'sort_market_cap'; sort_column_name = 'Market Cap'
            elif sort_choice == '2': sort_column = 'sort_vol_today'; sort_column_name = 'Vol (Today)'
            elif sort_choice == '3': sort_column = 'sort_median_range'; sort_column_name = 'Median Range %'
            elif sort_choice == '4': sort_column = 'sort_p25_range'; sort_column_name = 'P25 Range %'
            elif sort_choice == '5': sort_column = 'sort_avg_range'; sort_column_name = 'Avg Range %'

            if sort_column and sort_column in df.columns:
                df = df.sort_values(by=sort_column, ascending=False) # Sortiere nach der temporären Zahlenspalte
                print(f"\n---> Liste neu sortiert nach: {sort_column_name}")
                print("\nTop 15 der neu sortierten Liste:")
                print(df[display_columns].head(15).to_string(index=False)) # Zeige formatierte Spalten
            elif not sort_column: raise ValueError("Ungültige Wahl (nur 1-5 erlaubt)")
            else: raise ValueError(f"Fehler: Spalte '{sort_column}' nicht in Daten vorhanden.")

            num_assets_str = input(f"\nWie viele dieser {len(df)} Paare sollen in die Whitelist aufgenommen werden? (Zahl eingeben, 0 für keine): ")
            num_assets = int(num_assets_str)
            if 0 <= num_assets <= len(df): break
            else: print(f"FEHLER: Bitte eine Zahl zwischen 0 und {len(df)} eingeben.")
        except ValueError as e: print(f"UNGÜLTIGE EINGABE: {e}. Bitte erneut versuchen.")
        except Exception as e: print(f"Ein unerwarteter Fehler ist aufgetreten: {e}. Bitte erneut versuchen.")

    # --- Generiere die Whitelist Ausgabe ---
    if num_assets > 0:
        selected_pairs = df['Pair'].head(num_assets).tolist()
        print("\n\n" + "="*40)
        print(f" Freqtrade `pair_whitelist` für Top {num_assets} Paare ")
        print(f" (Sortiert nach: {sort_column_name})") # Nutze den gespeicherten Namen
        print("="*40)
        print("(Einfach kopieren und in deine Config einfügen)\n")
        print('        "pair_whitelist": [')
        for i, pair in enumerate(selected_pairs):
            print(f'            "{pair}"', end='')
            if i < len(selected_pairs) - 1: print(',')
            else: print('')
        print('        ],')
    elif num_assets == 0: print("\nKeine Whitelist generiert, da 0 ausgewählt wurde.")

else:
    print("\nKeine Coins gefunden, die ALLEN Kriterien entsprechen.")
    print(f"- Prüfe, ob MIN_AVG_DAILY_RANGE_PERCENT ({MIN_AVG_DAILY_RANGE_PERCENT}%) nicht zu hoch ist.")
    print("- Andere Filter könnten zu streng sein oder Marktbedingungen sind ungünstig.")

print("\n--- WICHTIGER HINWEIS ---")
print("Die Wahl der Sortierung beeinflusst stark, welche Coins oben stehen.")
print("Market Cap/Volumen = Etablierung; Median/P25 Range = Konsistenz; Avg Range = Max. Bewegung.")
print("Führe IMMER deine eigene Recherche (DYOR) durch!")
