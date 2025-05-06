# --- START OF FILE simple_backtester_top_coins_v2_realistic.py ---
# Version: 2.0
# Änderungen:
# - Handelsgebühren hinzugefügt (TRADING_FEE_PERCENT)
# - Optionale realistischere Order-Ausführung (EXECUTE_ON_NEXT_OPEN = True)
#   - Trades (außer SL) werden zum Open der nächsten Kerze ausgeführt
#   - Stop-Loss wird weiterhin intra-candle geprüft (low <= stop_loss_price)
# - Optionale einfache Slippage hinzugefügt (SLIPPAGE_PERCENT_PER_SIDE)

import importlib.util
import inspect
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import math
import threading
from queue import Queue, Empty
import json
import sys
import os

import numpy as np
import pandas as pd
import requests
import ccxt
from pycoingecko import CoinGeckoAPI
import pyarrow # Import prüfen

# --- Rich für schöne Ausgabe ---
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

# --- Freqtrade Parameter Klassen (für Auto-Detection) ---
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter, BooleanParameter
# Füge hier ggf. weitere Parametertypen hinzu, falls deine Strategien sie nutzen (z.B. RealParameter)


# ==============================================================================
# --- KONFIGURATION ---
# ==============================================================================
STRATEGY_FILE_PATH = Path("/freqtrade/user_data/strategies/test.py") # <-- ANPASSEN!!
USE_HYPEROPT_JSON_IF_FOUND = True # <-- Setze auf True, um gefundene JSON zu nutzen
QUOTE_ASSET = "USDC"
TIMEFRAME = '5m'
LOOKBACK_DAYS = 60
MIN_MARKET_CAP_USD = 500_000_000
MIN_AVG_QUOTE_VOLUME_LOOKBACK = 1_000_000 # Durchschnittliches TÄGLICHES Quote-Volumen über LOOKBACK_DAYS
EXCHANGE_ID = 'binance'
API_TIMEOUT = 30000 # ms
NUM_THREADS = 10
PRINT_THREAD_PROGRESS = True
CACHE_DIR = Path("./ohlcv_cache")
CACHE_MAX_AGE_DAYS = 1
# --- NEU: Realismus-Einstellungen ---
TRADING_FEE_PERCENT = 0.1 # Gebühr pro Trade in Prozent (z.B. 0.1 für 0.1%)
EXECUTE_ON_NEXT_OPEN = True # True: Realistischer (Signal -> Aktion nächste Kerze), False: Einfacher (Aktion auf Signalkerze)
SLIPPAGE_PERCENT_PER_SIDE = 0.00 # Zusätzliche % Slippage pro Kauf/Verkauf (z.B. 0.02 für 0.02%). 0 deaktiviert.
# --- Ende Realismus-Einstellungen ---

# --- NEU: Sortierung der Ergebnisse ---
# Mögliche Werte: "Total PnL %", "Profit Factor", "Sharpe Ratio", "Max Drawdown %", "Trades"
# Bei "Max Drawdown %" wird aufsteigend sortiert (geringster Drawdown zuerst), sonst absteigend.
SORT_RESULTS_BY = "Sharpe Ratio"
# ==============================================================================

# --- Globale Variablen & Initialisierung ---
# (Bleiben größtenteils gleich)
console = Console(); cg = CoinGeckoAPI(); markets = {}; results_list = []
results_lock = threading.Lock(); processed_count = 0; total_coins_to_process = 0
skipped_mcap = 0; skipped_insufficient_history = 0; final_candidate_list = []
processed_data_phase = 0; fetched_data_cache = {}; fetch_lock = threading.Lock()
processed_backtest_phase = 0; skipped_avg_volume_worker = 0; analyzed_count = 0
tf_in_seconds = 0
try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True); console.print(f"Cache-Verzeichnis: [cyan]{CACHE_DIR.resolve()}[/]")
    exchange_options = { 'options': { 'defaultType': 'spot', 'adjustForTimeDifference': True, }, 'timeout': API_TIMEOUT, 'enableRateLimit': True, }
    exchange = getattr(ccxt, EXCHANGE_ID)(exchange_options); console.print(f"[green]{exchange.name} CCXT Client initialisiert.[/]")
    markets = exchange.load_markets(); console.print(f"[green]{len(markets)} Märkte von {exchange.name} geladen.[/]")
    tf_in_seconds = exchange.parse_timeframe(TIMEFRAME); console.print(f"[dim]Timeframe '{TIMEFRAME}' entspricht {tf_in_seconds} Sekunden.[/dim]")
except Exception as e: console.print(f"[bold red]Fehler bei Initialisierung/Cache-Erstellung:[/bold red] {e}"); exit(1)

# --- Hilfsfunktionen ---
# (safe_api_call, load_strategy_from_file, get_cache_filename, get_ohlcv_data bleiben gleich)
def safe_api_call(func, *args, **kwargs):
    max_retries = 5; delay = 7
    for attempt in range(max_retries):
        try: return func(*args, **kwargs)
        except (ccxt.RateLimitExceeded) as e: time.sleep(delay); delay *= 1.5
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException, requests.exceptions.ReadTimeout) as e: time.sleep(delay); delay *= 1.5
        except ccxt.ExchangeError as e: time.sleep(delay); delay *= 1.5;
        except Exception as e: time.sleep(delay); delay *= 1.5
    return None

def load_strategy_from_file(filepath: Path):
    if not filepath.is_file(): raise FileNotFoundError(f"Strategie-Datei nicht gefunden: {filepath}")
    module_name = filepath.stem; spec = importlib.util.spec_from_file_location(module_name, str(filepath))
    if spec is None or spec.loader is None: raise ImportError(f"Konnte Spezifikation für Modul nicht laden: {filepath}")
    strategy_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(strategy_module)
    for name, obj in inspect.getmembers(strategy_module):
        if inspect.isclass(obj) and obj.__module__ == module_name:
            if IStrategy in obj.__mro__:
                 config = { 'strategy': name, 'minimal_roi': {"0": 10.0}, 'stoploss': -0.10, 'timeframe': TIMEFRAME, 'dry_run': True, 'exchange': {'name': EXCHANGE_ID, 'pair_whitelist': []}, 'pair_whitelist': [], } # Verwende globalen TIMEFRAME
                 try:
                     strategy_instance = obj(config=config)
                     strategy_instance.timeframe = getattr(strategy_instance, 'timeframe', config.get('timeframe', TIMEFRAME)) # Sicherstellen
                     strategy_instance.stoploss = getattr(strategy_instance, 'stoploss', config.get('stoploss', -0.10))
                     strategy_instance.minimal_roi = getattr(strategy_instance, 'minimal_roi', config.get('minimal_roi', {"0": 10.0}))
                     strategy_instance.startup_candle_count = getattr(strategy_instance, 'startup_candle_count', 30)
                     strategy_instance.use_sell_signal = getattr(strategy_instance, 'use_sell_signal', True)
                     strategy_instance.sell_profit_only = getattr(strategy_instance, 'sell_profit_only', False)
                     strategy_instance.ignore_roi_if_buy_signal = getattr(strategy_instance, 'ignore_roi_if_buy_signal', False)
                     return strategy_instance
                 except Exception as e: raise ImportError(f"Konnte Strategie '{name}' nicht korrekt instanziieren.") from e
    raise ImportError(f"Keine Klasse, die von IStrategy erbt, in {filepath} gefunden.")

def get_cache_filename(symbol: str, timeframe: str) -> Path:
    pair_slug = symbol.replace('/', '_'); return CACHE_DIR / f"{EXCHANGE_ID}_{pair_slug}_{timeframe}.parquet"

def get_ohlcv_data(symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime, startup_candles: int) -> pd.DataFrame | None:
    cache_file = get_cache_filename(symbol, timeframe); now = datetime.now(timezone.utc); load_from_cache = False; df_cached = None
    tf_in_ms = tf_in_seconds * 1000 # Verwende globale tf_in_seconds
    fetch_start_dt = start_dt - timedelta(milliseconds=startup_candles * tf_in_ms * 1.2) # Slightly more buffer
    fetch_start_timestamp = int(fetch_start_dt.timestamp() * 1000)
    fetch_end_timestamp = int(end_dt.timestamp() * 1000)
    if cache_file.is_file():
        cache_valid = False
        if CACHE_MAX_AGE_DAYS > 0:
            try:
                file_mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime, timezone.utc)
                if (now - file_mod_time) < timedelta(days=CACHE_MAX_AGE_DAYS): cache_valid = True
            except Exception: pass # Ignore errors reading file time
        else: cache_valid = True # Cache is always valid if age is 0 or less
        if cache_valid:
            try:
                df_cached = pd.read_parquet(cache_file)
                if not df_cached.empty:
                    cache_start_dt = df_cached.index.min(); cache_end_dt = df_cached.index.max()
                    # Prüfe, ob der Cache *mindestens* den benötigten Zeitraum abdeckt
                    if cache_start_dt <= fetch_start_dt and cache_end_dt >= end_dt:
                        load_from_cache = True
            except Exception as e:
                # console.print(f"[yellow]Warnung:[/yellow] Fehler beim Lesen des Cache {cache_file}: {e}")
                df_cached = None; load_from_cache = False

    if load_from_cache and df_cached is not None:
        # Extrahiere nur den wirklich benötigten Teil aus dem Cache
        df_needed = df_cached[(df_cached.index >= fetch_start_dt) & (df_cached.index <= end_dt)].copy()
        if not df_needed.empty and len(df_needed) >= startup_candles: # Sicherstellen, dass genug Kerzen da sind
            # console.print(f"[dim]Cache HIT für {symbol}[/dim]", end="\r")
            return df_needed
        else:
             # console.print(f"[dim]Cache teilw. {symbol}, lade neu...[/dim]", end="\r")
             load_from_cache = False # Cache nicht ausreichend, neu laden

    # console.print(f"[dim]API Abruf für {symbol}...[/dim]", end="\r")
    limit = 1000; all_ohlcv = []; current_timestamp = fetch_start_timestamp
    # Erhöhe max_loops leicht, falls der API-Abruf genau an der Grenze stockt
    max_loops = math.ceil((fetch_end_timestamp - fetch_start_timestamp) / tf_in_ms / limit) + 5
    loops = 0

    while current_timestamp <= fetch_end_timestamp and loops < max_loops:
        try:
            # console.print(f"[dim] Fetch {symbol} from {pd.to_datetime(current_timestamp, unit='ms', utc=True)}[/dim]", end='\r')
            ohlcv_batch = safe_api_call(exchange.fetch_ohlcv, symbol, timeframe, since=current_timestamp, limit=limit)
            if ohlcv_batch:
                # Verhindere Endlosschleife, wenn API immer dieselbe letzte Kerze liefert
                if all_ohlcv and ohlcv_batch[0][0] <= all_ohlcv[-1][0]:
                    break
                all_ohlcv.extend(ohlcv_batch)
                last_fetched_ts = ohlcv_batch[-1][0]
                current_timestamp = last_fetched_ts + tf_in_ms # Zum nächsten Zeitstempel springen
            else:
                break # Keine weiteren Daten von der API
            time.sleep(exchange.rateLimit / 1000 * 0.7) # Rate Limit respektieren
        except Exception as e:
            # console.print(f"\n[red]API Fehler beim OHLCV-Abruf für {symbol}: {e}[/red]")
            if loops > 5 : # Nicht ewig versuchen
                 console.print(f"\n[bold red]Mehrere API Fehler für {symbol}. Gebe auf.[/bold red]")
                 return None
            time.sleep(7) # Länger warten bei wiederholten Fehlern
        loops += 1
        if loops >= max_loops:
            console.print(f"\n[yellow]Warnung:[/yellow] Max Loops ({max_loops}) erreicht beim API-Download für {symbol}. Daten könnten unvollständig sein.")

    if not all_ohlcv:
        # console.print(f"\n[yellow]Warnung:[/yellow] Keine OHLCV-Daten für {symbol} im Zeitraum erhalten.")
        return None

    df_downloaded = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_downloaded['date'] = pd.to_datetime(df_downloaded['timestamp'], unit='ms', utc=True)
    df_downloaded = df_downloaded.drop_duplicates(subset='timestamp', keep='first')
    df_downloaded = df_downloaded.set_index('date')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_downloaded[col] = pd.to_numeric(df_downloaded[col], errors='coerce')
    df_downloaded.dropna(subset=['open', 'high', 'low', 'close'], inplace=True) # Volumen NaN ist okay

    if not df_downloaded.empty:
        try:
            # Kombiniere alten Cache mit neuen Daten, falls vorhanden
            df_to_save = df_downloaded
            if df_cached is not None and not df_cached.empty:
                 # Bevorzuge neu heruntergeladene Daten für überlappende Indizes
                 df_combined = pd.concat([df_cached[~df_cached.index.isin(df_downloaded.index)], df_downloaded])
                 # df_combined = df_combined.loc[~df_combined.index.duplicated(keep='last')] # Alternative
                 df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                 df_combined.sort_index(inplace=True)
                 df_to_save = df_combined
            df_to_save.to_parquet(cache_file)
            # console.print(f"[dim]Cache gespeichert: {cache_file.name}[/dim]", end='\r')
        except Exception as e:
            console.print(f"\n[red]Fehler[/red] beim Cache Speichern für {symbol} ({cache_file}): {e}")

    # Gebe den benötigten Teil zurück, auch wenn Cache vorher unvollständig war
    df_needed = df_downloaded[(df_downloaded.index >= fetch_start_dt) & (df_downloaded.index <= end_dt)].copy()
    if df_needed.empty or len(df_needed) < startup_candles // 2: # Brauchen mind. ~halbe Startup-Kerzen
        # console.print(f"\n[yellow]Warnung:[/yellow] Nicht genug Daten für {symbol} nach Download/Filterung ({len(df_needed)} Kerzen).")
        return None
    return df_needed

# --- *** ÜBERARBEITETE BACKTEST FUNKTION *** ---
def run_simple_backtest(
    dataframe: pd.DataFrame,
    strategy_instance,
    fee_percent: float = 0.0,
    execute_on_next_open: bool = True,
    slippage_percent: float = 0.0
) -> list:
    """
    Führt einen einfachen Backtest durch, optional mit Gebühren,
    Ausführung auf der nächsten Kerze und Slippage.
    """
    trades = []
    in_trade = False
    # Trade-relevante Variablen zurücksetzen pro Backtest
    entry_price_actual = 0.0      # Tatsächlicher Einstiegspreis (kann next_open sein)
    entry_price_after_fee_slippage = 0.0 # Preis nach Gebühren/Slippage für PnL
    entry_time_actual = None      # Tatsächlicher Einstiegszeitpunkt
    signal_entry_price = 0.0      # Preis zum Zeitpunkt des Signals (für SL-Berechnung)
    signal_entry_time = None      # Zeitpunkt des Signals
    stop_loss_price = 0.0         # Stop-Loss Trigger-Preis

    # --- Konfiguration auslesen / berechnen ---
    fixed_stop_loss_pct = getattr(strategy_instance, 'stoploss', -0.10)
    use_sell_signal = getattr(strategy_instance, 'use_sell_signal', True)
    sell_profit_only = getattr(strategy_instance, 'sell_profit_only', False)
    fee_factor = fee_percent / 100.0
    slippage_factor = slippage_percent / 100.0

    # Standardmäßig 'buy'/'sell' Spalten hinzufügen, falls nicht vorhanden
    if 'buy' not in dataframe.columns: dataframe['buy'] = 0
    if 'sell' not in dataframe.columns: dataframe['sell'] = 0

    # --- Vorbereitung für Next-Open Execution ---
    if execute_on_next_open:
        dataframe['next_open'] = dataframe['open'].shift(-1)
        # Iteriere nur bis zur vorletzten Zeile, da wir 'next_open' brauchen
        iterator = dataframe.iloc[:-1].iterrows()
    else:
        # Iteriere über alle Zeilen für die alte Methode
        iterator = dataframe.iterrows()

    # --- Haupt-Schleife über die Kerzen ---
    for index, row in iterator:
        current_time = index        # Zeit der aktuellen Kerze (für Signalfindung / SL-Check)
        current_close = row['close']# Schlusskurs der aktuellen Kerze
        current_low = row['low']    # Tiefstkurs der aktuellen Kerze (für SL)

        # --- Exit Logic ---
        if in_trade:
            exit_reason = None
            exit_price_raw = 0.0      # Ausstiegspreis vor Gebühren/Slippage
            exit_time_actual = None   # Tatsächlicher Ausstiegszeitpunkt

            # 1. STOP LOSS PRÜFUNG (höchste Priorität, intra-candle)
            #    Wird *immer* auf der aktuellen Kerze geprüft, unabhängig von execute_on_next_open!
            if current_low <= stop_loss_price:
                exit_reason = "Stop Loss"
                # Ausführung *sofort* zum Stop-Loss Preis (schlechtester Fall)
                exit_price_raw = stop_loss_price
                exit_time_actual = current_time # SL trifft in dieser Kerze
                # console.print(f"DEBUG {index}: SL Hit at {stop_loss_price:.4f} (Low: {current_low:.4f})")

            # 2. SELL SIGNAL PRÜFUNG (wenn kein SL ausgelöst wurde)
            elif row['sell'] == 1 and use_sell_signal:
                # Prüfe sell_profit_only Bedingung (basierend auf aktuellem Close vs. Einstieg)
                # Beachte: signal_entry_price ist der Referenzpreis vom Einstiegssignal
                if sell_profit_only and current_close <= signal_entry_price:
                    pass # Kein Verkauf bei Verlust, wenn sell_profit_only=True
                else:
                    exit_reason = "Sell Signal"
                    if execute_on_next_open:
                        # Ausführung zum Open der *nächsten* Kerze
                        next_open_price = row.get('next_open') # Sicherer Zugriff
                        if pd.isna(next_open_price): continue # Sollte nicht passieren durch iloc[:-1], aber sicher ist sicher
                        exit_price_raw = next_open_price
                        exit_time_actual = dataframe.index[dataframe.index.get_loc(index) + 1] # Zeit der nächsten Kerze
                    else:
                        # Ausführung zum Close der *aktuellen* Kerze (alte Methode)
                        exit_price_raw = current_close
                        exit_time_actual = current_time

            # 3. TRADE AUSFÜHREN & BUCHEN (wenn Ausstiegsgrund vorhanden)
            if exit_reason:
                # Berechne Ausstiegspreis nach Gebühren und Slippage
                # Slippage wirkt sich negativ beim Verkauf aus (man bekommt weniger)
                exit_price_after_fee_slippage = exit_price_raw * (1 - fee_factor - slippage_factor)

                # Berechne PnL basierend auf den tatsächlichen Preisen nach Gebühren/Slippage
                if entry_price_after_fee_slippage > 0: # Vermeide Division durch Null
                     profit_pct = (exit_price_after_fee_slippage / entry_price_after_fee_slippage) - 1
                else: profit_pct = 0.0 # Sollte nicht vorkommen, aber zur Sicherheit

                trades.append({
                    'entry_time': entry_time_actual,
                    'exit_time': exit_time_actual,
                    'entry_price': entry_price_actual,   # Ursprünglicher Ausführungspreis (ohne Fee/Slip)
                    'exit_price': exit_price_raw,        # Ursprünglicher Ausführungspreis (ohne Fee/Slip)
                    'profit_pct': profit_pct,            # PnL nach Gebühren & Slippage
                    'exit_reason': exit_reason
                })
                in_trade = False # Reset für nächsten Trade
                # Alle trade-spezifischen Variablen zurücksetzen
                entry_price_actual = 0.0; entry_price_after_fee_slippage = 0.0; entry_time_actual = None
                signal_entry_price = 0.0; signal_entry_time = None; stop_loss_price = 0.0

        # --- Entry Logic ---
        if not in_trade:
            # Prüfe BUY Signal auf aktueller Kerze
            if row['buy'] == 1:
                signal_entry_price = current_close # Preis bei Signal (für Referenz / sell_profit_only)
                signal_entry_time = current_time

                if execute_on_next_open:
                    # Einstieg zum Open der *nächsten* Kerze
                    next_open_price = row.get('next_open')
                    if pd.isna(next_open_price): continue # Nächste Kerze nicht verfügbar
                    entry_price_actual = next_open_price
                    entry_time_actual = dataframe.index[dataframe.index.get_loc(index) + 1]
                else:
                    # Einstieg zum Close der *aktuellen* Kerze (alte Methode)
                    entry_price_actual = current_close
                    entry_time_actual = current_time

                # Berechne Einstiegspreis inkl. Gebühren und Slippage
                # Slippage wirkt sich negativ beim Kauf aus (man zahlt mehr)
                entry_price_after_fee_slippage = entry_price_actual * (1 + fee_factor + slippage_factor)

                # Setze Stop-Loss basierend auf dem *tatsächlichen* Einstiegspreis (ohne Gebühren/Slippage)
                if fixed_stop_loss_pct < 0: # Nur wenn Stop-Loss aktiv ist
                    stop_loss_price = entry_price_actual * (1 + fixed_stop_loss_pct)
                else:
                    stop_loss_price = 0.0 # Deaktivierter Stop-Loss

                in_trade = True # Trade ist jetzt aktiv
                # console.print(f"DEBUG {entry_time_actual}: Entered at {entry_price_actual:.4f} (SL: {stop_loss_price:.4f})")


    # Optional: Entferne Hilfsspalte am Ende
    if 'next_open' in dataframe.columns and execute_on_next_open:
        del dataframe['next_open']

    return trades
# --- *** ENDE ÜBERARBEITETE BACKTEST FUNKTION *** ---


# --- ERWEITERTE calculate_metrics Funktion ---
# (Bleibt gleich, da sie auf `profit_pct` basiert, was jetzt Gebühren/Slippage enthält)
def calculate_metrics(trades: list, start_date: datetime, end_date: datetime) -> dict:
    total_trades = len(trades)
    base_metrics = {
        "Start Date": start_date.strftime('%Y-%m-%d'),
        "End Date": end_date.strftime('%Y-%m-%d'),
        "Total Trades": 0,
        "Win Rate": "0.00%",
        "Total PnL %": "0.00%",
        "Avg Win %": "0.00%",
        "Avg Loss %": "0.00%",
        "Profit Factor": "0.00",
        "Max Drawdown %": "0.00%",
        "Sharpe Ratio": "0.00"
    }
    if total_trades == 0: return base_metrics

    base_metrics["Total Trades"] = total_trades
    returns = [t['profit_pct'] for t in trades]
    wins = [t for t in trades if t['profit_pct'] > 0]
    losses = [t for t in trades if t['profit_pct'] <= 0]

    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0.0
    # Einfache Summe der Prozent-Profite (kann irreführend sein, besser wäre geometrisch, aber für Vergleich OK)
    total_pnl_pct = sum(returns) * 100

    avg_win_pct = (sum(t['profit_pct'] for t in wins) / len(wins)) * 100 if wins else 0.0
    avg_loss_pct = (sum(t['profit_pct'] for t in losses) / len(losses)) * 100 if losses else 0.0

    total_profit = sum(t['profit_pct'] for t in wins)
    total_loss = abs(sum(t['profit_pct'] for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

    # Max Drawdown (vereinfachte Equity-Kurve basierend auf %-Returns)
    max_drawdown = 0.0
    peak_equity = 1.0
    current_equity = 1.0
    for ret in returns:
        current_equity *= (1 + ret)
        if current_equity > peak_equity:
            peak_equity = current_equity
        drawdown = (peak_equity - current_equity) / peak_equity # Korrigierte Berechnung
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    # Korrektur: max_drawdown ist jetzt positiv, muss für Anzeige negativ gemacht werden
    max_drawdown_display = -max_drawdown * 100


    # Sharpe Ratio (vereinfacht, risk-free rate = 0, nicht annualisiert)
    sharpe_ratio = 0.0
    if total_trades >= 2:
        returns_np = np.array(returns)
        std_dev = np.std(returns_np)
        if std_dev > 0:
             mean_return = np.mean(returns_np)
             sharpe_ratio = mean_return / std_dev
             # Optional: Annualisierung (hier weggelassen, da komplexer und oft nicht nötig für Vorfilter)
             # trading_periods = len(dataframe) # Anzahl Kerzen im Backtest
             # trades_per_period = total_trades / trading_periods if trading_periods > 0 else 0
             # periods_per_year = (365 * 24 * 60 * 60) / tf_in_seconds if tf_in_seconds > 0 else 0
             # if periods_per_year > 0:
             #    sharpe_ratio *= np.sqrt(periods_per_year) # Annualisierung
        # else: sharpe_ratio bleibt 0

    base_metrics.update({
        "Win Rate": f"{win_rate:.2f}%",
        "Total PnL %": f"{total_pnl_pct:.2f}%",
        "Avg Win %": f"{avg_win_pct:.2f}%",
        "Avg Loss %": f"{avg_loss_pct:.2f}%",
        "Profit Factor": f"{profit_factor:.2f}" if profit_factor != np.inf else "Inf",
        "Max Drawdown %": f"{max_drawdown_display:.2f}%", # Negativer Wert für Drawdown
        "Sharpe Ratio": f"{sharpe_ratio:.2f}"
    })
    return base_metrics

# --- Worker für Phase 1: Prüfen der benötigten History ---
# (Bleibt gleich)
def history_check_worker(check_queue: Queue, required_start_dt: datetime, timeframe: str):
    global skipped_insufficient_history, final_candidate_list, processed_data_phase, total_coins_to_process
    while True:
        try:
            symbol = check_queue.get(block=False)
            try:
                if PRINT_THREAD_PROGRESS:
                     with results_lock: processed_data_phase += 1; current_progress = processed_data_phase
                     if current_progress % 20 == 0 or current_progress == total_coins_to_process:
                         console.print(f"Prüfe History: {current_progress}/{total_coins_to_process} ({symbol})...", end="\r")

                # Hole nur die erste Kerze, um das Startdatum zu prüfen
                ohlcv_age_check = safe_api_call(exchange.fetch_ohlcv, symbol, timeframe, since=0, limit=2)

                if ohlcv_age_check and len(ohlcv_age_check) > 0:
                    first_candle_ts = ohlcv_age_check[0][0]
                    first_candle_dt = pd.to_datetime(first_candle_ts, unit='ms', utc=True)
                    if first_candle_dt <= required_start_dt:
                        with results_lock: final_candidate_list.append(symbol)
                    else:
                        with results_lock: skipped_insufficient_history += 1
                else:
                    with results_lock: skipped_insufficient_history += 1
            except Exception as e:
                # console.print(f"\n[yellow]Warnung:[/yellow] Fehler bei History-Check für {symbol}: {e}")
                with results_lock: skipped_insufficient_history += 1
            finally:
                check_queue.task_done()
        except Empty: break
        except Exception as e:
             with results_lock: console.print(f"\n[bold red]Unerwarteter Fehler im History-Check Worker:[/bold red] {e}"); break


# --- *** ANGEPASSTER BACKTEST WORKER *** ---
def backtest_worker(
    backtest_queue: Queue,
    strategy_instance_template,
    start_dt: datetime,
    end_dt: datetime,
    timeframe: str,
    avg_vol_threshold: float,
    # NEUE Parameter für Realismus:
    fee_percent: float,
    execute_next_open: bool,
    slippage_percent: float
):
    """ Holt Daten, prüft Durchschnittsvolumen und führt Backtest durch (mit Realismus-Optionen). """
    global processed_backtest_phase, skipped_avg_volume_worker, analyzed_count, tf_in_seconds

    # Kopiere die Strategie-Instanz für diesen Thread, um Konflikte zu vermeiden
    strategy_instance = strategy_instance_template.__class__(config=strategy_instance_template.config)
    # Kopiere optimierte Parameter, falls vorhanden
    parameter_classes = (IntParameter, DecimalParameter, CategoricalParameter, BooleanParameter)
    for name, member in inspect.getmembers(strategy_instance_template):
         if isinstance(member, parameter_classes):
             try:
                 if hasattr(strategy_instance, name):
                    param_obj = getattr(strategy_instance, name)
                    if isinstance(param_obj, parameter_classes):
                        param_obj.value = member.value # Kopiere den Wert
             except Exception as e:
                 with results_lock: console.print(f"\n[yellow]Warnung:[/yellow] Fehler beim Kopieren von Parameter '{name}': {e}")
    # Auch Basis-Attribute kopieren
    strategy_instance.minimal_roi = strategy_instance_template.minimal_roi
    strategy_instance.stoploss = strategy_instance_template.stoploss
    # ... (weitere relevante Attribute ggf. hier kopieren)


    # Konstante für Berechnung des Tagesdurchschnitts
    SECONDS_PER_DAY = 86400
    candles_per_day = SECONDS_PER_DAY / tf_in_seconds if tf_in_seconds > 0 else 0

    while True:
        task_obtained = False; ccxt_symbol = None
        try:
            ccxt_symbol = backtest_queue.get(block=False); task_obtained = True

            if PRINT_THREAD_PROGRESS:
                with results_lock: processed_backtest_phase += 1; current_progress = processed_backtest_phase
                if current_progress % 10 == 0 or current_progress == total_coins_to_process: # Update seltener
                    console.print(f"Backtest: {current_progress}/{total_coins_to_process} ({ccxt_symbol})...", end="\r")

            # --- Daten holen (Nutzt Caching Funktion get_ohlcv_data) ---
            df = get_ohlcv_data(ccxt_symbol, timeframe, start_dt, end_dt, strategy_instance.startup_candle_count)

            if df is None or df.empty or len(df) < strategy_instance.startup_candle_count:
                # console.print(f"[dim]Skipped (Data): {ccxt_symbol}[/dim]")
                backtest_queue.task_done()
                continue # Nicht genug Daten

            # --- Durchschnittsvolumen-Filter ---
            if avg_vol_threshold > 0 and candles_per_day > 0:
                volume_ok = False
                try:
                    if 'close' in df.columns and 'volume' in df.columns:
                         df['quote_volume'] = pd.to_numeric(df['close'], errors='coerce') * pd.to_numeric(df['volume'], errors='coerce')
                         mean_candle_quote_volume = df['quote_volume'].mean()
                         if not pd.isna(mean_candle_quote_volume):
                             avg_daily_quote_volume = mean_candle_quote_volume * candles_per_day
                             if avg_daily_quote_volume >= avg_vol_threshold: volume_ok = True
                except Exception as e: pass # Bei Fehler überspringen

                if not volume_ok:
                    with results_lock: skipped_avg_volume_worker += 1
                    backtest_queue.task_done()
                    continue
            # --- Ende Volumenfilter ---

            with results_lock: analyzed_count += 1

            # --- Indikatoren berechnen ---
            metadata = {'pair': ccxt_symbol}
            df_populated = None
            try:
                df_populated = strategy_instance.populate_indicators(df.copy(), metadata)
                if not isinstance(df_populated, pd.DataFrame): raise TypeError("populate_indicators hat keinen DataFrame zurückgegeben.")
            except AttributeError:
                 console.print(f"\n[red]Fehler:[/red] Strategie '{strategy_instance.__class__.__name__}' implementiert 'populate_indicators' nicht."); backtest_queue.task_done(); continue
            except Exception as e:
                console.print(f"\n[red]Fehler[/red] in populate_indicators für {ccxt_symbol}: {e}"); backtest_queue.task_done(); continue

            # --- Signale berechnen ---
            try:
                # Direkte Zuweisung der Ergebnisse an neue Spalten
                df_entry = strategy_instance.populate_entry_trend(df_populated.copy(), metadata)
                df_exit = strategy_instance.populate_exit_trend(df_populated.copy(), metadata)

                # Standard Freqtrade Spaltennamen prüfen
                entry_col = 'enter_long' if 'enter_long' in df_entry.columns else 'buy' if 'buy' in df_entry.columns else None
                exit_col = 'exit_long' if 'exit_long' in df_exit.columns else 'sell' if 'sell' in df_exit.columns else None

                df_populated['buy'] = df_entry[entry_col].fillna(0).astype(int) if entry_col else 0
                df_populated['sell'] = df_exit[exit_col].fillna(0).astype(int) if exit_col else 0

            except AttributeError as e:
                 with results_lock: console.print(f"\n[red]Fehler:[/red] Strategie '{strategy_instance.__class__.__name__}' implementiert entry/exit Trend Methoden nicht korrekt. Details: {e}"); backtest_queue.task_done(); continue
            except Exception as e:
                 with results_lock: console.print(f"\n[red]Fehler[/red] in populate_entry/exit_trend für {ccxt_symbol}: {e}"); backtest_queue.task_done(); continue

            # --- Backtest durchführen (mit den neuen Parametern) ---
            trades = run_simple_backtest(
                df_populated,
                strategy_instance,
                fee_percent=fee_percent,
                execute_on_next_open=execute_next_open,
                slippage_percent=slippage_percent
            )

            # --- Ergebnisse speichern ---
            if trades:
                metrics = calculate_metrics(trades, start_dt, end_dt) # Nutzt erweiterte Funktion
                with results_lock: results_list.append({"Pair": ccxt_symbol, **metrics})

            backtest_queue.task_done()

        except Empty: break # Queue ist leer, Thread beenden
        except Exception as e:
            with results_lock: console.print(f"\n[bold red]Unerwarteter Fehler im Backtest-Worker für {ccxt_symbol or 'unbekannt'}:[/bold red] {e}")
            if task_obtained:
                try: backtest_queue.task_done() # Sicherstellen, dass Task abgeschlossen wird
                except ValueError: pass # Falls schon erledigt
                except Exception as e_td: 
                    with results_lock: console.print(f"\n[bold red]Kritischer Fehler:[/bold red] Konnte task_done nach Exception nicht aufrufen: {e_td}")
            break # Thread bei unerwartetem Fehler beenden


# ======== Hauptprogrammablauf ========
def main():
    global total_coins_to_process, processed_data_phase, processed_backtest_phase
    global skipped_mcap, skipped_insufficient_history, skipped_avg_volume_worker, analyzed_count
    global final_candidate_list
    params_to_display_detected = []

    start_time_script = time.time()

    console.print(Rule(f"[bold magenta]===== SCRIPT START (PID: {os.getpid()}) ====="))
    console.print(Rule(f"[bold yellow]Simple Backtester v2 (Fees, NextOpen, Slippage) - {QUOTE_ASSET} Paare[/]"))
    console.print(f"Strategie: [cyan]{STRATEGY_FILE_PATH}[/]")
    console.print(f"Exchange: [cyan]{EXCHANGE_ID}[/], Quote: [cyan]{QUOTE_ASSET}[/], TF: [cyan]{TIMEFRAME}[/]")
    console.print(f"Lookback: [cyan]{LOOKBACK_DAYS} Tage[/], Threads: [cyan]{NUM_THREADS}[/]")
    console.print(f"Prüfung: [cyan]Ausreichend History für {LOOKBACK_DAYS} Tage + Startup?[/] (Vorab geprüft)")
    if MIN_MARKET_CAP_USD > 0: console.print(f"Min. Market Cap: [cyan]${MIN_MARKET_CAP_USD:,.0f} USD[/] (Vorab geprüft)")
    if MIN_AVG_QUOTE_VOLUME_LOOKBACK > 0: console.print(f"Min. Avg Lookback Volumen: [cyan]{MIN_AVG_QUOTE_VOLUME_LOOKBACK:,.0f} {QUOTE_ASSET} / Tag[/] (im Worker geprüft)")
    else: console.print("Avg Lookback Volumen Filter: [dim]Deaktiviert[/dim]")
    # --- NEUE Realismus-Parameter anzeigen ---
    console.print(f"Handelsgebühr: [cyan]{TRADING_FEE_PERCENT:.3f}%[/]")
    console.print(f"Ausführung: [cyan]{'Nächste Kerze (Open)' if EXECUTE_ON_NEXT_OPEN else 'Signalkerze (Close)'}[/]")
    console.print(f"Slippage pro Seite: [cyan]{SLIPPAGE_PERCENT_PER_SIDE:.3f}%[/] {'[dim](Deaktiviert)[/dim]' if SLIPPAGE_PERCENT_PER_SIDE <= 0 else ''}")
    # --- Ende ---
    console.print(f"Cache Dir: [cyan]{CACHE_DIR.resolve()}[/], Max Cache Age: [cyan]{CACHE_MAX_AGE_DAYS if CACHE_MAX_AGE_DAYS > 0 else 'Unbegrenzt'} Tage[/]")


    # --- Schritt 1: Lade Strategie ---
    console.print(Rule("[bold yellow]Schritt 1: Lade Strategie[/]"))
    strategy_template = None
    try:
        strategy_template = load_strategy_from_file(STRATEGY_FILE_PATH)
        console.print(f"[green]Strategie '{strategy_template.__class__.__name__}' erfolgreich geladen.[/]")
        if TIMEFRAME != strategy_template.timeframe: console.print(f"[bold yellow]WARNUNG:[/bold yellow] Backtest-TF '{TIMEFRAME}' != Strategie-TF '{strategy_template.timeframe}'.")
        backtest_end_dt_main = datetime.now(timezone.utc)
        backtest_start_dt_main = backtest_end_dt_main - timedelta(days=LOOKBACK_DAYS)
        # Berechne benötigte Startup-Dauer basierend auf dem globalen TF
        startup_duration_main = timedelta(seconds=strategy_template.startup_candle_count * tf_in_seconds * 1.2) # Verwende tf_in_seconds
        required_data_start_dt_main = backtest_start_dt_main - startup_duration_main
        console.print(f"[dim]Früheste benötigte Kerze (ca.): {required_data_start_dt_main.strftime('%Y-%m-%d %H:%M')} UTC[/dim]")

        console.print("[dim]Ermittle automatisch definierte Strategie-Parameter...[/dim]")
        parameter_classes = (IntParameter, DecimalParameter, CategoricalParameter, BooleanParameter)
        params_to_display_detected = []
        for name, member in inspect.getmembers(strategy_template.__class__):
            if isinstance(member, parameter_classes): params_to_display_detected.append(name)
        params_to_display_detected.sort()
        if params_to_display_detected: console.print(f"[green]Gefundene Parameter zum Anzeigen:[/green] [cyan]{', '.join(params_to_display_detected)}[/]")
        else: console.print("[yellow]Keine optimierbaren Parameter (IntParameter, etc.) in der Strategie gefunden.[/yellow]")
    except (FileNotFoundError, ImportError) as e:
        console.print(f"[bold red]Fehler Strategie laden:[/bold red] {e}"); exit(1)
    if strategy_template is None:
         console.print("[bold red]Strategie konnte nicht geladen werden, Abbruch.[/bold red]"); exit(1)


    # --- Schritt 2: Prüfe & Wende Hyperopt Parameter an ---
    # (Logik bleibt gleich)
    console.print(Rule("[bold yellow]Schritt 2: Prüfe & Wende Hyperopt Parameter an[/]"))
    hyperopt_params_applied = False; potential_json_path = STRATEGY_FILE_PATH.with_suffix('.json')
    hyperopt_params_loaded = {} # Für spätere Prüfung im Bestätigungsdialog
    if potential_json_path.is_file():
        if USE_HYPEROPT_JSON_IF_FOUND:
            console.print(f"Passende Hyperopt-Datei gefunden: [cyan]{potential_json_path}. Versuche Parameter anzuwenden...[/]")
            try:
                with open(potential_json_path, 'r') as f: hyperopt_data = json.load(f)
                # Suche in 'strategy_params' oder 'params'
                if 'strategy_params' in hyperopt_data: hyperopt_params_loaded = hyperopt_data['strategy_params']
                elif 'params' in hyperopt_data: hyperopt_params_loaded = hyperopt_data['params']
                else: hyperopt_params_loaded = {} # Fallback, falls keine der Sektionen existiert

                if not hyperopt_params_loaded:
                     console.print("[yellow]Warnung:[/yellow] Keine 'params' oder 'strategy_params' Sektion in JSON gefunden.")
                else:
                    applied_count = 0; skipped_count = 0; not_found_count = 0
                    # Kombiniere alle Parameter-Sektionen aus der JSON
                    param_sections = {
                        **hyperopt_params_loaded.get('buy', {}),
                        **hyperopt_params_loaded.get('sell', {}),
                        **hyperopt_params_loaded.get('protection', {}),
                        # Füge Top-Level Parameter hinzu (falls nicht in buy/sell/prot)
                        **{k: v for k, v in hyperopt_params_loaded.items() if k not in ['buy', 'sell', 'protection', 'minimal_roi', 'stoploss']}
                    }

                    for param_name, param_value in param_sections.items():
                         if hasattr(strategy_template, param_name):
                             try:
                                 param_obj = getattr(strategy_template, param_name)
                                 # Prüfe, ob es ein Freqtrade Parameter-Objekt ist
                                 if isinstance(param_obj, parameter_classes):
                                     param_obj.value = param_value # Setze den Wert des Parameter-Objekts
                                     applied_count += 1
                                 # Optional: Erlaube Überschreiben von normalen Attributen, wenn nicht Parameter-Objekt?
                                 # else:
                                 #    setattr(strategy_template, param_name, param_value)
                                 #    applied_count += 1 # Zählen oder nicht? Hängt von Anforderung ab
                                 #    console.print(f"[dim]Info: Attribut '{param_name}' überschrieben (kein Parameter-Objekt).[/dim]")
                                 else: skipped_count +=1 # Kein Parameter-Objekt, überspringen
                             except Exception as e: console.print(f"[yellow]Warnung:[/yellow] Param '{param_name}' Apply-Fehler: {e}"); skipped_count += 1
                         else: not_found_count += 1

                    # Speziell minimal_roi und stoploss behandeln
                    if 'minimal_roi' in hyperopt_params_loaded:
                         try:
                             # Stelle sicher, dass das Format passt (manchmal als String in JSON)
                             roi_data = hyperopt_params_loaded['minimal_roi']
                             if isinstance(roi_data, dict):
                                 strategy_template.minimal_roi = {int(k): v for k, v in roi_data.items()} # Schlüssel zu int konvertieren
                                 applied_count += 1
                             else:
                                 console.print(f"[yellow]Warnung:[/yellow] 'minimal_roi' in JSON ist kein Dictionary.")
                                 skipped_count += 1
                         except Exception as e: console.print(f"[yellow]Warnung:[/yellow] Param 'minimal_roi' Apply-Fehler: {e}"); skipped_count += 1
                    if 'stoploss' in hyperopt_params_loaded:
                         try: strategy_template.stoploss = hyperopt_params_loaded['stoploss']; applied_count += 1
                         except Exception as e: console.print(f"[yellow]Warnung:[/yellow] Param 'stoploss' Apply-Fehler: {e}"); skipped_count += 1

                    console.print(f"[green]{applied_count} Hyperopt-Parameter angewendet.[/]");
                    if skipped_count > 0: console.print(f"[yellow]{skipped_count} Parameter konnten nicht angewendet werden (Typ/Fehler/Attribut).[/]")
                    if not_found_count > 0: console.print(f"[dim]{not_found_count} Parameter aus JSON in Strategie nicht gefunden.[/dim]")
                    if applied_count > 0: hyperopt_params_applied = True
            except json.JSONDecodeError as e: console.print(f"[red]Fehler[/red] beim Parsen der Hyperopt-JSON ({potential_json_path}): {e}. Standardwerte.")
            except Exception as e: console.print(f"[red]Fehler[/red] beim Verarbeiten der Hyperopt-Datei ({potential_json_path}): {e}. Standardwerte.")
        else: console.print(f"[dim]Hyperopt-Datei ({potential_json_path.name}) gefunden, ignoriert (Konfig=False). Standardwerte.[/dim]")
    else: console.print(f"[dim]Keine Hyperopt-Datei ({potential_json_path.name}) gefunden. Standardwerte.[/dim]")


    # --- Schritt 3: Hole Market Cap Daten ---
    # (Logik bleibt gleich)
    console.print(Rule("[bold yellow]Schritt 3: Hole Market Cap Daten von CoinGecko[/]"))
    mcap_map = {};
    if MIN_MARKET_CAP_USD > 0:
        console.print(f"Frage Top 250 Coins von CoinGecko für Market Cap Daten ab...")
        try:
            # Nutze safe_api_call für Robustheit
            cg_market_data_raw = safe_api_call(cg.get_coins_markets, vs_currency='usd', order='market_cap_desc', per_page=250, page=1, sparkline=False)
            if not cg_market_data_raw: console.print("[bold yellow]Warnung:[/bold yellow] Keine Mcap Daten von CG erhalten. Filter ignoriert.")
            else:
                 for coin in cg_market_data_raw:
                     symbol_upper = coin.get('symbol', '').upper(); market_cap = coin.get('market_cap')
                     if symbol_upper and market_cap is not None: mcap_map[symbol_upper] = market_cap
                 console.print(f"[green]{len(mcap_map)} Coins mit Market Cap Daten geladen.[/]")
        except Exception as e: console.print(f"[bold yellow]Warnung:[/bold yellow] Fehler beim Abrufen/Verarbeiten der CG Mcap Daten: {e}. Filter ignoriert."); mcap_map = {}
    else: console.print("[dim]Market Cap Filter deaktiviert.[/dim]")


    # --- Schritt 4: Finde Kandidaten nach Mcap ---
    # (Logik bleibt gleich)
    console.print(Rule("[bold yellow]Schritt 4: Finde Kandidaten nach Mcap[/]"))
    mcap_filtered_candidates = []
    skipped_mcap = 0
    console.print(f"Filtere {len(markets)} Binance Märkte (SPOT, Quote={QUOTE_ASSET}, Mcap)...")
    active_market_count = 0
    for symbol, market_data in markets.items():
        # Prüfe, ob der Markt aktiv ist, ein Spot-Markt ist und das richtige Quote-Asset hat
        if (market_data.get('active', False) and
            market_data.get('spot', False) and
            market_data.get('quote') == QUOTE_ASSET):
            active_market_count += 1
            base_asset = market_data.get('base')
            # Market Cap Filter anwenden, falls aktiv und Daten vorhanden
            if base_asset and MIN_MARKET_CAP_USD > 0 and mcap_map:
                market_cap = mcap_map.get(base_asset.upper(), 0) # Suche Upper-Case
                if market_cap < MIN_MARKET_CAP_USD:
                    skipped_mcap += 1
                    continue # Überspringe diesen Coin
            # Wenn alle Filter bestanden, zur Liste hinzufügen
            mcap_filtered_candidates.append(symbol)

    console.print(f"{len(mcap_filtered_candidates)} Kandidaten nach Spot/Quote Filterung.")
    if MIN_MARKET_CAP_USD > 0: console.print(f"[dim] ({skipped_mcap} von {active_market_count} aktiven {QUOTE_ASSET}-Paaren wegen Market Cap < ${MIN_MARKET_CAP_USD:,.0f} übersprungen)[/dim]")
    else: console.print(f"[dim] (Market Cap Filter war deaktiviert)[/dim]")


    # --- Schritt 5: Paralleler Vorab-Check der History ---
    # (Logik bleibt gleich)
    console.print(Rule(f"[bold yellow]Schritt 5: Parallele Prüfung der Datenhistorie ({NUM_THREADS} Threads)[/]"))
    check_queue = Queue()
    final_candidate_list = [] # Zurücksetzen für diesen Lauf
    skipped_insufficient_history = 0
    processed_data_phase = 0 # Zähler für diese Phase
    total_coins_to_process_phase1 = len(mcap_filtered_candidates)

    if total_coins_to_process_phase1 > 0:
        console.print(f"Prüfe für {total_coins_to_process_phase1} Kandidaten, ob Daten bis ca. {required_data_start_dt_main.strftime('%Y-%m-%d %H:%M')} UTC zurückreichen...")
        with results_lock: total_coins_to_process = total_coins_to_process_phase1 # Globaler Zähler für Fortschrittsanzeige
        for symbol in mcap_filtered_candidates: check_queue.put(symbol)

        check_threads = []
        for _ in range(NUM_THREADS):
            thread = threading.Thread(target=history_check_worker, args=(check_queue, required_data_start_dt_main, TIMEFRAME), daemon=True)
            thread.start(); check_threads.append(thread)

        # Warten, bis alle History-Checks abgeschlossen sind
        check_queue.join()
        # Kurze Pause, um sicherzustellen, dass alle Threads wirklich beendet sind (optional)
        time.sleep(0.2)

        if PRINT_THREAD_PROGRESS: print(" " * (console.width -1), end="\r") # Letzte Fortschrittsanzeige löschen
        console.print(f"[green]History-Prüfung abgeschlossen.[/]")
        console.print(f"[green]{len(final_candidate_list)} Paare haben ausreichende Historie.[/]")
        if skipped_insufficient_history > 0: console.print(f"[dim] ({skipped_insufficient_history} von {total_coins_to_process_phase1} wegen unzureichender Historie/Datenfehler übersprungen)[/dim]")
    else:
        console.print("[yellow]Keine Kandidaten nach Mcap-Filterung übrig. Backtest nicht möglich.[/]"); exit(0)


    # --- Schritt 6: Definiere Backtest-Zeitrahmen ---
    # (Bleibt gleich)
    start_dt = backtest_start_dt_main
    end_dt = backtest_end_dt_main
    console.print(Rule("[bold yellow]Schritt 6: Definiere Backtest-Zeitrahmen[/]"))
    console.print(f"Backtest Zeitraum: [cyan]{start_dt.strftime('%Y-%m-%d %H:%M')}[/] bis [cyan]{end_dt.strftime('%Y-%m-%d %H:%M')} UTC[/]")
    console.print(f"Daten inkl. Startup Candles benötigt ab: ~[cyan]{required_data_start_dt_main.strftime('%Y-%m-%d %H:%M')} UTC[/]")


    # --- Schritt 7: Bestätigungs-Dialog (mit neuen Parametern) ---
    console.print(Rule("[bold yellow]Schritt 7: Bestätigung[/]"))
    # Parameter-Quelle bestimmen
    if not USE_HYPEROPT_JSON_IF_FOUND: param_source_info = Text.from_markup("[dim]Standardwerte (Hyperopt-JSON ignoriert via Konfig)[/dim]")
    elif hyperopt_params_applied: param_source_info = Text.from_markup(f"aus [cyan]{potential_json_path.name}[/]")
    else: param_source_info = Text.from_markup("[dim]Standardwerte (Keine .json / Fehler / nicht angewendet)[/dim]")

    # Aktive Parameter anzeigen
    active_params_text = Text("Aktive Strategie-Parameter:\n", style="bold white")
    param_found_display = False
    if not params_to_display_detected: active_params_text = Text("Keine automatisch erkannten Parameter zum Anzeigen.", style="yellow")
    else:
        for param_name in params_to_display_detected:
            if hasattr(strategy_template, param_name):
                try:
                    param_obj = getattr(strategy_template, param_name)
                    if hasattr(param_obj, 'value'): # Ist es ein Freqtrade Parameter Objekt?
                        current_value = param_obj.value
                        # Prüfen, ob dieser spezifische Parameter in der geladenen Hyperopt-JSON war
                        # (auch in verschachtelten Sektionen wie 'buy' oder 'sell')
                        param_key_exists_in_hyperopt = False
                        if hyperopt_params_loaded:
                            if param_name in hyperopt_params_loaded.get('buy', {}): param_key_exists_in_hyperopt = True
                            elif param_name in hyperopt_params_loaded.get('sell', {}): param_key_exists_in_hyperopt = True
                            elif param_name in hyperopt_params_loaded.get('protection', {}): param_key_exists_in_hyperopt = True
                            elif param_name in hyperopt_params_loaded: param_key_exists_in_hyperopt = True # Auch Top-Level prüfen

                        value_style = "cyan" if (hyperopt_params_applied and param_key_exists_in_hyperopt) else "dim"
                        active_params_text.append(Text.assemble(f"  - {param_name}: ", (f"{current_value}", value_style), "\n"))
                        param_found_display = True
                    else:
                         # Falls es kein Parameter-Objekt ist, aber existiert (z.B. normales Attribut)
                         current_value = getattr(strategy_template, param_name)
                         active_params_text.append(f"  - {param_name}: {current_value} ([dim]Standard-Attribut[/dim])\n"); param_found_display = True
                except AttributeError: active_params_text.append(f"  - {param_name}: [red]Fehler beim Lesen[/red]\n")

        # Stoploss und ROI speziell behandeln
        is_sl_hyperopt = hyperopt_params_applied and 'stoploss' in hyperopt_params_loaded
        is_roi_hyperopt = hyperopt_params_applied and 'minimal_roi' in hyperopt_params_loaded
        active_params_text.append(Text.assemble(f"  - stoploss: ", (f"{strategy_template.stoploss}", "cyan" if is_sl_hyperopt else "dim"), "\n"))
        # ROI kann lang sein, evtl. kürzen oder nur Hinweis geben? Hier vollständig:
        roi_display = str(strategy_template.minimal_roi)
        if len(roi_display) > 60: roi_display = roi_display[:57] + "..." # Kürzen bei Bedarf
        active_params_text.append(Text.assemble(f"  - minimal_roi: ", (f"{roi_display}", "cyan" if is_roi_hyperopt else "dim"), "\n"))
        param_found_display = True

        if not param_found_display and params_to_display_detected: active_params_text = Text("Konnte Werte für erkannte Parameter nicht lesen.", style="red")


    # Zusammenfassung für Panel
    summary_text = Text.assemble(
        ("Strategie:", "bold white"), f" {strategy_template.__class__.__name__} ({STRATEGY_FILE_PATH.name})\n",
        ("Parameter Quelle:", "bold white"), " ", param_source_info, "\n", active_params_text,
        ("\nExchange:", "bold white"), f" {EXCHANGE_ID}\n", ("Quote Asset:", "bold white"), f" {QUOTE_ASSET}\n",
        ("Timeframe:", "bold white"), f" {TIMEFRAME}\n", ("Lookback:", "bold white"), f" {LOOKBACK_DAYS} Tage\n",
        ("Min. Market Cap:", "bold white"), f" ${MIN_MARKET_CAP_USD:,.0f}\n" if MIN_MARKET_CAP_USD > 0 else " Deaktiviert\n",
        ("History Check:", "bold white"), " Ausreichend Daten (Vorab geprüft)\n",
        ("Min. Avg Volume:", "bold white"), f" {MIN_AVG_QUOTE_VOLUME_LOOKBACK:,.0f} {QUOTE_ASSET}/Tag\n" if MIN_AVG_QUOTE_VOLUME_LOOKBACK > 0 else " Deaktiviert\n",
        # --- NEUE Parameter in der Zusammenfassung ---
        ("\nHandelsgebühr:", "bold white"), f" {TRADING_FEE_PERCENT:.3f}%\n",
        ("Order Ausführung:", "bold white"), f" {'Nächste Kerze (Open)' if EXECUTE_ON_NEXT_OPEN else 'Signalkerze (Close)'}\n",
        ("Slippage / Seite:", "bold white"), f" {SLIPPAGE_PERCENT_PER_SIDE:.3f}% {'(Deaktiviert)' if SLIPPAGE_PERCENT_PER_SIDE <= 0 else ''}\n",
        # --- Ende ---
        ("\nZu prüfende Paare:", "bold white"), f" {len(final_candidate_list)}\n",
        ("Threads:", "bold white"), f" {NUM_THREADS}",
    )
    console.print(Panel(summary_text, title="Backtest Konfiguration", border_style="yellow", padding=(1, 2)))

    if not final_candidate_list:
         console.print("\n[bold red]Keine Kandidaten nach allen Filtern übrig. Backtest wird nicht gestartet.[/bold red]"); exit(0)

    try:
        # Timeout für Input, falls Skript unbeaufsichtigt läuft (optional)
        # confirmation = console.input("\n[bold]Backtest starten? ([green]ja[/]/[red]nein[/]): [/]", timeout=60).lower().strip()
        confirmation = console.input("\n[bold]Backtest starten? ([green]ja[/]/[red]nein[/]): [/]").lower().strip()
        if confirmation not in ['yes', 'y', 'ja', 'j']:
            console.print("\n[yellow]Vorgang abgebrochen.[/]"); sys.exit(0)
        else:
            console.print("[green]Bestätigt. Starte parallele Verarbeitung...[/]")
    except (EOFError, KeyboardInterrupt): console.print("\n[yellow]Eingabe abgebrochen. Beende.[/]"); sys.exit(0)
    # except TimeoutError: console.print("\n[yellow]Timeout bei Eingabe. Beende.[/]"); sys.exit(0)


    # --- Schritt 8: Parallele Verarbeitung (Backtest mit neuen Parametern) ---
    console.print(Rule(f"[bold yellow]Schritt 8: Starte parallele Backtests ({NUM_THREADS} Threads)[/]"))
    backtest_queue = Queue()
    results_list.clear() # Alte Ergebnisse löschen
    for symbol in final_candidate_list: backtest_queue.put(symbol)

    total_coins_to_process_phase2 = len(final_candidate_list)
    with results_lock: total_coins_to_process = total_coins_to_process_phase2 # Für Fortschrittsanzeige

    processed_backtest_phase = 0; skipped_avg_volume_worker = 0; analyzed_count = 0 # Zähler zurücksetzen

    if total_coins_to_process_phase2 == 0: console.print("[yellow]Keine gültigen Coins zum Verarbeiten.[/]"); exit(0)

    console.print(f"Verarbeite {total_coins_to_process_phase2} Paare...")
    threads = []
    for _ in range(NUM_THREADS):
        # Übergebe ALLE notwendigen Parameter an den Worker
        thread = threading.Thread(
            target=backtest_worker,
            args=(
                backtest_queue,
                strategy_template,
                start_dt,
                end_dt,
                TIMEFRAME,
                MIN_AVG_QUOTE_VOLUME_LOOKBACK, # Volumenfilter
                # Neue Realismus-Parameter:
                TRADING_FEE_PERCENT,
                EXECUTE_ON_NEXT_OPEN,
                SLIPPAGE_PERCENT_PER_SIDE
            ),
            daemon=True # Threads beenden sich, wenn Hauptprogramm endet
        )
        thread.start(); threads.append(thread)

    # Warten, bis alle Backtests abgeschlossen sind
    backtest_queue.join()
    time.sleep(0.2) # Kurze Pause

    if PRINT_THREAD_PROGRESS: print(" " * (console.width -1), end="\r") # Letzte Fortschrittsanzeige löschen
    console.print(f"[green]Alle {total_coins_to_process_phase2} Kandidaten geprüft.[/]")
    console.print(f"[dim]Backtest-Zusammenfassung:[/dim]")
    console.print(f"[dim]  - {skipped_avg_volume_worker} wegen Durchschnittsvolumen im Worker übersprungen.[/dim]")
    console.print(f"[dim]  - {analyzed_count} tatsächlich analysiert (Filter passiert & Daten ok).[/dim]")
    console.print(f"[dim]  - {len(results_list)} Paare haben Trades generiert.[/dim]")


    # --- Schritt 9: Ergebnisse (Sortierung und Anzeige) ---
    console.print(Rule("[bold yellow]Schritt 9: Ergebnisse[/]"))
    if not results_list:
        console.print("[bold yellow]Keine Trades generiert / Alle Kandidaten hatten keine Signale oder wurden gefiltert.[/]");
        end_time_script = time.time()
        console.print(f"\nScript Laufzeit: {end_time_script - start_time_script:.2f} Sekunden.")
        console.print(Rule(f"[bold magenta]===== SCRIPT END (PID: {os.getpid()}) ====="))
        exit(0)

    # Sortierung der Ergebnisse (Logik bleibt gleich)
    try:
        sort_key = SORT_RESULTS_BY
        reverse_sort = True # Standard: Absteigend
        # Spezielle Behandlung für bestimmte Schlüssel
        if sort_key == "Max Drawdown %":
            reverse_sort = False # Aufsteigend (kleinster Drawdown ist besser)
            def get_sort_value(x):
                val_str = x.get(sort_key, "0.00%").replace('%', '')
                try: return float(val_str)
                except ValueError: return float('inf') # Fehler nach unten sortieren (schlechter Drawdown)
        elif sort_key == "Profit Factor":
            def get_sort_value(x):
                 pf_str = x.get(sort_key, "0.0");
                 try: return float('inf') if pf_str == "Inf" else float(pf_str)
                 except ValueError: return -float('inf')
        elif sort_key in ["Total PnL %", "Avg Win %", "Avg Loss %", "Win Rate"]:
             def get_sort_value(x):
                 val_str = x.get(sort_key, "0.00%").replace('%', '')
                 try: return float(val_str)
                 except ValueError: return -float('inf')
        elif sort_key == "Trades":
            sort_key = "Total Trades" # Interner Key
            def get_sort_value(x):
                 val_str = str(x.get(sort_key, "0"))
                 try: return int(val_str)
                 except ValueError: return -1
        elif sort_key == "Sharpe Ratio":
             def get_sort_value(x):
                 val_str = x.get(sort_key, "0.00")
                 try: return float(val_str)
                 except ValueError: return -float('inf')
        else:
            console.print(f"[yellow]Warnung:[/yellow] Ungültiger Sortierschlüssel '{sort_key}'. Sortiere nach 'Sharpe Ratio'.")
            sort_key = "Sharpe Ratio"; reverse_sort = True
            def get_sort_value(x):
                 val_str = x.get(sort_key, "0.00")
                 try: return float(val_str)
                 except ValueError: return -float('inf')

        results_list.sort(key=get_sort_value, reverse=reverse_sort)
        console.print(f"[dim]Ergebnisse sortiert nach '{SORT_RESULTS_BY}' ({'absteigend' if reverse_sort else 'aufsteigend'}, {len(results_list)} Paare mit Trades).[/dim]")

    except Exception as e: console.print(f"[yellow]Warnung: Sortierung fehlgeschlagen: {e}. Ergebnisse unsortiert.[/]")

    # Tabelle mit Ergebnissen (Spalten bleiben gleich)
    table = Table(title=f"Backtest: '{strategy_template.__class__.__name__}' | {QUOTE_ASSET} | {TIMEFRAME} | {EXCHANGE_ID}", show_header=True, header_style="bold magenta", border_style="dim blue")
    table.add_column("Rank", style="dim", width=4, justify="right")
    table.add_column("Pair", style="cyan", no_wrap=True)
    table.add_column("Trades", style="blue", justify="right")
    table.add_column("Win Rate", style="yellow", justify="right")
    table.add_column("Profit Factor", style="magenta", justify="right")
    table.add_column("Max DD %", style="red", justify="right") # Max Drawdown
    table.add_column("Sharpe", style="blue", justify="right")  # Sharpe Ratio
    table.add_column("Total PnL %", style="bold", justify="right") # PnL nach Gebühren/Slippage

    for index, res in enumerate(results_list):
        # Dynamische Farbgebung für PnL, Sharpe, Drawdown
        pnl_str = res.get("Total PnL %", "0.00%")
        pnl_style = "dim"
        try: pnl_val = float(pnl_str.replace('%', '')); pnl_style = "green" if pnl_val > 0 else "red" if pnl_val < 0 else "dim"
        except ValueError: pass

        sharpe_str = res.get("Sharpe Ratio", "0.00")
        sharpe_style = "dim"
        try: sharpe_val = float(sharpe_str); sharpe_style = "green" if sharpe_val > 1.0 else "yellow" if sharpe_val > 0.5 else "red" if sharpe_val < 0 else "dim"
        except ValueError: pass

        mdd_str = res.get("Max Drawdown %", "0.00%")
        mdd_style = "green" # Geringer Drawdown ist gut (näher an 0)
        try: mdd_val = float(mdd_str.replace('%','')); mdd_style = "red" if mdd_val < -25 else "yellow" if mdd_val < -15 else "green" # Schwellen anpassen
        except ValueError: pass

        table.add_row(
            str(index + 1),
            res.get("Pair", "N/A"),
            str(res.get("Total Trades", "0")),
            res.get("Win Rate", "0.00%"),
            res.get("Profit Factor", "0.00"),
            Text(mdd_str, style=mdd_style),
            Text(sharpe_str, style=sharpe_style),
            Text(pnl_str, style=pnl_style)
        )
    console.print(table)
    console.print(f"\n[dim]Getestet über {LOOKBACK_DAYS} Tage vom {start_dt.strftime('%Y-%m-%d')} bis {end_dt.strftime('%Y-%m-%d')}.[/dim]")
    console.print(f"[dim]Sortiert nach: [white]{SORT_RESULTS_BY}[/white] ({'absteigend' if reverse_sort else 'aufsteigend'}).[/dim]")
    console.print(f"[dim]Parameter: Fee={TRADING_FEE_PERCENT}%, NextOpen={EXECUTE_ON_NEXT_OPEN}, Slippage={SLIPPAGE_PERCENT_PER_SIDE}%[/dim]")


    # --- Schritt 10: Whitelist generieren ---
    # (Logik bleibt gleich)
    if results_list:
         console.print("\n"); console.print(Rule("[bold yellow]Freqtrade Whitelist generieren[/]"))
         num_results = len(results_list); num_assets_to_whitelist = 0
         try:
             while True:
                 prompt = f"Wie viele der Top [bold cyan]{num_results}[/] Paare (gemäß Sortierung) sollen in die Whitelist? ([cyan]Zahl[/], [cyan]0[/] für keine): "
                 num_assets_str = console.input(prompt)
                 try:
                     num_assets_to_whitelist = int(num_assets_str)
                     if 0 <= num_assets_to_whitelist <= num_results: break
                     else: console.print(f"[red]Fehler:[/red] Zahl zwischen 0 und {num_results} eingeben.")
                 except ValueError: console.print("[red]Fehler:[/red] Ungültige Zahl.")

             if num_assets_to_whitelist > 0:
                 ordered_whitelist_pairs = [res.get("Pair") for res in results_list[:num_assets_to_whitelist] if res.get("Pair")]
                 if ordered_whitelist_pairs:
                     whitelist_content = '    "pair_whitelist": [\n'
                     for i, pair_slash in enumerate(ordered_whitelist_pairs):
                         whitelist_content += f'        "{pair_slash}"'
                         whitelist_content += ',\n' if i < len(ordered_whitelist_pairs) - 1 else '\n'
                     whitelist_content += '    ],'
                     console.print(f"\n[green]Generiere Whitelist für die Top {num_assets_to_whitelist} Paare:[/]")
                     console.print(Text("↓↓↓ FÜR CONFIG KOPIEREN ↓↓↓", style="bold yellow"))
                     print(whitelist_content) # Direkte Ausgabe für einfaches Kopieren
                     console.print(Text("↑↑↑ FÜR CONFIG KOPIEREN ↑↑↑", style="bold yellow"))
                 else: console.print("[yellow]Konnte keine Paare für die Whitelist extrahieren.[/]")
             else: console.print("\n[yellow]Keine Whitelist generiert (Anzahl 0 gewählt).[/]")
         except (EOFError, KeyboardInterrupt):
             console.print("\n[yellow]Eingabe abgebrochen. Keine Whitelist generiert.[/]")


    # --- Schlussbemerkung (aktualisiert) ---
    end_time_script = time.time()
    console.print(f"\nScript Laufzeit: {end_time_script - start_time_script:.2f} Sekunden.")
    console.print("\n" + "="*80)
    disclaimer = "[bold orange1]WICHTIG:[/bold orange1] Vereinfachter Backtest! Berücksichtigt jetzt:"
    disclaimer += f"\n- Handelsgebühren ({TRADING_FEE_PERCENT}%)"
    disclaimer += f"\n- Orderausführung ({'Nächste Kerze Open' if EXECUTE_ON_NEXT_OPEN else 'Signalkerze Close'})"
    disclaimer += f"\n- Optionale Slippage ({SLIPPAGE_PERCENT_PER_SIDE}%)"
    disclaimer += "\nEs werden weiterhin KEINE ROI-Tabelle, Trailing Stops (außer dem fixen SL) oder Stake-Amount berücksichtigt."
    disclaimer += "\nStop-Loss wird auf Kerzenbasis ('low') geprüft."
    console.print(disclaimer)
    console.print("="*80)
    console.print(Rule(f"[bold magenta]===== SCRIPT END (PID: {os.getpid()}) ====="))
    sys.exit(0)

if __name__ == "__main__":
    main()

# --- END OF FILE simple_backtester_top_coins_v2_realistic.py ---
