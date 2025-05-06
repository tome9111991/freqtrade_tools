import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime
import time
import sys
import os

# --- User Configuration ---
# Hier die gewünschten Einstellungen vornehmen:
SYMBOL = "BTCUSDC"          # Handelspaar (z.B., BTCUSDT, ETHBTC, ADAEUR)
INTERVAL = "4h"             # Zeitrahmen/Intervall (z.B., 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M)
START_DATE = "2025-04-14"   # Startdatum (Format: YYYY-MM-DD oder YYYY-MM-DD HH:MM:SS)
END_DATE = "2025-04-17"     # Enddatum (Format: YYYY-MM-DD oder YYYY-MM-DD HH:MM:SS)
                            # Setze END_DATE = None, um Daten bis zur aktuellen Zeit zu laden
OUTPUT_FILENAME = None      # Gewünschter Dateiname für die CSV (z.B., "btc_ohlcv_data.csv")
                            # Setze OUTPUT_FILENAME = None für automatische Benennung (SYMBOL_INTERVAL_START_END_OHLCV.csv)
# --- End User Configuration ---


# Definiere die Spaltennamen, wie sie *ursprünglich* von der Binance API kommen
# ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
KLINE_COLUMNS_FROM_API = [
    'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades',
    'Taker_Buy_Base_Asset_Volume', 'Taker_Buy_Quote_Asset_Volume', 'Ignore'
]

# Definiere die Spalten, die wir *behalten* wollen, und ihre gewünschten Namen
OHLCV_COLUMNS_TO_KEEP = {
    'Open_Time': 'Timestamp', # Neuer Name für die Zeitspalte
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Volume': 'Volume'
}


# Gültige Binance Intervalle/Timeframes (zur Validierung)
VALID_INTERVALS = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M"
]

def date_to_milliseconds(date_str):
    """Konvertiert einen Datumsstring in Millisekunden seit Epoche."""
    if date_str is None:
        return None
    try:
        dt_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            dt_obj = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
    return int(dt_obj.timestamp() * 1000)

def get_binance_klines(client, symbol, interval, start_ms, end_ms):
    """Holt Klines von Binance in Chunks."""
    all_klines = []
    current_start_ms = start_ms
    limit = 1000 # Binance API Limit pro Request

    start_dt_str = datetime.fromtimestamp(start_ms/1000).strftime('%Y-%m-%d %H:%M:%S')
    end_dt_str = datetime.fromtimestamp(end_ms/1000).strftime('%Y-%m-%d %H:%M:%S')

    print(f"Fetching data for {symbol} ({interval}) from {start_dt_str} to {end_dt_str}...")

    while current_start_ms < end_ms:
        try:
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=str(current_start_ms),
                end_str=str(end_ms),
                limit=limit
            )

            if not klines:
                print("No more data found for the period or reached end date.")
                break

            all_klines.extend(klines)
            last_kline_time = klines[-1][0]
            current_start_ms = last_kline_time + 1

            last_date_str = datetime.fromtimestamp(last_kline_time / 1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  Fetched {len(klines)} records up to {last_date_str}. Total: {len(all_klines)}")

            time.sleep(0.2)

        except BinanceAPIException as e:
            print(f"Binance API Exception: {e}")
            print("Retrying after 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"An unexpected error occurred during fetch: {e}")
            return None

    print(f"Finished fetching. Total raw records retrieved: {len(all_klines)}")
    return all_klines

def main():
    """Hauptfunktion des Skripts."""

    print("--- Binance OHLCV Data Downloader ---")
    print(f"Symbol:   {SYMBOL}")
    print(f"Interval: {INTERVAL}")
    print(f"Start:    {START_DATE}")
    print(f"End:      {END_DATE if END_DATE else 'Current Time'}")
    print(f"Output:   {OUTPUT_FILENAME if OUTPUT_FILENAME else 'Automatic'}")
    print("------------------------------------")

    if INTERVAL not in VALID_INTERVALS:
        print(f"Error: Invalid interval '{INTERVAL}'. Please choose from: {', '.join(VALID_INTERVALS)}")
        sys.exit(1)

    client = Client(api_key=None, api_secret=None)
    try:
        client.ping()
        print("Successfully connected to Binance Public API.")
    except Exception as e:
        print(f"Error connecting to Binance API: {e}")
        sys.exit(1)

    try:
        start_ms = date_to_milliseconds(START_DATE)
        if start_ms is None:
             print("Error: START_DATE cannot be empty.")
             sys.exit(1)
    except ValueError as e:
        print(f"Error parsing start date '{START_DATE}': {e}")
        sys.exit(1)

    if END_DATE:
        try:
            end_ms = date_to_milliseconds(END_DATE)
        except ValueError as e:
            print(f"Error parsing end date '{END_DATE}': {e}")
            sys.exit(1)
    else:
        end_ms = int(datetime.now().timestamp() * 1000)
        print(f"End date not specified, using current time: {datetime.fromtimestamp(end_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}")

    if start_ms >= end_ms:
        print(f"Error: Start date ({START_DATE}) must be before end date ({datetime.fromtimestamp(end_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}).")
        sys.exit(1)

    # Hole die Rohdaten (immer noch mit allen Spalten von der API)
    klines_data = get_binance_klines(client, SYMBOL.upper(), INTERVAL, start_ms, end_ms)

    if klines_data is None:
        print("Failed to retrieve data due to errors.")
        sys.exit(1)
    if not klines_data:
         print("No data found for the specified parameters.")
         sys.exit(0)

    # Erstelle einen Pandas DataFrame aus den Rohdaten
    print("Processing data...")
    df_raw = pd.DataFrame(klines_data, columns=KLINE_COLUMNS_FROM_API)

    # --- Datenbereinigung und Auswahl ---
    # 1. Wähle nur die Spalten aus, die wir brauchen
    df_ohlcv = df_raw[list(OHLCV_COLUMNS_TO_KEEP.keys())].copy() # .copy() vermeidet SettingWithCopyWarning

    # 2. Benenne die Spalten um (insbesondere 'Open_Time' zu 'Timestamp')
    df_ohlcv.rename(columns=OHLCV_COLUMNS_TO_KEEP, inplace=True)

    # 3. Konvertiere den Timestamp zu einem lesbaren Datetime-Objekt
    df_ohlcv['Timestamp'] = pd.to_datetime(df_ohlcv['Timestamp'], unit='ms')

    # 4. Konvertiere die OHLCV-Spalten zu numerischen Typen (Floats)
    ohlcv_numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in ohlcv_numeric_cols:
        df_ohlcv[col] = pd.to_numeric(df_ohlcv[col], errors='coerce')

    # 5. Sortiere nach Zeitstempel (sollte schon so sein, aber zur Sicherheit)
    df_ohlcv.sort_values('Timestamp', inplace=True)

    # 6. Entferne eventuelle Duplikate basierend auf dem Zeitstempel
    initial_rows = len(df_ohlcv)
    df_ohlcv.drop_duplicates(subset=['Timestamp'], keep='first', inplace=True)
    if len(df_ohlcv) < initial_rows:
        print(f"Removed {initial_rows - len(df_ohlcv)} duplicate rows based on Timestamp.")

    # 7. Filtere erneut nach dem exakten Zeitbereich
    start_dt = datetime.fromtimestamp(start_ms / 1000)
    end_dt = datetime.fromtimestamp(end_ms / 1000) # Exklusiv

    if df_ohlcv['Timestamp'].dt.tz is not None:
        start_dt = start_dt.replace(tzinfo=df_ohlcv['Timestamp'].dt.tz)
        end_dt = end_dt.replace(tzinfo=df_ohlcv['Timestamp'].dt.tz)

    original_count_before_filter = len(df_ohlcv)
    df_ohlcv = df_ohlcv[(df_ohlcv['Timestamp'] >= start_dt) & (df_ohlcv['Timestamp'] < end_dt)]
    if len(df_ohlcv) < original_count_before_filter:
         print(f"Filtered out {original_count_before_filter - len(df_ohlcv)} records outside the precise time range ({start_dt} to {end_dt}).")


    if df_ohlcv.empty:
        print("No OHLCV data remains after processing and filtering.")
        sys.exit(0)

    # Bestimme den Output-Dateinamen
    output_file = OUTPUT_FILENAME
    if not output_file:
        start_date_str_formatted = datetime.fromtimestamp(start_ms/1000).strftime('%Y%m%d')
        if END_DATE:
            end_date_str_formatted = datetime.fromtimestamp(end_ms/1000).strftime('%Y%m%d')
        else:
             end_date_str_formatted = "now"

        # Füge "_OHLCV" hinzu, um klarzustellen, dass es sich um reduzierte Daten handelt
        output_file = f"{SYMBOL.upper()}_{INTERVAL}_{start_date_str_formatted}_to_{end_date_str_formatted}_OHLCV.csv"

    # Stelle sicher, dass der Output-Pfad existiert
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            sys.exit(1)

    # Speichere den finalen OHLCV DataFrame als CSV
    try:
        # Verwende ein Standard-Datumsformat für die CSV-Ausgabe
        df_ohlcv.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Successfully downloaded and processed {len(df_ohlcv)} OHLCV records.")
        print(f"Data saved to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"Error saving OHLCV data to CSV '{output_file}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
