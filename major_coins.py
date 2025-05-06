# -*- coding: utf-8 -*-

import operator
from binance.client import Client

# --- Konfiguration ---
TOP_N = 20  # Wie viele Top-Coins sollen angezeigt werden?
QUOTE_ASSET = "USDC" # Gegen welchen Stablecoin soll das Volumen gemessen werden? (USDT ist am gängigsten)
# Andere Optionen könnten BUSD, USDC, TUSD sein, aber USDT hat meist das höchste Volumen.
# --- Ende Konfiguration ---

def get_major_coins_by_volume_on_binance(top_n=20, quote_asset="USDT"):
    """
    Ermittelt die Top N Kryptowährungen auf Binance basierend auf dem 24h-Handelsvolumen
    gegen den angegebenen Quote-Asset (z.B. USDT).

    Args:
        top_n (int): Die Anzahl der Top-Coins, die zurückgegeben werden sollen.
        quote_asset (str): Der Quote-Asset (z.B. 'USDT', 'BUSD'), gegen den das Volumen gemessen wird.

    Returns:
        list: Eine Liste von Dictionaries, wobei jedes Dictionary einen Top-Coin
              und sein Volumen enthält, oder None bei einem Fehler.
              Format: [{'base_asset': 'BTC', 'symbol': 'BTCUSDT', 'volume': 12345678.90}, ...]
    """
    print(f"Ermittle die Top {top_n} Coins nach 24h Handelsvolumen gegen {quote_asset} auf Binance...")

    try:
        # Initialisiere den Binance Client (ohne API Keys für öffentliche Daten)
        client = Client()

        # Rufe die 24-Stunden-Ticker-Statistiken für alle Handelspaare ab
        tickers = client.get_ticker()

        # Filtere nach Paaren, die mit dem gewünschten Quote-Asset enden (z.B. 'BTCUSDT')
        # und extrahiere Symbol und Volumen
        volume_pairs = []
        for ticker in tickers:
            if ticker['symbol'].endswith(quote_asset):
                try:
                    # 'quoteVolume' ist das Volumen ausgedrückt im Quote-Asset (z.B. in USDT)
                    # Das ist oft aussagekräftiger als das Volumen in der Base-Asset-Menge.
                    volume = float(ticker['quoteVolume'])
                    base_asset = ticker['symbol'].replace(quote_asset, '')
                    volume_pairs.append({
                        'base_asset': base_asset,
                        'symbol': ticker['symbol'],
                        'volume': volume
                    })
                except ValueError:
                    # Überspringe Ticker, wenn das Volumen keine gültige Zahl ist
                    # print(f"Warnung: Konnte Volumen für {ticker['symbol']} nicht verarbeiten.") # Optional für Debugging
                    pass
                except KeyError:
                     # Überspringe Ticker, wenn erwartete Schlüssel fehlen
                    # print(f"Warnung: Fehlende Daten für {ticker['symbol']}.") # Optional für Debugging
                    pass


        # Sortiere die Paare nach Volumen in absteigender Reihenfolge
        volume_pairs.sort(key=operator.itemgetter('volume'), reverse=True)

        # Wähle die Top N Paare aus
        top_pairs = volume_pairs[:top_n]

        return top_pairs

    except Exception as e:
        print(f"\nFehler bei der Abfrage der Binance API: {e}")
        print("Stelle sicher, dass du eine Internetverbindung hast und die 'python-binance' Bibliothek installiert ist (`pip install python-binance`).")
        return None

# --- Hauptteil des Skripts ---
if __name__ == "__main__":
    top_coins_list = get_major_coins_by_volume_on_binance(TOP_N, QUOTE_ASSET)

    if top_coins_list:
        print("-" * 70)
        print(f"Top {len(top_coins_list)} Coins auf Binance nach 24h Handelsvolumen gegen {QUOTE_ASSET}:")
        print("-" * 70)
        # Gib die Ergebnisse formatiert aus
        for i, coin_data in enumerate(top_coins_list):
            # Formatiere das Volumen für bessere Lesbarkeit
            formatted_volume = f"{coin_data['volume']:,.2f} {QUOTE_ASSET}"
            print(f"{i+1:>3}. {coin_data['base_asset']:<10} (Paar: {coin_data['symbol']:<15} | Volumen: {formatted_volume})")
        print("-" * 70)
        print(f"\nStand: Aktuelle Daten von der Binance API.")
        print(f"Hinweis: Dies spiegelt die Handelsaktivität auf Binance wider. Marktbedingungen ändern sich.")
