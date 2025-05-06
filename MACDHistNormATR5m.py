# --- Freqtrade Strategie: Normalisiertes MACD Histogramm & Signallinie (5m) ---
# Dateiname: z.B. macd_hist_norm_atr_5m_strategy.py

import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import numpy as np # Für die Behandlung von Division durch Null
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame

class MACDHistNormATR5m(IStrategy):
    """
    MACD Strategie basierend auf ATR-NORMALISIERTEM Histogramm-Schwellenwerten
    und der Position der Signallinie (5m Chart).
    Versucht, die Strategie robuster gegenüber unterschiedlicher Volatilität zu machen.

    Kauft, wenn norm_macdhist <= buy_norm_threshold UND macdsignal < 0.
    Verkauft, wenn norm_macdhist >= sell_norm_threshold UND macdsignal > 0.

    Hinweis: Die NORMALISIERTEN Schwellenwerte und die ATR-Periode sind jetzt
           wichtige Optimierungsparameter! Paar-spezifische Optimierung
           via Hyperopt wird weiterhin für beste Ergebnisse empfohlen.
    """

    # --- Strategy Standardparameter ---
    minimal_roi = {
        "0": 0.05
    }
    stoploss = -0.05
    trailing_stop = False
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True
    timeframe = '5m'
    startup_candle_count: int = 35 # Ausreichend für MACD und ATR(14)

    # --- Hyperopt Parameter ---
    # MACD Perioden
    macd_fast = IntParameter(8, 16, default=12, space="buy", optimize=False)
    macd_slow = IntParameter(20, 35, default=26, space="buy", optimize=False)
    macd_signal = IntParameter(7, 12, default=9, space="buy", optimize=False)

    # ATR Periode für Normalisierung
    atr_period = IntParameter(10, 20, default=14, space="buy", optimize=False) # Hinzugefügt

    # NORMALISIERTE MACD Histogramm Schwellenwerte
    # Diese Werte repräsentieren jetzt ein Vielfaches des ATR.
    # Die Bereiche müssen angepasst werden (kleinere Zahlen als zuvor!)
    buy_norm_hist_threshold = DecimalParameter(-0.8, -0.2, default=-0.4, decimals=3, space="buy")      # -0.6, -0.28
    sell_norm_hist_threshold = DecimalParameter(0.2, 0.8, default=0.3, decimals=3, space="sell")        # 0.2, 0.5


       
# === KORRIGIERTE Plot Konfiguration ===
    plot_config = {
        'main_plot': {
            # Optional: Füge hier Dinge hinzu, die auf dem Preis-Chart liegen sollen
        },
        'subplots': {
            # Eigener Subplot für MACD Signal Linie
            "MACDSignal": {
                'macdsignal': {'color': 'turquoise'}  # KEIN 'type' mehr nötig
            },
            # Eigener Subplot für das NORMALISIERTE Histogramm
            "NormMACDHist": {
                'norm_macdhist': {'color': 'purple', 'type': 'bar', 'plotly': {'opacity': 0.7}} # 'type': 'bar' ist korrekt
            },
            # Optional: Eigener Subplot für ATR
            "ATR": {
                'atr': {'color': 'orange'}           # KEIN 'type' mehr nötig
            }
        }
    }
    # =======================================

    


    # --- Indikatoren-Berechnung ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Fügt MACD, ATR und das normalisierte Histogramm zum DataFrame hinzu.
        """
        # Hole Parameterwerte
        macd_fast = self.macd_fast.value
        macd_slow = self.macd_slow.value
        macd_signal = self.macd_signal.value
        atr_period = self.atr_period.value

        # Berechne MACD
        macd = ta.MACD(dataframe,
                       fastperiod=macd_fast,
                       slowperiod=macd_slow,
                       signalperiod=macd_signal)
        dataframe['macdhist'] = macd['macdhist']
        dataframe['macdsignal'] = macd['macdsignal']

        # Berechne ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=atr_period)

        # Berechne normalisiertes Histogramm
        # WICHTIG: Behandlung von ATR = 0 (passiert bei komplett flachen Kerzen)
        # Ersetze ATR=0 durch einen sehr kleinen Wert, um Division durch Null zu vermeiden
        safe_atr = dataframe['atr'].replace(0, np.nan).ffill().fillna(1e-9) # Fülle NaNs und ersetze 0
        dataframe['norm_macdhist'] = dataframe['macdhist'] / safe_atr

        return dataframe

    # --- Kaufsignal-Logik ---
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Definiert die Bedingungen für ein Kaufsignal:
        Normalisiertes Histogramm <= Schwellenwert UND Signallinie < 0
        """
        # Hole den normalisierten Schwellenwert für den Kauf
        buy_threshold = self.buy_norm_hist_threshold.value

        dataframe.loc[
            (
                # Bedingung 1: Normalisiertes Histogramm <= Kauf-Schwellenwert
                (dataframe['norm_macdhist'] <= buy_threshold) &

                # Bedingung 2: Signallinie < 0
                (dataframe['macdsignal'] < 0)

                # Optional: Füge hier weitere Filter hinzu
                # & (dataframe['volume'] > 0)

            ),
            'buy'] = 1 # Setze 'buy' auf 1 (Signal)

        return dataframe

    # --- Verkaufssignal-Logik ---
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Definiert die Bedingungen für ein Verkaufssignal:
        Normalisiertes Histogramm >= Schwellenwert UND Signallinie > 0
        """
        # Hole den normalisierten Schwellenwert für den Verkauf
        sell_threshold = self.sell_norm_hist_threshold.value

        dataframe.loc[
            (
                # Bedingung 1: Normalisiertes Histogramm >= Verkauf-Schwellenwert
                (dataframe['norm_macdhist'] >= sell_threshold) &

                # Bedingung 2: Signallinie > 0
                (dataframe['macdsignal'] > 0)

                # Optional: Füge hier weitere Filter hinzu
                # & (dataframe['volume'] > 0)
            ),
            'sell'] = 1 # Setze 'sell' auf 1 (Signal)
        return dataframe

    # --- Konfiguration für Verkaufssignale ---
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False