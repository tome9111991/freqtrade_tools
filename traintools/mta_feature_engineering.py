# --- START OF FILE mta_feature_engineering.py (KORRIGIERT v2) ---

# --- /home/tommi/freqtrade/user_data/strategies/includes/mta_feature_engineering.py ---
# Version mit erweiterten Features für Entry UND Exit, OHNE Supertrend, MIT Breakouts & Exhaustion/Reversal Features
# KORRIGIERTE VERSION v2: Import für Optional hinzugefügt, .bfill() an kritischen Stellen entfernt/ersetzt.
import pandas as pd
import numpy as np
import re
import logging
from pandas import DataFrame
import traceback # Für detaillierte Fehlermeldungen
from typing import Optional # <<< HINZUGEFÜGT: Import für Optional

# --- TA-Lib Handling ---
_talib_available = False
try:
    import talib
    _talib_available = True
    print("MTAFeatureProvider: TA-Lib gefunden.")
except ImportError:
    print("MTAFeatureProvider: Warnung: TA-Lib nicht gefunden. TA-Features werden fehlen.")
    # --- Verbesserter TA-Lib Dummy (unverändert) ---
    class TalibDummy:
        def __getattr__(self, name):
            real_func = None
            try: real_func = getattr(talib, name)
            except AttributeError: pass
            expected_outputs = 1
            if real_func and hasattr(real_func, 'output_names') and isinstance(real_func.output_names, (list, tuple)):
                expected_outputs = len(real_func.output_names)
            def method(*args, **kwargs):
                series_list = [s for s in args if isinstance(s, pd.Series)]
                if not series_list: return np.nan if expected_outputs == 1 else tuple([np.nan] * expected_outputs)
                index = series_list[0].index
                if expected_outputs > 1:
                    output_names = getattr(real_func, 'output_names', [f'out{i}' for i in range(expected_outputs)])
                    return [pd.Series(np.nan, index=index, name=f"{name.lower()}_{out_name}") for out_name in output_names]
                else:
                    output_name = getattr(real_func, 'output_names', ['real'])[0] if real_func else 'real'
                    return pd.Series(np.nan, index=index, name=f"{name.lower()}_{output_name}")
            return method
    talib = TalibDummy()
    print("MTAFeatureProvider: TA-Lib Dummy aktiviert.")

# Logger
logger = logging.getLogger(__name__)

# --- Hilfsfunktionen ---
def _safe_talib_call(talib_func_name, *args, **kwargs):
    """ Robuster Wrapper für TA-Lib Aufrufe über Funktionsnamen (String). KORRIGIERT: .bfill() entfernt. """
    # (Code unverändert zur letzten Version)
    try:
        talib_func = getattr(talib, talib_func_name)
    except AttributeError:
        logger.error(f"TA-Lib Funktion '{talib_func_name}' nicht gefunden.")
        series_list = [s for s in args if isinstance(s, pd.Series)]; index = series_list[0].index if series_list else pd.Index([])
        return pd.Series(np.nan, index=index)
    if not _talib_available and isinstance(talib, TalibDummy):
        logger.debug(f"TA-Lib Dummy Call für '{talib_func_name}'.")
        try: return talib_func(*args, **kwargs)
        except Exception as e:
             logger.error(f"Fehler im TA-Lib Dummy für {talib_func_name}: {e}")
             series_list = [s for s in args if isinstance(s, pd.Series)]; index = series_list[0].index if series_list else pd.Index([])
             return pd.Series(np.nan, index=index)
    series_list = [s for s in args if isinstance(s, pd.Series)]
    if not series_list: return pd.Series(np.nan, index=kwargs.get('index', pd.Index([])))
    original_index = series_list[0].index; output_length = len(original_index)
    min_series_len = min(len(s) for s in series_list) if series_list else 0
    timeperiod = 1
    timeperiod_keys = ['timeperiod', 'fastperiod', 'slowk_period', 'timeperiod1', 'period', 'slowperiod', 'signalperiod', 'fastk_period', 'nbdevup', 'nbdevdn']
    periods_found = [kwargs[key] for key in timeperiod_keys if key in kwargs and isinstance(kwargs[key], (int, float))]
    int_args = [a for a in args if isinstance(a, int) and a > 1]
    all_periods = periods_found + int_args
    if all_periods: timeperiod = max(timeperiod, int(max(all_periods)))
    if min_series_len < timeperiod and min_series_len > 0:
         logger.warning(f"Nicht genug Daten ({min_series_len}) für '{talib_func_name}' (Periode ~{timeperiod}).")
         output_info = getattr(talib_func, 'output_names', ['real'])
         if isinstance(output_info, (list, tuple)) and len(output_info) > 1: return [pd.Series(np.nan, index=original_index) for _ in output_info]
         else: return pd.Series(np.nan, index=original_index)
    elif min_series_len == 0:
        output_info = getattr(talib_func, 'output_names', ['real'])
        if isinstance(output_info, (list, tuple)) and len(output_info) > 1: return [pd.Series(np.nan, index=original_index) for _ in output_info]
        else: return pd.Series(np.nan, index=original_index)
    try:
        np_args = []
        for s in args:
            if isinstance(s, pd.Series):
                filled_s = s.astype(np.float64, errors='ignore').copy()
                if filled_s.isnull().any(): filled_s = filled_s.ffill()
                filled_s.replace([np.inf, -np.inf], np.nan, inplace=True)
                if filled_s.isnull().any(): filled_s = filled_s.fillna(0); logger.debug(f"Fülle verbleibende Anfangs-NaNs mit 0 für '{talib_func_name}'.")
                np_args.append(filled_s.values)
            else: np_args.append(s)
        result = talib_func(*np_args, **kwargs)
        output_names = getattr(talib_func, 'output_names', None)
        if isinstance(result, tuple):
            processed_results = []
            if output_names is None: output_names = [f'out{i}' for i in range(len(result))]
            for i, res_part in enumerate(result):
                name = f"{talib_func_name.lower()}_{output_names[i]}"
                if not isinstance(res_part, np.ndarray): continue
                if len(res_part) < output_length: res_part = np.concatenate((np.full(output_length - len(res_part), np.nan), res_part))
                elif len(res_part) > output_length: res_part = res_part[-output_length:]
                processed_results.append(pd.Series(res_part, index=original_index, name=name))
            return processed_results
        elif isinstance(result, np.ndarray):
             if output_names is None or not output_names: name = talib_func_name.lower()
             else: name = f"{talib_func_name.lower()}_{output_names[0]}"
             if len(result) < output_length: result = np.concatenate((np.full(output_length - len(result), np.nan), result))
             elif len(result) > output_length: result = result[-output_length:]
             return pd.Series(result, index=original_index, name=name)
        else: raise TypeError(f"Unerwarteter Typ von TA-Lib {talib_func_name}: {type(result)}")
    except ValueError as ve: logger.warning(f"ValueError in _safe_talib_call ({talib_func_name}): {ve}. Liefert NaNs.")
    except Exception as e: logger.error(f"Fehler in _safe_talib_call ({talib_func_name}): {e}."); logger.error(traceback.format_exc())
    try: output_info = getattr(getattr(talib, talib_func_name), 'output_names', ['real'])
    except AttributeError: output_info = ['real']
    if isinstance(output_info, (list, tuple)) and len(output_info) > 1: return [pd.Series(np.nan, index=original_index) for _ in output_info]
    else: return pd.Series(np.nan, index=original_index)


def parse_timeframe_to_minutes(tf_string):
    # (Code unverändert)
    if not tf_string: return 0
    match = re.match(r'(\d+)([mhdw])', tf_string)
    if match:
        value, unit = int(match.group(1)), match.group(2)
        if unit == 'm': return value
        if unit == 'h': return value * 60
        if unit == 'd': return value * 60 * 24
        if unit == 'w': return value * 60 * 24 * 7
    raise ValueError(f"Ungültiger Timeframe-String: {tf_string}")

# --- Feature Provider Klasse ---
class MTAFeatureProvider:
    """ KORRIGIERTE Version v2 des Feature Providers (Import Optional). """
    def __init__(self, base_tf: str = '15m', higher_tf: str = '1h'):
        self.base_tf = base_tf
        self.higher_tf = higher_tf
        try: self.base_tf_minutes = parse_timeframe_to_minutes(base_tf)
        except ValueError as e: logger.critical(f"MTAFeatureProvider Init Error: {e}"); raise
        logger.info(f"MTAFeatureProvider initialized (Base: {base_tf}, Higher: {higher_tf}) - KORRIGIERTE Version v2")

    def _calculate_base_ta_features(self, df: DataFrame) -> DataFrame:
        """ Berechnet Basis TA-Features. KORRIGIERT: Heikin Ashi Init. """
        # --- Momentum Indikatoren (unverändert) ---
        # ... (Code für RSI, Stoch, StochRSI, MACD, ROC, MOM, CCI) ...
        logger.debug("Berechne Momentum Indikatoren...")
        df['feature_rsi_14'] = _safe_talib_call('RSI', df['close'], timeperiod=14)
        df['feature_rsi_7'] = _safe_talib_call('RSI', df['close'], timeperiod=7)
        stoch_k, stoch_d = _safe_talib_call('STOCH', df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
        df['feature_stoch_k'] = stoch_k; df['feature_stoch_d'] = stoch_d
        rsi_temp = _safe_talib_call('RSI', df['close'], timeperiod=14)
        if isinstance(rsi_temp, pd.Series) and not rsi_temp.isnull().all():
            stochrsi_k, stochrsi_d = _safe_talib_call('STOCHRSI', rsi_temp, timeperiod=14, fastk_period=5, fastd_period=3)
            df['feature_stochrsi_k'] = stochrsi_k; df['feature_stochrsi_d'] = stochrsi_d
        else: df['feature_stochrsi_k'], df['feature_stochrsi_d'] = np.nan, np.nan
        df['feature_stochrsi_k'] = df['feature_stochrsi_k'].fillna(50); df['feature_stochrsi_d'] = df['feature_stochrsi_d'].fillna(50)
        macd, macdsignal, macdhist = _safe_talib_call('MACD', df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['feature_macd_macd'] = macd; df['feature_macd_macdsignal'] = macdsignal; df['feature_macd_macdhist'] = macdhist
        df['feature_macd_hist_rising'] = (df['feature_macd_macdhist'] > df['feature_macd_macdhist'].shift(1)).fillna(0).astype(int)
        df['feature_roc_3'] = _safe_talib_call('ROC', df['close'], timeperiod=3).fillna(0)
        df['feature_roc_10'] = _safe_talib_call('ROC', df['close'], timeperiod=10).fillna(0)
        df['feature_mom_14'] = _safe_talib_call('MOM', df['close'], timeperiod=14).fillna(0)
        df['feature_cci_20'] = _safe_talib_call('CCI', df['high'], df['low'], df['close'], timeperiod=20).fillna(0)

        # --- Trendstärke (unverändert) ---
        # ... (Code für ADX, ADXR, DI) ...
        logger.debug("Berechne Trendstärke Indikatoren...")
        df['feature_adx_14'] = _safe_talib_call('ADX', df['high'], df['low'], df['close'], timeperiod=14)
        df['feature_adxr_14'] = _safe_talib_call('ADXR', df['high'], df['low'], df['close'], timeperiod=14)
        df['feature_minus_di_14'] = _safe_talib_call('MINUS_DI', df['high'], df['low'], df['close'], timeperiod=14)
        df['feature_plus_di_14'] = _safe_talib_call('PLUS_DI', df['high'], df['low'], df['close'], timeperiod=14)
        df['feature_di_trend'] = (df['feature_plus_di_14'] > df['feature_minus_di_14']).astype(int)

        # --- Volatilität & Bollinger Bänder (unverändert) ---
        # ... (Code für BBands, ATR, Stddev) ...
        logger.debug("Berechne Volatilität & Bollinger Bänder...")
        bb_upper, bb_middle, bb_lower = _safe_talib_call('BBANDS', df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['feature_bb_upperband'] = bb_upper; df['feature_bb_middleband'] = bb_middle; df['feature_bb_lowerband'] = bb_lower
        if all(f in df and not df[f].isnull().all() for f in ['feature_bb_upperband', 'feature_bb_lowerband', 'feature_bb_middleband']):
            safe_bb_middle = df['feature_bb_middleband'].replace(0, np.nan)
            safe_bb_divisor = (df['feature_bb_upperband'] - df['feature_bb_lowerband']).replace(0, np.nan)
            df['feature_bb_width'] = ((safe_bb_divisor / safe_bb_middle) * 100).fillna(0)
            df['feature_bb_pct_b'] = ((df['close'] - df['feature_bb_lowerband']) / safe_bb_divisor).fillna(0.5)
            df['feature_price_above_upper_bb'] = (df['close'] > df['feature_bb_upperband']).astype(int)
            df['feature_price_below_lower_bb'] = (df['close'] < df['feature_bb_lowerband']).astype(int)
            df['feature_price_below_middle_bb'] = (df['close'] < df['feature_bb_middleband']).astype(int)
        else:
            df['feature_bb_width'], df['feature_bb_pct_b'] = 0.0, 0.5
            df['feature_price_above_upper_bb'] = 0; df['feature_price_below_lower_bb'] = 0; df['feature_price_below_middle_bb'] = 0
        df['feature_atr_14'] = _safe_talib_call('ATR', df['high'], df['low'], df['close'], timeperiod=14)
        safe_close_atr = df['close'].replace(0, np.nan)
        df['feature_atr_ratio'] = ((df['feature_atr_14'] / safe_close_atr) * 100).fillna(0)
        df['feature_stddev_20'] = df['close'].rolling(window=20, min_periods=15).std().fillna(0)
        safe_close_std = df['close'].replace(0, np.nan)
        df['feature_stddev_ratio'] = (df['feature_stddev_20'] / safe_close_std * 100).fillna(0)

        # --- Preis-Aktion / Candlesticks (unverändert) ---
        # ... (Code für CDL*) ...
        logger.debug("Berechne Candlestick Pattern Features...")
        df['feature_cdl_engulfing'] = (_safe_talib_call('CDLENGULFING', df['open'], df['high'], df['low'], df['close']) / 100).fillna(0)
        df['feature_cdl_hammer'] = (_safe_talib_call('CDLHAMMER', df['open'], df['high'], df['low'], df['close']) / 100).fillna(0)
        df['feature_cdl_doji'] = (_safe_talib_call('CDLDOJI', df['open'], df['high'], df['low'], df['close']) / 100).fillna(0)
        df['feature_cdl_shootingstar'] = (_safe_talib_call('CDLSHOOTINGSTAR', df['open'], df['high'], df['low'], df['close']) / 100).fillna(0)
        df['feature_cdl_hangingman'] = (_safe_talib_call('CDLHANGINGMAN', df['open'], df['high'], df['low'], df['close']) / 100).fillna(0)
        df['feature_cdl_darkcloudcover'] = (_safe_talib_call('CDLDARKCLOUDCOVER', df['open'], df['high'], df['low'], df['close']) / 100).fillna(0)
        cdl_cols = [col for col in df.columns if col.startswith('feature_cdl_')]
        df[cdl_cols] = df[cdl_cols].fillna(0)

        # --- Relative Preisposition & einfache Preisänderung (unverändert) ---
        # ... (Code für Rolling High/Low, Distanzen, pct_change) ...
        logger.debug("Berechne relative Preisposition Features...")
        rolling_high_20 = df['high'].rolling(window=20, min_periods=15).max(); rolling_low_20 = df['low'].rolling(window=20, min_periods=15).min()
        df['feature_high_20'] = rolling_high_20; df['feature_low_20'] = rolling_low_20
        safe_close_pos20 = df['close'].replace(0, np.nan)
        df['feature_dist_from_high_20_pct'] = ((df['feature_high_20'] - df['close']) / safe_close_pos20 * 100).fillna(0)
        df['feature_dist_from_low_20_pct'] = ((df['close'] - df['feature_low_20']) / safe_close_pos20 * 100).fillna(0)
        range_20 = (rolling_high_20 - rolling_low_20).replace(0, np.nan); df['feature_price_pct_in_range_20'] = ((df['close'] - rolling_low_20) / range_20).fillna(0.5)
        rolling_high_50 = df['high'].rolling(window=50, min_periods=40).max(); rolling_low_50 = df['low'].rolling(window=50, min_periods=40).min()
        df['feature_high_50'] = rolling_high_50; df['feature_low_50'] = rolling_low_50
        safe_close_pos50 = df['close'].replace(0, np.nan)
        df['feature_dist_from_high_50_pct'] = ((df['feature_high_50'] - df['close']) / safe_close_pos50 * 100).fillna(0)
        df['feature_dist_from_low_50_pct'] = ((df['close'] - df['feature_low_50']) / safe_close_pos50 * 100).fillna(0)
        range_50 = (rolling_high_50 - rolling_low_50).replace(0, np.nan); df['feature_price_pct_in_range_50'] = ((df['close'] - rolling_low_50) / range_50).fillna(0.5)
        df['feature_price_change_pct_1'] = (df['close'].pct_change(periods=1) * 100).fillna(0)
        df['feature_candle_body_pct'] = (abs(df['close'] - df['open']) / safe_close_pos20 * 100).fillna(0)
        df['feature_candle_wick_pct'] = (((df['high'] - df['low']) - abs(df['close'] - df['open'])) / safe_close_pos20 * 100).clip(lower=0).fillna(0)


        # --- Volumen (unverändert) ---
        # ... (Code für Volumen MA, Ratio, Change) ...
        logger.debug("Berechne Volumen Features...")
        df['feature_volume_ma_20'] = df['volume'].rolling(window=20, min_periods=15).mean().fillna(0)
        safe_volume_ma = df['feature_volume_ma_20'].replace(0, np.nan); df['feature_volume_ratio'] = (df['volume'] / safe_volume_ma).fillna(1.0)
        df['feature_volume_change_pct_1'] = (df['volume'].pct_change(periods=1) * 100).fillna(0)


        # --- Konsekutive Kerzen (unverändert) ---
        # ... (Code für consecutive up/down candles) ...
        logger.debug("Berechne konsekutive Kerzen...")
        df['price_up'] = (df['close'] > df['close'].shift(1)).astype(int); group_up = df['price_up'].ne(df['price_up'].shift()).cumsum()
        df['feature_consecutive_up_candles'] = df.groupby(group_up).cumcount() + 1; df.loc[df['price_up'] == 0, 'feature_consecutive_up_candles'] = 0
        df['price_down'] = (df['close'] < df['close'].shift(1)).astype(int); group_down = df['price_down'].ne(df['price_down'].shift()).cumsum()
        df['feature_consecutive_down_candles'] = df.groupby(group_down).cumcount() + 1; df.loc[df['price_down'] == 0, 'feature_consecutive_down_candles'] = 0
        df.drop(columns=['price_up', 'price_down'], inplace=True, errors='ignore')


        # --- Breakout Signale (unverändert) ---
        # ... (Code für breakout high/low) ...
        logger.debug("Berechne einfache Breakout Features...")
        df['feature_breakout_high_20_simple'] = (df['close'] > df['feature_high_20'].shift(1)).astype(int).fillna(0)
        df['feature_breakout_low_20_simple'] = (df['close'] < df['feature_low_20'].shift(1)).astype(int).fillna(0)


        # --- Moving Averages (unverändert) ---
        # ... (Code für MA 10, 20, 50, 200) ...
        logger.debug("Berechne Moving Averages...")
        df['feature_ma_10'] = df['close'].rolling(window=10, min_periods=8).mean()
        df['feature_ma_20'] = df['close'].rolling(window=20, min_periods=15).mean()
        df['feature_ma_50'] = df['close'].rolling(window=50, min_periods=40).mean()
        df['feature_ma_200'] = df['close'].rolling(window=200, min_periods=150).mean()


        # --- MA Kreuzungen (unverändert) ---
        # ... (Code für price_crossed_below/above_ma) ...
        logger.debug("Berechne MA Kreuzungen...")
        for ma_len in [10, 20, 50]:
            ma_col = f'feature_ma_{ma_len}'
            if ma_col in df.columns and not df[ma_col].isnull().all():
                df[f'feature_price_crossed_below_ma_{ma_len}'] = ((df['close'].shift(1) >= df[ma_col].shift(1)) & (df['close'] < df[ma_col])).astype(int).fillna(0)
                df[f'feature_price_crossed_above_ma_{ma_len}'] = ((df['close'].shift(1) <= df[ma_col].shift(1)) & (df['close'] > df[ma_col])).astype(int).fillna(0)
            else: df[f'feature_price_crossed_below_ma_{ma_len}'] = 0; df[f'feature_price_crossed_above_ma_{ma_len}'] = 0


        # --- Veränderungsraten (Diffs) von Indikatoren (unverändert) ---
        # ... (Code für _diff Features) ...
        logger.debug("Berechne Veränderungsraten (Diffs) für Indikatoren...")
        df['feature_rsi_14_diff'] = df['feature_rsi_14'].diff().fillna(0)
        df['feature_bb_pct_b_diff'] = df['feature_bb_pct_b'].diff().fillna(0)
        df['feature_macd_hist_diff'] = df['feature_macd_macdhist'].diff().fillna(0)
        df['feature_adx_14_diff'] = df['feature_adx_14'].diff().fillna(0)
        df['feature_volume_ratio_diff'] = df['feature_volume_ratio'].diff().fillna(0)

        # --- Heikin Ashi Features (KORRIGIERT: ha_open Initialisierung) ---
        logger.debug("Berechne Heikin Ashi Features (KORRIGIERT)...")
        try:
            ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            # === KORREKTUR: Sicherere ha_open Berechnung ohne bfill() ===
            ha_open_raw = (df['open'].shift(1) + df['close'].shift(1)) / 2
            ha_open = ha_open_raw.ffill() # Nur vorwärts füllen
            ha_open.fillna(df['open'], inplace=True) # Fülle verbleibende Anfangs-NaNs mit aktuellem Open
            # === Ende Korrektur ===

            df['feature_ha_close'] = ha_close; df['feature_ha_open'] = ha_open
            df['feature_ha_high'] = df[['high', 'feature_ha_open', 'feature_ha_close']].max(axis=1)
            df['feature_ha_low'] = df[['low', 'feature_ha_open', 'feature_ha_close']].min(axis=1)
            df['feature_ha_trend'] = (df['feature_ha_close'] > df['feature_ha_open']).astype(int)
            df['feature_ha_trend_changed'] = (df['feature_ha_trend'] != df['feature_ha_trend'].shift(1)).astype(int).fillna(0)
            safe_ha_close = df['feature_ha_close'].replace(0, np.nan)
            df['feature_ha_body_size_pct'] = (abs(df['feature_ha_close'] - df['feature_ha_open']) / safe_ha_close * 100).fillna(0)
            df['feature_ha_small_body'] = (df['feature_ha_body_size_pct'] < 0.1).astype(int)
            df['feature_ha_wick_upper_pct'] = ((df['feature_ha_high'] - df[['feature_ha_open', 'feature_ha_close']].max(axis=1)) / safe_ha_close * 100).clip(lower=0).fillna(0)
            df['feature_ha_wick_lower_pct'] = ((df[['feature_ha_open', 'feature_ha_close']].min(axis=1) - df['feature_ha_low']) / safe_ha_close * 100).clip(lower=0).fillna(0)
            df['feature_ha_long_upper_wick'] = (df['feature_ha_wick_upper_pct'] > df['feature_ha_body_size_pct'] * 1.5).astype(int)
            df['feature_ha_long_lower_wick'] = (df['feature_ha_wick_lower_pct'] > df['feature_ha_body_size_pct'] * 1.5).astype(int)
        except Exception as e_ha:
            logger.warning(f"Fehler bei Berechnung Heikin Ashi Features: {e_ha}. Setze HA Features auf NaN/0.")
            ha_cols = ['feature_ha_close', 'feature_ha_open', 'feature_ha_high', 'feature_ha_low','feature_ha_trend', 'feature_ha_trend_changed', 'feature_ha_body_size_pct','feature_ha_small_body', 'feature_ha_wick_upper_pct', 'feature_ha_wick_lower_pct','feature_ha_long_upper_wick', 'feature_ha_long_lower_wick']
            for col in ha_cols: df[col] = np.nan if any(x in col for x in ['close', 'open', 'high', 'low']) else 0

        # --- Finale Füllung ---
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        # Wichtig: ffill() zuerst, dann fillna(0) für verbleibende NaNs am Anfang
        df[feature_cols] = df[feature_cols].ffill().fillna(0)

        logger.debug(f"Basis TA-Features Berechnung abgeschlossen. {len(feature_cols)} Features.")
        return df

    def add_all_features(self, base_df: DataFrame, htf_df: Optional[DataFrame]) -> DataFrame:
        """
        Berechnet *alle* Features (Basis, HTF, Resampled, MTA, Abgeleitete).
        Akzeptiert optionales htf_df. KORRIGIERTE Version v2.
        """
        logger.info(f"Starte Feature-Berechnung (Base: {self.base_tf}, HTF: {self.higher_tf}) - KORRIGIERTE Version v2")
        # ... (Assertions für DatetimeIndex) ...
        if not isinstance(base_df.index, pd.DatetimeIndex): raise ValueError("add_all_features: base_df muss DatetimeIndex haben!")
        if htf_df is not None and not isinstance(htf_df.index, pd.DatetimeIndex): raise ValueError("add_all_features: htf_df muss DatetimeIndex haben, wenn übergeben!")

        df_base = base_df.copy()
        df_htf = htf_df.copy() if htf_df is not None else None
        # logger.debug("Kopien von base_df und htf_df erstellt.")

        htf_tf_name = self.higher_tf
        target_training_suffix = f'_{htf_tf_name}_inf'
        df_htf_resampled_suffixed = pd.DataFrame(index=df_base.index) # Leerer DF als Default

        # --- 1. & 2. Berechne HTF Features und Resample (NUR WENN HTF VORHANDEN) ---
        rename_dict_suffix = {} # Initialisiere hier für späteren Zugriff
        if df_htf is not None:
            # ... (Code für HTF Berechnung, Resampling, Renaming - unverändert zur letzten Version) ...
            logger.debug("Berechne HTF Indikatoren...")
            df_htf_calculated = self._calculate_base_ta_features(df_htf)
            if 'close' in df_htf_calculated.columns:
                 if 'feature_ma_50' not in df_htf_calculated.columns: df_htf_calculated['feature_ma_50'] = df_htf_calculated['close'].rolling(window=50, min_periods=40).mean()
                 if 'feature_ma_200' not in df_htf_calculated.columns: df_htf_calculated['feature_ma_200'] = df_htf_calculated['close'].rolling(window=200, min_periods=150).mean()
                 df_htf_calculated.rename(columns={'feature_ma_50': 'htf_ma_50', 'feature_ma_200': 'htf_ma_200'}, inplace=True)
            logger.debug("Resample und benenne HTF Features um...")
            htf_cols_to_resample = [col for col in df_htf_calculated.columns if col.startswith(('feature_', 'htf_'))]
            if not htf_cols_to_resample: logger.warning("Keine HTF Features zum Resamplen gefunden.")
            else:
                 df_htf_to_resample = df_htf_calculated[htf_cols_to_resample].copy()
                 if not df_htf_to_resample.index.is_unique: logger.warning("HTF Index nicht eindeutig."); df_htf_to_resample = df_htf_to_resample[~df_htf_to_resample.index.duplicated(keep='last')]
                 try:
                     df_htf_resampled = df_htf_to_resample.resample(f'{self.base_tf_minutes}min').ffill()
                     for col in df_htf_resampled.columns:
                         new_name = ""
                         if col.startswith('htf_'): new_name = f"feature_{htf_tf_name}_{col[4:]}{target_training_suffix}"
                         elif col.startswith('feature_'):
                             feature_name = col.replace('feature_', '').replace('macd_macdhist', 'macd_hist').replace('macd_macdsignal', 'macd_signal').replace('macd_macd', 'macd').replace('bbands_upperband', 'bb_upperband').replace('bbands_middleband', 'bb_middleband').replace('bbands_lowerband', 'bb_lowerband')
                             new_name = f"feature_{htf_tf_name}_{feature_name}{target_training_suffix}"
                         if new_name: rename_dict_suffix[col] = new_name
                     df_htf_resampled_suffixed = df_htf_resampled.rename(columns=rename_dict_suffix)
                     logger.debug(f"{len(rename_dict_suffix)} HTF Spalten umbenannt.")
                 except Exception as e_resample: logger.error(f"Fehler Resampling/Renaming: {e_resample}"); logger.error(traceback.format_exc())
        else: logger.warning("Kein HTF DataFrame übergeben.")

        # --- 3. Berechne Basis TF Features (unverändert) ---
        logger.debug("Berechne Basis TF Indikatoren...")
        df_base = self._calculate_base_ta_features(df_base)
        if isinstance(df_base.index, pd.DatetimeIndex):
             df_base['feature_hour'] = df_base.index.hour; df_base['feature_dayofweek'] = df_base.index.dayofweek
             df_base['feature_minuteofday'] = df_base.index.hour * 60 + df_base.index.minute
        for ma_len in [10, 20, 50, 200]:
             if f'feature_ma_{ma_len}' not in df_base.columns: df_base[f'feature_ma_{ma_len}'] = df_base['close'].rolling(window=ma_len, min_periods=int(ma_len*0.8)).mean()

        # --- 4. Mergen der Features (unverändert) ---
        logger.debug("Merge Basis TF und HTF Features...")
        df_final = df_base.join(df_htf_resampled_suffixed, how='left')
        if df_htf is not None: # Nur füllen, wenn HTF Daten da waren
            htf_feature_cols_final = list(rename_dict_suffix.values())
            if htf_feature_cols_final:
                existing_htf_cols = [col for col in htf_feature_cols_final if col in df_final.columns]
                if existing_htf_cols: df_final[existing_htf_cols] = df_final[existing_htf_cols].ffill()

        df_final = df_final.copy() # Erste Kopie
        # logger.debug("Erste Kopie nach Merge erstellt.")

        # --- 5. Berechne Abgeleitete Features (Logik unverändert, aber abhängig von existierenden HTF Spalten) ---
        logger.debug("Berechne abgeleitete Features...")
        # ... (Code für Distanzen, Ratios, Slopes - bleibt logisch gleich, wird aber nur ausgeführt, wenn HTF-Spalten da sind) ...
        for ma_len in [50, 200]:
             htf_ma_col = f'feature_{htf_tf_name}_ma_{ma_len}{target_training_suffix}'
             if htf_ma_col in df_final.columns:
                 if not df_final[htf_ma_col].isnull().all():
                     safe_htf_ma = df_final[htf_ma_col].replace(0, np.nan)
                     df_final[f'feature_dist_to_htf_ma_{ma_len}_pct'] = ((df_final['close'] / safe_htf_ma) - 1) * 100
                     df_final[f'feature_price_above_htf_ma_{ma_len}'] = (df_final['close'] > df_final[htf_ma_col]).astype(int)
                     df_final[f'feature_price_crossed_below_htf_ma_{ma_len}'] = ((df_final['close'].shift(1) >= df_final[htf_ma_col].shift(1)) & (df_final['close'] < df_final[htf_ma_col])).astype(int).fillna(0)
                     df_final[f'feature_price_crossed_above_htf_ma_{ma_len}'] = ((df_final['close'].shift(1) <= df_final[htf_ma_col].shift(1)) & (df_final['close'] > df_final[htf_ma_col])).astype(int).fillna(0)
                 else: df_final[f'feature_dist_to_htf_ma_{ma_len}_pct'] = 0.0; df_final[f'feature_price_above_htf_ma_{ma_len}'] = 0; df_final[f'feature_price_crossed_below_htf_ma_{ma_len}'] = 0; df_final[f'feature_price_crossed_above_htf_ma_{ma_len}'] = 0
             else: df_final[f'feature_dist_to_htf_ma_{ma_len}_pct'] = 0.0; df_final[f'feature_price_above_htf_ma_{ma_len}'] = 0; df_final[f'feature_price_crossed_below_htf_ma_{ma_len}'] = 0; df_final[f'feature_price_crossed_above_htf_ma_{ma_len}'] = 0
        rsi_base_col = 'feature_rsi_14'; rsi_htf_col = f'feature_{htf_tf_name}_rsi_14{target_training_suffix}'
        if rsi_base_col in df_final.columns and rsi_htf_col in df_final.columns and not df_final[rsi_base_col].isnull().all() and not df_final[rsi_htf_col].isnull().all():
            safe_rsi_htf = df_final[rsi_htf_col].replace(0, np.nan).replace(100, 99.99); safe_rsi_base = df_final[rsi_base_col].replace(0, 0.01); df_final['feature_rsi_base_htf_ratio'] = (safe_rsi_base / safe_rsi_htf).fillna(1.0).clip(0.1, 10)
        else: df_final['feature_rsi_base_htf_ratio'] = 1.0
        atr_ratio_base_col = 'feature_atr_ratio'; atr_ratio_htf_col = f'feature_{htf_tf_name}_atr_ratio{target_training_suffix}'
        if atr_ratio_base_col in df_final.columns and atr_ratio_htf_col in df_final.columns and not df_final[atr_ratio_base_col].isnull().all() and not df_final[atr_ratio_htf_col].isnull().all():
            safe_atr_ratio_htf = df_final[atr_ratio_htf_col].replace(0, np.nan); df_final['feature_volatility_base_htf_ratio'] = (df_final[atr_ratio_base_col] / safe_atr_ratio_htf).fillna(1.0).clip(0.1, 10)
        else: df_final['feature_volatility_base_htf_ratio'] = 1.0
        adx_base_col = 'feature_adx_14'; adx_htf_col = f'feature_{htf_tf_name}_adx_14{target_training_suffix}'
        if adx_base_col in df_final.columns and adx_htf_col in df_final.columns and not df_final[adx_base_col].isnull().all() and not df_final[adx_htf_col].isnull().all():
             df_final['feature_adx_base_htf_diff'] = (df_final[adx_base_col] - df_final[adx_htf_col]).fillna(0)
        else: df_final['feature_adx_base_htf_diff'] = 0.0
        # MA Slopes
        for ma_len in [10, 20, 50, 200]:
            ma_col = f'feature_ma_{ma_len}'; slope_col = f'{ma_col}_slope'; slope_norm_col = f'{ma_col}_slope_norm'
            if ma_col in df_final.columns and not df_final[ma_col].isnull().all():
                 df_final[slope_col] = df_final[ma_col].diff().fillna(0)
                 if 'feature_atr_14' in df_final.columns and not df_final['feature_atr_14'].isnull().all(): safe_atr = df_final['feature_atr_14'].replace(0, np.nan); df_final[slope_norm_col] = (df_final[slope_col] / safe_atr).fillna(0)
                 else: df_final[slope_norm_col] = 0.0
            else: df_final[slope_col] = 0.0; df_final[slope_norm_col] = 0.0
        ma_short_slope_col = 'feature_ma_10_slope'; ma_long_slope_col = 'feature_ma_50_slope'
        if ma_short_slope_col in df_final.columns and ma_long_slope_col in df_final.columns:
            safe_long_slope = df_final[ma_long_slope_col].replace(0, np.nan); df_final['feature_ma_slope_ratio_10_50'] = (df_final[ma_short_slope_col] / safe_long_slope).fillna(1.0).clip(-10, 10)
            df_final['feature_ma_slope_diff_10_50'] = (df_final[ma_short_slope_col] - df_final[ma_long_slope_col]).fillna(0.0)
        else: df_final['feature_ma_slope_ratio_10_50'] = 1.0; df_final['feature_ma_slope_diff_10_50'] = 0.0


        df_final = df_final.copy() # Zweite Kopie
        # logger.debug("Zweite Kopie nach Slope-Berechnungen erstellt.")

        # --- 6. Berechne alte MTA Interaktions-Features (Logik unverändert, aber prüft auf Spaltenexistenz) ---
        logger.debug("Berechne ursprüngliche MTA Interaktions-Features (feature_mta_*)...")
        # ... (Code für MTA Features, prüft auf Spaltenexistenz vor Berechnung) ...
        rsi_base = 'feature_rsi_14'; rsi_htf = f'feature_{htf_tf_name}_rsi_14{target_training_suffix}'; stoch_base = 'feature_stoch_k'; adx_base = 'feature_adx_14'; adx_htf = f'feature_{htf_tf_name}_adx_14{target_training_suffix}'
        ma50_htf = f'feature_{htf_tf_name}_ma_50{target_training_suffix}'; ma200_htf = f'feature_{htf_tf_name}_ma_200{target_training_suffix}'; bb_pct_b_base = 'feature_bb_pct_b'; macd_rising_base = 'feature_macd_hist_rising'
        mta_feature_cols = ['feature_mta_rsi_dip_in_strong_1h_up', 'feature_mta_stoch_bot_mom_1h_ok', 'feature_mta_bblow_in_1h_trend', 'feature_mta_5m_pause_in_1h_uptrend']
        required_mta_cols = ['close', rsi_base, stoch_base, bb_pct_b_base, macd_rising_base, adx_base]; required_mta_htf_cols = [ma50_htf, ma200_htf, rsi_htf, adx_htf]
        all_required_present = all(col in df_final.columns for col in required_mta_cols + required_mta_htf_cols)
        if not all_required_present:
            missing = [col for col in required_mta_cols + required_mta_htf_cols if col not in df_final.columns]; logger.warning(f"Fehlende Spalten für MTA-Berechnung: {missing}. Setze MTA Features auf 0.")
            for col in mta_feature_cols: df_final[col] = 0
        else:
            nan_check_ok = not df_final[required_mta_cols + required_mta_htf_cols].isnull().any().any()
            if not nan_check_ok: logger.warning("NaNs in benötigten MTA Spalten gefunden. Setze MTA Features auf 0."); [df_final.__setitem__(col, 0) for col in mta_feature_cols]
            else:
                try:
                    cond_1h_strong_uptrend = (df_final['close'] > df_final[ma50_htf]) & (df_final[ma50_htf] > df_final[ma200_htf]) & (df_final[adx_htf] > 25); cond_5m_rsi_low = (df_final[rsi_base] < 40); df_final['feature_mta_rsi_dip_in_strong_1h_up'] = (cond_5m_rsi_low & cond_1h_strong_uptrend).astype(int)
                    cond_5m_stoch_very_low = (df_final[stoch_base] < 20); cond_5m_macd_rising = df_final[macd_rising_base].astype(bool); cond_1h_not_oversold = (df_final[rsi_htf] > 35); df_final['feature_mta_stoch_bot_mom_1h_ok'] = (cond_5m_stoch_very_low & cond_5m_macd_rising & cond_1h_not_oversold).astype(int)
                    cond_5m_near_bblow = (df_final[bb_pct_b_base] < 0.15); cond_1h_trending = (df_final[adx_htf] > 20); df_final['feature_mta_bblow_in_1h_trend'] = (cond_5m_near_bblow & cond_1h_trending).astype(int)
                    cond_5m_adx_weak = (df_final[adx_base] < 20); cond_1h_ma_trend_up = (df_final[ma50_htf] > df_final[ma200_htf]); df_final['feature_mta_5m_pause_in_1h_uptrend'] = (cond_5m_adx_weak & cond_1h_ma_trend_up).astype(int)
                except Exception as e_mta: logger.error(f"Fehler bei MTA-Features: {e_mta}."); logger.error(traceback.format_exc()); [df_final.__setitem__(col, 0) for col in mta_feature_cols]
        for col in mta_feature_cols: # Sicherstellen, dass Spalten existieren
            if col not in df_final.columns: df_final[col] = 0


        # --- 7. Finale Bereinigung (unverändert) ---
        logger.debug("Finale Bereinigung (Inf -> NaN)...")
        df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_feature_cols = [col for col in df_final.columns if col.startswith('feature_')]
        final_nan_count = df_final[final_feature_cols].isnull().sum().sum()
        if final_nan_count > 0:
            logger.warning(f"Fülle {final_nan_count} verbleibende NaNs (nach Inf-Entfernung) mit ffill/0.")
            df_final[final_feature_cols] = df_final[final_feature_cols].ffill().fillna(0)

        # --- Abschluss (unverändert) ---
        logger.info(f"Feature-Berechnung abgeschlossen. Finale Spaltenanzahl: {len(df_final.columns)}")
        final_feature_count = len([col for col in df_final.columns if col.startswith('feature_')])
        logger.info(f"Anzahl 'feature_' Spalten generiert: {final_feature_count}")

        # --- Letzte Kopie (unverändert) ---
        # logger.debug("Erstelle finale Kopie des DataFrames.")
        df_final = df_final.copy()

        return df_final

# --- Ende der Klasse ---

# --- END OF FILE mta_feature_engineering.py (KORRIGIERT v2) ---