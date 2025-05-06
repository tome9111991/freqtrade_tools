# --- START OF FILE feature_engineering_neu.py --- (OHLCV Focus - V3 - Neue Features)

# --- START OF FILE mta_feature_engineering_base.py ---

# --- /home/tommi/freqtrade/user_data/strategies/includes/mta_feature_engineering_base.py ---
# Erweiterter MTAFeatureProvider - Fokus auf OHLCV-basierten Features
# Kompatibel mit Per-Pair-Training Skript v4.1 (ohne Übergabe von logger_adapter an add_all_features)
# MIT FIX FÜR PerformanceWarning & NEUEN FEATURES V3

import pandas as pd
import numpy as np
import re
import logging
from pandas import DataFrame
import traceback
from typing import Optional, Dict, Tuple, Any, List # Erweiterte Typ-Hinweise

# --- TA-Lib Handling (unverändert) ---
_talib_available = False
try:
    import talib
    _talib_available = True
    print("MTAFeatureProvider (OHLCV Focus V3): TA-Lib gefunden.")
except ImportError:
    print("MTAFeatureProvider (OHLCV Focus V3): Warnung: TA-Lib nicht gefunden. TA-Features werden fehlen.")
    # --- Verbesserter TA-Lib Dummy ---
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
    print("MTAFeatureProvider (OHLCV Focus V3): TA-Lib Dummy aktiviert.")

# Standard-Logger für dieses Modul
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Bei Bedarf aktivieren


# --- Hilfsfunktionen (unverändert) ---
def _safe_talib_call(talib_func_name, *args, **kwargs):
    """ Robuster Wrapper für TA-Lib Aufrufe über Funktionsnamen (String). """
    try:
        talib_func = getattr(talib, talib_func_name)
    except AttributeError:
        logger.error(f"TA-Lib Funktion '{talib_func_name}' nicht gefunden.")
        series_list = [s for s in args if isinstance(s, pd.Series)]; index = series_list[0].index if series_list else pd.Index([])
        expected_outputs = 1
        try:
             f = getattr(talib, talib_func_name) # Erneuter Versuch für Dummy/Real
             if hasattr(f, 'output_names') and isinstance(f.output_names, (list, tuple)):
                expected_outputs = len(f.output_names)
        except: pass
        if expected_outputs > 1: return [pd.Series(np.nan, index=index, name=f"{talib_func_name.lower()}_missing_func_out{i}") for i in range(expected_outputs)]
        else: return pd.Series(np.nan, index=index, name=f"{talib_func_name.lower()}_missing_func")

    if not _talib_available and isinstance(talib, TalibDummy):
        logger.debug(f"TA-Lib Dummy Call für '{talib_func_name}'.")
        try: return talib_func(*args, **kwargs)
        except Exception as e:
             logger.error(f"Fehler im TA-Lib Dummy für {talib_func_name}: {e}")
             series_list = [s for s in args if isinstance(s, pd.Series)]; index = series_list[0].index if series_list else pd.Index([])
             expected_outputs = 1
             if hasattr(talib_func, 'output_names') and isinstance(talib_func.output_names, (list, tuple)): expected_outputs = len(talib_func.output_names)
             if expected_outputs > 1: return [pd.Series(np.nan, index=index) for _ in range(expected_outputs)]
             else: return pd.Series(np.nan, index=index)

    series_list = [s for s in args if isinstance(s, pd.Series)]
    if not series_list:
         index = kwargs.get('index', pd.Index([]))
         expected_outputs = 1
         if hasattr(talib_func, 'output_names') and isinstance(talib_func.output_names, (list, tuple)): expected_outputs = len(talib_func.output_names)
         if expected_outputs > 1: return [pd.Series(np.nan, index=index) for _ in range(expected_outputs)]
         else: return pd.Series(np.nan, index=index)

    original_index = series_list[0].index; output_length = len(original_index)
    min_series_len = min(len(s) for s in series_list) if series_list else 0
    if min_series_len == 0:
         logger.warning(f"Leere Input-Series für TA-Lib Funktion '{talib_func_name}'.")
         expected_outputs = 1
         if hasattr(talib_func, 'output_names') and isinstance(talib_func.output_names, (list, tuple)): expected_outputs = len(talib_func.output_names)
         if expected_outputs > 1: return [pd.Series(np.nan, index=original_index) for _ in range(expected_outputs)]
         else: return pd.Series(np.nan, index=original_index)

    try:
        np_args = []
        min_required_len = 0
        for kw, val in kwargs.items():
            if 'timeperiod' in kw and isinstance(val, int): min_required_len = max(min_required_len, val)
        for s in args:
            if isinstance(s, pd.Series):
                filled_s = s.astype(np.float64, errors='ignore').copy()
                if filled_s.isnull().any(): filled_s = filled_s.ffill()
                filled_s.replace([np.inf, -np.inf], np.nan, inplace=True)
                if filled_s.isnull().any(): filled_s = filled_s.fillna(0)
                np_args.append(filled_s.values)
            else: np_args.append(s)

        valid_data_lengths = [len(arr) for arr in np_args if isinstance(arr, np.ndarray)]
        if valid_data_lengths and min(valid_data_lengths) < min_required_len:
             logger.warning(f"Nicht genug valide Daten für TA-Lib '{talib_func_name}' mit timeperiod={min_required_len}. Habe {min(valid_data_lengths)}. Gebe NaNs zurück.")
             expected_outputs = 1
             if hasattr(talib_func, 'output_names') and isinstance(talib_func.output_names, (list, tuple)): expected_outputs = len(talib_func.output_names)
             if expected_outputs > 1: return [pd.Series(np.nan, index=original_index) for _ in range(expected_outputs)]
             else: return pd.Series(np.nan, index=original_index)

        result = talib_func(*np_args, **kwargs)
        output_names = getattr(talib_func, 'output_names', None)
        if isinstance(result, tuple): # Multi-Output
            processed_results = []
            if output_names is None or len(output_names) != len(result): # Fallback
                 output_names = [f'out{i}' for i in range(len(result))]
            for i, res_part in enumerate(result):
                current_out_name = output_names[i] if i < len(output_names) else f'out{i}'
                name = f"{talib_func_name.lower()}_{current_out_name}"
                if not isinstance(res_part, np.ndarray):
                    logger.warning(f"Unerwarteter Typ in TA-Lib '{talib_func_name}' Output {i}: {type(res_part)}. Erstelle leere Series.")
                    res_series = pd.Series(np.nan, index=original_index, name=name)
                elif len(res_part) != output_length:
                     logger.warning(f"Längen-Mismatch bei TA-Lib '{talib_func_name}' Output {i}. Erwartet {output_length}, bekommen {len(res_part)}.")
                     res_series = pd.Series(np.nan, index=original_index, name=name)
                else:
                     res_series = pd.Series(res_part, index=original_index, name=name)
                processed_results.append(res_series)
            return processed_results
        elif isinstance(result, np.ndarray): # Single Output
             if output_names is None or not isinstance(output_names, (list, tuple)) or not output_names: name = f"{talib_func_name.lower()}_real"
             else: name = f"{talib_func_name.lower()}_{output_names[0]}"
             if len(result) != output_length:
                  logger.warning(f"Längen-Mismatch bei TA-Lib '{talib_func_name}' Output. Erwartet {output_length}, bekommen {len(result)}.")
                  return pd.Series(np.nan, index=original_index, name=name)
             else:
                  return pd.Series(result, index=original_index, name=name)
        else:
            raise TypeError(f"Unerwarteter Typ von TA-Lib {talib_func_name}: {type(result)}")

    except ValueError as ve: logger.warning(f"ValueError in _safe_talib_call ({talib_func_name}): {ve}. Liefert NaNs.")
    except Exception as e: logger.error(f"Fehler in _safe_talib_call ({talib_func_name}): {e}."); logger.error(traceback.format_exc())

    expected_outputs = 1
    if hasattr(talib_func, 'output_names') and isinstance(talib_func.output_names, (list, tuple)): expected_outputs = len(talib_func.output_names)
    func_base_name = talib_func_name.lower()
    out_names_fb = getattr(talib_func, 'output_names', [f'out{i}' for i in range(expected_outputs)]) if expected_outputs > 1 else ['real']
    if expected_outputs > 1:
        return [pd.Series(np.nan, index=original_index, name=f"{func_base_name}_{out_names_fb[i]}_error") for i in range(expected_outputs)]
    else:
        return pd.Series(np.nan, index=original_index, name=f"{func_base_name}_{out_names_fb[0]}_error")

def parse_timeframe_to_minutes(tf_string):
    if not tf_string: return 0
    match = re.match(r'(\d+)([mhdw])', tf_string)
    if match:
        value, unit = int(match.group(1)), match.group(2)
        if unit == 'm': return value
        if unit == 'h': return value * 60
        if unit == 'd': return value * 60 * 24
        if unit == 'w': return value * 60 * 24 * 7
    raise ValueError(f"Ungültiger Timeframe-String: {tf_string}")

# --- Feature Provider Klasse (OHLCV Fokus - V3) ---
class MTAFeatureProvider:
    """ Erweiterter MTAFeatureProvider mit Fokus auf OHLCV-basierten Features V3"""
    def __init__(self, base_tf: str = '5m', higher_tf: Optional[str] = '1h'):
        self.base_tf = base_tf
        self.higher_tf = higher_tf
        try:
            self.base_tf_minutes = parse_timeframe_to_minutes(base_tf)
            self.higher_tf_minutes = parse_timeframe_to_minutes(higher_tf) if higher_tf else 0
        except ValueError as e:
            logger.critical(f"MTAFeatureProvider Init Error: {e}"); raise

        logger.info(f"MTAFeatureProvider (OHLCV Focus V3) initialized (Base: {base_tf}, Higher: {higher_tf or 'None'})")


    def _calculate_base_ta_features(self, df: DataFrame) -> DataFrame:
        """ Berechnet erweiterte OHLCV-basierte TA-Features V3. """
        logger.debug(f"Starte _calculate_base_ta_features (OHLCV Focus V3)...")

        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_ohlcv if col not in df.columns]
        if missing_cols:
            logger.error(f"Fehlende Basis-Spalten im DataFrame: {missing_cols}")
            raise KeyError(f"Fehlende Basis-Spalten: {missing_cols}")

        for col in required_ohlcv:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Spalte '{col}' ist nicht numerisch ({df[col].dtype}). Versuche Konvertierung.")
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                     logger.warning(f"NaNs in '{col}' nach Konvertierung. Fülle mit ffill().fillna(0).")
                     df[col] = df[col].ffill().fillna(0)

        # --- Basis Indikatoren (RSI, BBands, MAs) ---
        logger.debug("Berechne Basis Indikatoren...")
        df['feature_rsi_14'] = _safe_talib_call('RSI', df['close'], timeperiod=14)
        df['feature_rsi_9'] = _safe_talib_call('RSI', df['close'], timeperiod=9)
        df['feature_rsi_26'] = _safe_talib_call('RSI', df['close'], timeperiod=26)

        bb_upper, bb_middle, bb_lower = _safe_talib_call('BBANDS', df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['feature_bb_upperband'] = bb_upper
        df['feature_bb_middleband'] = bb_middle
        df['feature_bb_lowerband'] = bb_lower

        df['feature_ma_9'] = df['close'].rolling(window=9, min_periods=5).mean()
        df['feature_ma_20'] = df['close'].rolling(window=20, min_periods=15).mean()
        df['feature_ma_50'] = df['close'].rolling(window=50, min_periods=40).mean()
        # NEU: Längere MAs
        df['feature_ma_100'] = df['close'].rolling(window=100, min_periods=80).mean()
        df['feature_ma_200'] = df['close'].rolling(window=200, min_periods=150).mean()


        # --- Erweiterte Volumen Features (OHLCV basiert) ---
        logger.debug("Berechne OHLCV-basierte Volumen-Features...")
        vol_rolling_mean_20 = df['volume'].rolling(window=20, min_periods=15).mean().replace(0, np.nan)
        df['feature_volume_roc_20'] = (df['volume'] / vol_rolling_mean_20) - 1
        vol_rolling_mean_50 = df['volume'].rolling(window=50, min_periods=40).mean().replace(0, np.nan)
        df['feature_volume_roc_50'] = (df['volume'] / vol_rolling_mean_50) - 1

        df['feature_obv'] = _safe_talib_call('OBV', df['close'], df['volume'])
        # NEU: Accumulation/Distribution Line
        df['feature_ad'] = _safe_talib_call('AD', df['high'], df['low'], df['close'], df['volume'])


        # --- Erweiterte Volatilitäts-Features ---
        logger.debug("Berechne erweiterte Volatilitäts-Features...")
        close_safe = df['close'].replace(0, np.nan)
        atr_14 = _safe_talib_call('ATR', df['high'], df['low'], df['close'], timeperiod=14)
        df['feature_atr_14_norm'] = (atr_14 / close_safe)
        atr_7 = _safe_talib_call('ATR', df['high'], df['low'], df['close'], timeperiod=7)
        df['feature_atr_7_norm'] = (atr_7 / close_safe)
        # NEU: Längere ATR & ATR Ratio
        atr_50 = _safe_talib_call('ATR', df['high'], df['low'], df['close'], timeperiod=50)
        df['feature_atr_50_norm'] = (atr_50 / close_safe)
        if isinstance(atr_7, pd.Series) and isinstance(atr_14, pd.Series):
             atr_14_safe = atr_14.replace(0, np.nan)
             df['feature_atr_7_14_ratio'] = (atr_7 / atr_14_safe).fillna(1.0)
        else: df['feature_atr_7_14_ratio'] = 1.0

        bb_middle_safe = df['feature_bb_middleband'].replace(0, np.nan)
        if 'feature_bb_upperband' in df.columns and 'feature_bb_lowerband' in df.columns and 'feature_bb_middleband' in df.columns and \
           isinstance(df['feature_bb_upperband'], pd.Series) and isinstance(df['feature_bb_lowerband'], pd.Series) and isinstance(df['feature_bb_middleband'], pd.Series):
            df['feature_bb_width_20_norm'] = (df['feature_bb_upperband'] - df['feature_bb_lowerband']) / bb_middle_safe
            # NEU: BB Width ROC
            bbw_roll_mean = df['feature_bb_width_20_norm'].rolling(window=10, min_periods=8).mean().replace(0, np.nan)
            df['feature_bb_width_roc_10'] = (df['feature_bb_width_20_norm'] / bbw_roll_mean) - 1
        else:
            logger.warning("BBands Spalten fehlen oder sind keine Series. Überspringe BB Width & ROC.")
            df['feature_bb_width_20_norm'] = 0.0
            df['feature_bb_width_roc_10'] = 0.0


        df['feature_stddev_20'] = df['close'].rolling(window=20, min_periods=15).std()
        df['feature_stddev_20_norm'] = df['feature_stddev_20'] / close_safe


        # --- Zusätzliche Momentum- und Oszillator-Features ---
        logger.debug("Berechne zusätzliche Momentum-/Oszillator-Features...")
        df['feature_cci_14'] = _safe_talib_call('CCI', df['high'], df['low'], df['close'], timeperiod=14)
        df['feature_cci_26'] = _safe_talib_call('CCI', df['high'], df['low'], df['close'], timeperiod=26)
        # NEU: Längerer CCI
        df['feature_cci_50'] = _safe_talib_call('CCI', df['high'], df['low'], df['close'], timeperiod=50)


        slowk, slowd = _safe_talib_call('STOCH', df['high'], df['low'], df['close'],
                                       fastk_period=14, slowk_period=3, slowk_matype=0,
                                       slowd_period=3, slowd_matype=0)
        df['feature_stoch_k'] = slowk
        df['feature_stoch_d'] = slowd
        if isinstance(slowk, pd.Series) and isinstance(slowd, pd.Series):
             df['feature_stoch_hist'] = slowk - slowd
             # NEU: Stochastik ROC (Rate of Change von %K)
             stoch_k_roll_mean = slowk.rolling(window=5, min_periods=3).mean().replace(0, np.nan)
             df['feature_stoch_k_roc_5'] = (slowk / stoch_k_roll_mean) - 1
        else:
             df['feature_stoch_hist'] = 0.0
             df['feature_stoch_k_roc_5'] = 0.0


        df['feature_roc_3'] = _safe_talib_call('ROC', df['close'], timeperiod=3)
        df['feature_roc_9'] = _safe_talib_call('ROC', df['close'], timeperiod=9)

        macd, macdsignal, macdhist = _safe_talib_call('MACD', df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['feature_macd'] = macd
        df['feature_macdsignal'] = macdsignal
        df['feature_macdhist'] = macdhist

        df['feature_willr_14'] = _safe_talib_call('WILLR', df['high'], df['low'], df['close'], timeperiod=14)


        # --- Trend- und Preispositions-Features ---
        logger.debug("Berechne Trend-/Preispositions-Features...")
        df['feature_adx_14'] = _safe_talib_call('ADX', df['high'], df['low'], df['close'], timeperiod=14)
        # NEU: Längerer ADX
        df['feature_adx_50'] = _safe_talib_call('ADX', df['high'], df['low'], df['close'], timeperiod=50)

        df['feature_di_plus_14'] = _safe_talib_call('PLUS_DI', df['high'], df['low'], df['close'], timeperiod=14)
        df['feature_di_minus_14'] = _safe_talib_call('MINUS_DI', df['high'], df['low'], df['close'], timeperiod=14)
        if isinstance(df['feature_di_plus_14'], pd.Series) and isinstance(df['feature_di_minus_14'], pd.Series):
             df['feature_di_diff_14'] = df['feature_di_plus_14'] - df['feature_di_minus_14']
        else: df['feature_di_diff_14'] = 0.0

        # Preis vs MAs (Distanz & binär) - jetzt auch mit 100, 200
        for p in [9, 20, 50, 100, 200]:
            ma_col = f'feature_ma_{p}'
            if ma_col in df.columns and pd.api.types.is_numeric_dtype(df[ma_col]):
                ma_safe = df[ma_col].replace(0, np.nan)
                df[f'feature_price_vs_ma{p}_dist'] = (df['close'] / ma_safe) - 1
                df[f'feature_price_above_ma{p}'] = (df['close'] > df[ma_col]).astype(int)
            else:
                logger.warning(f"MA-Spalte {ma_col} fehlt oder ist nicht numerisch. Überspringe abhängige Features.")
                df[f'feature_price_vs_ma{p}_dist'] = 0.0
                df[f'feature_price_above_ma{p}'] = 0

        # MA Cross (wie vorher)
        if 'feature_ma_9' in df.columns and 'feature_ma_20' in df.columns and \
           pd.api.types.is_numeric_dtype(df['feature_ma_9']) and \
           pd.api.types.is_numeric_dtype(df['feature_ma_20']):
            df['feature_ma9_above_ma20'] = (df['feature_ma_9'] > df['feature_ma_20']).astype(int)
        else: df['feature_ma9_above_ma20'] = 0

        # NEU: MA Alignment
        if all(f'feature_ma_{p}' in df.columns and pd.api.types.is_numeric_dtype(df[f'feature_ma_{p}']) for p in [9, 20, 50]):
            df['feature_ma_bullish_alignment'] = ((df['feature_ma_9'] > df['feature_ma_20']) & (df['feature_ma_20'] > df['feature_ma_50'])).astype(int)
            df['feature_ma_bearish_alignment'] = ((df['feature_ma_9'] < df['feature_ma_20']) & (df['feature_ma_20'] < df['feature_ma_50'])).astype(int)
        else:
             logger.warning("Nicht alle MAs (9, 20, 50) für Alignment vorhanden/numerisch.")
             df['feature_ma_bullish_alignment'] = 0
             df['feature_ma_bearish_alignment'] = 0


        # Relative Preispositionen (wie vorher)
        high_low_diff = (df['high'] - df['low']).replace(0, np.nan)
        df['feature_candle_pos'] = ((df['close'] - df['low']) / high_low_diff).fillna(0.5)
        rolling_low_50 = df['low'].rolling(window=50, min_periods=40).min()
        rolling_high_50 = df['high'].rolling(window=50, min_periods=40).max()
        rolling_range_50 = (rolling_high_50 - rolling_low_50).replace(0, np.nan)
        df['feature_pos_in_range50'] = ((df['close'] - rolling_low_50) / rolling_range_50).fillna(0.5)

        # --- Candlestick Patterns ---
        if _talib_available:
             logger.debug("Berechne Candlestick Patterns...")
             df['feature_cdl_doji'] = _safe_talib_call('CDLDOJI', df['open'], df['high'], df['low'], df['close']) / 100
             df['feature_cdl_hammer'] = _safe_talib_call('CDLHAMMER', df['open'], df['high'], df['low'], df['close']) / 100
             df['feature_cdl_engulfing'] = _safe_talib_call('CDLENGULFING', df['open'], df['high'], df['low'], df['close']) / 100
             df['feature_cdl_invhammer'] = _safe_talib_call('CDLINVERTEDHAMMER', df['open'], df['high'], df['low'], df['close']) / 100
             # NEU: Zusätzliche Muster
             df['feature_cdl_shootingstar'] = _safe_talib_call('CDLSHOOTINGSTAR', df['open'], df['high'], df['low'], df['close']) / 100
             df['feature_cdl_hangingman'] = _safe_talib_call('CDLHANGINGMAN', df['open'], df['high'], df['low'], df['close']) / 100
        else:
            logger.warning("TA-Lib nicht verfügbar, überspringe Candlestick Patterns.")
            df['feature_cdl_doji'] = 0; df['feature_cdl_hammer'] = 0; df['feature_cdl_engulfing'] = 0
            df['feature_cdl_invhammer'] = 0; df['feature_cdl_shootingstar'] = 0; df['feature_cdl_hangingman'] = 0

        # --- Rate of Change von Indikatoren (wie vorher) ---
        logger.debug("Berechne Rate of Change von Indikatoren...")
        for ind_col in ['feature_rsi_14', 'feature_cci_14', 'feature_macdhist']:
             if ind_col in df.columns and pd.api.types.is_numeric_dtype(df[ind_col]):
                 df[f'{ind_col}_diff1'] = df[ind_col].diff()
             else:
                 logger.warning(f"Spalte {ind_col} für Diff fehlt oder nicht numerisch. Überspringe.")
                 df[f'{ind_col}_diff1'] = 0.0

        # --- Zeitliche Features (wie vorher) ---
        logger.debug("Berechne zeitliche Features...")
        if isinstance(df.index, pd.DatetimeIndex):
             df['feature_hour_of_day'] = df.index.hour
             df['feature_day_of_week'] = df.index.dayofweek
        else:
             logger.warning("Index ist kein DatetimeIndex, zeitliche Features werden übersprungen.")
             df['feature_hour_of_day'] = 0; df['feature_day_of_week'] = 0

        # --- Lagged Features (wie vorher) ---
        logger.debug("Berechne Lagged Features...")
        df['feature_close_lag1'] = df['close'].shift(1)
        for lag_col in ['feature_rsi_14', 'feature_cci_14', 'feature_adx_14', 'feature_di_diff_14']:
             if lag_col in df.columns:
                 if isinstance(df[lag_col], pd.Series): df[f'{lag_col}_lag1'] = df[lag_col].shift(1)
                 else: logger.warning(f"Spalte {lag_col} für Lag ist kein Series-Objekt."); df[f'{lag_col}_lag1'] = 0.0
             else: logger.warning(f"Spalte {lag_col} für Lag fehlt."); df[f'{lag_col}_lag1'] = 0.0

        logger.debug(f"_calculate_base_ta_features abgeschlossen. {len([c for c in df.columns if c.startswith('feature_')])} Features aktuell.")
        return df.copy() # Wichtig für PerformanceWarning

    def add_all_features(self, base_df: DataFrame, htf_df: Optional[DataFrame]) -> DataFrame:
        """
        Hauptmethode: Berechnet Basis-Features, HTF-Features (falls vorhanden),
        resampelt HTF, mergt und berechnet abgeleitete/Interaktions-Features.
        Fokus auf OHLCV-Daten V3. Fix für PerformanceWarning mit pd.concat.
        """
        logger.info(f"Starte Feature-Berechnung (OHLCV Focus V3) (Base: {self.base_tf}, HTF: {self.higher_tf or 'None'})")

        if not isinstance(base_df.index, pd.DatetimeIndex): raise ValueError("base_df muss DatetimeIndex haben!")
        if htf_df is not None and not isinstance(htf_df.index, pd.DatetimeIndex): raise ValueError("htf_df muss DatetimeIndex haben!")

        base_df_copy = base_df.copy()
        htf_df_copy = htf_df.copy() if htf_df is not None else None

        logger.info(f"Berechne Features für Base TF ({self.base_tf})...")
        try:
            df_base_features = self._calculate_base_ta_features(base_df_copy)
        except KeyError as e:
             logger.critical(f"Fehler bei Basis-Feature-Berechnung (KeyError): {e}. Prüfe OHLCV-Spaltennamen.")
             raise
        except Exception as e:
             logger.critical(f"Unerwarteter Fehler bei Basis-Feature-Berechnung: {e}"); logger.error(traceback.format_exc())
             raise

        df_htf_features_resampled = pd.DataFrame(index=df_base_features.index)

        # --- HTF Verarbeitung ---
        if htf_df_copy is not None and self.higher_tf and self.higher_tf_minutes > self.base_tf_minutes:
            logger.info(f"Berechne Features für HTF ({self.higher_tf})...")
            try:
                df_htf_calculated = self._calculate_base_ta_features(htf_df_copy)
            except KeyError as e:
                 logger.critical(f"Fehler bei HTF-Feature-Berechnung (KeyError): {e}. Prüfe OHLCV-Spaltennamen im HTF.")
                 raise
            except Exception as e:
                 logger.critical(f"Unerwarteter Fehler bei HTF-Feature-Berechnung: {e}"); logger.error(traceback.format_exc())
                 raise

            htf_cols_to_resample = [col for col in df_htf_calculated.columns if col.startswith('feature_')]

            if htf_cols_to_resample:
                 df_htf_to_resample = df_htf_calculated[htf_cols_to_resample].copy()
                 if not df_htf_to_resample.index.is_unique:
                     logger.warning("HTF Index nicht eindeutig. Duplikate werden entfernt (letztes Vorkommen behalten).")
                     df_htf_to_resample = df_htf_to_resample[~df_htf_to_resample.index.duplicated(keep='last')]

                 logger.debug(f"Resample HTF Features von {self.higher_tf} zu {self.base_tf}...")
                 try:
                     df_htf_resampled_raw = df_htf_to_resample.resample(f'{self.base_tf_minutes}min').ffill()
                     rename_dict = {col: f"feature_{self.higher_tf}_{col.replace('feature_', '')}_inf" for col in df_htf_resampled_raw.columns}
                     df_htf_features_resampled = df_htf_resampled_raw.rename(columns=rename_dict)
                     logger.debug(f"{len(df_htf_features_resampled.columns)} HTF Features resampled und umbenannt.")
                 except Exception as e_resample:
                     logger.error(f"Fehler beim Resampling der HTF Features: {e_resample}"); logger.error(traceback.format_exc())
            else: logger.warning("Keine 'feature_' Spalten im HTF DataFrame gefunden zum Resamplen.")
        elif htf_df_copy is None: logger.info("Kein HTF DataFrame übergeben, überspringe HTF Feature Berechnung.")
        else: logger.warning(f"HTF ({self.higher_tf}) ist nicht höher als Base TF ({self.base_tf}) oder ungültig. Überspringe HTF.")

        # --- Merge Basis und HTF Features ---
        logger.debug("Merge Basis und HTF Features...")
        df_merged = df_base_features.join(df_htf_features_resampled, how='left')

        # Fülle NaNs, die durch das Join entstanden sein könnten
        feature_cols_merged = [col for col in df_merged.columns if col.startswith('feature_')]
        feature_cols_exist_in_merged = [col for col in feature_cols_merged if col in df_merged.columns]
        df_merged[feature_cols_exist_in_merged] = df_merged[feature_cols_exist_in_merged].ffill()
        logger.debug("NaNs nach Merge mit ffill gefüllt.")

        # --- Berechne abgeleitete/Interaktions-Features ---
        logger.debug("Berechne Interaktions-Features separat...")
        interaction_features_list = []

        # RSI Ratio
        rsi_base_col = 'feature_rsi_14'; rsi_htf_col = f'feature_{self.higher_tf}_rsi_14_inf'; rsi_ratio_name = 'feature_rsi_htf_ratio'
        if rsi_base_col in df_merged.columns and rsi_htf_col in df_merged.columns:
             safe_rsi_htf = df_merged[rsi_htf_col].replace(0, np.nan)
             interaction_features_list.append((df_merged[rsi_base_col] / safe_rsi_htf).fillna(1.0).rename(rsi_ratio_name))
             logger.debug(f"{rsi_ratio_name} berechnet.")
        else: interaction_features_list.append(pd.Series(1.0, index=df_merged.index, name=rsi_ratio_name))

        # CCI Differenz
        cci_base_col = 'feature_cci_14'; cci_htf_col = f'feature_{self.higher_tf}_cci_14_inf'; cci_diff_name = 'feature_cci_diff_base_htf'
        if cci_base_col in df_merged.columns and cci_htf_col in df_merged.columns:
            interaction_features_list.append((df_merged[cci_base_col] - df_merged[cci_htf_col]).rename(cci_diff_name))
            logger.debug(f"{cci_diff_name} berechnet.")
        else: interaction_features_list.append(pd.Series(0.0, index=df_merged.index, name=cci_diff_name))

        # ADX Ratio
        adx_base_col = 'feature_adx_14'; adx_htf_col = f'feature_{self.higher_tf}_adx_14_inf'; adx_ratio_name = 'feature_adx_htf_ratio'
        if adx_base_col in df_merged.columns and adx_htf_col in df_merged.columns:
             safe_adx_htf = df_merged[adx_htf_col].replace(0, np.nan)
             interaction_features_list.append((df_merged[adx_base_col] / safe_adx_htf).fillna(1.0).rename(adx_ratio_name))
             logger.debug(f"{adx_ratio_name} berechnet.")
        else: interaction_features_list.append(pd.Series(1.0, index=df_merged.index, name=adx_ratio_name))

        # Base Volatilität * HTF Trendstärke
        atr_norm_base_col = 'feature_atr_14_norm'
        adx_htf_col_eff = f'feature_{self.higher_tf}_adx_14_inf' if f'feature_{self.higher_tf}_adx_14_inf' in df_merged.columns else 'feature_adx_14'
        interaction_name = 'feature_interaction_volabase_trendhtf'
        if atr_norm_base_col in df_merged.columns and adx_htf_col_eff in df_merged.columns:
            if pd.api.types.is_numeric_dtype(df_merged[atr_norm_base_col]) and pd.api.types.is_numeric_dtype(df_merged[adx_htf_col_eff]):
                interaction_features_list.append((df_merged[atr_norm_base_col] * df_merged[adx_htf_col_eff]).rename(interaction_name))
                logger.debug(f"{interaction_name} berechnet.")
            else:
                logger.warning(f"Interaktions-Spalten {atr_norm_base_col} oder {adx_htf_col_eff} nicht numerisch.")
                interaction_features_list.append(pd.Series(0.0, index=df_merged.index, name=interaction_name))
        else:
            logger.warning(f"Interaktions-Spalten {atr_norm_base_col} oder {adx_htf_col_eff} fehlen.")
            interaction_features_list.append(pd.Series(0.0, index=df_merged.index, name=interaction_name))

        # NEU: Strong Uptrend Momentum Signal
        strong_uptrend_name = 'feature_strong_uptrend_momentum'
        base_above_ma50 = 'feature_price_above_ma50'
        htf_above_ma50 = f'feature_{self.higher_tf}_price_above_ma50_inf'
        base_rsi_strong = 'feature_rsi_14' # Verwenden wir den normalen RSI hier
        if all(c in df_merged.columns for c in [base_above_ma50, htf_above_ma50, base_rsi_strong]):
            cond1 = df_merged[base_above_ma50] == 1
            cond2 = df_merged[htf_above_ma50] == 1
            cond3 = df_merged[base_rsi_strong] > 55 # RSI > 55 als Beispiel für Momentum
            interaction_features_list.append((cond1 & cond2 & cond3).astype(int).rename(strong_uptrend_name))
            logger.debug(f"{strong_uptrend_name} berechnet.")
        else:
            logger.warning(f"Spalten für {strong_uptrend_name} fehlen.")
            interaction_features_list.append(pd.Series(0, index=df_merged.index, name=strong_uptrend_name))


        # Füge alle neuen Interaktionsfeatures auf einmal hinzu
        if interaction_features_list:
             logger.debug(f"Füge {len(interaction_features_list)} Interaktions-Features mit pd.concat hinzu...")
             new_cols_df = pd.concat(interaction_features_list, axis=1)
             cols_to_add = [col for col in new_cols_df.columns if col not in df_merged.columns]
             if cols_to_add:
                 df_merged = pd.concat([df_merged, new_cols_df[cols_to_add]], axis=1)
             else: logger.warning("Keine neuen Interaktionsfeatures zum Hinzufügen gefunden.")
        else: logger.debug("Keine Interaktions-Features berechnet.")


        # --- Finale Bereinigung ---
        logger.debug("Finale Bereinigung (Inf -> NaN, verbleibende NaN -> 0)...")
        final_feature_cols = [col for col in df_merged.columns if col.startswith('feature_')]
        for col in final_feature_cols:
            if col in df_merged.columns:
                if not pd.api.types.is_numeric_dtype(df_merged[col]):
                     logger.warning(f"Spalte '{col}' ist nicht numerisch ({df_merged[col].dtype}). Versuche Konvertierung.")
                     df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
            else: logger.warning(f"Spalte '{col}' während finaler Bereinigung nicht gefunden.")

        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_feature_cols_exist = [col for col in final_feature_cols if col in df_merged.columns]
        final_nan_count = df_merged[final_feature_cols_exist].isnull().sum().sum()
        if final_nan_count > 0:
            df_merged[final_feature_cols_exist] = df_merged[final_feature_cols_exist].ffill().fillna(0)
            logger.warning(f"{final_nan_count} verbleibende NaNs/Inf in Features wurden mit ffill().fillna(0) behandelt.")

        # --- Abschluss ---
        final_feature_count = len([col for col in df_merged.columns if col.startswith('feature_')])
        logger.info(f"Feature-Berechnung abgeschlossen. Anzahl 'feature_' Spalten generiert: {final_feature_count}")

        initial_cols = ['open', 'high', 'low', 'close', 'volume']
        # +++ START KORREKTUR: 'date' explizit hinzufügen, wenn vorhanden +++
        cols_to_keep = initial_cols + final_feature_cols_exist
        if 'date' in df_merged.columns:
             if 'date' not in cols_to_keep: # Nur hinzufügen, falls nicht schon drin
                 cols_to_keep.append('date')
                 logger.debug("Explizit 'date'-Spalte zur finalen Rückgabe hinzugefügt.")
        # +++ ENDE KORREKTUR +++

        cols_exist = [col for col in cols_to_keep if col in df_merged.columns]
        # Loggen, welche Spalten tatsächlich zurückgegeben werden
        logger.debug(f"Finale Spalten, die zurückgegeben werden: {cols_exist}")
        df_final = df_merged[cols_exist].copy()

        # Prüfe Index Typ vor Rückgabe (Sicherheitscheck)
        if not isinstance(df_final.index, pd.DatetimeIndex):
             logger.error(f"FEHLER: df_final hat keinen DatetimeIndex mehr vor Rückgabe aus add_all_features!")

        return df_final

# --- Ende der Klasse ---

# --- END OF FILE mta_feature_engineering_base.py ---

# --- END OF FILE feature_engineering_neu.py ---