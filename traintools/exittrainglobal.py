# --- START OF FILE exittrain.py (GLOBAL MODEL VERSION) ---
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna # Für HPO
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, f1_score, roc_auc_score
import joblib
import os
import glob
import re
import json
from datetime import datetime, timezone
import sys
import logging
import traceback

# --- Grundlegendes Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# *** Eindeutiger Logger-Name für Exit-Training ***
logger = logging.getLogger("exit_train_global_script")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ================================================================
# === START: BENUTZERKONFIGURATION (EXIT - GLOBAL) ===
# ================================================================
# --- Pfade ---
USER_CONFIG_USER_DATA_DIR = '/home/tommi/freqtrade/user_data' # <<< ANPASSEN
USER_CONFIG_DATA_DIR_SUFFIX = 'data/binance'
USER_CONFIG_MODELS_DIR_SUFFIX = 'models'
USER_CONFIG_PROVIDER_DIR_SUFFIX = 'strategies/includes'

# --- Zeitrahmen ---
USER_CONFIG_BASE_TIMEFRAME = '5m' # <<< ANPASSEN
USER_CONFIG_HIGHER_TIMEFRAME = '1h' # <<< ANPASSEN

# --- Trainingsdaten-Filter ---
USER_CONFIG_TIMERANGE = '20240201-20250410' # <<< ANPASSEN

# --- EXIT Zielvariable & Daten ---
USER_CONFIG_LOOKAHEAD_PERIODS = 3 # <<< ANPASSEN
USER_CONFIG_EXIT_TARGET_VOLA_MULT = 0.0 # <<< ANPASSEN
USER_CONFIG_EXIT_TARGET_THRESHOLD = 0.003 # <<< ANPASSEN (nur wenn Vola_Mult = 0)
USER_CONFIG_N_SPLITS = 5
USER_CONFIG_MIN_SAMPLES_COMBINED = 10000 # <<< ANPASSEN

# --- *** EXIT Penalty Konfiguration *** ---
USER_CONFIG_EXIT_PENALTY_ENABLE = False
USER_CONFIG_PENALTY_OVERSOLD_ENABLE = True
USER_CONFIG_PENALTY_RSI_LOW_THRESHOLD = 15.0
USER_CONFIG_PENALTY_DIST_LOW_50_PCT_LOW_THRESHOLD = 0.5
USER_CONFIG_PENALTY_HTF_STRONG_UPTREND_ENABLE = True
USER_CONFIG_PENALTY_HTF_RSI_HIGH_THRESHOLD = 75.0
USER_CONFIG_PENALTY_HTF_MA_TREND_CHECK = True

# --- HPO (Optuna) Konfiguration ---
USER_CONFIG_ENABLE_HPO = True
USER_CONFIG_HPO_TRIALS = 10

# --- Feste Modellparameter (Fallback) ---
USER_CONFIG_FIXED_PARAMS = {
    'learning_rate': 0.02, 'num_leaves': 20, 'max_depth': 7,
    'reg_alpha': 1e-7, 'reg_lambda': 1e-7, 'min_child_samples': 30,
    'subsample': 0.85, 'colsample_bytree': 0.7
}

# --- Training Konfiguration (EXIT) ---
# *** WICHTIG: Eindeutiger Prefix für Exit-Modelle! ***
USER_CONFIG_MODEL_PREFIX = 'mta_exit_model2_glob' # <<< ANPASSEN
USER_CONFIG_N_ESTIMATORS_FINAL = 3000
USER_CONFIG_EARLY_STOPPING_ROUNDS_FINAL = 50
# *** Ziel-Precision für Exit-Klasse (1 = Preis fällt) ***
USER_CONFIG_EXIT_MIN_PRECISION_TARGET = 0.50 # <<< ANPASSEN

# --- Test Subset ---
USER_CONFIG_TEST_PAIRS_SUBSET = None
# ================================================================
# === ENDE: BENUTZERKONFIGURATION (EXIT - GLOBAL) ===
# ================================================================

# --- Abgeleitete Pfade (unverändert) ---
user_data_dir = os.path.abspath(USER_CONFIG_USER_DATA_DIR)
data_directory = os.path.join(user_data_dir, USER_CONFIG_DATA_DIR_SUFFIX)
save_directory = os.path.join(user_data_dir, USER_CONFIG_MODELS_DIR_SUFFIX)
provider_base_dir = os.path.join(user_data_dir, os.path.dirname(USER_CONFIG_PROVIDER_DIR_SUFFIX)) if USER_CONFIG_PROVIDER_DIR_SUFFIX else user_data_dir
provider_module_path = USER_CONFIG_PROVIDER_DIR_SUFFIX.replace('/', '.') if USER_CONFIG_PROVIDER_DIR_SUFFIX else ''

# --- Provider importieren ---
MTAFeatureProvider = None
try:
    import_base_path = user_data_dir
    if import_base_path not in sys.path: sys.path.insert(0, import_base_path); logger.info(f"Added '{import_base_path}' to sys.path.")
    from strategies.includes.mta_feature_engineering import MTAFeatureProvider
    logger.info(f"MTAFeatureProvider successfully imported from '{provider_module_path or 'strategies.includes'}.mta_feature_engineering'.")
except Exception as e: logger.critical(f"FEHLER beim Import von MTAFeatureProvider: {e}"); logger.error(traceback.format_exc()); sys.exit(1)
if MTAFeatureProvider is None: logger.critical("Import von MTAFeatureProvider fehlgeschlagen."); sys.exit(1)

# --- Hilfsfunktionen ---
def parse_timerange(timerange_str):
    # (Code unverändert zu entrytrain)
    if not timerange_str: return None, None
    parts = timerange_str.split('-'); start_dt, end_dt = None, None; date_format = '%Y%m%d'
    if len(parts) != 2: raise ValueError(f"Ungültiges Timerange-Format: {timerange_str}. Erwartet 'YYYYMMDD-YYYYMMDD'.")
    start_str, end_str = parts
    if start_str:
        try: start_dt = datetime.strptime(start_str, date_format).replace(tzinfo=timezone.utc)
        except ValueError: raise ValueError(f"Ungültiges Startdatum: {start_str}.")
    if end_str:
        try: end_dt_naive = datetime.strptime(end_str, date_format); end_dt = end_dt_naive.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
        except ValueError: raise ValueError(f"Ungültiges Enddatum: {end_str}.")
    return start_dt, end_dt

def load_data_mta(data_dir_load, pairs, timeframes, start_dt=None, end_dt=None, timerange_str=""):
    # (Code unverändert zu entrytrain)
    data_dict = {tf: {} for tf in timeframes}; required_base_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    logger.info(f"Lade Daten für {len(pairs)} Paare und TFs {timeframes} aus '{data_dir_load}'...")
    if timerange_str: logger.info(f"Zeitfilter: {timerange_str} ({start_dt} bis {end_dt})")
    for tf in timeframes:
        logger.info(f"Lade Zeitrahmen: {tf}")
        loaded_pairs_tf = 0; skipped_filter_tf = 0; skipped_missing_cols = 0
        for pair_slash in pairs:
            pair_file_str = pair_slash.replace('/', '_'); search_pattern = os.path.join(data_dir_load, f'{pair_file_str}-{tf}.feather')
            file_list = glob.glob(search_pattern)
            if not file_list: continue
            filepath = file_list[0]
            try:
                df_pair = pd.read_feather(filepath)
                if not all(col in df_pair.columns for col in required_base_cols): logger.warning(f"Datei {filepath} fehlen Basisspalten."); skipped_missing_cols += 1; continue
                df_pair = df_pair[required_base_cols].copy(); df_pair['date'] = pd.to_datetime(df_pair['date'], utc=True); df_pair.set_index('date', inplace=True); df_pair.sort_index(inplace=True)
                if start_dt or end_dt:
                    original_len = len(df_pair); df_pair = df_pair.loc[start_dt:end_dt]
                    if df_pair.empty and original_len > 0: skipped_filter_tf += 1; continue
                    elif df_pair.empty: continue
                if not df_pair.empty: data_dict[tf][pair_slash] = df_pair; loaded_pairs_tf += 1
            except Exception as e: logger.error(f"Fehler Laden/Verarbeiten {filepath}: {e}")
        logger.info(f"{loaded_pairs_tf} Paare für {tf} geladen.");
        if skipped_filter_tf > 0: logger.info(f"{skipped_filter_tf} Paare für {tf} ohne Daten im Zeitrahmen.")
        if skipped_missing_cols > 0: logger.warning(f"{skipped_missing_cols} Dateien für {tf} wegen fehlender Spalten übersprungen.")
    valid_tf_keys = [tf for tf in timeframes if data_dict[tf]]
    if not valid_tf_keys: raise ValueError(f"Keine Daten für TFs {timeframes} geladen.")
    if len(valid_tf_keys) < len(timeframes):
        missing_tfs = [tf for tf in timeframes if tf not in valid_tf_keys]; logger.warning(f"Nicht alle TFs gefunden/behalten. Fehlend: {missing_tfs}. Nur mit: {valid_tf_keys}"); timeframes = valid_tf_keys
    if not timeframes: raise ValueError("Keine Timeframes übrig nach Validierung.")
    common_pairs = set.intersection(*[set(data_dict[tf].keys()) for tf in timeframes])
    if not common_pairs: error_msg = f"Keine gemeinsamen Paare für TFs {timeframes} gefunden"; raise ValueError(error_msg + (f" im Zeitrahmen '{timerange_str}'" if timerange_str else ""))
    logger.info(f"{len(common_pairs)} Paare haben Daten für alle TFs: {sorted(list(common_pairs))[:5]}...")
    final_data = {tf: {pair: df for pair, df in data_dict[tf].items() if pair in common_pairs} for tf in timeframes}
    final_data_clean = {tf: data for tf, data in final_data.items() if data}; final_timeframes = list(final_data_clean.keys())
    if not final_timeframes: raise ValueError("Keine Daten übrig nach Bereinigen auf gemeinsame Paare (final_timeframes leer).")
    common_pairs_final = set.intersection(*[set(final_data_clean[tf].keys()) for tf in final_timeframes])
    if not common_pairs_final: raise ValueError("Keine Daten übrig nach Bereinigen auf gemeinsame Paare (common_pairs leer).")
    final_data_clean = {tf: {pair: df for pair, df in final_data_clean[tf].items() if pair in common_pairs_final} for tf in final_timeframes}
    return final_data_clean, sorted(list(common_pairs_final)), final_timeframes

def create_exit_target(df, periods=5, threshold=0.0, vola_threshold_mult=0.5,
                       penalty_enable=False, penalty_oversold_enable=False, rsi_penalty_low_thr=25.0,
                       dist_low_50_pct_penalty_low_thr=1.5, penalty_htf_strong_uptrend_enable=False,
                       htf_rsi_penalty_high_thr=60.0, htf_ma_trend_check=True, htf_tf_name='1h'):
    """ Erstellt die Zielvariable 'target' für Exits. **MODIFIZIERT für globales Modell**. """
    target_col = 'target'
    if target_col in df.columns: df = df.drop(columns=[target_col])

    # *** WICHTIG: Shift pro Gruppe (Paar) berechnen! ***
    logger.debug(f"Exit Target: Berechne future close (shift by {-periods}) gruppiert nach 'pair'...")
    if 'pair' not in df.columns:
        raise ValueError("Spalte 'pair' fehlt im DataFrame für gruppierten Shift in create_exit_target.")
    future_close = df.groupby('pair')['close'].shift(-periods)

    base_target_condition = pd.Series(False, index=df.index)
    if vola_threshold_mult > 0:
        logger.debug(f"Exit Target: Preis < Close - {vola_threshold_mult} * ATR(14) in {periods} Perioden.")
        if 'feature_atr_14' not in df.columns: raise ValueError("ATR ('feature_atr_14') fehlt für vola-adjustiertes Exit-Target.")
        safe_atr = df['feature_atr_14'].fillna(0)
        base_target_condition = (future_close < df['close'] - vola_threshold_mult * safe_atr)
    elif threshold > 0:
        logger.debug(f"Exit Target: Preis < Close * (1 - {threshold:.4f}) in {periods} Perioden.")
        base_target_condition = (future_close / df['close'].replace(0, np.nan) - 1) < -threshold
    else: logger.warning("Kein gültiges Exit-Target definiert (weder Vola noch Threshold > 0).")

    base_target_condition = base_target_condition.fillna(False)
    initial_exit_count = base_target_condition.sum()

    if penalty_enable:
        logger.debug("Wende EXIT Penalty-Bedingungen an...")
        penalty = pd.Series(False, index=df.index) # Exit-Signal (1) wird zu 0, wenn Penalty=True

        if penalty_oversold_enable:
            cond_oversold = pd.Series(False, index=df.index); rsi_col = 'feature_rsi_14'; dist_low_col = 'feature_dist_from_low_50_pct'
            if rsi_col in df.columns: cond_oversold = cond_oversold | (df[rsi_col] < rsi_penalty_low_thr).fillna(False); logger.debug(f"  - Oversold Check: {rsi_col} < {rsi_penalty_low_thr}?")
            else: logger.warning(f"Exit Penalty-Feature {rsi_col} nicht gefunden.")
            if dist_low_col in df.columns: cond_oversold = cond_oversold | (df[dist_low_col] < dist_low_50_pct_penalty_low_thr).fillna(False); logger.debug(f"  - Oversold Check: {dist_low_col} < {dist_low_50_pct_penalty_low_thr}%?")
            else: logger.warning(f"Exit Penalty-Feature {dist_low_col} nicht gefunden.")
            penalty = penalty | cond_oversold
            logger.debug(f"  - Penalty (prevent exit) if OVERSOLD condition is TRUE.")

        if penalty_htf_strong_uptrend_enable:
             # Erwarte HTF Features im Format 'feature_TF_NAME_FEATURE_NAME_TF_inf'
             htf_suffix = f'_{htf_tf_name}_inf'; htf_rsi_col = f'feature_{htf_tf_name}_rsi_14{htf_suffix}'; htf_ma50_col = f'feature_{htf_tf_name}_ma_50{htf_suffix}'; htf_ma200_col = f'feature_{htf_tf_name}_ma_200{htf_suffix}'
             required_htf_cols = [htf_rsi_col];
             if htf_ma_trend_check: required_htf_cols.extend([htf_ma50_col, htf_ma200_col])

             missing_htf_cols = [col for col in required_htf_cols if col not in df.columns]
             if not missing_htf_cols:
                 cond_htf_rsi = (df[htf_rsi_col] > htf_rsi_penalty_high_thr).fillna(False); cond_htf_ma = True
                 if htf_ma_trend_check: cond_htf_ma = (df[htf_ma50_col] > df[htf_ma200_col]).fillna(False)
                 cond_htf_strong_up = cond_htf_rsi & cond_htf_ma; penalty = penalty | cond_htf_strong_up
                 log_msg_htf = f"  - Penalty (prevent exit) if HTF Trend strong UP ({htf_rsi_col} > {htf_rsi_penalty_high_thr}" + (f" AND {htf_ma50_col} > {htf_ma200_col}" if htf_ma_trend_check else "") + ")"
                 logger.debug(log_msg_htf)
             else: logger.warning(f"HTF Trend Penalty nicht anwendbar, fehlende Features: {missing_htf_cols}")

        final_target_condition = base_target_condition & (~penalty)
        num_penalized = (base_target_condition & penalty).sum()
        logger.info(f"Exit Penalties: {num_penalized} von {int(initial_exit_count)} ({num_penalized/max(1, initial_exit_count)*100:.2f}%) pot. Exit-Signalen unterdrückt.")
    else:
        final_target_condition = base_target_condition

    df[target_col] = final_target_condition.astype(int)
    final_exit_count = df[target_col].sum()
    logger.info(f"Exit Target erstellt: {final_exit_count} Exit-Signale (1) von {len(df)} ({final_exit_count/len(df)*100:.2f}%) nach Penalties.")
    return df

def objective(trial, X_train, y_train, X_test, y_test, study_name_prefix=""):
    # (Code unverändert zu entrytrain)
    params = { 'objective': 'binary', 'metric': 'auc', 'is_unbalance': True, 'n_estimators': 1000,
               'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
               'num_leaves': trial.suggest_int('num_leaves', 10, 100), 'max_depth': trial.suggest_int('max_depth', 3, 15),
               'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
               'min_child_samples': trial.suggest_int('min_child_samples', 5, 50), 'subsample': trial.suggest_float('subsample', 0.6, 1.0),
               'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 'n_jobs': -1, 'random_state': 42, 'verbose': -1 }
    try:
        model = lgb.LGBMClassifier(**params)
        early_stopping_callback_hpo = lgb.early_stopping(stopping_rounds=25, verbose=False)
        log_evaluation_callback_hpo = lgb.log_evaluation(period=-1)
        if len(y_train.value_counts()) < 2: raise ValueError("Nur eine Klasse in y_train für HPO.")
        eval_set_hpo = [(X_test, y_test)] if len(y_test.value_counts()) >= 2 else None
        if eval_set_hpo is None: logger.warning(f"HPO Trial {study_name_prefix}: Nur eine Klasse in y_test."); return 0.0
        model.fit(X_train, y_train, eval_set=eval_set_hpo, eval_metric='auc', callbacks=[early_stopping_callback_hpo, log_evaluation_callback_hpo])
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba); return auc
    except ValueError as e: logger.warning(f"HPO Trial {study_name_prefix} ValueError: {e}. Gebe 0.0 zurück."); return 0.0
    except Exception as e: logger.warning(f"HPO Trial {study_name_prefix} fehlgeschlagen: {e}"); return 0.0

# --- Hauptfunktion ---
def main():
    logger.info("--- Start: MTA GLOBAL EXIT Modell-Trainingsskript ---")
    # --- Konfiguration loggen ---
    logger.info("--- Verwendete EXIT Konfiguration ---")
    # ... (alle logger.info Aufrufe für Konfig) ...
    logger.info(f"  User Data Dir:       {user_data_dir}")
    logger.info(f"  Daten Verzeichnis:   {data_directory}")
    logger.info(f"  Modell Verzeichnis:  {save_directory}")
    logger.info(f"  Provider Verzeichnis:{provider_base_dir}/{USER_CONFIG_PROVIDER_DIR_SUFFIX}")
    logger.info(f"  Basis Timeframe:     {USER_CONFIG_BASE_TIMEFRAME}")
    logger.info(f"  Höherer Timeframe:   {USER_CONFIG_HIGHER_TIMEFRAME}")
    logger.info(f"  Timerange:           {USER_CONFIG_TIMERANGE}")
    logger.info(f"  Lookahead Perioden:  {USER_CONFIG_LOOKAHEAD_PERIODS}")
    logger.info(f"  Exit Target Vola M:  {USER_CONFIG_EXIT_TARGET_VOLA_MULT}")
    logger.info(f"  Exit Target Thr (%): {USER_CONFIG_EXIT_TARGET_THRESHOLD if USER_CONFIG_EXIT_TARGET_VOLA_MULT <= 0 else 'N/A (Vola genutzt)'}")
    logger.info(f"  Exit Penalties:      {USER_CONFIG_EXIT_PENALTY_ENABLE}")
    if USER_CONFIG_EXIT_PENALTY_ENABLE:
        logger.info(f"    Oversold Enable:   {USER_CONFIG_PENALTY_OVERSOLD_ENABLE}")
        if USER_CONFIG_PENALTY_OVERSOLD_ENABLE:
            logger.info(f"      RSI Low Thr:     {USER_CONFIG_PENALTY_RSI_LOW_THRESHOLD}")
            logger.info(f"      Dist Low Thr:    {USER_CONFIG_PENALTY_DIST_LOW_50_PCT_LOW_THRESHOLD}%")
        logger.info(f"    HTF Up Enable:     {USER_CONFIG_PENALTY_HTF_STRONG_UPTREND_ENABLE}")
        if USER_CONFIG_PENALTY_HTF_STRONG_UPTREND_ENABLE:
            logger.info(f"      HTF RSI High Thr:{USER_CONFIG_PENALTY_HTF_RSI_HIGH_THRESHOLD}")
            logger.info(f"      HTF MA Check:    {USER_CONFIG_PENALTY_HTF_MA_TREND_CHECK}")
    logger.info(f"  CV Splits:           {USER_CONFIG_N_SPLITS}")
    logger.info(f"  Min Samples Komb.:   {USER_CONFIG_MIN_SAMPLES_COMBINED}")
    logger.info(f"  HPO Aktiviert:       {USER_CONFIG_ENABLE_HPO}")
    if USER_CONFIG_ENABLE_HPO: logger.info(f"  HPO Trials:          {USER_CONFIG_HPO_TRIALS}")
    else: logger.info(f"  Feste Parameter:     {USER_CONFIG_FIXED_PARAMS}")
    logger.info(f"  Modell Präfix:       {USER_CONFIG_MODEL_PREFIX}") # Exit Prefix
    logger.info(f"  Finale Estimators:   {USER_CONFIG_N_ESTIMATORS_FINAL}")
    logger.info(f"  Finale EarlyStop:    {USER_CONFIG_EARLY_STOPPING_ROUNDS_FINAL}")
    logger.info(f"  Min Exit Precision:  {USER_CONFIG_EXIT_MIN_PRECISION_TARGET}") # Exit Precision
    if USER_CONFIG_TEST_PAIRS_SUBSET: logger.warning(f"!!! Test Subset Paare: {USER_CONFIG_TEST_PAIRS_SUBSET} (Nur diese werden geladen!)")
    else: logger.info(f"  Test Subset Paare:   None (Alle verfügbaren Paare werden geladen)")
    logger.info("-----------------------------")

    # --- Feature Provider & Timerange ---
    try: feature_provider = MTAFeatureProvider(base_tf=USER_CONFIG_BASE_TIMEFRAME, higher_tf=USER_CONFIG_HIGHER_TIMEFRAME)
    except Exception as e: logger.critical(f"Fehler bei Initialisierung MTAFeatureProvider: {e}"); logger.error(traceback.format_exc()); return
    try: start_dt, end_dt = parse_timerange(USER_CONFIG_TIMERANGE)
    except ValueError as e: logger.critical(f"Fehler im Timerange-Format: {e}"); return

    # --- Daten laden ---
    # (Code identisch zu entrytrain)
    try:
        search_pattern_base = os.path.join(data_directory, f'*-{USER_CONFIG_BASE_TIMEFRAME}.feather')
        all_potential_pairs = sorted([os.path.basename(f).replace(f'-{USER_CONFIG_BASE_TIMEFRAME}.feather', '').replace('_', '/') for f in glob.glob(search_pattern_base)])
        if not all_potential_pairs: raise FileNotFoundError(f"Keine '{USER_CONFIG_BASE_TIMEFRAME}' Feather-Dateien in '{data_directory}' gefunden.")
        logger.info(f"{len(all_potential_pairs)} mögliche Paare für {USER_CONFIG_BASE_TIMEFRAME} gefunden.")
        pairs_to_load = all_potential_pairs
        if USER_CONFIG_TEST_PAIRS_SUBSET:
            pairs_to_load = sorted([p for p in all_potential_pairs if p in USER_CONFIG_TEST_PAIRS_SUBSET])
            if not pairs_to_load: logger.critical(f"FEHLER: Keine der Testpaare {USER_CONFIG_TEST_PAIRS_SUBSET} gefunden."); return
            logger.warning(f"Lade nur Subset von {len(pairs_to_load)} Paaren: {pairs_to_load}")
        mta_data_dict, common_pairs_list, actual_timeframes = load_data_mta(data_directory, pairs_to_load, [USER_CONFIG_BASE_TIMEFRAME, USER_CONFIG_HIGHER_TIMEFRAME], start_dt, end_dt, USER_CONFIG_TIMERANGE)
        if USER_CONFIG_BASE_TIMEFRAME not in actual_timeframes or USER_CONFIG_HIGHER_TIMEFRAME not in actual_timeframes:
             missing = [tf for tf in [USER_CONFIG_BASE_TIMEFRAME, USER_CONFIG_HIGHER_TIMEFRAME] if tf not in actual_timeframes]; logger.critical(f"FEHLER: TFs {missing} nicht geladen."); return
        if not common_pairs_list: logger.critical(f"FEHLER: Keine gemeinsamen Paare nach Laden/Filtern."); return
    except FileNotFoundError as e: logger.critical(f"{e}"); return
    except ValueError as e: logger.critical(f"Fehler bei Datenvorbereitung: {e}"); return
    except Exception as e: logger.critical(f"Unerwarteter Fehler beim Laden: {e}"); logger.error(traceback.format_exc()); return

    # --- Daten Kombinieren ---
    # (Code identisch zu entrytrain)
    try:
        all_base_dfs = []; all_htf_dfs = []
        logger.info(f"Kombiniere Daten von {len(common_pairs_list)} Paaren...")
        for pair in common_pairs_list:
            if pair not in mta_data_dict[USER_CONFIG_BASE_TIMEFRAME] or pair not in mta_data_dict[USER_CONFIG_HIGHER_TIMEFRAME]: logger.warning(f"Überspringe {pair} beim Kombinieren (Daten fehlen)."); continue
            base_df = mta_data_dict[USER_CONFIG_BASE_TIMEFRAME][pair].copy(); htf_df = mta_data_dict[USER_CONFIG_HIGHER_TIMEFRAME][pair].copy()
            if base_df.empty or htf_df.empty: logger.warning(f"Überspringe {pair} beim Kombinieren (leere DFs)."); continue
            base_df['pair'] = pair; htf_df['pair'] = pair
            all_base_dfs.append(base_df); all_htf_dfs.append(htf_df)
        if not all_base_dfs or not all_htf_dfs: raise ValueError("Keine Daten nach dem Filtern für die Kombination übrig.")
        combined_base_df = pd.concat(all_base_dfs, ignore_index=False); combined_base_df.sort_index(inplace=True)
        combined_htf_df = pd.concat(all_htf_dfs, ignore_index=False); combined_htf_df.sort_index(inplace=True)
        logger.info(f"Kombinierte Basis-Daten: {combined_base_df.shape[0]} Zeilen, {len(combined_base_df['pair'].unique())} Paare")
        logger.info(f"Kombinierte HTF-Daten: {combined_htf_df.shape[0]} Zeilen, {len(combined_htf_df['pair'].unique())} Paare")
    except Exception as e: logger.critical(f"Fehler beim Kombinieren der Daten: {e}"); logger.error(traceback.format_exc()); return

    # --- Feature Engineering für kombinierte Daten ---
    # (Code identisch zu entrytrain)
    try:
        logger.info("Starte Feature Engineering für kombinierte Daten...")
        df_features_combined = feature_provider.add_all_features(combined_base_df, combined_htf_df)
        if 'pair' not in df_features_combined.columns and 'pair' in combined_base_df.columns: df_features_combined['pair'] = combined_base_df['pair']
        elif 'pair' not in df_features_combined.columns: raise ValueError("'pair' Spalte nach Feature Engineering verloren.")
        logger.info(f"Feature Engineering abgeschlossen. Shape: {df_features_combined.shape}")
    except Exception as e: logger.critical(f"Fehler Feature Engineering: {e}"); logger.error(traceback.format_exc()); return

    # --- *** EXIT Target Erstellung für kombinierte Daten *** ---
    try:
        logger.info("Erstelle EXIT Target für kombinierte Daten...")
        df_final_combined = create_exit_target( # <<< EXIT TARGET Funktion
            df_features_combined,
            periods=USER_CONFIG_LOOKAHEAD_PERIODS,
            threshold=USER_CONFIG_EXIT_TARGET_THRESHOLD if USER_CONFIG_EXIT_TARGET_VOLA_MULT <= 0 else 0,
            vola_threshold_mult=USER_CONFIG_EXIT_TARGET_VOLA_MULT,
            penalty_enable=USER_CONFIG_EXIT_PENALTY_ENABLE,
            penalty_oversold_enable=USER_CONFIG_PENALTY_OVERSOLD_ENABLE,
            rsi_penalty_low_thr=USER_CONFIG_PENALTY_RSI_LOW_THRESHOLD,
            dist_low_50_pct_penalty_low_thr=USER_CONFIG_PENALTY_DIST_LOW_50_PCT_LOW_THRESHOLD,
            penalty_htf_strong_uptrend_enable=USER_CONFIG_PENALTY_HTF_STRONG_UPTREND_ENABLE,
            htf_rsi_penalty_high_thr=USER_CONFIG_PENALTY_HTF_RSI_HIGH_THRESHOLD,
            htf_ma_trend_check=USER_CONFIG_PENALTY_HTF_MA_TREND_CHECK,
            htf_tf_name=USER_CONFIG_HIGHER_TIMEFRAME # Wichtig für Penalty
        )
    except Exception as e: logger.critical(f"Fehler bei EXIT Target-Erstellung: {e}"); logger.error(traceback.format_exc()); return


    # --- Phase 1: HPO (falls aktiviert) ---
    # (Code identisch zu entrytrain, außer study_name)
    best_params_glob = USER_CONFIG_FIXED_PARAMS.copy()
    if USER_CONFIG_ENABLE_HPO:
        logger.info("--- Starte HPO Phase (Globales EXIT Modell) ---")
        try:
            feature_columns_hpo = sorted([col for col in df_final_combined.columns if col.startswith('feature_')]); target_col_hpo = 'target'
            if not feature_columns_hpo or target_col_hpo not in df_final_combined.columns: raise ValueError("Keine Features/Target für HPO.")
            df_ml_hpo = df_final_combined[feature_columns_hpo + [target_col_hpo] + ['pair']].copy();
            for col in feature_columns_hpo: df_ml_hpo[col] = pd.to_numeric(df_ml_hpo[col], errors='coerce')
            df_ml_hpo.replace([np.inf, -np.inf], np.nan, inplace=True); df_ml_hpo[feature_columns_hpo] = df_ml_hpo[feature_columns_hpo].ffill()
            df_ml_hpo.dropna(subset=feature_columns_hpo + [target_col_hpo], inplace=True)
            if len(df_ml_hpo) < USER_CONFIG_MIN_SAMPLES_COMBINED // 2: raise ValueError(f"Zu wenige Daten ({len(df_ml_hpo)}) für HPO.");
            if len(df_ml_hpo[target_col_hpo].value_counts()) < 2: raise ValueError(f"Nur eine Target-Klasse für HPO.")
            logger.info(f"[HPO EXIT] Target-Verteilung: \n{df_ml_hpo[target_col_hpo].value_counts(normalize=True).round(4).to_string()}")
            X_hpo = df_ml_hpo[feature_columns_hpo]; y_hpo = df_ml_hpo[target_col_hpo]
            tscv_hpo = TimeSeriesSplit(n_splits=USER_CONFIG_N_SPLITS); train_idx_hpo, test_idx_hpo = list(tscv_hpo.split(X_hpo))[-1]
            if len(train_idx_hpo) == 0 or len(test_idx_hpo) == 0: raise ValueError("Leere Splits für HPO.")
            X_train_hpo, X_test_hpo = X_hpo.iloc[train_idx_hpo], X_hpo.iloc[test_idx_hpo]; y_train_hpo, y_test_hpo = y_hpo.iloc[train_idx_hpo], y_hpo.iloc[test_idx_hpo]
            if X_train_hpo.empty or X_test_hpo.empty or len(y_train_hpo.value_counts()) < 2 or len(y_test_hpo.value_counts()) < 2: raise ValueError("Leere/einklassige Sets für HPO.")
            # *** Anderer Study Name ***
            study_name = f"lgbm_exit_hpo_GLOBAL_{USER_CONFIG_MODEL_PREFIX}"
            study = optuna.create_study(direction="maximize", study_name=study_name)
            study.optimize(lambda trial: objective(trial, X_train_hpo, y_train_hpo, X_test_hpo, y_test_hpo, study_name), n_trials=USER_CONFIG_HPO_TRIALS, show_progress_bar=True)
            best_params_glob = study.best_params; logger.info(f"--- HPO EXIT Abgeschlossen ---"); logger.info(f"Bester AUC Score: {study.best_value:.6f}"); logger.info(f"Beste Parameter: {best_params_glob}")
        except Exception as e_hpo: logger.error(f"FEHLER HPO EXIT: {e_hpo}"); logger.error(traceback.format_exc()); logger.warning("Fahre mit festen Params fort."); best_params_glob = USER_CONFIG_FIXED_PARAMS.copy()
    else: logger.info("--- HPO Phase übersprungen ---"); logger.info(f"Verwende feste Parameter: {best_params_glob}")

    # --- Phase 2: Finales Training (Globales EXIT Modell) ---
    logger.info("--- Starte Finale Trainingsphase (Globales EXIT Modell) ---")
    try:
        # 1. ML Vorbereitung & Bereinigung
        # (Code identisch zu entrytrain)
        feature_columns = sorted([col for col in df_final_combined.columns if col.startswith('feature_')]); target_col = 'target'
        if not feature_columns or target_col not in df_final_combined.columns: raise ValueError("Keine Features/Target nach Engineering.")
        df_ml = df_final_combined[feature_columns + [target_col] + ['pair']].copy()
        for col in feature_columns: df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
        df_ml.replace([np.inf, -np.inf], np.nan, inplace=True); df_ml[feature_columns] = df_ml[feature_columns].ffill()
        remaining_nans = df_ml[feature_columns].isnull().sum().sum()
        if remaining_nans > 0: logger.warning(f"{remaining_nans} NaNs verbleiben nach ffill (werden durch dropna entfernt).")
        initial_rows = len(df_ml); df_ml.dropna(subset=feature_columns + [target_col], inplace=True); final_rows = len(df_ml)
        logger.info(f"Daten nach dropna: {final_rows} (von {initial_rows})")
        if final_rows < USER_CONFIG_MIN_SAMPLES_COMBINED: raise ValueError(f"Zu wenige Daten ({final_rows}) (Min: {USER_CONFIG_MIN_SAMPLES_COMBINED}).")
        if len(df_ml[target_col].value_counts()) < 2: raise ValueError(f"Nur eine Target-Klasse nach Bereinigung: {df_ml[target_col].value_counts()}")
        X = df_ml[feature_columns]; y = df_ml[target_col]; pairs_in_final_data = df_ml['pair']
        logger.info(f"Daten für ML: {len(X)} Zeilen"); logger.info(f"EXIT Target-Verteilung (Global): \n{y.value_counts(normalize=True).round(4).to_string()}")

        # 2. Daten aufteilen
        # (Code identisch zu entrytrain)
        tscv = TimeSeriesSplit(n_splits=USER_CONFIG_N_SPLITS)
        try: train_indices, test_indices = list(tscv.split(X))[-1]
        except Exception as e_split: raise ValueError(f"Fehler bei TimeSeriesSplit: {e_split}")
        if len(train_indices) == 0 or len(test_indices) == 0: raise ValueError("Leere Splits.")
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]; y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]; pairs_test = pairs_in_final_data.iloc[test_indices]
        if X_train.empty or X_test.empty or len(y_train.value_counts()) < 2: raise ValueError("Leere/einklassige Train/Test-Sets.")
        logger.info(f"Train: {len(X_train)} Zeilen, Test: {len(X_test)} Zeilen")

        # 3. Modell trainieren
        # (Code identisch zu entrytrain)
        final_params = best_params_glob.copy(); final_params['objective'] = 'binary'; final_params['metric'] = 'auc'; final_params['is_unbalance'] = True
        final_params['n_estimators'] = USER_CONFIG_N_ESTIMATORS_FINAL; final_params['n_jobs'] = -1; final_params['random_state'] = 42; final_params['verbose'] = -1
        model = lgb.LGBMClassifier(**final_params)
        early_stopping_callback_final = lgb.early_stopping(stopping_rounds=USER_CONFIG_EARLY_STOPPING_ROUNDS_FINAL, verbose=False)
        log_evaluation_callback_final = lgb.log_evaluation(period=500)
        logger.info(f"Starte finales globales EXIT-Training...")
        eval_set = [(X_test, y_test)] if len(y_test.value_counts()) >= 2 else None
        if eval_set is None: logger.warning("Nur eine Klasse in y_test. Early Stopping nur auf Training.")
        if X_train.isnull().values.any() or np.isinf(X_train.values).any(): raise ValueError("NaN/Inf in X_train VOR fit.")
        if y_train.isnull().values.any(): raise ValueError("NaN in y_train VOR fit.")
        model.fit(X_train, y_train, eval_set=eval_set, eval_metric='auc', callbacks=[early_stopping_callback_final, log_evaluation_callback_final])
        logger.info(f"Training beendet. Beste Iteration: {model.best_iteration_}")

        # 4. Modell evaluieren & Thresholds pro Paar finden
        logger.info(f"Evaluiere globales EXIT-Modell und finde Thresholds pro Paar...")
        pair_thresholds = {}
        pair_results = {} # Store detailed results per pair
        report_overall = {}

        if y_test.empty or len(y_test.value_counts()) < 2:
            logger.error("Testset ungültig für Evaluation.")
            report_overall = {'error': 'Invalid test set'}
        else:
            try:
                best_iter = model.best_iteration_ if model.best_iteration_ and model.best_iteration_ > 0 else -1
                X_test_eval = X_test[X_train.columns]
                if X_test_eval.isnull().values.any() or np.isinf(X_test_eval.values).any(): raise ValueError("NaN/Inf in X_test VOR predict.")

                logger.info("Mache Vorhersagen für das gesamte Testset...")
                y_pred_proba_full = model.predict_proba(X_test_eval, num_iteration=best_iter)[:, 1]
                if np.isnan(y_pred_proba_full).all(): raise ValueError("predict_proba nur NaNs.")

                unique_pairs_in_test = pairs_test.unique()
                logger.info(f"Finde EXIT-Thresholds für {len(unique_pairs_in_test)} Paare im Testset...")

                for pair_name in unique_pairs_in_test:
                    pair_mask = (pairs_test == pair_name)
                    y_test_pair = y_test[pair_mask]
                    y_pred_proba_pair = y_pred_proba_full[pair_mask]

                    if len(y_test_pair) < 10 or len(y_test_pair.value_counts()) < 2:
                        logger.warning(f"[{pair_name} EXIT] Zu wenige Daten (<10) oder nur eine Klasse im Testset. Setze Threshold auf 0.5.")
                        pair_thresholds[pair_name] = 0.5
                        pair_results[pair_name] = {'pair': pair_name, 'threshold': 0.5, 'error': 'Insufficient test data'}
                        continue

                    precision, recall, thresholds = precision_recall_curve(y_test_pair, y_pred_proba_pair)
                    valid_pr_indices = ~np.isnan(precision) & ~np.isnan(recall)
                    precision_valid = precision[valid_pr_indices]
                    recall_valid = recall[valid_pr_indices]
                    thresholds_valid = thresholds[np.where(valid_pr_indices)[0][:-1]] if len(thresholds) > 0 and np.sum(valid_pr_indices) > 1 else np.array([])

                    if len(precision_valid) <= 1 or len(thresholds_valid) == 0:
                         logger.warning(f"[{pair_name} EXIT] Zu wenige gültige Punkte für Threshold-Findung. Setze Threshold auf 0.5.")
                         pred_threshold_pair = 0.5; best_f1_pair_exit = 0.0; achieved_precision_exit = np.nan; achieved_recall_exit = np.nan
                    else:
                        # *** Suche Threshold basierend auf EXIT Precision ***
                        min_precision_target = USER_CONFIG_EXIT_MIN_PRECISION_TARGET
                        f1_scores = np.divide(2*recall_valid[:-1]*precision_valid[:-1], recall_valid[:-1]+precision_valid[:-1], out=np.zeros_like(recall_valid[:-1]), where=(recall_valid[:-1]+precision_valid[:-1])!=0)
                        f1_scores[np.isnan(f1_scores)] = 0
                        valid_precision_indices = np.where(precision_valid[:-1] >= min_precision_target)[0]

                        if len(valid_precision_indices) > 0:
                            valid_f1_scores = f1_scores[valid_precision_indices]
                            if np.max(valid_f1_scores) > 0:
                                ix_relative = np.argmax(valid_f1_scores); ix = valid_precision_indices[ix_relative]
                                if ix < len(thresholds_valid):
                                    pred_threshold_pair = thresholds_valid[ix]; best_f1_pair_exit = f1_scores[ix]; achieved_precision_exit=precision_valid[ix]; achieved_recall_exit=recall_valid[ix];
                                else: logger.warning(f"[{pair_name} EXIT] Index-Fehler (1). Setze Threshold auf 0.5."); pred_threshold_pair = 0.5; best_f1_pair_exit = 0.0; achieved_precision_exit=np.nan; achieved_recall_exit=np.nan
                            else: # Fallback Max Precision
                                logger.warning(f"[{pair_name} EXIT] Kein F1>0 für P>={min_precision_target:.2f}. Fallback: Max Precision.")
                                ix = np.argmax(precision_valid[:-1])
                                if ix < len(thresholds_valid): pred_threshold_pair = thresholds_valid[ix]; best_f1_pair_exit = f1_scores[ix]; achieved_precision_exit=precision_valid[ix]; achieved_recall_exit=recall_valid[ix];
                                else: logger.warning(f"[{pair_name} EXIT] Fallback Max P Index OOB. Nutze 0.5."); pred_threshold_pair=0.5; best_f1_pair_exit=0.0; achieved_precision_exit=np.nan; achieved_recall_exit=np.nan
                        else: # Fallback Max Precision
                            logger.warning(f"[{pair_name} EXIT] Kein Thresh mit P>={min_precision_target:.2f}. Fallback: Max Precision.")
                            ix = np.argmax(precision_valid[:-1])
                            if ix < len(thresholds_valid): pred_threshold_pair = thresholds_valid[ix]; best_f1_pair_exit = f1_scores[ix]; achieved_precision_exit=precision_valid[ix]; achieved_recall_exit=recall_valid[ix];
                            else: logger.warning(f"[{pair_name} EXIT] Fallback Max P Index OOB. Nutze 0.5."); pred_threshold_pair=0.5; best_f1_pair_exit=0.0; achieved_precision_exit=np.nan; achieved_recall_exit=np.nan

                    # Speichere Threshold
                    pair_thresholds[pair_name] = pred_threshold_pair

                    # Berechne Metriken für diesen Paar-Threshold
                    y_pred_class_pair = (y_pred_proba_pair > pred_threshold_pair).astype(int)
                    accuracy_pair = accuracy_score(y_test_pair, y_pred_class_pair)
                    # *** Andere Target-Namen ***
                    target_names_exit = ['Hold', 'Exit (Drop)']
                    report_dict_pair = classification_report(y_test_pair, y_pred_class_pair, target_names=target_names_exit, zero_division=0, output_dict=True)

                    # Speichere Ergebnisse für die Zusammenfassung
                    pair_results[pair_name] = {
                        'pair': pair_name,
                        'threshold': pred_threshold_pair,
                        # *** Fokus auf Exit Klasse (war 'Anstieg' im entrytrain) ***
                        'f1_Exit': report_dict_pair.get('Exit (Drop)', {}).get('f1-score', np.nan),
                        'prec_Exit': report_dict_pair.get('Exit (Drop)', {}).get('precision', np.nan),
                        'recall_Exit': report_dict_pair.get('Exit (Drop)', {}).get('recall', np.nan),
                        'support_Exit': report_dict_pair.get('Exit (Drop)', {}).get('support', np.nan),
                        'accuracy': accuracy_pair,
                        'test_samples': len(y_test_pair)
                    }

                # Gesamt-Report (optional, mit einem Standard-Threshold wie 0.5)
                y_pred_class_overall = (y_pred_proba_full > 0.5).astype(int)
                accuracy_overall = accuracy_score(y_test, y_pred_class_overall)
                logger.info(f"\nGesamt-Accuracy EXIT (@Thresh 0.5): {accuracy_overall:.4f}")
                target_names_exit = ['Hold', 'Exit (Drop)']
                report_str_overall = classification_report(y_test, y_pred_class_overall, target_names=target_names_exit, zero_division=0)
                logger.info(f"Gesamt Classification Report EXIT (@Thresh 0.5):\n{report_str_overall}")
                report_overall = classification_report(y_test, y_pred_class_overall, target_names=target_names_exit, zero_division=0, output_dict=True)
                report_overall['accuracy'] = accuracy_overall

            except Exception as e_eval:
                logger.error(f"Fehler bei Evaluation/Threshold-Findung EXIT: {e_eval}"); logger.error(traceback.format_exc())
                report_overall = {'error': str(e_eval)}

        # 5. Ergebnisse & Feature Importance anzeigen
        # (Code identisch zu entrytrain)
        if hasattr(model, 'feature_importances_'):
             try:
                 importances = model.feature_importances_
                 if importances is not None and len(importances) == len(X_train.columns):
                     logger.info(f"Top 10 Features (Globales EXIT Modell):"); feature_imp = pd.DataFrame({'Value': importances, 'Feature': X_train.columns}); logger.info("\n" + feature_imp.nlargest(10, "Value").to_string(index=False))
                 else: logger.warning("Feature Importances nicht ok.")
             except Exception as fi_e: logger.error(f"Fehler Feature Importance: {fi_e}")

        # 6. Modell & Features speichern
        # (Code identisch zu entrytrain, nutzt aber EXIT Prefix)
        os.makedirs(save_directory, exist_ok=True)
        base_tf_safe = USER_CONFIG_BASE_TIMEFRAME.replace('/', '_'); higher_tf_safe = USER_CONFIG_HIGHER_TIMEFRAME.replace('/', '_')
        base_filename = f"{USER_CONFIG_MODEL_PREFIX}_GLOBAL_{base_tf_safe}_WITH_{higher_tf_safe}" # EXIT Prefix
        model_filepath = os.path.join(save_directory, f"{base_filename}.joblib")
        features_filepath = os.path.join(save_directory, f"{base_filename}_features.json")
        threshold_filepath = os.path.join(save_directory, f"{USER_CONFIG_MODEL_PREFIX}_thresholds.json") # EXIT Prefix

        logger.info(f"Speichere globales EXIT-Modell: {model_filepath}"); joblib.dump(model, model_filepath)
        feature_list_to_save = list(X_train.columns)
        logger.info(f"Speichere globale Features ({len(feature_list_to_save)}): {features_filepath}")
        try:
            with open(features_filepath, 'w') as f: json.dump(feature_list_to_save, f, indent=4)
            logger.info(f"EXIT Modell/Features gespeichert.")
        except Exception as e_save: logger.error(f"Fehler Speichern EXIT Features: {e_save}")

        # 7. Thresholds speichern (paar-spezifisch)
        # (Code identisch zu entrytrain)
        if pair_thresholds:
            logger.info(f"Speichere paar-spezifische EXIT-Thresholds nach {threshold_filepath}...")
            try:
                threshold_dict_serializable = {k: float(v) if pd.notna(v) else 0.5 for k, v in pair_thresholds.items()}
                with open(threshold_filepath, 'w') as f: json.dump(threshold_dict_serializable, f, indent=4, sort_keys=True)
                logger.info(f"EXIT Thresholds für {len(threshold_dict_serializable)} Paare gespeichert.")
            except Exception as e: logger.error(f"Fehler Speichern EXIT Thresholds: {e}"); logger.error(traceback.format_exc())
        else: logger.warning("Keine paar-spezifischen EXIT-Thresholds zum Speichern gefunden.")

    except ValueError as e: logger.error(f"Globales EXIT Training gescheitert wegen ValueError: {e}."); logger.error(traceback.format_exc());
    except KeyError as e: logger.error(f"Globales EXIT Training gescheitert wegen KeyError: {e}."); logger.error(traceback.format_exc());
    except MemoryError as e: logger.error(f"Globales EXIT Training gescheitert wegen MemoryError: {e}."); logger.error(traceback.format_exc());
    except Exception as e: logger.error(f"Unerwarteter Fehler im globalen EXIT Training: {e}."); logger.error(traceback.format_exc());
    finally: logger.info(f"===== Globales EXIT Training abgeschlossen =====")

    # --- Ergebnisübersicht (Paar-spezifisch, Fokus auf Exit-Metriken) ---
    logger.info("\n--- EXIT Training abgeschlossen ---")
    logger.info("\n--- EXIT Ergebnisübersicht pro Paar (basierend auf globalem Modell & paar-spezifischem Threshold) ---")
    if pair_results:
        summary_data = list(pair_results.values())
        summary_df = pd.DataFrame(summary_data)
        # *** Sortiere nach Exit Precision/F1 ***
        summary_df = summary_df.sort_values(by=['prec_Exit', 'f1_Exit'], ascending=[False, False], na_position='last')
        logger.info("EXIT Paar-Ergebnis Zusammenfassung:\n" + summary_df[['pair', 'threshold', 'prec_Exit', 'recall_Exit', 'f1_Exit', 'accuracy', 'test_samples']].to_string(index=False, float_format="%.4f"))
    else: logger.info("Keine Einzelergebnisse für Zusammenfassung.")


# --- Skriptstart Schutz ---
if __name__ == "__main__":
    logger.info("exittrain.py (global) wird als Hauptskript ausgeführt.")
    main()
else:
    logger.warning("exittrain.py (global) wird als Modul importiert (unerwartet).")

# --- END OF FILE exittrain.py (GLOBAL MODEL VERSION) ---
