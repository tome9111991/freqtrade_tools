# --- START OF FILE exittrain.py (KORRIGIERT & Simplified Penalties) ---

# --- exittrain.py ---
# Angepasst von train.py für das Training von EXIT-Modellen
# KORRIGIERTE VERSION: .bfill() an kritischer Stelle im NaN-Handling entfernt.
# VERSION MIT VEREINFACHTEM PENALTY-SYSTEM
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
# Eindeutiger Logger-Name für Exit-Training
logger = logging.getLogger("exit_train_script")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ================================================================
# === START: BENUTZERKONFIGURATION (EXIT - Simplified Penalties) ===
# ================================================================
# --- Pfade ---
USER_CONFIG_USER_DATA_DIR = '/home/tommi/freqtrade/user_data' # <<< ANPASSEN
USER_CONFIG_DATA_DIR_SUFFIX = 'data/binance'
USER_CONFIG_MODELS_DIR_SUFFIX = 'models' # Gleicher Ordner wie Entry-Modelle ok? Oder Unterordner 'exit_models'?
USER_CONFIG_PROVIDER_DIR_SUFFIX = 'strategies/includes'

# --- Zeitrahmen ---
USER_CONFIG_BASE_TIMEFRAME = '30m' # <<< ANPASSEN (z.B. '1h', '30m')
USER_CONFIG_HIGHER_TIMEFRAME = '4h' # <<< ANPASSEN (z.B. '4h', '1d')

# --- Trainingsdaten-Filter ---
# WICHTIG: Genug Daten für Training + Validierung + Test sicherstellen
USER_CONFIG_TIMERANGE = '20230201-20250410' # <<< ANPASSEN

# --- EXIT Zielvariable & Daten ---
USER_CONFIG_LOOKAHEAD_PERIODS = 5 # <<< ANPASSEN: Wie viele Kerzen vorausschauen?
USER_CONFIG_EXIT_TARGET_VOLA_MULT = 0.6 # <<< ANPASSEN: ATR-Multiplikator für Kursfall (0 deaktiviert Vola-Target)
USER_CONFIG_EXIT_TARGET_THRESHOLD = 0.006 # <<< ANPASSEN: % Kursfall (nur wenn Vola_Mult = 0)
USER_CONFIG_N_SPLITS = 5 # Anzahl Splits für TimeSeriesSplit
USER_CONFIG_MIN_SAMPLES_PER_PAIR = 200 # Mindestanzahl Datenpunkte pro Paar nach Bereinigung

# --- *** VEREINFACHTE EXIT Penalty Konfiguration *** ---
# Exit-Target (1 = Preis fällt) wird 0 (Hold), wenn eine der aktivierten Penalty-Bedingungen zutrifft.
USER_CONFIG_EXIT_PENALTY_ENABLE = True # <<< ANPASSEN: Penalties AN/AUS

# Penalty 1: Markt extrem überverkauft? (Verhindere Panik-Exit)
# Wird aktiv, wenn RSI ODER Distanz zum Tief unter den Schwellwert fällt.
USER_CONFIG_PENALTY_OVERSOLD_ENABLE = True # <<< ANPASSEN
USER_CONFIG_PENALTY_RSI_LOW_THRESHOLD = 25.0  # <<< ANPASSEN: Unter diesem RSI -> kein Exit
USER_CONFIG_PENALTY_DIST_LOW_50_PCT_LOW_THRESHOLD = 1.5 # <<< ANPASSEN: Unter dieser %-Distanz zum Tief -> kein Exit

# Penalty 2: Übergeordneter (HTF) Trend noch stark AUFWÄRTS? (Verhindere Exit in starkem Uptrend)
# Wird aktiv, wenn HTF Indikatoren starken Aufwärtstrend anzeigen.
USER_CONFIG_PENALTY_HTF_STRONG_UPTREND_ENABLE = True # <<< ANPASSEN
USER_CONFIG_PENALTY_HTF_RSI_HIGH_THRESHOLD = 60.0 # <<< ANPASSEN: HTF RSI muss darüber sein UND...
USER_CONFIG_PENALTY_HTF_MA_TREND_CHECK = True     # <<< ANPASSEN: ... HTF MA50 > HTF MA200 muss gelten?

# --- HPO (Optuna) Konfiguration ---
USER_CONFIG_ENABLE_HPO = True # <<< ANPASSEN: HPO an/aus?
USER_CONFIG_HPO_TRIALS = 20    # <<< ANPASSEN: Anzahl HPO Versuche
USER_CONFIG_HPO_PAIR = 'SOL/USDC' # <<< ANPASSEN: Welches Paar für HPO? (None = erstes verfügbares)

# --- Feste Modellparameter (Falls HPO deaktiviert oder fehlschlägt) ---
USER_CONFIG_FIXED_PARAMS = { # <<< ANPASSEN: Gute Startwerte / Fallback
    'learning_rate': 0.02, 'num_leaves': 20, 'max_depth': 7,
    'reg_alpha': 1e-7, 'reg_lambda': 1e-7, 'min_child_samples': 30,
    'subsample': 0.85, 'colsample_bytree': 0.7
}

# --- Training Konfiguration (EXIT) ---
# WICHTIG: Eindeutiger Prefix für Exit-Modelle!
USER_CONFIG_MODEL_PREFIX = 'mta_exit_model2' # <<< ANPASSEN (muss mit Strategie übereinstimmen)
USER_CONFIG_N_ESTIMATORS_FINAL = 3000 # Max. Anzahl Bäume für finales Training
USER_CONFIG_EARLY_STOPPING_ROUNDS_FINAL = 50 # Runden für Early Stopping
# Ziel-Precision für Exit-Klasse (1 = Preis fällt) bei Threshold-Suche
USER_CONFIG_EXIT_MIN_PRECISION_TARGET = 0.55 # <<< ANPASSEN (z.B. 0.55, 0.60)

# +++ Fokus auf Teilmenge von Paaren für schnelle Tests +++
# Für vollen Lauf auskommentieren/auf None setzen
USER_CONFIG_TEST_PAIRS_SUBSET = None
#USER_CONFIG_TEST_PAIRS_SUBSET = ['BTC/USDC', 'ETH/USDC', 'SOL/USDC'] # Beispiel für Test
# ================================================================
# === ENDE: BENUTZERKONFIGURATION (EXIT - Simplified Penalties) ===
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
    # Stelle sicher, dass der *korrigierte* Provider importiert wird
    from strategies.includes.mta_feature_engineering import MTAFeatureProvider
    logger.info(f"MTAFeatureProvider successfully imported from '{provider_module_path or 'strategies.includes'}.mta_feature_engineering'.")
except Exception as e: logger.critical(f"FEHLER beim Import von MTAFeatureProvider: {e}"); logger.error(traceback.format_exc()); sys.exit(1)
if MTAFeatureProvider is None: logger.critical("Import von MTAFeatureProvider fehlgeschlagen."); sys.exit(1)

# --- Hilfsfunktionen (parse_timerange, load_data_mta, create_exit_target, objective) ---
# Diese Funktionen bleiben unverändert zur letzten Version
def parse_timerange(timerange_str):
    # (Code unverändert)
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
    # (Code unverändert)
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
    """ Erstellt die Zielvariable 'target' für Exits mit optionalen Penalties. (Unverändert) """
    # (Code unverändert zur letzten Version)
    target_col = 'target'
    if target_col in df.columns: df = df.drop(columns=[target_col])
    future_close = df['close'].shift(-periods)
    base_target_condition = pd.Series(False, index=df.index)
    if vola_threshold_mult > 0:
        logger.debug(f"Exit Target: Preis < Close - {vola_threshold_mult} * ATR(14) in {periods} Perioden.")
        if 'feature_atr_14' not in df.columns: raise ValueError("ATR ('feature_atr_14') fehlt für vola-adjustiertes Exit-Target.")
        safe_atr = df['feature_atr_14'].fillna(0)
        base_target_condition = (future_close < df['close'] - vola_threshold_mult * safe_atr)
    elif threshold > 0:
        logger.debug(f"Exit Target: Preis < Close * (1 - {threshold:.4f}) in {periods} Perioden.")
        base_target_condition = (future_close / df['close'].replace(0, np.nan) - 1) < -threshold
    else: logger.warning("Kein gültiges Exit-Target definiert.")
    base_target_condition = base_target_condition.fillna(False)
    initial_exit_count = base_target_condition.sum()
    if penalty_enable:
        logger.debug("Wende VEREINFACHTE EXIT Penalty-Bedingungen an...")
        penalty = pd.Series(False, index=df.index)
        if penalty_oversold_enable:
            cond_oversold = pd.Series(False, index=df.index); rsi_col = 'feature_rsi_14'; dist_low_col = 'feature_dist_from_low_50_pct'
            if rsi_col in df.columns: cond_oversold = cond_oversold | (df[rsi_col] < rsi_penalty_low_thr).fillna(False); logger.debug(f"  - Oversold Check: {rsi_col} < {rsi_penalty_low_thr}?")
            else: logger.warning(f"Exit Penalty-Feature {rsi_col} nicht gefunden.")
            if dist_low_col in df.columns: cond_oversold = cond_oversold | (df[dist_low_col] < dist_low_50_pct_penalty_low_thr).fillna(False); logger.debug(f"  - Oversold Check: {dist_low_col} < {dist_low_50_pct_penalty_low_thr}%?")
            else: logger.warning(f"Exit Penalty-Feature {dist_low_col} nicht gefunden.")
            penalty = penalty | cond_oversold
            logger.debug(f"  - Penalty (prevent exit) if OVERSOLD condition is TRUE.")
        if penalty_htf_strong_uptrend_enable:
             htf_suffix = f'_{htf_tf_name}_inf'; htf_rsi_col = f'feature_{htf_tf_name}_rsi_14{htf_suffix}'; htf_ma50_col = f'feature_{htf_tf_name}_ma_50{htf_suffix}'; htf_ma200_col = f'feature_{htf_tf_name}_ma_200{htf_suffix}'
             required_htf_cols = [htf_rsi_col];
             if htf_ma_trend_check: required_htf_cols.extend([htf_ma50_col, htf_ma200_col])
             if all(col in df.columns for col in required_htf_cols):
                 cond_htf_rsi = (df[htf_rsi_col] > htf_rsi_penalty_high_thr).fillna(False); cond_htf_ma = True
                 if htf_ma_trend_check: cond_htf_ma = (df[htf_ma50_col] > df[htf_ma200_col]).fillna(False)
                 cond_htf_strong_up = cond_htf_rsi & cond_htf_ma; penalty = penalty | cond_htf_strong_up
                 log_msg_htf = f"  - Penalty (prevent exit) if HTF Trend strong UP ({htf_rsi_col} > {htf_rsi_penalty_high_thr}" + (f" AND {htf_ma50_col} > {htf_ma200_col}" if htf_ma_trend_check else "") + ")"; logger.debug(log_msg_htf)
             else: logger.warning(f"HTF Trend Penalty nicht anwendbar, fehlende Features: {[col for col in required_htf_cols if col not in df.columns]}")
        final_target_condition = base_target_condition & (~penalty)
        num_penalized = (base_target_condition & penalty).sum(); logger.info(f"Exit Penalties: {num_penalized} von {int(initial_exit_count)} ({num_penalized/max(1, initial_exit_count)*100:.2f}%) pot. Exit-Signalen unterdrückt.")
    else: final_target_condition = base_target_condition
    df[target_col] = final_target_condition.astype(int)
    final_exit_count = df[target_col].sum(); logger.info(f"Exit Target erstellt: {final_exit_count} Exit-Signale (1) von {len(df)} ({final_exit_count/len(df)*100:.2f}%) nach Penalties.")
    return df

def objective(trial, X_train, y_train, X_test, y_test, pair_name):
    """ Optuna objective function (unverändert) """
    # (Code unverändert zur letzten Version)
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
        if eval_set_hpo is None: logger.warning(f"HPO Trial {pair_name}: Nur eine Klasse in y_test."); return 0.0
        model.fit(X_train, y_train, eval_set=eval_set_hpo, eval_metric='auc', callbacks=[early_stopping_callback_hpo, log_evaluation_callback_hpo])
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba); return auc
    except ValueError as e: logger.warning(f"HPO Trial {pair_name} ValueError: {e}. Gebe 0.0 zurück."); return 0.0
    except Exception as e: logger.warning(f"HPO Trial {pair_name} Fehler: {e}"); return 0.0

# --- Hauptfunktion ---
def main():
    logger.info("--- Start: MTA EXIT Modell-Trainingsskript (KORRIGIERT) ---")
    # --- Konfiguration loggen (unverändert) ---
    logger.info("--- Verwendete EXIT Konfiguration (Simplified Penalties) ---")
    # ... (alle logger.info Aufrufe für Konfig) ...
    logger.info(f"  User Data Dir:       {user_data_dir}")
    logger.info(f"  Daten Verzeichnis:   {data_directory}")
    # ... (Rest des Config Loggings) ...
    logger.info("-----------------------------")


    # --- Feature Provider & Timerange (unverändert) ---
    try: feature_provider = MTAFeatureProvider(base_tf=USER_CONFIG_BASE_TIMEFRAME, higher_tf=USER_CONFIG_HIGHER_TIMEFRAME)
    except Exception as e: logger.critical(f"Fehler Init Provider: {e}"); return
    try: start_dt, end_dt = parse_timerange(USER_CONFIG_TIMERANGE)
    except ValueError as e: logger.critical(f"Fehler Timerange: {e}"); return

    # --- Daten laden (unverändert) ---
    try:
        search_pattern_base = os.path.join(data_directory, f'*-{USER_CONFIG_BASE_TIMEFRAME}.feather')
        all_potential_pairs = sorted([os.path.basename(f).replace(f'-{USER_CONFIG_BASE_TIMEFRAME}.feather', '').replace('_', '/') for f in glob.glob(search_pattern_base)])
        if not all_potential_pairs: raise FileNotFoundError(f"Keine '{USER_CONFIG_BASE_TIMEFRAME}' Feather-Dateien in '{data_directory}'.")
        logger.info(f"{len(all_potential_pairs)} mögliche Paare gefunden.")
        mta_data_dict, common_pairs_list, actual_timeframes = load_data_mta(data_directory, all_potential_pairs, [USER_CONFIG_BASE_TIMEFRAME, USER_CONFIG_HIGHER_TIMEFRAME], start_dt, end_dt, USER_CONFIG_TIMERANGE)
        if USER_CONFIG_BASE_TIMEFRAME not in actual_timeframes or USER_CONFIG_HIGHER_TIMEFRAME not in actual_timeframes:
            missing = [tf for tf in [USER_CONFIG_BASE_TIMEFRAME, USER_CONFIG_HIGHER_TIMEFRAME] if tf not in actual_timeframes]; logger.critical(f"FEHLER: TFs {missing} nicht geladen."); return
        if not common_pairs_list: logger.critical(f"FEHLER: Keine gemeinsamen Paare."); return
    except Exception as e: logger.critical(f"Fehler Daten laden: {e}"); logger.error(traceback.format_exc()); return

    # --- Phase 1: HPO (falls aktiviert) ---
    best_params_glob = USER_CONFIG_FIXED_PARAMS.copy()
    if USER_CONFIG_ENABLE_HPO:
        logger.info("--- Starte HPO Phase (EXIT Model) ---")
        hpo_pair = USER_CONFIG_HPO_PAIR if USER_CONFIG_HPO_PAIR and USER_CONFIG_HPO_PAIR in common_pairs_list else common_pairs_list[0]
        if USER_CONFIG_HPO_PAIR and USER_CONFIG_HPO_PAIR not in common_pairs_list: logger.warning(f"HPO Paar '{USER_CONFIG_HPO_PAIR}' nicht gefunden. Verwende: {hpo_pair}")
        logger.info(f"Führe HPO für Paar '{hpo_pair}' durch ({USER_CONFIG_HPO_TRIALS} Trials)...")
        try:
            base_df_hpo = mta_data_dict[USER_CONFIG_BASE_TIMEFRAME][hpo_pair].copy()
            htf_df_hpo = mta_data_dict[USER_CONFIG_HIGHER_TIMEFRAME][hpo_pair].copy()
            df_features_hpo = feature_provider.add_all_features(base_df_hpo, htf_df_hpo)
            logger.info(f"[HPO-{hpo_pair}] Erstelle EXIT Target...")
            df_final_hpo = create_exit_target( df_features_hpo, periods=USER_CONFIG_LOOKAHEAD_PERIODS, threshold=USER_CONFIG_EXIT_TARGET_THRESHOLD if USER_CONFIG_EXIT_TARGET_VOLA_MULT <= 0 else 0, vola_threshold_mult=USER_CONFIG_EXIT_TARGET_VOLA_MULT, penalty_enable=USER_CONFIG_EXIT_PENALTY_ENABLE, penalty_oversold_enable=USER_CONFIG_PENALTY_OVERSOLD_ENABLE, rsi_penalty_low_thr=USER_CONFIG_PENALTY_RSI_LOW_THRESHOLD, dist_low_50_pct_penalty_low_thr=USER_CONFIG_PENALTY_DIST_LOW_50_PCT_LOW_THRESHOLD, penalty_htf_strong_uptrend_enable=USER_CONFIG_PENALTY_HTF_STRONG_UPTREND_ENABLE, htf_rsi_penalty_high_thr=USER_CONFIG_PENALTY_HTF_RSI_HIGH_THRESHOLD, htf_ma_trend_check=USER_CONFIG_PENALTY_HTF_MA_TREND_CHECK, htf_tf_name=USER_CONFIG_HIGHER_TIMEFRAME)

            feature_columns_hpo = sorted([col for col in df_final_hpo.columns if col.startswith('feature_')]); target_col_hpo = 'target'
            if not feature_columns_hpo or target_col_hpo not in df_final_hpo.columns: raise ValueError(f"Keine Features/Target für HPO.")

            df_ml_hpo = df_final_hpo[feature_columns_hpo + [target_col_hpo]].copy()
            for col in feature_columns_hpo: df_ml_hpo[col] = pd.to_numeric(df_ml_hpo[col], errors='coerce')
            # === KORRIGIERT: NaN/Inf Handling VOR dropna ===
            df_ml_hpo.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_ml_hpo[feature_columns_hpo] = df_ml_hpo[feature_columns_hpo].ffill() # Nur ffill
            # === Ende Korrektur ===
            df_ml_hpo.dropna(subset=feature_columns_hpo + [target_col_hpo], inplace=True)

            if len(df_ml_hpo) < USER_CONFIG_MIN_SAMPLES_PER_PAIR: raise ValueError(f"Zu wenig Daten ({len(df_ml_hpo)}) für HPO.")
            if len(df_ml_hpo[target_col_hpo].value_counts()) < 2: raise ValueError(f"Nur eine Target-Klasse für HPO.")
            logger.info(f"[HPO-{hpo_pair}] EXIT Target-Verteilung: \n{df_ml_hpo[target_col_hpo].value_counts(normalize=True).round(4).to_string()}")

            X_hpo = df_ml_hpo[feature_columns_hpo]; y_hpo = df_ml_hpo[target_col_hpo]
            tscv_hpo = TimeSeriesSplit(n_splits=USER_CONFIG_N_SPLITS); train_idx_hpo, test_idx_hpo = list(tscv_hpo.split(X_hpo))[-1]
            if len(train_idx_hpo) == 0 or len(test_idx_hpo) == 0: raise ValueError(f"Leere Splits für HPO.")
            X_train_hpo, X_test_hpo = X_hpo.iloc[train_idx_hpo], X_hpo.iloc[test_idx_hpo]; y_train_hpo, y_test_hpo = y_hpo.iloc[train_idx_hpo], y_hpo.iloc[test_idx_hpo]
            if X_train_hpo.empty or X_test_hpo.empty or len(y_train_hpo.value_counts()) < 2 or len(y_test_hpo.value_counts()) < 2: raise ValueError(f"Leere/einklassige Sets für HPO.")

            study = optuna.create_study(direction="maximize", study_name=f"lgbm_exit_hpo_{hpo_pair.replace('/','_')}_{USER_CONFIG_MODEL_PREFIX}")
            study.optimize(lambda trial: objective(trial, X_train_hpo, y_train_hpo, X_test_hpo, y_test_hpo, hpo_pair), n_trials=USER_CONFIG_HPO_TRIALS, show_progress_bar=True)
            best_params_glob = study.best_params; logger.info(f"--- HPO Abgeschlossen für {hpo_pair} ---"); logger.info(f"Bester AUC Score: {study.best_value:.6f}"); logger.info(f"Beste Parameter: {best_params_glob}")
        except Exception as e_hpo: logger.error(f"FEHLER HPO {hpo_pair}: {e_hpo}"); logger.error(traceback.format_exc()); logger.warning("Fahre mit festen Params fort."); best_params_glob = USER_CONFIG_FIXED_PARAMS.copy()
    else: logger.info("--- HPO Phase übersprungen ---"); logger.info(f"Verwende feste Parameter: {best_params_glob}")

    # --- Phase 2: Finales Training (für EXITS) ---
    logger.info("--- Starte Finale EXIT Trainingsphase ---")
    pairs_to_train = common_pairs_list
    if USER_CONFIG_TEST_PAIRS_SUBSET:
        # ... (Subset Handling unverändert) ...
        original_count = len(common_pairs_list); pairs_to_train = sorted([p for p in common_pairs_list if p in USER_CONFIG_TEST_PAIRS_SUBSET]); filtered_count = len(pairs_to_train)
        if not pairs_to_train: logger.critical(f"FEHLER: Keine Testpaare {USER_CONFIG_TEST_PAIRS_SUBSET} gefunden."); return
        logger.warning(f"!!! ACHTUNG: Testlauf! Nur {filtered_count}/{original_count} Paare (EXIT): {pairs_to_train}")
    if not pairs_to_train: logger.critical("FEHLER: Keine Paare zum Trainieren."); return

    processed_pairs = 0; skipped_pairs = 0; all_results = {}
    for pair_name in pairs_to_train:
        logger.info(f"===== Verarbeite Paar für EXIT: {pair_name} =====")
        try:
            # 1. Daten & Features (unverändert)
            if pair_name not in mta_data_dict[USER_CONFIG_BASE_TIMEFRAME] or pair_name not in mta_data_dict[USER_CONFIG_HIGHER_TIMEFRAME]: logger.warning(f"[{pair_name}] Daten fehlen."); skipped_pairs += 1; continue
            base_df_pair = mta_data_dict[USER_CONFIG_BASE_TIMEFRAME][pair_name].copy(); htf_df_pair = mta_data_dict[USER_CONFIG_HIGHER_TIMEFRAME][pair_name].copy()
            if base_df_pair.empty or htf_df_pair.empty: logger.warning(f"[{pair_name}] Leere DFs."); skipped_pairs += 1; continue
            df_features = feature_provider.add_all_features(base_df_pair, htf_df_pair)

            # Target-Aufruf (unverändert)
            logger.info(f"[{pair_name}] Erstelle EXIT Target...")
            df_final = create_exit_target( df_features, periods=USER_CONFIG_LOOKAHEAD_PERIODS, threshold=USER_CONFIG_EXIT_TARGET_THRESHOLD if USER_CONFIG_EXIT_TARGET_VOLA_MULT <= 0 else 0, vola_threshold_mult=USER_CONFIG_EXIT_TARGET_VOLA_MULT, penalty_enable=USER_CONFIG_EXIT_PENALTY_ENABLE, penalty_oversold_enable=USER_CONFIG_PENALTY_OVERSOLD_ENABLE, rsi_penalty_low_thr=USER_CONFIG_PENALTY_RSI_LOW_THRESHOLD, dist_low_50_pct_penalty_low_thr=USER_CONFIG_PENALTY_DIST_LOW_50_PCT_LOW_THRESHOLD, penalty_htf_strong_uptrend_enable=USER_CONFIG_PENALTY_HTF_STRONG_UPTREND_ENABLE, htf_rsi_penalty_high_thr=USER_CONFIG_PENALTY_HTF_RSI_HIGH_THRESHOLD, htf_ma_trend_check=USER_CONFIG_PENALTY_HTF_MA_TREND_CHECK, htf_tf_name=USER_CONFIG_HIGHER_TIMEFRAME)

            # 2. ML Vorbereitung & Bereinigung (KORRIGIERT)
            feature_columns = sorted([col for col in df_final.columns if col.startswith('feature_')]); target_col = 'target'
            if not feature_columns or target_col not in df_final.columns: raise ValueError("Keine Features/Target nach Engineering.")
            df_ml = df_final[feature_columns + [target_col]].copy()
            for col in feature_columns: df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
            # === KORRIGIERT: NaN/Inf Handling VOR dropna ===
            df_ml.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_ml[feature_columns] = df_ml[feature_columns].ffill() # Nur ffill
            remaining_nans = df_ml[feature_columns].isnull().sum().sum()
            if remaining_nans > 0: logger.warning(f"[{pair_name}] {remaining_nans} NaNs verbleiben nach ffill (werden durch dropna entfernt).")
            # === Ende Korrektur ===
            initial_rows = len(df_ml); df_ml.dropna(subset=feature_columns + [target_col], inplace=True); final_rows = len(df_ml)
            logger.info(f"[{pair_name}] Daten nach dropna: {final_rows} (von {initial_rows})")
            if initial_rows > 0 and final_rows == 0: logger.error(f"!!! [{pair_name}] Alle Zeilen durch dropna entfernt !!!"); raise ValueError("Alle Daten durch dropna entfernt.")
            if final_rows < USER_CONFIG_MIN_SAMPLES_PER_PAIR: raise ValueError(f"Zu wenige Daten ({final_rows}) nach Bereinigung (Min: {USER_CONFIG_MIN_SAMPLES_PER_PAIR}).")
            if len(df_ml[target_col].value_counts()) < 2: raise ValueError(f"Nur eine Target-Klasse nach Bereinigung. Verteilung: {df_ml[target_col].value_counts()}")

            X = df_ml[feature_columns]; y = df_ml[target_col]
            logger.info(f"[{pair_name}] Daten für ML: {len(X)} Zeilen"); logger.info(f"[{pair_name}] EXIT Target-Verteilung: \n{y.value_counts(normalize=True).round(4).to_string()}")

            # 3. Daten aufteilen (unverändert)
            tscv = TimeSeriesSplit(n_splits=USER_CONFIG_N_SPLITS)
            try: train_indices, test_indices = list(tscv.split(X))[-1]
            except Exception as e_split: raise ValueError(f"Fehler bei TimeSeriesSplit: {e_split}")
            if len(train_indices) == 0 or len(test_indices) == 0: raise ValueError("Leere Splits.")
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]; y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            if X_train.empty or X_test.empty or len(y_train.value_counts()) < 2: raise ValueError("Leere/einklassige Train/Test-Sets.")
            logger.info(f"[{pair_name}] Train: {len(X_train)} Zeilen, Test: {len(X_test)} Zeilen")

            # 4. Modell trainieren (unverändert)
            final_params = best_params_glob.copy(); final_params['objective'] = 'binary'; final_params['metric'] = 'auc'; final_params['is_unbalance'] = True
            final_params['n_estimators'] = USER_CONFIG_N_ESTIMATORS_FINAL; final_params['n_jobs'] = -1; final_params['random_state'] = 42; final_params['verbose'] = -1
            model = lgb.LGBMClassifier(**final_params)
            early_stopping_callback_final = lgb.early_stopping(stopping_rounds=USER_CONFIG_EARLY_STOPPING_ROUNDS_FINAL, verbose=False)
            log_evaluation_callback_final = lgb.log_evaluation(period=500)
            logger.info(f"[{pair_name}] Starte finales EXIT-Training...")
            eval_set = [(X_test, y_test)] if len(y_test.value_counts()) >= 2 else None
            if eval_set is None: logger.warning(f"[{pair_name}] Nur eine Klasse in y_test. Early Stopping nur auf Training.")
            if X_train.isnull().values.any() or np.isinf(X_train.values).any(): raise ValueError("NaN/Inf in X_train VOR fit trotz Bereinigung.")
            if y_train.isnull().values.any(): raise ValueError("NaN in y_train VOR fit trotz Bereinigung.")
            model.fit(X_train, y_train, eval_set=eval_set, eval_metric='auc', callbacks=[early_stopping_callback_final, log_evaluation_callback_final])
            logger.info(f"[{pair_name}] Training beendet. Beste Iteration: {model.best_iteration_}")

            # 5. Modell evaluieren (unverändert)
            logger.info(f"[{pair_name}] Evaluiere EXIT-Modell...")
            report_dict = {}; accuracy = np.nan; best_f1_exit = np.nan; pred_threshold = 0.5; report_str = "Eval übersprungen."; achieved_precision_exit = np.nan; achieved_recall_exit = np.nan
            if y_test.empty or len(y_test.value_counts()) < 2: logger.warning(f"[{pair_name}] Testset ungültig."); report_dict = {'error': 'Invalid test set'}; pred_threshold = np.nan
            else:
                try:
                    best_iter = model.best_iteration_ if model.best_iteration_ and model.best_iteration_ > 0 else -1
                    X_test_eval = X_test[X_train.columns];
                    if X_test_eval.isnull().values.any() or np.isinf(X_test_eval.values).any(): raise ValueError("NaN/Inf in X_test VOR predict.")
                    y_pred_proba = model.predict_proba(X_test_eval, num_iteration=best_iter)[:, 1]
                    if np.isnan(y_pred_proba).all(): raise ValueError("predict_proba nur NaNs.")

                    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
                    valid_pr_indices = ~np.isnan(precision) & ~np.isnan(recall); precision_valid = precision[valid_pr_indices]; recall_valid = recall[valid_pr_indices]
                    thresholds_valid = thresholds[np.where(valid_pr_indices)[0][:-1]] if len(thresholds) > 0 and np.sum(valid_pr_indices) > 1 else np.array([])
                    if len(precision_valid) <= 1 or len(thresholds_valid) == 0: raise ValueError("Zu wenige gültige Werte für Threshold-Findung.")

                    min_precision_target = USER_CONFIG_EXIT_MIN_PRECISION_TARGET; logger.info(f"Suche Threshold für Precision (Exit) >= {min_precision_target:.2f}...")
                    f1_scores = np.divide(2*recall_valid[:-1]*precision_valid[:-1], recall_valid[:-1]+precision_valid[:-1], out=np.zeros_like(recall_valid[:-1]), where=(recall_valid[:-1]+precision_valid[:-1])!=0)
                    f1_scores[np.isnan(f1_scores)] = 0
                    valid_precision_indices = np.where(precision_valid[:-1] >= min_precision_target)[0]

                    if len(valid_precision_indices) > 0:
                         valid_f1_scores = f1_scores[valid_precision_indices]
                         if np.max(valid_f1_scores) > 0:
                             ix_relative = np.argmax(valid_f1_scores); ix = valid_precision_indices[ix_relative]
                             if ix < len(thresholds_valid):
                                 pred_threshold = thresholds_valid[ix]; best_f1_exit = f1_scores[ix]; achieved_precision_exit=precision_valid[ix]; achieved_recall_exit=recall_valid[ix];
                                 logger.info(f"[{pair_name}] Bester EXIT Thresh (P>={min_precision_target:.2f}): {pred_threshold:.4f} -> F1={best_f1_exit:.4f} (P={achieved_precision_exit:.4f}, R={achieved_recall_exit:.4f})")
                             else: raise IndexError("Thresh-Index OOB.")
                         else: # Fallback Max Precision
                             logger.warning(f"[{pair_name}] Kein F1>0 für P>={min_precision_target:.2f}. Fallback: Max Precision.")
                             ix = np.argmax(precision_valid[:-1])
                             if ix < len(thresholds_valid): pred_threshold = thresholds_valid[ix]; best_f1_exit = f1_scores[ix]; achieved_precision_exit=precision_valid[ix]; achieved_recall_exit=recall_valid[ix]; logger.info(f"[{pair_name}] Bester Thresh (FALLBACK Max Precision): {pred_threshold:.4f} -> F1={best_f1_exit:.4f} (P={achieved_precision_exit:.4f}, R={achieved_recall_exit:.4f})")
                             else: logger.warning(f"[{pair_name}] Fallback Max P Index OOB. Nutze 0.5."); pred_threshold=0.5; best_f1_exit=0.0; achieved_precision_exit=np.nan; achieved_recall_exit=np.nan
                    else: # Fallback Max Precision
                         logger.warning(f"[{pair_name}] Kein Thresh mit P>={min_precision_target:.2f}. Fallback: Max Precision.")
                         ix = np.argmax(precision_valid[:-1])
                         if ix < len(thresholds_valid): pred_threshold = thresholds_valid[ix]; best_f1_exit = f1_scores[ix]; achieved_precision_exit=precision_valid[ix]; achieved_recall_exit=recall_valid[ix]; logger.info(f"[{pair_name}] Bester Thresh (FALLBACK Max Precision): {pred_threshold:.4f} -> F1={best_f1_exit:.4f} (P={achieved_precision_exit:.4f}, R={achieved_recall_exit:.4f})")
                         else: logger.warning(f"[{pair_name}] Fallback Max P Index OOB. Nutze 0.5."); pred_threshold=0.5; best_f1_exit=0.0; achieved_precision_exit=np.nan; achieved_recall_exit=np.nan

                    y_pred_class = (y_pred_proba > pred_threshold).astype(int); accuracy = accuracy_score(y_test, y_pred_class)
                    logger.info(f"[{pair_name}] Accuracy (@Thresh {pred_threshold:.4f}): {accuracy:.4f}")
                    target_names_exit = ['Hold', 'Exit (Drop)']; report_str = classification_report(y_test, y_pred_class, target_names=target_names_exit, zero_division=0)
                    logger.info(f"[{pair_name}] Classification Report (Exit):\n{report_str}")
                    report_dict = classification_report(y_test, y_pred_class, target_names=target_names_exit, zero_division=0, output_dict=True)
                    report_dict['used_threshold'] = pred_threshold; report_dict['best_f1_for_exit_class'] = best_f1_exit; report_dict['accuracy'] = accuracy
                    report_dict['precision_for_exit_class'] = achieved_precision_exit; report_dict['recall_for_exit_class'] = achieved_recall_exit
                except Exception as e_eval: logger.error(f"[{pair_name}] Fehler Evaluation: {e_eval}"); logger.error(traceback.format_exc()); report_str = f"Evaluation fehlgeschlagen: {e_eval}"; report_dict = {'error': str(e_eval)}; pred_threshold = np.nan

            # Ergebnisse & Feature Importance (unverändert)
            result_summary = report_dict.copy(); result_summary['pair'] = pair_name; result_summary['threshold'] = pred_threshold
            all_results[pair_name] = result_summary
            if hasattr(model, 'feature_importances_') and 'error' not in report_dict:
                try:
                    importances = model.feature_importances_
                    if importances is not None and len(importances) == len(X_train.columns): logger.info(f"[{pair_name}] Top 10 Features (Exit Model):"); feature_imp = pd.DataFrame({'Value': importances, 'Feature': X_train.columns}); logger.info("\n" + feature_imp.nlargest(10, "Value").to_string(index=False))
                    else: logger.warning(f"[{pair_name}] Feature Importances nicht ok.")
                except Exception as fi_e: logger.error(f"Fehler Feature Importance: {fi_e}")

            # 7. Modell & Features speichern (unverändert)
            if 'error' not in report_dict:
                os.makedirs(save_directory, exist_ok=True); safe_pair_name = pair_name.replace('/', '_'); base_tf_safe = USER_CONFIG_BASE_TIMEFRAME.replace('/', '_'); higher_tf_safe = USER_CONFIG_HIGHER_TIMEFRAME.replace('/', '_')
                base_filename = f"{USER_CONFIG_MODEL_PREFIX}_{safe_pair_name}_{base_tf_safe}_WITH_{higher_tf_safe}"
                model_filepath = os.path.join(save_directory, f"{base_filename}.joblib"); features_filepath = os.path.join(save_directory, f"{base_filename}_features.json")
                logger.info(f"[{pair_name}] Speichere EXIT-Modell: {model_filepath}"); joblib.dump(model, model_filepath)
                feature_list_to_save = list(X_train.columns); logger.info(f"[{pair_name}] Speichere Features ({len(feature_list_to_save)}): {features_filepath}")
                try:
                    with open(features_filepath, 'w') as f: json.dump(feature_list_to_save, f, indent=4); logger.info(f"[{pair_name}] EXIT-Modell/Features gespeichert.")
                except Exception as e_save: logger.error(f"[{pair_name}] Fehler Speichern Features: {e_save}")
            else: logger.warning(f"[{pair_name}] EXIT-Modell/Features NICHT gespeichert wegen Fehler.")
            processed_pairs += 1

        except ValueError as e: logger.error(f"Überspringe Paar {pair_name} wegen ValueError: {e}."); skipped_pairs += 1
        except KeyError as e: logger.error(f"Überspringe Paar {pair_name} wegen KeyError: {e}."); skipped_pairs += 1
        except MemoryError as e: logger.error(f"Überspringe Paar {pair_name} wegen MemoryError: {e}."); skipped_pairs += 1
        except Exception as e: logger.error(f"Unerwarteter Fehler bei {pair_name}: {e}."); logger.error(traceback.format_exc()); skipped_pairs += 1
        finally: logger.info(f"===== Paar {pair_name} (EXIT) abgeschlossen =====")
    # --- Ende for-Schleife ---

    # --- Ergebnisübersicht (unverändert) ---
    logger.info("\n--- EXIT Training abgeschlossen ---")
    logger.info(f"Erfolgreich: {processed_pairs} Paare."); logger.info(f"Übersprungen: {skipped_pairs} Paare.")
    logger.info("\n--- EXIT Ergebnisübersicht (Precision & F1 der 'Exit (Drop)'-Klasse @ Threshold) ---")
    summary_data = []
    for pair, report in all_results.items():
        f1_exit = np.nan; prec_exit = np.nan; rec_exit = np.nan; th = report.get('threshold', np.nan)
        if isinstance(report, dict) and 'error' not in report:
            f1_exit = report.get('best_f1_for_exit_class', np.nan)
            prec_exit = report.get('precision_for_exit_class', np.nan)
            rec_exit = report.get('recall_for_exit_class', np.nan)
        summary_data.append({'Pair': pair, 'Threshold': th, 'Prec_Exit': prec_exit if pd.notna(prec_exit) else np.nan, 'Recall_Exit': rec_exit if pd.notna(rec_exit) else np.nan, 'F1_Exit': f1_exit if pd.notna(f1_exit) else np.nan})
    if summary_data:
        summary_df = pd.DataFrame(summary_data); summary_df = summary_df.sort_values(by=['Prec_Exit', 'F1_Exit'], ascending=[False, False], na_position='last')
        summary_df = summary_df[['Pair', 'Threshold', 'Prec_Exit', 'Recall_Exit', 'F1_Exit']]
        logger.info("EXIT Ergebnis Zusammenfassung:\n" + summary_df.to_string(index=False, float_format="%.4f"))
    else: logger.info("Keine Ergebnisse für Zusammenfassung.")

    # --- Thresholds speichern (unverändert) ---
    if summary_data:
        threshold_filepath = os.path.join(save_directory, f"{USER_CONFIG_MODEL_PREFIX}_thresholds.json")
        logger.info(f"Speichere EXIT-Thresholds nach {threshold_filepath}...")
        threshold_dict = {item['Pair']: item['Threshold'] for item in summary_data if 'Pair' in item and 'Threshold' in item and pd.notna(item['Threshold'])}
        try:
            with open(threshold_filepath, 'w') as f: json.dump(threshold_dict, f, indent=4, sort_keys=True); logger.info(f"EXIT-Thresholds für {len(threshold_dict)} Paare gespeichert.")
        except Exception as e: logger.error(f"Fehler Speichern EXIT-Thresholds: {e}"); logger.error(traceback.format_exc())
    else: logger.warning("Keine Ergebnisse, keine EXIT-Threshold-Datei gespeichert.")

# --- Skriptstart Schutz ---
if __name__ == "__main__":
    logger.info("exittrain.py wird als Hauptskript ausgeführt.")
    main()
else:
    logger.warning("exittrain.py wird als Modul importiert (unerwartet).")

# --- END OF FILE exittrain.py (KORRIGIERT & Simplified Penalties) ---
