# --- START OF FILE entrytrainglobal.py (Vorschlag für finales Training) ---
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
logger = logging.getLogger("entry_train_global_script")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ================================================================
# === START: BENUTZERKONFIGURATION (ENTRY - GLOBAL - FINAL) ===
# ================================================================
# --- Pfade ---
USER_CONFIG_USER_DATA_DIR = '/home/tommi/freqtrade/user_data'
USER_CONFIG_DATA_DIR_SUFFIX = 'data/binance'
USER_CONFIG_MODELS_DIR_SUFFIX = 'models'
USER_CONFIG_PROVIDER_DIR_SUFFIX = 'strategies/includes'

# --- Zeitrahmen ---
USER_CONFIG_BASE_TIMEFRAME = '5m' # Deine Basis-TF
USER_CONFIG_HIGHER_TIMEFRAME = '1h' # Deine HTF

# --- Trainingsdaten-Filter ---
# <<< Langer Zeitraum, wie beim Exit-Modell >>>
USER_CONFIG_TIMERANGE = '20240201-20250410'

# --- Zielvariable & Daten ---
# <<< Angepasst basierend auf dem erfolgreichen Lauf von 19:15 >>>
USER_CONFIG_LOOKAHEAD_PERIODS = 6  # 3 Stunden (wie im Lauf von 19:15)
USER_CONFIG_TARGET_VOLA_MULT = 0.0 # Vola aus
USER_CONFIG_TARGET_THRESHOLD = 0.005 # 1% Anstieg (wie im Lauf von 19:15, lieferte ~24% Targets)
USER_CONFIG_N_SPLITS = 5
USER_CONFIG_MIN_SAMPLES_COMBINED = 10000

# --- Penalty Konfiguration für create_target ---
# <<< Penalties wieder aktivieren, um überkaufte Situationen etc. zu filtern >>>
USER_CONFIG_PENALTY_ENABLE = False
USER_CONFIG_PENALTY_RSI_THRESHOLD = 85.0
USER_CONFIG_PENALTY_STOCHRSI_K_THRESHOLD = 90.0
USER_CONFIG_PENALTY_DIST_LOW_50_PCT_THRESHOLD = 15.0 # % Abstand vom 50er Tief
USER_CONFIG_PENALTY_BB_PCT_B_THRESHOLD = 1.15 # Über oberem BBand

# --- HPO (Optuna) Konfiguration ---
USER_CONFIG_ENABLE_HPO = False # Anlassen, um optimale Parameter für diesen Datensatz zu finden
USER_CONFIG_HPO_TRIALS = 15   # Oder 30 für schnelleren Lauf, wenn HPO lange dauert

# --- Feste Modellparameter (Fallback, falls HPO fehlschlägt) ---
# (Kannst du lassen oder anpassen, HPO sollte aber funktionieren)
USER_CONFIG_FIXED_PARAMS = {
    'learning_rate': 0.024594082887298703,
    'num_leaves': 55,
    'max_depth': 12,
    'reg_alpha': 0.06688234715477308,
    'reg_lambda': 0.0001898428536491572,
    'min_child_samples': 26,
    'subsample': 0.9554178198767475,
    'colsample_bytree': 0.9059501760903735
}
# --- Training Konfiguration ---
# <<< Finaler Prefix für das Entry-Modell (ohne _test) >>>
USER_CONFIG_MODEL_PREFIX = 'mta_model_entry2_glob'
USER_CONFIG_N_ESTIMATORS_FINAL = 3000
USER_CONFIG_EARLY_STOPPING_ROUNDS_FINAL = 50
USER_CONFIG_MIN_PRECISION_TARGET = 0.50 # Dein ursprüngliches Ziel für Entry

# --- Test Subset ---
USER_CONFIG_TEST_PAIRS_SUBSET = None # Alle Paare verwenden
# ================================================================
# === ENDE: BENUTZERKONFIGURATION (ENTRY - GLOBAL) ===
# ================================================================

# --- Abgeleitete Pfade, Provider Import, Hilfsfunktionen ---
# (Code unverändert)
user_data_dir = os.path.abspath(USER_CONFIG_USER_DATA_DIR)
data_directory = os.path.join(user_data_dir, USER_CONFIG_DATA_DIR_SUFFIX)
save_directory = os.path.join(user_data_dir, USER_CONFIG_MODELS_DIR_SUFFIX)
provider_base_dir = os.path.join(user_data_dir, os.path.dirname(USER_CONFIG_PROVIDER_DIR_SUFFIX)) if USER_CONFIG_PROVIDER_DIR_SUFFIX else user_data_dir
provider_module_path = USER_CONFIG_PROVIDER_DIR_SUFFIX.replace('/', '.') if USER_CONFIG_PROVIDER_DIR_SUFFIX else ''
MTAFeatureProvider = None
try:
    import_base_path = user_data_dir
    if import_base_path not in sys.path: sys.path.insert(0, import_base_path); logger.info(f"Added '{import_base_path}' to sys.path.")
    from strategies.includes.mta_feature_engineering import MTAFeatureProvider
    logger.info(f"MTAFeatureProvider successfully imported from '{provider_module_path or 'strategies.includes'}.mta_feature_engineering'.")
except Exception as e: logger.critical(f"FEHLER beim Import von MTAFeatureProvider: {e}"); logger.error(traceback.format_exc()); sys.exit(1)
if MTAFeatureProvider is None: logger.critical("Import von MTAFeatureProvider fehlgeschlagen."); sys.exit(1)

def parse_timerange(timerange_str):
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

def create_target(df, periods=5, threshold=0.0, vola_threshold_mult=0.5,
                  penalty_enable=False, rsi_penalty_thr=80.0, stochrsi_k_penalty_thr=90.0,
                  dist_low_50_pct_penalty_thr=10.0, bb_pct_b_penalty_thr=1.1):
    # (Code unverändert, nutzt jetzt .groupby('pair').shift())
    target_col = 'target'
    if target_col in df.columns: df = df.drop(columns=[target_col])
    logger.debug(f"Target: Berechne future close (shift by {-periods}) gruppiert nach 'pair'...")
    if 'pair' not in df.columns: raise ValueError("Spalte 'pair' fehlt für gruppierten Shift.")
    future_close = df.groupby('pair')['close'].shift(-periods)
    base_target_condition = pd.Series(False, index=df.index)
    if vola_threshold_mult > 0:
        logger.debug(f"Target: Preis > Close + {vola_threshold_mult} * ATR(14) in {periods} Perioden.")
        if 'feature_atr_14' not in df.columns: raise ValueError("ATR ('feature_atr_14') fehlt.")
        safe_atr = df['feature_atr_14'].fillna(0)
        base_target_condition = (future_close > df['close'] + vola_threshold_mult * safe_atr)
    elif threshold > 0:
        logger.debug(f"Target: Preis > Close * (1 + {threshold:.4f}) in {periods} Perioden.")
        base_target_condition = (future_close / df['close'].replace(0, np.nan) - 1) > threshold
    else: logger.warning("Kein gültiges Target definiert.")
    base_target_condition = base_target_condition.fillna(False)
    initial_target_count = base_target_condition.sum()
    # Hinzufügen: Loggen der initialen Anzahl
    logger.info(f"Target Basisbedingung: {initial_target_count} positive Targets ({initial_target_count/len(df)*100:.2f}%) VOR Penalties.")
    if penalty_enable:
        logger.debug("Wende Penalty-Bedingungen auf Target an...")
        penalty = pd.Series(False, index=df.index)
        cond_map = { 'feature_rsi_14': ('>', rsi_penalty_thr), 'feature_stochrsi_k': ('>', stochrsi_k_penalty_thr), 'feature_dist_from_low_50_pct': ('>', dist_low_50_pct_penalty_thr), 'feature_bb_pct_b': ('>', bb_pct_b_penalty_thr) }
        for col, (op, thr) in cond_map.items():
            if col in df.columns:
                condition = (df[col] > thr).fillna(False); penalty = penalty | condition; logger.debug(f"  - Penalty if {col} {op} {thr}")
            else: logger.warning(f"Penalty-Feature {col} nicht gefunden.")
        final_target_condition = base_target_condition & (~penalty)
        num_penalized = (base_target_condition & penalty).sum()
        logger.info(f"Target Penalties: {num_penalized} von {int(initial_target_count)} ({num_penalized/max(1, initial_target_count)*100:.2f}%) pos. Targets auf 0 gesetzt.")
    else:
        final_target_condition = base_target_condition; logger.info("Target Penalties deaktiviert.")
    df[target_col] = final_target_condition.astype(int)
    final_target_count = df[target_col].sum()
    logger.info(f"Target erstellt: {final_target_count} positive Targets (1) von {len(df)} ({final_target_count/len(df)*100:.2f}%) nach Penalties.")
    return df

def objective(trial, X_train, y_train, X_test, y_test, study_name_prefix=""):
    # (Code unverändert)
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
    logger.info("--- Start: MTA GLOBAL ENTRY Modell-Trainingsskript ---")
    # --- Konfiguration loggen ---
    # (unverändert)
    logger.info("--- Verwendete Konfiguration ---")
    # ... (alle logger.info Aufrufe für Konfig) ...
    logger.info(f"  User Data Dir:       {user_data_dir}")
    logger.info(f"  Daten Verzeichnis:   {data_directory}")
    # ... (Rest des Config Loggings) ...
    logger.info("-----------------------------")

    # --- Feature Provider & Timerange ---
    # (unverändert)
    try: feature_provider = MTAFeatureProvider(base_tf=USER_CONFIG_BASE_TIMEFRAME, higher_tf=USER_CONFIG_HIGHER_TIMEFRAME)
    except Exception as e: logger.critical(f"Fehler Init Provider: {e}"); return
    try: start_dt, end_dt = parse_timerange(USER_CONFIG_TIMERANGE)
    except ValueError as e: logger.critical(f"Fehler Timerange: {e}"); return

    # --- Daten laden ---
    # (unverändert)
    try:
        search_pattern_base = os.path.join(data_directory, f'*-{USER_CONFIG_BASE_TIMEFRAME}.feather')
        all_potential_pairs = sorted([os.path.basename(f).replace(f'-{USER_CONFIG_BASE_TIMEFRAME}.feather', '').replace('_', '/') for f in glob.glob(search_pattern_base)])
        if not all_potential_pairs: raise FileNotFoundError(f"Keine '{USER_CONFIG_BASE_TIMEFRAME}' Feather-Dateien.")
        logger.info(f"{len(all_potential_pairs)} mögliche Paare gefunden.")
        pairs_to_load = all_potential_pairs
        if USER_CONFIG_TEST_PAIRS_SUBSET:
            pairs_to_load = sorted([p for p in all_potential_pairs if p in USER_CONFIG_TEST_PAIRS_SUBSET])
            if not pairs_to_load: logger.critical(f"FEHLER: Keine Testpaare {USER_CONFIG_TEST_PAIRS_SUBSET} gefunden."); return
            logger.warning(f"Lade nur Subset: {pairs_to_load}")
        mta_data_dict, common_pairs_list, actual_timeframes = load_data_mta(data_directory, pairs_to_load, [USER_CONFIG_BASE_TIMEFRAME, USER_CONFIG_HIGHER_TIMEFRAME], start_dt, end_dt, USER_CONFIG_TIMERANGE)
        if USER_CONFIG_BASE_TIMEFRAME not in actual_timeframes or USER_CONFIG_HIGHER_TIMEFRAME not in actual_timeframes:
             missing = [tf for tf in [USER_CONFIG_BASE_TIMEFRAME, USER_CONFIG_HIGHER_TIMEFRAME] if tf not in actual_timeframes]; logger.critical(f"FEHLER: TFs {missing} nicht geladen."); return
        if not common_pairs_list: logger.critical(f"FEHLER: Keine gemeinsamen Paare."); return
    except Exception as e: logger.critical(f"Fehler Daten laden: {e}"); return

    # --- Daten Kombinieren ---
    # (unverändert)
    try:
        all_base_dfs = []; all_htf_dfs = []
        logger.info(f"Kombiniere Daten von {len(common_pairs_list)} Paaren...")
        for pair in common_pairs_list:
            if pair not in mta_data_dict[USER_CONFIG_BASE_TIMEFRAME] or pair not in mta_data_dict[USER_CONFIG_HIGHER_TIMEFRAME]: continue
            base_df = mta_data_dict[USER_CONFIG_BASE_TIMEFRAME][pair].copy(); htf_df = mta_data_dict[USER_CONFIG_HIGHER_TIMEFRAME][pair].copy()
            if base_df.empty or htf_df.empty: continue
            base_df['pair'] = pair; htf_df['pair'] = pair
            all_base_dfs.append(base_df); all_htf_dfs.append(htf_df)
        if not all_base_dfs or not all_htf_dfs: raise ValueError("Keine Daten zum Kombinieren.")
        combined_base_df = pd.concat(all_base_dfs, ignore_index=False); combined_base_df.sort_index(inplace=True)
        combined_htf_df = pd.concat(all_htf_dfs, ignore_index=False); combined_htf_df.sort_index(inplace=True)
        logger.info(f"Kombinierte Basis-Daten: {combined_base_df.shape[0]} Zeilen, {len(combined_base_df['pair'].unique())} Paare")
        logger.info(f"Kombinierte HTF-Daten: {combined_htf_df.shape[0]} Zeilen, {len(combined_htf_df['pair'].unique())} Paare")
    except Exception as e: logger.critical(f"Fehler Kombinieren: {e}"); return

    # --- Feature Engineering ---
    # (unverändert)
    try:
        logger.info("Starte Feature Engineering für kombinierte Daten...")
        df_features_combined = feature_provider.add_all_features(combined_base_df, combined_htf_df)
        if 'pair' not in df_features_combined.columns and 'pair' in combined_base_df.columns: df_features_combined['pair'] = combined_base_df['pair']
        elif 'pair' not in df_features_combined.columns: raise ValueError("'pair' Spalte verloren.")
        logger.info(f"Feature Engineering abgeschlossen. Shape: {df_features_combined.shape}")
    except Exception as e: logger.critical(f"Fehler Feature Engineering: {e}"); return

    # --- Target Erstellung ---
    # (unverändert)
    try:
        logger.info("Erstelle Target für kombinierte Daten...")
        df_final_combined = create_target( df_features_combined, periods=USER_CONFIG_LOOKAHEAD_PERIODS, threshold=USER_CONFIG_TARGET_THRESHOLD if USER_CONFIG_TARGET_VOLA_MULT <= 0 else 0, vola_threshold_mult=USER_CONFIG_TARGET_VOLA_MULT, penalty_enable=USER_CONFIG_PENALTY_ENABLE, rsi_penalty_thr=USER_CONFIG_PENALTY_RSI_THRESHOLD, stochrsi_k_penalty_thr=USER_CONFIG_PENALTY_STOCHRSI_K_THRESHOLD, dist_low_50_pct_penalty_thr=USER_CONFIG_PENALTY_DIST_LOW_50_PCT_THRESHOLD, bb_pct_b_penalty_thr=USER_CONFIG_PENALTY_BB_PCT_B_THRESHOLD )
        # <<< Prüfe hier, ob genügend Targets generiert wurden >>>
        if df_final_combined['target'].sum() < len(df_final_combined) * 0.001: # Beispiel: Weniger als 0.1% Targets
            logger.critical(f"FEHLER: Extrem wenige positive Targets ({df_final_combined['target'].sum()}). Bitte Target-Parameter prüfen!")
            return # Abbruch, wenn zu wenige Targets
    except Exception as e: logger.critical(f"Fehler Target-Erstellung: {e}"); return

    # --- Phase 1: HPO ---
    # (unverändert)
    best_params_glob = USER_CONFIG_FIXED_PARAMS.copy()
    if USER_CONFIG_ENABLE_HPO:
        logger.info("--- Starte HPO Phase (Globales Modell) ---")
        try:
            feature_columns_hpo = sorted([col for col in df_final_combined.columns if col.startswith('feature_')]); target_col_hpo = 'target'
            if not feature_columns_hpo or target_col_hpo not in df_final_combined.columns: raise ValueError("Keine Features/Target für HPO.")
            df_ml_hpo = df_final_combined[feature_columns_hpo + [target_col_hpo] + ['pair']].copy();
            for col in feature_columns_hpo: df_ml_hpo[col] = pd.to_numeric(df_ml_hpo[col], errors='coerce')
            df_ml_hpo.replace([np.inf, -np.inf], np.nan, inplace=True); df_ml_hpo[feature_columns_hpo] = df_ml_hpo[feature_columns_hpo].ffill()
            df_ml_hpo.dropna(subset=feature_columns_hpo + [target_col_hpo], inplace=True)
            if len(df_ml_hpo) < USER_CONFIG_MIN_SAMPLES_COMBINED // 2: raise ValueError(f"Zu wenige Daten ({len(df_ml_hpo)}) für HPO.");
            if len(df_ml_hpo[target_col_hpo].value_counts()) < 2: raise ValueError(f"Nur eine Target-Klasse für HPO.")
            logger.info(f"[HPO] Target-Verteilung: \n{df_ml_hpo[target_col_hpo].value_counts(normalize=True).round(4).to_string()}")
            X_hpo = df_ml_hpo[feature_columns_hpo]; y_hpo = df_ml_hpo[target_col_hpo]
            tscv_hpo = TimeSeriesSplit(n_splits=USER_CONFIG_N_SPLITS); train_idx_hpo, test_idx_hpo = list(tscv_hpo.split(X_hpo))[-1]
            if len(train_idx_hpo) == 0 or len(test_idx_hpo) == 0: raise ValueError("Leere Splits für HPO.")
            X_train_hpo, X_test_hpo = X_hpo.iloc[train_idx_hpo], X_hpo.iloc[test_idx_hpo]; y_train_hpo, y_test_hpo = y_hpo.iloc[train_idx_hpo], y_hpo.iloc[test_idx_hpo]
            if X_train_hpo.empty or X_test_hpo.empty or len(y_train_hpo.value_counts()) < 2 or len(y_test_hpo.value_counts()) < 2: raise ValueError("Leere/einklassige Sets für HPO.")
            study_name = f"lgbm_entry_hpo_GLOBAL_{USER_CONFIG_MODEL_PREFIX}"; study = optuna.create_study(direction="maximize", study_name=study_name)
            study.optimize(lambda trial: objective(trial, X_train_hpo, y_train_hpo, X_test_hpo, y_test_hpo, study_name), n_trials=USER_CONFIG_HPO_TRIALS, show_progress_bar=True)
            best_params_glob = study.best_params; logger.info(f"--- HPO Abgeschlossen ---"); logger.info(f"Bester AUC Score: {study.best_value:.6f}"); logger.info(f"Beste Parameter: {best_params_glob}")
        except Exception as e_hpo: logger.error(f"FEHLER während HPO: {e_hpo}"); logger.error(traceback.format_exc()); logger.warning("Fahre mit festen Parametern fort."); best_params_glob = USER_CONFIG_FIXED_PARAMS.copy()
    else: logger.info("--- HPO Phase übersprungen ---"); logger.info(f"Verwende feste Parameter: {best_params_glob}")

    # --- Phase 2: Finales Training ---
    logger.info("--- Starte Finale Trainingsphase (Globales Modell) ---")
    # *** FIX: Initialisiere pair_results VOR dem try-Block ***
    pair_results = {}
    model = None # Initialisiere model auch
    try:
        # 1. ML Vorbereitung & Bereinigung
        # (unverändert)
        feature_columns = sorted([col for col in df_final_combined.columns if col.startswith('feature_')]); target_col = 'target'
        if not feature_columns or target_col not in df_final_combined.columns: raise ValueError("Keine Features/Target.")
        df_ml = df_final_combined[feature_columns + [target_col] + ['pair']].copy()
        for col in feature_columns: df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
        df_ml.replace([np.inf, -np.inf], np.nan, inplace=True); df_ml[feature_columns] = df_ml[feature_columns].ffill()
        remaining_nans = df_ml[feature_columns].isnull().sum().sum()
        if remaining_nans > 0: logger.warning(f"{remaining_nans} NaNs bleiben nach ffill.")
        initial_rows = len(df_ml); df_ml.dropna(subset=feature_columns + [target_col], inplace=True); final_rows = len(df_ml)
        logger.info(f"Daten nach dropna: {final_rows} (von {initial_rows})")
        if final_rows < USER_CONFIG_MIN_SAMPLES_COMBINED: raise ValueError(f"Zu wenige Daten ({final_rows}) (Min: {USER_CONFIG_MIN_SAMPLES_COMBINED}).")
        if len(df_ml[target_col].value_counts()) < 2: raise ValueError(f"Nur eine Target-Klasse: {df_ml[target_col].value_counts()}")
        X = df_ml[feature_columns]; y = df_ml[target_col]; pairs_in_final_data = df_ml['pair']
        logger.info(f"Daten für ML: {len(X)} Zeilen"); logger.info(f"Target-Verteilung (Global): \n{y.value_counts(normalize=True).round(4).to_string()}")

        # 2. Daten aufteilen
        # (unverändert)
        tscv = TimeSeriesSplit(n_splits=USER_CONFIG_N_SPLITS)
        try: train_indices, test_indices = list(tscv.split(X))[-1]
        except Exception as e_split: raise ValueError(f"Fehler TimeSeriesSplit: {e_split}")
        if len(train_indices) == 0 or len(test_indices) == 0: raise ValueError("Leere Splits.")
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]; y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]; pairs_test = pairs_in_final_data.iloc[test_indices]
        if X_train.empty or X_test.empty or len(y_train.value_counts()) < 2: raise ValueError("Leere/einklassige Train/Test-Sets.")
        logger.info(f"Train: {len(X_train)} Zeilen, Test: {len(X_test)} Zeilen")

        # 3. Modell trainieren
        # (unverändert, aber mit Prüfung ob eval_set gültig ist)
        final_params = best_params_glob.copy(); final_params['objective'] = 'binary'; final_params['metric'] = 'auc'; final_params['is_unbalance'] = True
        final_params['n_estimators'] = USER_CONFIG_N_ESTIMATORS_FINAL; final_params['n_jobs'] = -1; final_params['random_state'] = 42; final_params['verbose'] = -1
        model = lgb.LGBMClassifier(**final_params) # model wird hier zugewiesen
        early_stopping_callback_final = lgb.early_stopping(stopping_rounds=USER_CONFIG_EARLY_STOPPING_ROUNDS_FINAL, verbose=False)
        log_evaluation_callback_final = lgb.log_evaluation(period=500)
        logger.info(f"Starte finales globales Training...")

        eval_set_list = None
        eval_metric_list = None
        if len(y_test.value_counts()) >= 2:
            eval_set_list = [(X_test, y_test)]
            eval_metric_list = ['auc']
            logger.info("Early stopping wird auf Testset mit AUC durchgeführt.")
        else:
            # LightGBM braucht *irgendein* eval_set für early stopping,
            # aber wir können nicht 'auc' verwenden. Wir nehmen Train.
            # Besser wäre es, wenn y_test immer beide Klassen hätte.
            eval_set_list = [(X_train, y_train)] # Eval auf Training (nicht ideal)
            eval_metric_list = ['binary_logloss'] # Oder eine andere Metrik
            logger.warning("Nur eine Klasse in y_test. Early Stopping auf Training mit LogLoss.")

        if X_train.isnull().values.any() or np.isinf(X_train.values).any(): raise ValueError("NaN/Inf in X_train VOR fit.")
        if y_train.isnull().values.any(): raise ValueError("NaN in y_train VOR fit.")

        model.fit(X_train, y_train, eval_set=eval_set_list, eval_metric=eval_metric_list, callbacks=[early_stopping_callback_final, log_evaluation_callback_final])
        logger.info(f"Training beendet. Beste Iteration: {model.best_iteration_}")

        # 4. Modell evaluieren & Thresholds pro Paar finden
        # (Code unverändert, außer Initialisierung oben)
        logger.info(f"Evaluiere globales Modell und finde Thresholds pro Paar...")
        pair_thresholds = {}
        # pair_results wurde schon initialisiert
        report_overall = {}
        if y_test.empty: # Prüfung auf leeres y_test
            logger.error("Testset (y_test) ist leer für Evaluation.")
            report_overall = {'error': 'Empty test set'}
        elif len(y_test.value_counts()) < 2:
             logger.warning("Testset hat nur eine Klasse. Evaluation nur bedingt möglich.")
             report_overall = {'error': 'Single class in test set'}
             # Hier könnte man zumindest Accuracy auf dem Testset berechnen
             try:
                 best_iter = model.best_iteration_ if model.best_iteration_ and model.best_iteration_ > 0 else -1
                 X_test_eval = X_test[X_train.columns]
                 if X_test_eval.isnull().values.any() or np.isinf(X_test_eval.values).any(): raise ValueError("NaN/Inf in X_test VOR predict.")
                 y_pred_proba_full = model.predict_proba(X_test_eval, num_iteration=best_iter)[:, 1]
                 # Da nur eine Klasse, Threshold-Findung nicht sinnvoll -> setze 0.5
                 unique_pairs_in_test = pairs_test.unique()
                 for pair_name in unique_pairs_in_test:
                     pair_thresholds[pair_name] = 0.5
                     pair_results[pair_name] = {'pair': pair_name, 'threshold': 0.5, 'error': 'Single class test set'}
                 y_pred_class_overall = (y_pred_proba_full > 0.5).astype(int)
                 accuracy_overall = accuracy_score(y_test, y_pred_class_overall)
                 logger.info(f"\nGesamt-Accuracy (@Thresh 0.5): {accuracy_overall:.4f} (Nur Klasse 0 im Testset!)")
                 report_overall['accuracy'] = accuracy_overall
             except Exception as e_eval_single:
                  logger.error(f"Fehler bei Single-Class Evaluation: {e_eval_single}")
                  report_overall['error'] = f"Single class eval error: {e_eval_single}"
        else: # Normalfall: Beide Klassen im Testset
            try:
                best_iter = model.best_iteration_ if model.best_iteration_ and model.best_iteration_ > 0 else -1
                X_test_eval = X_test[X_train.columns];
                if X_test_eval.isnull().values.any() or np.isinf(X_test_eval.values).any(): raise ValueError("NaN/Inf in X_test VOR predict.")
                logger.info("Mache Vorhersagen für das gesamte Testset...")
                y_pred_proba_full = model.predict_proba(X_test_eval, num_iteration=best_iter)[:, 1]
                if np.isnan(y_pred_proba_full).all(): raise ValueError("predict_proba nur NaNs.")
                unique_pairs_in_test = pairs_test.unique(); logger.info(f"Finde Thresholds für {len(unique_pairs_in_test)} Paare...")
                for pair_name in unique_pairs_in_test:
                    pair_mask = (pairs_test == pair_name); y_test_pair = y_test[pair_mask]; y_pred_proba_pair = y_pred_proba_full[pair_mask]
                    if len(y_test_pair) < 10 or len(y_test_pair.value_counts()) < 2:
                        logger.warning(f"[{pair_name}] <10 Samples oder nur 1 Klasse. Setze Threshold 0.5."); pair_thresholds[pair_name] = 0.5; pair_results[pair_name] = {'pair': pair_name, 'threshold': 0.5, 'error': 'Insufficient test data'}; continue
                    precision, recall, thresholds = precision_recall_curve(y_test_pair, y_pred_proba_pair); valid_pr_indices = ~np.isnan(precision) & ~np.isnan(recall); precision_valid = precision[valid_pr_indices]; recall_valid = recall[valid_pr_indices]; thresholds_valid = thresholds[np.where(valid_pr_indices)[0][:-1]] if len(thresholds) > 0 and np.sum(valid_pr_indices) > 1 else np.array([])
                    if len(precision_valid) <= 1 or len(thresholds_valid) == 0:
                         logger.warning(f"[{pair_name}] Zu wenige Punkte f. Threshold. Setze 0.5."); pred_threshold_pair = 0.5; best_f1_pair = 0.0; achieved_precision = np.nan; achieved_recall = np.nan
                    else:
                        min_precision_target = USER_CONFIG_MIN_PRECISION_TARGET; f1_scores = np.divide(2*recall_valid[:-1]*precision_valid[:-1], recall_valid[:-1]+precision_valid[:-1], out=np.zeros_like(recall_valid[:-1]), where=(recall_valid[:-1]+precision_valid[:-1])!=0); f1_scores[np.isnan(f1_scores)] = 0; valid_precision_indices = np.where(precision_valid[:-1] >= min_precision_target)[0]
                        if len(valid_precision_indices) > 0:
                            valid_f1_scores = f1_scores[valid_precision_indices]
                            if np.max(valid_f1_scores) > 0:
                                ix_relative = np.argmax(valid_f1_scores); ix = valid_precision_indices[ix_relative]
                                if ix < len(thresholds_valid): pred_threshold_pair = thresholds_valid[ix]; best_f1_pair = f1_scores[ix]; achieved_precision=precision_valid[ix]; achieved_recall=recall_valid[ix];
                                else: logger.warning(f"[{pair_name}] Index-Fehler (1). Setze 0.5."); pred_threshold_pair = 0.5; best_f1_pair = 0.0; achieved_precision=np.nan; achieved_recall=np.nan
                            else:
                                logger.warning(f"[{pair_name}] Kein F1>0 für P>={min_precision_target:.2f}. Fallback: Max F1."); ix = np.argmax(f1_scores)
                                if ix < len(thresholds_valid) and f1_scores[ix] > 0: pred_threshold_pair = thresholds_valid[ix]; best_f1_pair = f1_scores[ix]; achieved_precision=precision_valid[ix]; achieved_recall=recall_valid[ix];
                                else: logger.warning(f"[{pair_name}] Fallback F1 0/OOB. Nutze 0.5."); pred_threshold_pair=0.5; best_f1_pair=0.0; achieved_precision=np.nan; achieved_recall=np.nan
                        else:
                            logger.warning(f"[{pair_name}] Kein Thresh mit P>={min_precision_target:.2f}. Fallback: Max F1."); ix = np.argmax(f1_scores)
                            if ix < len(thresholds_valid) and f1_scores[ix] > 0: pred_threshold_pair = thresholds_valid[ix]; best_f1_pair = f1_scores[ix]; achieved_precision=precision_valid[ix]; achieved_recall=recall_valid[ix];
                            else: logger.warning(f"[{pair_name}] Fallback F1 0/OOB. Nutze 0.5."); pred_threshold_pair=0.5; best_f1_pair=0.0; achieved_precision=np.nan; achieved_recall=np.nan
                    pair_thresholds[pair_name] = pred_threshold_pair
                    y_pred_class_pair = (y_pred_proba_pair > pred_threshold_pair).astype(int); accuracy_pair = accuracy_score(y_test_pair, y_pred_class_pair)
                    report_dict_pair = classification_report(y_test_pair, y_pred_class_pair, target_names=['Kein Anstieg', 'Anstieg'], zero_division=0, output_dict=True)
                    pair_results[pair_name] = { 'pair': pair_name, 'threshold': pred_threshold_pair, 'f1_Anstieg': report_dict_pair.get('Anstieg', {}).get('f1-score', np.nan), 'prec_Anstieg': report_dict_pair.get('Anstieg', {}).get('precision', np.nan), 'recall_Anstieg': report_dict_pair.get('Anstieg', {}).get('recall', np.nan), 'support_Anstieg': report_dict_pair.get('Anstieg', {}).get('support', np.nan), 'accuracy': accuracy_pair, 'test_samples': len(y_test_pair) }
                y_pred_class_overall = (y_pred_proba_full > 0.5).astype(int); accuracy_overall = accuracy_score(y_test, y_pred_class_overall)
                logger.info(f"\nGesamt-Accuracy (@Thresh 0.5): {accuracy_overall:.4f}")
                report_str_overall = classification_report(y_test, y_pred_class_overall, target_names=['Kein Anstieg', 'Anstieg'], zero_division=0); logger.info(f"Gesamt Classification Report (@Thresh 0.5):\n{report_str_overall}")
                report_overall = classification_report(y_test, y_pred_class_overall, target_names=['Kein Anstieg', 'Anstieg'], zero_division=0, output_dict=True); report_overall['accuracy'] = accuracy_overall
            except Exception as e_eval: logger.error(f"Fehler Evaluation/Threshold: {e_eval}"); logger.error(traceback.format_exc()); report_overall = {'error': str(e_eval)}

        # 5. Ergebnisse & Feature Importance anzeigen
        # (unverändert)
        if model and hasattr(model, 'feature_importances_'): # Prüfe ob model existiert
             try:
                 importances = model.feature_importances_
                 if importances is not None and len(importances) == len(X_train.columns):
                     logger.info(f"Top 10 Features (Globales Modell):"); feature_imp = pd.DataFrame({'Value': importances, 'Feature': X_train.columns}); logger.info("\n" + feature_imp.nlargest(10, "Value").to_string(index=False))
                 else: logger.warning("Feature Importances nicht ok.")
             except Exception as fi_e: logger.error(f"Fehler Feature Importance: {fi_e}")

        # 6. Modell & Features speichern (nur wenn Modell trainiert wurde)
        if model: # <<< Prüfe ob Modell existiert
            os.makedirs(save_directory, exist_ok=True)
            base_tf_safe = USER_CONFIG_BASE_TIMEFRAME.replace('/', '_'); higher_tf_safe = USER_CONFIG_HIGHER_TIMEFRAME.replace('/', '_')
            base_filename = f"{USER_CONFIG_MODEL_PREFIX}_GLOBAL_{base_tf_safe}_WITH_{higher_tf_safe}"
            model_filepath = os.path.join(save_directory, f"{base_filename}.joblib"); features_filepath = os.path.join(save_directory, f"{base_filename}_features.json"); threshold_filepath = os.path.join(save_directory, f"{USER_CONFIG_MODEL_PREFIX}_thresholds.json")
            logger.info(f"Speichere globales Modell: {model_filepath}"); joblib.dump(model, model_filepath)
            feature_list_to_save = list(X_train.columns) # X_train existiert sicher hier
            logger.info(f"Speichere globale Features ({len(feature_list_to_save)}): {features_filepath}")
            try:
                with open(features_filepath, 'w') as f: json.dump(feature_list_to_save, f, indent=4); logger.info(f"Modell/Features gespeichert.")
            except Exception as e_save: logger.error(f"Fehler Speichern Features: {e_save}")
        else:
            logger.error("Modell wurde nicht trainiert, nichts wird gespeichert.")

        # 7. Thresholds speichern (nur wenn sie berechnet wurden)
        if pair_thresholds:
            logger.info(f"Speichere paar-spezifische Thresholds nach {threshold_filepath}...")
            try:
                threshold_dict_serializable = {k: float(v) if pd.notna(v) else 0.5 for k, v in pair_thresholds.items()}
                with open(threshold_filepath, 'w') as f: json.dump(threshold_dict_serializable, f, indent=4, sort_keys=True)
                logger.info(f"Thresholds für {len(threshold_dict_serializable)} Paare gespeichert.")
            except Exception as e: logger.error(f"Fehler Speichern Thresholds: {e}"); logger.error(traceback.format_exc())
        elif 'error' not in report_overall: # Nur warnen, wenn kein genereller Fehler vorlag
             logger.warning("Keine paar-spezifischen Thresholds zum Speichern gefunden (möglicherweise Single-Class Testset).")

    except ValueError as e: logger.error(f"Globales Training gescheitert: {e}."); logger.error(traceback.format_exc());
    except KeyError as e: logger.error(f"Globales Training gescheitert: {e}."); logger.error(traceback.format_exc());
    except MemoryError as e: logger.error(f"Globales Training gescheitert: {e}."); logger.error(traceback.format_exc());
    except Exception as e: logger.error(f"Unerwarteter Fehler: {e}."); logger.error(traceback.format_exc());
    finally: logger.info(f"===== Globales Training abgeschlossen =====")

    # --- Ergebnisübersicht (Paar-spezifisch) ---
    logger.info("\n--- Training abgeschlossen ---")
    logger.info("\n--- Ergebnisübersicht pro Paar (basierend auf globalem Modell & paar-spezifischem Threshold) ---")
    if pair_results: # Prüft ob pair_results existiert und nicht leer ist
        summary_data = list(pair_results.values())
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(by='f1_Anstieg', ascending=False, na_position='last')
        logger.info("Paar-Ergebnis Zusammenfassung:\n" + summary_df[['pair', 'threshold', 'f1_Anstieg', 'prec_Anstieg', 'recall_Anstieg', 'accuracy', 'test_samples']].to_string(index=False, float_format="%.4f"))
    else: logger.info("Keine Einzelergebnisse für Zusammenfassung vorhanden (möglicherweise Fehler im Training oder Single-Class Testset).")


# --- Skriptstart Schutz ---
if __name__ == "__main__":
    logger.info("entrytrain.py (global) wird als Hauptskript ausgeführt.")
    main()
else:
    logger.warning("entrytrain.py (global) wird als Modul importiert (unerwartet).")

# --- END OF FILE entrytrainglobal.py (GLOBAL MODEL VERSION - FIX ATTEMPT) ---
