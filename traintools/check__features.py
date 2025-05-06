import joblib
import traceback

# --- Passe diesen Pfad an! ---
# Nimm eine Datei für ein Paar, das im Log vorkommt, z.B. ETH/USDC
feature_file = '/home/tommi/freqtrade/user_data/models/_ETH_USDC_5m_WITH_1h_features.joblib'
# -----------------------------

try:
    features = joblib.load(feature_file)
    print(f"--- Features in {feature_file} ---")
    # Sortiere die Liste alphabetisch für leichtere Lesbarkeit
    features.sort()
    print(features)

    print("\n--- Checking for specific missing 5m features (as per log warning) ---")
    # Liste der Features aus der letzten Warnung
    missing_5m_in_warning = ['feature_dist_from_high_20', 'feature_high_20', 'feature_ma_10',
                             'feature_ma_200', 'feature_ma_50', 'feature_volume_ma_20',
                             'feature_volume_ratio']
    found_count = 0
    for f in missing_5m_in_warning:
        is_present = f in features
        print(f"'{f}' in features list: {is_present}")
        if is_present:
            found_count += 1

    print(f"\nFound {found_count} out of {len(missing_5m_in_warning)} missing 5m features in the saved list.")
    if found_count < len(missing_5m_in_warning):
        print("==> PROBLEM: It seems these 5m base features are NOT included in the saved feature list from training!")
    else:
        print("==> OK: All listed 5m base features ARE included in the saved feature list.")


except FileNotFoundError:
    print(f"Error: File not found at {feature_file}")
    print("Please check the path.")
except Exception as e:
    print(f"An error occurred loading {feature_file}:")
    print(traceback.format_exc())
