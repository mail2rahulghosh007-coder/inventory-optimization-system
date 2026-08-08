import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split

DATA_DIR = 'data'
MODEL_DIR = 'models'

dagshub.init(repo_owner='mail2rahulghosh007-coder', repo_name='inventory_optimization_system', mlflow=True)
print("DagsHub MLflow Tracking initialized successfully.")


def pinball_loss(y_true, y_pred, alpha):
    err = y_true - y_pred
    return np.mean(np.maximum(alpha * err, (alpha - 1) * err))


def build_lookup(df_subset):
    """Per (store_nbr, item_nbr): sorted unique dates + corresponding unit_sales,
    for fast, leakage-safe 'as of date X' lookups."""
    lookup = {}
    for (store_nbr, item_nbr), grp in df_subset.groupby(['store_nbr', 'item_nbr']):
        g = grp.sort_values('date').drop_duplicates(subset='date', keep='last')
        lookup[(store_nbr, item_nbr)] = (
            g['date'].values.astype('datetime64[ns]'),
            g['unit_sales'].values.astype(float)
        )
    return lookup


def compute_baselines(val_index, df_subset, lookup, fallback_value):
    """For each validation row, using ONLY history strictly before that row's date
    (no leakage), compute: naive last value, rolling 7-day mean, seasonal (t-7)."""
    naive_last = np.full(len(val_index), np.nan)
    rolling_mean_7 = np.full(len(val_index), np.nan)
    seasonal_naive = np.full(len(val_index), np.nan)

    rows = df_subset.loc[val_index, ['store_nbr', 'item_nbr', 'date']]

    for i, (idx, row) in enumerate(rows.iterrows()):
        key = (row['store_nbr'], row['item_nbr'])
        dates, sales = lookup.get(key, (np.array([], dtype='datetime64[ns]'), np.array([])))
        target_date = np.datetime64(row['date'])

        pos = np.searchsorted(dates, target_date, side='left')  # first date >= target
        before_pos = pos - 1

        if before_pos >= 0:
            naive_last[i] = sales[before_pos]
            window_start = max(0, before_pos - 6)
            rolling_mean_7[i] = sales[window_start:before_pos + 1].mean()
        else:
            naive_last[i] = fallback_value
            rolling_mean_7[i] = fallback_value

        seasonal_target = target_date - np.timedelta64(7, 'D')
        s_pos = np.searchsorted(dates, seasonal_target, side='left')
        if s_pos < len(dates) and dates[s_pos] == seasonal_target:
            seasonal_naive[i] = sales[s_pos]
        else:
            seasonal_naive[i] = naive_last[i]  # fallback if exactly-7-days-ago missing

        if (i + 1) % 50000 == 0:
            print(f"  ...{i+1}/{len(val_index)} baseline rows computed")

    return naive_last, rolling_mean_7, seasonal_naive


def run_baseline_suite(df, feature_cols, perishable_flag, alpha, model_name):
    print(f"\n=== Baselines for {model_name} (alpha={alpha}) ===")
    df_sub = df[df['perishable'] == perishable_flag]
    X = df_sub[feature_cols]
    y = df_sub['unit_sales']

    # IDENTICAL split call/order/params as training script -> same val rows
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    fallback_value = float(y_train.median())
    lookup = build_lookup(df_sub)

    naive_last, rolling_mean_7, seasonal_naive = compute_baselines(
        X_val.index, df_sub, lookup, fallback_value
    )
    y_val_arr = y_val.values.astype(float)

    results = {
        'naive_last_value': pinball_loss(y_val_arr, naive_last, alpha),
        'rolling_mean_7d': pinball_loss(y_val_arr, rolling_mean_7, alpha),
        'seasonal_naive_t7': pinball_loss(y_val_arr, seasonal_naive, alpha),
    }

    for name, loss in results.items():
        print(f"  {name}: pinball_loss = {loss:.4f}")

    mlflow.set_experiment(f"Favorita_{model_name}_Optimization")
    with mlflow.start_run(run_name=f"baseline_comparison_{model_name}"):
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("n_val_rows", len(y_val))
        for name, loss in results.items():
            mlflow.log_metric(f"pinball_loss_{name}", loss)

    return results


if __name__ == '__main__':
    print("Loading retail_features.csv...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'retail_features.csv'), parse_dates=['date'])

    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].astype('category').cat.codes

    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))

    perishable_results = run_baseline_suite(
        df, feature_cols, perishable_flag=1, alpha=0.250, model_name="PERISHABLE"
    )
    nonperishable_results = run_baseline_suite(
        df, feature_cols, perishable_flag=0, alpha=0.778, model_name="NON_PERISHABLE"
    )

    print("\n=== SUMMARY ===")
    print("Perishable  | XGBoost (Optuna) = 2.3215  | Baselines:", perishable_results)
    print("Non-perish. | XGBoost (Optuna) = 2.9726  | Baselines:", nonperishable_results)