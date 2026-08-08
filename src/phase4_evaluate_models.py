import os
import joblib
import pandas as pd
import numpy as np

# Directories
DATA_DIR = 'data'
MODEL_DIR = 'models'

# Load models and saved feature column list
model_perishable = joblib.load(os.path.join(MODEL_DIR, 'model_perishable.pkl'))
model_nonperishable = joblib.load(os.path.join(MODEL_DIR, 'model_nonperishable.pkl'))
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))

# Critical Ratios
CRITICAL_RATIO_PERISHABLE = 0.250
CRITICAL_RATIO_NON_PERISHABLE = 0.778

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, 'retail_features.csv'), parse_dates=['date'])

# Encode object/categorical columns (city, state, etc.)
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].astype('category').cat.codes

# Define test set (using last 30 days of dataset)
max_date = df['date'].max()
test_start_date = max_date - pd.Timedelta(days=30)
test_df = df[df['date'] >= test_start_date].copy()

def pinball_loss(y_true, y_pred, alpha):
    """Calculates Pinball / Quantile Loss."""
    err = y_true - y_pred
    return np.mean(np.maximum(alpha * err, (alpha - 1) * err))

def evaluate_model(model, df_subset, alpha, name):
    print(f"\n=== Evaluating {name} Model (alpha={alpha:.3f}) ===")
    
    X = df_subset[feature_cols]
    y_true = df_subset['unit_sales']
    
    y_pred = model.predict(X)
    
    # Calculate metrics
    loss = pinball_loss(y_true, y_pred, alpha)
    understock_rate = np.mean(y_true > y_pred)
    coverage_rate = np.mean(y_true <= y_pred)
    
    print(f"Test Set Size:      {len(df_subset)} rows")
    print(f"Pinball Loss:       {loss:.4f}")
    print(f"Empirical Coverage: {coverage_rate * 100:.2f}% (Target: {alpha * 100:.1f}%)")
    print(f"Stockout Rate:      {understock_rate * 100:.2f}%")

if __name__ == '__main__':
    # Evaluate Perishable
    test_p = test_df[test_df['perishable'] == 1]
    evaluate_model(model_perishable, test_p, CRITICAL_RATIO_PERISHABLE, "PERISHABLE")
    
    # Evaluate Non-Perishable
    test_np = test_df[test_df['perishable'] == 0]
    evaluate_model(model_nonperishable, test_np, CRITICAL_RATIO_NON_PERISHABLE, "NON-PERISHABLE")
    
    print("\nPhase 4 evaluation complete!")