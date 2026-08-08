# phase2_feature_engineering.py
#
# PURPOSE:
# Raw sales numbers alone don't tell the model WHY demand moves -- promotions,
# holidays, and recent trend all shift demand independently of any single
# average. This phase builds those context features so the model can learn
# more than a naive rolling mean would capture.

import pandas as pd
import numpy as np
import os

DATA_DIR = 'data'

df = pd.read_csv(os.path.join(DATA_DIR, 'retail_subset.csv'), parse_dates=['date'])
df = df.sort_values(['store_nbr', 'item_nbr', 'date']).reset_index(drop=True)

# ---- 1. Date-based features ----
# WHY: demand has strong weekly/monthly seasonality (weekend spikes, month-end
# effects) that a model can't infer from a raw date without these being
# broken out into usable numeric/categorical signals.
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# ---- 2. Lag / rolling features, computed PER (store, item) pair ----
# WHY: these give the model a sense of recent trend for that specific
# store-item combination -- without them, the model has no way to know if
# demand for this item at this store is currently trending up or down.
# IMPORTANT: shift(1) before rolling() ensures we never use today's own
# sales to predict today's sales (that would be leakage).
grp = df.groupby(['store_nbr', 'item_nbr'])['unit_sales']

df['lag_7d_sales'] = grp.shift(7)
df['rolling_7d_avg'] = grp.transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
df['rolling_30d_avg'] = grp.transform(lambda x: x.shift(1).rolling(30, min_periods=5).mean())
df['rolling_30d_std'] = grp.transform(lambda x: x.shift(1).rolling(30, min_periods=5).std())

# ---- 3. onpromotion as integer (it may load as bool/string depending on source) ----
df['onpromotion'] = df['onpromotion'].fillna(False).astype(int)

# ---- 4. Encode categoricals ----
# WHY: tree models need numeric input; one-hot avoids implying false
# ordering between categories (same reasoning as the pricing project).
df = pd.get_dummies(df, columns=['family', 'type', 'day_of_week'],
                     prefix=['fam', 'store_type', 'dow'])

# ---- 5. Drop rows where lag features are NaN ----
# WHY: the first ~30 days of each (store, item) pair's history can't have a
# valid rolling_30d_avg yet -- keeping them would feed the model incomplete
#/imputed-as-zero signal that doesn't reflect real history.
before = len(df)
df = df.dropna(subset=['lag_7d_sales', 'rolling_30d_avg'])
print(f"Dropped {before - len(df)} rows with insufficient history for lag features")

print("\nFinal shape:", df.shape)
print("perishable=1 rows:", (df['perishable'] == 1).sum())
print("perishable=0 rows:", (df['perishable'] == 0).sum())

df.to_csv(os.path.join(DATA_DIR, 'retail_features.csv'), index=False)
print(f"\nSaved to {DATA_DIR}/retail_features.csv")

import os
import joblib

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Save the feature columns list so that FastAPI & Phase 3/5 can use it
feature_cols = [col for col in df.columns if col not in ['date', 'unit_sales', 'id', 'store_nbr', 'item_nbr', 'perishable']]
joblib.dump(feature_cols, os.path.join('models', 'feature_list.pkl'))
print("Saved models/feature_list.pkl successfully.")
