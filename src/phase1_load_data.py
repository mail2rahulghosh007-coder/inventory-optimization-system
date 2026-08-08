# phase1_load_data.py
#
# PURPOSE:
# The full Favorita training data has ~125 million rows (store x item x date).
# That's too large to iterate on quickly and unnecessary for a portfolio-scale
# project. This phase loads only the needed columns, picks a manageable subset
# (top stores by volume, top items by frequency), and merges in item/store
# metadata -- including the 'perishable' flag, which is central to this
# project's inventory-optimization logic later.

import pandas as pd
import os

DATA_DIR = 'data'

# ---- Load only the columns we need (saves memory on a 5GB+ file) ----
print("Loading train.csv (this may take a few minutes)...")
train = pd.read_csv(
    os.path.join(DATA_DIR, 'train.csv'),
    usecols=['date', 'store_nbr', 'item_nbr', 'unit_sales', 'onpromotion'],
    parse_dates=['date'],
    dtype={'store_nbr': 'int32', 'item_nbr': 'int32', 'unit_sales': 'float32'}
)
print(f"Full train shape: {train.shape}")

# ---- Load metadata ----
items = pd.read_csv(os.path.join(DATA_DIR, 'items.csv'))    # item_nbr, family, class, perishable
stores = pd.read_csv(os.path.join(DATA_DIR, 'stores.csv'))  # store_nbr, city, state, type, cluster

# ---- Drop returns (negative unit_sales represent product returns, not demand) ----
before = len(train)
train = train[train['unit_sales'] >= 0]
print(f"Dropped {before - len(train)} rows with negative unit_sales (returns)")

# ---- Subset: top N stores by transaction volume, top M items by frequency ----
# WHY: this keeps enough history PER (store, item) combination for meaningful
# time-series CV later, while keeping the dataset small enough to iterate on
# locally -- unlike the earlier pricing project where each product only had
# ~13 data points, here each (store, item) pair will have up to ~1600 days
# of history (the full date range of the competition).
N_STORES = 20
N_ITEMS = 100

top_stores = train['store_nbr'].value_counts().head(N_STORES).index
top_items = train['item_nbr'].value_counts().head(N_ITEMS).index

df = train[train['store_nbr'].isin(top_stores) & train['item_nbr'].isin(top_items)].copy()
print(f"Subset shape: {df.shape} ({N_STORES} stores x {N_ITEMS} items)")

# ---- Merge item metadata (brings in 'perishable', 'family', 'class') ----
df = df.merge(items, on='item_nbr', how='left')

# ---- Merge store metadata (brings in 'type', 'city', 'state', 'cluster') ----
df = df.merge(stores, on='store_nbr', how='left')

# ---- Sanity checks ----
print("\nDate range:", df['date'].min(), "to", df['date'].max())
print("Unique (store, item) combinations:", df.groupby(['store_nbr', 'item_nbr']).ngroups)
print("Perishable item rows:", (df['perishable'] == 1).sum(),
      "| Non-perishable item rows:", (df['perishable'] == 0).sum())

missing = df.isnull().sum()
missing = missing[missing > 0]
print("\nMissing values:", "None" if missing.empty else missing.to_dict())

# ---- Save ----
os.makedirs(DATA_DIR, exist_ok=True)
df.to_csv(os.path.join(DATA_DIR, 'retail_subset.csv'), index=False)
print(f"\nSaved subset to {DATA_DIR}/retail_subset.csv")
