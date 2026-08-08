# app/streamlit_app_standalone.py
# Deployment version for Streamlit Community Cloud: loads the quantile
# regression models DIRECTLY inside Streamlit (no separate FastAPI process).

import os
import math
import logging
import json
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Favorita Demand Forecasting", layout="centered")

MODEL_DIR = 'models'
DATA_PATH = os.path.join('data', 'retail_features.csv')

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'requests.log'),
    level=logging.INFO,
    format='%(message)s'
)
monitor_logger = logging.getLogger('forecast_monitor')

# Column names that correspond to each day-of-week one-hot slot, matching
# feature_list.pkl's 'dow_0'..'dow_6' naming
DOW_COLS = ['dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6']


@st.cache_resource
def load_models():
    model_perishable = joblib.load(os.path.join(MODEL_DIR, 'model_perishable.pkl'))
    model_nonperishable = joblib.load(os.path.join(MODEL_DIR, 'model_nonperishable.pkl'))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))
    return model_perishable, model_nonperishable, feature_cols


@st.cache_data(show_spinner=True, ttl=3600)
def load_feature_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, parse_dates=['date'])
    return None


def build_input_features(latest_row, target_date, feature_cols):
    """Builds the feature dict for prediction, starting from the item's
    latest known static/rolling features (from latest_row), then
    OVERWRITING the date-dependent features (day-of-week, month, weekend)
    with values computed from the user's actually selected target_date.

    BUG FIXED: the previous version left dow_0..dow_6 (and is_weekend)
    at whatever historical values happened to be in latest_row, never
    updating them for the date the user actually asked about -- meaning
    changing the forecast date had no effect on these features, which
    contributed to identical predictions across different inputs.
    """
    feature_dict = latest_row.to_dict()
    for col in ['date', 'unit_sales', 'id', 'store_nbr', 'item_nbr',
                'perishable', 'onpromotion']:
        feature_dict.pop(col, None)

    # ---- Overwrite date-dependent features using the user's target_date ----
    feature_dict['month'] = target_date.month
    weekday = target_date.weekday()  # Monday=0 ... Sunday=6
    feature_dict['is_weekend'] = 1 if weekday >= 5 else 0
    for col in DOW_COLS:
        if col in feature_cols:
            feature_dict[col] = 0
    dow_col = f'dow_{weekday}'
    if dow_col in feature_cols:
        feature_dict[dow_col] = 1

    # ---- Safely handle any remaining non-numeric values ----
    # (rather than silently zeroing every string, which could hide a real
    # feature-encoding problem -- log a warning instead so it's visible)
    for k, v in list(feature_dict.items()):
        if isinstance(v, str):
            monitor_logger.warning(json.dumps({
                'warning': f"Non-numeric value found for feature '{k}': '{v}' -- "
                           f"replaced with 0. This may indicate a feature "
                           f"encoding mismatch worth investigating."
            }))
            feature_dict[k] = 0
        elif pd.isna(v):
            feature_dict[k] = 0

    return feature_dict


def predict_demand(store_nbr, item_nbr, onpromotion, perishable, feature_dict,
                    model_perishable, model_nonperishable, feature_cols):
    input_features = dict(feature_dict)
    input_features['store_nbr'] = store_nbr
    input_features['item_nbr'] = item_nbr
    input_features['onpromotion'] = onpromotion
    input_features['perishable'] = perishable

    input_df = pd.DataFrame([input_features])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    if perishable == 1:
        raw_pred = model_perishable.predict(input_df)[0]
        alpha_used = 0.25
    else:
        raw_pred = model_nonperishable.predict(input_df)[0]
        alpha_used = 0.778

    recommended_stock = math.ceil(max(0.0, float(raw_pred)))
    return recommended_stock, alpha_used


def check_input_drift(store_nbr, item_nbr, all_data):
    reasons = []
    if all_data is not None:
        exact_match_count = len(all_data[(all_data['store_nbr'] == store_nbr) &
                                           (all_data['item_nbr'] == item_nbr)])
        if exact_match_count == 0:
            reasons.append("no direct history for this exact store+item combination")
        item_history_count = len(all_data[all_data['item_nbr'] == item_nbr])
        if item_history_count < 10:
            reasons.append(f"this item has only {item_history_count} historical records overall")
    return reasons


st.title("🛒 Favorita Retail Demand Forecasting")
st.markdown("Quantile Regression based Inventory Optimization Dashboard")

try:
    model_perishable, model_nonperishable, feature_cols = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Could not load models: {e}")

df_features = load_feature_dataset()

if df_features is None:
    st.error(
        "⚠️ Could not load `data/retail_features.csv`. Predictions cannot be "
        "generated without it. Check that this file is actually present in "
        "the deployed repo (not just a DVC pointer)."
    )

if models_loaded and df_features is not None:
    st.sidebar.header("Forecast Parameters")

    store_nbr = st.sidebar.number_input("Store Number", min_value=1, max_value=54, value=25)
    available_items = df_features['item_nbr'].unique()
    item_nbr = st.sidebar.selectbox("Select Item ID", options=available_items)

    target_date = st.sidebar.date_input("Forecast Date", value=datetime.today())
    onpromotion = st.sidebar.selectbox("Is On Promotion?", options=[0, 1],
                                         format_func=lambda x: "Yes" if x == 1 else "No")

    matched_rows = df_features[(df_features['store_nbr'] == int(store_nbr)) &
                                 (df_features['item_nbr'] == int(item_nbr))]
    if matched_rows.empty:
        matched_rows = df_features[df_features['item_nbr'] == int(item_nbr)]

    latest_row = matched_rows.sort_values('date').iloc[-1] if not matched_rows.empty else None

    if latest_row is not None:
        perishable = int(latest_row.get('perishable', 1))
        last_sale = latest_row.get('unit_sales', 'N/A')
        st.sidebar.info(f"**Auto-Detected Perishable:** {'Yes' if perishable == 1 else 'No'}\n\n"
                        f"**Last Recorded Daily Sale:** {last_sale} units")
    else:
        perishable = 1

    if st.button("Generate Demand Forecast", type="primary"):
        if latest_row is None:
            st.error(f"No historical data found for item {item_nbr} at all. Cannot forecast.")
        else:
            feature_dict = build_input_features(latest_row, target_date, feature_cols)

            try:
                recommended_stock, alpha_used = predict_demand(
                    int(store_nbr), int(item_nbr), int(onpromotion), int(perishable),
                    feature_dict, model_perishable, model_nonperishable, feature_cols
                )

                st.success(f"Forecast for {target_date} Generated Successfully!")

                col1, col2 = st.columns(2)
                col1.metric("Recommended Stock", f"{recommended_stock} units")
                col2.metric("Target Risk Level (Alpha)", f"{alpha_used}")

                drift_reasons = check_input_drift(int(store_nbr), int(item_nbr), df_features)
                if drift_reasons:
                    st.warning("⚠️ Lower-confidence prediction: " + "; ".join(drift_reasons) + ".")

                log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'store_nbr': int(store_nbr),
                    'item_nbr': int(item_nbr),
                    'target_date': str(target_date),
                    'perishable': int(perishable),
                    'onpromotion': int(onpromotion),
                    'quantile_alpha': alpha_used,
                    'predicted_unit_sales': recommended_stock,
                    'drift_flags': drift_reasons
                }
                monitor_logger.info(json.dumps(log_entry))

                with st.expander("Raw result & Monitoring Log"):
                    st.json(log_entry)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.divider()
st.caption(
    "Models: Quantile Regression (XGBoost) -- separate models for perishable "
    "(alpha=0.25) and non-perishable (alpha=0.778) items."
)