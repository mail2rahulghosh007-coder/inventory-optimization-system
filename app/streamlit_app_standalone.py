# app/streamlit_app_standalone.py
# Deployment version for Streamlit Community Cloud.
#
# IMPORTANT MEMORY FIX:
# The previous version loaded the entire data/retail_features.csv (702MB), which exceeded 
# the free-tier RAM limit (~1GB) and caused an OOM crash during the second prediction.
# Now it loads data/inference_snapshot.parquet (a few MBs), which is pre-built using 
# the build_inference_snapshot.py script, containing the latest row + drift counts 
# for each (store_nbr, item_nbr) combination.

import os
import math
import logging
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Favorita Demand Forecasting", layout="centered")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
    MODEL_DIR = os.path.join(ROOT_DIR, 'models')
    SNAPSHOT_PATH = os.path.join(ROOT_DIR, 'data', 'inference_snapshot.parquet')
    LOG_DIR = os.path.join(ROOT_DIR, 'logs')

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, 'requests.log'),
        level=logging.INFO,
        format='%(message)s'
    )
    monitor_logger = logging.getLogger('forecast_monitor')

    DOW_COLS = ['dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6']

    @st.cache_resource
    def load_models():
        m_perish_path = os.path.join(MODEL_DIR, 'model_perishable.pkl')
        m_nonperish_path = os.path.join(MODEL_DIR, 'model_nonperishable.pkl')
        feat_path = os.path.join(MODEL_DIR, 'feature_list.pkl')

        for p in [m_perish_path, m_nonperish_path, feat_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing model file at: {p}")

        model_perishable = joblib.load(m_perish_path)
        model_nonperishable = joblib.load(m_nonperish_path)
        feature_cols = joblib.load(feat_path)
        return model_perishable, model_nonperishable, feature_cols

    @st.cache_data(show_spinner=True, ttl=3600)
    def load_snapshot():
        if not os.path.exists(SNAPSHOT_PATH):
            raise FileNotFoundError(
                f"Missing {SNAPSHOT_PATH}. Run build_inference_snapshot.py first "
                f"and commit the resulting parquet file to the repo."
            )
        df = pd.read_parquet(SNAPSHOT_PATH)
        # Reduce dtypes to further lower memory usage
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')
        return df

    model_perishable, model_nonperishable, feature_cols = load_models()
    df_snapshot = load_snapshot()
    models_loaded = True

except Exception as e:
    models_loaded = False
    df_snapshot = None
    st.error("🚨 FATAL STARTUP ERROR DETECTED:")
    st.exception(e)
    import traceback
    st.code(traceback.format_exc())
    st.stop()


def build_input_features(latest_row, target_date, feature_cols):
    feature_dict = latest_row.to_dict()
    for col in ['date', 'unit_sales', 'id', 'store_nbr', 'item_nbr',
                'perishable', 'onpromotion', 'exact_match_count',
                'item_history_count']:
        feature_dict.pop(col, None)

    feature_dict['month'] = target_date.month
    weekday = target_date.weekday()
    feature_dict['is_weekend'] = 1 if weekday >= 5 else 0
    for col in DOW_COLS:
        if col in feature_cols:
            feature_dict[col] = 0
    dow_col = f'dow_{weekday}'
    if dow_col in feature_cols:
        feature_dict[dow_col] = 1

    for k, v in list(feature_dict.items()):
        if isinstance(v, str):
            monitor_logger.warning(json.dumps({
                'warning': f"Non-numeric value found for feature '{k}': '{v}' -- replaced with 0."
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

    input_df = input_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    if perishable == 1:
        raw_pred = model_perishable.predict(input_df)[0]
        alpha_used = 0.25
    else:
        raw_pred = model_nonperishable.predict(input_df)[0]
        alpha_used = 0.778

    recommended_stock = math.ceil(max(0.0, float(raw_pred)))
    return recommended_stock, alpha_used


def check_input_drift(row):
    reasons = []
    if row.get('exact_match_count', 1) == 0:
        reasons.append("no direct history for this exact store+item combination")
    item_hist = row.get('item_history_count', 999)
    if item_hist < 10:
        reasons.append(f"this item has only {int(item_hist)} historical records overall")
    return reasons


st.title("🛒 Favorita Retail Demand Forecasting")
st.markdown("Quantile Regression based Inventory Optimization Dashboard")

if models_loaded and df_snapshot is not None:
    st.sidebar.header("Forecast Parameters")

    store_nbr = st.sidebar.number_input("Store Number", min_value=1, max_value=54, value=25)
    available_items = df_snapshot['item_nbr'].unique()
    item_nbr = st.sidebar.selectbox("Select Item ID", options=available_items)

    target_date = st.sidebar.date_input("Forecast Date", value=datetime.today())
    onpromotion = st.sidebar.selectbox("Is On Promotion?", options=[0, 1],
                                       format_func=lambda x: "Yes" if x == 1 else "No")

    matched = df_snapshot[(df_snapshot['store_nbr'] == int(store_nbr)) &
                          (df_snapshot['item_nbr'] == int(item_nbr))]
    if matched.empty:
        # If this item is not available in that store, take the item's most recent row 
        # (from any store) as a fallback
        item_rows = df_snapshot[df_snapshot['item_nbr'] == int(item_nbr)]
        matched = item_rows.sort_values('date').tail(1) if not item_rows.empty else item_rows

    latest_row = matched.iloc[-1] if not matched.empty else None

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

                drift_reasons = check_input_drift(latest_row)
                if drift_reasons:
                    st.warning("⚠️ Lower-confidence prediction: " + "; ".join(drift_reasons) + ".")

                log_entry = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
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
                st.error("🚨 PREDICTION ERROR:")
                st.exception(e)
                import traceback
                st.code(traceback.format_exc())

st.divider()
st.caption(
    "Models: Quantile Regression (XGBoost) -- separate models for perishable "
    "(alpha=0.25) and non-perishable (alpha=0.778) items."
)