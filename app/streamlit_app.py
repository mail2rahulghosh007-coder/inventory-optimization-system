import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Favorita Demand Forecasting", layout="centered")

st.title("🛒 Favorita Retail Demand Forecasting")
st.markdown("Quantile Regression based Inventory Optimization Dashboard")

# Load engineered features dataset
@st.cache_data
def load_feature_dataset():
    features_path = os.path.join('data', 'retail_features.csv')
    if os.path.exists(features_path):
        return pd.read_csv(features_path, parse_dates=['date'])
    return None

df_features = load_feature_dataset()

st.sidebar.header("Forecast Parameters")

# Store selection
store_nbr = st.sidebar.number_input("Store Number", min_value=1, max_value=54, value=25)

if df_features is not None:
    # Get available items from dataset
    available_items = df_features['item_nbr'].unique()
    item_nbr = st.sidebar.selectbox("Select Item ID", options=available_items)
    
    # Extract latest recorded history for selected store & item
    matched_rows = df_features[(df_features['store_nbr'] == store_nbr) & (df_features['item_nbr'] == item_nbr)]
    
    if matched_rows.empty:
        # Fallback to item history if store match isn't present
        matched_rows = df_features[df_features['item_nbr'] == item_nbr]
    
    latest_row = matched_rows.sort_values('date').iloc[-1] if not matched_rows.empty else None
else:
    item_nbr = st.sidebar.number_input("Item ID", value=1496)
    latest_row = None

# Automatically detect perishable status and show historical sales
if latest_row is not None:
    perishable = int(latest_row['perishable'])
    last_sale = latest_row.get('unit_sales', 'N/A')
    st.sidebar.info(f"**Auto-Detected Perishable:** {'Yes' if perishable == 1 else 'No'}\n\n**Last Recorded Daily Sale:** {last_sale} units")
else:
    perishable = 1

target_date = st.sidebar.date_input("Forecast Date")
onpromotion = st.sidebar.selectbox("Is On Promotion?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

API_URL = "http://127.0.0.1:8000/predict"

if st.button("Generate Demand Forecast", type="primary"):
    if latest_row is not None:
        # Convert row to dictionary and strip non-feature columns
        feature_dict = latest_row.to_dict()
        for col in ['date', 'unit_sales', 'id', 'store_nbr', 'item_nbr', 'perishable', 'onpromotion']:
            feature_dict.pop(col, None)
            
        # Clean string/null data
        for k, v in feature_dict.items():
            if isinstance(v, str) or pd.isna(v):
                feature_dict[k] = 0
    else:
        feature_dict = {}

    payload = {
        "store_nbr": int(store_nbr),
        "item_nbr": int(item_nbr),
        "onpromotion": int(onpromotion),
        "perishable": int(perishable),
        "features": feature_dict
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Forecast for {target_date} Generated Successfully!")
            
            col1, col2 = st.columns(2)
            col1.metric("Recommended Stock", f"{result['predicted_unit_sales']} units")
            col2.metric("Target Risk Level (Alpha)", f"{result['quantile_alpha']}")
            
            st.json(result)
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Could not connect to FastAPI server at {API_URL}.")