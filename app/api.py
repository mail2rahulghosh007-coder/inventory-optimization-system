import os
import math
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Favorita Demand Forecasting API",
    description="Quantile Regression API for Perishable and Non-Perishable retail items.",
    version="1.1"
)

MODEL_DIR = 'models'

# Load models and features at startup
try:
    model_perishable = joblib.load(os.path.join(MODEL_DIR, 'model_perishable.pkl'))
    model_nonperishable = joblib.load(os.path.join(MODEL_DIR, 'model_nonperishable.pkl'))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))
except Exception as e:
    raise RuntimeError(f"Failed to load models or feature list: {str(e)}")

# Request Body Schema
class ForecastRequest(BaseModel):
    store_nbr: int
    item_nbr: int
    onpromotion: int
    perishable: int
    features: dict  # Dynamic key-value pairs for remaining features

@app.get("/")
def health_check():
    return {"status": "online", "message": "Favorita Forecasting API is up and running!"}

@app.post("/predict")
def predict_demand(payload: ForecastRequest):
    try:
        data = payload.dict()
        is_perishable = data['perishable']
        input_features = data['features']
        
        # Base input values
        input_features['store_nbr'] = data['store_nbr']
        input_features['item_nbr'] = data['item_nbr']
        input_features['onpromotion'] = data['onpromotion']
        input_features['perishable'] = is_perishable

        # Convert to DataFrame
        input_df = pd.DataFrame([input_features])
        
        # Fill missing features if any
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
                
        input_df = input_df[feature_cols]

        # Predict using appropriate quantile model
        if is_perishable == 1:
            raw_pred = model_perishable.predict(input_df)[0]
            alpha_used = 0.25
        else:
            raw_pred = model_nonperishable.predict(input_df)[0]
            alpha_used = 0.778

        # Convert float to positive integer (Ceiling value for inventory safety stock)
        recommended_stock = math.ceil(max(0.0, float(raw_pred)))

        return {
            "store_nbr": data['store_nbr'],
            "item_nbr": data['item_nbr'],
            "perishable": is_perishable,
            "quantile_alpha": alpha_used,
            "predicted_unit_sales": int(recommended_stock)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))