import os
import joblib
import pandas as pd
import numpy as np

# Directories
DATA_DIR = 'data'
MODEL_DIR = 'models'

# Load models and feature list
model_perishable = joblib.load(os.path.join(MODEL_DIR, 'model_perishable.pkl'))
model_nonperishable = joblib.load(os.path.join(MODEL_DIR, 'model_nonperishable.pkl'))
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))

# Load dataset for inference batch
df = pd.read_csv(os.path.join(DATA_DIR, 'retail_features.csv'), parse_dates=['date'])

# Encode object/categorical columns (city, state, etc.)
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].astype('category').cat.codes

def generate_inventory_recommendations(df_input):
    """Generates quantile demand predictions for inventory planning."""
    df_sample = df_input.tail(100).copy()  # Infer on latest 100 records
    
    X = df_sample[feature_cols]
    
    preds = []
    for idx, row in df_sample.iterrows():
        is_perishable = row['perishable']
        x_single = X.loc[[idx]]
        
        if is_perishable == 1:
            pred = model_perishable.predict(x_single)[0]
        else:
            pred = model_nonperishable.predict(x_single)[0]
            
        preds.append(max(0.0, float(pred)))
        
    df_sample['recommended_stock'] = np.ceil(preds).astype(int)
    
    output_cols = ['date', 'store_nbr', 'item_nbr', 'perishable', 'unit_sales', 'recommended_stock']
    existing_cols = [c for c in output_cols if c in df_sample.columns]
    
    return df_sample[existing_cols]

if __name__ == '__main__':
    print("=== Running Phase 5 Batch Inference ===")
    results = generate_inventory_recommendations(df)
    
    print("\nSample Recommendations:")
    print(results.head(10).to_string(index=False))
    
    output_path = os.path.join(DATA_DIR, 'inventory_recommendations.csv')
    results.to_csv(output_path, index=False)
    print(f"\nSaved inference recommendations to {output_path}")
    print("Phase 5 inference complete!")