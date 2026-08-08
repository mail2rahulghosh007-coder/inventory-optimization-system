import os
import joblib
import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.xgboost
import dagshub
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Suppress Optuna verbose logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize DagsHub MLflow tracking automatically
dagshub.init(repo_owner='mail2rahulghosh007-coder', repo_name='inventory_optimization_system', mlflow=True)
print("DagsHub MLflow Tracking initialized successfully.")

def pinball_loss(y_true, y_pred, alpha):
    """Calculates Pinball Loss for Quantile Regression."""
    err = y_true - y_pred
    return np.mean(np.maximum(alpha * err, (alpha - 1) * err))

def train_quantile_model_with_optuna(X_train, y_train, X_val, y_val, alpha, model_name, n_trials=15):
    print(f"\n=== Starting Optuna Tuning for {model_name} (alpha={alpha}) ===")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'objective': 'reg:quantileerror',  # Corrected objective function name
            'quantile_alpha': alpha,
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return pinball_loss(y_val, preds, alpha)

    mlflow.set_experiment(f"Favorita_{model_name}_Optimization")
    
    with mlflow.start_run(run_name=f"optuna_study_{model_name}"):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"Best Pinball Loss ({model_name}): {best_score:.4f}")
        
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_pinball_loss", best_score)
        
        final_params = {
            **best_params,
            'objective': 'reg:quantileerror',  # Corrected objective function name
            'quantile_alpha': alpha,
            'random_state': 42,
            'n_jobs': -1
        }
        
        best_model = XGBRegressor(**final_params)
        best_model.fit(X_train, y_train)
        
        # Log model artifact to DagsHub MLflow
        mlflow.xgboost.log_model(best_model, artifact_path=f"model_{model_name.lower()}")
        
        return best_model

if __name__ == '__main__':
    print("Loading retail_features.csv...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'retail_features.csv'), parse_dates=['date'])
    
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].astype('category').cat.codes

    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))
    
    # Train Perishable Model
    df_p = df[df['perishable'] == 1]
    X_p = df_p[feature_cols]
    y_p = df_p['unit_sales']
    X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(X_p, y_p, test_size=0.2, random_state=42)
    
    model_perishable = train_quantile_model_with_optuna(
        X_train_p, y_train_p, X_val_p, y_val_p, 
        alpha=0.250, model_name="PERISHABLE", n_trials=15
    )
    joblib.dump(model_perishable, os.path.join(MODEL_DIR, 'model_perishable.pkl'))
    print("Saved model_perishable.pkl successfully.")
    
    # Train Non-Perishable Model
    df_np = df[df['perishable'] == 0]
    X_np = df_np[feature_cols]
    y_np = df_np['unit_sales']
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
    
    model_nonperishable = train_quantile_model_with_optuna(
        X_train_np, y_train_np, X_val_np, y_val_np, 
        alpha=0.778, model_name="NON_PERISHABLE", n_trials=15
    )
    joblib.dump(model_nonperishable, os.path.join(MODEL_DIR, 'model_nonperishable.pkl'))
    print("Saved model_nonperishable.pkl successfully.")
    
    print("\nPhase 3 Optuna Training and MLflow Logging Complete!")