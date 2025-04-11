import os
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import sys
import json
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.processor import create_preprocessor, split_features_target

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../logs/bayesian_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """
    Load the processed data for model optimization
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def optimize_random_forest(X_train, X_test, y_train, y_test, n_calls=10):
    """
    Bayesian optimization for RandomForestClassifier
    """
    space = [
        Integer(50, 500, name='n_estimators'),
        Integer(5, 30, name='max_depth'),
        Integer(2, 20, name='min_samples_split'),
        Integer(1, 10, name='min_samples_leaf'),
        Categorical(['sqrt', 'log2', None], name='max_features')
    ]
    
    @use_named_args(space)
    def objective(**params):
        model = RandomForestClassifier(random_state=42, **params)
        model.fit(X_train, y_train)
        return -accuracy_score(y_test, model.predict(X_test))
    
    logger.info("Starting Bayesian optimization for RandomForestClassifier...")
    res_gp = gp_minimize(objective, space, n_calls=n_calls, random_state=42, verbose=True)
    
    # Get the best parameters
    best_params = {
        'n_estimators': res_gp.x[0],
        'max_depth': res_gp.x[1],
        'min_samples_split': res_gp.x[2],
        'min_samples_leaf': res_gp.x[3],
        'max_features': res_gp.x[4]
    }
    
    logger.info(f"Best parameters for RandomForestClassifier: {best_params}")
    logger.info(f"Best accuracy: {-res_gp.fun:.4f}")
    
    # Train model with best parameters
    best_model = RandomForestClassifier(random_state=42, **best_params)
    best_model.fit(X_train, y_train)
    
    return best_model, best_params, -res_gp.fun

def optimize_xgboost(X_train, X_test, y_train, y_test, n_calls=10):
    """
    Bayesian optimization for XGBoostClassifier
    """
    space = [
        Integer(50, 500, name='n_estimators'),
        Real(0.01, 0.3, "log-uniform", name='learning_rate'),
        Integer(3, 15, name='max_depth'),
        Integer(1, 10, name='min_child_weight'),
        Real(0, 1.0, name='gamma'),
        Real(0.5, 1.0, name='subsample'),
        Real(0.5, 1.0, name='colsample_bytree'),
        Real(0, 5.0, name='reg_alpha'),
        Real(0, 5.0, name='reg_lambda')
    ]
    
    @use_named_args(space)
    def objective(**params):
        model = XGBClassifier(random_state=42, **params)
        model.fit(X_train, y_train)
        return -accuracy_score(y_test, model.predict(X_test))
    
    logger.info("Starting Bayesian optimization for XGBoostClassifier...")
    res_gp = gp_minimize(objective, space, n_calls=n_calls, random_state=42, verbose=True)
    
    # Get the best parameters
    best_params = {
        'n_estimators': res_gp.x[0],
        'learning_rate': res_gp.x[1],
        'max_depth': res_gp.x[2],
        'min_child_weight': res_gp.x[3],
        'gamma': res_gp.x[4],
        'subsample': res_gp.x[5],
        'colsample_bytree': res_gp.x[6],
        'reg_alpha': res_gp.x[7],
        'reg_lambda': res_gp.x[8]
    }
    
    logger.info(f"Best parameters for XGBoostClassifier: {best_params}")
    logger.info(f"Best accuracy: {-res_gp.fun:.4f}")
    
    # Train model with best parameters
    best_model = XGBClassifier(random_state=42, **best_params)
    best_model.fit(X_train, y_train)
    
    return best_model, best_params, -res_gp.fun

def optimize_lightgbm(X_train, X_test, y_train, y_test, n_calls=10):
    """
    Bayesian optimization for LGBMClassifier
    """
    space = [
        Integer(50, 500, name='n_estimators'),
        Real(0.01, 0.3, "log-uniform", name='learning_rate'),
        Integer(3, 15, name='max_depth'),
        Integer(20, 150, name='num_leaves'),
        Real(0.5, 1.0, name='subsample'),
        Real(0.5, 1.0, name='colsample_bytree'),
        Real(0, 5.0, name='reg_alpha'),
        Real(0, 5.0, name='reg_lambda')
    ]
    
    @use_named_args(space)
    def objective(**params):
        model = LGBMClassifier(random_state=42, **params)
        model.fit(X_train, y_train)
        return -accuracy_score(y_test, model.predict(X_test))
    
    logger.info("Starting Bayesian optimization for LGBMClassifier...")
    res_gp = gp_minimize(objective, space, n_calls=n_calls, random_state=42, verbose=True)
    
    # Get the best parameters
    best_params = {
        'n_estimators': res_gp.x[0],
        'learning_rate': res_gp.x[1],
        'max_depth': res_gp.x[2],
        'num_leaves': res_gp.x[3],
        'subsample': res_gp.x[4],
        'colsample_bytree': res_gp.x[5],
        'reg_alpha': res_gp.x[6],
        'reg_lambda': res_gp.x[7]
    }
    
    logger.info(f"Best parameters for LGBMClassifier: {best_params}")
    logger.info(f"Best accuracy: {-res_gp.fun:.4f}")
    
    # Train model with best parameters
    best_model = LGBMClassifier(random_state=42, **best_params)
    best_model.fit(X_train, y_train)
    
    return best_model, best_params, -res_gp.fun

def optimize_catboost(X_train, X_test, y_train, y_test, n_calls=10):
    """
    Bayesian optimization for CatBoostClassifier
    """
    space = [
        Integer(50, 500, name='iterations'),
        Real(0.01, 0.3, "log-uniform", name='learning_rate'),
        Integer(3, 10, name='depth'),
        Real(1, 10, name='l2_leaf_reg'),
        Categorical(['Bayesian', 'Bernoulli', 'MVS'], name='bootstrap_type'),
        Categorical(['SymmetricTree', 'Depthwise', 'Lossguide'], name='grow_policy')
    ]
    
    @use_named_args(space)
    def objective(iterations, learning_rate, depth, l2_leaf_reg, bootstrap_type, grow_policy):
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            bootstrap_type=bootstrap_type,
            grow_policy=grow_policy,
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        return -accuracy_score(y_test, model.predict(X_test))
    
    logger.info("Starting Bayesian optimization for CatBoostClassifier...")
    res_gp = gp_minimize(objective, space, n_calls=n_calls, random_state=42, verbose=True)
    
    # Get the best parameters
    best_params = {
        'iterations': res_gp.x[0],
        'learning_rate': res_gp.x[1],
        'depth': res_gp.x[2],
        'l2_leaf_reg': res_gp.x[3],
        'bootstrap_type': res_gp.x[4],
        'grow_policy': res_gp.x[5]
    }
    
    logger.info(f"Best parameters for CatBoostClassifier: {best_params}")
    logger.info(f"Best accuracy: {-res_gp.fun:.4f}")
    
    # Train model with best parameters
    best_model = CatBoostClassifier(random_state=42, verbose=0, **best_params)
    best_model.fit(X_train, y_train)
    
    return best_model, best_params, -res_gp.fun

def save_optimized_model(model, model_name, preprocessor, params, accuracy, date_str=None):
    """
    Save the optimized model, preprocessor, and hyperparameters
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Create directories if they don't exist
    model_dir = f"../../models/optimized/{date_str}"
    hyperparams_dir = f"../../models/hyperparams"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(hyperparams_dir, exist_ok=True)
    
    # Save model
    model_path = f"{model_dir}/{model_name}.pkl"
    preprocessor_path = f"{model_dir}/preprocessor.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Convert NumPy data types to Python native types for JSON serialization
    serializable_params = {}
    for k, v in params.items():
        if isinstance(v, (np.int64, np.int32)):
            serializable_params[k] = int(v)
        elif isinstance(v, (np.float64, np.float32)):
            serializable_params[k] = float(v)
        else:
            serializable_params[k] = v
    
    # Save hyperparameters
    hyperparam_info = {
        'model_name': model_name,
        'date': date_str,
        'params': serializable_params,
        'accuracy': float(accuracy) if isinstance(accuracy, (np.float64, np.float32)) else accuracy
    }
    
    hyperparam_path = f"{hyperparams_dir}/{model_name}_{date_str}.json"
    with open(hyperparam_path, 'w') as f:
        json.dump(hyperparam_info, f, indent=2)
    
    logger.info(f"Saved optimized model to {model_path}")
    logger.info(f"Saved hyperparameters to {hyperparam_path}")

def main():
    """
    Main function to perform Bayesian optimization of models
    """
    # Load the data
    data_path = "../../data/processed/statcast_data_processed_latest.csv"
    df = load_data(data_path)
    
    # Create preprocessor
    preprocessor, _, _ = create_preprocessor(df)
    
    # Split features and target
    X, y = split_features_target(df)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocess the data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    logger.info(f"Preprocessed data shapes:")
    logger.info(f"X_train: {X_train_preprocessed.shape}")
    logger.info(f"X_test: {X_test_preprocessed.shape}")
    
    # Extract date from the data file name for model versioning
    import re
    date_match = re.search(r'statcast_data_processed_(\d{4}-\d{2}-\d{2})\.csv', data_path)
    if date_match:
        date_str = date_match.group(1)
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Optimize RandomForestClassifier
    rf_model, rf_params, rf_accuracy = optimize_random_forest(
        X_train_preprocessed, X_test_preprocessed, y_train, y_test
    )
    save_optimized_model(rf_model, "random_forest", preprocessor, rf_params, rf_accuracy, date_str)
    
    # Optimize XGBoost
    xgb_model, xgb_params, xgb_accuracy = optimize_xgboost(
        X_train_preprocessed, X_test_preprocessed, y_train, y_test
    )
    save_optimized_model(xgb_model, "xgboost", preprocessor, xgb_params, xgb_accuracy, date_str)
    
    # Optimize LightGBM
    lgbm_model, lgbm_params, lgbm_accuracy = optimize_lightgbm(
        X_train_preprocessed, X_test_preprocessed, y_train, y_test
    )
    save_optimized_model(lgbm_model, "light_gbm", preprocessor, lgbm_params, lgbm_accuracy, date_str)

    # Optimize catboost
    cat_model, cat_params, cat_accuracy = optimize_catboost(
        X_train_preprocessed, X_test_preprocessed, y_train, y_test
    )
    save_optimized_model(cat_model, "catboost", preprocessor, cat_params, cat_accuracy, date_str)
    
    # Compare models and save the best one
    models = {
        'random_forest': (rf_model, rf_accuracy),
        'xgboost': (xgb_model, xgb_accuracy),
        'light_gbm': (lgbm_model, lgbm_accuracy),
        'catboost': (cat_model, cat_accuracy)
    }
    
    best_model_name = max(models, key=lambda k: models[k][1])
    best_model, best_accuracy = models[best_model_name]
    
    # Save the best model as the "latest"
    latest_model_path = f"../../models/latest/best_model.pkl"
    latest_preprocessor_path = f"../../models/latest/preprocessor.pkl"
    latest_model_info_path = f"../../models/latest/model_info.json"
    
    os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)
    
    joblib.dump(best_model, latest_model_path)
    joblib.dump(preprocessor, latest_preprocessor_path)
    
    model_info = {
        'model_name': best_model_name,
        'date': date_str,
        'accuracy': best_accuracy
    }
    
    with open(latest_model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")
    logger.info(f"Saved best model to {latest_model_path}")
    
    # Create comparison report
    comparison = pd.DataFrame({
        'random_forest': {'accuracy': rf_accuracy, 'params': str(rf_params)},
        'xgboost': {'accuracy': xgb_accuracy, 'params': str(xgb_params)},
        'light_gbm': {'accuracy': lgbm_accuracy, 'params': str(lgbm_params)},
        'catboost': {'accuracy': cat_accuracy, 'params': str(cat_params)},
    }).T
    
    comparison_path = f"../../reports/hyperparameter_optimization_{date_str}.csv"
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    comparison.to_csv(comparison_path)
    
    logger.info(f"Saved model comparison to {comparison_path}")

if __name__ == "__main__":
    main()