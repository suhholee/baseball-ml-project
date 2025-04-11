import os
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib
from datetime import datetime
import logging
import os
import sys
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.processor import create_preprocessor, split_features_target

# Configure logging
log_dir = "../../logs"
os.makedirs(log_dir, exist_ok=True) 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../logs/model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """
    Load the processed data for model training
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, label_mapping=None):
    """
    Train and evaluate a single model
    """
    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        # Log model parameters
        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)
        
        # Train the model
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        # Generate classification report
        if label_mapping is not None:
            target_names = [label_mapping[i] for i in range(len(label_mapping))]
            report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        else:
            report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log the model
        mlflow.sklearn.log_model(model, model_name)
        
        # Create and save confusion matrix
        cm_path = f"../../reports/figures/{model_name}_confusion_matrix.png"
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        
        from sklearn.metrics import ConfusionMatrixDisplay
        fig, ax = plt.subplots(figsize=(12, 10))
        if label_mapping is not None:
            display_labels = [label_mapping[i] for i in range(len(label_mapping))]
            disp = ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, display_labels=display_labels, 
                xticks_rotation=45, ax=ax
            )
        else:
            disp = ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, xticks_rotation=45, ax=ax
            )
        plt.tight_layout()
        plt.savefig(cm_path)
        
        # Log the figure as an artifact
        mlflow.log_artifact(cm_path)
        
        return model, accuracy, report

def get_model_configurations():
    """
    Create configurations for multiple models with expanded hyperparameter sets
    """
    model_configs = {
        'random_forest': {
            'model': RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                random_state=42
            )
        },
        'xgboost': {
            'model': XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=10,
                min_child_weight=3,
                gamma=0.2,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )
        },
        'lightgbm': {
            'model': LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=10,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )
        },
        'catboost': {
            'model': CatBoostClassifier(
                iterations=200,
                learning_rate=0.1,
                depth=8,
                l2_leaf_reg=3,
                bootstrap_type='Bayesian',
                grow_policy='SymmetricTree',
                random_state=42,
                verbose=0
            )
        }
    }
    
    return model_configs

def train_models(X_train, X_test, y_train, y_test, label_mapping=None):
    """
    Train multiple models and compare their performance
    """
    # Get model configurations
    model_configs = get_model_configurations()
    
    results = {}
    best_accuracy = 0
    best_model = None
    best_model_name = None
    
    for name, config in model_configs.items():
        try:
            model = config['model']
            trained_model, accuracy, report = train_and_evaluate_model(
                model, X_train, X_test, y_train, y_test, name, label_mapping
            )
            
            results[name] = {
                'model': trained_model,
                'accuracy': accuracy,
                'report': report
            }
            
            logger.info(f"{name} accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = trained_model
                best_model_name = name
        except Exception as e:
            logger.error(f"Error training {name} model: {str(e)}")
    
    logger.info(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")
    return results, best_model, best_model_name

def save_model(model, model_name, preprocessor, date_str=None):
    """
    Save the trained model and preprocessor
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Create directories if they don't exist
    os.makedirs(f"../../models/{date_str}", exist_ok=True)
    os.makedirs("../../models/latest", exist_ok=True)
    
    # Save model with date
    model_path = f"../../models/{date_str}/{model_name}.pkl"
    preprocessor_path = f"../../models/{date_str}/preprocessor.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Also save as latest
    latest_model_path = f"../../models/latest/{model_name}.pkl"
    latest_preprocessor_path = f"../../models/latest/preprocessor.pkl"
    
    joblib.dump(model, latest_model_path)
    joblib.dump(preprocessor, latest_preprocessor_path)
    
    logger.info(f"Saved model to {model_path} and {latest_model_path}")
    logger.info(f"Saved preprocessor to {preprocessor_path} and {latest_preprocessor_path}")

def main():
    """
    Main function to train and evaluate models
    """
    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("baseball-hit-prediction")
    
    # Load the data
    data_path = "../../data/processed/statcast_data_processed_latest.csv"
    df = load_data(data_path)
    
    # Load the target label mapping
    try:
        label_mapping_path = "../../models/result_mapping.npy"
        label_mapping = np.load(label_mapping_path, allow_pickle=True)
        label_mapping = {i: label for i, label in enumerate(label_mapping)}
        logger.info(f"Loaded label mapping: {label_mapping}")
    except FileNotFoundError:
        logger.warning("Label mapping file not found. Proceeding without label mapping.")
        label_mapping = None
    
    # Create preprocessor
    preprocessor, cat_cols, num_cols = create_preprocessor(df)
    
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
    
    # Train and evaluate models
    results, best_model, best_model_name = train_models(
        X_train_preprocessed, X_test_preprocessed, y_train, y_test, label_mapping
    )
    
    # Extract date from the data file name for model versioning
    import re
    date_match = re.search(r'statcast_data_processed_(\d{4}-\d{2}-\d{2})\.csv', data_path)
    if date_match:
        date_str = date_match.group(1)
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Save the best model
    save_model(best_model, best_model_name, preprocessor, date_str)
    
    # Generate model comparison report
    report_path = f"../../reports/model_comparison_{date_str}.csv"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    comparison = pd.DataFrame({
        name: {
            'accuracy': info['accuracy'],
            'precision': info['report']['weighted avg']['precision'],
            'recall': info['report']['weighted avg']['recall'],
            'f1': info['report']['weighted avg']['f1-score']
        }
        for name, info in results.items()
    }).T
    
    comparison.to_csv(report_path)
    logger.info(f"Saved model comparison to {report_path}")
    
    return results

if __name__ == "__main__":
    main()