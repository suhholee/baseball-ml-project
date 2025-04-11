import os
import pandas as pd
import numpy as np
import joblib
import json
import logging
import glob
import re
from datetime import datetime
import sys
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from google.cloud import storage

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.processor import create_preprocessor, split_features_target

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../logs/final_model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_latest_processed_data():
    """
    Load the most recent processed dataset
    """
    try:
        # Path to latest processed data
        data_path = "../../data/processed/statcast_data_processed_latest.csv"
        
        # Check if file exists
        if not os.path.exists(data_path):
            # Try to find any processed data file
            processed_files = glob.glob("../../data/processed/statcast_data_processed_*.csv")
            if not processed_files:
                raise FileNotFoundError("No processed data files found")
            
            # Sort by date in filename (newest first)
            def extract_date(filename):
                match = re.search(r'statcast_data_processed_(\d{4}-\d{2}-\d{2})\.csv', filename)
                if match:
                    try:
                        return datetime.strptime(match.group(1), '%Y-%m-%d')
                    except ValueError:
                        return datetime.fromtimestamp(os.path.getmtime(filename))
                else:
                    return datetime.fromtimestamp(os.path.getmtime(filename))
            
            data_path = max(processed_files, key=extract_date)
        
        # Load the data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded processed data from {data_path} with shape {df.shape}")
        return df, data_path
    
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        raise

def load_best_hyperparameters():
    """
    Load the best hyperparameters from optimization results
    """
    try:
        # Check for model info in the latest directory
        model_info_path = "../../models/latest/model_info.json"
        
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            best_model_name = model_info.get('model_name')
            logger.info(f"Best model according to model_info.json: {best_model_name}")
            
            # Find the corresponding hyperparameters file
            hyperparams_dir = "../../models/hyperparams"
            if os.path.exists(hyperparams_dir):
                # Find hyperparameter files for the best model
                hyperparam_files = glob.glob(f"{hyperparams_dir}/{best_model_name}_*.json")
                
                if hyperparam_files:
                    # Get the most recent hyperparameter file
                    latest_hyperparam_file = max(hyperparam_files, key=os.path.getmtime)
                    
                    with open(latest_hyperparam_file, 'r') as f:
                        hyperparams = json.load(f)
                    
                    logger.info(f"Loaded hyperparameters from {latest_hyperparam_file}")
                    return hyperparams
                else:
                    logger.warning(f"No hyperparameter files found for {best_model_name}")
            else:
                logger.warning("Hyperparameters directory not found")
        
        # If we couldn't find the best model hyperparameters, look for any hyperparameter file
        hyperparams_dir = "../../models/hyperparams"
        if os.path.exists(hyperparams_dir):
            hyperparam_files = glob.glob(f"{hyperparams_dir}/*.json")
            
            if hyperparam_files:
                # Get the most recent hyperparameter file
                latest_hyperparam_file = max(hyperparam_files, key=os.path.getmtime)
                
                with open(latest_hyperparam_file, 'r') as f:
                    hyperparams = json.load(f)
                
                logger.info(f"Loaded hyperparameters from {latest_hyperparam_file}")
                return hyperparams
        
        # If we couldn't find any hyperparameters, return default
        logger.warning("No hyperparameter files found, using default configuration")
        return {
            'model_name': 'xgboost',
            'params': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 10,
                'min_child_weight': 3,
                'gamma': 0.2,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        }
    
    except Exception as e:
        logger.error(f"Error loading hyperparameters: {str(e)}")
        # Return default configuration
        return {
            'model_name': 'xgboost',
            'params': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 10,
                'min_child_weight': 3,
                'gamma': 0.2,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        }

def load_selected_features():
    """
    Load the selected features from feature analysis
    """
    try:
        # Check for selected features file
        feature_dir = "../../reports/feature_importance"
        if os.path.exists(feature_dir):
            feature_files = glob.glob(f"{feature_dir}/selected_features_*.json")
            
            if feature_files:
                # Get the most recent file
                latest_feature_file = max(feature_files, key=os.path.getmtime)
                
                with open(latest_feature_file, 'r') as f:
                    feature_data = json.load(f)
                
                selected_features = feature_data.get('selected_features', [])
                logger.info(f"Loaded {len(selected_features)} selected features from {latest_feature_file}")
                
                # Filter out 'Season' feature if present
                if 'Season' in selected_features:
                    selected_features.remove('Season')
                    logger.info("Removed 'Season' from selected features")
                
                return selected_features
            else:
                logger.warning("No selected features files found")
                return None
        else:
            logger.warning("Feature importance directory not found")
            return None
    
    except Exception as e:
        logger.error(f"Error loading selected features: {str(e)}")
        return None

def create_model(model_name, params):
    """
    Create a model instance with the specified parameters
    """
    model_name = model_name.lower()
    
    if model_name == 'random_forest':
        return RandomForestClassifier(random_state=42, **params)
    elif model_name == 'xgboost':
        return XGBClassifier(random_state=42, **params)
    elif model_name == 'light_gbm' or model_name == 'lightgbm':
        return LGBMClassifier(random_state=42, **params)
    elif model_name == 'catboost':
        return CatBoostClassifier(random_state=42, verbose=0, **params)
    else:
        logger.warning(f"Unknown model type: {model_name}. Using XGBoost instead.")
        return XGBClassifier(random_state=42)

def filter_dataframe_by_features(df, selected_features, target_col='result'):
    """
    Filter DataFrame to include only selected features and target
    """
    if selected_features:
        # Ensure target_col is not in selected_features
        features = [f for f in selected_features if f != target_col]
        
        # Check if features exist in the DataFrame
        existing_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Some selected features are not in the DataFrame: {missing_features}")
        
        if not existing_features:
            logger.warning("None of the selected features are in the DataFrame. Using all columns.")
            return df
        
        # Include target column
        columns_to_keep = existing_features + [target_col]
        
        # Filter DataFrame
        filtered_df = df[columns_to_keep]
        logger.info(f"Filtered DataFrame to {len(existing_features)} features plus target")
        return filtered_df
    else:
        logger.info("No feature selection applied, using all columns")
        return df

def save_to_gcs(model, preprocessor, bucket_name, model_prefix='models', preprocessor_prefix='preprocessors'):
    """
    Save model and preprocessor to Google Cloud Storage
    """
    try:
        # Initialize storage client
        storage_client = storage.Client()
        
        # Get bucket
        bucket = storage_client.bucket(bucket_name)
        
        # Create blob names with current date
        date_str = datetime.now().strftime("%Y-%m-%d")
        model_blob_name = f"{model_prefix}/final_model_{date_str}.pkl"
        preprocessor_blob_name = f"{preprocessor_prefix}/preprocessor_{date_str}.pkl"
        
        # Also create "latest" blob names
        model_latest_blob_name = f"{model_prefix}/final_model_latest.pkl"
        preprocessor_latest_blob_name = f"{preprocessor_prefix}/preprocessor_latest.pkl"
        
        # Save model and preprocessor locally first
        local_model_path = "../../models/final/final_model.pkl"
        local_preprocessor_path = "../../models/final/preprocessor.pkl"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
        
        # Save locally
        joblib.dump(model, local_model_path)
        joblib.dump(preprocessor, local_preprocessor_path)
        
        # Create blobs and upload
        model_blob = bucket.blob(model_blob_name)
        preprocessor_blob = bucket.blob(preprocessor_blob_name)
        model_latest_blob = bucket.blob(model_latest_blob_name)
        preprocessor_latest_blob = bucket.blob(preprocessor_latest_blob_name)
        
        # Upload
        model_blob.upload_from_filename(local_model_path)
        preprocessor_blob.upload_from_filename(local_preprocessor_path)
        model_latest_blob.upload_from_filename(local_model_path)
        preprocessor_latest_blob.upload_from_filename(local_preprocessor_path)
        
        logger.info(f"Model uploaded to gs://{bucket_name}/{model_blob_name}")
        logger.info(f"Preprocessor uploaded to gs://{bucket_name}/{preprocessor_blob_name}")
        logger.info(f"Model (latest) uploaded to gs://{bucket_name}/{model_latest_blob_name}")
        logger.info(f"Preprocessor (latest) uploaded to gs://{bucket_name}/{preprocessor_latest_blob_name}")
        
        return {
            'model_gcs_uri': f"gs://{bucket_name}/{model_blob_name}",
            'preprocessor_gcs_uri': f"gs://{bucket_name}/{preprocessor_blob_name}",
            'model_latest_gcs_uri': f"gs://{bucket_name}/{model_latest_blob_name}",
            'preprocessor_latest_gcs_uri': f"gs://{bucket_name}/{preprocessor_latest_blob_name}"
        }
    
    except Exception as e:
        logger.error(f"Error saving to GCS: {str(e)}")
        return None

def train_and_evaluate_final_model(df, hyperparams, selected_features=None):
    """
    Train the final model with optimized hyperparameters and selected features
    """
    try:
        # Filter DataFrame by selected features if provided
        if selected_features:
            filtered_df = filter_dataframe_by_features(df, selected_features)
        else:
            filtered_df = df
        
        # Create preprocessor
        preprocessor, cat_cols, num_cols = create_preprocessor(filtered_df)
        
        # Split features and target
        X, y = split_features_target(filtered_df)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Preprocess the data
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
        
        # Create model
        model_name = hyperparams.get('model_name', 'xgboost')
        params = hyperparams.get('params', {})
        
        logger.info(f"Creating {model_name} model with optimized parameters")
        model = create_model(model_name, params)
        
        # Train model
        logger.info("Training final model...")
        model.fit(X_train_preprocessed, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_preprocessed)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Final model accuracy: {accuracy:.4f}")
        
        # Generate classification report
        try:
            # Try to load label mapping
            label_mapping_path = "../../models/result_mapping.npy"
            if os.path.exists(label_mapping_path):
                label_mapping = np.load(label_mapping_path, allow_pickle=True)
                target_names = [str(label) for label in label_mapping]
            else:
                target_names = None
            
            report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            
            # Save report
            report_dir = "../../reports/final_model"
            os.makedirs(report_dir, exist_ok=True)
            
            date_str = datetime.now().strftime("%Y-%m-%d")
            report_path = f"{report_dir}/final_model_report_{date_str}.csv"
            report_df.to_csv(report_path)
            
            logger.info(f"Saved classification report to {report_path}")
            
            # Generate confusion matrix
            fig, ax = plt.subplots(figsize=(12, 10))
            
            if target_names:
                disp = ConfusionMatrixDisplay.from_predictions(
                    y_test, y_pred, display_labels=target_names,
                    xticks_rotation=45, ax=ax
                )
            else:
                disp = ConfusionMatrixDisplay.from_predictions(
                    y_test, y_pred, xticks_rotation=45, ax=ax
                )
            
            plt.tight_layout()
            
            cm_path = f"{report_dir}/final_model_confusion_matrix_{date_str}.png"
            plt.savefig(cm_path)
            
            logger.info(f"Saved confusion matrix to {cm_path}")
        
        except Exception as e:
            logger.error(f"Error generating classification report: {str(e)}")
        
        # Save model and preprocessor
        model_dir = "../../models/final"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = f"{model_dir}/final_model.pkl"
        preprocessor_path = f"{model_dir}/preprocessor.pkl"
        
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        
        logger.info(f"Saved final model to {model_path}")
        logger.info(f"Saved preprocessor to {preprocessor_path}")
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_features': X_train_preprocessed.shape[1],
            'num_samples': X_train.shape[0],
            'params': params,
            'selected_features': selected_features
        }
        
        metadata_path = f"{model_dir}/final_model_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved model metadata to {metadata_path}")
        
        # Return model and preprocessor
        return model, preprocessor, accuracy
    
    except Exception as e:
        logger.error(f"Error training final model: {str(e)}")
        raise

def main():
    """
    Main function to train the final model
    """
    try:
        # Load the processed data
        df, data_path = load_latest_processed_data()
        
        # Remove Season column if it exists
        if 'Season' in df.columns:
            df = df.drop(columns=['Season'])
            logger.info("Removed 'Season' column from the dataset")
        
        # Load hyperparameters
        hyperparams = load_best_hyperparameters()
        
        # Load selected features
        selected_features = load_selected_features()
        
        # Train final model
        model, preprocessor, accuracy = train_and_evaluate_final_model(df, hyperparams, selected_features)
        
        # Save to GCS (optional)
        try:
            gcs_bucket_name = "baseball-ml-data"
            gcs_uris = save_to_gcs(model, preprocessor, gcs_bucket_name)
            
            if gcs_uris:
                logger.info("Model and preprocessor successfully uploaded to GCS")
                
                # Save GCS URIs as metadata
                metadata_path = "../../models/final/gcs_uris.json"
                with open(metadata_path, 'w') as f:
                    json.dump(gcs_uris, f, indent=2)
                
                logger.info(f"Saved GCS URIs to {metadata_path}")
        except Exception as e:
            logger.warning(f"Could not save to GCS (this is optional): {str(e)}")
        
        logger.info("Final model training completed successfully!")
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()