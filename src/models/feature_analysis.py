import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap
import logging
from datetime import datetime
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.processor import split_features_target

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../logs/feature_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data_and_model():
    """
    Load the data and trained model for feature analysis
    """
    try:
        # Load the data
        data_path = "../../data/processed/statcast_data_processed_latest.csv"
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path} with shape {df.shape}")
        
        # Load the model
        model_path = "../../models/latest/best_model.pkl"
        preprocessor_path = "../../models/latest/preprocessor.pkl"
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Loaded model from {model_path}")
        
        return df, model, preprocessor
    except Exception as e:
        logger.error(f"Error loading data or model: {str(e)}")
        raise

def perform_feature_importance_analysis(model, X, y, preprocessor, feature_names):
    """
    Analyze feature importance using multiple methods
    """
    # Directory for saving feature importance visualizations
    output_dir = "../../reports/feature_importance"
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Preprocess the data
    X_preprocessed = preprocessor.transform(X)

    # Get feature names after preprocessing
    if hasattr(preprocessor, 'get_feature_names_out'):
        all_feature_names = preprocessor.get_feature_names_out()
    else:
        all_feature_names = [f"feature_{i}" for i in range(X_preprocessed.shape[1])]

    # ✅ Wrap into a DataFrame to preserve feature names
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=all_feature_names)
    
    # 1. Built-in feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        
        # Get feature names after preprocessing
        if hasattr(preprocessor, 'get_feature_names_out'):
            all_feature_names = preprocessor.get_feature_names_out()
        else:
            # For older scikit-learn versions
            all_feature_names = [f"feature_{i}" for i in range(X_preprocessed.shape[1])]
        
        # Create DataFrame for feature importances
        importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance': feature_importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Save to CSV
        importance_path = f"{output_dir}/feature_importance_{date_str}.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Saved feature importance to {importance_path}")
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=importance_df.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{output_dir}/feature_importance_plot_{date_str}.png"
        plt.savefig(plot_path)
        logger.info(f"Saved feature importance plot to {plot_path}")
        
        # Return top features for further analysis
        top_features = importance_df['feature'].head(10).tolist()
    else:
        logger.info("Model does not have built-in feature_importances_ attribute")
        top_features = []
    
    # 2. Permutation importance
    try:
        logger.info("Calculating permutation importance...")
        perm_importance = permutation_importance(model, X_preprocessed, y, n_repeats=10, random_state=42)
        
        if hasattr(preprocessor, 'get_feature_names_out'):
            perm_feature_names = preprocessor.get_feature_names_out()
        else:
            perm_feature_names = [f"feature_{i}" for i in range(X_preprocessed.shape[1])]
        
        perm_importance_df = pd.DataFrame({
            'feature': perm_feature_names,
            'importance': perm_importance.importances_mean
        })
        
        perm_importance_df = perm_importance_df.sort_values('importance', ascending=False)
        
        # Save to CSV
        perm_importance_path = f"{output_dir}/permutation_importance_{date_str}.csv"
        perm_importance_df.to_csv(perm_importance_path, index=False)
        logger.info(f"Saved permutation importance to {perm_importance_path}")
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=perm_importance_df.head(20))
        plt.title('Top 20 Permutation Feature Importances')
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{output_dir}/permutation_importance_plot_{date_str}.png"
        plt.savefig(plot_path)
        logger.info(f"Saved permutation importance plot to {plot_path}")
        
        # Update top features
        if not top_features:
            top_features = perm_importance_df['feature'].head(10).tolist()
            
    except Exception as e:
        logger.error(f"Error calculating permutation importance: {str(e)}")
    
    # 3. SHAP analysis:
    try:
        logger.info("Performing SHAP analysis...")
        if X_preprocessed.shape[0] > 1000:
            sample_indices = np.random.choice(X_preprocessed.shape[0], 1000, replace=False)
            if isinstance(X_preprocessed, pd.DataFrame):
                X_sample = X_preprocessed.iloc[sample_indices]
            else:
                X_sample = X_preprocessed[sample_indices]
            y_sample = y[sample_indices]
        else:
            X_sample = X_preprocessed
            y_sample = y
        
        # Create explainer based on model type
        try:
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, X_sample)
                shap_values = explainer(X_sample)

                # Handle multi-class case
                if isinstance(shap_values, list):
                    logger.info("Multi-class SHAP values detected (list format)")
                    plt.figure(figsize=(12, 10))
                    try:
                        shap.summary_plot(
                            shap_values,
                            X_sample,
                            feature_names=perm_feature_names,
                            show=False,
                            check_additivity=False 
                        )
                    except TypeError:
                        shap.summary_plot(
                            shap_values,
                            X_sample,
                            feature_names=perm_feature_names,
                            show=False
                        )
                else:
                    # For models where SHAP returns Explanation object
                    logger.info("Standard SHAP values detected")
                    plt.figure(figsize=(12, 10))
                    try:
                        shap.summary_plot(
                            shap_values,
                            X_sample,
                            feature_names=perm_feature_names,
                            show=False,
                            check_additivity=False 
                        )
                    except TypeError:
                        shap.summary_plot(
                            shap_values,
                            X_sample,
                            feature_names=perm_feature_names,
                            show=False
                        )
            else:
                # TreeExplainer for tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Handle multi-class case
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    logger.info(f"Multi-class SHAP values detected with {len(shap_values)} classes")
                    plt.figure(figsize=(12, 10))
                    try:
                        shap.summary_plot(
                            shap_values,
                            X_sample,
                            feature_names=perm_feature_names,
                            show=False,
                            check_additivity=False 
                        )
                    except TypeError:
                        shap.summary_plot(
                            shap_values,
                            X_sample,
                            feature_names=perm_feature_names,
                            show=False
                        )
                else:
                    plt.figure(figsize=(12, 10))
                    try:
                        shap.summary_plot(
                            shap_values,
                            X_sample,
                            feature_names=perm_feature_names,
                            show=False,
                            check_additivity=False 
                        )
                    except TypeError:
                        shap.summary_plot(
                            shap_values,
                            X_sample,
                            feature_names=perm_feature_names,
                            show=False
                        )
        except Exception as e:
            logger.error(f"Error in SHAP explainer initialization: {str(e)}")
            logger.info("Falling back to simpler SHAP approach")
            # Fallback to a simpler approach
            explainer = shap.Explainer(model.predict, X_sample)
            shap_values = explainer(X_sample)
            plt.figure(figsize=(12, 10))
            try:
                shap.summary_plot(
                    shap_values,
                    X_sample,
                    feature_names=perm_feature_names,
                    show=False,
                    check_additivity=False 
                )
            except TypeError:
                shap.summary_plot(
                    shap_values,
                    X_sample,
                    feature_names=perm_feature_names,
                    show=False
                  )
            
        # Save plot
        shap_summary_path = f"{output_dir}/shap_summary_plot_{date_str}.png"
        plt.savefig(shap_summary_path)
        plt.close()
        logger.info(f"Saved SHAP summary plot to {shap_summary_path}")

    except Exception as e:
        logger.error(f"Error performing SHAP analysis: {str(e)}")
    
    return top_features

def analyze_outcome_specific_features(df, feature_names, label_mapping=None):
    """
    Analyze how features relate to specific outcomes
    """
    output_dir = "../../reports/feature_importance"
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    if label_mapping is None:
        try:
            label_mapping_path = "../../models/result_mapping.npy"
            label_mapping = np.load(label_mapping_path, allow_pickle=True)
            label_mapping = {i: label for i, label in enumerate(label_mapping)}
        except FileNotFoundError:
            logger.warning("Label mapping file not found. Using numeric labels.")
            unique_results = df['result'].unique()
            label_mapping = {i: str(label) for i, label in enumerate(unique_results)}

    if 'result' in df.columns:
        if df['result'].dtype in [np.int64, np.int32, np.float64, int, float]:
            # Ensure mapping covers all values if mapping is used
            if all(item in label_mapping for item in df['result'].unique()):
                df['result_label'] = df['result'].map(label_mapping)
            else:
                logger.warning("Label mapping doesn't cover all numeric result values. Using results directly.")
                df['result_label'] = df['result'].astype(str)
        elif isinstance(df['result'].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(df['result']):
            df['result_label'] = df['result']
        else:
            df['result_label'] = df['result'].astype(str)
    else:
        logger.error("'result' column not found for outcome analysis.")
        return pd.DataFrame() # Return empty if no result column

    if 'result_label' not in df.columns:
        logger.error("Failed to create 'result_label'. Cannot perform outcome analysis.")
        return pd.DataFrame()

    outcomes = df['result_label'].unique()
    outcome_analysis = {}
    summary_rows = []

    logger.info(f"Analyzing features for outcomes: {outcomes}")

    for outcome in outcomes:
        outcome_data = df[df['result_label'] == outcome]
        if outcome_data.empty:
            logger.warning(f"No data found for outcome: {outcome}")
            continue
        outcome_analysis[outcome] = {}

        # Iterate through feature names passed from main
        for feature in feature_names:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                try:
                    stats = {
                        'mean': outcome_data[feature].mean(),
                        'median': outcome_data[feature].median(),
                        'std': outcome_data[feature].std(),
                        'min': outcome_data[feature].min(),
                        'max': outcome_data[feature].max()
                    }
                    outcome_analysis[outcome][feature] = stats
                    summary_rows.append({
                        'outcome': outcome,
                        'feature': feature,
                        **stats # Unpack the stats dictionary
                    })
                except Exception as e:
                    logger.warning(f"Could not calculate stats for numeric feature '{feature}' and outcome '{outcome}': {e}")

    summary_df = pd.DataFrame(summary_rows)

    # Save summary stats CSV (only includes numeric features now)
    if not summary_df.empty:
        summary_path = f"{output_dir}/outcome_feature_analysis_{date_str}.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved outcome-specific numeric feature analysis to {summary_path}")
    else:
        logger.warning("No numeric feature statistics generated for outcomes.")
        
    return summary_df

def select_top_features(importance_df, threshold=0.9):
    """
    Select top features based on cumulative importance
    """
    # Sort by importance if not already sorted
    sorted_df = importance_df.sort_values('importance', ascending=False)
    
    # Calculate cumulative importance
    total_importance = sorted_df['importance'].sum()
    sorted_df['importance_normalized'] = sorted_df['importance'] / total_importance
    sorted_df['cumulative_importance'] = sorted_df['importance_normalized'].cumsum()
    
    # Select features up to threshold
    selected_features = sorted_df[sorted_df['cumulative_importance'] <= threshold]['feature'].tolist()
    
    # Always include at least the top feature
    if not selected_features:
        selected_features = [sorted_df.iloc[0]['feature']]
    
    logger.info(f"Selected {len(selected_features)} features explaining {threshold*100:.1f}% of model importance")
    
    # Save the feature selection results
    output_dir = "../../reports/feature_importance"
    date_str = datetime.now().strftime("%Y-%m-%d")
    selection_path = f"{output_dir}/selected_features_{date_str}.json"
    
    import json
    with open(selection_path, 'w') as f:
        json.dump({
            'selected_features': selected_features,
            'importance_threshold': threshold,
            'feature_count': len(selected_features),
            'date': date_str
        }, f, indent=2)
    
    logger.info(f"Saved selected features to {selection_path}")
    
    return selected_features

def main():
    """
    Main function to perform feature importance analysis
    """
    try:
        # Load data and model
        df, model, preprocessor = load_data_and_model()
        
        # Split features and target
        X, y = split_features_target(df)
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Perform feature importance analysis
        top_features = perform_feature_importance_analysis(model, X, y, preprocessor, feature_names)
        
        # Load label mapping
        try:
            label_mapping_path = "../../models/result_mapping.npy"
            label_mapping = np.load(label_mapping_path, allow_pickle=True)
            label_mapping = {i: label for i, label in enumerate(label_mapping)}
        except FileNotFoundError:
            logger.warning("Label mapping file not found. Proceeding without label mapping.")
            label_mapping = None
        
        # Analyze how features relate to specific outcomes
        summary_df = analyze_outcome_specific_features(df, feature_names, label_mapping)
        
        # Generate feature relationships
        correlation_matrix = df.select_dtypes(include=[np.number]).corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                    vmin=-1, vmax=1, square=True)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # Save correlation matrix
        output_dir = "../../reports/feature_importance"
        corr_matrix_path = f"{output_dir}/correlation_matrix_{datetime.now().strftime('%Y-%m-%d')}.png"
        plt.savefig(corr_matrix_path)
        plt.close()
        logger.info(f"Saved correlation matrix to {corr_matrix_path}")
        
        # Find importance file for feature selection
        importance_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and 'feature_importance' in f]
        if importance_files:
            # Use permutation importance if available, otherwise use the most recent importance file
            perm_files = [f for f in importance_files if 'permutation' in f]
            if perm_files:
                latest_file = max(perm_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
            else:
                latest_file = max(importance_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
            
            importance_df = pd.read_csv(os.path.join(output_dir, latest_file))
            
            # Select features explaining 90% of importance
            selected_features = select_top_features(importance_df, threshold=0.9)
            logger.info(f"Top selected features: {selected_features[:5]}...")
        
        logger.info("Feature analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in feature analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()