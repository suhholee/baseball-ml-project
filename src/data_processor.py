import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseballDataProcessor:
    """
    A class for processing baseball data, particularly for predicting xwOBAcon
    based on bat tracking metrics and quality of contact metrics.
    """
    
    def __init__(self, raw_data_dir='../data/raw', processed_data_dir='../data/processed'):
        """
        Initialize the data processor with directory containing data files
        
        Args:
            raw_data_dir (str): Directory where raw data files are stored
            processed_data_dir (str): Directory where processed data files will be stored
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.merged_data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.preprocessor = None
    
    def load_merged_data(self, filepath=None):
        """
        Load merged bat tracking and quality of contact data
        
        Args:
            filepath (str, optional): Path to the merged data file
                                      If None, uses the latest file in raw_data_dir
        
        Returns:
            pandas.DataFrame: Loaded and merged dataset
        """
        if filepath is None:
            # Find the latest merged data file
            import glob
            merged_files = glob.glob(os.path.join(self.raw_data_dir, "merged_xwobacon_*.csv"))
            if not merged_files:
                logger.error("No merged data files found")
                return None
            
            # Get the most recent file
            filepath = max(merged_files, key=os.path.getctime)
            logger.info(f"Using latest merged data file: {filepath}")
        
        try:
            # Load merged data
            self.merged_data = pd.read_csv(filepath)
            logger.info(f"Loaded merged data: {self.merged_data.shape[0]} rows, {self.merged_data.shape[1]} columns")
            return self.merged_data
        except Exception as e:
            logger.error(f"Error loading merged data: {e}")
            return None
    
    def explore_data(self):
        """
        Perform exploratory data analysis on the merged dataset,
        excluding metrics directly related to or derived from xwOBAcon
        
        Returns:
            dict: Dictionary with EDA results
        """
        if self.merged_data is None:
            logger.error("No data loaded. Call load_merged_data() first.")
            return None
        
        # Dataframe information
        info = {}
        info['shape'] = self.merged_data.shape
        info['columns'] = self.merged_data.columns.tolist()
        
        # Missing values analysis
        missing_values = self.merged_data.isnull().sum()
        info['missing_values'] = missing_values[missing_values > 0].to_dict()
        
        # Basic statistics
        info['numeric_stats'] = self.merged_data.describe().to_dict()
        
        # Define metrics to exclude from correlation analysis (related to xwOBAcon)
        exclude_from_correlation = [
            'xwobacon',         # The target itself
            'xwoba',            # Derived from same components
            'woba',             # Actual outcome related to xwOBAcon
            'est_woba',         # Another expected metric
            'est_ba',           # Expected batting average 
            'est_slg',          # Expected slugging
            'est_woba_minus_woba_diff', # Derived difference
            'est_ba_minus_ba_diff',     # Derived difference
            'est_slg_minus_slg_diff'    # Derived difference
        ]
        
        # Correlation with target variable (xwobacon)
        # Select numeric columns only
        numeric_cols = self.merged_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'xwobacon' in numeric_cols:
            # Calculate correlation with target, excluding related metrics
            correlations = {}
            for col in numeric_cols:
                if col != 'xwobacon' and col not in exclude_from_correlation:
                    # Skip columns with missing values for correlation calculation
                    if self.merged_data[[col, 'xwobacon']].isnull().any().any():
                        cleaned_data = self.merged_data[[col, 'xwobacon']].dropna()
                        if len(cleaned_data) > 5:  # Ensure enough data for correlation
                            corr, p_value = pearsonr(cleaned_data[col], cleaned_data['xwobacon'])
                            correlations[col] = {'correlation': corr, 'p_value': p_value}
                    else:
                        corr, p_value = pearsonr(self.merged_data[col], self.merged_data['xwobacon'])
                        correlations[col] = {'correlation': corr, 'p_value': p_value}
            
            # Sort by absolute correlation value
            correlations = {k: v for k, v in sorted(correlations.items(), 
                                                key=lambda item: abs(item[1]['correlation']), 
                                                reverse=True)}
            info['xwobacon_correlations'] = correlations
        
        return info
    
    def prepare_features_target(self, target_col='xwobacon'):
        """
        Prepare features and target variable from the merged dataset
        
        Args:
            target_col (str): Name of the target column (default: 'xwobacon')
            
        Returns:
            tuple: X (features) and y (target) pandas DataFrames
        """
        if self.merged_data is None:
            logger.error("No data loaded. Call load_merged_data() first.")
            return None, None
        
        # Check if target column exists
        if target_col not in self.merged_data.columns:
            logger.error(f"Target column '{target_col}' not found in dataset")
            return None, None
        
        # Exclude non-feature columns
        exclude_cols = ['id', 'name', 'player_id', 'player_name', 'last_name, first_name',
                        'year', 'season', target_col, 'woba', 'ba', 'slg']
        
        # Also exclude columns derived from target to prevent data leakage
        leakage_cols = ['xwoba', 'est_woba', 'est_woba_minus_woba_diff', 
                        'est_ba', 'est_ba_minus_ba_diff', 
                        'est_slg', 'est_slg_minus_slg_diff']
        
        exclude_cols.extend(leakage_cols)
        
        # Get feature columns
        feature_cols = [col for col in self.merged_data.columns 
                        if col not in exclude_cols and 
                        np.issubdtype(self.merged_data[col].dtype, np.number)]
        
        logger.info(f"Selected {len(feature_cols)} feature columns: {feature_cols}")
        
        # Create feature matrix and target vector
        X = self.merged_data[feature_cols].copy()
        y = self.merged_data[target_col].copy()
        
        # Store feature names for later interpretation
        self.feature_names = feature_cols
        self.X = X
        self.y = y
        
        return X, y
    
    def create_preprocessor(self, handle_missing='drop'):
        """
        Create a preprocessing pipeline for features
        
        Args:
            handle_missing (str): How to handle missing values:
                'drop' - Drop samples with missing values
                'flag' - Add indicator features for missing values
                'none' - Do nothing (model must handle missing values)
                
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        if self.X is None:
            logger.error("Features not prepared. Call prepare_features_target() first.")
            return None
        
        # For numerical features
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        steps = []
        
        # Handle missing values according to strategy
        if handle_missing == 'drop':
            # No transformer needed - we'll drop rows with missing values during fitting
            pass
        elif handle_missing == 'flag':
            # Add indicator variables for missing values
            steps.append(('indicators', SimpleImputer(strategy='constant', fill_value=0, add_indicator=True)))
        elif handle_missing == 'none':
            # Do nothing - some models like XGBoost can handle missing values natively
            pass
        
        numeric_transformer = Pipeline(steps=steps) if steps else None
        
        # Create the column transformer
        if numeric_transformer:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ])
        else:
            # If we're just dropping missing values, no transformer needed
            preprocessor = 'passthrough'
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def engineer_features(self):
        """
        Engineer additional features from existing data
        
        Returns:
            pandas.DataFrame: DataFrame with engineered features
        """
        if self.X is None:
            logger.error("Features not prepared. Call prepare_features_target() first.")
            return None
        
        # Create a copy to avoid modifying the original
        X_engineered = self.X.copy()
        
        # Feature 1: Contact quality - bat speed interaction
        if all(col in X_engineered.columns for col in ['avg_bat_speed', 'contact']):
            X_engineered['bat_speed_contact_ratio'] = X_engineered['avg_bat_speed'] / X_engineered['contact'].replace(0, 1)
        
        # Feature 2: Squared up efficiency 
        if all(col in X_engineered.columns for col in ['squared_up_per_bat_contact', 'squared_up_per_swing']):
            X_engineered['squared_up_efficiency'] = X_engineered['squared_up_per_bat_contact'] * X_engineered['squared_up_per_swing']
        
        # Feature 3: Contact quality composite
        # Combines hard swing rate, squared up rate, and blast rate
        composite_cols = [col for col in ['hard_swing_rate', 'squared_up_per_bat_contact', 'blast_per_bat_contact'] 
                           if col in X_engineered.columns]
        
        if len(composite_cols) >= 2:
            X_engineered['contact_quality_composite'] = X_engineered[composite_cols].mean(axis=1)
        
        # Feature 4: Swing efficiency ratio
        if all(col in X_engineered.columns for col in ['avg_bat_speed', 'swing_length']):
            X_engineered['swing_efficiency_ratio'] = X_engineered['avg_bat_speed'] / X_engineered['swing_length'].replace(0, 1)
        
        # Feature 5: Whiff-to-contact ratio
        if all(col in X_engineered.columns for col in ['whiff_per_swing', 'batted_ball_event_per_swing']):
            X_engineered['whiff_to_contact_ratio'] = X_engineered['whiff_per_swing'] / X_engineered['batted_ball_event_per_swing'].replace(0, 0.001)
        
        # Feature 6: Power potential score
        power_cols = [col for col in ['avg_bat_speed', 'hard_swing_rate', 'blast_per_bat_contact', 'hard_hit_percent'] 
                       if col in X_engineered.columns]
        
        if len(power_cols) >= 2:
            # Normalize each feature to 0-1 scale for this calculation
            normalized_power_features = X_engineered[power_cols].copy()
            for col in power_cols:
                min_val = normalized_power_features[col].min()
                max_val = normalized_power_features[col].max()
                if max_val > min_val:
                    normalized_power_features[col] = (normalized_power_features[col] - min_val) / (max_val - min_val)
                else:
                    normalized_power_features[col] = 0
            
            # Compute power potential score as weighted average
            X_engineered['power_potential_score'] = normalized_power_features.mean(axis=1)
        
        # Feature 7: Sweet spot optimization score (new feature)
        sweet_spot_cols = [col for col in ['sweet_spot_percent', 'barrel_batted_rate'] 
                          if col in X_engineered.columns]
        
        if len(sweet_spot_cols) >= 1:
            X_engineered['sweet_spot_optimization'] = X_engineered[sweet_spot_cols].mean(axis=1)
        
        # Log the engineered features
        new_features = [col for col in X_engineered.columns if col not in self.X.columns]
        logger.info(f"Engineered {len(new_features)} new features: {new_features}")
        
        # Update feature names list
        self.feature_names = X_engineered.columns.tolist()
        self.X = X_engineered
        
        # Save processed data to the processed directory
        if hasattr(self, 'processed_data_dir'):
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            output_path = os.path.join(self.processed_data_dir, f"processed_features_{timestamp}.csv")
            X_engineered.to_csv(output_path, index=False)
            logger.info(f"Saved processed features to {output_path}")
        
        return X_engineered
    
    def visualize_correlations(self, top_n=15):
        """
        Visualize correlations between features and target
        
        Args:
            top_n (int): Number of top correlated features to visualize
            
        Returns:
            matplotlib.figure.Figure: Figure with correlation plots
        """
        if self.X is None or self.y is None:
            logger.error("Features and target not prepared. Call prepare_features_target() first.")
            return None
        
        # Calculate correlations
        data_for_corr = pd.concat([self.X, self.y.rename('xwobacon')], axis=1)
        correlations = data_for_corr.corr()['xwobacon'].drop('xwobacon')
        
        # Sort by absolute correlation value
        correlations = correlations.abs().sort_values(ascending=False)
        top_features = correlations.index[:top_n]
        
        # Create subplot with top features
        fig, axes = plt.subplots(nrows=min(5, len(top_features)), ncols=min(3, len(top_features) // 5 + 1), 
                                figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            if i < len(axes):
                sns.scatterplot(x=self.X[feature], y=self.y, ax=axes[i])
                axes[i].set_title(f'{feature} (corr: {correlations[feature]:.3f})')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('xwOBAcon')
        
        # Remove unused subplots
        for i in range(len(top_features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.processed_data_dir, "feature_correlations.png")
        plt.savefig(output_path, dpi=300)
        logger.info(f"Saved correlation visualization to {output_path}")
        
        return fig
    
    def visualize_feature_importance(self, model, top_n=15):
        """
        Visualize feature importance from a trained model
        
        Args:
            model: Trained model with feature_importances_ attribute
            top_n (int): Number of top features to visualize
            
        Returns:
            matplotlib.figure.Figure: Figure with feature importance plot
        """
        if not hasattr(model, 'feature_importances_'):
            logger.error("Model does not have feature_importances_ attribute")
            return None
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Match importance with feature name
        if len(importances) != len(self.feature_names):
            logger.warning("Number of features in model doesn't match feature names")
            return None
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({'feature': self.feature_names, 'importance': importances})
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
        ax.set_title('Feature Importance for xwOBAcon Prediction')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.processed_data_dir, "feature_importance.png")
        plt.savefig(output_path, dpi=300)
        logger.info(f"Saved feature importance visualization to {output_path}")
        
        return fig
    
    def get_processed_data(self):
        """
        Get the processed features and target
        
        Returns:
            tuple: X (processed features) and y (target)
        """
        return self.X, self.y


if __name__ == "__main__":
    processor = BaseballDataProcessor()
    merged_data = processor.load_merged_data()
    
    if merged_data is not None:
        # Explore data
        eda_results = processor.explore_data()

        # Print dataset shape
        print(f"Dataset shape: {eda_results['shape'][0]} rows, {eda_results['shape'][1]} columns")

        # Print missing values if any
        if eda_results['missing_values']:
            print("Missing values:")
            for col, count in eda_results['missing_values'].items():
                print(f"  {col}: {count}")
        else:
            print("No missing values found in the dataset")

        # Print correlations
        print("\nTop correlations with xwOBAcon:")
        for feature, values in list(eda_results['xwobacon_correlations'].items())[:10]:
            print(f"{feature}: {values['correlation']:.3f} (p-value: {values['p_value']:.3f})")

        # Prepare features and target
        X, y = processor.prepare_features_target()
        
        # Engineer features
        X_engineered = processor.engineer_features()
        
        # Create preprocessor
        preprocessor = processor.create_preprocessor()
        
        print(f"Final dataset: {X_engineered.shape[0]} samples, {X_engineered.shape[1]} features")