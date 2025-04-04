import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, List, Any
import os
import re
from datetime import datetime
import glob

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and engineer features for modeling
    
    Args:
        df: Raw DataFrame with cleaned Statcast data
        
    Returns:
        DataFrame with processed features
    """
    # Create a copy of the DataFrame
    processed_df = df.copy()
    
    # Drop the Rk. column if it exists
    if 'Rk.' in processed_df.columns:
        processed_df = processed_df.drop(columns=['Rk.'])
        print("Dropped 'Rk.' column")
    
    # Remove rows with "--" values
    rows_before = processed_df.shape[0]
    for col in processed_df.columns:
        mask = processed_df[col].astype(str) == '--'
        if mask.any():
            print(f"Found {mask.sum()} rows with '--' values in column '{col}'")
            processed_df = processed_df[~mask]
    
    rows_after = processed_df.shape[0]
    if rows_before > rows_after:
        print(f"Removed {rows_before - rows_after} rows with '--' values")
    
    # Rename columns for consistency
    column_mapping = {
        'Pitch (MPH)': 'pitch_velocity',
        'Perceived Velocity': 'perceived_velocity',
        'Spin Rate (RPM)': 'spin_rate',
        'Vertical Release Pt (ft)': 'vertical_release',
        'Horizontal Release Pt (ft)': 'horizontal_release',
        'Extension (ft)': 'extension',
        'Arm Angle': 'arm_angle',
        'PX (ft)': 'px',
        'PZ (ft)': 'pz',
        'EV (MPH)': 'ev',
        'Adj. EV (MPH)': 'adj_ev',
        'Bat Speed': 'bat_speed',
        'LA (°)': 'launch_angle',
        'Dist (ft)': 'hit_distance',
        'Swing Length (ft)': 'swing_length',
        'Pitch Type': 'pitch_type',
        'Player': 'player',
        'Team': 'team',
        'Vs.': 'opponent',
        'Game Date': 'game_date',
        'xwOBA': 'xwoba',
        'Result': 'result'
    }
    
    # Rename columns that exist in the DataFrame
    existing_columns = {k: v for k, v in column_mapping.items() if k in processed_df.columns}
    processed_df = processed_df.rename(columns=existing_columns)
    
    # Convert numeric columns
    numeric_columns = ['pitch_velocity', 'perceived_velocity', 'spin_rate', 
                       'vertical_release', 'horizontal_release', 'extension',
                       'arm_angle', 'px', 'pz', 'ev', 'adj_ev', 'bat_speed',
                       'launch_angle', 'hit_distance', 'swing_length', 'xwoba']
    
    # Convert columns that exist to numeric
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Convert date to datetime
    if 'game_date' in processed_df.columns:
        processed_df['game_date'] = pd.to_datetime(processed_df['game_date'], errors='coerce')
        
        # Extract date features
        processed_df['month'] = processed_df['game_date'].dt.month
        processed_df['day_of_week'] = processed_df['game_date'].dt.dayofweek
        processed_df['is_weekend'] = processed_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Create derived features
    # Delta EV (exit velocity minus pitch velocity)
    if 'ev' in processed_df.columns and 'pitch_velocity' in processed_df.columns:
        processed_df['delta_ev'] = processed_df['ev'] - processed_df['pitch_velocity']
    
    # Estimated attack angle
    if 'launch_angle' in processed_df.columns and 'pz' in processed_df.columns:
        # Simple approximation: higher pz means steeper approach
        processed_df['est_attack_angle'] = processed_df['launch_angle'] - (10 - processed_df['pz']) * 3
    
    # Contact quality score
    if 'ev' in processed_df.columns and 'launch_angle' in processed_df.columns:
        # Optimal launch angle is around 10-30 degrees for line drives and power
        processed_df['contact_quality'] = processed_df['ev'] * (1 - abs(processed_df['launch_angle'] - 20) / 50)
    
    # Swing efficiency
    if 'ev' in processed_df.columns and 'swing_length' in processed_df.columns:
        # Shorter swing with higher EV means more efficient
        processed_df['swing_efficiency'] = processed_df['ev'] / (processed_df['swing_length'] * 100)
    
    # Drop rows with missing target
    initial_rows = processed_df.shape[0]
    if 'xwoba' in processed_df.columns:
        processed_df = processed_df.dropna(subset=['xwoba'])
        dropped_rows = initial_rows - processed_df.shape[0]
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with missing xwoba values")
    
    return processed_df

def create_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Create a preprocessor for model training
    
    Args:
        df: DataFrame with features
        
    Returns:
        Tuple containing:
        - ColumnTransformer preprocessor
        - List of categorical column names
        - List of numerical column names
    """
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target variable from features if present
    if 'xwoba' in numerical_cols:
        numerical_cols.remove('xwoba')
    
    # Create preprocessing pipelines
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor, categorical_cols, numerical_cols

def split_features_target(df: pd.DataFrame, target_col: str = 'xwoba') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        
    Returns:
        Tuple of (X, y) - features and target
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return X, y

def save_processed_data(df: pd.DataFrame, file_path: str):
    """
    Save processed DataFrame to CSV
    
    Args:
        df: DataFrame to save
        file_path: Path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"Saved processed data to {file_path}")

if __name__ == "__main__":
    # Example usage - load the data and test the functions
    try:
        # Get all files matching the pattern
        input_file_pattern = "../../data/raw/statcast_data_*.csv"
        matching_files = glob.glob(input_file_pattern)
        
        if not matching_files:
            raise FileNotFoundError(f"No files found matching pattern: {input_file_pattern}")
        
        # Print found files
        print(f"Found {len(matching_files)} matching files:")
        for file in matching_files:
            print(f"  - {os.path.basename(file)}")
        
        # Sort files by date in filename
        def extract_date(filename):
            # Try to extract date in format YYYY-MM-DD
            match = re.search(r'statcast_data_(\d{4}-\d{2}-\d{2})\.csv', filename)
            if match:
                date_str = match.group(1)
                try:
                    return datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    # If parsing fails, use file modification time as fallback
                    return datetime.fromtimestamp(os.path.getmtime(filename))
            else:
                # If no date found in filename, use file modification time
                return datetime.fromtimestamp(os.path.getmtime(filename))
        
        # Sort by extracted date (newest first)
        newest_file = max(matching_files, key=extract_date)
        print(f"\nUsing most recent file: {os.path.basename(newest_file)}")
        
        # Extract date from filename for output filename
        date_match = re.search(r'statcast_data_(\d{4}-\d{2}-\d{2})\.csv', newest_file)
        if date_match:
            date_str = date_match.group(1)
            output_file_path = f"../../data/processed/statcast_data_processed_{date_str}.csv"
        else:
            # If no date in filename, use current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            output_file_path = f"../../data/processed/statcast_data_processed_{current_date}.csv"
        
        # Also save as 'latest' for convenience
        latest_output_path = "../../data/processed/statcast_data_processed_latest.csv"
        
        print(f"Loading data from {newest_file}")
        df = pd.read_csv(newest_file)
        
        # Process features
        print("Processing features...")
        processed_df = prepare_features(df)
        print(f"Original data shape: {df.shape}")
        print(f"Processed data shape: {processed_df.shape}")
        
        # Save processed data with date
        save_processed_data(processed_df, output_file_path)
        
        # Also save as latest
        save_processed_data(processed_df, latest_output_path)
        
        # Show sample of processed data
        print("\nSample of processed data:")
        print(processed_df.head(3))
        
        # Create preprocessor
        print("\nCreating preprocessor...")
        preprocessor, cat_cols, num_cols = create_preprocessor(processed_df)
        print(f"Categorical columns: {len(cat_cols)}")
        print(f"Numerical columns: {len(num_cols)}")
        
        # Split features and target
        print("\nSplitting features and target...")
        X, y = split_features_target(processed_df)
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")