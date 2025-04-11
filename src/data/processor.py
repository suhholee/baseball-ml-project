import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import re
from datetime import datetime
import glob

def prepare_features(df):
    """
    Process and engineer features for modeling hit outcomes based on
    swing mechanics and pitch characteristics
    """
    # Create a copy of the DataFrame
    processed_df = df.copy()
    
    # Drop unnecessary columns if they exist
    columns_to_drop = ['Rk.', 'Player', 'Team', 'Game Date', 'Vs.', 'Season']
    existing_columns_to_drop = [col for col in columns_to_drop if col in processed_df.columns]

    if existing_columns_to_drop:
        processed_df = processed_df.drop(columns=existing_columns_to_drop)
        print(f"Dropped columns: {', '.join(existing_columns_to_drop)}")

    
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
    
    # Convert numeric columns (excluding EV and LA which we will drop later)
    numeric_columns = ['pitch_velocity', 'perceived_velocity', 'spin_rate', 
                       'vertical_release', 'horizontal_release', 'extension',
                       'arm_angle', 'px', 'pz', 'bat_speed', 'swing_length']
    
    # Convert columns that exist to numeric
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Process the target variable (Result)
    if 'result' in processed_df.columns:
        # Check the unique values and standardize
        unique_results = processed_df['result'].unique()
        print(f"Unique values in 'result' column: {unique_results}")
        
        # Make sure we have no missing results
        initial_rows = processed_df.shape[0]
        processed_df = processed_df.dropna(subset=['result'])
        dropped_rows = initial_rows - processed_df.shape[0]
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with missing result values")
    else:
        print("WARNING: 'result' column not found in the dataset")
    
    # Create derived features based on swing mechanics and pitch characteristics
    # Pitch location categorization (inside/outside, high/low)
    if 'px' in processed_df.columns and 'pz' in processed_df.columns:
        # Horizontal location (px): negative = inside to right-handed batter
        processed_df['inside_pitch'] = processed_df['px'] < 0
        # Vertical location
        processed_df['high_pitch'] = processed_df['pz'] > 3.2
        processed_df['low_pitch'] = processed_df['pz'] < 2.0
    
    # Pitch type grouping using full pitch names directly
    if 'pitch_type' in processed_df.columns:
        # Define categories based on full pitch names
        fastball_types = [
            '4-Seam Fastball', '2-Seam Fastball', 'Cutter', 'Sinker', 'Split-Finger', 'Fastball'
        ]
        breaking_types = [
            'Slider', 'Sweeper', 'Curveball', 'Knuckle Curve', 'Slurve'
        ]
        offspeed_types = [
            'Changeup', 'Forkball', 'Eephus'
        ]

        def categorize_pitch(pitch):
            if pitch in fastball_types:
                return 'Fastball'
            elif pitch in breaking_types:
                return 'Breaking'
            elif pitch in offspeed_types:
                return 'Offspeed'
            else:
                return 'Other'

        # Apply categorization
        processed_df['pitch_category'] = processed_df['pitch_type'].apply(categorize_pitch)
    
    # Swing mechanics features
    if 'swing_length' in processed_df.columns and 'bat_speed' in processed_df.columns:
        # Create efficiency metric
        processed_df['swing_efficiency_ratio'] = processed_df['bat_speed'] / processed_df['swing_length']
    
    # Create interaction features between pitch and swing
    if 'pitch_velocity' in processed_df.columns and 'bat_speed' in processed_df.columns:
        processed_df['speed_differential'] = processed_df['bat_speed'] - processed_df['pitch_velocity']
    
    # Drop columns that tends to predict results more straightforwardly
    columns_to_drop = ['ev', 'adj_ev', 'launch_angle', 'hit_distance', 'xwoba']
    columns_to_drop = [col for col in columns_to_drop if col in processed_df.columns]
    
    if columns_to_drop:
        processed_df = processed_df.drop(columns=columns_to_drop)
        print(f"Dropped columns: {', '.join(columns_to_drop)}")
    
    # Outcome grouping
    if 'result' in processed_df.columns:
        # Create a new column for grouped outcomes
        processed_df['result_grouped'] = processed_df['result']
        
        # Define outcome groups
        # Group 1: Singles
        single_outcomes = ['Single']
        # Group 2: Extra-base hits
        extra_base_outcomes = ['Double', 'Triple'] 
        # Group 3: Home Runs (separate category)
        home_run_outcomes = ['Home Run']
        # Group 3: Outs (regular outs that don't advance runners)
        out_outcomes = ['Field Out', 'Force Out', 'Fielder\'s Choice Out', 'Double Play', 'Grounded Into Double Play', 'Sac Fly Double Play', 'Triple Play', 'Sac Fly', 'Fielder\'s Choice']
        
        # Apply the mapping
        processed_df.loc[processed_df['result'].isin(single_outcomes), 'result_grouped'] = 'Single'
        processed_df.loc[processed_df['result'].isin(extra_base_outcomes), 'result_grouped'] = 'Extra-Base Hit'
        processed_df.loc[processed_df['result'].isin(home_run_outcomes), 'result_grouped'] = 'Home Run'
        processed_df.loc[processed_df['result'].isin(out_outcomes), 'result_grouped'] = 'Out'
        processed_df.loc[processed_df['result'] == 'Field Error', 'result_grouped'] = 'Out'
        
        # Print the distribution of grouped outcomes
        group_counts = processed_df['result_grouped'].value_counts()
        print("\nGrouped outcome distribution:")
        print(group_counts)
        
        # Preserve the original result column as 'result_original'
        processed_df['result_original'] = processed_df['result']
        
        # Replace the result column with the grouped version
        processed_df['result'] = processed_df['result_grouped']
        
        # Drop the intermediate column
        processed_df = processed_df.drop(columns=['result_grouped'])
        
    return processed_df

def create_preprocessor(df, target_col='result'):
    """
    Create a preprocessor for model training
    """
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target variable from features if present
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    elif target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    # Remove non-feature columns
    cols_to_exclude = ['player', 'team', 'opponent', 'game_date', 'result_original']
    categorical_cols = [col for col in categorical_cols if col not in cols_to_exclude]
    
    print(f"Using {len(numerical_cols)} numerical features: {numerical_cols}")
    print(f"Using {len(categorical_cols)} categorical features: {categorical_cols}")
    
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
        ],
        remainder='drop'
    )
    
    return preprocessor, categorical_cols, numerical_cols

def split_features_target(df, target_col='result', min_samples_per_class=5):
    """
    Split DataFrame into features and target with optional class filtering
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Check class distribution before encoding
    class_counts = df[target_col].value_counts()
    print(f"Class distribution before filtering:")
    print(class_counts)
    
    # Find rare classes with fewer than min_samples_per_class
    rare_classes = class_counts[class_counts < min_samples_per_class].index.tolist()
    
    if rare_classes:
        print(f"Found {len(rare_classes)} rare classes with fewer than {min_samples_per_class} samples: {rare_classes}")
        print(f"Filtering out rare classes to ensure proper stratification")
        
        # Filter out instances of rare classes
        df_filtered = df[~df[target_col].isin(rare_classes)]
        
        # Report the difference
        filtered_out = len(df) - len(df_filtered)
        print(f"Removed {filtered_out} instances of rare classes ({filtered_out/len(df)*100:.2f}% of dataset)")
        
        # Use the filtered DataFrame
        df = df_filtered
    
    # Encode the target if it's categorical
    if df[target_col].dtype == 'object' or df[target_col].dtype == 'category':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[target_col])
        
        # Print encoding mapping for reference
        mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        print(f"Target encoding: {mapping}")
        
        # Save the encoder mapping for later use
        result_mapping_path = "../../models/result_mapping.npy"
        os.makedirs(os.path.dirname(result_mapping_path), exist_ok=True)
        np.save(result_mapping_path, label_encoder.classes_)
        print(f"Saved result mapping to {result_mapping_path}")
    else:
        y = df[target_col].values
    
    # Get features (all columns except target and original result)
    drop_cols = [target_col]
    if 'result_original' in df.columns:
        drop_cols.append('result_original')
    
    X = df.drop(columns=drop_cols)
    
    return X, y

def save_processed_data(df, file_path):
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