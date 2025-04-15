import functions_framework
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from google.cloud import storage
import tempfile
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scrape_statcast_data(url):
    """
    Scrape data from Baseball Savant using BeautifulSoup
    """
    # Send request with appropriate headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    logger.info("Fetching data from Baseball Savant...")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching URL: {response.status_code}")
    
    # Parse HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table
    table = soup.find('table', id='search_results')
    if table is None:
        raise Exception("Table with id 'search_results' not found.")
    
    # Get headers from <thead>
    headers = []
    thead = table.find('thead')
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all('th')]
    
    if not headers:
        raise Exception("Table headers not found.")
    
    # Remove empty "Graphs" column if it exists
    if headers[-1] == "Graphs" or headers[-1] == "":
        headers = headers[:-1]
    
    # Get data rows from <tbody>
    rows = []
    tbody = table.find('tbody')
    if tbody:
        for tr in tbody.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all('td')]
            
            # Remove the last column (Graphs) if it exists
            if len(cells) > 0 and len(cells) == len(headers) + 1:
                cells = cells[:-1]
            
            # Skip rows with no data
            if len(cells) == 0 or all(cell == "" for cell in cells):
                continue
            
            # Ensure row length matches header length
            if len(cells) != len(headers):
                logger.warning(f"Row length {len(cells)} does not match header length {len(headers)}. Skipping row.")
                continue
                
            rows.append(cells)
    
    # Create DataFrame
    if not rows:
        raise Exception("No data rows found in the table.")
    
    df = pd.DataFrame(rows, columns=headers)
    
    logger.info(f"Scraped {len(rows)} rows with {len(headers)} columns")
    return df

def clean_statcast_data(df):
    """
    Clean the Statcast data
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Replace empty strings (or whitespace-only strings) with NaN
    df_clean = df_clean.replace(r'^\s*$', pd.NA, regex=True)
    
    # Drop rows that are completely empty (all columns are NaN)
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.dropna(how='all')
    dropped_rows = initial_rows - df_clean.shape[0]
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} empty rows")
    
    # Check for columns with all empty values and drop them
    empty_cols = []
    for col in df_clean.columns:
        # Check if column is all NaN or empty strings
        if df_clean[col].isna().all() or (df_clean[col].astype(str).str.strip() == '').all():
            empty_cols.append(col)
    
    if empty_cols:
        logger.info(f"Dropping empty columns: {', '.join(empty_cols)}")
        df_clean = df_clean.drop(columns=empty_cols)
    
    # Define columns to remove if they exist
    columns_to_remove = [
        "Downward Movement w/ Gravity (in)",
        "Glove/Arm-Side Movement (in)",
        "Vertical Movement w/o Gravity (in)",
        "Movement Toward/Away from Batter (in)"
    ]
    
    # Drop the specified columns
    cols_before = set(df_clean.columns)
    df_clean = df_clean.drop(columns=[col for col in columns_to_remove if col in df_clean.columns])
    cols_after = set(df_clean.columns)
    removed = cols_before - cols_after
    if removed:
        logger.info(f"Removed specific columns: {', '.join(removed)}")
    
    # Additional check for nearly empty columns (>95% missing)
    missing_percentage = df_clean.isna().mean()
    nearly_empty_cols = missing_percentage[missing_percentage > 0.95].index.tolist()
    
    if nearly_empty_cols:
        logger.info(f"Dropping nearly empty columns (>95% missing): {', '.join(nearly_empty_cols)}")
        df_clean = df_clean.drop(columns=nearly_empty_cols)
    
    # Show missing value counts
    missing_counts = df_clean.isna().sum()
    if missing_counts.sum() > 0:
        logger.info("\nMissing values per column:")
        for col, count in missing_counts[missing_counts > 0].items():
            logger.info(f"{col}: {count} missing values")
    
    return df_clean

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
        logger.info(f"Dropped columns: {', '.join(existing_columns_to_drop)}")

    
    # Remove rows with "--" values
    rows_before = processed_df.shape[0]
    for col in processed_df.columns:
        mask = processed_df[col].astype(str) == '--'
        if mask.any():
            logger.info(f"Found {mask.sum()} rows with '--' values in column '{col}'")
            processed_df = processed_df[~mask]
    
    rows_after = processed_df.shape[0]
    if rows_before > rows_after:
        logger.info(f"Removed {rows_before - rows_after} rows with '--' values")
    
    # Rename columns for consistency
    column_mapping = {
        'Pitch (MPH)': 'pitch_velocity',
        'Perceived Velocity': 'perceived_velocity',
        'Spin Rate (RPM)': 'spin_rate',
        'Vertical Release Pt (ft)': 'vertical_release',
        'Horizontal Release Pt (ft)': 'horizontal_release',
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
                       'arm_angle', 'px', 'pz', 'bat_speed', 'swing_length']
    
    # Convert columns that exist to numeric
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Process the target variable (Result)
    if 'result' in processed_df.columns:
        # Check the unique values and standardize
        unique_results = processed_df['result'].unique()
        logger.info(f"Unique values in 'result' column: {unique_results}")
        
        # Make sure we have no missing results
        initial_rows = processed_df.shape[0]
        processed_df = processed_df.dropna(subset=['result'])
        dropped_rows = initial_rows - processed_df.shape[0]
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with missing result values")
    else:
        logger.warning("WARNING: 'result' column not found in the dataset")
    
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
    
    # Drop columns that tend to predict results more straightforwardly
    columns_to_drop = ['ev', 'adj_ev', 'launch_angle', 'hit_distance', 'xwoba']
    columns_to_drop = [col for col in columns_to_drop if col in processed_df.columns]
    
    if columns_to_drop:
        processed_df = processed_df.drop(columns=columns_to_drop)
        logger.info(f"Dropped columns: {', '.join(columns_to_drop)}")
    
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
        # Group 4: Outs (regular outs that don't advance runners)
        out_outcomes = ['Field Out', 'Force Out', 'Fielder\'s Choice Out', 'Double Play', 
                        'Grounded Into Double Play', 'Sac Fly Double Play', 'Triple Play', 
                        'Sac Fly', 'Fielder\'s Choice']
        
        # Apply the mapping
        processed_df.loc[processed_df['result'].isin(single_outcomes), 'result_grouped'] = 'Single'
        processed_df.loc[processed_df['result'].isin(extra_base_outcomes), 'result_grouped'] = 'Extra-Base Hit'
        processed_df.loc[processed_df['result'].isin(home_run_outcomes), 'result_grouped'] = 'Home Run'
        processed_df.loc[processed_df['result'].isin(out_outcomes), 'result_grouped'] = 'Out'
        processed_df.loc[processed_df['result'] == 'Field Error', 'result_grouped'] = 'Out'
        
        # Print the distribution of grouped outcomes
        group_counts = processed_df['result_grouped'].value_counts()
        logger.info("\nGrouped outcome distribution:")
        logger.info(group_counts)
        
        # Preserve the original result column as 'result_original'
        processed_df['result_original'] = processed_df['result']
        
        # Replace the result column with the grouped version
        processed_df['result'] = processed_df['result_grouped']
        
        # Drop the intermediate column
        processed_df = processed_df.drop(columns=['result_grouped'])
        
    return processed_df

def save_to_gcs(df, bucket_name, filename):
    """
    Save DataFrame to Google Cloud Storage
    """
    # Initialize storage client
    storage_client = storage.Client()
    
    # Get bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Create blob
    blob = bucket.blob(filename)
    
    # Convert DataFrame to CSV and upload
    csv_data = df.to_csv(index=False)
    blob.upload_from_string(csv_data, content_type='text/csv')
    
    return f"gs://{bucket_name}/{filename}"

def main(seasons=None):
    """
    Main function to scrape, clean, and save Statcast data
    
    Args:
        seasons: List of seasons to scrape (e.g., [2023, 2024, 2025])
    """
    if seasons is None:
        # Default to current year and previous 2 years
        current_year = datetime.now().year
        seasons = [current_year - 2, current_year - 1, current_year]
    
    # Create empty DataFrames to store combined data
    combined_raw_df = pd.DataFrame()
    combined_clean_df = pd.DataFrame()
    
    for season in seasons:
        # Create base URL for Statcast data for the current season (hits into play)
        url = (f"https://baseballsavant.mlb.com/statcast_search?hfPT=&hfAB=&hfGT=R%7C&hfPR="
               f"hit%5C.%5C.into%5C.%5C.play%7C&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC="
               f"&hfSea={season}%7C&hfSit=&player_type=batter&hfOuts=&hfOpponent=&pitcher_throws="
               f"&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfMo=&hfTeam=&home_road=&hfRO="
               f"&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&metric_1=&group_by=name-event"
               f"&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=fangraphs_est_woba_numer"
               f"&sort_order=desc&chk_event_release_speed=on&chk_event_effective_speed=on"
               f"&chk_event_release_spin_rate=on&chk_event_release_pos_z=on&chk_event_release_pos_x=on"
               f"&chk_event_release_extension=on&chk_event_plate_x=on&chk_event_plate_z=on"
               f"&chk_event_arm_angle=on&chk_event_pitch_name=on&chk_event_api_break_z_with_gravity=on"
               f"&chk_event_api_break_x_arm=on&chk_event_api_break_z_induced=on&chk_event_api_break_x_batter_in=on"
               f"&chk_event_launch_speed=on&chk_event_hyper_speed=on&chk_event_sweetspot_speed_mph=on"
               f"&chk_event_launch_angle=on&chk_event_hit_distance_sc=on&chk_event_swing_length=on"
               f"&chk_event_fangraphs_est_woba_numer=on#results")
        
        try:
            # Scrape the data for this season
            logger.info(f"\nStarting data scraping for {season} season...")
            df = scrape_statcast_data(url)
            logger.info(f"Scraped data for {season} with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Add season column for reference
            df['Season'] = season
            
            # Append to combined raw DataFrame
            combined_raw_df = pd.concat([combined_raw_df, df], ignore_index=True)
            
            # Clean the data
            logger.info(f"\nCleaning data for {season}...")
            df_clean = clean_statcast_data(df)
            
            # Append to combined clean DataFrame
            combined_clean_df = pd.concat([combined_clean_df, df_clean], ignore_index=True)
            logger.info(f"Added {df_clean.shape[0]} rows from {season} to combined dataset")
            
        except Exception as e:
            logger.error(f"Error scraping data for {season}: {str(e)}")
            logger.info("Continuing with other seasons...")
            continue
    
    # Check if we have any data
    if combined_clean_df.empty:
        raise Exception("Failed to scrape data from any season")
    
    logger.info(f"\nFinal combined dataset has {combined_clean_df.shape[0]} rows and {combined_clean_df.shape[1]} columns")
    
    # Process features on the combined clean data
    logger.info("\nProcessing features...")
    processed_df = prepare_features(combined_clean_df)
    logger.info(f"Processed data shape: {processed_df.shape}")
    
    # Count class distribution
    if 'result' in processed_df.columns:
        result_counts = processed_df['result'].value_counts()
        logger.info("\nClass distribution in processed dataset:")
        logger.info(result_counts)
    
    # Create filename with current date and season range
    current_date = time.strftime("%Y-%m-%d")
    season_range = f"{min(seasons)}-{max(seasons)}" if len(seasons) > 1 else str(seasons[0])
    raw_filename = f"statcast_data_{season_range}_{current_date}.csv"
    processed_filename = f"statcast_data_processed_{current_date}.csv"
    
    # Save to GCS 
    bucket_name = "baseball-ml-data"
    
    # Upload raw data
    try:
        # Upload raw data
        raw_gcs_uri = save_to_gcs(combined_raw_df, bucket_name, f"raw/{raw_filename}")
        logger.info(f"Raw data uploaded to {raw_gcs_uri}")
        
        # Upload latest raw data
        latest_raw_gcs_uri = save_to_gcs(combined_raw_df, bucket_name, "latest/statcast_data_latest.csv")
        logger.info(f"Latest raw data uploaded to {latest_raw_gcs_uri}")
        
        # Upload processed data with date
        processed_gcs_uri = save_to_gcs(processed_df, bucket_name, f"processed/{processed_filename}")
        logger.info(f"Processed data uploaded to {processed_gcs_uri}")
        
        # Upload latest processed data
        latest_processed_gcs_uri = save_to_gcs(processed_df, bucket_name, "latest/statcast_data_processed_latest.csv")
        logger.info(f"Latest processed data uploaded to {latest_processed_gcs_uri}")
        
        return {
            "status": "success",
            "message": "Data successfully scraped, processed, and uploaded",
            "raw_data_uri": raw_gcs_uri,
            "processed_data_uri": processed_gcs_uri,
            "rows_scraped": combined_raw_df.shape[0],
            "rows_processed": processed_df.shape[0]
        }
    
    except Exception as e:
        logger.error(f"Error uploading data to GCS: {str(e)}")
        return {
            "status": "error",
            "message": f"Error uploading data to GCS: {str(e)}"
        }

@functions_framework.http
def scrape_baseball_data(request):
    """
    HTTP Cloud Function that scrapes and processes baseball data
    """
    try:
        # Get current year and previous 2 years
        current_year = datetime.now().year
        seasons = [current_year - 2, current_year - 1, current_year]
        
        logger.info(f"Starting data scraping for seasons: {seasons}")
        
        # Run the main function
        result = main(seasons=seasons)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in cloud function: {str(e)}")
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        }

# For local testing
if __name__ == "__main__":
    main(seasons=[2023, 2024, 2025])