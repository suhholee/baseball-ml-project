import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from typing import Optional
from google.cloud import storage

def scrape_statcast_data(url: str) -> pd.DataFrame:
    """
    Scrape data from Baseball Savant using BeautifulSoup
    
    Args:
        url: The URL to scrape
        
    Returns:
        DataFrame containing the scraped data
    """
    # Send request with appropriate headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print("Fetching data from Baseball Savant...")
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
                print(f"Warning: Row length {len(cells)} does not match header length {len(headers)}. Skipping row.")
                continue
                
            rows.append(cells)
    
    # Create DataFrame
    if not rows:
        raise Exception("No data rows found in the table.")
    
    df = pd.DataFrame(rows, columns=headers)
    
    print(f"Scraped {len(rows)} rows with {len(headers)} columns")
    return df

def clean_statcast_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Statcast data"""
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Replace empty strings (or whitespace-only strings) with NaN
    df_clean = df_clean.replace(r'^\s*$', pd.NA, regex=True)
    
    # Drop rows that are completely empty (all columns are NaN)
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.dropna(how='all')
    dropped_rows = initial_rows - df_clean.shape[0]
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} empty rows")
    
    # Check for columns with all empty values and drop them
    empty_cols = []
    for col in df_clean.columns:
        # Check if column is all NaN or empty strings
        if df_clean[col].isna().all() or (df_clean[col].astype(str).str.strip() == '').all():
            empty_cols.append(col)
    
    if empty_cols:
        print(f"Dropping empty columns: {', '.join(empty_cols)}")
        df_clean = df_clean.drop(columns=empty_cols)
    
    # Define columns to remove if they exist
    columns_to_remove = [
        "Downward Movement w/ Gravity (in)",
        "Glove/Arm-Side Movement (in)",
        "Vertical Movement w/o Gravity (in)",
        "Movement Toward/Away from Batter (in)"
    ]
    
    # Drop the specified columns
    cols_before = set(df_clean.columns)
    df_clean = df_clean.drop(columns=[col for col in columns_to_remove if col in df_clean.columns])
    cols_after = set(df_clean.columns)
    removed = cols_before - cols_after
    if removed:
        print(f"Removed specific columns: {', '.join(removed)}")
    
    # Additional check for nearly empty columns (>95% missing)
    missing_percentage = df_clean.isna().mean()
    nearly_empty_cols = missing_percentage[missing_percentage > 0.95].index.tolist()
    
    if nearly_empty_cols:
        print(f"Dropping nearly empty columns (>95% missing): {', '.join(nearly_empty_cols)}")
        df_clean = df_clean.drop(columns=nearly_empty_cols)
    
    # Show data types for each column (helpful for debugging)
    print("\nColumn data types after cleaning:")
    for col in df_clean.columns:
        print(f"{col}: {df_clean[col].dtype}")
    
    # Show missing value counts
    missing_counts = df_clean.isna().sum()
    if missing_counts.sum() > 0:
        print("\nMissing values per column:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"{col}: {count} missing values")
    
    return df_clean

def save_to_gcs(df: pd.DataFrame, bucket_name: str, filename: str) -> str:
    """
    Save DataFrame to Google Cloud Storage
    
    Args:
        df: DataFrame to save
        bucket_name: GCS bucket name
        filename: Filename to save as
        
    Returns:
        GCS URI of the saved file
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

def main(local_save: bool = True, gcs_upload: bool = False):
    """
    Main function to scrape, clean, and save Statcast data
    
    Args:
        local_save: Whether to save data locally
        gcs_upload: Whether to upload data to GCS
    """
    # Create base URL for Statcast data (2025 season, hits into play)
    url = ("https://baseballsavant.mlb.com/statcast_search?hfPT=&hfAB=&hfGT=R%7C&hfPR="
           "hit%5C.%5C.into%5C.%5C.play%7C&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC="
           "&hfSea=2025%7C&hfSit=&player_type=batter&hfOuts=&hfOpponent=&pitcher_throws="
           "&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfMo=&hfTeam=&home_road=&hfRO="
           "&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&metric_1=&group_by=name-event"
           "&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=fangraphs_est_woba_numer"
           "&sort_order=desc&chk_event_release_speed=on&chk_event_effective_speed=on"
           "&chk_event_release_spin_rate=on&chk_event_release_pos_z=on&chk_event_release_pos_x=on"
           "&chk_event_release_extension=on&chk_event_plate_x=on&chk_event_plate_z=on"
           "&chk_event_arm_angle=on&chk_event_pitch_name=on&chk_event_api_break_z_with_gravity=on"
           "&chk_event_api_break_x_arm=on&chk_event_api_break_z_induced=on&chk_event_api_break_x_batter_in=on"
           "&chk_event_launch_speed=on&chk_event_hyper_speed=on&chk_event_sweetspot_speed_mph=on"
           "&chk_event_launch_angle=on&chk_event_hit_distance_sc=on&chk_event_swing_length=on"
           "&chk_event_fangraphs_est_woba_numer=on#results")
    
    try:
        # Scrape the data
        print("Starting data scraping...")
        df = scrape_statcast_data(url)
        print(f"Scraped data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Check for empty columns before cleaning
        empty_cols_before = [col for col in df.columns if df[col].astype(str).str.strip().eq('').all()]
        if empty_cols_before:
            print(f"Found empty columns before cleaning: {', '.join(empty_cols_before)}")
        
        # Display first few rows of raw data
        print("\nFirst few rows of raw data:")
        print(df.head(3))
        
        # Clean the data
        print("\nCleaning data...")
        df_clean = clean_statcast_data(df)
        print(f"Cleaned data has {df_clean.shape[0]} rows and {df_clean.shape[1]} columns")
        
        # Double check for any remaining empty columns
        empty_cols_after = [col for col in df_clean.columns if df_clean[col].isna().all()]
        if empty_cols_after:
            print(f"WARNING: Still found empty columns after cleaning: {', '.join(empty_cols_after)}")
            print("Removing these columns...")
            df_clean = df_clean.drop(columns=empty_cols_after)
            print(f"Final cleaned data has {df_clean.shape[0]} rows and {df_clean.shape[1]} columns")
        
        # Display first few rows of cleaned data
        print("\nFirst few rows of cleaned data:")
        print(df_clean.head(3))
        
        # Create filename with current date
        current_date = time.strftime("%Y-%m-%d")
        filename = f"statcast_data_{current_date}.csv"
        
        # Save locally if requested
        if local_save:
            # Create data directories if they don't exist
            os.makedirs("../../data/raw", exist_ok=True)
            
            # Save raw and cleaned data
            df_clean.to_csv(f"../../data/raw/{filename}", index=False)
            print(f"Data saved locally to ../../data/raw/{filename} and ../../data/processed/{filename}")
        
        # Upload to GCS if requested
        if gcs_upload:
            bucket_name = "baseball-ml-data"
            
            # Upload raw data
            raw_gcs_uri = save_to_gcs(df, bucket_name, f"raw/{filename}")
            print(f"Raw data uploaded to {raw_gcs_uri}")
            
            # Also save as "latest" for easy reference
            latest_gcs_uri = save_to_gcs(df_clean, bucket_name, "latest/statcast_data_latest.csv")
            print(f"Latest data uploaded to {latest_gcs_uri}")
        
        return df_clean
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == '__main__':
    # When running locally
    main(local_save=True, gcs_upload=False)