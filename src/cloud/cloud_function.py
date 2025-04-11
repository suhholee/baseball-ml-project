import functions_framework
import logging
import requests
from bs4 import BeautifulSoup
import csv
import tempfile
import time
from datetime import datetime
from google.cloud import storage
import traceback
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GCS bucket configuration
GCS_BUCKET_NAME = "baseball-ml-data"  # This is set in the environment variable

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
    
    # Check if we have any data
    if not rows:
        raise Exception("No data rows found in the table.")
    
    logger.info(f"Scraped {len(rows)} rows with {len(headers)} columns")
    
    # Return headers and rows directly instead of using pandas
    return headers, rows

def clean_data(headers, rows):
    """
    Clean the data without using pandas
    """
    # Filter out empty rows
    filtered_rows = []
    for row in rows:
        if any(cell.strip() for cell in row):  # Check if any cell has content
            filtered_rows.append(row)
    
    logger.info(f"Cleaned data from {len(rows)} to {len(filtered_rows)} rows")
    
    # Add Season column if not already present
    if 'Season' not in headers:
        headers.append('Season')
        for row in filtered_rows:
            row.append(str(datetime.now().year))  # Add current year
    
    return headers, filtered_rows

def save_to_gcs(headers, rows, bucket_name, prefix, filename):
    """
    Save data to Google Cloud Storage as CSV
    """
    # Initialize storage client
    storage_client = storage.Client()
    
    # Get bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Create blob
    blob_name = f"{prefix}/{filename}"
    blob = bucket.blob(blob_name)
    
    # Create CSV in memory
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    
    # Write headers and rows
    csv_writer.writerow(headers)
    csv_writer.writerows(rows)
    
    # Upload to GCS
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
    
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    logger.info(f"Saved data to {gcs_uri}")
    
    return gcs_uri

def generate_statcast_url(season, event_type="hit..into..play"):
    """
    Generate URL for Statcast data with specified parameters
    """
    url = (f"https://baseballsavant.mlb.com/statcast_search?hfPT=&hfAB=&hfGT=R%7C&hfPR="
           f"{event_type}%7C&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC="
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
    
    return url

@functions_framework.http
def scrape_statcast_http(request):
    """
    HTTP Cloud Function to scrape Statcast data
    """
    try:
        # Get request parameters
        request_json = request.get_json(silent=True)
        request_args = request.args
        
        # Parse parameters
        if request_json and 'season' in request_json:
            season = int(request_json['season'])
        elif request_args and 'season' in request_args:
            season = int(request_args['season'])
        else:
            season = datetime.now().year
        
        if request_json and 'include_previous_season' in request_json:
            include_previous = request_json['include_previous_season']
        elif request_args and 'include_previous_season' in request_args:
            include_previous = request_args['include_previous_season'].lower() == 'true'
        else:
            include_previous = False
        
        # Log request info
        logger.info(f"Received request to scrape Statcast data for {season} season")
        logger.info(f"Include previous season: {include_previous}")
        
        # Determine seasons to scrape
        seasons_to_scrape = [season]
        if include_previous:
            seasons_to_scrape.append(season - 1)
        
        # Scrape and save data for each season
        all_results = []
        
        for season_year in seasons_to_scrape:
            try:
                # Generate URL
                url = generate_statcast_url(season_year)
                
                # Scrape data
                headers, rows = scrape_statcast_data(url)
                
                # Add Season column with explicit value
                if 'Season' not in headers:
                    headers.append('Season')
                    for row in rows:
                        row.append(str(season_year))
                
                # Clean data
                headers, cleaned_rows = clean_data(headers, rows)
                
                # Create filename
                current_date = datetime.now().strftime("%Y-%m-%d")
                filename = f"statcast_data_{season_year}_{current_date}.csv"
                
                # Save raw data to GCS
                raw_gcs_uri = save_to_gcs(headers, cleaned_rows, GCS_BUCKET_NAME, "raw", filename)
                
                # Also save as latest
                if season_year == season:
                    latest_filename = "statcast_data_latest.csv"
                    latest_gcs_uri = save_to_gcs(headers, cleaned_rows, GCS_BUCKET_NAME, "latest", latest_filename)
                
                all_results.append({
                    "season": season_year,
                    "rows_scraped": len(cleaned_rows),
                    "uri": raw_gcs_uri
                })
                
                logger.info(f"Successfully scraped and saved data for {season_year} season")
                
            except Exception as e:
                logger.error(f"Error scraping data for {season_year} season: {str(e)}")
                all_results.append({
                    "season": season_year,
                    "error": str(e)
                })
        
        # Return results
        return {
            "status": "success",
            "message": f"Scraped data for {len(all_results)} seasons",
            "results": all_results
        }
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }, 500

@functions_framework.cloud_event
def scrape_statcast_scheduled(cloud_event):
    """
    Cloud Function to be triggered by Cloud Scheduler
    """
    try:
        # Log event information
        logger.info(f"Received scheduled event: {cloud_event.id}")
        
        # Get current season
        seasons_to_scrape = [2023, 2024, 2025]
        
        # Scrape current season data
        url = generate_statcast_url(seasons_to_scrape)
        headers, rows = scrape_statcast_data(url)
        
        # Add Season column with explicit value
        if 'Season' not in headers:
            headers.append('Season')
            for row in rows:
                row.append(str(seasons_to_scrape))
        
        # Clean data
        headers, cleaned_rows = clean_data(headers, rows)
        
        # Create filenames
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"statcast_data_{seasons_to_scrape}_{current_date}.csv"
        latest_filename = "statcast_data_latest.csv"
        
        # Save data to GCS
        raw_gcs_uri = save_to_gcs(headers, cleaned_rows, GCS_BUCKET_NAME, "raw", filename)
        latest_gcs_uri = save_to_gcs(headers, cleaned_rows, GCS_BUCKET_NAME, "latest", latest_filename)
        
        return {
            "success": True,
            "message": f"Successfully scraped and saved data for {seasons_to_scrape} season",
            "raw_uri": raw_gcs_uri,
            "latest_uri": latest_gcs_uri,
            "rows_scraped": len(cleaned_rows)
        }
    
    except Exception as e:
        logger.error(f"Error processing scheduled event: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "error": str(e)
        }