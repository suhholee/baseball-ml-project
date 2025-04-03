import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import backoff

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseballSavantScraper:
    """
    A class to scrape statcast data from Baseball Savant (baseballsavant.mlb.com)
    """
    
    def __init__(self, output_dir='data'):
        """
        Initialize the scraper with a directory to save data
        
        Args:
            output_dir (str): Directory where data will be saved
        """
        self.base_url = "https://baseballsavant.mlb.com/statcast_search/csv"
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def get_latest_season_year(self):
        """
        Determine the latest MLB season year with available data
        
        Returns:
            int: Latest season year with data
        """
        current_date = datetime.now()
        current_year = current_date.year
        
        # If we're in the MLB season (April to October)
        if 4 <= current_date.month <= 10:
            return current_year
        
        # During off-season, check if current year data exists
        # If not, use previous year
        try:
            logger.info(f"Checking if data exists for {current_year} season...")
            params = {
                'all': 'true',
                'hfSea': f'{current_year}|',
                'player_type': 'batter',
                'game_date_gt': f'{current_year}-03-18',
                'game_date_lt': f'{current_year}-04-30',
                'min_pitches': '0',
                'min_results': '0',
                'group_by': 'name',
                'sort_col': 'pitches',
                'sort_order': 'desc',
                'min_pas': '0',
                'type': 'details'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200 and len(response.content) > 500:
                logger.info(f"Found data for {current_year} season")
                return current_year
            else:
                logger.info(f"No data found for {current_year}, using {current_year-1} season")
                return current_year - 1
                
        except Exception as e:
            logger.warning(f"Error checking current year data: {e}")
            # If an error occurs, safely assume previous year
            return current_year - 1

    def create_session_with_retries(self, retries=3, backoff_factor=0.5, timeout=60):
        """
        Create a requests session with retry logic
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.timeout = timeout
        return session

    def fetch_statcast_data(self, start_date, end_date, player_type='batter', min_results=25):
        """
        Fetch Statcast data for a specific date range with better error handling and retries
        """
        season_year = datetime.strptime(start_date, "%Y-%m-%d").year
        
        params = {
            'all': 'true',
            'hfPT': '',
            'hfAB': '',
            'hfGT': '',
            'hfPR': '',
            'hfZ': '',
            'stadium': '',
            'hfBBL': '',
            'hfNewZones': '',
            'hfPull': '',
            'hfC': '',
            'hfSea': f'{season_year}|',
            'hfSit': '',
            'player_type': player_type,
            'hfOuts': '',
            'opponent': '',
            'pitcher_throws': '',
            'batter_stands': '',
            'hfSA': '',
            'game_date_gt': start_date,
            'game_date_lt': end_date,
            'hfInfield': '',
            'team': '',
            'position': '',
            'hfOutfield': '',
            'hfRO': '',
            'home_road': '',
            'hfFlag': '',
            'hfBBT': '',
            'metric_1': '',
            'hfInn': '',
            'min_pitches': str(min_results),
            'min_results': str(min_results),
            'group_by': 'name',
            'sort_col': 'pitches',
            'player_event_sort': 'api_p_release_speed',
            'sort_order': 'desc',
            'min_pas': str(min_results),
            'type': 'details'
        }
        
        try:
            logger.info(f"Fetching data from {start_date} to {end_date}...")
            session = self.create_session_with_retries(timeout=60)  # Increased timeout
            response = session.get(self.base_url, params=params)
            
            if response.status_code == 200:
                # Check if response content is valid
                if response.content and len(response.content) > 100:
                    df = pd.read_csv(pd.io.common.StringIO(response.content.decode('utf-8')))
                    logger.info(f"Successfully fetched {len(df)} rows of data")
                    return df
                else:
                    logger.warning("Response was successful but contained insufficient data")
                    return pd.DataFrame()
            else:
                logger.warning(f"Failed to fetch data: Status code {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return pd.DataFrame()
    
    def fetch_most_recent_season_data(self, player_type='batter', chunk_size=7, min_results=25):
        """
        Fetch data for the most recent season with available data
        
        Args:
            player_type (str): Either 'batter' or 'pitcher'
            chunk_size (int): Number of days per request
            min_results (int): Minimum number of results to include a player
            
        Returns:
            pandas.DataFrame: DataFrame containing the season's Statcast data
        """
        # Get the latest season year with data
        year = self.get_latest_season_year()
        logger.info(f"Fetching most recent season data (year: {year})")
        
        # Define season date range
        # Use actual dates for completed seasons or appropriate date range for in-progress seasons
        current_date = datetime.now()
        
        if year < current_date.year:
            # Past season - use full date range
            season_start = f"{year}-03-18"
            season_end = f"{year}-12-01" 
        else:
            # Current season - use data up until yesterday
            season_start = f"{year}-03-18"
            yesterday = current_date - timedelta(days=1)
            season_end = yesterday.strftime("%Y-%m-%d")
        
        logger.info(f"Using date range: {season_start} to {season_end}")
        
        start_date = datetime.strptime(season_start, "%Y-%m-%d")
        end_date = datetime.strptime(season_end, "%Y-%m-%d")
        
        all_data = []
        current_date = start_date
        
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=chunk_size), end_date)
            
            # Format dates for API request
            from_date = current_date.strftime("%Y-%m-%d")
            to_date = chunk_end.strftime("%Y-%m-%d")
            
            # Fetch data for this date range
            df = self.fetch_statcast_data(from_date, to_date, player_type, min_results)
            
            if not df.empty:
                all_data.append(df)
                
            # Add a delay to avoid overloading the server
            time.sleep(3)
            
            # Move to next chunk
            current_date = chunk_end + timedelta(days=1)
        
        # Combine all chunks into a single DataFrame
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save to CSV
            output_file = os.path.join(self.output_dir, f"statcast_{player_type}_{year}.csv")
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(combined_df)} records to {output_file}")
            
            return combined_df
        else:
            logger.warning("No data was collected for the specified parameters")
            return pd.DataFrame()
    
    def fetch_quality_of_contact_data(self, year=None):
        """
        Fetch quality of contact metrics for batters in a specific season
        
        Args:
            year (int, optional): The season year to fetch data for. 
                                  If None, uses the most recent season.
            
        Returns:
            pandas.DataFrame: DataFrame containing quality of contact metrics
        """
        # If year is not specified, get the latest season
        if year is None:
            year = self.get_latest_season_year()
            
        # This endpoint provides leaderboard data with quality of contact metrics
        url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year={year}&position=&team=&min=q&csv=true"
        
        try:
            logger.info(f"Fetching quality of contact data for {year}...")
            response = requests.get(url, timeout=30)  # Added timeout
            
            if response.status_code == 200:
                # Check if response content is valid
                if response.content and len(response.content) > 100:
                    df = pd.read_csv(pd.io.common.StringIO(response.content.decode('utf-8')))
                    
                    # Save to CSV
                    output_file = os.path.join(self.output_dir, f"quality_of_contact_{year}.csv")
                    df.to_csv(output_file, index=False)
                    
                    logger.info(f"Successfully fetched quality of contact data for {len(df)} batters")
                    return df
                else:
                    logger.warning("Response was successful but contained insufficient data")
                    return pd.DataFrame()
            else:
                logger.warning(f"Failed to fetch data: Status code {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return pd.DataFrame()
            
    def fetch_bat_tracking_data(self, year=None, min_pa=50):
        """
        Fetch bat tracking data from Baseball Savant using CSV download
        
        Args:
            year (int, optional): The season year to fetch data for.
                                  If None, uses the most recent season.
            min_pa (int): Minimum plate appearances filter
            
        Returns:
            pandas.DataFrame: DataFrame containing bat tracking metrics
        """
        # If year is not specified, get the latest season
        if year is None:
            year = self.get_latest_season_year()
            
        logger.info(f"Fetching bat tracking data for {year} with min_pa={min_pa}...")
        
        # Construct CSV download URL
        base_url = "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
        csv_url = f"{base_url}?type=batter&year={year}&position=&team=&min={min_pa}&csv=true"
        
        try:
            logger.info(f"Requesting URL: {csv_url}")
            response = requests.get(csv_url, timeout=30)
            
            if response.status_code == 200:
                # Check if response content is valid
                if response.content and len(response.content) > 100:
                    try:
                        # Try to parse as CSV
                        df = pd.read_csv(pd.io.common.StringIO(response.content.decode('utf-8')))
                        
                        # Check if we got a valid DataFrame with data
                        if len(df) > 0 and len(df.columns) > 1:
                            logger.info(f"Successfully fetched bat tracking data: {len(df)} rows, {len(df.columns)} columns")
                            
                            # Clean up data - convert percentage strings to floats, etc.
                            for col in df.columns:
                                # Skip the first column (usually player name)
                                if df.columns.get_loc(col) == 0:
                                    continue
                                
                                # Convert percentage strings to floats
                                if df[col].dtype == 'object' and df[col].str.contains('%').any():
                                    df[col] = df[col].str.replace('%', '').astype(float) / 100
                                
                                # Try to convert other numeric columns to float
                                if df[col].dtype == 'object':
                                    try:
                                        df[col] = df[col].str.replace(',', '').astype(float)
                                    except:
                                        pass
                            
                            # Save to CSV
                            output_file = os.path.join(self.output_dir, f"bat_tracking_{year}.csv")
                            df.to_csv(output_file, index=False)
                            
                            logger.info(f"Saved bat tracking data to {output_file}")
                            return df
                        else:
                            logger.warning("Received CSV data but it appears to be empty or invalid")
                            # Try to print the first part of the response for debugging
                            logger.warning(f"Response preview: {response.content[:500]}")
                            return pd.DataFrame()
                    except Exception as e:
                        logger.error(f"Error parsing CSV data: {e}")
                        # Save the raw response for inspection
                        error_file = os.path.join(self.output_dir, f"bat_tracking_error_{year}.txt")
                        with open(error_file, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"Saved error response to {error_file}")
                        return pd.DataFrame()
                else:
                    logger.warning("Response was successful but contained insufficient data")
                    return pd.DataFrame()
            else:
                logger.warning(f"Failed to fetch data: Status code {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"An error occurred during request: {e}")
            return pd.DataFrame()
    
    def fetch_most_recent_bat_tracking_data(self, min_pa=50):
        """
        Fetch bat tracking data for the most recent MLB season
        
        Args:
            min_pa (int): Minimum plate appearances filter
            
        Returns:
            pandas.DataFrame: Bat tracking data for most recent season
        """
        # Get the latest season year with data
        year = self.get_latest_season_year()
        
        # Fetch the data
        return self.fetch_bat_tracking_data(year=year, min_pa=min_pa)
    
    def merge_with_quality_of_contact(self, bat_data, qoc_data):
        """
        Merge bat tracking data with quality of contact data
        
        Args:
            bat_data (pandas.DataFrame): Bat tracking metrics
            qoc_data (pandas.DataFrame): Quality of contact metrics
            
        Returns:
            pandas.DataFrame: Merged dataset with both bat tracking and quality of contact metrics
        """
        # Check if either dataset is empty
        if bat_data is None or bat_data.empty:
            logger.error("Bat tracking data is empty, cannot merge")
            return None
            
        if qoc_data is None or qoc_data.empty:
            logger.error("Quality of contact data is empty, cannot merge")
            return None
        
        logger.info(f"Merging datasets: bat_data has {bat_data.shape[0]} rows, qoc_data has {qoc_data.shape[0]} rows")
        
        # Check if player_id is available in both
        if 'player_id' in bat_data.columns and 'player_id' in qoc_data.columns:
            merge_key = 'player_id'
        else:
            # Fall back to player name
            # Try to identify player name columns
            bat_name_columns = [col for col in bat_data.columns if any(name in col.lower() for name in ['player', 'name', 'hitter', 'batter'])]
            qoc_name_columns = [col for col in qoc_data.columns if any(name in col.lower() for name in ['player', 'name', 'hitter', 'batter'])]
            
            if bat_name_columns:
                bat_key = bat_name_columns[0]
            else:
                # Use first column as a fallback
                bat_key = bat_data.columns[0]
            
            if qoc_name_columns:
                qoc_key = qoc_name_columns[0]
            else:
                # Use first column as a fallback
                qoc_key = qoc_data.columns[0]
            
            logger.info(f"Using merge keys: {bat_key} from bat_data and {qoc_key} from qoc_data")
            
            # Create temporary columns with standardized format for merging
            bat_data['merge_key'] = bat_data[bat_key].astype(str).str.upper().str.strip()
            qoc_data['merge_key'] = qoc_data[qoc_key].astype(str).str.upper().str.strip()
            
            merge_key = 'merge_key'
        
        # Merge the datasets
        merged_df = pd.merge(bat_data, qoc_data, on=merge_key, how='inner', suffixes=('_bat', '_qoc'))
        
        # Remove temporary merge key if created
        if merge_key == 'merge_key':
            merged_df = merged_df.drop(columns=['merge_key'])
        
        logger.info(f"Merged dataset contains {len(merged_df)} players and {len(merged_df.columns)} columns")
        
        # Save merged dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = os.path.join(self.output_dir, f"merged_bat_qoc_data_{timestamp}.csv")
        merged_df.to_csv(output_file, index=False)
        logger.info(f"Saved merged dataset to {output_file}")
        
        return merged_df


if __name__ == "__main__":
    # Initialize the scraper
    scraper = BaseballSavantScraper(output_dir='../data/raw')
    
    # Get current season data for batters
    season_data = scraper.fetch_most_recent_season_data(player_type='batter', min_results=25)
    
    # Get quality of contact data
    qoc_data = scraper.fetch_quality_of_contact_data()
    
    # Get bat tracking data
    bat_data = scraper.fetch_most_recent_bat_tracking_data(min_pa=25)
    
    # Check if we got bat tracking data
    if bat_data is not None and not bat_data.empty:
        print(f"Got bat tracking data with {len(bat_data)} rows and {len(bat_data.columns)} columns")
        print("Columns:", bat_data.columns.tolist())
        
        # Merge with quality of contact data
        merged_data = scraper.merge_with_quality_of_contact(bat_data, qoc_data)
        if merged_data is not None:
            print(f"Final merged dataset has {merged_data.shape[1]} features")