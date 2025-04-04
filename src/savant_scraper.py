import requests
import pandas as pd
import time
import os
from datetime import datetime
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseballSavantScraper:
    """
    A class to scrape xwOBAcon prediction data from Baseball Savant
    """
    
    def __init__(self, output_dir='../data/raw'):
        """
        Initialize the scraper with a directory to save data
        
        Args:
            output_dir (str): Directory where data will be saved
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
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
    
    def get_available_seasons(self):
        """
        Determine all available seasons for bat tracking data
        
        Returns:
            list: List of years with available data
        """
        # Bat tracking data is available from 2015
        current_year = datetime.now().year
        available_years = []
        base_url = "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
        
        for year in range(2015, current_year + 1):
            csv_url = f"{base_url}?type=batter&year={year}&position=&team=&min=25&csv=true"
            try:
                logger.info(f"Checking if bat tracking data exists for {year}...")
                response = requests.get(csv_url, timeout=10)
                
                if response.status_code == 200 and len(response.content) > 500:
                    logger.info(f"Found bat tracking data for {year}")
                    available_years.append(year)
                else:
                    logger.info(f"No bat tracking data found for {year}")
            except Exception as e:
                logger.warning(f"Error checking {year} data: {e}")
        
        return available_years
            
    def fetch_bat_tracking_data(self, year, min_pa=25):
        """
        Fetch bat tracking data from Baseball Savant
        
        Args:
            year (int): The season year to fetch data for
            min_pa (int): Minimum plate appearances filter
            
        Returns:
            pandas.DataFrame: DataFrame containing bat tracking metrics
        """
        logger.info(f"Fetching bat tracking data for {year} with min_pa={min_pa}...")
        base_url = "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
        csv_url = f"{base_url}?type=batter&year={year}&position=&team=&min={min_pa}&csv=true"
        
        try:
            session = self.create_session_with_retries()
            logger.info(f"Requesting URL: {csv_url}")
            response = session.get(csv_url)
            
            if response.status_code == 200:
                # Check if response content is valid
                if response.content and len(response.content) > 100:
                    try:
                        df = pd.read_csv(pd.io.common.StringIO(response.content.decode('utf-8')))
                        
                        # Check if we got a valid DataFrame with data
                        if len(df) > 0 and len(df.columns) > 1:
                            logger.info(f"Successfully fetched bat tracking data: {len(df)} rows, {len(df.columns)} columns")
                            
                            # Clean up data - convert percentage strings to floats, etc.
                            for col in df.columns:
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
                            
                            # Add year column for multi-year datasets
                            df['season'] = year
                            
                            # Save to CSV
                            output_file = os.path.join(self.output_dir, f"bat_tracking_{year}.csv")
                            df.to_csv(output_file, index=False)
                            
                            logger.info(f"Saved bat tracking data to {output_file}")
                            return df
                        else:
                            logger.warning("Received CSV data but it appears to be empty or invalid")
                            return pd.DataFrame()
                    except Exception as e:
                        logger.error(f"Error parsing CSV data: {e}")
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
    
    def fetch_xwobacon_data(self, year, min_pa=25):
        """
        Fetch xwOBAcon and quality of contact metrics for batters
        
        Args:
            year (int): Year to fetch data for
            min_pa (int): Minimum plate appearances filter
            
        Returns:
            pandas.DataFrame: DataFrame containing quality of contact metrics
        """
        logger.info(f"Fetching xwOBAcon data for {year}...")
        
        # Custom leaderboard URL
        base_url = "https://baseballsavant.mlb.com/leaderboard/custom"
        params = {
            "year": year,
            "type": "batter",
            "filter": "",
            "min": str(min_pa),
            "selections": "pa,k_percent,bb_percent,woba,xwoba,xwobacon,sweet_spot_percent,barrel_batted_rate,hard_hit_percent,avg_best_speed,avg_hyper_speed,whiff_percent,swing_percent",
            "chart": "false",
            "x": "pa",
            "y": "pa",
            "r": "no",
            "chartType": "beeswarm",
            "sort": "xwobacon",
            "sortDir": "desc",
            "csv": "true"
        }
        
        try:
            session = self.create_session_with_retries()
            response = session.get(base_url, params=params)
            
            if response.status_code == 200 and len(response.content) > 100:
                df = pd.read_csv(pd.io.common.StringIO(response.content.decode('utf-8')))
                
                # Check if xwOBAcon column exists
                if 'xwobacon' not in df.columns.str.lower():
                    logger.error("xwOBAcon column not found in the data")
                    return pd.DataFrame()
                
                # Add year column for multi-year datasets
                df['season'] = year
                
                logger.info(f"Successfully fetched xwOBAcon data: {len(df)} rows")
                
                # Save the data
                output_file = os.path.join(self.output_dir, f"xwobacon_data_{year}.csv")
                df.to_csv(output_file, index=False)
                logger.info(f"Saved to {output_file}")
                return df
            else:
                logger.warning(f"Failed to fetch data. Status: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching xwOBAcon data: {e}")
            return pd.DataFrame()
    
    def fetch_multi_season_data(self, start_year=None, end_year=None, min_pa=25):
        """
        Fetch data for multiple seasons
        
        Args:
            start_year (int, optional): First year to fetch data for
            end_year (int, optional): Last year to fetch data for
            min_pa (int): Minimum plate appearances
            
        Returns:
            tuple: (all_bat_data, all_xwobacon_data)
        """
        if start_year is None or end_year is None:
            available_years = self.get_available_seasons()
            if not available_years:
                logger.error("No available seasons detected")
                return None, None
                
            if start_year is None:
                start_year = min(available_years)
            if end_year is None:
                end_year = max(available_years)
        
        all_bat_data = []
        all_xwobacon_data = []
        
        for year in range(start_year, end_year + 1):
            # Fetch bat tracking data
            bat_data = self.fetch_bat_tracking_data(year, min_pa)
            if not bat_data.empty:
                all_bat_data.append(bat_data)
            
            # Fetch xwOBAcon data
            xwobacon_data = self.fetch_xwobacon_data(year, min_pa)
            if not xwobacon_data.empty:
                all_xwobacon_data.append(xwobacon_data)
            
            # Add a delay to avoid overloading the server
            time.sleep(5)
        
        # Combine all data
        combined_bat_data = pd.concat(all_bat_data, ignore_index=True) if all_bat_data else pd.DataFrame()
        combined_xwobacon_data = pd.concat(all_xwobacon_data, ignore_index=True) if all_xwobacon_data else pd.DataFrame()
        
        # Save combined data
        if not combined_bat_data.empty:
            output_file = os.path.join(self.output_dir, f"bat_tracking_combined_{start_year}_{end_year}.csv")
            combined_bat_data.to_csv(output_file, index=False)
            logger.info(f"Saved combined bat tracking data to {output_file}")
        
        if not combined_xwobacon_data.empty:
            output_file = os.path.join(self.output_dir, f"xwobacon_combined_{start_year}_{end_year}.csv")
            combined_xwobacon_data.to_csv(output_file, index=False)
            logger.info(f"Saved combined xwOBAcon data to {output_file}")
        
        return combined_bat_data, combined_xwobacon_data
    
    def merge_with_quality_of_contact(self, bat_data, qoc_data):
        """
        Merge bat tracking data with quality of contact data
        
        Args:
            bat_data (pandas.DataFrame): Bat tracking metrics
            qoc_data (pandas.DataFrame): Quality of contact metrics
            
        Returns:
            pandas.DataFrame: Merged dataset with both bat tracking and quality of contact metrics
        """
        if bat_data is None or bat_data.empty:
            logger.error("Bat tracking data is empty, cannot merge")
            return None
            
        if qoc_data is None or qoc_data.empty:
            logger.error("Quality of contact data is empty, cannot merge")
            return None
        
        logger.info(f"Merging datasets: bat_data has {bat_data.shape[0]} rows, qoc_data has {qoc_data.shape[0]} rows")
        
        # Check if player_id is available in both
        if 'player_id' in bat_data.columns and 'player_id' in qoc_data.columns:
            merge_keys = ['player_id', 'season']  # Add season for multi-year data
        else:
            # Fall back to player name and season
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
            
            # Use both merge_key and season for multi-year data
            merge_keys = ['merge_key', 'season']
        
        # Merge the datasets
        merged_df = pd.merge(bat_data, qoc_data, on=merge_keys, how='inner', suffixes=('_bat', '_qoc'))
        
        # Remove temporary merge key if created
        if 'merge_key' in merge_keys:
            merged_df = merged_df.drop(columns=['merge_key'])
        
        logger.info(f"Merged dataset contains {len(merged_df)} players and {len(merged_df.columns)} columns")
        
        # Save merged dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = os.path.join(self.output_dir, f"merged_xwobacon_data_{timestamp}.csv")
        merged_df.to_csv(output_file, index=False)
        logger.info(f"Saved merged dataset to {output_file}")
        
        return merged_df
    
    def update_with_latest_data(self, min_pa=25):
        """
        Update dataset with the latest weekly data for the current season
        
        Args:
            min_pa (int): Minimum plate appearances
            
        Returns:
            pandas.DataFrame: Updated merged dataset
        """
        current_year = datetime.now().year
        
        # Fetch latest data for current year
        latest_bat_data = self.fetch_bat_tracking_data(current_year, min_pa)
        latest_xwobacon_data = self.fetch_xwobacon_data(current_year, min_pa)
        
        # Merge latest data
        if not latest_bat_data.empty and not latest_xwobacon_data.empty:
            latest_merged = self.merge_with_quality_of_contact(latest_bat_data, latest_xwobacon_data)
            
            # Load historical data
            import glob
            pattern = os.path.join(self.output_dir, "merged_xwobacon_data_*.csv")
            existing_files = sorted(glob.glob(pattern))
            
            if existing_files and existing_files[-1] != latest_merged:
                # Combine with historical data
                try:
                    historical_data = pd.read_csv(existing_files[-1])
                    
                    # Remove current season data from historical dataset
                    if 'season' in historical_data.columns:
                        historical_data = historical_data[historical_data['season'] != current_year]
                    
                    # Combine with latest data
                    combined_data = pd.concat([historical_data, latest_merged], ignore_index=True)
                    
                    # Save combined dataset
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    output_file = os.path.join(self.output_dir, f"merged_xwobacon_complete_{timestamp}.csv")
                    combined_data.to_csv(output_file, index=False)
                    logger.info(f"Saved updated complete dataset to {output_file}")
                    
                    return combined_data
                except Exception as e:
                    logger.error(f"Error combining historical and latest data: {e}")
                    return latest_merged
            else:
                return latest_merged
        else:
            logger.warning("Could not fetch latest data for update")
            return None


if __name__ == "__main__":
    scraper = BaseballSavantScraper()
    bat_data, xwobacon_data = scraper.fetch_multi_season_data(start_year=2015, min_pa=25)
    if bat_data is not None and not bat_data.empty and xwobacon_data is not None and not xwobacon_data.empty:
        merged_data = scraper.merge_with_quality_of_contact(bat_data, xwobacon_data)
        if merged_data is not None:
            print(f"Final merged dataset has {merged_data.shape[0]} players and {merged_data.shape[1]} features")