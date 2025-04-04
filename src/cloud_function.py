import functions_framework
from google.cloud import storage
import logging
import sys
import os
import json
from datetime import datetime

# Add local directory to Python path to import the scraper
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the scraper
from savant_scraper import BaseballSavantScraper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@functions_framework.http
def weekly_baseball_data_update(request):
    """
    Cloud Function to update baseball data on a weekly basis
    
    Args:
        request (flask.Request): HTTP request object
        
    Returns:
        dict: HTTP response with update status
    """
    try:
        logger.info("Starting weekly baseball data update")
        
        # Check if this is a manual execution or scheduled
        if request.method == 'POST':
            request_json = request.get_json(silent=True)
            if request_json and 'min_pa' in request_json:
                min_pa = request_json['min_pa']
            else:
                min_pa = 25
        else:
            min_pa = 25
        
        # GCS bucket configuration (if needed)
        bucket_name = os.environ.get('BUCKET_NAME', 'baseball-data-bucket')
        raw_data_dir = os.environ.get('RAW_DATA_DIR', 'data/raw')
        
        # Initialize scraper
        scraper = BaseballSavantScraper(output_dir=raw_data_dir)
        
        # Option 1: For regular weekly updates
        updated_data = scraper.update_with_latest_data(min_pa=min_pa)
        
        if updated_data is not None:
            # Upload to GCS if needed
            if os.environ.get('USE_GCS', 'false').lower() == 'true':
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                
                # List local files to upload
                import glob
                files_to_upload = glob.glob(os.path.join(raw_data_dir, "merged_xwobacon*.csv"))
                
                # Upload each file
                for file_path in files_to_upload:
                    blob_name = os.path.join('data', os.path.basename(file_path))
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(file_path)
                    logger.info(f"Uploaded {file_path} to {blob_name}")
            
            # Log success
            result = {
                'statusCode': 200,
                'body': f'Successfully updated data with {len(updated_data)} players',
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Weekly update completed successfully with {len(updated_data)} players")
            return result
        else:
            result = {
                'statusCode': 400,
                'body': 'No data was updated',
                'timestamp': datetime.now().isoformat()
            }
            logger.warning("Weekly update completed but no data was updated")
            return result
    
    except Exception as e:
        logger.error(f"Error in weekly baseball data update: {e}")
        return {
            'statusCode': 500,
            'body': f'Error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }

# This is for local testing
if __name__ == "__main__":
    # Create a mock request
    from collections import namedtuple
    MockRequest = namedtuple('MockRequest', ['method'])
    mock_request = MockRequest(method='GET')
    
    # Execute function
    result = weekly_baseball_data_update(mock_request)
    print(result)