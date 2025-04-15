#!/bin/bash

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <project_id> [region]"
    echo "Example: $0 my-project-id us-west1"
    exit 1
fi

# Get the project ID from the first argument
PROJECT_ID=$1

# Set the region, default to us-west1 if not provided
REGION=${2:-"us-central1"}

# Set the service name
SERVICE_NAME="baseball-data-scraper"

# Set the bucket name
BUCKET_NAME="baseball-ml-data"

echo "Deploying Cloud Run service '$SERVICE_NAME' to project '$PROJECT_ID' in region '$REGION'..."

# Create proper Dockerfile first
cat > Dockerfile << 'EOL'
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Environment variables
ENV PORT=8080

# Command to run the application
CMD python -m functions_framework --target=scrape_baseball_data --port=${PORT}
EOL

# Build the container image
echo "Building container image..."
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"
gcloud builds submit --tag $IMAGE_NAME --project $PROJECT_ID

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 1 \
  --set-env-vars BUCKET_NAME=$BUCKET_NAME \
  --allow-unauthenticated \
  --project $PROJECT_ID

# Wait for the deployment to complete
echo "Waiting for deployment to complete..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)' --project $PROJECT_ID)

if [ -n "$SERVICE_URL" ]; then
    echo "Cloud Run service successfully deployed!"
    echo "Service URL: $SERVICE_URL"
    
    # Set up the Cloud Scheduler job for weekly runs
    JOB_NAME="weekly_baseball_data_scraper"
    SCHEDULE="0 10 * * 3"  # 10:00 UTC on Wednesdays (3:00 AM PST)
    
    # Check if the job already exists
    gcloud scheduler jobs describe $JOB_NAME \
        --project=$PROJECT_ID \
        --location=$REGION > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        # Update existing job
        echo "Updating existing Cloud Scheduler job '$JOB_NAME'..."
        gcloud scheduler jobs update http $JOB_NAME \
            --project=$PROJECT_ID \
            --location=$REGION \
            --schedule="$SCHEDULE" \
            --uri="$SERVICE_URL" \
            --http-method="GET" \
            --time-zone="America/Los_Angeles" \
            --attempt-deadline=1800s \
            --description="Weekly Baseball Data Scraper (runs every Wednesday at 3:00 AM PST)"
    else
        # Create a new job
        echo "Creating Cloud Scheduler job '$JOB_NAME'..."
        gcloud scheduler jobs create http $JOB_NAME \
            --project=$PROJECT_ID \
            --location=$REGION \
            --schedule="$SCHEDULE" \
            --uri="$SERVICE_URL" \
            --http-method="GET" \
            --time-zone="America/Los_Angeles" \
            --attempt-deadline=1800s \
            --description="Weekly Baseball Data Scraper (runs every Wednesday at 3:00 AM PST)"
    fi
    
    echo "Cloud Scheduler job set up to run every Wednesday at 3:00 AM PST!"
    echo "Deployment complete!"
    
    # Optional: Test the deployment
    echo "Would you like to test the deployment now? (y/n)"
    read -r test_deployment
    if [[ "$test_deployment" =~ ^[Yy]$ ]]; then
        echo "Testing the deployment..."
        echo "This may take several minutes depending on the data size."
        echo "Sending request to $SERVICE_URL"
        curl -X GET "$SERVICE_URL"
        echo ""
        echo "Check the logs for detailed progress:"
        echo "gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\" --limit 50 --project $PROJECT_ID"
    fi
else
    echo "Error: Deployment failed or timed out. Please check the logs for more information."
    exit 1
fi