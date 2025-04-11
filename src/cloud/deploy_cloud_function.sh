#!/bin/bash

# Updated Cloud Function deployment script
# Usage: ./deploy_cloud_function.sh [project_id] [region]

# Default values
PROJECT_ID=${1:-"baseball-ml"}
REGION=${2:-"us-central1"}
FUNCTION_NAME="scrape-statcast-data"
RUNTIME="python39"

# Create temporary directory for deployment
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Create requirements.txt
cat > $TEMP_DIR/requirements.txt << EOF
beautifulsoup4==4.12.2
requests==2.31.0
google-cloud-storage==2.10.0
functions-framework==3.4.0
EOF

# Copy the cloud function code
cp cloud_function.py $TEMP_DIR/main.py

# Change to temp directory 
cd $TEMP_DIR

# Deploy HTTP function
echo "Deploying HTTP function..."
gcloud functions deploy ${FUNCTION_NAME}-http \
  --gen2 \
  --runtime=$RUNTIME \
  --region=$REGION \
  --source=. \
  --entry-point=scrape_statcast_http \
  --trigger-http \
  --memory=512MB \
  --timeout=540s \
  --project=$PROJECT_ID \
  --set-env-vars="GCS_BUCKET_NAME=baseball-ml-data"

# Create Pub/Sub topic if it doesn't exist
TOPIC_NAME="weekly-statcast-scrape"
if ! gcloud pubsub topics describe $TOPIC_NAME --project=$PROJECT_ID &>/dev/null; then
  echo "Creating Pub/Sub topic: $TOPIC_NAME"
  gcloud pubsub topics create $TOPIC_NAME --project=$PROJECT_ID
fi

# Deploy scheduled function
echo "Deploying scheduled function..."
gcloud functions deploy ${FUNCTION_NAME}-scheduled \
  --gen2 \
  --runtime=$RUNTIME \
  --region=$REGION \
  --source=. \
  --entry-point=scrape_statcast_scheduled \
  --trigger-topic=$TOPIC_NAME \
  --memory=512MB \
  --timeout=540s \
  --project=$PROJECT_ID \
  --set-env-vars="GCS_BUCKET_NAME=baseball-ml-data"

# Create Cloud Scheduler job for weekly execution
JOB_NAME="weekly-statcast-scrape"
SCHEDULE="0 3 * * 1"  # 3:00 AM every Monday
echo "Creating Cloud Scheduler job..."
gcloud scheduler jobs create pubsub $JOB_NAME \
  --schedule="$SCHEDULE" \
  --topic="projects/$PROJECT_ID/topics/$TOPIC_NAME" \
  --message-body='{"scrape":"weekly"}' \
  --location=$REGION \
  --project=$PROJECT_ID

# Clean up temporary directory
cd - > /dev/null
rm -rf $TEMP_DIR
echo "Cleaned up temporary directory"

echo "Deployment completed successfully!"
echo "HTTP function URL:"
gcloud functions describe ${FUNCTION_NAME}-http --gen2 --region=$REGION --project=$PROJECT_ID --format="value(serviceConfig.uri)"