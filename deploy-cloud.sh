#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting Cloud Run deployment process..."

# Get the root directory
cd "$(dirname "$0")"

# Submit to Cloud Build (which will handle building, pushing and deploying)
echo "🏗️ Submitting to Cloud Build..."
gcloud builds submit --config cloudbuild.yaml .

echo "✅ Deployment process completed!"

# Show the service URL
echo "🌐 Service URL:"
gcloud run services describe handwriting-synthesis --platform managed --region asia-northeast3 --format 'value(status.url)' 