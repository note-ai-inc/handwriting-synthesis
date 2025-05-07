#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting Cloud Run deployment process..."

# Get the root directory
cd "$(dirname "$0")"

# Build and push images using docker-compose
echo "🏗️ Building and pushing images..."
docker-compose build
docker-compose push

# Submit to Cloud Build (which will handle building, pushing and deploying)
echo "🏗️ Submitting to Cloud Build..."
gcloud builds submit --config cloudbuild.yaml .

echo "✅ Deployment process completed!"

# Show the service URLs
echo "🌐 Service URLs:"
echo "Handwriting Synthesis:"
gcloud run services describe handwriting-synthesis --platform managed --region asia-northeast3 --format 'value(status.url)'
echo "Handwriting Quality:"
gcloud run services describe handwriting-quality --platform managed --region asia-northeast3 --format 'value(status.url)' 