#!/bin/bash
set -e

# Expect the full image URI as the first argument
if [ -z "$1" ]; then
  echo "Error: Container image URI argument is missing."
  exit 1
fi

CONTAINER_IMAGE_URI="$1"
# Extract base image name (everything before the first ':')
BASE_IMAGE_NAME=$(echo "$CONTAINER_IMAGE_URI" | cut -d':' -f1)

echo "[Deploy Script] Updating containers on $(hostname)"

echo "[Deploy Script] Pulling images..."
docker-compose pull

echo "[Deploy Script] Stopping existing containers..."
docker-compose down

echo "[Deploy Script] Starting new containers..."
docker-compose up -d

echo "[Deploy Script] Update complete on $(hostname)" 