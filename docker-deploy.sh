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

echo "[Deploy Script] Updating container on $(hostname) to $CONTAINER_IMAGE_URI"

echo "[Deploy Script] Pulling image..."
docker pull "$CONTAINER_IMAGE_URI"

echo "[Deploy Script] Checking for running containers ($BASE_IMAGE_NAME)..."
RUNNING_CONTAINERS=$(docker ps -q --filter ancestor="$BASE_IMAGE_NAME")
if [ -n "$RUNNING_CONTAINERS" ]; then
  echo "[Deploy Script] Stopping existing containers..."
  docker stop $RUNNING_CONTAINERS
fi

echo "[Deploy Script] Checking for stopped containers ($BASE_IMAGE_NAME)..."
STOPPED_CONTAINERS=$(docker ps -aq --filter ancestor="$BASE_IMAGE_NAME")
if [ -n "$STOPPED_CONTAINERS" ]; then
  echo "[Deploy Script] Removing stopped containers..."
  docker rm $STOPPED_CONTAINERS
fi

echo "[Deploy Script] Starting new container..."
docker run -d -p 80:8000 --restart always "$CONTAINER_IMAGE_URI"

echo "[Deploy Script] Update complete on $(hostname)" 