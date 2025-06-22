#!/bin/bash
set -e

IMAGE_TAG=${1:-dynamic-gsplats-dev}
PROJECT_DIR=$(pwd)

echo "Running Docker image: $IMAGE_TAG with GPU support and mounted volume"
docker run --rm -it \
  --gpus all \
  --shm-size=1g \
  -v "$PROJECT_DIR":/dynamic-gsplats \
  -w /dynamic-gsplats \
  "$IMAGE_TAG"
