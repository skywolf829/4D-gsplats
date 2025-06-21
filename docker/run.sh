#!/bin/bash
set -e

IMAGE_TAG=${1:-dynamic-gsplats-dev}
PROJECT_DIR=$(pwd)

echo "Running Docker image: $IMAGE_TAG with GPU support and mounted volume"
docker run --rm -it \
  --gpus all \
  -v "$PROJECT_DIR":/dynamic-gsplats \
  -w /dynamic-gsplats \
  "$IMAGE_TAG" \
  bash -c "source .venv/bin/activate && exec bash"
