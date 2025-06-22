#!/bin/bash
set -e
export DOCKER_BUILDKIT=1

# Image names
BASE_TAG=dynamic-gsplats-base
FINAL_TAG=${1:-dynamic-gsplats-dev}

# Build base image
docker build -f docker/Dockerfile.base -t "$BASE_TAG" .

# Build final image using the base
docker build -f docker/Dockerfile -t "$FINAL_TAG" .
