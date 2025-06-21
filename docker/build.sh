#!/bin/bash
set -e
export DOCKER_BUILDKIT=1
IMAGE_TAG=${1:-dynamic-gsplats-dev}
docker build -f docker/Dockerfile -t "$IMAGE_TAG" .
