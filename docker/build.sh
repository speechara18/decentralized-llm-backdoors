#!/bin/bash
# Build and push the ICLScan image to the EPFL RCP registry.
# Run this from the docker/ directory on your machine (WSL).
set -e

IMAGE="registry.rcp.epfl.ch/sacs-peechara/iclscan:latest"

echo ">>> Building ${IMAGE} for linux/amd64 ..."
docker buildx build \
    -f Dockerfile \
    --platform linux/amd64 \
    -t "${IMAGE}" \
    --load .

echo ">>> Build done. Pushing to registry ..."
docker push "${IMAGE}"

echo ">>> Pushed ${IMAGE}"
