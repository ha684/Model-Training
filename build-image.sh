#!/bin/bash
# Build Docker image cho Qwen3-VL Pipeline

set -e

VERSION=${1:-latest}
IMAGE_NAME="qwen3-vl-pipeline"
REGISTRY=${REGISTRY:-}  # Set này nếu muốn push lên private registry

echo "🔨 Building Docker image: ${IMAGE_NAME}:${VERSION}"

# Build image
docker build -t ${IMAGE_NAME}:${VERSION} .

# Tag latest
docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:latest

echo "✅ Build complete!"
echo ""
echo "Image tags:"
docker images | grep ${IMAGE_NAME}

echo ""
echo "📦 To save image to file:"
echo "   docker save ${IMAGE_NAME}:${VERSION} | gzip > ${IMAGE_NAME}-${VERSION}.tar.gz"
echo ""
echo "📤 To push to registry (optional):"
echo "   docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
echo "   docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
echo ""
echo "🚀 To run:"
echo "   docker-compose up pipeline"
