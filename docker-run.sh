#!/bin/bash
# Helper script để chạy pipeline trong Docker

set -e

ACTION=${1:-run}

case $ACTION in
    run)
        echo "🚀 Running pipeline..."
        docker-compose up pipeline
        ;;
    
    run-with-ui)
        echo "🚀 Starting Prefect server + worker..."
        docker-compose up -d prefect-server postgres
        echo "⏳ Waiting for Prefect server..."
        sleep 15
        
        echo "📦 Deploying pipeline..."
        docker exec prefect_server prefect work-pool create --type process default 2>/dev/null || true
        docker-compose run --rm pipeline prefect deploy --all
        
        echo "🏃 Starting worker..."
        docker-compose up -d prefect-worker
        
        echo ""
        echo "✅ Setup complete!"
        echo "   Prefect UI: http://localhost:4200"
        echo "   Trigger run: prefect deployment run qwen3_vl_pipeline/local"
        echo "   Or use UI → Deployments → Run button"
        ;;
    
    shell)
        echo "🐚 Opening interactive shell..."
        docker-compose run --rm pipeline /bin/bash
        ;;
    
    logs)
        echo "📋 Showing logs..."
        docker-compose logs -f pipeline
        ;;
    
    stop)
        echo "🛑 Stopping all services..."
        docker-compose down
        ;;
    
    clean)
        echo "🧹 Cleaning up..."
        docker-compose down -v
        docker system prune -f
        ;;
    
    build)
        VERSION=${2:-latest}
        echo "🔨 Building image version: $VERSION"
        docker build -t qwen3-vl-pipeline:$VERSION .
        docker tag qwen3-vl-pipeline:$VERSION qwen3-vl-pipeline:latest
        echo "✅ Build complete!"
        ;;
    
    save)
        VERSION=${2:-latest}
        OUTPUT="qwen3-vl-pipeline-${VERSION}.tar.gz"
        echo "💾 Saving image to $OUTPUT..."
        docker save qwen3-vl-pipeline:$VERSION | gzip > $OUTPUT
        echo "✅ Saved: $OUTPUT ($(du -h $OUTPUT | cut -f1))"
        ;;
    
    load)
        FILE=${2}
        if [ -z "$FILE" ]; then
            echo "❌ Please provide image file: $0 load <file.tar.gz>"
            exit 1
        fi
        echo "📥 Loading image from $FILE..."
        docker load < $FILE
        echo "✅ Loaded!"
        docker images | grep qwen3-vl-pipeline
        ;;
    
    *)
        echo "Usage: $0 {run|run-with-ui|shell|logs|stop|clean|build|save|load}"
        echo ""
        echo "Commands:"
        echo "  run          - Run pipeline (ephemeral mode)"
        echo "  run-with-ui  - Run with Prefect UI"
        echo "  shell        - Open interactive shell"
        echo "  logs         - Show logs"
        echo "  stop         - Stop all services"
        echo "  clean        - Clean up containers and volumes"
        echo "  build [ver]  - Build Docker image"
        echo "  save [ver]   - Save image to .tar.gz"
        echo "  load <file>  - Load image from .tar.gz"
        exit 1
        ;;
esac
