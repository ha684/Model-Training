#!/bin/bash
# Helper script để chạy pipeline trong Docker

set -e

ACTION=${1:-run}

case $ACTION in
    run)
        echo "🚀 Running pipeline (SIMPLE MODE - no retry/resume)..."
        echo "⚠️  This mode runs 'python pipeline.py' directly"
        echo "   Fail = restart từ đầu. Use 'run-production' for retry/resume."
        echo ""
        docker-compose --profile simple up pipeline-simple
        ;;
    
    run-production)
        echo "🚀 Starting Production Mode (with retry/resume)..."
        echo ""
        echo "Step 1: Starting Prefect server..."
        docker-compose up -d prefect-server postgres
        
        echo "⏳ Waiting for server to be ready..."
        sleep 15
        
        echo ""
        echo "Step 2: Creating work pool..."
        docker exec prefect_server prefect work-pool create --type process default 2>/dev/null || echo "   Work pool already exists"
        
        echo ""
        echo "Step 3: Deploying pipeline..."
        docker-compose run --rm \
          -e PREFECT_API_URL=http://prefect-server:4200/api \
          prefect-worker prefect deploy --all
        
        echo ""
        echo "Step 4: Starting worker..."
        docker-compose up -d prefect-worker
        
        echo ""
        echo "=========================================="
        echo "✅ Production setup complete!"
        echo "=========================================="
        echo ""
        echo "Trigger run:"
        echo "  CLI: prefect deployment run qwen3_vl_pipeline/local"
        echo "  UI: http://localhost:4200 → Deployments → Run button"
        echo ""
        echo "Monitor:"
        echo "  docker-compose logs -f prefect-worker"
        echo ""
        echo "Retry failed tasks:"
        echo "  UI → Flow Runs → Click failed run → Retry button"
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
        echo "Usage: $0 {run|run-production|shell|logs|stop|clean|build|save|load}"
        echo ""
        echo "Commands:"
        echo "  run               - Run pipeline (SIMPLE mode - no retry/resume)"
        echo "  run-production    - Run with Prefect deployment (HAS retry/resume)"
        echo "  shell             - Open interactive shell"
        echo "  logs              - Show logs"
        echo "  stop              - Stop all services"
        echo "  clean             - Clean up containers and volumes"
        echo "  build [ver]       - Build Docker image"
        echo "  save [ver]        - Save image to .tar.gz"
        echo "  load <file>       - Load image from .tar.gz"
        echo ""
        echo "Recommendations:"
        echo "  Testing:    ./docker-run.sh run"
        echo "  Production: ./docker-run.sh run-production"
        exit 1
        ;;
esac
