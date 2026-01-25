#!/bin/bash
# Setup Prefect deployment (one-time)

set -e

echo "🔧 Prefect Deployment Setup"
echo "============================"

# Check if server is running
echo ""
echo "Checking Prefect server..."
if curl -s http://127.0.0.1:4200/api/health >/dev/null 2>&1; then
    echo "✅ Prefect server is running"
else
    echo "❌ Prefect server is NOT running"
    echo ""
    echo "Please start server in another terminal:"
    echo "   prefect server start"
    echo ""
    read -p "Press Enter when server is ready..."
fi

# Create work pool
echo ""
echo "Creating work pool 'default'..."
if prefect work-pool inspect default >/dev/null 2>&1; then
    echo "✅ Work pool 'default' already exists"
else
    prefect work-pool create --type process default
    echo "✅ Work pool 'default' created"
fi

# Deploy pipeline
echo ""
echo "Deploying pipeline..."
prefect deploy --all
echo "✅ Deployment created"

# Summary
echo ""
echo "============================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Start worker (Terminal 3):"
echo "   prefect worker start --pool default"
echo ""
echo "2. Trigger run:"
echo "   prefect deployment run qwen3_vl_pipeline/local"
echo ""
echo "3. Monitor:"
echo "   UI: http://127.0.0.1:4200"
echo ""
echo "To restart from failed task:"
echo "   UI → Flow Runs → Click failed run → Restart button"
