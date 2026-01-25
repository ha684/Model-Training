# Docker Deployment Guide

Quick guide để deploy Qwen3-VL Pipeline lên máy production bằng Docker.

## 📦 Workflow: Dev → Prod

### Tại máy Dev (build image)

```bash
# 1. Build image
./docker-run.sh build v1.0
# hoặc Windows
.\docker-run.ps1 build v1.0

# 2. Save image ra file
./docker-run.sh save v1.0
# → tạo file: qwen3-vl-pipeline-v1.0.tar.gz (~5-10GB)
```

### Transfer sang máy Prod

```bash
# SCP
scp qwen3-vl-pipeline-v1.0.tar.gz user@prod-server:/tmp/

# hoặc USB/shared drive
cp qwen3-vl-pipeline-v1.0.tar.gz /mnt/share/
```

### Tại máy Prod (load & run)

```bash
# 1. Load image
./docker-run.sh load qwen3-vl-pipeline-v1.0.tar.gz

# 2. Setup environment
cat > .env << EOF
HF_TOKEN=hf_xxx
GOOGLE_CHAT_WEBHOOK_URL=https://chat.googleapis.com/v1/spaces/xxx/messages?key=xxx
EOF

# 3. Tạo baseline reports (nếu chưa có)
mkdir -p reports
cat > reports/sft_best.json << EOF
{
  "winrate": 0.5,
  "accuracy": 0.5,
  "p95_latency_ms": 1200.0,
  "format_pass_rate": 0.98,
  "model_repo": "baseline",
  "model_revision": "baseline",
  "run_id": "baseline",
  "stage": "sft",
  "timestamp": "2026-01-01T00:00:00Z"
}
EOF

cat > reports/grpo_prod.json << EOF
{
  "winrate": 0.6,
  "accuracy": 0.6,
  "p95_latency_ms": 1200.0,
  "format_pass_rate": 0.98,
  "model_repo": "baseline",
  "model_revision": "baseline",
  "run_id": "baseline",
  "stage": "grpo",
  "timestamp": "2026-01-01T00:00:00Z"
}
EOF

# 4. Run pipeline
./docker-run.sh run

# hoặc với Prefect UI
./docker-run.sh run-with-ui
```

## 🔧 Config trong Docker

### Volumes được mount

```yaml
volumes:
  - ./data:/app/data         # Input data
  - ./outputs:/app/outputs   # Training outputs
  - ./runs:/app/runs         # Pipeline artifacts
  - ./reports:/app/reports   # Baseline reports
  - ./configs:/app/configs   # Config files (editable)
  - ./.env:/app/.env:ro      # Environment vars
```

**→ Có thể edit config từ host machine, container sẽ dùng config mới!**

### Edit config

```bash
# Tại host machine (không cần vào container)
vim configs/run.yaml

# Change:
train_sft:
  batch_size: 4  # Thay đổi batch size
  num_gpus: 2    # Thay đổi số GPU

# Run lại
docker-compose restart pipeline
```

## 🚀 Common Tasks

### Run pipeline

```bash
./docker-run.sh run
```

### View logs

```bash
./docker-run.sh logs
```

### Debug trong container

```bash
./docker-run.sh shell
# Inside container:
python pipeline.py
# hoặc
python -m qwen3_vl.evaluation_cli --base_url http://10.0.64.77:8000 --model xxx --dataset xxx --out test.json
```

### Stop pipeline

```bash
./docker-run.sh stop
```

### Update code (rebuild)

```bash
# Tại máy dev: rebuild
./docker-run.sh build v1.1
./docker-run.sh save v1.1

# Transfer sang prod
scp qwen3-vl-pipeline-v1.1.tar.gz user@prod-server:/tmp/

# Tại prod: load image mới
./docker-run.sh stop
./docker-run.sh load qwen3-vl-pipeline-v1.1.tar.gz
./docker-run.sh run
```

## 📊 Monitoring

### Check outputs

```bash
# Artifacts
ls -lh runs/

# Training checkpoints
ls -lh outputs/

# Logs
docker-compose logs -f pipeline
```

### Check GPU usage

```bash
# From host
nvidia-smi

# From container
docker exec qwen3_pipeline nvidia-smi
```

## 🐛 Troubleshooting

### Container không start

```bash
# Check logs
docker-compose logs pipeline

# Check GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Out of memory

```bash
# Giảm batch size trong configs/run.yaml
train_sft:
  batch_size: 1
  grad_accum: 16  # Tăng grad accum để maintain effective batch size
```

### Port conflicts

```bash
# vLLM port bị chiếm
docker exec qwen3_pipeline pkill -f vllm
```

### Disk space

```bash
# Clean old outputs
rm -rf outputs/old_*
rm -rf runs/2024*  # Old runs

# Clean Docker cache
docker system prune -a
```

## 🔐 Security Notes

### Protect .env

```bash
# Không commit .env
echo ".env" >> .gitignore

# Set permissions
chmod 600 .env
```

### Private registry (optional)

Nếu muốn dùng private Docker registry thay vì save/load file:

```bash
# Tại dev: push
docker tag qwen3-vl-pipeline:v1.0 registry.company.com/qwen3-vl-pipeline:v1.0
docker push registry.company.com/qwen3-vl-pipeline:v1.0

# Tại prod: pull
docker login registry.company.com
docker pull registry.company.com/qwen3-vl-pipeline:v1.0
docker tag registry.company.com/qwen3-vl-pipeline:v1.0 qwen3-vl-pipeline:latest
./docker-run.sh run
```

## 📝 Complete Example

```bash
# === Tại máy DEV ===
# Build
./docker-run.sh build v1.0

# Test local
./docker-run.sh run

# Save image
./docker-run.sh save v1.0
# → qwen3-vl-pipeline-v1.0.tar.gz

# === Transfer ===
scp qwen3-vl-pipeline-v1.0.tar.gz prod-server:/tmp/

# === Tại máy PROD ===
ssh prod-server
cd /workspace/pipeline

# Load image
./docker-run.sh load /tmp/qwen3-vl-pipeline-v1.0.tar.gz

# Setup .env (one-time)
cat > .env << 'EOF'
HF_TOKEN=hf_xxx
GOOGLE_CHAT_WEBHOOK_URL=https://xxx
EOF

# Create baseline reports (one-time)
mkdir -p reports
# ... (copy từ section trên)

# Run
./docker-run.sh run

# Monitor
./docker-run.sh logs

# Check results
ls -lh runs/$(ls -t runs/ | head -1)/
```

Done! 🎉
