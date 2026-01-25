# Command Cheatsheet

## 🎯 Production Setup (Recommended)

### One-time Setup

```bash
# Terminal 1: Server (keep running)
prefect server start

# Terminal 2: Deploy (run once, or when code updates)
./setup-prefect.sh  # Auto setup
# hoặc manual:
# prefect work-pool create --type process default
# prefect deploy --all

# Terminal 3: Worker (keep running)
prefect worker start --pool default
```

### Trigger Runs

```bash
# CLI
prefect deployment run qwen3_vl_pipeline/local

# UI
http://127.0.0.1:4200 → Deployments → Run
```

### Retry/Resume

```
UI → Flow Runs → Click failed run → Retry/Restart button
```

---

## 🧪 Testing (Simple Mode)

```bash
python pipeline.py
```

---

## 🐳 Docker Commands

### Build & Save

```bash
# Build
./docker-run.sh build v1.0

# Test
./docker-run.sh run

# Save để transfer
./docker-run.sh save v1.0
```

### Load & Run (Prod Machine)

```bash
# Load
./docker-run.sh load qwen3-vl-pipeline-v1.0.tar.gz

# Setup .env
cat > .env << EOF
HF_TOKEN=hf_xxx
GOOGLE_CHAT_WEBHOOK_URL=https://xxx
EOF

# Run
./docker-run.sh run

# hoặc với UI
./docker-run.sh run-with-ui
```

---

## 🔍 Monitoring

```bash
# Logs
./docker-run.sh logs
# hoặc
tail -f runs/*/commands.log

# GPU usage
nvidia-smi

# Artifacts
ls -lh runs/$(ls -t runs/ | head -1)/

# MLflow
open https://dagshub.com/robotpro682004/my-first-repo.mlflow
```

---

## 🐛 Troubleshooting

### vLLM port stuck

```bash
# Host
lsof -ti:8001 | xargs kill -9

# Docker
docker exec qwen3_pipeline pkill -f vllm
```

### Rebuild image

```bash
./docker-run.sh stop
./docker-run.sh build v1.1
./docker-run.sh run
```

### Check Prefect status

```bash
# Work pools
prefect work-pool ls

# Deployments
prefect deployment ls

# Workers
prefect worker ls

# Flow runs
prefect flow-run ls
```

---

## 📝 Config Changes

### Edit config

```bash
vim configs/run.yaml
# Change batch size, GPUs, etc
```

### Apply changes

**Without deployment:**
```bash
python pipeline.py  # Restart từ đầu
```

**With deployment:**
```bash
# If code changed:
prefect deploy --all  # Re-deploy

# If only config changed:
# Just trigger new run, worker will use new config
prefect deployment run qwen3_vl_pipeline/local
```

---

## 🔄 Common Workflows

### Daily Training

```bash
# Schedule daily at 2AM
prefect deployment set-schedule qwen3_vl_pipeline/local \
  --cron "0 2 * * *"
```

### Manual Trigger

```bash
# CLI
prefect deployment run qwen3_vl_pipeline/local

# UI
http://127.0.0.1:4200 → Deployments → Run
```

### Retry Failed

```
UI → Flow Runs → Click failed → Retry button
```

### Resume from Checkpoint

```
UI → Flow Runs → Click failed → Restart button
→ Prefect skips completed tasks
```

---

## 📊 Check Results

```bash
# Latest run
RUN_ID=$(ls -t runs/ | head -1)
echo "Latest run: $RUN_ID"

# Decision
cat runs/$RUN_ID/decision.json | jq .

# SFT metrics
cat runs/$RUN_ID/sft_eval_report.json | jq .

# GRPO metrics
cat runs/$RUN_ID/grpo_eval_report.json | jq .

# Commands executed
cat runs/$RUN_ID/commands.log
```

---

## 🛑 Emergency Stop

```bash
# Stop pipeline
docker-compose down
# hoặc
pkill -f pipeline.py

# Kill vLLM
pkill -f vllm
```

---

## 📦 Docker Quick Ref

```bash
# Build
./docker-run.sh build <version>

# Save
./docker-run.sh save <version>

# Load
./docker-run.sh load <file.tar.gz>

# Run
./docker-run.sh run

# Run with UI
./docker-run.sh run-with-ui

# Shell
./docker-run.sh shell

# Logs
./docker-run.sh logs

# Stop
./docker-run.sh stop

# Clean
./docker-run.sh clean
```
