# 🚀 Quickstart Guide

## Chọn mode:

### 1️⃣ **Simple Run** (Testing)
→ 1 command, no setup  
→ ❌ Fail = restart từ đầu

### 2️⃣ **Prefect Deployment** (Production ⭐)
→ Retry từ UI, resume từ failed step  
→ ✅ Best cho long-running pipelines

### 3️⃣ **Docker** (Deploy máy khác)
→ Protect code, portable  
→ ✅ Combine với deployment = best of both

---

## 1️⃣ Simple Run (Testing)

```bash
# Install deps
pip install -r requirements.txt

# Setup .env
cp .env.example .env
# Edit .env: add HF_TOKEN

# Run
python pipeline.py
```

**⚠️ Warning:** Training mất 5+ giờ, fail ở GRPO → mất 5 giờ!

---

## 2️⃣ Prefect Deployment (Recommended ⭐)

**One-time setup:**

```bash
# Terminal 1: Server
prefect server start

# Terminal 2: Auto setup
./setup-prefect.sh
# Windows: .\setup-prefect.ps1

# Terminal 3: Worker
prefect worker start --pool default
```

**Trigger runs:**

```bash
# CLI
prefect deployment run qwen3_vl_pipeline/local

# hoặc UI
http://127.0.0.1:4200 → Deployments → Run
```

**Retry khi fail:**

```
Train SFT ✅ (2h) → Eval ❌ (network error)
```

UI → Click "Retry" → Eval chạy lại, **không train lại SFT!**

**That's it!** ✨

---

## 2️⃣ Prefect Deployment (Production)

**Setup (one-time):**

```bash
# Terminal 1: Server
prefect server start

# Terminal 2: Auto setup
./setup-prefect.sh
# Windows: .\setup-prefect.ps1
# Tự động: create work pool + deploy pipeline

# Terminal 3: Worker
prefect worker start --pool default
```

**Run pipeline:**

```bash
# Trigger từ CLI
prefect deployment run qwen3_vl_pipeline/local

# hoặc UI: http://127.0.0.1:4200
→ Deployments → qwen3_vl_pipeline/local → Run
```

**Retry/Resume khi fail:**

```
Train SFT ✅ (2h) → Deploy ✅ → Eval ❌ (error)
```

UI → Flow Runs → Click failed run → **Retry** button  
→ Chỉ retry Eval, **KHÔNG train lại SFT!**

**Save hours of GPU time!** 💰

---

## 3️⃣ Docker Deployment

### Tại máy Dev

```bash
# Build image
./docker-run.sh build v1.0
# Windows: .\docker-run.ps1 build v1.0

# Save để chuyển sang máy khác
./docker-run.sh save v1.0
# → qwen3-vl-pipeline-v1.0.tar.gz
```

### Transfer

```bash
# USB, SCP, hoặc share drive
scp qwen3-vl-pipeline-v1.0.tar.gz user@prod:/tmp/
```

### Tại máy Prod

```bash
# Load image
./docker-run.sh load qwen3-vl-pipeline-v1.0.tar.gz

# Setup .env
cat > .env << EOF
HF_TOKEN=hf_xxx
GOOGLE_CHAT_WEBHOOK_URL=https://xxx
EOF

# Create baseline reports
mkdir -p reports
cat > reports/sft_best.json << 'EOF'
{"winrate": 0.5, "accuracy": 0.5, "p95_latency_ms": 1200, "format_pass_rate": 0.98, "model_repo": "baseline", "model_revision": "baseline", "run_id": "baseline", "stage": "sft", "timestamp": "2026-01-01T00:00:00Z"}
EOF

cat > reports/grpo_prod.json << 'EOF'
{"winrate": 0.6, "accuracy": 0.6, "p95_latency_ms": 1200, "format_pass_rate": 0.98, "model_repo": "baseline", "model_revision": "baseline", "run_id": "baseline", "stage": "grpo", "timestamp": "2026-01-01T00:00:00Z"}
EOF

# Run pipeline
./docker-run.sh run
```

**Done!** 🎉

---

## 📖 Full Docs

- **README.md** - Full documentation
- **DOCKER_GUIDE.md** - Docker chi tiết
- **INSTRUCTION.md** - Pipeline architecture

---

## 🔧 Config

Edit `configs/run.yaml`:

```yaml
hf:
  org_or_user: "ha684"  # ← Đổi thành org của bạn

train_sft:
  num_gpus: 1
  batch_size: 2

train_grpo:
  num_gpus: 4
  batch_size: 4

notify:
  backend: "google_chat"  # hoặc "slack", "email"
```

---

## 📊 Monitoring

### Logs
```bash
# Direct run
tail -f runs/*/commands.log

# Docker
./docker-run.sh logs
```

### Artifacts
```
runs/<run_id>/
├── sft_eval_report.json
├── grpo_eval_report.json
├── decision.json
└── meta.json
```

### MLflow
```
https://dagshub.com/robotpro682004/my-first-repo.mlflow
```

---

## ❓ Help

```bash
# Direct run
python pipeline.py --help

# Docker
./docker-run.sh
# Shows all commands
```

**Common issues → README.md Troubleshooting section**
