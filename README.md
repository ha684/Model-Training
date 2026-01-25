# Qwen3-VL Pipeline

End-to-end orchestration: SFT → eval → gate → GRPO → eval → gate → promote

**📚 Guides:**
- **QUICKSTART.md** - Quick reference
- **DEPLOYMENT_EXAMPLE.md** - Tại sao cần deployment? (Real cost savings example)
- **PREFECT_MODES.md** - 3 modes explained
- **DOCKER_GUIDE.md** - Docker deployment

---

## 🚀 Quickstart - Chọn Mode

| Mode | Use Case | Setup | Retry/Resume? |
|------|----------|-------|---------------|
| **Simple** | Testing, one-off | 1 command | ❌ Chạy lại từ đầu |
| **Deployment** | Production | 4 terminals | ✅ Retry từ UI, resume từ failed step |
| **Docker** | Deploy máy khác | Build + transfer image | ✅ (nếu dùng deployment) |

### Option 1: Simple Mode (Testing)

```bash
# Setup
cp .env.example .env
# Edit .env: add HF_TOKEN

# Run
python pipeline.py
```

**⚠️ Fail → restart từ đầu. Train 5 giờ rồi fail = mất 5 giờ!**

### Option 2: Prefect Deployment (Production ⭐)

**One-time setup:**
```bash
# Terminal 1: Server
prefect server start

# Terminal 2: Setup (chạy script tự động)
./setup-prefect.sh
# Windows: .\setup-prefect.ps1

# Terminal 3: Worker  
prefect worker start --pool default
```

**Trigger runs:**
```bash
# CLI
prefect deployment run qwen3_vl_pipeline/local

# hoặc UI: http://127.0.0.1:4200 → Run button
```

**✅ Fail? → Click Retry trong UI, chỉ chạy lại failed task!**

### Option 3: Docker

**Tại máy phát triển (build image):**

```bash
# 1. Build image
docker build -t qwen3-vl-pipeline:latest .
# hoặc
./build-image.sh v1.0

# 2. Save image để chuyển sang máy khác
docker save qwen3-vl-pipeline:latest | gzip > qwen3-vl-pipeline.tar.gz
```

**Tại máy production (load & run):**

```bash
# 1. Load image
docker load < qwen3-vl-pipeline.tar.gz

# 2. Tạo .env
cat > .env << EOF
HF_TOKEN=hf_xxx
GOOGLE_CHAT_WEBHOOK_URL=https://chat.googleapis.com/v1/spaces/xxx/messages?key=xxx
EOF

# 3. Run pipeline
docker-compose up pipeline

# Hoặc chạy trực tiếp
docker run --rm \
  --gpus all \
  --env-file .env \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/runs:/app/runs \
  qwen3-vl-pipeline:latest
```

---

## 🎯 Prefect Deployment Workflow (Production)

### Tại sao cần Deployment?

**Flow Run trực tiếp** (`python pipeline.py`):
```
SFT (2h) → vLLM → Eval → Gate ✅ → GRPO (3h) → vLLM → Eval ❌
                                                              ↑
                                                         Fail here
→ Phải chạy lại TỪ ĐẦU (5+ hours wasted!)
```

**Deployment + Worker**:
```
SFT (2h) ✅ → vLLM ✅ → Eval ✅ → Gate ✅ → GRPO (3h) ✅ → vLLM ✅ → Eval ❌
                                                                      ↑
                                                                  Fail here
→ Click "Retry" trong UI → chỉ chạy lại Eval (saved 5+ hours!)
```

### Setup (One-time)

#### Step 1: Start Prefect Server

**Terminal 1:**
```bash
prefect server start
# Server: http://127.0.0.1:4200
```

#### Step 2: Create Work Pool

**Terminal 2:**
```bash
# Check existing pools
prefect work-pool ls

# Create default pool
prefect work-pool create --type process default
```

#### Step 3: Deploy Pipeline

```bash
# Option A: Deploy tất cả từ prefect.yaml
prefect deploy --all

# Option B: Custom deploy
prefect deployment build pipeline.py:pipeline_flow \
  -n local \
  -p default \
  --apply
```

Output:
```
✅ Deployment 'qwen3_vl_pipeline/local' created
```

#### Step 4: Start Worker

**Terminal 3:**
```bash
prefect worker start --pool default
```

Output:
```
Worker 'default' polling for work...
```

### Trigger Runs

#### Option A: UI (Recommended)

1. Mở http://127.0.0.1:4200
2. **Deployments** → `qwen3_vl_pipeline/local`
3. Click **"Run"** button
4. (Optional) Override params → **Submit**
5. Theo dõi progress trong **Flow Runs** tab

#### Option B: CLI

```bash
prefect deployment run qwen3_vl_pipeline/local
```

With custom config:
```bash
prefect deployment run qwen3_vl_pipeline/local \
  --param config_path=configs/test.yaml
```

### Resume/Retry từ Failures (CHỈ CÓ VỚI DEPLOYMENT!)

**⚠️ Important:** Retry/Resume **CHỈ HOẠT ĐỘNG** với Deployment + Worker, KHÔNG hoạt động với `python pipeline.py`!

#### Scenario 1: Network error during eval

**Timeline:**
```
[10:00] Train SFT starts...
[12:00] Train SFT ✅ (saved state)
[12:01] Deploy vLLM ✅ (saved state)  
[12:05] Eval SFT ❌ (network timeout)
```

**Fix (CHỈ VỚI DEPLOYMENT):**
1. UI → Flow Run detail
2. Click failed task → **"Retry"** button  
3. Prefect check saved state:
   - Train SFT = completed → **SKIP**
   - Deploy vLLM = completed → **SKIP**
   - Eval SFT = failed → **RETRY**

```
[12:10] Retry Eval SFT...
[12:15] Eval SFT ✅
[12:16] Continue: Gate → GRPO → ...
```

**→ Saved 2 hours training!** ✨

#### Scenario 2: GRPO OOM

**Timeline:**
```
[10:00] Train SFT (2h) ✅
[12:00] Deploy + Eval + Gate ✅
[12:30] Train GRPO starts...
[15:30] Train GRPO ❌ (OOM after 3h)
```

**Fix (CHỈ VỚI DEPLOYMENT):**
1. Edit `configs/run.yaml`:
   ```yaml
   train_grpo:
     batch_size: 2  # giảm từ 4
     grad_accum: 2  # tăng để maintain effective batch
   ```
2. UI → Click **"Restart"** button
3. Prefect check saved state:
   - Train SFT = completed → **SKIP**
   - Deploy vLLM SFT = completed → **SKIP**
   - Eval SFT = completed → **SKIP**
   - Gate SFT = completed → **SKIP**
   - Train GRPO = failed → **RESTART với config mới**

```
[15:35] Restart Train GRPO với batch_size=2...
[18:35] Train GRPO ✅
[18:36] Continue: Deploy GRPO → Eval → ...
```

**→ Saved 5.5 hours!** 🎉

#### Scenario 3: Nếu dùng `python pipeline.py` (KHÔNG CÓ DEPLOYMENT)

**Timeline:**
```
[10:00] Train SFT (2h)...
[12:00] Train SFT ✅
[12:30] Train GRPO (3h)...
[15:30] Train GRPO ❌ (OOM)
```

**Fix:**
1. Edit config
2. Chạy lại: `python pipeline.py`
3. Pipeline restart:
   - Train SFT từ đầu (2h) 😢
   - Gate
   - Train GRPO (3h)

```
[15:35] Train SFT lại... (wasted 2h)
[17:35] Train SFT ✅
[18:05] Train GRPO...
[21:05] Train GRPO ✅
```

**→ Wasted 2 hours!** 😭

---

## 🆚 Summary: Direct Run vs Deployment

| Scenario | Direct Run | Deployment |
|----------|------------|------------|
| Eval fail ở 2h mark | ❌ Restart từ đầu | ✅ Retry chỉ eval (save 2h) |
| GRPO OOM ở 5h mark | ❌ Restart từ đầu | ✅ Resume từ GRPO (save 5h) |
| Network glitch | ❌ Crash + restart | ✅ Auto retry (Prefect config) |
| Schedule daily | ❌ Manual trigger | ✅ Cron schedule |

**→ Cho pipeline dài (5+ giờ), LUÔN DÙNG DEPLOYMENT!**

### Schedule Runs

```bash
# Chạy mỗi ngày lúc 2AM
prefect deployment set-schedule qwen3_vl_pipeline/local \
  --cron "0 2 * * *"

# Hoặc interval
prefect deployment set-schedule qwen3_vl_pipeline/local \
  --interval 86400  # 24 hours
```

### Docker với Deployment

**docker-compose.yml** đã include:
- `prefect-server`: UI server
- `prefect-worker`: Worker để execute jobs
- `postgres`: Database cho server

**Run:**
```bash
# Start server + worker
docker-compose up -d prefect-server postgres prefect-worker

# Deploy (trong container)
docker exec prefect_worker prefect deploy --all

# Trigger từ UI hoặc CLI
prefect deployment run qwen3_vl_pipeline/local
```

---

## 📦 Docker Setup Chi Tiết

### Prerequisites

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 2. Install NVIDIA Container Toolkit (cho GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 3. Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Build & Deploy

**Windows:**
```powershell
.\build-image.ps1 -Version v1.0
```

**Linux/Mac:**
```bash
bash build-image.sh v1.0
```

### Running Modes

#### Mode 1: Simple Run (Ephemeral Prefect)

```bash
# Chỉ chạy pipeline, không UI
docker-compose up pipeline
```

Logs → console + `runs/<run_id>/`

#### Mode 2: With Prefect UI

```bash
# Start cả server + worker + pipeline
docker-compose up -d prefect-server postgres
docker-compose up prefect-worker

# Mở UI: http://localhost:4200
# Trigger pipeline từ UI hoặc CLI:
docker exec qwen3_pipeline python pipeline.py
```

#### Mode 3: Interactive Shell

```bash
# Vào container để debug
docker-compose run --rm pipeline /bin/bash

# Inside container:
python pipeline.py
# hoặc
python -m qwen3_vl.evaluation_cli --base_url http://host:8000 --model xxx --dataset xxx --out runs/test.json
```

### Docker Compose Services

```yaml
services:
  pipeline:         # Main pipeline runner
  prefect-server:  # Prefect UI (optional)
  prefect-worker:  # Worker cho deployments (optional)
  postgres:        # DB cho Prefect server (optional)
```

**Chạy từng service:**

```bash
# Chỉ pipeline (không cần Prefect UI)
docker-compose up pipeline

# Full stack (có UI)
docker-compose up -d prefect-server postgres
docker-compose up prefect-worker
```

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Required
HF_TOKEN=hf_xxx

# Notification (chọn 1)
GOOGLE_CHAT_WEBHOOK_URL=https://chat.googleapis.com/v1/spaces/xxx
# hoặc
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx
# hoặc
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

### Pipeline Config (configs/run.yaml)

```yaml
hf:
  org_or_user: "ha684"  # Thay bằng org của bạn

train_sft:
  num_gpus: 1
  batch_size: 2
  # ... other params

train_grpo:
  num_gpus: 4
  batch_size: 4
  # ... other params

gate:
  sft:
    min_winrate_delta_vs_sft_baseline: -0.01
  grpo:
    min_winrate_delta_vs_grpo_baseline: 0.02
```

---

## 📊 Monitoring

### 1. Console Logs

```bash
# Follow logs
docker-compose logs -f pipeline
```

### 2. Artifacts

```
runs/<run_id>/
├── commands.log           # Commands executed (HF_TOKEN masked)
├── sft_eval_report.json   # SFT metrics
├── grpo_eval_report.json  # GRPO metrics
├── decision.json          # Gate decisions
├── meta.json              # Full metadata
└── vllm.log              # vLLM logs
```

### 3. MLflow

```
https://dagshub.com/robotpro682004/my-first-repo.mlflow
```

### 4. Prefect UI (nếu dùng server)

```
http://localhost:4200
```

---

## 🐛 Troubleshooting

### Docker issues

```bash
# Check GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check logs
docker-compose logs pipeline

# Rebuild image sau khi sửa code
docker-compose build --no-cache pipeline
docker-compose up pipeline
```

### Pipeline fails

```bash
# Xem logs chi tiết
docker-compose logs -f pipeline

# Vào container để debug
docker-compose run --rm pipeline /bin/bash
# Inside:
python pipeline.py
```

### Port conflicts

```bash
# vLLM port bị chiếm
docker exec qwen3_pipeline netstat -tuln | grep 8001
# Kill process nếu cần
docker exec qwen3_pipeline pkill -f vllm
```

### Out of disk space

```bash
# Clean Docker cache
docker system prune -a
docker volume prune
```

---

## 📝 Notes

### MLflow & ngrok helpers

```bash
mlflow server --host 0.0.0.0 --port 5000
ngrok http 5000 --domain=smart-kind-macaque.ngrok-free.app
```

### Datasets

- `vl-vi-conv-mix-v1.3` = latest prod
- `vl-vi-conv-mix-v1.2` = SFT dataset
- `vl-vi-conv-mix-v1.1` = processed merged

### Image Distribution

**Build tại máy dev:**
```bash
docker build -t qwen3-vl-pipeline:v1.0 .
docker save qwen3-vl-pipeline:v1.0 | gzip > qwen3-vl-pipeline-v1.0.tar.gz
```

**Deploy tại máy prod:**
```bash
# Copy file .tar.gz qua máy prod
scp qwen3-vl-pipeline-v1.0.tar.gz user@prod-server:/tmp/

# SSH vào máy prod
ssh user@prod-server

# Load image
docker load < /tmp/qwen3-vl-pipeline-v1.0.tar.gz

# Setup .env
cat > .env << EOF
HF_TOKEN=hf_xxx
GOOGLE_CHAT_WEBHOOK_URL=https://xxx
EOF

# Run
docker-compose up -d pipeline
```

**Hoặc dùng private registry:**
```bash
# Push từ máy dev
docker tag qwen3-vl-pipeline:v1.0 your-registry.com/qwen3-vl-pipeline:v1.0
docker push your-registry.com/qwen3-vl-pipeline:v1.0

# Pull tại máy prod
docker pull your-registry.com/qwen3-vl-pipeline:v1.0
docker tag your-registry.com/qwen3-vl-pipeline:v1.0 qwen3-vl-pipeline:latest
docker-compose up -d pipeline
```
