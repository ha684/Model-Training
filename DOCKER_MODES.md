# Docker Modes Explained

## ⚠️ Critical: Docker có 2 modes khác nhau!

---

## Mode 1: Simple Run (Testing)

```bash
./docker-run.sh run
```

**Cách hoạt động:**
```
Docker Container
├── python pipeline.py  ← Chạy TRỰC TIẾP
│   ├── Train SFT
│   ├── Deploy vLLM
│   ├── Eval
│   └── ... (sequential execution)
└── Fail → Container exit → phải chạy lại từ đầu
```

**Docker compose:**
```yaml
pipeline-simple:
  command: python pipeline.py  # ← Direct execution
```

**❌ Không có:**
- Retry từ UI
- Resume từ failed step
- Task state tracking

**✅ Use case:**
- Quick testing
- One-off runs
- Development

---

## Mode 2: Production (Deployment + Worker)

```bash
./docker-run.sh run-production
```

**Cách hoạt động:**
```
┌─────────────────────────────────────────────┐
│ Container: prefect-server                   │
│ - Store deployments                         │
│ - Store task states                         │
│ - Serve UI (port 4200)                      │
└─────────────────────────────────────────────┘
                    ↕ API
┌─────────────────────────────────────────────┐
│ Container: prefect-worker                   │
│ 1. Pull job from queue                      │
│ 2. Execute: Train SFT → Save state ✅       │
│ 3. Execute: Deploy vLLM → Save state ✅     │
│ 4. Execute: Eval SFT → Fail ❌              │
│    └→ Save failure state                    │
│ 5. Wait for retry/restart command          │
│                                             │
│ When user clicks "Retry" in UI:            │
│ 6. Load saved states                        │
│ 7. Skip: Train SFT (completed)             │
│ 8. Skip: Deploy vLLM (completed)           │
│ 9. Retry: Eval SFT                          │
│ 10. Continue: Gate → GRPO → ...            │
└─────────────────────────────────────────────┘
```

**Docker compose:**
```yaml
prefect-server:
  command: prefect server start

prefect-worker:
  command: prefect worker start --pool default
  # Worker execute jobs từ queue
```

**✅ Có:**
- Retry từ UI
- Resume từ failed step
- Task state tracking
- Schedule
- Queue management

**✅ Use case:**
- Production
- Long-running pipelines
- Need reliability

---

## 📊 Comparison

### Scenario: Pipeline fail ở GRPO (sau 5h)

#### Mode 1: Simple

```bash
./docker-run.sh run
```

```
Container start
[10:00] Train SFT (2h)...
[12:00] Train SFT ✅
[12:30] Train GRPO (3h)...
[15:30] Train GRPO ❌ Fail
[15:30] Container exit

Fix config...

./docker-run.sh run  # Restart
[15:35] Train SFT (2h)... ← LẶP LẠI!
[17:35] Train SFT ✅
[18:05] Train GRPO (3h)...
[21:05] Done

Total: 5h + 5.5h = 10.5h
```

#### Mode 2: Production

```bash
./docker-run.sh run-production
prefect deployment run qwen3_vl_pipeline/local
```

```
Server + Worker start
[10:00] Worker pull job
[10:00] Train SFT (2h)...
[12:00] Train SFT ✅ → State saved to server
[12:30] Train GRPO (3h)...
[15:30] Train GRPO ❌ Fail → State saved
[15:30] Worker waiting for next job

Fix config...

UI → Click "Restart"
[15:35] Worker load saved states
[15:35] Skip Train SFT (completed)
[15:35] Restart Train GRPO (3h)...
[18:35] Done

Total: 5h + 3h = 8h (saved 2.5h!)
```

---

## 🎯 Which Mode?

### Use Simple Mode when:
- ✅ Testing pipeline quickly
- ✅ One-off experimental runs
- ✅ Pipeline mất < 1h (ok để restart)

### Use Production Mode when:
- ✅ Pipeline mất 5+ giờ
- ✅ Training expensive (GPU cost)
- ✅ High chance of failures (network, OOM)
- ✅ Need schedule chạy định kỳ
- ✅ Production deployment

---

## 🚀 Quick Commands

### Simple Mode (Testing)

```bash
# Build & test
./docker-run.sh build v1.0
./docker-run.sh run

# Logs
./docker-run.sh logs

# Stop
./docker-run.sh stop
```

### Production Mode (Deployment)

```bash
# Build
./docker-run.sh build v1.0

# Setup & start
./docker-run.sh run-production

# Trigger run
prefect deployment run qwen3_vl_pipeline/local

# Monitor UI
open http://localhost:4200

# Retry failed
UI → Flow Runs → Click failed → Retry

# Stop
./docker-run.sh stop
```

---

## 📝 Config trong Production Mode

File structure tại máy prod:

```
/workspace/pipeline/
├── docker-run.sh                 # Helper script
├── docker-compose.yml            # Services config
├── .env                          # Secrets
├── configs/                      # Mounted volume
│   └── run.yaml                  # Editable từ host
├── reports/                      # Mounted volume
│   ├── sft_best.json
│   └── grpo_prod.json
├── outputs/                      # Mounted volume (training artifacts)
└── runs/                         # Mounted volume (pipeline artifacts)
```

**Edit config từ host:**
```bash
# Host machine (không cần vào container)
vim configs/run.yaml

# Change batch size
train_grpo:
  batch_size: 2

# Worker sẽ dùng config mới cho run tiếp theo
```

**Trigger run mới:**
```bash
prefect deployment run qwen3_vl_pipeline/local
```

---

## 🔍 Monitoring Production Mode

### Check worker status

```bash
docker-compose logs -f prefect-worker
```

### Check deployments

```bash
docker exec prefect_worker prefect deployment ls
```

### Check runs

```bash
docker exec prefect_worker prefect flow-run ls
```

### UI

```
http://localhost:4200
→ Flow Runs: See all runs, states, logs
→ Deployments: Trigger new runs
→ Work Pools: See worker status
```

Done! 🎉
