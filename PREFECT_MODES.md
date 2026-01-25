# Prefect Modes Explained

## 🤔 3 cách chạy pipeline

### 1. Simple Run (Ephemeral)

```bash
python pipeline.py
```

**Cách hoạt động:**
- Python execute trực tiếp như function call bình thường
- Prefect decorator (@flow, @task) chỉ log, không có queue
- Fail → exception → exit

**Ví dụ:**
```
[10:00] Train SFT starts...
[12:00] Train SFT done ✅
[12:01] Deploy vLLM...
[12:05] Eval SFT...
[12:10] Eval fail ❌ (network timeout)
[12:10] Pipeline crashes
```

→ Muốn retry? Chạy lại `python pipeline.py` → **Train lại từ đầu (2h)**

---

### 2. Flow Run với Server (có UI nhưng vẫn restart từ đầu)

```bash
# Terminal 1
prefect server start

# Terminal 2
export PREFECT_API_URL="http://127.0.0.1:4200/api"
python pipeline.py
```

**Cách hoạt động:**
- Giống Simple Run nhưng log lên server
- Thấy UI, graph, logs
- **NHƯNG**: Fail → vẫn phải chạy lại từ đầu

**Ví dụ:**
```
[10:00] Train SFT starts...
[12:00] Train SFT done ✅
[12:01] Deploy vLLM...
[12:05] Eval SFT...
[12:10] Eval fail ❌ (network timeout)
[12:10] Pipeline crashes
```

→ Trong UI: thấy logs, biết fail ở đâu  
→ Nhưng muốn retry? Vẫn phải chạy lại `python pipeline.py` → **Train lại từ đầu**

---

### 3. Deployment + Worker (Production 🌟)

```bash
# Terminal 1: Server
prefect server start

# Terminal 2: Setup
prefect work-pool create --type process default
prefect deploy --all

# Terminal 3: Worker
prefect worker start --pool default

# Terminal 4: Trigger
prefect deployment run qwen3_vl_pipeline/local
```

**Cách hoạt động:**
- Pipeline code → package → upload to server
- Worker pull job từ queue
- Prefect **track state** của từng task
- Fail → state saved, có thể retry individual task

**Ví dụ:**
```
[10:00] Train SFT starts...
[12:00] Train SFT done ✅ (state saved)
[12:01] Deploy vLLM... ✅ (state saved)
[12:05] Eval SFT...
[12:10] Eval fail ❌ (network timeout, state saved)
```

→ Trong UI: Click "Retry" button  
→ Prefect check saved state:  
  - Train SFT = completed ✅ → skip  
  - Deploy vLLM = completed ✅ → skip  
  - Eval SFT = failed → **chỉ retry task này**

```
[12:15] Retry Eval SFT...
[12:20] Eval SFT done ✅
[12:21] Continue: Gate Check → Train GRPO → ...
```

**→ Tiết kiệm 2 giờ training!**

---

## 📊 So sánh chi tiết

### Scenario: Pipeline fail ở GRPO (sau 5 giờ)

```
Train SFT (2h) → Deploy → Eval → Gate → Train GRPO (3h) → FAIL (OOM)
```

#### Mode 1: Simple Run
```bash
python pipeline.py
# → Fail at 5h mark
# Fix config (giảm batch size)
python pipeline.py
# → Train SFT lại từ đầu (2h wasted!)
```

#### Mode 2: Flow Run + Server
```bash
export PREFECT_API_URL="http://127.0.0.1:4200/api"
python pipeline.py
# → Fail at 5h mark
# Fix config
python pipeline.py
# → Train SFT lại từ đầu (2h wasted!)
# Chỉ khác: có UI đẹp để xem, nhưng vẫn phải restart từ đầu
```

#### Mode 3: Deployment + Worker ⭐
```bash
prefect deployment run qwen3_vl_pipeline/local
# → Fail at 5h mark
# Fix config
# UI → Click "Restart" button
# → Skip Train SFT (completed), chỉ chạy Train GRPO
# → Save 2 hours!
```

---

## 🎯 Recommendations

### Development/Testing
→ **Simple Run**: `python pipeline.py`  
- Fast iteration  
- No overhead  
- OK to restart từ đầu

### Production
→ **Deployment + Worker**  
- Long-running pipelines (SFT + GRPO = 5+ hours)  
- Expensive operations (training costs GPU time + money)  
- Need reliability (retry network errors, resume from failures)  
- Schedule chạy định kỳ

---

## 🚀 Quick Setup cho Production

**Automated setup:**
```bash
# Terminal 1
prefect server start

# Terminal 2  
./setup-prefect.sh  # Auto create pool + deploy
# Windows: .\setup-prefect.ps1

# Terminal 3
prefect worker start --pool default

# Trigger
prefect deployment run qwen3_vl_pipeline/local
```

**Manual setup:**
```bash
# Terminal 1: Server
prefect server start

# Terminal 2: Deploy
prefect work-pool create --type process default
prefect deploy --all

# Terminal 3: Worker
prefect worker start --pool default

# Terminal 4: Trigger
prefect deployment run qwen3_vl_pipeline/local
```

---

## 💡 Pro Tips

### 1. Keep server + worker running

```bash
# Use tmux/screen để giữ terminals
tmux new -s prefect
# Terminal 1: prefect server start
# Terminal 2: prefect worker start --pool default
# Detach: Ctrl+B, D
```

### 2. Auto-restart worker (systemd)

Create `/etc/systemd/system/prefect-worker.service`:
```ini
[Unit]
Description=Prefect Worker
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/workspace/pipeline
Environment="PREFECT_API_URL=http://127.0.0.1:4200/api"
ExecStart=/usr/bin/prefect worker start --pool default
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable prefect-worker
sudo systemctl start prefect-worker
```

### 3. Schedule daily runs

```bash
# Chạy mỗi ngày lúc 2AM
prefect deployment set-schedule qwen3_vl_pipeline/local \
  --cron "0 2 * * *"
```

### 4. Monitor logs

```bash
# Worker logs
tail -f ~/.prefect/logs/*.log

# Pipeline artifacts
tail -f runs/*/commands.log
```

---

## 🔍 Debugging

### Worker không pick up jobs

```bash
# Check work pool
prefect work-pool ls
prefect work-pool inspect default

# Check deployments
prefect deployment ls

# Check worker
prefect worker ls
```

### Restart flow từ failed task

1. UI → Flow Runs → Click failed run
2. Click **"Restart"** button
3. Prefect sẽ:
   - Load saved state
   - Skip completed tasks
   - Chỉ chạy failed task onwards

### Clear failed runs

```bash
# Clear old failed runs
prefect flow-run delete <run-id>
```

Done! 🎉
