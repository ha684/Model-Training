# Real Example: Tại sao cần Deployment?

## 📖 Scenario thực tế

Bạn chạy pipeline lúc 10:00 sáng, training SFT + GRPO mất tổng 5 giờ.

---

## ❌ Không dùng Deployment

```bash
python pipeline.py
```

**Timeline:**
```
10:00 - Start
10:00 - Train SFT begins...
12:00 - Train SFT complete ✅
12:01 - Push to Hub...
12:05 - Deploy vLLM SFT...
12:10 - Eval SFT...
12:30 - Gate check ✅
12:31 - Train GRPO begins...
15:31 - Train GRPO complete ✅
15:32 - Push to Hub...
15:36 - Deploy vLLM GRPO...
15:41 - Eval GRPO...
16:01 - Eval GRPO ❌ FAIL (network timeout)
16:01 - Pipeline crashes
```

**Bạn fix network issue và chạy lại:**

```bash
python pipeline.py  # 16:05
```

**Timeline (lần 2):**
```
16:05 - Train SFT begins AGAIN... 😭
18:05 - Train SFT complete ✅ (DUP, đã train lúc 10:00 rồi!)
18:06 - Push to Hub...
18:10 - Deploy vLLM SFT...
18:15 - Eval SFT...
18:35 - Gate check ✅
18:36 - Train GRPO begins AGAIN... 😭
21:36 - Train GRPO complete ✅ (DUP, đã train lúc 12:31 rồi!)
21:37 - Push to Hub...
21:41 - Deploy vLLM GRPO...
21:46 - Eval GRPO...
22:06 - Eval GRPO ✅ (finally!)
22:07 - Promote ✅
22:08 - Done
```

**Result:**
- First run: 6 giờ (10:00 → 16:01)
- Second run: 6 giờ (16:05 → 22:08)
- **Total: 12 giờ**
- **Wasted: 5 giờ training lặp lại!**

---

## ✅ Dùng Deployment + Worker

**Setup (one-time):**
```bash
# Terminal 1
prefect server start

# Terminal 2
./setup-prefect.sh

# Terminal 3
prefect worker start --pool default
```

**Trigger run:**
```bash
prefect deployment run qwen3_vl_pipeline/local  # 10:00
```

**Timeline:**
```
10:00 - Worker picks up job
10:00 - Train SFT begins...
12:00 - Train SFT complete ✅ → STATE SAVED
12:01 - Push to Hub... ✅ → STATE SAVED
12:05 - Deploy vLLM SFT... ✅ → STATE SAVED
12:10 - Eval SFT... ✅ → STATE SAVED
12:30 - Gate check ✅ → STATE SAVED
12:31 - Train GRPO begins...
15:31 - Train GRPO complete ✅ → STATE SAVED
15:32 - Push to Hub... ✅ → STATE SAVED
15:36 - Deploy vLLM GRPO... ✅ → STATE SAVED
15:41 - Eval GRPO...
16:01 - Eval GRPO ❌ FAIL (network timeout) → STATE SAVED
```

**Bạn vào UI và click "Retry":**

```
16:05 - Prefect loads saved state:
        - Train SFT: completed ✅ → SKIP
        - Push Hub SFT: completed ✅ → SKIP
        - Deploy vLLM SFT: completed ✅ → SKIP
        - Eval SFT: completed ✅ → SKIP
        - Gate SFT: completed ✅ → SKIP
        - Train GRPO: completed ✅ → SKIP
        - Push Hub GRPO: completed ✅ → SKIP
        - Deploy vLLM GRPO: completed ✅ → SKIP
        - Eval GRPO: failed → RETRY THIS ONLY
16:05 - Deploy vLLM GRPO (re-deploy)...
16:10 - Eval GRPO...
16:30 - Eval GRPO ✅
16:31 - Promote ✅
16:32 - Done
```

**Result:**
- First run: 6 giờ (10:00 → 16:01)
- Retry: 30 phút (16:05 → 16:32)
- **Total: 6.5 giờ**
- **Saved: 5.5 giờ!**

---

## 💰 Cost Savings

Training on 4x A100 GPUs:
- **Simple run:** 12 giờ × $4/GPU/h × 4 GPUs = **$192**
- **Deployment:** 6.5 giờ × $4/GPU/h × 4 GPUs = **$104**
- **Savings: $88** cho 1 lần retry! 💸

Với pipeline chạy hàng tuần:
- Probability of failure: ~20% (network, OOM, etc)
- Average saves/week: $88 × 0.2 = $17.6
- **Saves ~$900/year** just from retry!

---

## 🎯 Conclusion

**Cho pipeline của bạn:**
- SFT: 2h
- GRPO: 3h
- Total: 5h+ per run

→ **PHẢI DÙNG DEPLOYMENT!**

Không dùng deployment = mất thời gian + tiền bạc mỗi khi có failure.

---

## 🚀 Quick Setup Commands

```bash
# Terminal 1: Server (giữ chạy mãi)
prefect server start

# Terminal 2: Deploy (one-time, hoặc khi update code)
./setup-prefect.sh

# Terminal 3: Worker (giữ chạy mãi)
prefect worker start --pool default

# Terminal 4: Trigger (mỗi khi muốn chạy)
prefect deployment run qwen3_vl_pipeline/local

# Monitor: http://127.0.0.1:4200
```

Done! 🎉
