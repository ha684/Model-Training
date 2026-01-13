# merge_and_push.py
from unsloth import FastVisionModel
import os
# 1. Cấu hình
LORA_MODEL_ID = (
    "ha684/qwen3-vl-2b-lora-finetuned"  # Tên repo LoRA bạn vừa push ở Phase 1
)
NEW_MODEL_ID = "ha684/qwen3-vl-2b-1.0"  # Tên repo mới cho bản Full
HF_TOKEN = "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"  # Token của bạn (hoặc lấy từ os.environ)
os.environ["HF_TOKEN"] = HF_TOKEN
# 2. Load lại Model & Adapter
# Lưu ý: Load ở chế độ 16bit (không phải 4bit) để merge chất lượng tốt nhất
print("⏳ Loading LoRA and Base model...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=LORA_MODEL_ID,
    load_in_4bit=False,  # QUAN TRỌNG: Để False để merge sạch
    load_in_16bit=True,
)   

# 3. Merge & Push (Chọn 1 trong các cách dưới đây)

# --- CÁCH A: Merge ra 16bit (Dùng cho vLLM, SGLang) ---
print(f"🚀 Pushing Merged 16bit model to {NEW_MODEL_ID}...")
model.push_to_hub_merged(
    NEW_MODEL_ID, tokenizer, save_method="merged_16bit", token=HF_TOKEN
)

# --- CÁCH B: Merge ra GGUF (Dùng cho Ollama/Llama.cpp) ---
# GGUF_MODEL_ID = "ha684/qwen2.5-vl-3b-gguf"
# print(f"🚀 Pushing GGUF model to {GGUF_MODEL_ID}...")
# model.push_to_hub_gguf(
#     GGUF_MODEL_ID,
#     tokenizer,
#     quantization_method = "q4_k_m", # q4_k_m, q8_0, f16
#     token = HF_TOKEN
# )

print("✅ DONE! Check your Hugging Face Hub.")
