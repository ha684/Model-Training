"""
This script is used to analyze the token length distribution of the dataset.

python qwen3_vl/analyze_token_length.py
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from config.fields_config import get_extraction_prompt
from PIL import Image
from io import BytesIO

# ==========================================
# Configuration Variables
# ==========================================
DATASET_NAME = "ha684/senko-V1.2"
MODEL_NAME = "unsloth/Qwen3-VL-2B-Instruct"
HF_TOKEN = "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"
SPLIT = "train"
SAMPLE_LIMIT = 500
MAX_PIXELS = 12845056


def convert_to_conversation(sample, instruction):
    try:
        multiturn = sample.get("is_multiturn", False)
        image = sample["image"]
        
        # HuggingFace dataset image format: dict with 'bytes' and 'path'
        if isinstance(image, dict):
            if 'bytes' in image and image['bytes']:
                # Load từ bytes
                image = Image.open(BytesIO(image['bytes']))
            elif 'path' in image and image['path']:
                # Load từ path
                image = Image.open(image['path'])
        elif isinstance(image, str):
            # Nếu là string path
            image = Image.open(image)
        # Nếu đã là PIL Image thì giữ nguyên

        if not multiturn:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "image": image},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["text"]}],
                },
            ]
            return {"messages": conversation}

        text_data = sample["text"]
        raw_data = json.loads(text_data) if isinstance(text_data, str) else text_data

        conversation = []
        for item in raw_data:
            if "user" in item and "assistant" in item:
                conversation.append({"role": "user", "content": item["user"]})
                conversation.append({"role": "assistant", "content": item["assistant"]})
            elif "role" in item or "from" in item:
                role = item.get("role", item.get("from", "")).lower()
                content = item.get("content", item.get("value", ""))
                if role == "human": role = "user"
                if role == "gpt": role = "assistant"
                conversation.append({"role": role, "content": content})
            else:
                continue

        image_added = False
        for message in conversation:
            if message["role"] == "user":
                current_content = message["content"]
                if isinstance(current_content, str):
                    message["content"] = [
                        {"type": "text", "text": current_content},
                        {"type": "image", "image": image}
                    ]
                elif isinstance(current_content, list):
                    if not any(x.get("type") == "image" for x in current_content):
                        message["content"].append({"type": "image", "image": image})
                image_added = True
                break
        
        if not image_added:
             # Fallback nếu không tìm thấy chỗ nhét ảnh, nhét vào message user đầu tiên
             if len(conversation) > 0 and conversation[0]['role'] == 'user':
                 conversation[0]['content'] = [
                     {"type": "text", "text": str(conversation[0]['content'])},
                     {"type": "image", "image": image}
                 ]

        return {"messages": conversation}
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None


def main():
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN

    print(f"⏳ Loading Processor: {MODEL_NAME}...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading processor: {e}")
        return

    print(f"📊 Loading Dataset: {DATASET_NAME} ({SPLIT})...")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    
    if SAMPLE_LIMIT > 0:
        dataset = dataset.select(range(min(len(dataset), SAMPLE_LIMIT)))
        print(f"⚠️ Limiting analysis to first {len(dataset)} samples.")

    instruction_prompt = "Describe this image." 
    try:
        instruction_prompt = get_extraction_prompt()
    except ImportError:
        print("⚠️ Could not import get_extraction_prompt. Using default prompt.")

    token_lengths = []
    
    print("🚀 Starting analysis (This involves image processing, so it might take a moment)...")
    
    for sample in tqdm(dataset):
        chat_dict = convert_to_conversation(sample, instruction_prompt)
        if not chat_dict: 
            continue
        
        messages = chat_dict["messages"]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        seq_len = inputs.input_ids.shape[1]
        token_lengths.append(seq_len)

    # ==========================================
    # STATS & PLOTTING
    # ==========================================
    token_lengths = np.array(token_lengths)
    
    mean_len = np.mean(token_lengths)
    median_len = np.median(token_lengths)
    p90 = np.percentile(token_lengths, 90)
    p95 = np.percentile(token_lengths, 95)
    p99 = np.percentile(token_lengths, 99)
    max_len = np.max(token_lengths)

    print("\n" + "="*40)
    print("📈 TOKEN LENGTH STATISTICS")
    print("="*40)
    print(f"Count:  {len(token_lengths)}")
    print(f"Min:    {np.min(token_lengths)}")
    print(f"Mean:   {mean_len:.2f}")
    print(f"Median: {median_len:.2f}")
    print(f"P90:    {p90:.2f}")
    print(f"P95:    {p95:.2f}  <-- Recommended candidates")
    print(f"P99:    {p99:.2f}  <-- Recommended candidates")
    print(f"Max:    {max_len}")
    print("="*40)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.histplot(token_lengths, kde=True, bins=50, color='skyblue', edgecolor='black')
    
    plt.axvline(p95, color='orange', linestyle='--', label=f'P95: {int(p95)}')
    plt.axvline(p99, color='red', linestyle='--', label=f'P99: {int(p99)}')
    plt.axvline(8192, color='green', linestyle='-', label='Current Max (8192)')

    plt.title(f'Token Length Distribution (Text + Images)\nDataset: {DATASET_NAME}')
    plt.xlabel('Sequence Length (Tokens)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    output_file = "token_distribution.png"
    plt.savefig(output_file)
    print(f"\n✅ Plot saved to {output_file}")
    
    # Recommendation logic
    print("\n💡 RECOMMENDATION:")
    if p99 < 8192:
        print(f"You can safely REDUCE --max_seq_length to {int(p99)} or {int(p95)} to save VRAM and speed up training.")
    else:
        print(f"⚠️ Your data exceeds 8192 tokens. Consider INCREASING --max_seq_length to {int(p99)} or truncating data.")


if __name__ == "__main__":
    main()