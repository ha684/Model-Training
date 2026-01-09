# analyze_token_length.py
"""
This script is used to analyze the token length distribution of the dataset.

python qwen3_vl/analyze_token_length.py --dataset_name "ha684/senkoV1" --model_name "unsloth/Qwen3-VL-2B-Instruct" --hf_token "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" --sample_limit 100
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info  # Thư viện chuẩn thường dùng với Qwen-VL
from config.fields_config import get_extraction_prompt

# ==========================================
# 1. UTILS (Copy từ code của bạn để đảm bảo tính nhất quán)
# ==========================================
def convert_to_conversation(sample, instruction):
    try:
        multiturn = sample.get("is_multiturn", False)
        image = sample["image"]

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

# ==========================================
# 2. MAIN ANALYSIS LOGIC
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Token Length Distribution for Qwen3-VL")
    parser.add_argument("--dataset_name", type=str, default="ha684/senko1", help="HuggingFace dataset name")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model path/name for tokenizer") 
    # Lưu ý: Qwen3-VL chưa official rộng rãi, nếu bạn dùng checkpoint local thì trỏ path vào, 
    # nếu không hãy dùng Qwen2.5-VL hoặc Qwen2-VL vì tokenization logic cho ảnh tương tự.
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample_limit", type=int, default=-1, help="Limit number of samples to analyze (for speed)")
    parser.add_argument("--max_pixels", type=int, default=12845056, help="Max pixels for Qwen image processor (affects token count)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    print(f"⏳ Loading Processor: {args.model_name}...")
    try:
        # Load Processor (bao gồm Tokenizer và Image Processor)
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading processor: {e}")
        return

    print(f"📊 Loading Dataset: {args.dataset_name} ({args.split})...")
    dataset = load_dataset(args.dataset_name, split=args.split)
    
    if args.sample_limit > 0:
        dataset = dataset.select(range(min(len(dataset), args.sample_limit)))
        print(f"⚠️ Limiting analysis to first {len(dataset)} samples.")

    # Import prompt instruction từ file config của bạn nếu có, hoặc dùng default
    # Ở đây tôi hardcode giả lập để script chạy độc lập
    instruction_prompt = "Describe this image." 
    try:
        instruction_prompt = get_extraction_prompt()
    except ImportError:
        print("⚠️ Could not import get_extraction_prompt. Using default prompt.")

    token_lengths = []
    
    print("🚀 Starting analysis (This involves image processing, so it might take a moment)...")
    
    for sample in tqdm(dataset):
        # 1. Convert format
        chat_dict = convert_to_conversation(sample, instruction_prompt)
        if not chat_dict: continue
        
        messages = chat_dict["messages"]

        # 2. Prepare text string using chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 3. Extract Vision Info (Images/Videos)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 4. Tokenize & Process Image
        # Qwen-VL xử lý ảnh dynamic resolution nên số token phụ thuộc vào kích thước ảnh thật
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 5. Get Length
        seq_len = inputs.input_ids.shape[1]
        token_lengths.append(seq_len)

    # ==========================================
    # 3. STATS & PLOTTING
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

    plt.title(f'Token Length Distribution (Text + Images)\nDataset: {args.dataset_name}')
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