import os
import sys
from pathlib import Path
from io import BytesIO
import base64

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

CURRENT_DIR = os.getcwd()
LOCAL_CACHE_DIR = os.path.join(CURRENT_DIR, "local_hf_cache")

if not os.path.exists(LOCAL_CACHE_DIR):
    os.makedirs(LOCAL_CACHE_DIR)
    print(f"📁 Created local cache directory: {LOCAL_CACHE_DIR}")

os.environ["HF_HOME"] = LOCAL_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(LOCAL_CACHE_DIR, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(LOCAL_CACHE_DIR, "hub")
HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN

import torch
import json
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Import utils functions
from qwen3_vl.utils import detect_language, get_system_and_user_prompt


MODEL_ID = "ha684/qwen3-vl-2b-22-01-26"
DATASET_ID = "ha684/vl-vi-en-ja-conv-mix-v1.3"
OUTPUT_DIR = "vlm-w8a8-int8"
REPO_ID = "ha684/qwen3-vl-2b-22-01-26-w8a8-int8"
MAX_SEQ_LEN = 16384
NUM_CALIB = 256


def main():
    print(f"🚀 Loading model: {MODEL_ID}")
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        cache_dir=LOCAL_CACHE_DIR
    )
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        dtype="auto",
        trust_remote_code=True,
        cache_dir=LOCAL_CACHE_DIR,
    )
    print(f"Tổng số visual blocks: {len(model.visual.blocks)}")
    print(f"✅ Model loaded on device: {model.device}")

    def preprocess_and_tokenize(example):
        """Preprocess and tokenize - logic giống convert_to_conversation nhưng dùng base64"""
        try:
            image = example["image"]
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue())
            encoded_image_text = encoded_image.decode("utf-8")
            base64_image = f"data:image;base64,{encoded_image_text}"
            
            multiturn = example.get("is_multiturn", False)
            
            if not multiturn:
                text_content = example.get("text", "")
                language = detect_language(text_content)
                system_prompt, user_prompt = get_system_and_user_prompt(
                    language, sample=example
                )
                
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": base64_image},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": example["text"]}],
                    },
                ]
            else:
                text_data = example["text"]
                raw_data = json.loads(text_data) if isinstance(text_data, str) else text_data
                messages = []
                
                for item in raw_data:
                    if "user" in item and "assistant" in item:
                        messages.append({"role": "user", "content": item["user"]})
                        messages.append({"role": "assistant", "content": item["assistant"]})
                    elif "role" in item or "from" in item:
                        role = item.get("role", item.get("from", "")).lower()
                        content = item.get("content", item.get("value", ""))
                        if role == "human":
                            role = "user"
                        if role == "gpt":
                            role = "assistant"
                        messages.append({"role": role, "content": content})
                    else:
                        raise ValueError(f"Unknown conversation format keys: {list(item.keys())}")
                
                image_added = False
                for message in messages:
                    if message["role"] == "user":
                        current_content = message["content"]
                        if isinstance(current_content, str):
                            message["content"] = [
                                {"type": "text", "text": current_content},
                                {"type": "image", "image": base64_image},
                            ]
                        elif isinstance(current_content, list):
                            if not any(x.get("type") == "image" for x in current_content):
                                message["content"].append({"type": "image", "image": base64_image})
                        image_added = True
                        break
                
                if not image_added:
                    raise ValueError("Multiturn sample has no 'user' role to insert image")
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            return processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                max_length=MAX_SEQ_LEN,
                truncation=True,
            )
        except Exception as e:
            print(f"Error processing example: {e}")
            return {}

    print("⏳ Loading dataset...")
    local_cache = os.environ["HF_DATASETS_CACHE"]
    
    ds = load_dataset(
        DATASET_ID, 
        split=f"train[:{NUM_CALIB}]",
        cache_dir=local_cache
    )
    
    print(f"📊 Processing {len(ds)} samples...")
    calib_ds = ds.map(
        preprocess_and_tokenize, 
        remove_columns=ds.column_names,
        desc="Tokenizing",
        num_proc=4,
        batched=False
    )
    
    print(f"✅ Using {len(calib_ds)} calibration samples")

    def data_collator(batch):
        assert len(batch) == 1
        return {
            key: (
                torch.tensor(value)
                if key != "pixel_values"
                else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
            )
            for key, value in batch[0].items()
        }

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=[
            "re:.*lm_head",
            "re:visual.*",
            "re:model.visual.*",
            "re:.*mlp.gate$",
        ],
    )

    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_CALIB,
        data_collator=data_collator,
    )

    print(f"✅ Saving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR, save_compressed=True)
    processor.save_pretrained(OUTPUT_DIR)

    if REPO_ID:
        print(f"☁️  Pushing to Hub: {REPO_ID}")
        create_repo(REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)
        upload_folder(repo_id=REPO_ID, folder_path=OUTPUT_DIR, token=HF_TOKEN)
        print("🎉 Upload Done!")


if __name__ == "__main__":
    main()