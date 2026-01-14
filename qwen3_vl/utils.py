# qwen3-vl/utils.py
import json
from typing import Dict

import jiwer
import numpy as np
from torch.utils.data import Dataset as TorchDataset


def convert_to_conversation(sample: Dict, instruction: str) -> Dict:
    try:
        multiturn = sample["is_multiturn"]
        image = sample["image"]

        # 1. Xử lý Production Data (Single turn)
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

                if role == "human":
                    role = "user"
                if role == "gpt":
                    role = "assistant"

                conversation.append({"role": role, "content": content})

            else:
                raise ValueError(
                    f"Unknown conversation format keys: {list(item.keys())}"
                )

        image_added = False

        for message in conversation:
            if message["role"] == "user":
                current_content = message["content"]

                if isinstance(current_content, str):
                    message["content"] = [
                        {"type": "text", "text": current_content},
                        {"type": "image", "image": image},
                    ]
                elif isinstance(current_content, list):
                    if not any(x.get("type") == "image" for x in current_content):
                        message["content"].append({"type": "image", "image": image})

                image_added = True
                break

        if not image_added:
            raise ValueError(
                "Sample is multiturn but has no 'user' role to insert image after normalization."
            )

        return {"messages": conversation}

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ FATAL ERROR DETECTED: {e}")
        print(f"❌ Source File: {sample.get('source_file', 'Unknown')}")
        print("-" * 20 + " BAD SAMPLE CONTENT " + "-" * 20)

        debug_sample = sample.copy()
        debug_sample["image"] = (
            f"<Image Object: {sample['image'].size if hasattr(sample['image'], 'size') else 'Unknown'}>"
        )
        print(json.dumps(debug_sample, indent=4, ensure_ascii=False, default=str))
        print("=" * 60 + "\n")
        raise e


class LazyVisionDataset(TorchDataset):
    def __init__(self, hf_dataset, instruction: str):
        self.dataset = hf_dataset
        self.instruction = instruction

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return convert_to_conversation(sample, self.instruction)


def compute_metrics_cer(eval_preds, tokenizer):
    """
    Tính toán Character Error Rate (CER) cho việc đánh giá OCR.
    Lưu ý: Đây là Teacher-Forced CER (tính trên next-token prediction),
    giúp đánh giá nhanh mà không cần chạy generation (rất chậm).
    """
    preds, labels = eval_preds

    # Preds thường là logits, cần lấy argmax để ra token id
    if isinstance(preds, tuple):
        preds = preds[0]

    # Lấy token có xác suất cao nhất
    pred_ids = np.argmax(preds, axis=-1)

    # Thay thế -100 trong labels bằng pad_token_id để tokenizer không lỗi
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode batch thành list string
    # skip_special_tokens=True để bỏ qua <|endoftext|>, padding, v.v.
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Tính CER
    cer = jiwer.cer(decoded_labels, decoded_preds)

    # Trả về dict metrics (tên phải khớp để config nhận diện)
    return {"cer": cer}
