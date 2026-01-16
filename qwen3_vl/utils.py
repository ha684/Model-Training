# qwen3-vl/utils.py
import json
from typing import Dict

import jiwer
import numpy as np
from torch.utils.data import Dataset as TorchDataset


REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

def convert_to_conversation(sample: Dict, instruction: str) -> Dict:
    try:
        multiturn = sample["is_multiturn"]
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

class LazyVisionGRPODataset(TorchDataset):
    def __init__(self, hf_dataset, instruction: str, processor):
        """
        hf_dataset: Dataset gốc từ HuggingFace (chưa map/load ảnh vào RAM)
        instruction: Câu lệnh mặc định (ví dụ: "Solve this problem.")
        processor: Processor của Qwen2-VL (để dùng apply_chat_template)
        """
        self.dataset = hf_dataset
        self.instruction = instruction
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # 1. Xử lý ảnh (Load ảnh tại thời điểm gọi để tiết kiệm RAM)
        # Giả sử key ảnh trong dataset của bạn là "image"
        image = sample["image"]
        
        # Convert sang RGB nếu cần (tránh lỗi kênh Alpha)
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
            
        # (Tùy chọn) Resize ảnh nếu quá lớn để tiết kiệm VRAM khi training
        # if hasattr(image, "resize"):
        #     image = image.resize((512, 512)) 

        # 2. Xây dựng nội dung Prompt (User Prompt)
        # Chúng ta chèn yêu cầu Reasoning vào đây
        # Logic: Câu hỏi gốc + "Hãy suy nghĩ trong thẻ..."
        
        # Lấy câu hỏi từ data (giả sử key là 'text' hoặc 'question')
        question_text = sample.get("text") or sample.get("question", "")
        
        # Tạo câu lệnh đầy đủ cho GRPO
        user_content_text = (
            f"{self.instruction}\n\nProblem: {question_text}\n\n"
            f"First provide your reasoning or working out on how you would go about solving the question "
            f"between {REASONING_START} and {REASONING_END}. "
            f"Then output your final answer between {SOLUTION_START} and {SOLUTION_END}."
        )

        # 3. Tạo list messages theo chuẩn Qwen-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image}, # Truyền object ảnh PIL
                    {"type": "text", "text": user_content_text},
                ],
            }
        ]

        # 4. Dùng Processor để tạo thành chuỗi Prompt text
        # Lưu ý: add_generation_prompt=True để thêm token của Assistant vào cuối
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 5. Lấy Ground Truth (Đáp án đúng) để tính điểm thưởng (Reward)
        # Giả sử key đáp án trong data là 'answer' hoặc 'text' (phần assistant cũ)
        # Bạn cần trích xuất đúng cái "đáp án cuối cùng" để so khớp
        ground_truth = sample.get("answer") or sample.get("solution") or sample.get("text", "")
        
        # Nếu dataset cũ của bạn lưu cả đoạn hội thoại trong 'text', bạn cần parse để lấy đáp án đúng.
        # Ở đây tôi giả định bạn có cột chứa đáp án thô.

        return {
            "prompt": prompt_text,   # GRPOTrainer cần key này
            "image": image,          # Để Collator xử lý pixel_values
            "ground_truth": ground_truth # Để Reward function so sánh (không train trực tiếp cái này)
        }
        
def compute_metrics_cer(eval_preds, tokenizer):
    """
    Calculate Character Error Rate (CER) for OCR evaluation.
    Note: This is Teacher-Forced CER (based on next-token prediction),
    which helps evaluate quickly without running generation (very slow).
    """
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    pred_ids = np.argmax(preds, axis=-1)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    cer = jiwer.cer(decoded_labels, decoded_preds)

    return {"cer": cer}
