# qwen3-vl/utils.py
import json
from typing import Dict
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

        # 2. Xử lý Multi-turn Data
        text_data = sample["text"]
        raw_data = json.loads(text_data) if isinstance(text_data, str) else text_data
        
        # --- BƯỚC 2.1: NORMALIZATION (QUAN TRỌNG) ---
        # Chuyển mọi format (User/Assistant pair, ShareGPT, v.v.) về chuẩn OpenAI
        conversation = []
        
        for item in raw_data:
            # Case A: Format cặp "user" & "assistant" (như lỗi bạn gặp)
            if "user" in item and "assistant" in item:
                conversation.append({"role": "user", "content": item["user"]})
                conversation.append({"role": "assistant", "content": item["assistant"]})
            
            # Case B: Format chuẩn "role" & "content" hoặc ShareGPT "from" & "value"
            elif "role" in item or "from" in item:
                role = item.get("role", item.get("from", "")).lower()
                content = item.get("content", item.get("value", ""))
                
                # Map tên role lạ về chuẩn
                if role == "human": role = "user"
                if role == "gpt": role = "assistant"
                
                conversation.append({"role": role, "content": content})
            
            else:
                # Nếu không thuộc format nào -> Data rác thực sự -> Crash để báo
                raise ValueError(f"Unknown conversation format keys: {list(item.keys())}")

        # --- BƯỚC 2.2: CHÈN ẢNH VÀO TURN USER ĐẦU TIÊN ---
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
                    # Tránh duplicate ảnh nếu chạy lại nhiều lần
                    if not any(x.get("type") == "image" for x in current_content):
                        message["content"].append({"type": "image", "image": image})
                
                image_added = True
                break
        
        if not image_added:
            raise ValueError("Sample is multiturn but has no 'user' role to insert image after normalization.")

        return {"messages": conversation}

    except Exception as e:
        # Giữ nguyên phần Debug in lỗi cực mạnh này, nó rất hữu ích
        print("\n" + "="*60)
        print(f"❌ FATAL ERROR DETECTED: {e}")
        print(f"❌ Source File: {sample.get('source_file', 'Unknown')}")
        print("-" * 20 + " BAD SAMPLE CONTENT " + "-" * 20)
        
        debug_sample = sample.copy()
        debug_sample["image"] = f"<Image Object: {sample['image'].size if hasattr(sample['image'], 'size') else 'Unknown'}>"
        print(json.dumps(debug_sample, indent=4, ensure_ascii=False, default=str))
        print("="*60 + "\n")
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