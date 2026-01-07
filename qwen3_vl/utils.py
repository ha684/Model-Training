# qwen3-vl/utils.py
import json
from typing import Dict
from torch.utils.data import Dataset as TorchDataset

def convert_to_conversation(sample: Dict, instruction: str) -> Dict:
    try:
        multiturn = sample["is_multiturn"]
        image = sample["image"]

        # 1. Production Data (Single turn + Instruction)
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

        # 2. Multi-turn Data processing
        text_data = sample["text"]
        if isinstance(text_data, str):
            conversation = json.loads(text_data)
        else:
            conversation = text_data

        image_added = False
        
        for message in conversation:
            # --- KIỂM TRA LỖI LOGIC Ở ĐÂY ---
            # Nếu data thiếu key "role", code sẽ crash ngay tại dòng dưới
            # và nhảy vào block except để in log.
            if message["role"] == "user":
                current_content = message["content"]

                if isinstance(current_content, str):
                    message["content"] = [
                        {"type": "text", "text": current_content},
                        {"type": "image", "image": image}
                    ]
                elif isinstance(current_content, list):
                    message["content"].append({"type": "image", "image": image})
                
                image_added = True
                break
        
        if not image_added:
            # Nếu chạy hết vòng lặp mà không add được ảnh (không có role user?)
            # Thì đây cũng là data lỗi -> Raise lỗi luôn
            raise ValueError("Sample is multiturn but has no 'user' role to insert image.")

        return {"messages": conversation}

    except Exception as e:
        # --- DEBUG KHU VỰC CRASH ---
        print("\n" + "="*60)
        print(f"❌ FATAL ERROR DETECTED: {e}")
        print(f"❌ Source File: {sample.get('source_file', 'Unknown')}")
        print("-" * 20 + " BAD SAMPLE CONTENT " + "-" * 20)
        
        # In sample ra (loại bỏ object ảnh để print cho gọn text)
        debug_sample = sample.copy()
        debug_sample["image"] = f"<Image Object: {sample['image'].size if hasattr(sample['image'], 'size') else 'Unknown'}>"
        
        # Dùng json.dumps để in format đẹp, dễ nhìn structure lỗi
        print(json.dumps(debug_sample, indent=4, ensure_ascii=False, default=str))
        print("="*60 + "\n")
        
        # Raise lỗi để dừng training ngay lập tức
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