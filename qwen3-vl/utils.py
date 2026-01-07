# qwen3-vl/utils.py
import json
from typing import Dict
from torch.utils.data import Dataset as TorchDataset

def convert_to_conversation(sample: Dict, instruction: str) -> Dict:
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

    try:
        text_data = sample["text"]
        if isinstance(text_data, str):
            conversation = json.loads(text_data)
        else:
            conversation = text_data

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
                    message["content"].append({"type": "image", "image": image})
                
                image_added = True
                break
        
        if not image_added:
            conversation.insert(0, {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image."}]
            })

        return {"messages": conversation}

    except Exception as e:
        print(f"Error processing multi-turn sample: {e}")
        return {"messages": []}


class LazyVisionDataset(TorchDataset):
    def __init__(self, hf_dataset, instruction: str):
        self.dataset = hf_dataset
        self.instruction = instruction
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return convert_to_conversation(sample, self.instruction)