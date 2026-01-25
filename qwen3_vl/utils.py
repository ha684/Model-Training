import json
import os
from typing import Dict, List, Tuple

from config import (
    FIELD_DEFINITIONS,
    FieldDefinition,
    generate_extraction_prompt,
)
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoProcessor


def detect_language(data: Dict | str) -> str:
    """
    Detect language from JSON keys (not values).
    Returns: 'vi' for Vietnamese, 'ja' for Japanese, 'en' for English

    Args:
        data: Either a dictionary or a JSON string that will be parsed
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            raise

    if not isinstance(data, dict):
        return "en"

    all_keys = []
    metadata = data.get("metadata", [])
    if metadata:
        if isinstance(metadata, list) and len(metadata) > 0:
            item = metadata[0]
            if isinstance(item, dict):
                all_keys.extend(item.keys())
        elif isinstance(metadata, dict):
            all_keys.extend(metadata.keys())

    table_rows = data.get("table_rows", [])
    if table_rows and isinstance(table_rows, list) and len(table_rows) > 0:
        item = table_rows[0]
        if isinstance(item, dict):
            all_keys.extend(item.keys())

    all_keys_str = " ".join(all_keys)
    japanese_chars = sum(
        1
        for c in all_keys_str
        if "\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff"
    )
    if japanese_chars > 0:
        return "ja"

    vietnamese_fields = [
        "Số phiếu",
        "Ngày chứng từ",
        "Ngày chứng từ(dd/mm/yyyy)",
        "Ngày xuất",
        "Ngày xuất(dd/mm/yyyy)",
        "Kho nhận",
        "Địa chỉ",
        "Mã AR",
        "Mã FG",
        "Đvt",
        "Số lượng",
    ]
    for vn_field in vietnamese_fields:
        if any(vn_field in key for key in all_keys):
            return "vi"

    english_fields = [
        "document_number",
        "document_date",
        "export_date",
        "receiving_warehouse",
        "address",
        "ar_code",
        "product_code",
        "unit",
        "quantity",
    ]
    if any(field in all_keys for field in english_fields):
        return "en"

    return "en"


def detect_fields_in_sample(
    sample: Dict, language: str = "en"
) -> List[FieldDefinition]:
    """
    Detect which fields are present (non-null) in the sample's structured_output.
    """
    structured_output = None
    if "structured_output" in sample:
        structured_output = sample["structured_output"]
    elif "text" in sample:
        text = sample["text"]
        try:
            structured_output = json.loads(text) if isinstance(text, str) else text
        except:
            return [f for f in FIELD_DEFINITIONS if f.key == "product_code"]
    elif "metadata" in sample or "table_rows" in sample:
        structured_output = sample

    if not structured_output:
        return [f for f in FIELD_DEFINITIONS if f.key == "product_code"]

    if isinstance(structured_output, str):
        try:
            structured_output = json.loads(structured_output)
        except:
            return [f for f in FIELD_DEFINITIONS if f.key == "product_code"]

    if not isinstance(structured_output, dict):
        return [f for f in FIELD_DEFINITIONS if f.key == "product_code"]

    present_fields = []

    metadata = structured_output.get("metadata", [])
    if isinstance(metadata, list) and len(metadata) > 0:
        metadata_dict = metadata[0]
        if isinstance(metadata_dict, dict):
            for key in metadata_dict.keys():
                for field_def in FIELD_DEFINITIONS:
                    if field_def.field_type == "metadata":
                        field_name = field_def.get_name(language)
                        if field_name == key:
                            present_fields.append(field_def)
                            break

    table_rows = structured_output.get("table_rows", [])

    product_code_field = next(
        (f for f in FIELD_DEFINITIONS if f.key == "product_code"), None
    )
    if product_code_field:
        present_fields.append(product_code_field)

    if table_rows and isinstance(table_rows, list) and len(table_rows) > 0:
        first_row = table_rows[0]
        if isinstance(first_row, dict):
            for key in first_row.keys():
                if product_code_field and key == product_code_field.get_name(language):
                    continue
                for field_def in FIELD_DEFINITIONS:
                    if field_def.field_type == "table":
                        field_name = field_def.get_name(language)
                        if field_name == key:
                            present_fields.append(field_def)
                            break

    return present_fields


def get_system_and_user_prompt(
    language: str, sample: Dict = None, custom_fields: List[FieldDefinition] = None
) -> Tuple[str, str]:
    """
    Get system prompt and user prompt based on language and detected fields.

    Args:
        language: Language code ('en', 'ja', 'vi')
        sample: Optional sample to detect fields from
        custom_fields: Optional custom field list to use

    Returns:
        (system_prompt, user_prompt)
    """
    user_prompts = {
        "en": "Extract all required fields and return in JSON format for me.",
        "vi": "Hãy trích xuất tất cả các trường cần thiết và trả về dưới dạng JSON cho tôi.",
        "ja": "必要なフィールドをすべて抽出してJSON形式で返してください。",
    }

    fields_to_use = custom_fields
    if fields_to_use is None and sample is not None:
        fields_to_use = detect_fields_in_sample(sample, language)

    system_prompt = generate_extraction_prompt(
        language=language, custom_fields=fields_to_use
    )
    user_prompt = user_prompts.get(language, user_prompts["en"])

    return system_prompt, user_prompt


def convert_to_conversation(sample: Dict) -> Dict:
    try:
        multiturn = sample.get("is_multiturn", False)
        image = sample["image"]

        if not multiturn:
            text_content = sample.get("text", "")
            language = detect_language(text_content)
            system_prompt, user_prompt = get_system_and_user_prompt(
                language, sample=sample
            )

            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt},
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
        print(f"ERROR: {e}")
        print("-" * 60)
        print("=" * 60 + "\n")
        raise e


class LazyVisionDataset(TorchDataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return convert_to_conversation(sample)

    @property
    def column_names(self):
        if hasattr(self.dataset, "column_names"):
            return ["messages"]
        return ["messages"]


class LazyVisionGRPODataset(TorchDataset):
    def __init__(self, hf_dataset: HuggingFaceDataset, tokenizer: AutoProcessor):
        self.tokenizer = tokenizer

        self.dataset = hf_dataset.filter(
            lambda x: not x.get("is_multiturn", False),
            desc="Filtering non-multiturn samples")

        print(
            f"✅ Filtered dataset: {len(self.dataset)} non-multiturn samples "
            f"from {len(hf_dataset)} total samples"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        image = sample["image"]
        
        if hasattr(image, "resize"):
            image = image.resize((512, 512))
        
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        
        answer_text = sample.get("text", "")
        language = detect_language(answer_text)
        system_prompt, user_prompt = get_system_and_user_prompt(language, sample=sample)
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "prompt": prompt_text,
            "image": image,
            "answer": answer_text
        }
        


def get_best_checkpoint(output_dir: str) -> str:
    """
    """
    if not os.path.isdir(output_dir):
        return None
    
    checkpoints = []
    
    for item in os.listdir(output_dir):
        path = os.path.join(output_dir, item)
        if os.path.isdir(path) and item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[-1])
                checkpoints.append((step, path))
            except ValueError:
                continue
    
    emergency_state = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(emergency_state):
        try:
            with open(emergency_state, 'r') as f:
                state = json.load(f)
                global_step = state.get('global_step', 0)
                checkpoints.append((global_step, output_dir))
        except:
            pass
    
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    best_step, best_path = checkpoints[0]
    
    return best_path