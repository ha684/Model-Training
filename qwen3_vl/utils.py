# qwen3-vl/utils.py
import json
from typing import Dict, List, Tuple

from torch.utils.data import Dataset as TorchDataset

from config import (
    FIELD_DEFINITIONS,
    FieldDefinition,
    generate_extraction_prompt,
)


def detect_language(text: str) -> str:
    """
    Detect language from text content.
    Returns: 'vi' for Vietnamese, 'ja' for Japanese, 'en' for English
    """
    if not text:
        return "en"

    japanese_chars = sum(
        1 for c in text if "\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff"
    )
    if japanese_chars > len(text) * 0.1:
        return "ja"

    vietnamese_chars = (
        "ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ"
    )
    if any(c in text.lower() for c in vietnamese_chars):
        return "vi"

    return "en"


def detect_fields_in_sample(
    sample: Dict, language: str = "en"
) -> List[FieldDefinition]:
    """
    Detect which fields are present (non-null) in the sample's structured_output.

    Args:
        sample: Dataset sample with 'text' or 'structured_output' field
        language: Language code for field name mapping

    Returns:
        List of FieldDefinitions that are present in the sample
    """
    structured_output = None

    if "structured_output" in sample:
        structured_output = sample["structured_output"]
    elif "text" in sample:
        text = sample["text"]
        try:
            structured_output = json.loads(text) if isinstance(text, str) else text
        except:
            return FIELD_DEFINITIONS

    if not structured_output:
        return FIELD_DEFINITIONS

    if isinstance(structured_output, str):
        try:
            structured_output = json.loads(structured_output)
        except:
            return FIELD_DEFINITIONS

    present_fields = []

    metadata = structured_output.get("metadata", {})
    for field_def in FIELD_DEFINITIONS:
        if field_def.field_type == "metadata":
            field_name = field_def.get_name(language)
            if field_name in metadata and metadata[field_name] is not None:
                present_fields.append(field_def)

    table_rows = structured_output.get("table_rows", [])
    if table_rows:
        first_row = table_rows[0]
        for field_def in FIELD_DEFINITIONS:
            if field_def.field_type == "table":
                field_name = field_def.get_name(language)
                if field_name in first_row and first_row[field_name] is not None:
                    present_fields.append(field_def)

    if not present_fields:
        return FIELD_DEFINITIONS

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
        print(f"Conversation: {conversation}")
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
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return convert_to_conversation(sample)


class LazyVisionGRPODataset(TorchDataset):
    def __init__(self, hf_dataset, instruction: str, processor):
        self.dataset = hf_dataset
        self.instruction = instruction
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]

        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")

        answer_text = sample.get("text")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.instruction},
                ],
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt_text, "image": image, "answer": answer_text}
