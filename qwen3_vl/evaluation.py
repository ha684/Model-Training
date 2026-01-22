#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for vLLM-served vision-language model on extraction task.
Supports multi-language evaluation (English, Japanese, Vietnamese).
Compares model predictions against ground truth labels from validation dataset.
Optimized with concurrent requests.
"""

import sys
from pathlib import Path

import re
sys.path.insert(0, str(Path(__file__).parent.parent))
import asyncio
import base64
import json
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Tuple
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import aiohttp
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from tqdm.asyncio import tqdm as atqdm
from difflib import SequenceMatcher

from config import (
    FIELD_DEFINITIONS,
    FieldDefinition,
    generate_extraction_prompt,
)

load_dotenv()

HF_TOKEN = os.getenv("hf_token")
base_url = "https://478daa2e70d4.ngrok-free.app"
VLLM_API_URL = f"{base_url}/v1/chat/completions"
MODEL_NAME = "ha684/qwen3-vl-2b-20-01-26-2epochs"
DATASET_NAME = "ha684/vl-vi-en-ja-conv-mix-v1.3"
DATASET_SPLIT = "validation"
MAX_CONCURRENT_REQUESTS = 10
TIMEOUT = 180
MAX_RETRIES = 3
RETRY_DELAY = 2 


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
        if isinstance(metadata, list):
            metadata = metadata[0] if metadata else {}
        all_keys.extend(metadata.keys())

    table_rows = data.get("table_rows", [])
    if table_rows and isinstance(table_rows, list) and len(table_rows) > 0:
        all_keys.extend(table_rows[0].keys())

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
    # Check if any Vietnamese field exists in keys (exact match or as substring)
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
    
    Rules:
    - Metadata: Detect exact keys present in sample (can be empty)
    - Table: Always include product_code, detect unit and quantity if present

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
            return [f for f in FIELD_DEFINITIONS if f.key == "product_code"]
    elif "metadata" in sample or "table_rows" in sample:
        # Sample is already the structured output dict
        structured_output = sample

    if not structured_output:
        return [f for f in FIELD_DEFINITIONS if f.key == "product_code"]

    if isinstance(structured_output, str):
        try:
            structured_output = json.loads(structured_output)
        except:
            return [f for f in FIELD_DEFINITIONS if f.key == "product_code"]

    present_fields = []

    # Handle metadata - detect keys in order they appear in data
    metadata = structured_output.get("metadata", [])
    if isinstance(metadata, list) and len(metadata) > 0:
        metadata_dict = metadata[0]
        for key in metadata_dict.keys():
            for field_def in FIELD_DEFINITIONS:
                if field_def.field_type == "metadata":
                    field_name = field_def.get_name(language)
                    if field_name == key:
                        present_fields.append(field_def)
                        break

    # Handle table_rows - detect fields in order they appear in data
    table_rows = structured_output.get("table_rows", [])

    # Always add product_code first
    product_code_field = next((f for f in FIELD_DEFINITIONS if f.key == "product_code"), None)
    if product_code_field:
        present_fields.append(product_code_field)

    # Detect other table fields in order they appear
    if table_rows and len(table_rows) > 0:
        first_row = table_rows[0]
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


def image_to_base64(image) -> str:
    """Convert PIL Image to base64 string."""
    if isinstance(image, dict):
        if 'bytes' in image:
            return base64.b64encode(image['bytes']).decode('utf-8')
        image = image.get('image', image)
    
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    if isinstance(image, bytes):
        return base64.b64encode(image).decode('utf-8')
    
    raise ValueError(f"Unsupported image type: {type(image)}")


async def call_vllm_api_async(
    session: aiohttp.ClientSession, image, system_prompt: str, user_prompt: str, retry_count: int = 0
) -> str:
    """Call vLLM API with image and prompts asynchronously with retry logic."""
    image_base64 = image_to_base64(image)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 1.2,
        "min_p": 0.1,
    }

    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('api_key', '')}",
            "Content-Type": "application/json",
        }
        async with session.post(
            VLLM_API_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT),
        ) as response:
            response.raise_for_status()
            result = await response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"API call failed (attempt {retry_count + 1}/{MAX_RETRIES}): {e}. Retrying in {RETRY_DELAY}s...")
            await asyncio.sleep(RETRY_DELAY)
            return await call_vllm_api_async(session, image, system_prompt, user_prompt, retry_count + 1)
        else:
            print(f"API call failed after {MAX_RETRIES} retries: {e}")
            return None


def is_fuzzy_match(s1: str, s2: str, threshold=0.9) -> bool:
    """
    Kiểm tra độ tương đồng giữa 2 chuỗi. 
    Trả về True nếu độ tương đồng >= threshold (0.9 = 90%).
    """
    if s1 is None or s2 is None:
        return s1 == s2
    
    s1 = str(s1).strip().lower()
    s2 = str(s2).strip().lower()
    
    if s1 == s2:
        return True
    
    ratio = SequenceMatcher(None, s1, s2).ratio()
    
    if s1 in s2 or s2 in s1:
        if abs(len(s1) - len(s2)) <= 5:
            return True

    return ratio >= threshold


def parse_model_output(output: str) -> Dict:
    """Parse model output JSON, handling markdown code blocks."""
    if output is None:
        return None

    try:
        cleaned = output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Output: {output}...")
        return None


def normalize_value(value: Any) -> Any:
    """Normalize values for comparison - lowercase and strip whitespace."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (int, float)):
        return float(value)
    return value


def normalize_ar_code(value: str) -> str:
    """
    Xử lý đặc thù Mã AR linh hoạt:
    - Cắt bỏ phần đuôi sau dấu gạch dưới (_) hoặc khoảng trắng ( )
    """
    if value is None: 
        return None
    s = str(value).strip()
    parts = re.split(r'[_\s]+', s)
    return parts[0].strip() if parts else value


def normalize_so_phieu(value: str) -> str:
    """
    Normalize Số phiếu for comparison by removing all spaces, dots, and dashes.
    """
    if value is None:
        return None
    
    normalized = value.strip().lower()
    normalized = normalized.replace(' ', '').replace('.', '').replace('-', '')
    return normalized


def normalize_dia_chi(value: str) -> str:
    """
    Normalize Địa chỉ for fuzzy comparison by removing spaces and converting to lowercase.
    """
    if value is None:
        return None
    
    normalized = value.strip().lower()
    normalized = normalized.replace(' ', '')
    return normalized


def normalize_date_field(value: str) -> str:
    """
    Normalize date fields by removing parentheses and their content.
    """
    if value is None:
        return None
    
    normalized = re.sub(r'\([^)]*\)', '', value)
    return normalized.strip()


def get_field_mappings(language: str) -> Dict[str, str]:
    """
    Get field name mappings for different fields based on language.
    Returns dict mapping field types to actual field names in that language.
    """
    mappings = {
        "en": {
            "document_number": "document_number",
            "document_date": "document_date",
            "export_date": "export_date",
            "receiving_warehouse": "receiving_warehouse",
            "address": "address",
            "ar_code": "ar_code",
            "product_code": "product_code",
            "unit": "unit",
            "quantity": "quantity",
        },
        "vi": {
            "document_number": "Số phiếu",
            "document_date": "Ngày chứng từ",
            "export_date": "Ngày xuất",
            "receiving_warehouse": "Kho nhận",
            "address": "Địa chỉ",
            "ar_code": "Mã AR",
            "product_code": "Mã FG",
            "unit": "Đvt",
            "quantity": "Số lượng",
        },
        "ja": {
            "document_number": "伝票番号",
            "document_date": "伝票日付",
            "export_date": "出庫日付",
            "receiving_warehouse": "入庫倉庫",
            "address": "住所",
            "ar_code": "ARコード",
            "product_code": "製品コード",
            "unit": "単位",
            "quantity": "数量",
        }
    }
    
    return mappings.get(language, mappings["en"])


def compare_metadata(pred_meta: Dict, gt_meta: Dict, language: str) -> Dict[str, Dict]:
    """Compare metadata fields between prediction and ground truth."""
    results = {}

    if isinstance(gt_meta, list):
        gt_meta = gt_meta[0] if gt_meta else {}
    if isinstance(pred_meta, list):
        pred_meta = pred_meta[0] if pred_meta else {}

    # Get field mappings for this language
    field_map = get_field_mappings(language)
    
    # Define fuzzy and date fields using language-specific names
    fuzzy_fields = [field_map["receiving_warehouse"], field_map["address"]]
    date_field_base_names = [
        normalize_date_field(field_map["document_date"]),
        normalize_date_field(field_map["export_date"])
    ]

    for field in gt_meta.keys():
        pred_original = pred_meta.get(field)
        gt_original = gt_meta.get(field)
        
        correct = False
        
        normalized_field = normalize_date_field(field)
        
        # 1. Xử lý AR code
        if field == field_map["ar_code"]:
            pred_val = normalize_ar_code(pred_original)
            gt_val = normalize_ar_code(gt_original)
            correct = (pred_val == gt_val)
            
        # 2. Xử lý document number
        elif field == field_map["document_number"]:
            pred_val = normalize_so_phieu(pred_original)
            gt_val = normalize_so_phieu(gt_original)
            correct = (pred_val == gt_val)
            
        # 3. Xử lý Fuzzy fields (warehouse, address)
        elif field in fuzzy_fields:
            p_norm = normalize_dia_chi(pred_original)
            g_norm = normalize_dia_chi(gt_original)
            correct = is_fuzzy_match(p_norm, g_norm, threshold=0.85)
            
        # 4. Xử lý các trường ngày
        elif normalized_field in date_field_base_names:
            pred_val = None
            for pred_field_name, pred_field_value in pred_meta.items():
                if normalize_date_field(pred_field_name) == normalized_field:
                    pred_val = normalize_value(pred_field_value)
                    break
            
            gt_val = normalize_value(gt_original)
            correct = (pred_val == gt_val)
                
        # 5. Các trường còn lại
        else:
            pred_val = normalize_value(pred_original)
            gt_val = normalize_value(gt_original)
            if gt_val is None and pred_val is None:
                correct = True
            elif gt_val is None or pred_val is None:
                correct = (gt_val == pred_val)
            else:
                correct = (pred_val == gt_val)

        results[field] = {
            "correct": correct,
            "pred": pred_original,
            "gt": gt_original,
        }

    return results


def compare_table_rows(
    pred_rows: List[Dict], gt_rows: List[Dict], language: str
) -> Tuple[int, int, List[Dict]]:
    """
    Compare table rows between prediction and ground truth.
    A row is correct if product_code and quantity match exactly.
    """
    if not gt_rows:
        return 0, 0, []

    field_map = get_field_mappings(language)
    product_code_field = field_map["product_code"]
    quantity_field = field_map["quantity"]

    total_rows = len(gt_rows)
    correct_rows = 0
    row_details = []

    pred_by_code = {
        normalize_value(row.get(product_code_field)): row 
        for row in (pred_rows or [])
    }

    for gt_row in gt_rows:
        gt_code = normalize_value(gt_row.get(product_code_field))
        gt_qty = normalize_value(gt_row.get(quantity_field))

        row_detail = {"gt_code": gt_code, "match": False, "errors": []}

        if gt_code not in pred_by_code:
            row_detail["errors"].append(f"Missing product: {gt_code}")
            row_details.append(row_detail)
            continue

        pred_row = pred_by_code[gt_code]
        pred_qty = normalize_value(pred_row.get(quantity_field))

        qty_match = (gt_qty == pred_qty) if (gt_qty is not None and pred_qty is not None) else False

        if not qty_match:
            row_detail["errors"].append(
                f"Quantity mismatch: pred={pred_qty} vs gt={gt_qty}"
            )

        if qty_match:
            row_detail["match"] = True
            correct_rows += 1

        row_details.append(row_detail)

    return correct_rows, total_rows, row_details


async def evaluate_single_async(
    session: aiohttp.ClientSession,
    image,
    ground_truth: str,
    source_file: str = None,
    page_number: int = None,
    retry_count: int = 0,
) -> Dict:
    """Evaluate a single example asynchronously with retry logic."""
    try:
        gt_data = json.loads(ground_truth)
    except Exception as e:
        print(f"Failed to parse ground truth: {e}")
        return None

    language = detect_language(gt_data)
    
    system_prompt, user_prompt = get_system_and_user_prompt(language, gt_data)

    gt_has_table_rows = bool(gt_data.get("table_rows"))

    try:
        model_output = await call_vllm_api_async(session, image, system_prompt, user_prompt)
        
        if model_output is None:
            if retry_count < MAX_RETRIES:
                print(f"Evaluation failed (attempt {retry_count + 1}/{MAX_RETRIES}) for {source_file}. Retrying...")
                await asyncio.sleep(RETRY_DELAY)
                return await evaluate_single_async(session, image, ground_truth, source_file, page_number, retry_count + 1)
            else:
                return {
                    "success": False,
                    "skipped": False,
                    "error": "API call failed after retries",
                    "metadata_scores": {},
                    "table_rows_correct": 0,
                    "table_rows_total": len(gt_data.get("table_rows", [])) if gt_has_table_rows else 0,
                    "model_response": None,
                    "label": gt_data,
                    "source_file": source_file,
                    "page_number": page_number,
                    "language": language,
                    "gt_has_table_rows": gt_has_table_rows,
                }
        
        pred_data = parse_model_output(model_output)

        if pred_data is None:
            if retry_count < MAX_RETRIES:
                print(f"JSON parsing failed (attempt {retry_count + 1}/{MAX_RETRIES}) for {source_file}. Retrying...")
                await asyncio.sleep(RETRY_DELAY)
                return await evaluate_single_async(session, image, ground_truth, source_file, page_number, retry_count + 1)
            else:
                return {
                    "success": False,
                    "skipped": False,
                    "error": "JSON parsing failed after retries",
                    "metadata_scores": {},
                    "table_rows_correct": 0,
                    "table_rows_total": len(gt_data.get("table_rows", [])) if gt_has_table_rows else 0,
                    "model_response": model_output,
                    "label": gt_data,
                    "source_file": source_file,
                    "page_number": page_number,
                    "language": language,
                    "gt_has_table_rows": gt_has_table_rows,
                }

        # Evaluate metadata
        metadata_scores = compare_metadata(
            pred_data.get("metadata", {}), 
            gt_data.get("metadata", {}),
            language
        )

        # Check if model correctly predicted no table_rows when GT has none
        pred_has_table_rows = bool(pred_data.get("table_rows"))
        table_structure_correct = (gt_has_table_rows == pred_has_table_rows)

        # Evaluate table rows only if GT has table_rows
        if gt_has_table_rows:
            rows_correct, rows_total, row_details = compare_table_rows(
                pred_data.get("table_rows", []), 
                gt_data.get("table_rows", []),
                language
            )
        else:
            # GT has no table_rows - check if model also didn't generate any
            rows_correct = 0
            rows_total = 0
            row_details = []
            
            # If model incorrectly generated table_rows when GT has none
            if pred_has_table_rows:
                row_details.append({
                    "error": f"Model incorrectly generated {len(pred_data.get('table_rows', []))} table rows when GT has none"
                })

        return {
            "success": True,
            "skipped": False,
            "metadata_scores": metadata_scores,
            "table_rows_correct": rows_correct,
            "table_rows_total": rows_total,
            "row_details": row_details,
            "model_response": pred_data,
            "label": gt_data,
            "source_file": source_file,
            "page_number": page_number,
            "language": language,
            "gt_has_table_rows": gt_has_table_rows,
            "pred_has_table_rows": pred_has_table_rows,
            "table_structure_correct": table_structure_correct,
        }
    except Exception as e:
        # Retry on any unexpected error
        if retry_count < MAX_RETRIES:
            print(f"Unexpected error (attempt {retry_count + 1}/{MAX_RETRIES}) for {source_file}: {e}. Retrying...")
            await asyncio.sleep(RETRY_DELAY)
            return await evaluate_single_async(session, image, ground_truth, source_file, page_number, retry_count + 1)
        else:
            print(f"Evaluation failed after {MAX_RETRIES} retries for {source_file}: {e}")
            return {
                "success": False,
                "skipped": False,
                "error": str(e),
                "metadata_scores": {},
                "table_rows_correct": 0,
                "table_rows_total": 0,
                "model_response": None,
                "label": gt_data if 'gt_data' in locals() else None,
                "source_file": source_file,
                "page_number": page_number,
                "language": language if 'language' in locals() else None,
                "gt_has_table_rows": gt_has_table_rows if 'gt_has_table_rows' in locals() else None,
            }


async def process_batch(
    session: aiohttp.ClientSession,
    batch_examples: List[Dict],
    pbar: atqdm,
) -> List[Dict]:
    """Process a batch of examples concurrently."""
    tasks = []
    for example in batch_examples:
        task = evaluate_single_async(
            session,
            example["image"],
            example["text"],
            example.get("source_file"),
            example.get("page_number"),
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    pbar.update(len(batch_examples))

    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"\nError processing example: {result}")
            valid_results.append(None)
        else:
            valid_results.append(result)

    return valid_results


async def evaluate_dataset_async(dataset):
    """Evaluate entire dataset with concurrent requests."""
    total_examples = len(dataset)
    all_results = []

    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS, 
        limit_per_host=MAX_CONCURRENT_REQUESTS
    )
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        pbar = atqdm(total=total_examples, desc="Evaluating")

        for i in range(0, total_examples, MAX_CONCURRENT_REQUESTS):
            batch_slice = dataset[i : i + MAX_CONCURRENT_REQUESTS]
            batch = []
            batch_size = min(MAX_CONCURRENT_REQUESTS, total_examples - i)
            for j in range(batch_size):
                example = {
                    "image": batch_slice["image"][j],
                    "text": batch_slice["text"][j],
                    "source_file": batch_slice.get("source_file", [None] * batch_size)[j],
                    "page_number": batch_slice.get("page_number", [None] * batch_size)[j],
                }
                batch.append(example)
            
            batch_results = await process_batch(session, batch, pbar)
            all_results.extend(batch_results)

        pbar.close()

    return all_results


def main():
    """Main evaluation loop."""
    print(f"Loading dataset: {DATASET_NAME} ({DATASET_SPLIT})")
    dataset = load_dataset(
        "parquet", 
        data_files={"test": f"hf://datasets/{DATASET_NAME}/data/test-*.parquet"},
        split="test"
    )

    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    print(f"Concurrent requests: {MAX_CONCURRENT_REQUESTS}")

    print("\nStarting evaluation...")
    results = asyncio.run(evaluate_dataset_async(dataset))

    total_examples = len(dataset)
    successful_parses = 0
    skipped_examples = 0
    metadata_field_scores = defaultdict(list)
    total_rows_correct = 0
    total_rows_count = 0
    table_structure_correct_count = 0
    table_structure_total_count = 0
    
    # Track stats by language
    language_stats = defaultdict(lambda: {
        "total": 0,
        "successful": 0,
        "metadata_scores": defaultdict(list),
        "rows_correct": 0,
        "rows_total": 0,
        "table_structure_correct": 0,
        "table_structure_total": 0,
    })

    for idx, result in enumerate(results):
        if result is None:
            print(f"\n[{idx}] Evaluation completely failed")
            continue

        if result.get("skipped", False):
            skipped_examples += 1
            continue

        language = result.get("language", "unknown")
        language_stats[language]["total"] += 1

        # Track table structure accuracy
        if "table_structure_correct" in result:
            table_structure_total_count += 1
            language_stats[language]["table_structure_total"] += 1
            if result["table_structure_correct"]:
                table_structure_correct_count += 1
                language_stats[language]["table_structure_correct"] += 1

        if result["success"]:
            successful_parses += 1
            language_stats[language]["successful"] += 1

            for field, score_info in result["metadata_scores"].items():
                correct = score_info["correct"]
                metadata_field_scores[field].append(1 if correct else 0)
                language_stats[language]["metadata_scores"][field].append(1 if correct else 0)

            rows_correct = result["table_rows_correct"]
            rows_total = result["table_rows_total"]
            
            # Only count rows if GT has table_rows
            if result.get("gt_has_table_rows", True):
                total_rows_correct += rows_correct
                total_rows_count += rows_total
                language_stats[language]["rows_correct"] += rows_correct
                language_stats[language]["rows_total"] += rows_total
        else:
            # Count failed evaluations
            if "error" in result:
                print(f"\n[{idx}] Failed: {result.get('error')} - {result.get('source_file')}")

    evaluated_examples = total_examples - skipped_examples

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal examples: {total_examples}")
    print(f"Skipped examples (no table_rows in GT): {skipped_examples}")
    print(f"Evaluated examples: {evaluated_examples}")
    print(
        f"Successful parses: {successful_parses}/{evaluated_examples} "
        f"({100 * successful_parses / evaluated_examples:.1f}%)" 
        if evaluated_examples > 0 else "Successful parses: 0/0"
    )

    # Print overall metadata accuracy
    print("\n--- OVERALL METADATA FIELD ACCURACY ---")
    for field in sorted(metadata_field_scores.keys()):
        scores = metadata_field_scores[field]
        accuracy = np.mean(scores) * 100
        print(f"  {field:30s}: {accuracy:5.1f}% ({sum(scores)}/{len(scores)})")

    if metadata_field_scores:
        overall_metadata = (
            np.mean([np.mean(scores) for scores in metadata_field_scores.values()])
            * 100
        )
        print(f"\n  {'Overall Metadata Accuracy':30s}: {overall_metadata:5.1f}%")

    # Print overall table rows accuracy
    print("\n--- OVERALL TABLE ROWS ACCURACY ---")
    if total_rows_count > 0:
        row_accuracy = 100 * total_rows_correct / total_rows_count
        print(
            f"  Correct rows: {total_rows_correct}/{total_rows_count} ({row_accuracy:.1f}%)"
        )
    else:
        print("  No table rows to evaluate")
    
    # Print table structure accuracy (whether model correctly detected presence/absence of table)
    print("\n--- TABLE STRUCTURE DETECTION ACCURACY ---")
    if table_structure_total_count > 0:
        structure_accuracy = 100 * table_structure_correct_count / table_structure_total_count
        print(
            f"  Correct structure detection: {table_structure_correct_count}/{table_structure_total_count} ({structure_accuracy:.1f}%)"
        )
        print("  (Whether model correctly identified if document has table_rows or not)")
    else:
        print("  No structure detection to evaluate")

    # Print per-language statistics
    print("\n" + "=" * 80)
    print("PER-LANGUAGE STATISTICS")
    print("=" * 80)
    
    for lang in sorted(language_stats.keys()):
        stats = language_stats[lang]
        print(f"\n--- {lang.upper()} ---")
        print(f"  Total: {stats['total']}")
        print(f"  Successful: {stats['successful']}/{stats['total']} "
              f"({100 * stats['successful'] / stats['total']:.1f}%)" 
              if stats['total'] > 0 else "  Successful: 0/0")
        
        if stats['metadata_scores']:
            print(f"\n  Metadata accuracy:")
            for field in sorted(stats['metadata_scores'].keys()):
                scores = stats['metadata_scores'][field]
                accuracy = np.mean(scores) * 100
                print(f"    {field:30s}: {accuracy:5.1f}% ({sum(scores)}/{len(scores)})")
        
        if stats['rows_total'] > 0:
            row_acc = 100 * stats['rows_correct'] / stats['rows_total']
            print(f"\n  Table rows accuracy: {stats['rows_correct']}/{stats['rows_total']} ({row_acc:.1f}%)")
        
        if stats['table_structure_total'] > 0:
            structure_acc = 100 * stats['table_structure_correct'] / stats['table_structure_total']
            print(f"  Table structure detection: {stats['table_structure_correct']}/{stats['table_structure_total']} ({structure_acc:.1f}%)")

    # Save results
    output_file = "evaluation_results.json"
    simplified_results = []
    for i, r in enumerate(results):
        if r is not None:
            has_error = False
            
            if not r.get("success", False) and not r.get("skipped", False):
                has_error = True
            
            if r.get("success", False):
                for field, score_info in r.get("metadata_scores", {}).items():
                    if not score_info["correct"]:
                        has_error = True
                        break
                
                if r.get("table_rows_total", 0) > 0:
                    if r.get("table_rows_correct", 0) < r.get("table_rows_total", 0):
                        has_error = True
            
            if has_error:
                simplified_results.append(
                    {
                        "source_file": r.get("source_file", f"example_{i}"),
                        "page_number": r.get("page_number"),
                        "language": r.get("language"),
                        "model_response": r.get("model_response"),
                        "label": r.get("label"),
                        "success": r.get("success", False),
                        "skipped": r.get("skipped", False),
                        "metadata_scores": r.get("metadata_scores", {}),
                        "table_rows_correct": r.get("table_rows_correct", 0),
                        "table_rows_total": r.get("table_rows_total", 0),
                        "row_details": r.get("row_details", []),
                        "gt_has_table_rows": r.get("gt_has_table_rows"),
                        "pred_has_table_rows": r.get("pred_has_table_rows"),
                        "table_structure_correct": r.get("table_structure_correct"),
                    }
                )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total_examples": total_examples,
                    "skipped_examples": skipped_examples,
                    "evaluated_examples": evaluated_examples,
                    "successful_parses": successful_parses,
                    "total_errors": len(simplified_results),
                    "metadata_accuracy": {
                        k: np.mean(v) * 100 for k, v in metadata_field_scores.items()
                    },
                    "table_rows_accuracy": 100 * total_rows_correct / total_rows_count
                    if total_rows_count > 0
                    else 0,
                    "table_structure_accuracy": 100 * table_structure_correct_count / table_structure_total_count
                    if table_structure_total_count > 0
                    else 0,
                    "language_breakdown": {
                        lang: {
                            "total": stats["total"],
                            "successful": stats["successful"],
                            "metadata_accuracy": {
                                k: np.mean(v) * 100 
                                for k, v in stats["metadata_scores"].items()
                            } if stats["metadata_scores"] else {},
                            "table_rows_accuracy": 100 * stats["rows_correct"] / stats["rows_total"]
                            if stats["rows_total"] > 0 else 0,
                            "table_structure_accuracy": 100 * stats["table_structure_correct"] / stats["table_structure_total"]
                            if stats["table_structure_total"] > 0 else 0,
                        }
                        for lang, stats in language_stats.items()
                    }
                },
                "errors": simplified_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDetailed results saved to: {output_file}")
    print(f"Total errors saved: {len(simplified_results)}")


if __name__ == "__main__":
    main()