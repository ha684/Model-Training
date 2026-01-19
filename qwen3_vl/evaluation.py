#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for vLLM-served vision-language model on extraction task.
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

import aiohttp
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from tqdm.asyncio import tqdm as atqdm
from difflib import SequenceMatcher
from config.fields_config_vn import get_extraction_prompt_vi

load_dotenv()

HF_TOKEN = os.getenv("hf_token")
base_url = os.getenv("base_url", "https://8fad34849342.ngrok-free.app")
VLLM_API_URL = f"{base_url}/v1/chat/completions"
MODEL_NAME = "ha684/qwen3-vl-2b-grpo-V1.2"
DATASET_NAME = "ha684/vl-vi-conv-mix-v1.1"
DATASET_SPLIT = "validation"
MAX_CONCURRENT_REQUESTS = 2
TIMEOUT = 180


def image_to_base64(image) -> str:
    """Convert PIL Image to base64 string."""
    # Handle case where image might be a dict (from datasets)
    if isinstance(image, dict):
        # If it's already a PIL Image wrapped in dict
        if 'bytes' in image:
            return base64.b64encode(image['bytes']).decode('utf-8')
        # Otherwise try to get the actual image
        image = image.get('image', image)
    
    # If it's a PIL Image
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # If it's already bytes
    if isinstance(image, bytes):
        return base64.b64encode(image).decode('utf-8')
    
    raise ValueError(f"Unsupported image type: {type(image)}")


async def call_vllm_api_async(
    session: aiohttp.ClientSession, image, prompt: str
) -> str:
    """Call vLLM API with image and prompt asynchronously."""
    image_base64 = image_to_base64(image)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 1.5,
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
        print(f"API call failed: {e}")
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
    
    # Nếu khớp hoàn toàn
    if s1 == s2:
        return True
    
    # Tính tỉ lệ tương đồng (SequenceMatcher)
    ratio = SequenceMatcher(None, s1, s2).ratio()
    
    # Logic bổ sung: Nếu chuỗi này chứa chuỗi kia (tránh lỗi do OCR thêm ký tự rác ở cuối)
    if s1 in s2 or s2 in s1:
        # Nếu độ dài chênh lệch không quá lớn (ví dụ không quá 5 ký tự rác)
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
        print(f"Output: {output[:200]}...")
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
    - Chấp nhận 1 hay nhiều dấu _, dấu cách đều được xử lý
    Ví dụ: 
      'AR1830__141' -> 'AR1830'
      'AR1830_141'  -> 'AR1830'
      'AR1830 141'  -> 'AR1830'
    """
    if value is None: return None
    s = str(value).strip()
    parts = re.split(r'[_\s]+', s)
    return parts[0].strip() if parts else value


def normalize_so_phieu(value: str) -> str:
    """
    Normalize Số phiếu for comparison by removing all spaces, dots, and dashes.
    Examples:
    - "252.104.868.0263" -> "2521048680263"
    - "252. 104. 868.0263" -> "2521048680263"
    - "252-104-866-7210" -> "2521048667210"
    - "25 2-104.866-7210" -> "2521048667210"
    """
    if value is None:
        return None
    
    normalized = value.strip().lower()
    # Remove all spaces, dots, and dashes
    normalized = normalized.replace(' ', '').replace('.', '').replace('-', '')
    return normalized


def normalize_dia_chi(value: str) -> str:
    """
    Normalize Địa chỉ for fuzzy comparison by removing spaces and converting to lowercase.
    This helps handle minor OCR errors and spacing differences.
    """
    if value is None:
        return None
    
    normalized = value.strip().lower()
    normalized = normalized.replace(' ', '')
    return normalized


def compare_metadata(pred_meta: Dict, gt_meta: Dict) -> Dict[str, Dict]:
    """Compare metadata fields between prediction and ground truth."""
    results = {}

    if isinstance(gt_meta, list):
        gt_meta = gt_meta[0] if gt_meta else {}
    if isinstance(pred_meta, list):
        pred_meta = pred_meta[0] if pred_meta else {}

    fuzzy_fields = ["Kho nhận", "Địa chỉ"]

    for field in gt_meta.keys():
        pred_original = pred_meta.get(field)
        gt_original = gt_meta.get(field)
        
        correct = False
        
        # 1. Xử lý Mã AR
        if field == "Mã AR":
            pred_val = normalize_ar_code(pred_original)
            gt_val = normalize_ar_code(gt_original)
            correct = (pred_val == gt_val)
            
        # 2. Xử lý Số phiếu
        elif field == "Số phiếu":
            pred_val = normalize_so_phieu(pred_original)
            gt_val = normalize_so_phieu(gt_original)
            correct = (pred_val == gt_val)
            
        # 3. Xử lý Kho nhận và Địa chỉ (Fuzzy Match)
        elif field in fuzzy_fields:
            # Vẫn dùng normalize_dia_chi để xóa bớt khoảng trắng trước khi so sánh mờ
            p_norm = normalize_dia_chi(pred_original)
            g_norm = normalize_dia_chi(gt_original)
            correct = is_fuzzy_match(p_norm, g_norm, threshold=0.85) # Ngưỡng 85% là khá an toàn
            
        # 4. Các trường còn lại
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
    pred_rows: List[Dict], gt_rows: List[Dict]
) -> Tuple[int, int, List[Dict]]:
    """
    Compare table rows between prediction and ground truth.
    A row is correct if:
    - Mã FG matches exactly (after normalization)
    - Số lượng matches exactly
    """
    if not gt_rows:
        return 0, 0, []

    total_rows = len(gt_rows)
    correct_rows = 0
    row_details = []

    pred_by_fg = {normalize_value(row.get("Mã FG")): row for row in (pred_rows or [])}

    for gt_row in gt_rows:
        gt_fg = normalize_value(gt_row.get("Mã FG"))
        gt_qty = normalize_value(gt_row.get("Số lượng"))

        row_detail = {"gt_fg": gt_fg, "match": False, "errors": []}

        if gt_fg not in pred_by_fg:
            row_detail["errors"].append(f"Missing FG: {gt_fg}")
            row_details.append(row_detail)
            continue

        pred_row = pred_by_fg[gt_fg]
        pred_qty = normalize_value(pred_row.get("Số lượng"))

        # Check if quantities match
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
    prompt: str,
    source_file: str = None,
    page_number: int = None,
) -> Dict:
    """Evaluate a single example asynchronously."""
    try:
        gt_data = json.loads(ground_truth)
    except:
        return None

    # Check if ground truth has table_rows
    gt_has_table_rows = bool(gt_data.get("table_rows"))
    
    # Skip evaluation if label doesn't have table_rows
    if not gt_has_table_rows:
        return {
            "success": False,
            "skipped": True,
            "reason": "No table_rows in ground truth",
            "metadata_scores": {},
            "table_rows_correct": 0,
            "table_rows_total": 0,
            "model_response": None,
            "label": gt_data,
            "source_file": source_file,
            "page_number": page_number,
        }

    model_output = await call_vllm_api_async(session, image, prompt)
    pred_data = parse_model_output(model_output)

    if pred_data is None:
        return {
            "success": False,
            "skipped": False,
            "metadata_scores": {},
            "table_rows_correct": 0,
            "table_rows_total": len(gt_data.get("table_rows", [])),
            "model_response": model_output,
            "label": gt_data,
            "source_file": source_file,
            "page_number": page_number,
        }

    # Evaluate metadata
    metadata_scores = compare_metadata(
        pred_data.get("metadata", {}), gt_data.get("metadata", {})
    )

    # Evaluate table rows
    rows_correct, rows_total, row_details = compare_table_rows(
        pred_data.get("table_rows", []), gt_data.get("table_rows", [])
    )

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
    }


async def process_batch(
    session: aiohttp.ClientSession,
    batch_examples: List[Dict],
    prompt: str,
    pbar: atqdm,
) -> List[Dict]:
    """Process a batch of examples concurrently."""
    tasks = []
    for example in batch_examples:
        task = evaluate_single_async(
            session,
            example["image"],
            example["text"],
            prompt,
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


async def evaluate_dataset_async(dataset, prompt: str):
    """Evaluate entire dataset with concurrent requests."""
    total_examples = len(dataset)
    all_results = []

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, limit_per_host=MAX_CONCURRENT_REQUESTS)
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
            
            batch_results = await process_batch(session, batch, prompt, pbar)
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

    prompt = get_extraction_prompt_vi()

    print("\nStarting evaluation...")
    results = asyncio.run(evaluate_dataset_async(dataset, prompt))

    total_examples = len(dataset)
    successful_parses = 0
    skipped_examples = 0
    metadata_field_scores = defaultdict(list)
    total_rows_correct = 0
    total_rows_count = 0

    for idx, result in enumerate(results):
        if result is None:
            print(f"\n[{idx}] Ground truth parsing failed")
            continue

        # Skip examples without table_rows in ground truth
        if result.get("skipped", False):
            skipped_examples += 1
            continue

        if result["success"]:
            successful_parses += 1

            for field, score_info in result["metadata_scores"].items():
                correct = score_info["correct"]
                metadata_field_scores[field].append(1 if correct else 0)

            rows_correct = result["table_rows_correct"]
            rows_total = result["table_rows_total"]
            total_rows_correct += rows_correct
            total_rows_count += rows_total

    # Calculate evaluated examples (excluding skipped)
    evaluated_examples = total_examples - skipped_examples

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal examples: {total_examples}")
    print(f"Skipped examples (no table_rows in GT): {skipped_examples}")
    print(f"Evaluated examples: {evaluated_examples}")
    print(
        f"Successful parses: {successful_parses}/{evaluated_examples} ({100 * successful_parses / evaluated_examples:.1f}%)" if evaluated_examples > 0 else "Successful parses: 0/0"
    )

    print("\n--- METADATA FIELD ACCURACY ---")
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

    print("\n--- TABLE ROWS ACCURACY ---")
    if total_rows_count > 0:
        row_accuracy = 100 * total_rows_correct / total_rows_count
        print(
            f"  Correct rows: {total_rows_correct}/{total_rows_count} ({row_accuracy:.1f}%)"
        )
    else:
        print("  No table rows to evaluate")

    output_file = "evaluation_results.json"
    simplified_results = []
    for i, r in enumerate(results):
        if r is not None:
            # Only save results that have errors
            has_error = False
            
            # Check if parsing failed
            if not r.get("success", False) and not r.get("skipped", False):
                has_error = True
            
            # Check if any metadata field is incorrect
            if r.get("success", False):
                for field, score_info in r.get("metadata_scores", {}).items():
                    if not score_info["correct"]:
                        has_error = True
                        break
                
                # Check if any table rows are incorrect
                if r.get("table_rows_total", 0) > 0:
                    if r.get("table_rows_correct", 0) < r.get("table_rows_total", 0):
                        has_error = True
            
            # Only append if there's an error
            if has_error:
                simplified_results.append(
                    {
                        "source_file": r.get("source_file", f"example_{i}"),
                        "page_number": r.get("page_number"),
                        "model_response": r.get("model_response"),
                        "label": r.get("label"),
                        "success": r.get("success", False),
                        "skipped": r.get("skipped", False),
                        "metadata_scores": r.get("metadata_scores", {}),
                        "table_rows_correct": r.get("table_rows_correct", 0),
                        "table_rows_total": r.get("table_rows_total", 0),
                        "row_details": r.get("row_details", []),
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