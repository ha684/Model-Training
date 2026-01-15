#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fix ha684/senko-V1.1 dataset:
- Find records with is_multiturn = False (from ha684/senko1)
- Merge Số lượng(Thùng) and Số lượng(Pcs) into single Số lượng field
- Fix Số phiếu (set to null if < 5 characters)
- Change date keys from mm/dd/yyyy to dd/mm/yyyy format (keep values unchanged)
- Push back to HuggingFace
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

HF_TOKEN = os.getenv("hf_token")
DATASET_NAME = "ha684/senko-V1.1"
OUTPUT_JSON = "fixed_senko_v1.1.json"


def convert_date_format(date_str: str) -> str:
    """
    Convert date from mm/dd/yyyy to dd/mm/yyyy format.
    Returns original string if conversion fails.
    """
    if not date_str or date_str == "null" or date_str is None:
        return date_str
    
    try:
        # Try parsing as mm/dd/yyyy
        date_obj = datetime.strptime(str(date_str), "%m/%d/%Y")
        # Convert to dd/mm/yyyy
        return date_obj.strftime("%d/%m/%Y")
    except:
        # If parsing fails, return original
        return date_str


def fix_all_fields(data: Dict, is_multiturn: bool) -> Dict:
    """
    Fix all fields - only for records with is_multiturn=False.
    - Merge Số lượng(Thùng) and Số lượng(Pcs) into Số lượng
    - Fix Số phiếu (set to null if < 5 characters)
    - Changes date key names (keep values unchanged).
    """
    # Only process records from ha684/senko1 (is_multiturn = False)
    if is_multiturn:
        return data
    
    # Handle metadata (it's an array, not dict)
    if "metadata" in data and isinstance(data["metadata"], list):
        for metadata in data["metadata"]:
            if not isinstance(metadata, dict):
                continue
            
            # Fix Số phiếu - set to null if length < 5 (likely page number)
            if "Số phiếu" in metadata:
                so_phieu = metadata["Số phiếu"]
                if so_phieu is not None:
                    # Remove spaces, dots, dashes to check actual digit count
                    cleaned = str(so_phieu).replace(" ", "").replace(".", "").replace("-", "")
                    if len(cleaned) < 5:
                        metadata["Số phiếu"] = None
            
            # Fix Ngày chứng từ - only change key, keep value
            if "Ngày chứng từ(mm/dd/yyyy)" in metadata:
                date_value = metadata.pop("Ngày chứng từ(mm/dd/yyyy)")
                metadata["Ngày chứng từ(dd/mm/yyyy)"] = date_value
            
            # Fix Ngày xuất - only change key, keep value
            if "Ngày xuất(mm/dd/yyyy)" in metadata:
                date_value = metadata.pop("Ngày xuất(mm/dd/yyyy)")
                metadata["Ngày xuất(dd/mm/yyyy)"] = date_value
    
    # Handle table rows - merge quantity fields
    if "table_rows" in data and isinstance(data["table_rows"], list):
        for row in data["table_rows"]:
            qty_thung = row.get("Số lượng(Thùng)")
            qty_pcs = row.get("Số lượng(Pcs)")
            
            # Get numeric values
            thung_val = None
            pcs_val = None
            
            if qty_thung is not None:
                try:
                    thung_val = float(qty_thung)
                except:
                    pass
            
            if qty_pcs is not None:
                try:
                    pcs_val = float(qty_pcs)
                except:
                    pass
            
            # Take the larger value
            if thung_val is not None and pcs_val is not None:
                max_val = max(thung_val, pcs_val)
            elif thung_val is not None:
                max_val = thung_val
            elif pcs_val is not None:
                max_val = pcs_val
            else:
                max_val = None
            
            # Remove old fields and add new field
            row.pop("Số lượng(Thùng)", None)
            row.pop("Số lượng(Pcs)", None)
            row["Số lượng"] = max_val
    
    return data


def fix_dataset(mode: str = "json"):
    """
    Fix the dataset and either save to JSON or push to HuggingFace.
    
    Args:
        mode: Either "json" to save locally or "push" to upload to HuggingFace
    """
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train", token=HF_TOKEN)
    
    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    
    # Count records to fix
    records_to_fix = sum(1 for ex in dataset if not ex.get("is_multiturn", True))
    print(f"Records with is_multiturn=False (from ha684/senko1): {records_to_fix}")
    
    # Process each example
    fixed_data = []
    fixed_count = 0
    
    for example in tqdm(dataset, desc="Processing examples"):
        try:
            is_multiturn = example.get("is_multiturn", True)
            
            # Parse the text field (JSON string)
            text_data = json.loads(example["text"])
            
            # Fix date fields only for records with is_multiturn=False
            if not is_multiturn:
                fixed_text_data = fix_all_fields(text_data, is_multiturn)
                fixed_count += 1
            else:
                fixed_text_data = text_data
            
            # Create fixed example
            fixed_example = {
                "image": example["image"],
                "text": json.dumps(fixed_text_data, ensure_ascii=False),
                "source_file": example.get("source_file"),
                "page_number": example.get("page_number"),
                "is_multiturn": is_multiturn,
            }
            
            fixed_data.append(fixed_example)
            
        except Exception as e:
            print(f"\nError processing example: {e}")
            print(f"Source: {example.get('source_file')}, Page: {example.get('page_number')}")
            # Keep original if error
            fixed_data.append({
                "image": example["image"],
                "text": example["text"],
                "source_file": example.get("source_file"),
                "page_number": example.get("page_number"),
                "is_multiturn": example.get("is_multiturn", True),
            })
    
    print(f"\nFixed {fixed_count} records")
    
    if mode == "json":
        # Save to JSON for review (only the fixed records)
        output_data = []
        for item in fixed_data:
            if not item["is_multiturn"]:
                output_data.append({
                    "text": json.loads(item["text"]),
                    "source_file": item["source_file"],
                    "page_number": item["page_number"],
                    "is_multiturn": item["is_multiturn"],
                })
        
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nFixed records saved to: {OUTPUT_JSON}")
        print(f"Total fixed records in JSON: {len(output_data)}")
        print("\nReview the JSON file. If everything looks good, run with mode='push'")
        
    elif mode == "push":
        # Update dataset in-place to avoid memory issues
        print("\nUpdating dataset...")
        
        # Update the text field in the original dataset
        def update_text(example, idx):
            example["text"] = fixed_data[idx]["text"]
            return example
        
        updated_dataset = dataset.map(
            update_text,
            with_indices=True,
            desc="Updating dataset"
        )
        
        print(f"Pushing to HuggingFace: {DATASET_NAME}")
        updated_dataset.push_to_hub(
            DATASET_NAME,
            token=HF_TOKEN,
            commit_message="Fix senko1 records: merge quantity fields, fix Số phiếu, rename date keys to dd/mm/yyyy"
        )
        
        print(f"\nDataset successfully pushed to: {DATASET_NAME}")
        print(f"Total examples: {len(updated_dataset)}")
        print(f"Fixed examples: {fixed_count}")
    
    else:
        print(f"Invalid mode: {mode}. Use 'json' or 'push'")


def main():
    """Main function with mode selection."""
    import sys
    
    # Default to json mode for safety
    mode = "json"
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode not in ["json", "push"]:
        print("Usage: python fix_senko_v1.1_dates.py [json|push]")
        print("  json - Save fixed data to JSON for review (default)")
        print("  push - Push fixed data to HuggingFace")
        sys.exit(1)
    
    if mode == "push":
        confirm = input("Are you sure you want to push to HuggingFace? This will overwrite the dataset. (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(0)
    
    fix_dataset(mode)


if __name__ == "__main__":
    main()