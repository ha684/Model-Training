#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fix dataset ha684/senko1 (Train & Validation splits):
- SAFE MODE: Checks if fields exist before merging to avoid overwriting already fixed data.
- Merge Số lượng(Thùng) and Số lượng(Pcs) into single Số lượng field (take max value)
- Convert date keys from mm/dd/yyyy to dd/mm/yyyy format (keep values unchanged)
- Fix Số phiếu if too short
- Output to JSON for review or push to HuggingFace (handles all splits)
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

HF_TOKEN = os.getenv("hf_token")
DATASET_NAME = "ha684/senko1"
OUTPUT_JSON = "fixed_senko1_safe.json"


def convert_date_format(date_str: str) -> str:
    """
    Convert date from mm/dd/yyyy to dd/mm/yyyy format.
    Returns original string if conversion fails.
    """
    if not date_str or date_str == "null":
        return date_str
    
    try:
        # Try parsing as mm/dd/yyyy
        date_obj = datetime.strptime(date_str, "%m/%d/%Y")
        # Convert to dd/mm/yyyy
        return date_obj.strftime("%d/%m/%Y")
    except:
        # If parsing fails, return original
        return date_str


def merge_quantity_fields(data: Dict) -> Dict:
    """
    Merge Số lượng(Thùng) and Số lượng(Pcs) into single Số lượng field.
    SAFE VERSION: Only merges if old keys exist.
    """
    # Handle metadata (it's an array, not dict)
    if "metadata" in data and isinstance(data["metadata"], list):
        for metadata in data["metadata"]:
            if not isinstance(metadata, dict):
                continue
            
            # Fix Số phiếu - set to null if length < 5 (likely page number)
            if "Số phiếu" in metadata:
                so_phieu = metadata["Số phiếu"]
                if so_phieu is not None:
                    cleaned = str(so_phieu).replace(" ", "").replace(".", "").replace("-", "")
                    if len(cleaned) < 5:
                        metadata["Số phiếu"] = None
            
            # Convert date keys ONLY (Check if key exists first - Safe)
            if "Ngày chứng từ(mm/dd/yyyy)" in metadata:
                date_value = metadata.pop("Ngày chứng từ(mm/dd/yyyy)", None)
                metadata["Ngày chứng từ(dd/mm/yyyy)"] = date_value
            
            # Convert date keys ONLY (Check if key exists first - Safe)
            if "Ngày xuất(mm/dd/yyyy)" in metadata:
                date_value = metadata.pop("Ngày xuất(mm/dd/yyyy)", None)
                metadata["Ngày xuất(dd/mm/yyyy)"] = date_value
    
    # Handle table rows
    if "table_rows" in data and isinstance(data["table_rows"], list):
        for row in data["table_rows"]:
            # --- SAFE CHECK: Only proceed if old fields exist ---
            has_thung = "Số lượng(Thùng)" in row
            has_pcs = "Số lượng(Pcs)" in row
            
            if not has_thung and not has_pcs:
                # If neither exists, assume it's already fixed or doesn't need fixing
                continue
            # ----------------------------------------------------

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
            
            # Remove old fields and update new field
            row.pop("Số lượng(Thùng)", None)
            row.pop("Số lượng(Pcs)", None)
            
            # Update "Số lượng"
            # Logic: If we found old keys, we overwrite "Số lượng" with the new calculation.
            # If max_val is found, use it. 
            row["Số lượng"] = max_val
    
    return data


def process_example(example: Dict) -> Dict:
    """Wrapper to fix text field in each example."""
    try:
        text_data = json.loads(example["text"])
        fixed_text_data = merge_quantity_fields(text_data)
        example["text"] = json.dumps(fixed_text_data, ensure_ascii=False)
    except Exception as e:
        pass
    return example


def fix_dataset(mode: str = "json"):
    """
    Fix the dataset and either save to JSON or push to HuggingFace.
    """
    print(f"Loading dataset: {DATASET_NAME}")
    # Load all splits (returns DatasetDict)
    dataset = load_dataset(DATASET_NAME, token=HF_TOKEN)
    
    print(f"Dataset loaded. Splits found: {list(dataset.keys())}")
    for split in dataset.keys():
        print(f" - {split}: {len(dataset[split])} examples")
    
    # Apply processing to all splits
    print("\nProcessing dataset (all splits)...")
    updated_dataset = dataset.map(process_example, desc="Fixing examples")

    if mode == "json":
        output_data = []
        for split_name in updated_dataset.keys():
            print(f"Scanning split '{split_name}'...")
            split_data = updated_dataset[split_name]
            
            for example in tqdm(split_data, desc=f"Exporting {split_name}"):
                try:
                    output_data.append({
                        "split": split_name,
                        "text": json.loads(example["text"]),
                        "source_file": example.get("source_file"),
                        "page_number": example.get("page_number"),
                    })
                except:
                    continue
        
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nFixed data saved to: {OUTPUT_JSON}")
        
    elif mode == "push":
        print(f"\nPushing all splits to HuggingFace: {DATASET_NAME}")
        updated_dataset.push_to_hub(
            DATASET_NAME,
            token=HF_TOKEN,
            commit_message="Fix: Merge quantity fields (Safe Mode), fix keys (Train & Validation)"
        )
        print(f"\nDataset successfully pushed to: {DATASET_NAME}")
    
    else:
        print(f"Invalid mode: {mode}")


def main():
    import sys
    mode = "json"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode == "push":
        confirm = input("Are you sure you want to push? (yes/no): ")
        if confirm.lower() != "yes":
            sys.exit(0)
    
    fix_dataset(mode)


if __name__ == "__main__":
    main()