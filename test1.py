#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fix ha684/senko-V1.1 dataset (Train & Test splits):
- Optimized to avoid MemoryError by disabling image decoding.
- Pushes to a NEW repository (TARGET_DATASET) instead of overwriting.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

# Added Image import
from datasets import Dataset, DatasetDict, load_dataset, Image
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

HF_TOKEN = os.getenv("hf_token")

# --- CẤU HÌNH REPO ---
SOURCE_DATASET = "ha684/senko-V1.1"       # Repo nguồn (tải về)
TARGET_DATASET = "ha684/senko-V1.2" # Repo đích (push lên cái mới)
OUTPUT_JSON = "fixed_senko_v1.1.json"
# ---------------------

def fix_all_fields(data: Dict, is_multiturn: bool) -> Dict:
    """
    Fix all fields - only for records with is_multiturn=False.
    """
    if is_multiturn:
        return data
    
    # Handle metadata
    if "metadata" in data and isinstance(data["metadata"], list):
        for metadata in data["metadata"]:
            if not isinstance(metadata, dict):
                continue
            
            # Fix Số phiếu
            if "Số phiếu" in metadata:
                so_phieu = metadata["Số phiếu"]
                if so_phieu is not None:
                    cleaned = str(so_phieu).replace(" ", "").replace(".", "").replace("-", "")
                    if len(cleaned) < 5:
                        metadata["Số phiếu"] = None
            
            # Fix keys for dates
            if "Ngày chứng từ(mm/dd/yyyy)" in metadata:
                date_value = metadata.pop("Ngày chứng từ(mm/dd/yyyy)")
                metadata["Ngày chứng từ(dd/mm/yyyy)"] = date_value
            
            if "Ngày xuất(mm/dd/yyyy)" in metadata:
                date_value = metadata.pop("Ngày xuất(mm/dd/yyyy)")
                metadata["Ngày xuất(dd/mm/yyyy)"] = date_value
    
    # Handle table rows
    if "table_rows" in data and isinstance(data["table_rows"], list):
        for row in data["table_rows"]:
            qty_thung = row.get("Số lượng(Thùng)")
            qty_pcs = row.get("Số lượng(Pcs)")
            
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
            
            if thung_val is not None and pcs_val is not None:
                max_val = max(thung_val, pcs_val)
            elif thung_val is not None:
                max_val = thung_val
            elif pcs_val is not None:
                max_val = pcs_val
            else:
                max_val = None
            
            row.pop("Số lượng(Thùng)", None)
            row.pop("Số lượng(Pcs)", None)
            row["Số lượng"] = max_val
    
    return data


def process_example(example: Dict) -> Dict:
    """
    Wrapper function to apply fixes to a single dataset example.
    """
    try:
        is_multiturn = example.get("is_multiturn", True)
        
        if not is_multiturn:
            text_data = json.loads(example["text"])
            fixed_text_data = fix_all_fields(text_data, is_multiturn)
            example["text"] = json.dumps(fixed_text_data, ensure_ascii=False)
            
    except Exception as e:
        pass
        
    return example


def fix_dataset(mode: str = "json"):
    print(f"Loading source dataset: {SOURCE_DATASET}")
    dataset = load_dataset(SOURCE_DATASET, token=HF_TOKEN)
    
    # --- FIX FOR MEMORY ERROR ---
    print("Optimizing memory: Disabling image decoding...")
    for split in dataset.keys():
        if "image" in dataset[split].column_names:
            dataset[split] = dataset[split].cast_column("image", Image(decode=False))
    # -----------------------------

    print(f"Dataset loaded. Splits found: {list(dataset.keys())}")

    print("\nProcessing dataset (all splits)...")
    updated_dataset = dataset.map(process_example, desc="Fixing records")
    
    if mode == "json":
        output_data = []
        
        for split_name in updated_dataset.keys():
            print(f"Scanning split '{split_name}' for fixed records...")
            split_data = updated_dataset[split_name]
            
            for item in tqdm(split_data, desc=f"Exporting {split_name}"):
                if not item.get("is_multiturn", True):
                    output_data.append({
                        "split": split_name,
                        "text": json.loads(item["text"]),
                        "source_file": item.get("source_file"),
                        "page_number": item.get("page_number"),
                        "is_multiturn": item.get("is_multiturn"),
                    })
        
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nFixed records saved to: {OUTPUT_JSON}")
        
    elif mode == "push":
        print(f"\nPreparing to push to NEW repository: {TARGET_DATASET}")
        
        # Push to the NEW target repo
        updated_dataset.push_to_hub(
            TARGET_DATASET,
            token=HF_TOKEN,
            commit_message="Fix senko1 records: merge quantity fields, fix keys"
        )
        
        print(f"\nDataset successfully pushed to NEW repo: https://huggingface.co/{TARGET_DATASET}")


def main():
    import sys
    mode = "json"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode not in ["json", "push"]:
        print("Usage: python fix_senko_new_repo.py [json|push]")
        sys.exit(1)
    
    if mode == "push":
        print(f"Source: {SOURCE_DATASET}")
        print(f"Target: {TARGET_DATASET}")
        confirm = input(f"Confirm pushing to '{TARGET_DATASET}'? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(0)
    
    fix_dataset(mode)


if __name__ == "__main__":
    main()