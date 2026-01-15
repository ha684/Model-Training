#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fix dataset on HuggingFace:
- Merge Số lượng(Thùng) and Số lượng(Pcs) into single Số lượng field (take max value)
- Convert date keys from mm/dd/yyyy to dd/mm/yyyy format (keep values unchanged)
- Fix Số phiếu if too short
- Output to JSON for review or push to HuggingFace (memory-safe)
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
DATASET_NAME = "ha684/senko1"
OUTPUT_JSON = "fixed_dataset.json"


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
    Takes the larger value between the two.
    Also fixes Số phiếu if it's too short (likely a page number).
    Converts date keys and values.
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
                    # Remove spaces, dots, dashes to check actual digit count
                    cleaned = str(so_phieu).replace(" ", "").replace(".", "").replace("-", "")
                    if len(cleaned) < 5:
                        metadata["Số phiếu"] = None
            
            # Convert date keys ONLY (keep values as is)
            if "Ngày chứng từ(mm/dd/yyyy)" in metadata:
                date_value = metadata.pop("Ngày chứng từ(mm/dd/yyyy)", None)
                metadata["Ngày chứng từ(dd/mm/yyyy)"] = date_value
            
            # Convert date keys ONLY (keep values as is)
            if "Ngày xuất(mm/dd/yyyy)" in metadata:
                date_value = metadata.pop("Ngày xuất(mm/dd/yyyy)", None)
                metadata["Ngày xuất(dd/mm/yyyy)"] = date_value
    
    # Handle table rows
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
            
            # Take the larger value (max, not sum)
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
    
    if mode == "json":
        # Process each example for JSON output
        output_data = []
        
        for example in tqdm(dataset, desc="Processing examples"):
            try:
                # Parse the text field (JSON string)
                text_data = json.loads(example["text"])
                
                # Fix the data
                fixed_text_data = merge_quantity_fields(text_data)
                
                output_data.append({
                    "text": fixed_text_data,
                    "source_file": example.get("source_file"),
                    "page_number": example.get("page_number"),
                })
                
            except Exception as e:
                print(f"\nError processing example: {e}")
                print(f"Source: {example.get('source_file')}, Page: {example.get('page_number')}")
                # Keep original if error
                output_data.append({
                    "text": json.loads(example["text"]),
                    "source_file": example.get("source_file"),
                    "page_number": example.get("page_number"),
                })
        
        # Save to JSON for review
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nFixed data saved to: {OUTPUT_JSON}")
        print(f"Total examples: {len(output_data)}")
        print("\nReview the JSON file. If everything looks good, run with mode='push'")
        
    elif mode == "push":
        # Update dataset in-place to avoid memory issues
        print("\nUpdating dataset...")
        
        def fix_example(example):
            """Fix text field in each example"""
            try:
                text_data = json.loads(example["text"])
                fixed_text_data = merge_quantity_fields(text_data)
                example["text"] = json.dumps(fixed_text_data, ensure_ascii=False)
            except Exception as e:
                print(f"\nError processing example: {e}")
                print(f"Source: {example.get('source_file')}, Page: {example.get('page_number')}")
            
            return example
        
        updated_dataset = dataset.map(
            fix_example,
            desc="Fixing examples"
        )
        
        print(f"Pushing to HuggingFace: {DATASET_NAME}")
        updated_dataset.push_to_hub(
            DATASET_NAME,
            token=HF_TOKEN,
            commit_message="Fix: Merge quantity fields to Số lượng, fix Số phiếu, rename date keys to dd/mm/yyyy"
        )
        
        print(f"\nDataset successfully pushed to: {DATASET_NAME}")
        print(f"Total examples: {len(updated_dataset)}")
    
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
        print("Usage: python fix_senko1_safe.py [json|push]")
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