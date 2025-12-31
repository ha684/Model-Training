# -*- coding: utf-8 -*-
"""
Data Preprocessing Script.

This script cleans JSON data extracted by the pipeline.
Specifically, it normalizes the 'Mã AR' field (and others if specified)
to ensure consistent formatting for model fine-tuning.

Usage:
    python preprocess_data.py --input ./output/results.json --output ./output/clean_dataset.json
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def normalize_ar_code(value: str) -> str:
    """
    Normalizes the AR Code.
    
    Logic:
    1. Trims leading/trailing whitespace.
    2. Replaces any sequence of spaces or underscores with a single underscore.
    
    Examples:
    "AR4750 141"    -> "AR4750_141"
    "AR4750__141"   -> "AR4750_141"
    "AR4750_ 141"   -> "AR4750_141"
    " AR4750 141 "  -> "AR4750_141"
    """
    if not isinstance(value, str):
        return value
    
    # Strip whitespace from ends
    cleaned = value.strip()
    
    # Regex: Match 1 or more occurrences of whitespace (\s) OR underscore (_)
    # and replace them with a single underscore.
    cleaned = re.sub(r'[\s_]+', '_', cleaned)
    
    return cleaned

def process_dataset(input_path: str, output_path: str, target_field: str = "Mã AR"):
    """Loads, cleans, and saves the dataset."""
    
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        logger.info(f"Loading data from {input_path}...")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            logger.error("Invalid JSON format. Expected a list of records.")
            sys.exit(1)

        total_records = len(data)
        modified_count = 0
        processed_records = 0

        logger.info(f"Found {total_records} records. Normalizing field: '{target_field}'")

        for entry in data:
            # Check if metadata exists and is a list (as per your pipeline schema)
            if "metadata" in entry and isinstance(entry["metadata"], list):
                for meta_item in entry["metadata"]:
                    if target_field in meta_item and meta_item[target_field]:
                        original_value = meta_item[target_field]
                        new_value = normalize_ar_code(original_value)
                        
                        # Only update and log if actual change occurred
                        if original_value != new_value:
                            meta_item[target_field] = new_value
                            modified_count += 1
                            
                            # Log first 5 changes as sample, then every 50th
                            if modified_count <= 5 or modified_count % 50 == 0:
                                logger.info(f"Modified: '{original_value}' -> '{new_value}'")

            processed_records += 1

        # Save result
        logger.info(f"Saving cleaned data to {output_path}...")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("=" * 40)
        logger.info("Preprocessing Complete")
        logger.info(f"Total Records Checked: {processed_records}")
        logger.info(f"Total Values Modified: {modified_count}")
        logger.info("=" * 40)

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {input_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset for Fine-tuning"
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the extracted results JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to save the cleaned JSON file"
    )
    parser.add_argument(
        "-f", "--field",
        default="Mã AR",
        help="The specific metadata field to normalize (default: 'Mã AR')"
    )

    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.field)

if __name__ == "__main__":
    main()