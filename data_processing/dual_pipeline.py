# -*- coding: utf-8 -*-
"""
Dual-Pass Data Extraction Pipeline with Validation.

Pass 1: Extract with English field names (fields_config)
Pass 2: Extract with Vietnamese field names (fields_config_vn) for validation

Usage:
    python data_processing/dual_pipeline.py --input ./data --output ./output/results.json --api-keys ./api_keys.txt
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai
from PIL import Image
from data_processing.apikey_manager import APIKeyManager
from data_processing.utils import (
    parse_gemini_response, load_image, pdf_to_images, 
    append_result, save_checkpoint, load_checkpoint, get_supported_files
)
from data_processing.utils import ProcessingCheckpoint
from config.fields_config_en import (
    METADATA_FIELDS,
    TABLE_ROW_FIELDS,
    get_extraction_prompt,
)
from config.fields_config_vn import (
    METADATA_FIELDS_VI as METADATA_FIELDS_VN,
    TABLE_ROW_FIELDS_VI as TABLE_ROW_FIELDS_VN,
    get_extraction_prompt_vi as get_extraction_prompt_vn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dual_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

FIELD_MAPPING = {
    "Số phiếu": "document_number",
    "Ngày chứng từ(dd/mm/yyyy)": "document_date",
    "Ngày xuất(dd/mm/yyyy)": "export_date",
    "Kho nhận": "receiving_warehouse",
    "Địa chỉ": "address",
    "Mã AR": "ar_code",
    
    "Mã FG": "product_code",
    "Đvt": "unit",
    "Số lượng": "quantity",
}


def convert_vn_to_en(result_vn: dict) -> dict:
    """Convert Vietnamese field names to English field names."""
    if not result_vn:
        return result_vn
    
    converted = {}
    
    if "metadata" in result_vn:
        metadata_list = result_vn["metadata"] if isinstance(result_vn["metadata"], list) else [result_vn["metadata"]]
        converted["metadata"] = []
        for item in metadata_list:
            converted_item = {
                FIELD_MAPPING.get(k, k): v
                for k, v in item.items()
            }
            converted["metadata"].append(converted_item)
    
    if "table_rows" in result_vn:
        rows_list = result_vn["table_rows"] if isinstance(result_vn["table_rows"], list) else [result_vn["table_rows"]]
        converted["table_rows"] = []
        for row in rows_list:
            converted_row = {
                FIELD_MAPPING.get(k, k): v
                for k, v in row.items()
            }
            converted["table_rows"].append(converted_row)
    
    return converted


def validate_and_normalize_result(result: dict, metadata_fields: list, table_fields: list) -> dict:
    """Validate and normalize extracted result to match expected schema."""
    normalized = {
        "metadata": [],
        "table_rows": [],
    }

    if "metadata" in result:
        if isinstance(result["metadata"], list):
            normalized["metadata"] = result["metadata"]
        elif isinstance(result["metadata"], dict):
            normalized["metadata"] = [result["metadata"]]

    for item in normalized["metadata"]:
        for field_name in metadata_fields:
            if field_name not in item:
                item[field_name] = None

    if "table_rows" in result:
        if isinstance(result["table_rows"], list):
            normalized["table_rows"] = result["table_rows"]
        elif isinstance(result["table_rows"], dict):
            normalized["table_rows"] = [result["table_rows"]]

    for item in normalized["table_rows"]:
        for field_name in table_fields:
            if field_name not in item:
                item[field_name] = None

    return normalized


def normalize_value(value):
    """Normalize value for comparison (handle type differences)."""
    if value is None:
        return None
    
    # Try to convert to number (for quantity field)
    if isinstance(value, str):
        # Try to parse as number
        try:
            # Remove whitespace
            value = value.strip()
            # Try int first
            if '.' not in value:
                return int(value)
            # Then float
            return float(value)
        except (ValueError, AttributeError):
            # Not a number, return as lowercase string
            return value.lower()
    
    # Already a number
    if isinstance(value, (int, float)):
        return value
    
    # For other types, convert to string and lowercase
    return str(value).lower()


def normalize_result(result: dict) -> dict:
    """Normalize result for comparison."""
    normalized = {
        "metadata": [],
        "table_rows": []
    }
    
    # Normalize metadata
    if "metadata" in result:
        metadata_list = result["metadata"] if isinstance(result["metadata"], list) else [result["metadata"]]
        for item in metadata_list:
            normalized_item = {}
            for key, value in item.items():
                normalized_item[key] = normalize_value(value)
            normalized["metadata"].append(normalized_item)
    
    # Normalize table rows
    if "table_rows" in result:
        rows_list = result["table_rows"] if isinstance(result["table_rows"], list) else [result["table_rows"]]
        for row in rows_list:
            normalized_row = {}
            for key, value in row.items():
                normalized_row[key] = normalize_value(value)
            normalized["table_rows"].append(normalized_row)
    
    return normalized


def results_match(result1: dict, result2: dict) -> bool:
    """
    Compare two extraction results after normalization.
    Handles type differences (int vs string for numbers).
    """
    try:
        # Normalize both results
        norm1 = normalize_result(result1)
        norm2 = normalize_result(result2)
        
        # Convert to JSON and compare
        json1 = json.dumps(norm1, sort_keys=True, ensure_ascii=False)
        json2 = json.dumps(norm2, sort_keys=True, ensure_ascii=False)
        return json1 == json2
    except:
        return False


def log_mismatch(source_file: str, page_num: int, review_log_path: str):
    """Append mismatch entry to review log."""
    log_entry = {
        "source_file": source_file,
        "page_number": page_num,
        "flagged_at": datetime.now().isoformat()
    }
    
    # Read existing log
    if Path(review_log_path).exists():
        with open(review_log_path, 'r', encoding='utf-8') as f:
            try:
                log_data = json.load(f)
                if not isinstance(log_data, list):
                    log_data = []
            except:
                log_data = []
    else:
        log_data = []
    
    # Append new entry
    log_data.append(log_entry)
    
    # Write back
    with open(review_log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)


def extract_from_image(
    image: Image.Image,
    model: genai.GenerativeModel,
    prompt: str,
    key_manager: APIKeyManager,
    metadata_fields: list,
    table_fields: list,
    max_retries: int = 3,
) -> Optional[dict]:
    """Extract data from a single image using Gemini."""
    for attempt in range(max_retries):
        try:
            key_manager.wait_for_rate_limit()

            response = model.generate_content([prompt, image])

            if not key_manager.record_request():
                raise Exception("Daily request limit reached for this API key")

            if response.text:
                result = parse_gemini_response(response.text)
                if result:
                    return validate_and_normalize_result(result, metadata_fields, table_fields)

            logger.warning(f"Empty or invalid response on attempt {attempt + 1}")

        except Exception as e:
            error_msg = str(e).lower()

            if any(keyword in error_msg for keyword in ["quota", "rate", "limit", "daily"]):
                logger.warning(f"Rate/quota limit hit: {e}")
                raise

            if any(keyword in error_msg for keyword in ["api key", "invalid", "unauthorized"]):
                logger.warning(f"API key error: {e}")
                raise

            logger.warning(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(2**attempt)

    return None


def process_file(
    file_path: str,
    model: genai.GenerativeModel,
    prompt_en: str,
    prompt_vn: str,
    checkpoint: ProcessingCheckpoint,
    output_path: str,
    review_log_path: str,
    checkpoint_path: str,
    key_manager: APIKeyManager,
) -> bool:
    """Process a single file with dual-pass validation."""
    file_ext = Path(file_path).suffix.lower()
    file_name = Path(file_path).name

    # Get or initialize file progress
    if file_path not in checkpoint.processed_files:
        checkpoint.processed_files[file_path] = {"pages": [], "completed": False}

    file_progress = checkpoint.processed_files[file_path]

    if file_progress["completed"]:
        logger.info(f"Skipping already completed file: {file_name}")
        return True

    logger.info(f"Processing: {file_name}")

    if file_ext == ".pdf":
        # Process PDF
        images = pdf_to_images(file_path)
        if not images:
            logger.error(f"Failed to convert PDF: {file_name}")
            return True

        for page_num, image in enumerate(images, start=1):
            if page_num in file_progress["pages"]:
                logger.info(f"  Skipping page {page_num}/{len(images)} (already processed)")
                continue

            logger.info(f"  Processing page {page_num}/{len(images)}")
            checkpoint.current_file = file_path
            checkpoint.current_page = page_num

            # Check if we already have Pass 1 cached in checkpoint
            page_key = f"{file_path}:page{page_num}"
            if not hasattr(checkpoint, 'pass1_cache'):
                checkpoint.pass1_cache = {}
            
            result_en = checkpoint.pass1_cache.get(page_key)
            
            # PASS 1: Extract with English prompt (if not cached)
            if not result_en:
                logger.info(f"    Pass 1/2 (English fields)...")
                result_en = extract_from_image(image, model, prompt_en, key_manager, METADATA_FIELDS, TABLE_ROW_FIELDS)

                if not result_en:
                    logger.warning(f"  ✗ Pass 1 failed for page {page_num}")
                    output_entry = {
                        "source_file": file_path,
                        "source_type": "pdf",
                        "page_number": page_num,
                        "total_pages": len(images),
                        "extracted_at": datetime.now().isoformat(),
                        "metadata": [],
                        "table_rows": [],
                        "error": "Pass 1 extraction failed",
                    }
                    append_result(output_entry, output_path)
                    
                    # Update checkpoint
                    file_progress["pages"].append(page_num)
                    checkpoint.total_processed += 1
                    save_checkpoint(checkpoint, checkpoint_path)
                    continue
                
                # Cache Pass 1 result
                checkpoint.pass1_cache[page_key] = result_en
                save_checkpoint(checkpoint, checkpoint_path)
                logger.info(f"    ✓ Pass 1 completed and cached")
            else:
                logger.info(f"    ↻ Pass 1 already completed (using cached result)")

            # PASS 2: Extract with Vietnamese prompt
            logger.info(f"    Pass 2/2 (Vietnamese fields for validation)...")
            result_vn = extract_from_image(image, model, prompt_vn, key_manager, METADATA_FIELDS_VN, TABLE_ROW_FIELDS_VN)

            if not result_vn:
                logger.warning(f"  ✗ Pass 2 failed for page {page_num}")
                # Still save Pass 1 result
                output_entry = {
                    "source_file": file_path,
                    "source_type": "pdf",
                    "page_number": page_num,
                    "total_pages": len(images),
                    "extracted_at": datetime.now().isoformat(),
                    "metadata": result_en.get("metadata", []),
                    "table_rows": result_en.get("table_rows", []),
                    "validation_status": "pass_2_failed",
                }
                append_result(output_entry, output_path)
                
                # Log for review since we couldn't validate
                log_mismatch(file_path, page_num, review_log_path)
                logger.warning(f"  ⚠ Flagged for review (Pass 2 failed)")
                
                # Mark as complete and clear cache
                file_progress["pages"].append(page_num)
                checkpoint.total_processed += 1
                if page_key in checkpoint.pass1_cache:
                    del checkpoint.pass1_cache[page_key]
                save_checkpoint(checkpoint, checkpoint_path)
                continue

            # Convert Vietnamese result to English for comparison
            result_vn_converted = convert_vn_to_en(result_vn)

            # Compare results
            is_match = results_match(result_en, result_vn_converted)

            # Always save Pass 1 result (English fields)
            output_entry = {
                "source_file": file_path,
                "source_type": "pdf",
                "page_number": page_num,
                "total_pages": len(images),
                "extracted_at": datetime.now().isoformat(),
                "metadata": result_en.get("metadata", []),
                "table_rows": result_en.get("table_rows", []),
                "validation_status": "matched" if is_match else "mismatched",
            }
            append_result(output_entry, output_path)

            if is_match:
                logger.info(f"  ✓ Extracted data (validated)")
            else:
                # Log mismatch for review
                log_mismatch(file_path, page_num, review_log_path)
                logger.warning(f"  ⚠ Extracted data (MISMATCH - flagged for review)")
                logger.warning(f"  → Pass 1 (English): {json.dumps(result_en, ensure_ascii=False)}")
                logger.warning(f"  → Pass 2 (Vietnamese): {json.dumps(result_vn, ensure_ascii=False)}")
                logger.warning(f"  → Pass 2 (Converted): {json.dumps(result_vn_converted, ensure_ascii=False)}")

            # Update checkpoint and clear cache
            file_progress["pages"].append(page_num)
            checkpoint.total_processed += 1
            if page_key in checkpoint.pass1_cache:
                del checkpoint.pass1_cache[page_key]
            save_checkpoint(checkpoint, checkpoint_path)

        file_progress["completed"] = True
        save_checkpoint(checkpoint, checkpoint_path)

    else:
        # Process image
        try:
            image = load_image(file_path)

            # Check if we already have Pass 1 cached
            page_key = f"{file_path}:page1"
            if not hasattr(checkpoint, 'pass1_cache'):
                checkpoint.pass1_cache = {}
            
            result_en = checkpoint.pass1_cache.get(page_key)

            # PASS 1 (if not cached)
            if not result_en:
                logger.info(f"  Pass 1/2 (English fields)...")
                result_en = extract_from_image(image, model, prompt_en, key_manager, METADATA_FIELDS, TABLE_ROW_FIELDS)

                if not result_en:
                    logger.warning(f"  ✗ Pass 1 failed")
                    output_entry = {
                        "source_file": file_path,
                        "source_type": "image",
                        "page_number": 1,
                        "total_pages": 1,
                        "extracted_at": datetime.now().isoformat(),
                        "metadata": [],
                        "table_rows": [],
                        "error": "Pass 1 extraction failed",
                    }
                    append_result(output_entry, output_path)
                    
                    file_progress["pages"].append(1)
                    file_progress["completed"] = True
                    checkpoint.total_processed += 1
                    save_checkpoint(checkpoint, checkpoint_path)
                    return True
                
                # Cache Pass 1 result
                checkpoint.pass1_cache[page_key] = result_en
                save_checkpoint(checkpoint, checkpoint_path)
                logger.info(f"  ✓ Pass 1 completed and cached")
            else:
                logger.info(f"  ↻ Pass 1 already completed (using cached result)")

            # PASS 2
            logger.info(f"  Pass 2/2 (Vietnamese fields for validation)...")
            result_vn = extract_from_image(image, model, prompt_vn, key_manager, METADATA_FIELDS_VN, TABLE_ROW_FIELDS_VN)

            if not result_vn:
                logger.warning(f"  ✗ Pass 2 failed")
                output_entry = {
                    "source_file": file_path,
                    "source_type": "image",
                    "page_number": 1,
                    "total_pages": 1,
                    "extracted_at": datetime.now().isoformat(),
                    "metadata": result_en.get("metadata", []),
                    "table_rows": result_en.get("table_rows", []),
                    "validation_status": "pass_2_failed",
                }
                append_result(output_entry, output_path)
                log_mismatch(file_path, 1, review_log_path)
                logger.warning(f"  ⚠ Flagged for review (Pass 2 failed)")
                
                file_progress["pages"].append(1)
                file_progress["completed"] = True
                checkpoint.total_processed += 1
                if page_key in checkpoint.pass1_cache:
                    del checkpoint.pass1_cache[page_key]
                save_checkpoint(checkpoint, checkpoint_path)
                return True

            # Convert Vietnamese result to English for comparison
            result_vn_converted = convert_vn_to_en(result_vn)

            # Compare results
            is_match = results_match(result_en, result_vn_converted)

            # Save Pass 1 result (English fields)
            output_entry = {
                "source_file": file_path,
                "source_type": "image",
                "page_number": 1,
                "total_pages": 1,
                "extracted_at": datetime.now().isoformat(),
                "metadata": result_en.get("metadata", []),
                "table_rows": result_en.get("table_rows", []),
                "validation_status": "matched" if is_match else "mismatched",
            }
            append_result(output_entry, output_path)

            if is_match:
                logger.info("  ✓ Extracted data (validated)")
            else:
                log_mismatch(file_path, 1, review_log_path)
                logger.warning("  ⚠ Extracted data (MISMATCH - flagged for review)")
                logger.warning(f"  → Pass 1 (English): {json.dumps(result_en, ensure_ascii=False)}")
                logger.warning(f"  → Pass 2 (Vietnamese): {json.dumps(result_vn, ensure_ascii=False)}")
                logger.warning(f"  → Pass 2 (Converted): {json.dumps(result_vn_converted, ensure_ascii=False)}")

            file_progress["pages"].append(1)
            file_progress["completed"] = True
            checkpoint.total_processed += 1
            if page_key in checkpoint.pass1_cache:
                del checkpoint.pass1_cache[page_key]
            save_checkpoint(checkpoint, checkpoint_path)

        except Exception as e:
            logger.error(f"Error processing image {file_name}: {e}")
            raise

    return True


def run_pipeline(
    input_path: str,
    output_path: str,
    api_keys_file: str,
    review_log_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Run the dual-pass data extraction pipeline."""

    # Setup paths
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = str(output_dir / "checkpoint.json")
    
    if review_log_path is None:
        review_log_path = str(output_dir / "needs_review.json")

    # Load API keys
    key_manager = APIKeyManager()
    if not key_manager.load_keys(api_keys_file):
        logger.error("Failed to load API keys. Exiting.")
        sys.exit(1)

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)

    # Get files to process
    files = get_supported_files(input_path)
    if not files:
        logger.error(f"No supported files found in: {input_path}")
        sys.exit(1)

    # Get extraction prompts
    prompt_en = get_extraction_prompt()  # English fields
    prompt_vn = get_extraction_prompt_vn()  # Vietnamese fields

    logger.info("=" * 60)
    logger.info("Starting DUAL-PASS data extraction pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Review log: {review_log_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Files to process: {len(files)}")
    logger.info(f"API keys: {len(key_manager.keys)}")
    logger.info("Pass 1: English field names (saved to output)")
    logger.info("Pass 2: Vietnamese field names (for validation)")
    logger.info("NOTE: Each page/image will be processed TWICE (2x API calls)")
    logger.info("=" * 60)
    logger.info(key_manager.get_status_summary())
    logger.info("=" * 60)

    for file_idx, file_path in enumerate(files, start=1):
        logger.info(f"\n[{file_idx}/{len(files)}] {Path(file_path).name}")

        # Configure Gemini with current API key
        while key_manager.has_available_keys():
            current_key = key_manager.get_current_key()
            if not current_key:
                logger.error("No available API keys!")
                sys.exit(1)

            try:
                genai.configure(api_key=current_key)
                
                model = genai.GenerativeModel("gemini-3-flash-preview",generation_config=genai.GenerationConfig(temperature=0.5))

                success = process_file(
                    file_path=file_path,
                    model=model,
                    prompt_en=prompt_en,
                    prompt_vn=prompt_vn,
                    checkpoint=checkpoint,
                    output_path=output_path,
                    review_log_path=review_log_path,
                    checkpoint_path=checkpoint_path,
                    key_manager=key_manager,
                )

                if success:
                    break

            except Exception as e:
                error_msg = str(e).lower()

                if "daily" in error_msg:
                    reason = "daily_limit"
                elif any(kw in error_msg for kw in ["quota", "rate", "limit"]):
                    reason = "rate_limit"
                elif any(kw in error_msg for kw in ["api key", "invalid", "unauthorized", "forbidden"]):
                    reason = "invalid_key"
                else:
                    logger.error(f"Unexpected error: {e}")
                    save_checkpoint(checkpoint, checkpoint_path)
                    raise

                if not key_manager.mark_current_failed(reason):
                    logger.error("All API keys exhausted! Stopping pipeline.")
                    logger.info(key_manager.get_status_summary())
                    save_checkpoint(checkpoint, checkpoint_path)
                    sys.exit(1)
                logger.info("Retrying with new API key...")

        if not key_manager.has_available_keys():
            logger.error("All API keys exhausted! Stopping pipeline.")
            logger.info(key_manager.get_status_summary())
            save_checkpoint(checkpoint, checkpoint_path)
            sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Total items processed: {checkpoint.total_processed}")
    logger.info(f"Output saved to: {output_path}")
    logger.info(f"Review log saved to: {review_log_path}")
    logger.info("=" * 60)
    logger.info(key_manager.get_status_summary())
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dual-pass extraction with validation (English + Vietnamese)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dual_pipeline.py --input ./data --output ./output/results.json --api-keys ./api_keys.txt
  python dual_pipeline.py -i ./documents -o ./extracted.json -r ./review.json -k ./keys.txt
        """,
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory or file path",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "-k", "--api-keys",
        required=True,
        help="Path to API keys file",
    )
    parser.add_argument(
        "-r", "--review-log",
        default=None,
        help="Path to review log file (default: needs_review.json in output dir)",
    )
    parser.add_argument(
        "-c", "--checkpoint",
        default=None,
        help="Path to checkpoint file (default: checkpoint.json in output dir)",
    )

    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        api_keys_file=args.api_keys,
        review_log_path=args.review_log,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()