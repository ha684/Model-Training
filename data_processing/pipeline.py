# -*- coding: utf-8 -*-
"""
Data Extraction Pipeline using Gemini API.

This pipeline processes PDF files and images, extracts structured data
using Gemini Vision API, and saves results to JSON with checkpoint support.

Usage:
    python data_processing/pipeline.py --input ./data --output ./output/results.json --api-keys ./api_keys.txt
"""

import argparse
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
from data_processing.utils import parse_gemini_response, load_image, pdf_to_images, append_result, save_checkpoint, load_checkpoint, get_supported_files
from data_processing.utils import ProcessingCheckpoint
from config.fields_config import (
    METADATA_FIELDS,
    TABLE_ROW_FIELDS,
    get_extraction_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def validate_and_normalize_result(result: dict) -> dict:
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
        for field_name in METADATA_FIELDS:
            if field_name not in item:
                item[field_name] = None

    if "table_rows" in result:
        if isinstance(result["table_rows"], list):
            normalized["table_rows"] = result["table_rows"]
        elif isinstance(result["table_rows"], dict):
            normalized["table_rows"] = [result["table_rows"]]

    for item in normalized["table_rows"]:
        for field_name in TABLE_ROW_FIELDS:
            if field_name not in item:
                item[field_name] = None

    return normalized

def extract_from_image(
    image: Image.Image,
    model: genai.GenerativeModel,
    prompt: str,
    key_manager: APIKeyManager,
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
                    return validate_and_normalize_result(result)

            logger.warning(f"Empty or invalid response on attempt {attempt + 1}")

        except Exception as e:
            error_msg = str(e).lower()

            if (
                "quota" in error_msg
                or "rate" in error_msg
                or "limit" in error_msg
                or "daily" in error_msg
            ):
                logger.warning(f"Rate/quota limit hit: {e}")
                raise

            # Check for invalid API key
            if (
                "api key" in error_msg
                or "invalid" in error_msg
                or "unauthorized" in error_msg
            ):
                logger.warning(f"API key error: {e}")
                raise

            logger.warning(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff

    return None


def process_file(
    file_path: str,
    model: genai.GenerativeModel,
    prompt: str,
    checkpoint: ProcessingCheckpoint,
    output_path: str,
    checkpoint_path: str,
    key_manager: APIKeyManager,
) -> bool:
    """Process a single file (PDF or image). Returns False if API keys exhausted."""
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
                logger.info(
                    f"  Skipping page {page_num}/{len(images)} (already processed)"
                )
                continue

            logger.info(f"  Processing page {page_num}/{len(images)}")
            checkpoint.current_file = file_path
            checkpoint.current_page = page_num

            result = extract_from_image(image, model, prompt, key_manager)

            if result:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "pdf",
                    "page_number": page_num,
                    "total_pages": len(images),
                    "extracted_at": datetime.now().isoformat(),
                    "metadata": result.get("metadata", []),
                    "table_rows": result.get("table_rows", []),
                }
                append_result(output_entry, output_path)
                logger.info(f"  ✓ Extracted data from page {page_num}")
            else:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "pdf",
                    "page_number": page_num,
                    "total_pages": len(images),
                    "extracted_at": datetime.now().isoformat(),
                    "metadata": [],
                    "table_rows": [],
                    "error": "Failed to extract data",
                }
                append_result(output_entry, output_path)
                logger.warning(f"  ✗ Failed to extract data from page {page_num}")

            # Update checkpoint
            file_progress["pages"].append(page_num)
            checkpoint.total_processed += 1
            save_checkpoint(checkpoint, checkpoint_path)

        file_progress["completed"] = True
        save_checkpoint(checkpoint, checkpoint_path)

    else:
        # Process image
        try:
            image = load_image(file_path)

            result = extract_from_image(image, model, prompt, key_manager)

            if result:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "image",
                    "page_number": 1,
                    "total_pages": 1,
                    "extracted_at": datetime.now().isoformat(),
                    "metadata": result.get("metadata", []),
                    "table_rows": result.get("table_rows", []),
                }
                append_result(output_entry, output_path)
                logger.info("  ✓ Extracted data")
            else:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "image",
                    "page_number": 1,
                    "total_pages": 1,
                    "extracted_at": datetime.now().isoformat(),
                    "metadata": [],
                    "table_rows": [],
                    "error": "Failed to extract data",
                }
                append_result(output_entry, output_path)
                logger.warning("  ✗ Failed to extract data")

            file_progress["pages"].append(1)
            file_progress["completed"] = True
            checkpoint.total_processed += 1
            save_checkpoint(checkpoint, checkpoint_path)

        except Exception as e:
            logger.error(f"Error processing image {file_name}: {e}")
            raise

    return True


def run_pipeline(
    input_path: str,
    output_path: str,
    api_keys_file: str,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Run the data extraction pipeline."""

    # Setup paths
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = str(output_dir / "checkpoint.json")

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

    # Get extraction prompt
    prompt = get_extraction_prompt()

    # Calculate estimated time
    total_max_requests = len(key_manager.keys) * key_manager.REQUESTS_PER_DAY
    logger.info("=" * 60)
    logger.info("Starting data extraction pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Files to process: {len(files)}")
    logger.info(
        f"API keys: {len(key_manager.keys)} keys × {key_manager.REQUESTS_PER_DAY} req/day = {total_max_requests} max requests/day"
    )
    logger.info(
        f"Request interval: {key_manager.MIN_REQUEST_INTERVAL}s (5 req/min limit)"
    )
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
                model = genai.GenerativeModel("gemini-3-pro-preview")

                success = process_file(
                    file_path=file_path,
                    model=model,
                    prompt=prompt,
                    checkpoint=checkpoint,
                    output_path=output_path,
                    checkpoint_path=checkpoint_path,
                    key_manager=key_manager,
                )

                if success:
                    break  # Move to next file

            except Exception as e:
                error_msg = str(e).lower()

                # Determine failure reason
                if "daily" in error_msg:
                    reason = "daily_limit"
                elif (
                    "quota" in error_msg or "rate" in error_msg or "limit" in error_msg
                ):
                    reason = "rate_limit"
                elif (
                    "api key" in error_msg
                    or "invalid" in error_msg
                    or "unauthorized" in error_msg
                    or "forbidden" in error_msg
                ):
                    reason = "invalid_key"
                else:
                    logger.error(f"Unexpected error: {e}")
                    save_checkpoint(checkpoint, checkpoint_path)
                    raise

                # Try to rotate to next key
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
    logger.info("=" * 60)
    logger.info(key_manager.get_status_summary())
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract data from PDFs and images using Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --input ./data --output ./output/results.json --api-keys ./api_keys.txt
  python pipeline.py -i ./documents -o ./extracted.json -k ./keys.txt
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input directory or file path (supports PDF, PNG, JPG, JPEG, WebP, GIF, BMP)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "-k",
        "--api-keys",
        required=True,
        help="Path to text file containing Gemini API keys (one per line)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Path to checkpoint file (default: same directory as output)",
    )

    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        api_keys_file=args.api_keys,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
