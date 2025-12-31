# -*- coding: utf-8 -*-
"""
Data Extraction Pipeline using Gemini API.

This pipeline processes PDF files and images, extracts structured data
using Gemini Vision API, and saves results to JSON with checkpoint support.

Usage:
    python pipeline.py --input ./data --output ./output/results.json --api-keys ./api_keys.txt
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path to allow importing from config
sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image

from config.fields_config import (
    METADATA_FIELDS,
    TABLE_ROW_FIELDS,
    get_extraction_prompt,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ProcessingCheckpoint:
    """Tracks processing progress for resume capability."""

    processed_files: dict = field(
        default_factory=dict
    )  # file_path -> {"pages": [processed_page_numbers], "completed": bool}
    current_file: Optional[str] = None
    current_page: int = 0
    total_processed: int = 0
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {
            "processed_files": self.processed_files,
            "current_file": self.current_file,
            "current_page": self.current_page,
            "total_processed": self.total_processed,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingCheckpoint":
        return cls(
            processed_files=data.get("processed_files", {}),
            current_file=data.get("current_file"),
            current_page=data.get("current_page", 0),
            total_processed=data.get("total_processed", 0),
            last_updated=data.get("last_updated", ""),
        )


@dataclass
class APIKeyManager:
    """Manages multiple API keys with automatic rotation on failure.

    Rate limits (Google Gemini free tier):
    - 5 requests per minute per key
    - 20 requests per day per key
    """

    keys: list = field(default_factory=list)
    current_index: int = 0
    failed_keys: set = field(default_factory=set)

    # Rate limiting config
    REQUESTS_PER_MINUTE: int = 5
    REQUESTS_PER_DAY: int = 20
    MIN_REQUEST_INTERVAL: float = 13.0  # 60/5 = 12s, use 13s for safety margin

    # Track usage per key: {key_index: {"daily_count": int, "last_request_time": float, "date": str}}
    key_usage: dict = field(default_factory=dict)

    def _get_today(self) -> str:
        """Get today's date as string for tracking daily usage."""
        return datetime.now().strftime("%Y-%m-%d")

    def _init_key_usage(self, key_index: int) -> None:
        """Initialize or reset usage tracking for a key."""
        today = self._get_today()
        if key_index not in self.key_usage:
            self.key_usage[key_index] = {
                "daily_count": 0,
                "last_request_time": 0.0,
                "date": today,
            }
        # Reset daily count if it's a new day
        elif self.key_usage[key_index]["date"] != today:
            self.key_usage[key_index]["daily_count"] = 0
            self.key_usage[key_index]["date"] = today
            # Also remove from failed_keys if it was marked due to daily limit
            if key_index in self.failed_keys:
                self.failed_keys.discard(key_index)
                logger.info(f"API key {key_index + 1} reset for new day")

    def load_keys(self, keys_file: str) -> bool:
        """Load API keys from a text file (one key per line)."""
        try:
            with open(keys_file, "r", encoding="utf-8") as f:
                self.keys = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
            if not self.keys:
                logger.error(f"No API keys found in {keys_file}")
                return False
            logger.info(f"Loaded {len(self.keys)} API key(s)")
            logger.info(
                f"Rate limits: {self.REQUESTS_PER_MINUTE} req/min, {self.REQUESTS_PER_DAY} req/day per key"
            )
            logger.info(
                f"Request interval: {self.MIN_REQUEST_INTERVAL}s between requests"
            )
            # Initialize usage tracking for all keys
            for i in range(len(self.keys)):
                self._init_key_usage(i)
            return True
        except FileNotFoundError:
            logger.error(f"API keys file not found: {keys_file}")
            return False
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            return False

    def get_current_key(self) -> Optional[str]:
        """Get the current active API key."""
        # Check for new day and reset if needed
        for i in range(len(self.keys)):
            self._init_key_usage(i)

        available_keys = [
            k for i, k in enumerate(self.keys) if i not in self.failed_keys
        ]
        if not available_keys:
            return None
        # Find next available key
        while self.current_index in self.failed_keys:
            self.current_index = (self.current_index + 1) % len(self.keys)
            if all(i in self.failed_keys for i in range(len(self.keys))):
                return None
        return self.keys[self.current_index]

    def wait_for_rate_limit(self) -> None:
        """Wait if needed to respect rate limits (5 req/min = 12s between requests)."""
        self._init_key_usage(self.current_index)
        usage = self.key_usage[self.current_index]

        if usage["last_request_time"] > 0:
            elapsed = time.time() - usage["last_request_time"]
            if elapsed < self.MIN_REQUEST_INTERVAL:
                wait_time = self.MIN_REQUEST_INTERVAL - elapsed
                logger.info(
                    f"⏳ Rate limit: waiting {wait_time:.1f}s before next request..."
                )
                time.sleep(wait_time)

    def record_request(self) -> bool:
        """Record a request and check if daily limit is reached.
        Returns False if daily limit reached (key should be rotated).
        """
        self._init_key_usage(self.current_index)
        usage = self.key_usage[self.current_index]

        usage["daily_count"] += 1
        usage["last_request_time"] = time.time()

        remaining = self.REQUESTS_PER_DAY - usage["daily_count"]
        logger.info(
            f"📊 API key {self.current_index + 1}: {usage['daily_count']}/{self.REQUESTS_PER_DAY} requests today ({remaining} remaining)"
        )

        if usage["daily_count"] >= self.REQUESTS_PER_DAY:
            logger.warning(
                f"API key {self.current_index + 1} reached daily limit ({self.REQUESTS_PER_DAY} requests)"
            )
            return False
        return True

    def mark_current_failed(self, reason: str = "error") -> bool:
        """Mark current key as failed and rotate to next. Returns False if no keys left."""
        self.failed_keys.add(self.current_index)
        logger.warning(
            f"API key {self.current_index + 1} marked as failed ({reason}). Failed keys: {len(self.failed_keys)}/{len(self.keys)}"
        )

        if len(self.failed_keys) >= len(self.keys):
            logger.error("All API keys have been exhausted!")
            return False

        # Rotate to next available key
        self.current_index = (self.current_index + 1) % len(self.keys)
        while self.current_index in self.failed_keys:
            self.current_index = (self.current_index + 1) % len(self.keys)

        logger.info(f"🔄 Rotated to API key {self.current_index + 1}")
        return True

    def has_available_keys(self) -> bool:
        """Check if there are any available API keys."""
        # Check for new day reset first
        for i in range(len(self.keys)):
            self._init_key_usage(i)
        return len(self.failed_keys) < len(self.keys)

    def get_status_summary(self) -> str:
        """Get a summary of all API keys status."""
        lines = ["API Keys Status:"]
        for i in range(len(self.keys)):
            self._init_key_usage(i)
            usage = self.key_usage[i]
            status = (
                "❌ FAILED"
                if i in self.failed_keys
                else ("✅ CURRENT" if i == self.current_index else "⏸️ AVAILABLE")
            )
            lines.append(
                f"  Key {i + 1}: {usage['daily_count']}/{self.REQUESTS_PER_DAY} requests - {status}"
            )
        return "\n".join(lines)


# ============================================================================
# Helper Functions
# ============================================================================


def load_checkpoint(checkpoint_path: str) -> ProcessingCheckpoint:
    """Load checkpoint from file or create new one."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            checkpoint = ProcessingCheckpoint.from_dict(data)
            logger.info(
                f"Loaded checkpoint: {checkpoint.total_processed} items processed"
            )
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
    return ProcessingCheckpoint()


def save_checkpoint(checkpoint: ProcessingCheckpoint, checkpoint_path: str) -> None:
    """Save checkpoint to file."""
    checkpoint.last_updated = datetime.now().isoformat()
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)


def append_result(result: dict, output_path: str) -> None:
    """Append a single result to the output JSON file."""
    # Load existing results or create new list
    results = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except (json.JSONDecodeError, Exception):
            results = []

    results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def get_supported_files(input_dir: str) -> list:
    """Get all supported files (PDFs and images) from input directory."""
    supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    files = []

    input_path = Path(input_dir)
    if input_path.is_file():
        if input_path.suffix.lower() in supported_extensions:
            files.append(str(input_path))
    else:
        for ext in supported_extensions:
            files.extend([str(p) for p in input_path.rglob(f"*{ext}")])
            files.extend([str(p) for p in input_path.rglob(f"*{ext.upper()}")])

    # Remove duplicates and sort
    files = sorted(list(set(files)))
    logger.info(f"Found {len(files)} supported file(s)")
    return files


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_image(file_path: str) -> Image.Image:
    """Load an image file."""
    return Image.open(file_path)


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list:
    """Convert PDF pages to PIL Images."""
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        logger.info(f"Converted PDF '{pdf_path}' to {len(images)} page(s)")
        return images
    except Exception as e:
        logger.error(f"Failed to convert PDF '{pdf_path}': {e}")
        return []


def parse_gemini_response(response_text: str) -> Optional[dict]:
    """Parse JSON response from Gemini, handling markdown code blocks."""
    text = response_text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        logger.debug(f"Raw response: {response_text}")
        return None


def validate_and_normalize_result(result: dict) -> dict:
    """Validate and normalize extracted result to match expected schema."""
    normalized = {
        "metadata": [],
        "table_rows": [],
    }

    # Handle metadata
    if "metadata" in result:
        if isinstance(result["metadata"], list):
            normalized["metadata"] = result["metadata"]
        elif isinstance(result["metadata"], dict):
            normalized["metadata"] = [result["metadata"]]

    # Ensure metadata has all required fields
    for item in normalized["metadata"]:
        for field_name in METADATA_FIELDS:
            if field_name not in item:
                item[field_name] = None

    # Handle table_rows
    if "table_rows" in result:
        if isinstance(result["table_rows"], list):
            normalized["table_rows"] = result["table_rows"]
        elif isinstance(result["table_rows"], dict):
            normalized["table_rows"] = [result["table_rows"]]

    # Ensure table_rows have all required fields
    for item in normalized["table_rows"]:
        for field_name in TABLE_ROW_FIELDS:
            if field_name not in item:
                item[field_name] = None

    return normalized


# ============================================================================
# Main Processing
# ============================================================================


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
            # Wait for rate limit before making request
            key_manager.wait_for_rate_limit()

            response = model.generate_content([prompt, image])

            # Record the request for rate limiting
            if not key_manager.record_request():
                # Daily limit reached, trigger key rotation
                raise Exception("Daily request limit reached for this API key")

            if response.text:
                result = parse_gemini_response(response.text)
                if result:
                    return validate_and_normalize_result(result)

            logger.warning(f"Empty or invalid response on attempt {attempt + 1}")

        except Exception as e:
            error_msg = str(e).lower()

            # Check for quota/rate limit errors
            if (
                "quota" in error_msg
                or "rate" in error_msg
                or "limit" in error_msg
                or "daily" in error_msg
            ):
                logger.warning(f"Rate/quota limit hit: {e}")
                raise  # Re-raise to trigger key rotation

            # Check for invalid API key
            if (
                "api key" in error_msg
                or "invalid" in error_msg
                or "unauthorized" in error_msg
            ):
                logger.warning(f"API key error: {e}")
                raise  # Re-raise to trigger key rotation

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
                model = genai.GenerativeModel("gemini-3-flash-preview")

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
