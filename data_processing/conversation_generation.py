# -*- coding: utf-8 -*-
"""
Conversation Generation Pipeline using Gemini API.

This pipeline processes PDF files and images, generates multi-turn Q&A conversations
about the content using Gemini Vision API, and saves results to JSON with checkpoint support.

Usage:
    python data_processing/conversation_generation.py --input ./data --output ./output/conversations.json --api-keys ./api_keys.txt
"""

import argparse
import ast
import json
import logging
import random
import re
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
    ProcessingCheckpoint,
    append_result,
    get_conversation_prompt,
    get_lazy_user_prompt,
    get_reasoning_prompt,
    get_supported_files,
    get_visual_spatial_correction_prompt,
    load_checkpoint,
    load_image,
    pdf_to_images,
    save_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("conversation_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Set the output language for generated conversations
# Options: "vi" (Vietnamese), "en" (English), "zh" (Chinese), etc.
OUTPUT_LANGUAGE = "japanese"  # Change this to your desired language
# ----------------------

# --- PROMPT INJECTION TO FIX JSON FORMATTING ---
# We append strict formatting rules to whatever prompt comes from utils
STRICT_JSON_INSTRUCTION = """
---
IMPORTANT OUTPUT RULES:
1. Return ONLY valid JSON. No markdown fencing (```json) inside string values.
2. CRITICAL: Do NOT use actual line breaks (newlines) inside JSON string values. Use the literal characters '\\n' for line breaks.
   - WRONG: "text": "Line 1
     Line 2"
   - CORRECT: "text": "Line 1\\nLine 2"
3. Ensure all quotes inside strings are escaped properly.
"""

PROMPT_EXTRACTION = (
    get_conversation_prompt(output_language=OUTPUT_LANGUAGE) + STRICT_JSON_INSTRUCTION
)
PROMPT_REASONING = (
    get_reasoning_prompt(output_language=OUTPUT_LANGUAGE) + STRICT_JSON_INSTRUCTION
)
PROMPT_LAZY_USER = (
    get_lazy_user_prompt(output_language=OUTPUT_LANGUAGE) + STRICT_JSON_INSTRUCTION
)
PROMPT_VISUAL_SPATIAL_CORRECTION = (
    get_visual_spatial_correction_prompt(output_language=OUTPUT_LANGUAGE)
    + STRICT_JSON_INSTRUCTION
)
# -----------------------------------------------


def robust_parse_json(text: str) -> Optional[dict]:
    """
    V2: A much more robust JSON parser that handles:
    1. Nested Markdown blocks
    2. Unescaped newlines (common LLM reasoning error)
    3. Trailing commas or Python-style dicts
    """
    # 1. Cleaning: Find the outer braces
    start_idx = text.find("{")
    end_idx = text.rfind("}")

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None

    json_str = text[start_idx : end_idx + 1]

    # 2. Strategy A: Standard JSON Load
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass  # Fallthrough to Strategy B

    # 3. Strategy B: Python AST Eval (The "Nuclear Option")
    # Python's literal_eval is more forgiving with newlines in strings than strict JSON
    try:
        # We need to map JSON keywords to Python keywords for AST to work
        # This regex replaces true/false/null with True/False/None outside of strings is risky,
        # so we do a simple replace and hope it doesn't hit content words.
        # A safer way is using replace on the exact tokens if we had a tokenizer, but this usually works for LLM output.

        # Simple fix for common JSON vs Python differences
        py_str = (
            json_str.replace("true", "True")
            .replace("false", "False")
            .replace("null", "None")
        )

        # AST handles unescaped newlines inside strings automatically!
        parsed = ast.literal_eval(py_str)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass

    # 4. Strategy C: Desperate Cleaning (Regex)
    # If both above fail, try to remove control characters that might be breaking it
    try:
        # Remove raw newlines inside the string (dangerous but might save a result)
        # This collapses lines, but keeps the JSON valid structure
        clean_str = re.sub(r"(?<!\\)\n", "\\n", json_str)
        return json.loads(clean_str)
    except Exception:
        pass

    return None


def validate_conversation_result(result: dict) -> dict:
    """Validate and normalize conversation generation result."""
    conversation = []
    if "conversation" in result:
        # Ensure it's a list
        if not isinstance(result["conversation"], list):
            return []

        for turn in result["conversation"]:
            # FIX: Check if turn is actually a dictionary to prevent 'str' has no attribute 'get' error
            if not isinstance(turn, dict):
                logger.warning(
                    f"  Skipping malformed turn (not a dict): {str(turn)[:50]}..."
                )
                continue

            assistant_content = turn.get("assistant", "")

            if isinstance(assistant_content, (dict, list)):
                assistant_content = json.dumps(assistant_content, ensure_ascii=False)
            elif not isinstance(assistant_content, str):
                assistant_content = str(assistant_content)

            normalized_turn = {
                "user": turn.get("user", ""),
                "assistant": assistant_content,
            }
            conversation.append(normalized_turn)

    return conversation


def get_weighted_prompt():
    """Select a prompt based on defined weights."""
    prompts = [
        PROMPT_EXTRACTION,
        PROMPT_REASONING,
        PROMPT_LAZY_USER,
        PROMPT_VISUAL_SPATIAL_CORRECTION,
    ]
    # Tỷ lệ: Extraction 30%, Reasoning 40%, Lazy User 10%, Spatial 20%
    weights = [0.3, 0.4, 0.1, 0.2]
    selected_prompt = random.choices(prompts, weights=weights, k=1)[0]
    return selected_prompt


def get_prompt_type_name(prompt_content: str) -> str:
    """Helper to get string name of the prompt for logging/saving."""
    if prompt_content == PROMPT_REASONING:
        return "reasoning"
    if prompt_content == PROMPT_LAZY_USER:
        return "lazy_user"
    if prompt_content == PROMPT_VISUAL_SPATIAL_CORRECTION:
        return "visual_spatial_correction"
    return "extraction"


def generate_conversation_from_image(
    image: Image.Image,
    model: genai.GenerativeModel,
    prompt: str,
    key_manager: APIKeyManager,
    max_retries: int = 3,
) -> Optional[dict]:
    """Generate conversation data from a single image using Gemini."""
    for attempt in range(max_retries):
        try:
            key_manager.wait_for_rate_limit()

            response = model.generate_content([prompt, image])

            if not key_manager.record_request():
                raise Exception("Daily request limit reached for this API key")

            if response.text:
                # USE NEW ROBUST PARSER V2
                result = robust_parse_json(response.text)

                if result:
                    validated = validate_conversation_result(result)
                    if validated:
                        return validated
                    else:
                        logger.warning(
                            f"Parsed JSON but validation returned empty on attempt {attempt + 1}"
                        )
                else:
                    logger.warning(f"Failed to parse JSON on attempt {attempt + 1}")
                    # Optional: Log the start of the failing text for debugging
                    # logger.info(f"Failing Response Snippet: {response.text[:200]}...")

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

            if (
                "api key" in error_msg
                or "invalid" in error_msg
                or "unauthorized" in error_msg
            ):
                logger.warning(f"API key error: {e}")
                raise

            logger.warning(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(2**attempt)

    return None


def process_file(
    file_path: str,
    model: genai.GenerativeModel,
    checkpoint: ProcessingCheckpoint,
    output_path: str,
    checkpoint_path: str,
    key_manager: APIKeyManager,
) -> bool:
    """Process a single file (PDF or image). Returns False if API keys exhausted."""
    file_ext = Path(file_path).suffix.lower()
    file_name = Path(file_path).name

    if file_path not in checkpoint.processed_files:
        checkpoint.processed_files[file_path] = {"pages": [], "completed": False}

    file_progress = checkpoint.processed_files[file_path]

    if file_progress["completed"]:
        logger.info(f"Skipping already completed file: {file_name}")
        return True

    logger.info(f"Processing: {file_name}")

    # --- RANDOM PROMPT SELECTION ---
    current_prompt = get_weighted_prompt()
    prompt_type_str = get_prompt_type_name(current_prompt)

    # --- LOGGING THE CHOSEN PROMPT ---
    logger.info(f"  [PROMPT SELECTED] Type: {prompt_type_str.upper()}")
    # -------------------------------

    if file_ext == ".pdf":
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

            result = generate_conversation_from_image(
                image, model, current_prompt, key_manager
            )

            if result:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "pdf",
                    "page_number": page_num,
                    "total_pages": len(images),
                    "generated_at": datetime.now().isoformat(),
                    "conversation": result,
                    "prompt_type": prompt_type_str,
                }
                append_result(output_entry, output_path)
                logger.info(
                    f"  ✓ Generated conversation for page {page_num} ({len(result)} turns)"
                )
            else:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "pdf",
                    "page_number": page_num,
                    "total_pages": len(images),
                    "generated_at": datetime.now().isoformat(),
                    "conversation": [],
                    "error": "Failed to generate conversation",
                    "prompt_type": prompt_type_str,
                }
                append_result(output_entry, output_path)
                logger.warning(
                    f"  ✗ Failed to generate conversation for page {page_num}"
                )

            file_progress["pages"].append(page_num)
            checkpoint.total_processed += 1
            save_checkpoint(checkpoint, checkpoint_path)

        file_progress["completed"] = True
        save_checkpoint(checkpoint, checkpoint_path)

    else:
        # IMAGE PROCESSING
        try:
            image = load_image(file_path)

            result = generate_conversation_from_image(
                image, model, current_prompt, key_manager
            )

            if result:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "image",
                    "page_number": 1,
                    "total_pages": 1,
                    "generated_at": datetime.now().isoformat(),
                    "conversation": result,
                    "prompt_type": prompt_type_str,
                }
                append_result(output_entry, output_path)
                logger.info(f"  ✓ Generated conversation ({len(result)} turns)")
            else:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "image",
                    "page_number": 1,
                    "total_pages": 1,
                    "generated_at": datetime.now().isoformat(),
                    "conversation": [],
                    "error": "Failed to generate conversation",
                    "prompt_type": prompt_type_str,
                }
                append_result(output_entry, output_path)
                logger.warning("  ✗ Failed to generate conversation")

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
    """Run the conversation generation pipeline."""

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = str(output_dir / "conversation_checkpoint.json")

    key_manager = APIKeyManager()
    if not key_manager.load_keys(api_keys_file):
        logger.error("Failed to load API keys. Exiting.")
        sys.exit(1)

    checkpoint = load_checkpoint(checkpoint_path)

    files = get_supported_files(input_path)
    if not files:
        logger.error(f"No supported files found in: {input_path}")
        sys.exit(1)

    total_max_requests = len(key_manager.keys) * key_manager.REQUESTS_PER_DAY
    logger.info("=" * 60)
    logger.info("Starting conversation generation pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Files to process: {len(files)}")
    logger.info(
        f"API keys: {len(key_manager.keys)} keys × {key_manager.REQUESTS_PER_DAY} req/day = {total_max_requests} max requests/day"
    )
    logger.info("=" * 60)
    logger.info(key_manager.get_status_summary())
    logger.info("=" * 60)

    for file_idx, file_path in enumerate(files, start=1):
        logger.info(f"\n[{file_idx}/{len(files)}] {Path(file_path).name}")

        while key_manager.has_available_keys():
            current_key = key_manager.get_current_key()
            if not current_key:
                logger.error("No available API keys!")
                sys.exit(1)

            try:
                genai.configure(api_key=current_key)
                generation_config = genai.GenerationConfig(
                    temperature=1.0,
                )
                model = genai.GenerativeModel(
                    "gemini-3-flash-preview", generation_config=generation_config
                )

                success = process_file(
                    file_path=file_path,
                    model=model,
                    checkpoint=checkpoint,
                    output_path=output_path,
                    checkpoint_path=checkpoint_path,
                    key_manager=key_manager,
                )

                if success:
                    break

            except Exception as e:
                error_msg = str(e).lower()

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
    logger.info(f"Total conversations generated: {checkpoint.total_processed}")
    logger.info(f"Output saved to: {output_path}")
    logger.info("=" * 60)
    logger.info(key_manager.get_status_summary())
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate multi-turn Q&A conversations from document images using Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python conversation_generation_pipeline.py --input ./data --output ./output/conversations.json --api-keys ./api_keys.txt
  python conversation_generation_pipeline.py -i ./documents -o ./conversations.json -k ./keys.txt
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
        help="Output JSON file path for conversations",
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
