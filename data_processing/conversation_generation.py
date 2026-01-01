# -*- coding: utf-8 -*-
"""
Conversation Generation Pipeline using Gemini API.

This pipeline processes PDF files and images, generates multi-turn Q&A conversations
about the content using Gemini Vision API, and saves results to JSON with checkpoint support.

Usage:
    python data_processing/conversation_generation.py --input ./data --output ./output/conversations.json --api-keys ./api_keys.txt
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("conversation_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def get_conversation_prompt() -> str:
    """Generate prompt for creating multi-turn Q&A conversations about document images."""
    prompt = """You are an expert in analyzing Vietnamese business documents and creating natural, realistic conversations about their content.

TASK:
Analyze the provided document image and generate a realistic multi-turn conversation (3-6 turns) between a user asking questions and an AI assistant answering based on the document's content.

CORE PRINCIPLES:

1. **Naturalness & Diversity**
   - Questions should sound like real users asking about real documents
   - Vary question complexity, structure, and focus areas
   - Avoid repetitive patterns or formulaic phrasing
   - Mix formal and casual Vietnamese styles naturally

2. **Question Type Variety** (distribute across conversations, not all in one):
   - Direct information retrieval (dates, codes, names, addresses)
   - Quantitative queries (counts, totals, amounts)
   - Comparative analysis (which has more/less, differences)
   - Existence verification (is X present? does it contain Y?)
   - Relational reasoning (connections between fields, implications)
   - Clarification requests (units, specific details, confirmations)

3. **Conversation Flow**
   - Questions should build logically but unpredictably
   - Later questions can reference earlier context
   - Include both broad and narrow scopes
   - Mix simple lookups with analytical questions
   - Natural transitions between topics

4. **Answer Quality**
   - Base all answers STRICTLY on visible document information
   - Be precise with numbers, codes, and names
   - If information is unclear or absent, acknowledge it naturally
   - Use conversational but professional Vietnamese
   - Provide complete, helpful responses without over-explaining

IMPORTANT GUIDELINES:

✓ DO:
- Create unique, context-specific questions for each document type
- Use natural Vietnamese with proper diacritics
- Vary question lengths and complexity
- Include domain-specific terminology where appropriate
- Make questions that someone would genuinely ask
- Allow natural conversation flow with occasional tangents

✗ DON'T:
- Use the same question templates repeatedly
- Ask questions in predictable orders
- Hallucinate information not in the document
- Over-explain or be overly verbose
- Use unnatural or robotic language
- Force all question types into every conversation

OUTPUT FORMAT:
Return a JSON object with this structure:

{
  "conversation": [
    {
      "turn": 1,
      "user": "<Natural Vietnamese question>",
      "assistant": "<Accurate answer from document>"
    },
    {
      "turn": 2,
      "user": "<Follow-up or new question>",
      "assistant": "<Answer>"
    }
    // 3-6 turns total
  ]
}

QUALITY CHECKLIST:
- [ ] 3-6 conversation turns
- [ ] Questions feel natural and varied
- [ ] All answers are accurate and document-based
- [ ] Proper Vietnamese language throughout
- [ ] Logical but non-formulaic conversation flow
- [ ] No information fabrication
- [ ] Valid JSON format
- [ ] Questions specific to this document's context

Generate a high-quality, realistic conversation now:"""

    return prompt


def validate_conversation_result(result: dict) -> dict:
    """Validate and normalize conversation generation result."""
    normalized = {
        "document_summary": result.get("document_summary", ""),
        "conversation": [],
        "key_information_covered": result.get("key_information_covered", []),
    }

    if "conversation" in result and isinstance(result["conversation"], list):
        for idx, turn in enumerate(result["conversation"], start=1):
            normalized_turn = {
                "turn": turn.get("turn", idx),
                "user": turn.get("user", ""),
                "assistant": turn.get("assistant", ""),
            }
            normalized["conversation"].append(normalized_turn)

    return normalized


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
                result = parse_gemini_response(response.text)
                if result:
                    return validate_conversation_result(result)

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
    prompt: str,
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

    if file_ext == ".pdf":
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

            result = generate_conversation_from_image(image, model, prompt, key_manager)

            if result:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "pdf",
                    "page_number": page_num,
                    "total_pages": len(images),
                    "generated_at": datetime.now().isoformat(),
                    "document_summary": result.get("document_summary", ""),
                    "conversation": result.get("conversation", []),
                    "key_information_covered": result.get("key_information_covered", []),
                }
                append_result(output_entry, output_path)
                logger.info(f"  ✓ Generated conversation for page {page_num} ({len(result.get('conversation', []))} turns)")
            else:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "pdf",
                    "page_number": page_num,
                    "total_pages": len(images),
                    "generated_at": datetime.now().isoformat(),
                    "document_summary": "",
                    "conversation": [],
                    "key_information_covered": [],
                    "error": "Failed to generate conversation",
                }
                append_result(output_entry, output_path)
                logger.warning(f"  ✗ Failed to generate conversation for page {page_num}")

            file_progress["pages"].append(page_num)
            checkpoint.total_processed += 1
            save_checkpoint(checkpoint, checkpoint_path)

        file_progress["completed"] = True
        save_checkpoint(checkpoint, checkpoint_path)

    else:
        try:
            image = load_image(file_path)

            result = generate_conversation_from_image(image, model, prompt, key_manager)

            if result:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "image",
                    "page_number": 1,
                    "total_pages": 1,
                    "generated_at": datetime.now().isoformat(),
                    "document_summary": result.get("document_summary", ""),
                    "conversation": result.get("conversation", []),
                    "key_information_covered": result.get("key_information_covered", []),
                }
                append_result(output_entry, output_path)
                logger.info(f"  ✓ Generated conversation ({len(result.get('conversation', []))} turns)")
            else:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "image",
                    "page_number": 1,
                    "total_pages": 1,
                    "generated_at": datetime.now().isoformat(),
                    "document_summary": "",
                    "conversation": [],
                    "key_information_covered": [],
                    "error": "Failed to generate conversation",
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

    prompt = get_conversation_prompt()

    total_max_requests = len(key_manager.keys) * key_manager.REQUESTS_PER_DAY
    logger.info("=" * 60)
    logger.info("Starting conversation generation pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Files to process: {len(files)}")
    logger.info(f"API keys: {len(key_manager.keys)} keys × {key_manager.REQUESTS_PER_DAY} req/day = {total_max_requests} max requests/day")
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
                model = genai.GenerativeModel("gemini-2.0-flash-exp")

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
                    break

            except Exception as e:
                error_msg = str(e).lower()

                if "daily" in error_msg:
                    reason = "daily_limit"
                elif "quota" in error_msg or "rate" in error_msg or "limit" in error_msg:
                    reason = "rate_limit"
                elif "api key" in error_msg or "invalid" in error_msg or "unauthorized" in error_msg or "forbidden" in error_msg:
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