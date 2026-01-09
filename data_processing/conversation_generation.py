# -*- coding: utf-8 -*-
"""
Conversation Generation Pipeline using Gemini API.

This pipeline processes PDF files and images, generates multi-turn Q&A conversations
about the content using Gemini Vision API, and saves results to JSON with checkpoint support.

Usage:
    python data_processing/conversation_generation.py --input ./data --output ./output/conversations.json --api-keys ./api_keys.txt
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
    """Generate diverse format conversations for VLM training."""
    prompt = """You are an expert in analyzing Vietnamese business documents and creating natural, realistic conversations with DIVERSE OUTPUT FORMATS.

TASK:
Analyze the provided document image and generate a realistic multi-turn conversation (4-8 turns) between a user asking questions and an AI assistant answering in VARIOUS FORMATS.

CORE PRINCIPLES:

1. **Format Diversity** (CRITICAL for VLM training):
   
   Mix these formats naturally across conversations:
   
   a) **Natural Text Response** (25-30%):
      - User: "Tổng số lượng là bao nhiêu?"
      - Assistant: "Tổng cộng có 11.100 sản phẩm trong phiếu này."
   
   b) **JSON Format** (20-25%):
      - User: "Lấy thông tin nhà cung cấp dưới dạng JSON nhé"
      - Assistant: {"ten": "KOIKEYA", "sdt": "+84251368", "email": null}
      - MUST be raw JSON, no markdown code blocks
   
   c) **Markdown Tables** (15-20%):
      - User: "Liệt kê sản phẩm dưới dạng bảng"
      - Assistant:
        | STT | Tên sản phẩm | Số lượng |
        |-----|-------------|----------|
        | 1   | Snack A     | 100      |
   
   d) **Structured Key-Value Pairs** (10-15%):
      - User: "Cho tôi thông tin tóm tắt về phiếu"
      - Assistant:
        Số phiếu: PXB25110053
        Ngày: 03/11/2025
        Nhà cung cấp: KOIKEYA VIETNAM
        Tổng số lượng: 11,100 pcs
   
   e) **Summary + Detail Format** (8-12%):
      - User: "Tóm tắt phiếu này"
      - Assistant:
        **Tóm tắt:** Phiếu giao hàng có 8 loại snack, tổng 11,100 sản phẩm.
        
        Chi tiết:
        - Snack Bắp (4 loại): 6,000 pcs
          • Karamucho vị cay: 2,400 pcs (cao nhất)
          • Karamucho phô mai: 600 pcs
        - Snack Khoai tây (4 loại): 5,100 pcs
   
   f) **Comparison / Side-by-side** (5-10%):
      - User: "So sánh số lượng 3 sản phẩm đầu"
      - Assistant:
        Snack Bắp (500VS29):    2,400 pcs
        Snack Bắp (517VS29):      600 pcs  
        Snack Bắp (519VS29):      600 pcs
        
        → Sản phẩm 500VS29 có số lượng cao nhất (gấp 4 lần)
   
   g) **Markdown Lists** (5-8%):
      - User: "Các mặt hàng có những gì?"
      - Assistant:
        1. Snack Bắp Karamucho - vị cay (2,400 pcs)
        2. Snack Bắp Karamucho - vị phô mai (600 pcs)
        3. Snack khoai tây - vị cay (1,500 pcs)
   
   h) **Hierarchical Nested Text** (3-5%):
      - User: "Tổng quan cấu trúc phiếu"
      - Assistant:
        📋 PHIẾU GIAO HÀNG PXB25110053
        ├─ Thông tin chung
        │  ├─ Ngày: 03/11/2025
        │  └─ Nhà cung cấp: KOIKEYA
        │
        ├─ Sản phẩm (8 loại)
        │  ├─ Snack Bắp: 6,000 pcs
        │  └─ Snack Khoai tây: 5,100 pcs
        │
        └─ Tổng cộng: 11,100 pcs
   
   i) **CSV Format** (2-5%):
      - User: "Export danh sách CSV"
      - Assistant:
        STT,Mã số,Tên hàng hóa,Số lượng
        1,500VS29,"Snack Bắp Karamucho",2400
        2,517VS29,"Snack Bắp phô mai",600
   
   j) **Q&A / FAQ Style** (2-5%):
      - User: "Cho tôi biết: số phiếu? ngày? email? tổng?"
      - Assistant:
        Q: Số phiếu là gì?
        A: PXB25110053
        
        Q: Ngày giao hàng?
        A: 03/11/2025
        
        Q: Có email không?
        A: Không có thông tin email
        
        Q: Tổng số lượng?
        A: 11,100 pcs

2. **Format Mixing Strategies**:
   
   Each conversation should use 2-4 different formats naturally:
   
   **Strategy A: Progressive Enhancement** (30%):
```
   Turn 1: Text answer
   Turn 2: "Làm thành bảng markdown" → table
   Turn 3: "Chuyển sang JSON" → JSON
   Turn 4: "So sánh top 3" → comparison
```
   
   **Strategy B: Format-First** (25%):
```
   Turn 1: "Lấy tất cả thông tin JSON" → full JSON
   Turn 2: "Có email không?" → null handling
   Turn 3: "Liệt kê sản phẩm dạng bảng" → markdown table
   Turn 4: "Tóm tắt lại" → summary+detail
```
   
   **Strategy C: Mixed Query** (25%):
```
   Turn 1: "Tổng số lượng?" → text
   Turn 2: "Liệt kê sản phẩm" → list
   Turn 3: "So sánh top 3" → comparison
   Turn 4: "Export CSV" → CSV
```
   
   **Strategy D: Deep Dive** (20%):
```
   Turn 1: "Tóm tắt phiếu" → summary
   Turn 2: "Chi tiết sản phẩm dạng bảng" → table
   Turn 3: "Thông tin nhà cung cấp key-value" → structured
   Turn 4: "Có những trường nào null?" → null checking
   Turn 5: "Chuyển sang JSON hoàn chỉnh" → JSON
```

3. **Negative Questions & Null Handling** (20-30% of conversations):
   
   Include questions about NON-EXISTENT fields in various formats:
   
   Text format:
   - "Email của nhà cung cấp là gì?" → "Không có thông tin email trong phiếu"
   - "Có ghi chú về phí vận chuyển không?" → "Tài liệu không đề cập"
   
   JSON format:
   - "Lấy email và fax JSON" → {"email": null, "fax": "+84251368"}
   
   Key-value format:
   - Email: Không có
   - Fax: +84(0)251 368 1369
   
   In tables:
   - | Field | Value |
     | Email | Không có |

4. **Question Type Variety**:
   
   **Basic Extraction** (25-30%):
   - "Số phiếu là gì?"
   - "Lấy thông tin nhà cung cấp"
   
   **Format-specific requests** (25-30%):
   - "Cho tôi dưới dạng JSON"
   - "Làm bảng markdown"
   - "Export CSV"
   - "Tóm tắt có cấu trúc"
   
   **Arithmetic & Comparison** (15-20%):
   - "Tổng số lượng?"
   - "So sánh 3 sản phẩm"
   - "Sản phẩm nào nhiều nhất?"
   
   **Transformation** (10-15%):
   - "Đổi sang format khác"
   - "Chuyển sang bảng"
   - "Format lại dạng key-value"
   
   **Verification & Negative** (10-15%):
   - "Có email không?"
   - "Những trường nào bị thiếu?"

5. **Answer Quality by Format**:
   
   **Text responses:**
   - Natural Vietnamese, concise
   - "Tổng cộng có 11,100 sản phẩm"
   
   **JSON responses:**
   - Raw JSON, no code blocks
   - Explicit null for missing
   - Consistent naming
   
   **Markdown tables:**
   - Proper alignment
   - Vietnamese headers
   - Include units
   
   **Key-Value:**
   - Clear labels
   - Aligned colons
   - One entry per line
   
   **Summary+Detail:**
   - Bold summary
   - Indented details
   - Bullet points for sub-items
   
   **Comparison:**
   - Right-aligned numbers
   - Clear spacing
   - Include conclusion
   
   **CSV:**
   - Proper escaping
   - Quoted strings with commas
   - Consistent delimiters
   
   **Hierarchical:**
   - Tree structure symbols
   - Clear indentation
   - Emoji/icons optional
   
   **Q&A:**
   - Clear Q: and A: labels
   - One line break between pairs
   - Concise answers

CRITICAL RULES:

**JSON Format:**
- Return RAW JSON without ```json ... ``` blocks
- Use null for missing fields, not "null" string
- Properly escape if nested in outer JSON
- Keep numbers as numbers

**Markdown:**
- Tables must have proper | alignment
- Lists use proper numbering or bullets
- No mixing tabs and spaces

**CSV:**
- Escape commas in values with quotes
- Consistent encoding (UTF-8)
- Header row when appropriate

**Null Handling:**
- Be explicit: "Không có", null, "N/A"
- Consistent across formats
- Don't make up data

FORMAT DISTRIBUTION (target across dataset):
- Natural text:        25-30%
- JSON:                20-25%
- Markdown tables:     15-20%
- Key-Value pairs:     10-15%
- Summary+Detail:       8-12%
- Comparison:           5-10%
- Markdown lists:       5-8%
- Hierarchical:         3-5%
- CSV:                  2-5%
- Q&A style:            2-5%

CONVERSATION LENGTH:
- 4-5 turns: Simple documents, 2-3 formats
- 6-7 turns: Medium complexity, 3-4 formats
- 8 turns: Complex documents, 4-5 formats

Each conversation should feel natural - don't force all formats into one conversation.

OUTPUT FORMAT:
{
  "conversation": [
    {
      "user": "<Natural question with/without format request>",
      "assistant": "<Answer in appropriate format>"
    }
    // 4-8 turns
  ]
}

QUALITY CHECKLIST:
- [ ] 4-8 conversation turns (adjust based on document complexity)
- [ ] 2-4 different formats used naturally
- [ ] 20-30% chance: includes null/negative questions
- [ ] All formats properly structured
- [ ] JSON is raw (no code blocks)
- [ ] Natural format transitions
- [ ] All answers accurate
- [ ] Proper Vietnamese
- [ ] Valid outer JSON

EXAMPLES:

Example 1 - Progressive (6 turns):
{
  "conversation": [
    {"user": "Số phiếu và ngày?", 
    "assistant": "Số phiếu: PXB25110053, Ngày: 03/11/2025"},
    {"user": "Chuyển sang JSON nhé",
    "assistant": "{\"so_phieu\": \"PXB25110053\", \"ngay\": \"03/11/2025\"}"},
    {"user": "Thêm email nhà cung cấp",
     "assistant": "{\"so_phieu\": \"PXB25110053\", \"ngay\": \"03/11/2025\", \"email\": null}"},
    {"user": "Liệt kê 3 sản phẩm đầu dạng bảng",
     "assistant": "| STT | Tên | Số lượng |\n|-----|-----|----------|\n| 1 | Snack A | 2400 |"},
    {"user": "So sánh số lượng của chúng",
     "assistant": "Snack A: 2,400 pcs\nSnack B:   600 pcs\nSnack C:   600 pcs\n\n→ Snack A cao nhất (gấp 4 lần)"},
    {"user": "Export CSV tất cả sản phẩm",
     "assistant": "STT,Mã,Tên,Số lượng\n1,500VS29,\"Snack A\",2400\n2,517VS29,\"Snack B\",600"}
  ]
}

Example 2 - Mixed (5 turns):
{
  "conversation": [
    {"user": "Tóm tắt phiếu này",
     "assistant": "**Tóm tắt:** Phiếu PXB25110053 ngày 03/11/2025, có 8 loại snack.\n\nChi tiết:\n- Snack Bắp: 6,000 pcs\n- Snack Khoai tây: 5,100 pcs"},
    {"user": "Thông tin nhà cung cấp dạng key-value",
     "assistant": "Tên: KOIKEYA VIETNAM\nĐịa chỉ: N3-2 St., Long Duc IP\nĐiện thoại: +84(0)251 368 1368\nEmail: Không có"},
    {"user": "Cấu trúc tổng quan",
     "assistant": "📋 PHIẾU PXB25110053\n├─ Ngày: 03/11/2025\n├─ NCC: KOIKEYA\n├─ Sản phẩm: 8 loại\n└─ Tổng: 11,100 pcs"},
    {"user": "Chi tiết sản phẩm JSON",
     "assistant": "{\"products\": [{\"ma\": \"500VS29\", \"qty\": 2400}, {\"ma\": \"517VS29\", \"qty\": 600}]}"},
    {"user": "Có những field nào thiếu?",
     "assistant": "Q: Email nhà cung cấp?\nA: Không có\n\nQ: Ngày sản xuất?\nA: Không ghi\n\nQ: Ghi chú?\nA: Để trống"}
  ]
}
"""
    return prompt

def validate_conversation_result(result: dict) -> dict:
    """Validate and normalize conversation generation result."""
    conversation = []
    if "conversation" in result:
        for turn in result["conversation"]:
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
                    "conversation": result,
                }
                append_result(output_entry, output_path)
                logger.info(f"  ✓ Generated conversation for page {page_num} ({len(result)} turns)")
            else:
                output_entry = {
                    "source_file": file_path,
                    "source_type": "pdf",
                    "page_number": page_num,
                    "total_pages": len(images),
                    "generated_at": datetime.now().isoformat(),
                    "conversation": [],
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
                    "conversation": result,
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