# -*- coding: utf-8 -*-
"""
Synthetic Data Generation Pipeline using Local Model (Pass 1) and Gemini (Pass 2).

This pipeline:
1. Uses a local model to generate multi-turn conversations (2-3 turns)
2. Uses Gemini to answer those prompts and create ground truth data
3. Outputs in ShareGPT format (user/assistant multi-turn conversation)

Usage:
    python synthetic_data_pipeline.py --input ./data --output ./output/synthetic_sharegpt.json --api-keys ./api_keys.txt
"""

import argparse
import logging
import sys
import time
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from data_processing.apikey_manager import APIKeyManager
from data_processing.utils import (
    load_image, 
    pdf_to_images, 
    save_checkpoint, 
    load_checkpoint, 
    get_supported_files,
    ProcessingCheckpoint
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("synthetic_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


PASS1_PROMPT = """Bạn là một người dùng thực tế đang chat với AI để trích xuất dữ liệu từ tài liệu.

BƯỚC 1: QUAN SÁT TÀI LIỆU
Xem kỹ tài liệu và xác định:
- Các trường thông tin CÓ SẴN trong tài liệu (số phiếu, ngày, tên kho, sản phẩm, số lượng...)
- Các trường PHÙ HỢP với loại tài liệu này nhưng KHÔNG CÓ (email, phone, địa chỉ, fax...)

BƯỚC 2: TẠO HỘI THOẠI
Tạo 2-3 câu hỏi liên tiếp như đang chat (chỉ có câu hỏi của user, không có AI response):

**Câu 1** - Hỏi ban đầu:
- Chọn 2-3 trường CÓ SẴN trong tài liệu
- Thêm 1 trường KHÔNG CÓ nhưng hỏi tự nhiên (đừng nói "nếu có", "dù không có")
- Yêu cầu JSON format

**Câu 2** - Hỏi tiếp:
- Chọn các trường khác từ tài liệu (chi tiết sản phẩm, bảng số liệu...)
- Hoặc yêu cầu sửa/bổ sung
- Nói tự nhiên: "à thêm...", "cho thêm...", "còn..."

**Câu 3** - Tùy chọn (50% có):
- Hỏi thêm hoặc điều chỉnh format
- Hoặc tổng hợp

PHONG CÁCH:
✅ Chat tự nhiên: "cho mình...", "lấy...", "à còn..."
✅ Ngắn gọn, đời thường
✅ Việt/Anh/mix thoải mái
✅ Tên trường tùy ý dựa trên nội dung thực tế
✅ Hỏi field không có như field bình thường

❌ TRÁNH: "vui lòng", "thực hiện", "xin hãy"
❌ TRÁNH: Từ lạ, typo
❌ TRÁNH: Dùng các trường KHÔNG liên quan đến loại tài liệu này

VÍ DỤ FLOW (KHÔNG copy y nguyên):

*Nếu là phiếu xuất kho:*
- Có thể hỏi: số phiếu, ngày, kho, sản phẩm, số lượng, đơn vị...
- Không có nhưng hợp lý: email kho, phone, địa chỉ giao hàng...

*Nếu là hóa đơn:*
- Có thể hỏi: số HĐ, ngày, khách hàng, tổng tiền, thuế...
- Không có nhưng hợp lý: email KH, fax, website...

*Nếu là biên lai:*
- Có thể hỏi: số biên lai, người nộp, số tiền, nội dung...
- Không có nhưng hợp lý: CMND, địa chỉ, phone...

PHONG CÁCH NÓI VÀ TÊN TRƯỜNG:

**Ví dụ casual:**
"Lấy [tên trường từ doc] và [tên trường không có] cho mình. JSON nhé"
"À cho thêm [chi tiết từ bảng]"

**Ví dụ mix:**
"Get [field 1], [field 2] and [field không có]. JSON format"
"Thêm [data từ bảng] vào"

**Ví dụ 3 turns:**
"Cho mình [field 1], [field 2] và [field không có] dạng JSON"
"Thêm [chi tiết] vào"
"Đổi format [field] sang [format khác]"

OUTPUT:
```json
{
  "turns": [
    "câu hỏi 1 - dựa trên nội dung thực tế",
    "câu hỏi 2 - dựa trên nội dung thực tế",
    "câu hỏi 3 (có thể bỏ)"
  ]
}
```

CHÚ Ý:
- QUAN SÁT tài liệu trước khi tạo câu hỏi
- Chọn tên trường DỰA TRÊN nội dung thực tế trong ảnh
- Đừng dùng tên trường generic nếu doc có tên cụ thể
- Mỗi câu là lượt chat của USER, không có AI response
"""


def parse_local_model_response(response_text: str) -> Optional[List[str]]:
    """Parse response from local model to extract the list of turn prompts."""
    try:
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        result = json.loads(text)
        
        # Validate structure
        if "turns" in result and isinstance(result["turns"], list):
            turns = result["turns"]
            if 2 <= len(turns) <= 3:
                return turns
            else:
                logger.warning(f"Invalid number of turns: {len(turns)}")
                return None
        else:
            logger.warning("Response missing 'turns' field or not a list")
            return None
            
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from local model: {e}")
        logger.debug(f"Response text: {response_text[:500]}")
        return None
    except Exception as e:
        logger.warning(f"Error parsing local model response: {e}")
        return None


def parse_gemini_response(response_text: str) -> Optional[str]:
    """Parse response from Gemini and return clean JSON text."""
    try:
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # Validate it's proper JSON
        json.loads(text)
        return text
            
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from Gemini: {e}")
        logger.debug(f"Response text: {text[:500]}")
        return None
    except Exception as e:
        logger.warning(f"Error parsing Gemini response: {e}")
        return None


def generate_multiturn_prompts_with_local_model(
    image: Image.Image,
    local_llm: ChatOpenAI,
    max_retries: int = 3
) -> Optional[List[str]]:
    """
    Pass 1: Use local model to generate multi-turn prompts (2-3 turns).
    
    Returns:
        List of prompt strings if successful, None otherwise
    """
    import base64
    from io import BytesIO
    
    for attempt in range(max_retries):
        try:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": PASS1_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    }
                ]
            )
            
            # Invoke local model
            response = local_llm.invoke([message])
            
            if response.content:
                result = parse_local_model_response(response.content)
                if result:
                    logger.info(f"✓ Generated {len(result)}-turn conversation")
                    for i, turn in enumerate(result, 1):
                        logger.info(f"  Turn {i}: {turn[:80]}...")
                    return result
                else:
                    logger.warning(f"Invalid response structure on attempt {attempt + 1}")
            else:
                logger.warning(f"Empty response on attempt {attempt + 1}")
                
        except Exception as e:
            logger.warning(f"Local model attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return None


def extract_with_gemini_multiturn(
    image: Image.Image,
    turns: List[str],
    model: genai.GenerativeModel,
    key_manager: APIKeyManager,
    max_retries: int = 3
) -> Optional[List[str]]:
    """
    Pass 2: Use Gemini's native multi-turn conversation to answer prompts.
    Returns list of assistant responses (one per turn).
    """
    assistant_responses = []
    
    # Start a chat session with the image
    chat = model.start_chat(history=[])
    
    # System instruction for all turns
    system_instruction = """IMPORTANT:
- Return ONLY valid JSON format
- For fields that exist in the document: extract actual value
- For fields that DON'T exist in the document: set to null
- Do NOT hallucinate or make up missing data
- Consider context from previous turns in this conversation
- If asked to modify/reformat previous data, do so accordingly"""
    
    for turn_idx, user_prompt in enumerate(turns, 1):
        logger.info(f"    Turn {turn_idx}/{len(turns)}: Getting response...")
        
        if turn_idx == 1:
            message_content = [
                image,
                f"{user_prompt}\n\n{system_instruction}"
            ]
        else:
            message_content = f"{user_prompt}\n\n{system_instruction}"
        
        # Make API call with retry logic
        success = False
        for attempt in range(max_retries):
            try:
                key_manager.wait_for_rate_limit()
                
                response = chat.send_message(message_content)
                
                if not key_manager.record_request():
                    raise Exception("Daily request limit reached for this API key")
                
                if response.text:
                    cleaned_response = parse_gemini_response(response.text)
                    if cleaned_response:
                        assistant_responses.append(cleaned_response)
                        logger.info(f"    ✓ Turn {turn_idx} completed")
                        success = True
                        break
                
                logger.warning(f"Empty or invalid response on attempt {attempt + 1}")
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if (
                    "quota" in error_msg
                    or "rate" in error_msg
                    or "limit" in error_msg
                    or "daily" in error_msg
                    or "api key" in error_msg
                    or "invalid" in error_msg
                    or "unauthorized" in error_msg
                ):
                    # Let the outer handler deal with API issues
                    raise
                
                logger.warning(f"Turn {turn_idx} attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        if not success:
            logger.error(f"Failed to get response for turn {turn_idx}")
            return None
    
    return assistant_responses


def append_sharegpt_conversation(conversation: Dict, output_path: str):
    """Append a ShareGPT conversation to the output file."""
    output_file = Path(output_path)
    
    # Read existing data
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    # Append new conversation
    data.append(conversation)
    
    # Write back
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_image_for_synthetic_data(
    image: Image.Image,
    file_path: str,
    page_num: int,
    total_pages: int,
    local_llm: ChatOpenAI,
    gemini_model: genai.GenerativeModel,
    key_manager: APIKeyManager,
    output_path: str,
    checkpoint: ProcessingCheckpoint,
    checkpoint_path: str
) -> bool:
    """Process a single image to generate multi-turn synthetic data in ShareGPT format."""
    
    # Pass 1: Local model generates multi-turn prompts
    logger.info(f"  Pass 1: Generating multi-turn conversation with local model...")
    user_turns = generate_multiturn_prompts_with_local_model(image, local_llm)
    
    if not user_turns:
        logger.error("  ✗ Failed to generate multi-turn prompts")
        return False
    
    # Pass 2: Gemini answers each turn
    logger.info(f"  Pass 2: Getting responses from Gemini for {len(user_turns)} turns...")
    assistant_responses = extract_with_gemini_multiturn(
        image, 
        user_turns,
        gemini_model,
        key_manager
    )
    
    if not assistant_responses or len(assistant_responses) != len(user_turns):
        logger.error("  ✗ Failed to get complete multi-turn responses from Gemini")
        return False
    
    # Create ShareGPT format multi-turn conversation
    conversations = []
    for user_msg, assistant_msg in zip(user_turns, assistant_responses):
        conversations.append({
            "from": "user",
            "value": user_msg
        })
        conversations.append({
            "from": "assistant",
            "value": assistant_msg
        })
    
    sharegpt_conversation = {
        "conversations": conversations,
        "source": file_path,
        "page": page_num,
        "total_pages": total_pages,
        "num_turns": len(user_turns),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to output
    append_sharegpt_conversation(sharegpt_conversation, output_path)
    logger.info(f"  ✓ Successfully generated {len(user_turns)}-turn ShareGPT conversation")
    
    return True


def process_file(
    file_path: str,
    local_llm: ChatOpenAI,
    gemini_model: genai.GenerativeModel,
    key_manager: APIKeyManager,
    checkpoint: ProcessingCheckpoint,
    output_path: str,
    checkpoint_path: str,
) -> bool:
    """Process a single file (PDF or image) for synthetic data generation."""
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
            
            success = process_image_for_synthetic_data(
                image=image,
                file_path=file_path,
                page_num=page_num,
                total_pages=len(images),
                local_llm=local_llm,
                gemini_model=gemini_model,
                key_manager=key_manager,
                output_path=output_path,
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path
            )
            
            # Update checkpoint regardless of success
            file_progress["pages"].append(page_num)
            checkpoint.total_processed += 1
            save_checkpoint(checkpoint, checkpoint_path)
        
        file_progress["completed"] = True
        save_checkpoint(checkpoint, checkpoint_path)
        
    else:
        # Process image
        try:
            image = load_image(file_path)
            
            success = process_image_for_synthetic_data(
                image=image,
                file_path=file_path,
                page_num=1,
                total_pages=1,
                local_llm=local_llm,
                gemini_model=gemini_model,
                key_manager=key_manager,
                output_path=output_path,
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path
            )
            
            file_progress["pages"].append(1)
            file_progress["completed"] = True
            checkpoint.total_processed += 1
            save_checkpoint(checkpoint, checkpoint_path)
            
        except Exception as e:
            logger.error(f"Error processing image {file_name}: {e}")
            raise
    
    return True


def run_synthetic_pipeline(
    input_path: str,
    output_path: str,
    api_keys_file: str,
    local_api_key: str,
    local_base_url: str,
    local_model_name: str = "nxchat-vision",
    checkpoint_path: Optional[str] = None,
) -> None:
    """Run the synthetic data generation pipeline."""
    
    # Setup paths
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_path is None:
        checkpoint_path = str(output_dir / "checkpoint.json")
    
    # Initialize output file as empty list if it doesn't exist
    if not Path(output_path).exists():
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f)
    
    # Initialize local model
    logger.info("Initializing local model...")
    local_llm = ChatOpenAI(
        model=local_model_name,
        api_key=local_api_key,
        base_url=local_base_url,
        temperature=0.8,  # Higher temperature for more variety
    )
    
    # Load Gemini API keys
    key_manager = APIKeyManager()
    if not key_manager.load_keys(api_keys_file):
        logger.error("Failed to load Gemini API keys. Exiting.")
        sys.exit(1)
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Get files to process
    files = get_supported_files(input_path)
    if not files:
        logger.error(f"No supported files found in: {input_path}")
        sys.exit(1)
    
    # Log pipeline info
    total_max_requests = len(key_manager.keys) * key_manager.REQUESTS_PER_DAY
    logger.info("=" * 60)
    logger.info("Starting MULTI-TURN SYNTHETIC DATA GENERATION pipeline")
    logger.info("Output format: ShareGPT (2-3 turn conversations)")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Files to process: {len(files)}")
    logger.info(f"Local model: {local_model_name}")
    logger.info(f"Gemini API keys: {len(key_manager.keys)} keys × {key_manager.REQUESTS_PER_DAY} req/day = {total_max_requests} max requests/day")
    logger.info("Note: Each conversation will use 2-3 Gemini API calls")
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
                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                
                success = process_file(
                    file_path=file_path,
                    local_llm=local_llm,
                    gemini_model=gemini_model,
                    key_manager=key_manager,
                    checkpoint=checkpoint,
                    output_path=output_path,
                    checkpoint_path=checkpoint_path,
                )
                
                if success:
                    break  # Move to next file
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Determine failure reason
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
                
                # Try to rotate to next key
                if not key_manager.mark_current_failed(reason):
                    logger.error("All Gemini API keys exhausted! Stopping pipeline.")
                    logger.info(key_manager.get_status_summary())
                    save_checkpoint(checkpoint, checkpoint_path)
                    sys.exit(1)
                logger.info("Retrying with new API key...")
        
        if not key_manager.has_available_keys():
            logger.error("All Gemini API keys exhausted! Stopping pipeline.")
            logger.info(key_manager.get_status_summary())
            save_checkpoint(checkpoint, checkpoint_path)
            sys.exit(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("Multi-turn synthetic data generation completed successfully!")
    logger.info(f"Total conversations generated: {checkpoint.total_processed}")
    logger.info(f"Output saved to: {output_path}")
    logger.info("=" * 60)
    logger.info(key_manager.get_status_summary())
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate multi-turn synthetic data in ShareGPT format using Local Model (Pass 1) and Gemini (Pass 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python synthetic_data_pipeline.py --input ./data --output ./output/sharegpt_multiturn.json --api-keys ./gemini_keys.txt
  
  # Using environment variables from .env file:
  python synthetic_data_pipeline.py -i ./data -o ./sharegpt_multiturn.json -k ./keys.txt
        """,
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory or file path (supports PDF, PNG, JPG, JPEG, WebP, GIF, BMP)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSON file path (ShareGPT multi-turn format)",
    )
    parser.add_argument(
        "-k", "--api-keys",
        required=True,
        help="Path to text file containing Gemini API keys (one per line)",
    )
    parser.add_argument(
        "--local-api-key",
        default=None,
        help="API key for local model (default: from .env file)",
    )
    parser.add_argument(
        "--local-base-url",
        default=None,
        help="Base URL for local model (default: from .env file)",
    )
    parser.add_argument(
        "--local-model",
        default="nxchat-vision",
        help="Local model name (default: nxchat-vision)",
    )
    parser.add_argument(
        "-c", "--checkpoint",
        default=None,
        help="Path to checkpoint file (default: same directory as output)",
    )
    
    args = parser.parse_args()
    
    # Get local model credentials
    local_api_key = args.local_api_key or os.getenv("api_key")
    local_base_url = args.local_base_url or os.getenv("base_url")
    
    if not local_api_key or not local_base_url:
        logger.error("Local model credentials not provided. Please use --local-api-key and --local-base-url, or set api_key and base_url in .env file")
        sys.exit(1)
    
    run_synthetic_pipeline(
        input_path=args.input,
        output_path=args.output,
        api_keys_file=args.api_keys,
        local_api_key=local_api_key,
        local_base_url=local_base_url,
        local_model_name=args.local_model,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()