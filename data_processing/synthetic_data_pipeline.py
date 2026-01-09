# -*- coding: utf-8 -*-
"""
Synthetic Data Generation Pipeline using Local Model (Pass 1) and Gemini (Pass 2).

This pipeline:
1. Uses a local model to generate multi-turn conversations (2-3 turns)
2. Uses Gemini to answer those prompts and create ground truth data
3. Outputs in ShareGPT format (user/assistant multi-turn conversation)

Usage:
    python data_processing/synthetic_data_pipeline.py --input ./data --output ./output/synthetic_sharegpt.json --api-keys ./api_keys.txt
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

PASS1_PROMPT = """You are a real user chatting with an AI to extract information from a document.

TASK: Generate 2-3 natural chat questions (ONLY user questions, NO AI responses)

STEP 1 - ANALYZE THE DOCUMENT:
- Identify document type (delivery note, invoice, receipt, purchase order...)
- List all AVAILABLE fields/information
- Think of reasonable fields for this document type that are NOT present in the image

STEP 2 - CREATE DIVERSE CONVERSATIONS:

**GOLDEN RULES:**
1. **DIVERSE STYLES** - Each conversation should be different:
   - Formal/casual/mixed style
   - Pure Vietnamese, pure English, or Vietnamese-English mix
   - Sometimes request JSON, sometimes plain text, sometimes tables, sometimes bullet points
   
2. **DIVERSE QUESTION STRUCTURES**:
   - Direct question: "What's the invoice number?"
   - List request: "Show me customer information"
   - Summary request: "What's the total quantity and value?"
   - Conditional: "Which products have quantity > 100?"
   - Comparison: "Which product is most expensive?"
   - Format request: "List as table/list/JSON/whatever"

3. **DIVERSE CONTENT**:
   - Turn 1: Can ask for overview OR details OR single field
   - Turn 2: Can ask for more OR modify OR clarify OR different format
   - Turn 3 (50% chance): Can summarize OR extract differently OR verify

**DIVERSITY EXAMPLES (DON'T copy, just reference ideas):**

*Style 1 - Casual Vietnamese + JSON:*
"Lấy số phiếu và ngày cho mình. JSON nhé"
"À còn list sản phẩm nữa"

*Style 2 - Formal Vietnamese + Text:*
"Cho tôi biết thông tin người gửi và người nhận"
"Số lượng và đơn giá của từng sản phẩm là bao nhiêu?"

*Style 3 - English + Table:*
"Get the order number and date"
"List all items in a table format"
"What's the total amount?"

*Style 4 - Detailed questions:*
"How many product types are in this document?"
"Which product has the highest quantity?"

*Style 5 - General overview:*
"Show me all information from this form"
"Reformat it as a markdown table"

*Style 6 - Conditional queries:*
"List products with unit price above 100k"
"Are there any expired products?"

*Style 7 - Vietnamese-English mix:*
"Get order number, date và customer info"
"List all items dạng bảng"
"Tổng tiền bao nhiêu?"

*Style 8 - Casual English:*
"grab the invoice details for me"
"also need the product breakdown"

*Style 9 - Simple plain text request:*
"What's the delivery date and total weight?"
"List the products with their quantities"

*Style 10 - Direct Vietnamese:*
"Ngày giao hàng và tổng khối lượng là bao nhiêu?"
"Có mấy loại sản phẩm trong phiếu này?"

**SPEAKING STYLE:**
✅ Natural: "cho mình", "lấy giúp", "show me", "can you", "được không"
✅ Short or long depending on mood
✅ Sometimes polite, sometimes casual
✅ Sometimes ask 1 field, sometimes ask many
✅ Field names based on actual document content (don't use generic names)
✅ Keep questions SIMPLE and DIRECT - don't add unnecessary context

❌ AVOID:
- Always asking for JSON
- Always being too formal with "vui lòng", "xin hãy", "please kindly"
- Asking the same pattern repeatedly
- Asking for fields irrelevant to document type
- Typos or spelling errors
- Adding weird justifications like "cần để vào slide", "for presentation", "để báo cáo"
- Over-explaining why you need the information

**NULL FIELD INJECTION TECHNIQUE:**
- 30-40% of conversations should include 1 non-existent field
- Ask NATURALLY as if the field exists (don't say "if available", "though not present")
- Reasonable examples:
  + Delivery note → ask for email, phone of recipient (often not present)
  + Invoice → ask for company website, fax (often not present)
  + Purchase order → ask for specific delivery address (may not be present)

**ENSURE DIVERSITY:**
Output format distribution:
- JSON format: ~40% conversations
- Plain text: ~30% conversations  
- Table/List format: ~20% conversations
- Mixed format: ~10% conversations

Language distribution:
- Pure Vietnamese: ~50%
- Vietnamese-English mix: ~30%
- Pure English: ~20%

Question scope:
- Ask 1-2 fields: ~40%
- Ask many fields: ~40%
- Ask summary/conditional: ~20%

OUTPUT:
```json
{
  "turns": [
    "question 1 - based on actual content, diverse style",
    "question 2 - natural follow-up, can change format",
    "question 3 (optional) - if needed for addition or adjustment"
  ]
}
```

IMPORTANT NOTES:
- OBSERVE the document carefully first
- Each conversation must be DIFFERENT in style, format, tone
- Don't repeat the same patterns
- Field names must be based on ACTUAL content
- The more NATURAL and DIVERSE, the better
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
    Pass 2: Use Gemini generate_content purely (Stateless) mimicking chat.
    """
    assistant_responses = []
    
    # Lịch sử chat thủ công
    history_contents = [] 
    
    system_instruction = """You are a helpful document extraction assistant. Follow these rules STRICTLY:
1. **FORMAT ADHERENCE**: 
   - If user asks for JSON → return ONLY valid JSON
   - If user asks for plain text/văn bản → return natural language text (NO JSON)
   - If user asks for table/bảng → return markdown table format
   - If user asks for list/bullet points → return list format
   - If user doesn't specify format → choose the most natural format (prefer JSON for structured data)
2. **DATA EXTRACTION**:
   - For fields that exist in the document: extract the actual value
   - For fields that DON'T exist: set to null (in JSON) or state "không có" / "not available" (in text)
   - NEVER hallucinate or make up missing data
3. **CONVERSATION CONTEXT**:
   - Remember previous turns in this conversation
   - If asked to modify/reformat previous data, do so accordingly
   - If asked to add more fields, include previous data + new data
4. **OUTPUT QUALITY**:
   - Be concise and accurate
   - Use the same language as the user's question
   - Maintain consistency across turns"""

    for turn_idx, user_prompt in enumerate(turns, 1):
        logger.info(f"    Turn {turn_idx}/{len(turns)}: Getting response...")
        
        # Combine user prompt with system instruction
        current_prompt = f"{user_prompt}\n\n{system_instruction}"
        
        if turn_idx == 1:
            user_content = {
                "role": "user",
                "parts": [image, current_prompt]
            }
        else:
            user_content = {
                "role": "user",
                "parts": [current_prompt]
            }
            
        history_contents.append(user_content)

        response_text = None
        for attempt in range(max_retries):
            try:
                # Rate limit (RPM) check logic within KeyManager
                key_manager.wait_for_rate_limit()
                
                response = model.generate_content(history_contents)
                
                # Record successful usage
                if key_manager.record_request():
                    pass
                
                if response.text:
                    # For JSON responses, validate and clean
                    if any(keyword in user_prompt.lower() for keyword in ['json', 'JSON']):
                        response_text = parse_gemini_response(response.text)
                        if response_text:
                            break
                    else:
                        # For non-JSON responses, just clean markdown fences if any
                        response_text = response.text.strip()
                        if "```" in response_text:
                            response_text = response_text.replace("```json", "").replace("```", "").strip()
                        break
                        
            except Exception as e:
                error_str = str(e).lower()
                
                # --- PHẦN SỬA ĐỔI QUAN TRỌNG TẠI ĐÂY ---
                # Nếu gặp lỗi Quota/Limit (429), KHÔNG ĐƯỢC RETRY, phải RAISE ngay lập tức
                # để hàm main bắt được và đổi key.
                if "quota" in error_str or "429" in error_str or "limit" in error_str:
                    logger.warning(f"Turn {turn_idx} hit HARD LIMIT (Quota): {e}")
                    raise e  # Ném lỗi ra ngoài để hàm run_synthetic_pipeline bắt và đổi key
                
                # Các lỗi khác (network, server error 500...) thì mới retry
                logger.warning(f"Turn {turn_idx} failed attempt {attempt}: {e}")
                time.sleep(2 ** attempt) # Backoff exponential

        if not response_text:
            return None

        assistant_responses.append(response_text)
        
        history_contents.append({
            "role": "model",
            "parts": [response_text]
        })

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
                gemini_model = genai.GenerativeModel("gemini-3-flash-preview")
                
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
  python data_processing/synthetic_data_pipeline.py --input ./data --output ./output/sharegpt_multiturn.json --api-keys ./gemini_keys.txt
  
  # Using environment variables from .env file:
  python data_processing/synthetic_data_pipeline.py -i ./data -o ./sharegpt_multiturn.json -k ./keys.txt
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