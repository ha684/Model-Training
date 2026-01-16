from data_processing.apikey_manager import ProcessingCheckpoint
from PIL import Image
from pdf2image import convert_from_path
import json
import os
from pathlib import Path
from datetime import datetime
import base64
import logging
from typing import Optional
import json_repair

logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_path: str) -> ProcessingCheckpoint:
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            checkpoint = ProcessingCheckpoint.from_dict(data)
            logger.info(f"Loaded checkpoint: {checkpoint.total_processed} items processed")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
    return ProcessingCheckpoint()


def save_checkpoint(checkpoint: ProcessingCheckpoint, checkpoint_path: str) -> None:
    checkpoint.last_updated = datetime.now().isoformat()
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)


def append_result(result: dict, output_path: str) -> None:
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

    files = sorted(list(set(files)))
    logger.info(f"Found {len(files)} supported file(s)")
    return files


def load_image(file_path: str) -> Image.Image:
    return Image.open(file_path)


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list:
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        logger.info(f"Converted PDF '{pdf_path}' to {len(images)} page(s)")
        return images
    except Exception as e:
        logger.error(f"Failed to convert PDF '{pdf_path}': {e}")
        return []

def parse_gemini_response(response_text: str) -> Optional[dict]:
    text = response_text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            logger.info("JSON chuẩn bị lỗi, đang cố gắng repair...")
            decoded = json_repair.repair_json(text, return_objects=True)
            return decoded
        except Exception as e:
            logger.warning(f"Failed to repair/parse JSON response: {e} \n Response text: {response_text}")
            return None

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_conversation_prompt() -> str:
   """Generate diverse format conversations for VLM training with hard negatives and logic validation."""
   prompt = """You are an expert in analyzing Vietnamese business documents and creating natural, realistic conversations with DIVERSE OUTPUT FORMATS and CRITICAL REASONING.

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
        | STT | Tên sản phẩm | Số lượng (pcs) |
        | :--- | :--- | :--- |
        | 1 | Snack A | 100 pcs |
   
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

3. **HARD NEGATIVES & TRICK QUESTIONS** (CRITICAL - 25-35% of conversations):
   
   Generate questions that test the model's ability to distinguish confusable information:
   
   **Types of Hard Negatives:**
   
   a) **Number Confusion** - Testing if model confuses similar numbers:
      - Identify all numbers in the document (invoice numbers, codes, quantities, dates, etc.)
      - Create questions that ask about one field but mention a number from another field
      - Principles:
        • Use suffixes/prefixes from one field to trick about another
        • Use partial matches from codes/IDs
        • Mix date components with other numbers
        • Confuse quantity with price or vice versa
      
      Example patterns (DO NOT copy - generate based on actual document):
      - "Mã sản phẩm có chứa số [X] đúng không?" (where X is from a different field)
      - "Tổng cộng là [Y] đúng không?" (where Y is a line item, not the total)
      - "Số phiếu kết thúc bằng [Z] phải không?" (where Z is from customer code)
      
      Correct Assistant responses should:
      - Clearly identify what field the confused number actually belongs to
      - State the correct value for the asked field
      - Specify the location in the document
   
   b) **Unit Confusion** - Testing if model confuses measurement units:
      - Identify all units used (pcs, thùng, cái, hộp, kg, lít, etc.)
      - Ask about items using wrong but plausible units
      - Principles:
        • Confuse container units (thùng, hộp) with piece units (cái, chiếc)
        • Confuse count units with weight/volume units
        • Ask if measurements are in units not present in document
      
      Example patterns (adapt to document):
      - "Sản phẩm [A] tính theo cái hay theo thùng?"
      - "Đơn vị của mục [B] là kg phải không?"
      
      Correct responses should:
      - State the actual unit shown in the document
      - Clarify what the unit represents
      - Note if the asked unit doesn't appear
   
   c) **Field Confusion** - Testing if model confuses similar field names:
      - Identify fields with similar meanings (e.g., "Mã KH" vs "Số phiếu", "Ngày tạo" vs "Ngày giao")
      - Create questions that conflate these fields
      - Principles:
        • Confuse customer/supplier codes with invoice numbers
        • Confuse creation date with delivery date
        • Confuse subtotal with total
        • Confuse unit price with total price
      
      Example patterns:
      - "Mã khách hàng và số đơn hàng có giống nhau không?"
      - "Ngày in phiếu và ngày giao hàng là một không?"
      
      Correct responses should:
      - Clearly distinguish between the two fields
      - Provide both values if both exist
      - Explain the difference in meaning
   
   d) **Location-based Confusion** - Testing spatial understanding:
      - Identify information that appears in multiple locations (headers, footers, tables)
      - Ask about values from wrong sections
      - Principles:
        • Confuse header totals with table totals
        • Confuse footer notes with body content
        • Ask if information from one section applies to another
      
      Example patterns:
      - "Số ở góc phải trên là tổng số lượng đúng không?"
      - "Thông tin ở cuối bảng có phải là tổng cộng không?"
   
   e) **Ambiguous Queries** - Testing interpretation:
      - Ask intentionally vague questions that could refer to multiple things
      - Principles:
        • Use pronouns without clear antecedents ("Cái đó là gì?")
        • Ask about "first/last" without specifying what list
        • Use relative terms without anchors ("Mục tiếp theo")
      
      Correct responses should:
      - Ask for clarification OR
      - List all possible interpretations with their values
   
   **Distribution:**
   - 25-35% of conversations should include at least 1-2 hard negative questions
   - Vary the types across the dataset
   - Don't use the same confusion pattern twice in one conversation
   - Make the trick questions look natural, not obviously wrong
   
   **CRITICAL:** 
   - Generate confusion based on ACTUAL document content, not generic examples
   - Use numbers/codes/units that EXIST in the image
   - Vary the confusion patterns across different documents
   - Don't create repetitive patterns that the model can memorize

4. **ARITHMETIC & LOGIC VALIDATION** (15-25% of conversations):
   
   Create questions that require computation or logical verification:
   
   a) **Sum Verification**:
      - "Kiểm tra xem tổng các dòng có khớp với số tổng cộng không?"
      - Assistant should: Add up line items, compare with stated total, confirm match/mismatch
      
   b) **Subtotal Validation**:
      - "Tính tổng số lượng của nhóm sản phẩm [X]?"
      - Assistant should: Identify relevant rows, sum them, present result
      
   c) **Cross-field Calculation**:
      - "Nếu mỗi thùng có 24 cái, tổng cộng bao nhiêu cái?"
      - Assistant should: Multiply stated quantity by conversion factor
      
   d) **Comparison Logic**:
      - "Sản phẩm nào nhiều hơn: [A] hay [B]? Nhiều hơn bao nhiêu?"
      - Assistant should: Extract both values, compare, calculate difference
      
   e) **Percentage Calculation**:
      - "[A] chiếm bao nhiêu phần trăm trong tổng số?"
      - Assistant should: Calculate ratio, present as percentage
      
   f) **Consistency Check**:
      - "Số lượng sản phẩm ghi trong bảng có khớp với mô tả ở đầu phiếu không?"
      - Assistant should: Compare counts from different sections
   
   **Principles:**
   - Use ACTUAL numbers from the document
   - Vary calculation complexity (simple addition to multi-step computation)
   - Include both correct and incorrect verification scenarios
   - Assistant should show calculation steps when relevant
   - Format numeric results consistently with document style
   
   **Distribution:**
   - 1-2 arithmetic questions per conversation when applicable
   - Prioritize documents with tabular data
   - Mix verification (checking existing totals) with computation (deriving new values)

5. **Negative Questions & Null Handling** (20-30% of conversations):
   
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

6. **Question Type Variety**:
   
   **Basic Extraction** (20-25%):
   - "Số phiếu là gì?"
   - "Lấy thông tin nhà cung cấp"
   
   **Format-specific requests** (20-25%):
   - "Cho tôi dưới dạng JSON"
   - "Làm bảng markdown"
   - "Export CSV"
   - "Tóm tắt có cấu trúc"
   
   **Arithmetic & Comparison** (15-25%):
   - "Tổng số lượng?"
   - "So sánh 3 sản phẩm"
   - "Sản phẩm nào nhiều nhất?"
   - "Kiểm tra tổng có đúng không?"
   
   **Hard Negatives** (15-25%):
   - Number confusion questions
   - Unit confusion questions
   - Field confusion questions
   - Location-based tricks
   
   **Transformation** (10-15%):
   - "Đổi sang format khác"
   - "Chuyển sang bảng"
   - "Format lại dạng key-value"
   
   **Verification & Negative** (10-15%):
   - "Có email không?"
   - "Những trường nào bị thiếu?"

7. **Answer Quality by Format**:
   
   **Text responses:**
   - Natural Vietnamese, concise
   - "Tổng cộng có 11,100 sản phẩm"
   - For corrections: Clearly state what's wrong, provide correct answer, cite location
   
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
   
   **Arithmetic/Verification:**
   - Show calculation steps when helpful
   - Clearly state result
   - Note if values match or don't match expectations

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

**Hard Negatives:**
- Base confusion on ACTUAL document content
- Use real numbers/codes/units from the image
- Vary confusion types across dataset
- Make corrections clear and educational

**Arithmetic:**
- Use actual numbers from document
- Show calculation when helpful
- State results clearly
- Format consistently

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
- 4-5 turns: Simple documents, 2-3 formats, 0-1 hard negative
- 6-7 turns: Medium complexity, 3-4 formats, 1-2 hard negatives, 1 arithmetic
- 8 turns: Complex documents, 4-5 formats, 2-3 hard negatives, 1-2 arithmetic

Each conversation should feel natural - don't force all features into one conversation.

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
- [ ] 25-35% chance: includes 1-2 hard negative questions (number/unit/field confusion)
- [ ] 15-25% chance: includes 1-2 arithmetic/verification questions
- [ ] 20-30% chance: includes null/negative questions
- [ ] All formats properly structured
- [ ] JSON is raw (no code blocks)
- [ ] Natural format transitions
- [ ] All answers accurate and based on actual document content
- [ ] Hard negatives use real document data, not generic examples
- [ ] Corrections are clear and cite specific locations
- [ ] Arithmetic shows calculation when helpful
- [ ] Proper Vietnamese
- [ ] Valid outer JSON

DIVERSITY GUIDELINES:
- Generate confusion patterns based on what's actually in each unique document
- Vary which fields you create confusion about across different samples
- Don't create the same trick question pattern repeatedly
- If a document has many similar items, vary which items you ask about
- Make hard negatives feel like genuine user confusion, not artificial tests
- Balance the conversation - don't make every question a trick question

EXAMPLES:

Example 1 - With Hard Negative (6 turns):
{
  "conversation": [
    {"user": "Số phiếu là bao nhiêu?", 
     "assistant": "Số phiếu là PXB25110053"},
    {"user": "053 đó là số phiếu đúng không?",
     "assistant": "Không, 053 chỉ là 3 số cuối của số phiếu. Số phiếu đầy đủ là PXB25110053, ghi ở góc trên bên phải của phiếu."},
    {"user": "Kiểm tra tổng số lượng các dòng có khớp không?",
     "assistant": "Dòng 1: 2,400 pcs\nDòng 2: 600 pcs\nDòng 3: 600 pcs\n...\nTổng tính: 11,100 pcs\nTổng ghi trong phiếu: 11,100 pcs\n\n✓ Số liệu khớp chính xác."},
    {"user": "Liệt kê 3 sản phẩm đầu dạng bảng",
     "assistant": "| STT | Tên sản phẩm | Số lượng |\n| :--- | :--- | ---: |\n| 1 | Snack Bắp Karamucho cay | 2,400 pcs |\n| 2 | Snack Bắp phô mai | 600 pcs |\n| 3 | Snack Bắp tương đen | 600 pcs |"},
    {"user": "Đơn vị tính là cái hay thùng?",
     "assistant": "Đơn vị trong bảng ghi là 'pcs' (pieces - cái/chiếc), không phải thùng. Tất cả số lượng đều tính theo đơn vị pcs."},
    {"user": "Export thành JSON",
     "assistant": "{\"so_phieu\": \"PXB25110053\", \"ngay\": \"03/11/2025\", \"tong_so_luong\": 11100, \"don_vi\": \"pcs\"}"}
  ]
}

Example 2 - With Arithmetic (5 turns):
{
  "conversation": [
    {"user": "Tóm tắt phiếu này",
     "assistant": "**Tóm tắt:** Phiếu xuất hàng PXB25110053 ngày 03/11/2025, có 8 loại snack từ KOIKEYA VIETNAM.\n\nChi tiết:\n- Snack Bắp: 4 loại, 6,000 pcs\n- Snack Khoai tây: 4 loại, 5,100 pcs\n- Tổng: 11,100 pcs"},
    {"user": "Snack Bắp chiếm bao nhiêu phần trăm trong tổng số?",
     "assistant": "Snack Bắp: 6,000 pcs\nTổng cộng: 11,100 pcs\n\nTính toán: (6,000 / 11,100) × 100 = 54.05%\n\nSnack Bắp chiếm khoảng 54% tổng số lượng."},
    {"user": "Sản phẩm nào nhiều nhất? Nhiều hơn sản phẩm thứ 2 bao nhiêu?",
     "assistant": "Sản phẩm nhiều nhất: Snack Bắp Karamucho cay (500VS29) - 2,400 pcs\nSản phẩm thứ 2: Có 6 sản phẩm bằng nhau - 600 pcs mỗi loại\n\nChênh lệch: 2,400 - 600 = 1,800 pcs\n\n→ Sản phẩm đầu nhiều gấp 4 lần sản phẩm thứ 2."},
    {"user": "Thông tin nhà cung cấp dạng JSON",
     "assistant": "{\"ten\": \"KOIKEYA VIETNAM CO., LTD\", \"dia_chi\": \"N3-2 St., Long Duc IP, Long Thanh, Dong Nai\", \"dien_thoai\": \"+84(0)251 368 1368\", \"fax\": \"+84(0)251 368 1369\", \"email\": null}"},
    {"user": "Có ghi mã số thuế không?",
     "assistant": "Không có thông tin mã số thuế trong phiếu xuất kho này."}
  ]
}
"""
   return prompt

def get_reasoning_prompt() -> str:
    """Generate reasoning-focused conversations that teach models to think step-by-step before answering."""
    prompt = """You are an expert in analyzing Vietnamese business documents and creating conversations that TEACH STEP-BY-STEP REASONING before providing final answers.

TASK:
Analyze the provided document image and generate a realistic multi-turn conversation (5-8 turns) where the AI assistant demonstrates EXPLICIT REASONING PROCESS before answering questions.

CORE PRINCIPLE: **Think Out Loud Before Answering**

The assistant MUST show its thinking process using one of these reasoning patterns:

1. **Sequential Analysis Pattern** (30-35%):
   - Break down the question into sub-steps
   - Analyze each step explicitly
   - Combine findings to reach conclusion
   
   Example:
   User: "Hóa đơn này có vấn đề gì không?"
   Assistant: "Tôi sẽ kiểm tra từng phần:
   
   **Bước 1: Kiểm tra thông tin cơ bản**
   - Số hóa đơn: PXB25110053 ✓
   - Ngày tháng: 03/11/2025 ✓
   - Nhà cung cấp: KOIKEYA VIETNAM ✓
   
   **Bước 2: Kiểm tra tính toán**
   - Dòng 1: 2,400 pcs
   - Dòng 2: 600 pcs
   - ...
   - Tổng tính được: 11,100 pcs
   - Tổng ghi trên phiếu: 11,100 pcs ✓
   
   **Bước 3: Kiểm tra thông tin bắt buộc**
   - Địa chỉ: Có ✓
   - Điện thoại: Có ✓
   - Email: Không có ⚠️
   
   **Kết luận:** Phiếu hợp lệ về mặt số liệu, chỉ thiếu thông tin email không bắt buộc."

2. **Comparative Reasoning Pattern** (25-30%):
   - Identify items to compare
   - List relevant attributes of each
   - Analyze differences/similarities
   - Draw conclusion with evidence
   
   Example:
   User: "Sản phẩm nào đáng chú ý nhất trong phiếu này?"
   Assistant: "Để xác định sản phẩm đáng chú ý, tôi so sánh các tiêu chí:
   
   **Theo số lượng:**
   - Snack Bắp Karamucho cay: 2,400 pcs (cao nhất)
   - Các sản phẩm còn lại: 600-1,500 pcs
   → Karamucho cay gấp 4 lần sản phẩm khác
   
   **Theo tỷ trọng:**
   - Karamucho cay: 2,400/11,100 = 21.6%
   - Sản phẩm khác: 5.4%-13.5%
   → Chiếm tỷ trọng lớn nhất
   
   **Kết luận:** Snack Bắp Karamucho cay (500VS29) là sản phẩm đáng chú ý nhất vì có số lượng cao nhất (2,400 pcs) và chiếm 21.6% tổng đơn hàng."

3. **Detective/Verification Pattern** (20-25%):
   - State the claim/question to verify
   - Gather evidence from document
   - Check consistency across sections
   - Identify discrepancies or confirm accuracy
   
   Example:
   User: "Tổng số lượng ghi 11,100 pcs có đúng không?"
   Assistant: "Tôi sẽ xác minh bằng cách cộng từng dòng:
   
   **Thu thập dữ liệu:**
   Dòng 1: 2,400 pcs
   Dòng 2: 600 pcs
   Dòng 3: 600 pcs
   Dòng 4: 600 pcs
   Dòng 5: 1,500 pcs
   Dòng 6: 600 pcs
   Dòng 7: 1,500 pcs
   Dòng 8: 1,500 pcs
   
   **Tính toán:**
   2,400 + 600 + 600 + 600 + 1,500 + 600 + 1,500 + 1,500
   = (2,400 + 1,500 + 1,500 + 1,500) + (600 × 4)
   = 6,900 + 2,400
   = 9,300 pcs... Khoan đã, để tôi đếm lại.
   
   = 2,400 + (600 × 3) + (1,500 × 3) + 600
   = 2,400 + 1,800 + 4,500 + 2,400
   = 11,100 pcs ✓
   
   **Kết luận:** Tổng số lượng 11,100 pcs là chính xác."

4. **Elimination/Deduction Pattern** (15-20%):
   - List all possibilities
   - Rule out options one by one
   - Arrive at answer through elimination
   
   Example:
   User: "Sản phẩm 500VS29 là loại nào?"
   Assistant: "Tôi sẽ tìm bằng cách loại trừ:
   
   **Các khả năng:**
   - Snack Bắp (4 loại)
   - Snack Khoai tây (4 loại)
   
   **Kiểm tra thông tin:**
   Dòng 1 có mã 500VS29, số lượng 2,400 pcs
   
   **Đối chiếu với mô tả:**
   - Tên ghi: "Snack Bắp Karamucho..."
   - Các mã 5xxVS29 thường là dòng Bắp
   - Các mã 2xxVS29 thường là dòng Khoai tây
   
   **Loại trừ:**
   ❌ Không phải Khoai tây (mã không bắt đầu bằng 2)
   ❌ Không phải các sản phẩm 600 pcs (vì có 2,400 pcs)
   ✓ Phù hợp với Snack Bắp số lượng lớn
   
   **Kết luận:** 500VS29 là Snack Bắp Karamucho vị cay (dòng đầu tiên)."

5. **What-If/Scenario Analysis Pattern** (10-15%):
   - Propose hypothetical scenario
   - Analyze implications
   - Calculate potential outcomes
   
   Example:
   User: "Nếu mỗi thùng chứa 24 cái, cần bao nhiêu thùng cho từng sản phẩm?"
   Assistant: "Tôi sẽ tính toán cho từng sản phẩm:
   
   **Công thức:** Số thùng = Tổng số lượng ÷ 24 (làm tròn lên)
   
   **Tính toán:**
   1. Snack Bắp Karamucho cay: 2,400 pcs
      → 2,400 ÷ 24 = 100 thùng (chia hết)
   
   2. Snack Bắp phô mai: 600 pcs
      → 600 ÷ 24 = 25 thùng (chia hết)
   
   3. Snack Khoai tây cay: 1,500 pcs
      → 1,500 ÷ 24 = 62.5 → Cần 63 thùng (dư 12 cái)
   
   **Tổng hợp:**
   - Tổng thùng cần: 100 + 25 + 25 + 25 + 63 + 25 + 63 + 63 = 389 thùng
   - Có 3 loại sản phẩm sẽ có thùng lẻ (1,500 pcs)
   - Tổng cộng: 389 thùng, trong đó 3 thùng chỉ chứa 12 cái thay vì 24 cái"

REASONING TRIGGER QUESTIONS:

**Verification Questions** (Must show calculation):
- "Kiểm tra xem tổng có đúng không?"
- "Phiếu này có vấn đề gì không?"
- "Số liệu có khớp không?"
- "Có chỗ nào sai không?"

**Comparison Questions** (Must compare attributes):
- "Sản phẩm nào đáng chú ý nhất?"
- "So sánh 3 sản phẩm đầu"
- "Loại nào nhiều hơn: A hay B?"
- "Điểm khác biệt giữa X và Y là gì?"

**Analysis Questions** (Must break down into steps):
- "Phân tích cấu trúc đơn hàng"
- "Đánh giá tổng quan phiếu này"
- "Tóm tắt các điểm chính"
- "Phiếu này cho biết điều gì?"

**Deduction Questions** (Must use elimination):
- "Mã 500VS29 là sản phẩm nào?"
- "Trường nào đang thiếu?"
- "Thông tin nào không có trong phiếu?"

**Scenario Questions** (Must calculate hypothetical):
- "Nếu... thì sao?"
- "Giả sử... thì cần bao nhiêu?"
- "Trường hợp... thì kết quả ra sao?"

CRITICAL RULES FOR REASONING:

1. **Show Working**: Always display intermediate steps, not just final answer
2. **Label Steps**: Use "Bước 1:", "Kiểm tra:", "Tính toán:", etc.
3. **Explicit Math**: Write out calculations: "2,400 + 600 = 3,000" not "tổng là 3,000"
4. **Acknowledge Uncertainty**: "Để tôi đếm lại", "Khoan, có vẻ...", "Cần kiểm tra..."
5. **Self-Correction**: Occasionally show minor errors then correct: "...= 9,300 pcs... Khoan đã, để tôi đếm lại..."
6. **Cite Evidence**: Always reference where info comes from: "Dòng 3 ghi...", "Ở góc trên bên phải..."
7. **Structured Output**: Use bullets, numbered lists, sections to organize thinking
8. **Conclusion Label**: Always end with "**Kết luận:**" before final answer

REASONING QUALITY INDICATORS:

✓ Good Reasoning:
- Shows 3+ intermediate steps
- Explicitly states assumptions
- References specific document locations
- Uses calculations with numbers shown
- Organizes thoughts with structure
- Ends with clear conclusion

✗ Bad Reasoning (Avoid):
- Jumps directly to answer
- "Tổng là 11,100 pcs" without showing addition
- Vague references: "trong phiếu có ghi"
- Unstructured wall of text
- No explicit conclusion

CONVERSATION STRUCTURE:

**Turn Distribution** (5-8 turns total):

Turn 1-2: Warm-up questions (simple extraction)
- User: "Số phiếu là gì?"
- Assistant: "Số phiếu là PXB25110053" (brief, no reasoning needed)

Turn 3-5: CORE REASONING QUESTIONS (must use reasoning patterns)
- User: Verification/Comparison/Analysis question
- Assistant: DETAILED reasoning response with steps

Turn 6-7: Follow-up or transformation
- User: "Chuyển kết quả sang JSON" or "Còn vấn đề nào khác?"
- Assistant: Can be brief or require additional reasoning

Turn 8: Optional complex scenario
- User: "Nếu X thì Y?"
- Assistant: Scenario analysis with reasoning

**Reasoning Density:**
- 60-70% of turns should trigger explicit reasoning
- At least 3 turns per conversation must show step-by-step thinking
- Balance between simple extraction and complex reasoning

HARD NEGATIVES WITH REASONING:

When user asks tricky questions, assistant must:
1. Identify the confusion
2. Explain why it's confusing
3. Show the correct logic
4. Cite specific evidence

Example:
User: "Tổng cộng là 2,400 pcs đúng không?"
Assistant: "Để tôi kiểm tra câu hỏi này:

**Phân tích:**
- Số 2,400 xuất hiện ở dòng 1 (số lượng sản phẩm đầu tiên)
- Câu hỏi hỏi về "tổng cộng"

**Kiểm tra tổng thực tế:**
Tổng cộng = Dòng 1 + Dòng 2 + ... + Dòng 8
           = 2,400 + 600 + ... + 1,500
           = 11,100 pcs

**Kết luận:**
Không, tổng cộng là 11,100 pcs, không phải 2,400 pcs. Số 2,400 là số lượng của sản phẩm đầu tiên (Snack Bắp Karamucho cay), chiếm 21.6% tổng đơn hàng."

FORMAT GUIDELINES:

Assistant can still use various formats (JSON, tables, etc.) but must:
- Show reasoning BEFORE presenting formatted output
- Explain why that format is chosen
- Include reasoning in natural text, then format result

Example:
User: "Tổng hợp thông tin dạng JSON"
Assistant: "Tôi sẽ thu thập các trường thông tin chính:

**Thông tin cơ bản:**
- Số phiếu: PXB25110053 (góc trên phải)
- Ngày: 03/11/2025 (dòng đầu bảng)
- Nhà cung cấp: KOIKEYA VIETNAM CO., LTD (header)

**Thông tin liên hệ:**
- Điện thoại: +84(0)251 368 1368 (có)
- Fax: +84(0)251 368 1369 (có)
- Email: Không có trong phiếu (null)

**Thống kê:**
- Tổng số lượng: 11,100 pcs (đã kiểm tra)
- Số loại sản phẩm: 8 loại

**Kết quả JSON:**
{
  "so_phieu": "PXB25110053",
  "ngay": "03/11/2025",
  "nha_cung_cap": {
    "ten": "KOIKEYA VIETNAM CO., LTD",
    "dien_thoai": "+84(0)251 368 1368",
    "fax": "+84(0)251 368 1369",
    "email": null
  },
  "tong_so_luong": 11100,
  "so_loai_san_pham": 8
}"

OUTPUT FORMAT:
{
  "conversation": [
    {
      "user": "<Question requiring reasoning>",
      "assistant": "<Step-by-step reasoning process>\n\n**Kết luận:** <Final answer>"
    }
    // 5-8 turns, 60-70% with explicit reasoning
  ]
}

QUALITY CHECKLIST:
- [ ] 5-8 conversation turns
- [ ] At least 3 turns show explicit step-by-step reasoning
- [ ] Reasoning uses clear structure (bullets, numbering, sections)
- [ ] All calculations shown explicitly with numbers
- [ ] Conclusions clearly labeled
- [ ] Evidence cited with specific locations
- [ ] At least 2 different reasoning patterns used
- [ ] Includes 1-2 verification/checking questions
- [ ] Natural Vietnamese with technical precision
- [ ] Valid outer JSON format

EXAMPLES:

Example 1 - Verification Pattern:
{
  "conversation": [
    {"user": "Số phiếu là gì?", 
     "assistant": "Số phiếu là PXB25110053"},
    {"user": "Kiểm tra xem tổng số lượng có đúng không?",
     "assistant": "Tôi sẽ cộng từng dòng để xác minh:\n\n**Thu thập dữ liệu từ bảng:**\n- Dòng 1 (500VS29): 2,400 pcs\n- Dòng 2 (517VS29): 600 pcs\n- Dòng 3 (519VS29): 600 pcs\n- Dòng 4 (531VS29): 600 pcs\n- Dòng 5 (200VS29): 1,500 pcs\n- Dòng 6 (205VS29): 600 pcs\n- Dòng 7 (210VS29): 1,500 pcs\n- Dòng 8 (228VS29): 1,500 pcs\n\n**Tính tổng:**\n2,400 + 600 + 600 + 600 + 1,500 + 600 + 1,500 + 1,500\n= 2,400 + (600 × 4) + (1,500 × 3)\n= 2,400 + 2,400 + 4,500\n= 9,300 pcs... Khoan đã, để tôi đếm lại dòng có 1,500.\n\nCó 3 dòng 1,500 pcs: dòng 5, 7, 8\nCó 4 dòng 600 pcs: dòng 2, 3, 4, 6\nCó 1 dòng 2,400 pcs: dòng 1\n\n= (1,500 × 3) + (600 × 4) + 2,400\n= 4,500 + 2,400 + 2,400 + 2,400\n= 11,100 pcs ✓\n\n**So sánh với số ghi trên phiếu:**\nTổng ghi ở cuối bảng: 11,100 pcs\n\n**Kết luận:** Tổng số lượng 11,100 pcs là chính xác, khớp với tổng các dòng."},
    {"user": "Sản phẩm nào chiếm tỷ trọng lớn nhất?",
     "assistant": "Để xác định sản phẩm chiếm tỷ trọng lớn nhất:\n\n**Bước 1: Xác định tổng**\nTổng: 11,100 pcs\n\n**Bước 2: Tính tỷ trọng từng sản phẩm**\n- Dòng 1 (2,400 pcs): 2,400/11,100 = 21.6%\n- Dòng 5, 7, 8 (1,500 pcs): 1,500/11,100 = 13.5% (mỗi loại)\n- Dòng 2, 3, 4, 6 (600 pcs): 600/11,100 = 5.4% (mỗi loại)\n\n**Bước 3: So sánh**\n21.6% > 13.5% > 5.4%\n\n**Kết luận:** Snack Bắp Karamucho vị cay (500VS29) chiếm tỷ trọng lớn nhất với 21.6% (2,400/11,100 pcs)."},
    {"user": "Có thông tin email không?",
     "assistant": "Không có thông tin email trong phiếu xuất kho này. Chỉ có số điện thoại và fax."}
  ]
}

Example 2 - Comparative Reasoning:
{
  "conversation": [
    {"user": "So sánh số lượng Snack Bắp và Snack Khoai tây",
     "assistant": "Tôi sẽ so sánh hai nhóm sản phẩm:\n\n**Nhóm Snack Bắp:**\n- Dòng 1: 2,400 pcs\n- Dòng 2: 600 pcs\n- Dòng 3: 600 pcs\n- Dòng 4: 600 pcs\nTổng: 2,400 + 600 + 600 + 600 = 4,200 pcs\n\n**Nhóm Snack Khoai tây:**\n- Dòng 5: 1,500 pcs\n- Dòng 6: 600 pcs\n- Dòng 7: 1,500 pcs\n- Dòng 8: 1,500 pcs\nTổng: 1,500 + 600 + 1,500 + 1,500 = 5,100 pcs\n\n**Phân tích:**\n- Snack Khoai tây: 5,100 pcs (46%)\n- Snack Bắp: 4,200 pcs (38%)\n- Chênh lệch: 900 pcs\n\n**Kết luận:** Snack Khoai tây nhiều hơn Snack Bắp 900 pcs (21% nhiều hơn). Tuy nhiên, Snack Bắp có 1 sản phẩm chiếm số lượng lớn (2,400 pcs), trong khi Khoai tây phân bổ đều hơn."},
    {"user": "Nếu mỗi thùng chứa 24 cái, nhóm nào cần nhiều thùng hơn?",
     "assistant": "**Tính toán số thùng:**\n\nSnack Bắp (4,200 pcs):\n4,200 ÷ 24 = 175 thùng\n\nSnack Khoai tây (5,100 pcs):\n5,100 ÷ 24 = 212.5 → cần 213 thùng\n\n**Phân tích:**\n- Khoai tây: 213 thùng\n- Bắp: 175 thùng\n- Chênh lệch: 38 thùng\n- Khoai tây sẽ có 1 thùng lẻ (12 cái)\n\n**Kết luận:** Snack Khoai tây cần nhiều thùng hơn (213 vs 175), tương ứng với số lượng sản phẩm lớn hơn."}
  ]
}
"""
    return prompt
 
def get_lazy_user_prompt() -> str:
    """Generate conversations with casual, vague, and imperfect user queries to teach context recovery."""
    prompt = """You are an expert in analyzing Vietnamese business documents and creating realistic conversations where users ask questions in CASUAL, VAGUE, IMPERFECT language - like real workplace communication.

TASK:
Analyze the provided document image and generate a realistic multi-turn conversation (5-7 turns) where the USER asks questions using natural, imperfect Vietnamese (like a busy colleague or boss), and the ASSISTANT responds naturally and helpfully while handling ambiguity.

CORE PRINCIPLE: **Real People Don't Talk Like Robots**

USER QUERY TYPES (Mix these naturally):

1. **Ultra-Short Queries** (25-30%):
   - "tổng?"
   - "bao nhiu"
   - "số nào"
   - "ngày mấy"
   - "cái j"
   - "email?"
   - "sao z"
   
   Assistant must infer context from document or previous turns.

2. **Typos & Lower Case** (20-25%):
   - "tổng bnhieu" (bao nhiêu)
   - "cái mã 500vs kia là j" (là gì)
   - "kiểm tr xem" (kiểm tra)
   - "có đúg ko" (đúng không)
   - "sp nào nhìu nhất" (sản phẩm, nhiều)
   - "nha cung cap ten j" (nhà cung cấp tên gì)
   - "ngày tháng bn" (bao nhiêu)
   
   Assistant should understand and respond naturally.

3. **Missing Context** (15-20%):
   - "cái đó là gì?" (đó = what?)
   - "bao nhiêu?" (asking about what quantity?)
   - "đúng không?" (what is correct?)
   - "còn nữa không?"
   - "mấy cái?"
   - "ở đâu?"
   
   Assistant should either:
   - Ask for clarification: "Bạn muốn hỏi về tổng số lượng hay số phiếu?"
   - Infer from context: "Nếu bạn hỏi về tổng số lượng thì là 11,100 pcs."

4. **Pronouns Without Antecedents** (10-15%):
   - "nó là gì?" 
   - "của ai?"
   - "mấy cái kia?"
   - "cái thứ 2"
   - "mục đầu"
   
   Assistant should identify what "nó", "kia", "đầu" refers to from document.

5. **Slang & Colloquialisms** (10-15%):
   - "oke cho tao coi"
   - "xem nào"
   - "cho xíu"
   - "có éo nào" (vulgar - use sparingly)
   - "dzô xem"
   - "check hộ cái"
   
   Assistant responds naturally despite casual tone.

6. **Incomplete Sentences** (10-15%):
   - "còn email thì"
   - "nếu mỗi thùng 24"
   - "so với"
   - "còn lại"
   - "trừ đi"
   
   Assistant completes the thought logically.

7. **Mixed Vietnamese-English** (5-10%):
   - "total là bnhieu"
   - "check lại coi"
   - "list ra"
   - "sort theo quantity"
   - "có issue gì ko"
   
   Assistant can respond in Vietnamese or mirror the code-mixing naturally.

8. **Vague References** (5-10%):
   - "cái cao nhất"
   - "mấy thứ đầu"
   - "chỗ số"
   - "phần dưới"
   - "bên kia"
   
   Assistant should identify specific items/locations.

9. **Implied Questions** (5-10%):
   - "xem tổng" (implicit: tổng là bao nhiêu?)
   - "mã 500" (implicit: mã 500 là gì?)
   - "sai" (implicit: cái gì sai?)
   - "thiếu" (implicit: thiếu cái gì?)
   
   Assistant should understand the implicit question.

10. **Follow-up Without Full Context** (5-10%):
    - User: "tổng bao nhiêu"
    - Assistant: "11,100 pcs"
    - User: "của cái nào" (which one?)
    - Assistant: "Đó là tổng tất cả sản phẩm trong phiếu. Bạn muốn xem tổng của nhóm nào riêng không?"

ASSISTANT RESPONSE STRATEGIES:

**Strategy 1: Direct Answer** (50-60% of unclear queries):
When context is inferrable from document:
- User: "tổng bnhieu"
- Assistant: "Tổng số lượng là 11,100 pcs."

**Strategy 2: Clarifying Question** (20-30%):
When genuinely ambiguous:
- User: "bao nhiêu?"
- Assistant: "Bạn muốn biết tổng số lượng, số loại sản phẩm, hay số tiền?"

**Strategy 3: Multiple Interpretations** (10-15%):
Offer 2-3 possible answers:
- User: "cái đầu"
- Assistant: "Nếu bạn hỏi về sản phẩm đầu tiên thì là Snack Bắp Karamucho (2,400 pcs). Nếu hỏi về số đầu phiếu thì là PXB25110053."

**Strategy 4: Gentle Correction** (5-10%):
Fix typos implicitly:
- User: "sp nào nhìu nhất"
- Assistant: "Sản phẩm nhiều nhất là Snack Bắp Karamucho vị cay với 2,400 pcs."
(Note: Assistant writes correctly but doesn't criticize user's spelling)

ASSISTANT TONE GUIDELINES:

✓ Natural and helpful:
- Conversational, not robotic
- Don't mock typos or casual language
- Stay helpful even with vague questions

✓ Context-aware:
- Reference previous turns
- Remember what user asked before
- Build on conversation history

✓ Concise when user is brief:
- User: "tổng?"
- Assistant: "11,100 pcs." (not a paragraph)

✓ Detailed when user needs clarity:
- User: "cái đó là j"
- Assistant: "Mã 500VS29 là Snack Bắp Karamucho vị cay, sản phẩm có số lượng cao nhất trong phiếu (2,400 pcs)."

CONVERSATION PATTERNS:

**Pattern A: Progressive Laziness** (30%):
```
Turn 1: "Số phiếu là gì?" (formal)
Turn 2: "ngày mấy" (getting casual)
Turn 3: "tổng?" (ultra-short)
Turn 4: "cái nào nhìu nhất" (typos appear)
Turn 5: "check hộ xem đúng ko" (very casual)
```

**Pattern B: Consistently Casual** (25%):
```
All turns use lowercase, typos, slang:
Turn 1: "cho tao xem tổng"
Turn 2: "còn sp thì j"
Turn 3: "cái mã 500 kia"
Turn 4: "có đúg ko"
Turn 5: "oke r"
```

**Pattern C: Ambiguous + Clarification** (25%):
```
Turn 1: "bao nhiêu?" 
→ Assistant asks for clarification
Turn 2: User clarifies or Assistant gives multiple options
Turn 3: "cái đó" (vague pronoun)
→ Assistant infers or asks again
Turn 4: More specific question emerges
```

**Pattern D: Code-Mixing** (20%):
```
Mix Vietnamese and English:
Turn 1: "check total xem"
Turn 2: "list top 3"
Turn 3: "có issue gì ko"
Turn 4: "export json"
```

LAZY USER BEHAVIORS TO SIMULATE:

1. **No punctuation**: "tổng bao nhiêu cho tao xem"
2. **Missing diacritics**: "nha cung cap ten gi" (nhà cung cấp tên gì)
3. **Abbreviations**: "sp" (sản phẩm), "sl" (số lượng), "ncc" (nhà cung cấp)
4. **Teen code**: "bn" (bao nhiêu), "đc" (được), "ko" (không), "cx" (cũng)
5. **Trailing particles**: "...à", "...nha", "...đi"
6. **Impatient tone**: "nhanh lên", "cái gì mà lâu z"
7. **Assuming context**: Ask about "mã 500" without saying it's a product code
8. **Lazy references**: "cái kia", "mấy cái đầu", "chỗ số"

COMMON TYPO PATTERNS (Vietnamese):

- bnhieu → bao nhiêu
- j → gì
- nhìu → nhiều
- ko/k → không
- đc → được
- sp → sản phẩm
- ncc → nhà cung cấp
- sl → số lượng
- bn → bao nhiêu
- cx → cũng
- đúg → đúng
- chính xác → chính xác (might drop tone marks: chinh xac)

ASSISTANT HANDLING STRATEGIES:

**For Typos:**
- Understand and respond correctly
- Don't point out the typo
- Use correct spelling in response

**For Vagueness:**
Option 1 - Infer: "Nếu bạn hỏi về X thì..."
Option 2 - Clarify: "Bạn muốn hỏi về A, B hay C?"
Option 3 - Both: "Tổng số lượng là 11,100 pcs. Nếu bạn hỏi về loại cụ thể thì cho tôi biết thêm nhé."

**For Slang:**
- Stay natural but don't be overly formal
- Don't use vulgar language back (even if user does)
- Respond naturally without being stiff

**For Missing Context:**
- Look at conversation history
- Check document for likely reference
- Make educated guess or ask politely

**For Incomplete Sentences:**
- Complete the thought logically
- User: "còn email thì"
- Assistant: "Phiếu này không có thông tin email, chỉ có số điện thoại và fax."

FORMAT MIXING:

User can request formats casually:
- "làm bảng" → markdown table
- "json đi" → JSON output
- "list ra" → bullet points
- "cho dễ nhìn" → structured format

Assistant should still return proper formatted output.

CONVERSATION LENGTH: 5-7 turns
- Turn 1-2: Casual questions (may need clarification)
- Turn 3-4: Follow-ups (often vague, reference previous context)
- Turn 5-6: Specific requests or corrections
- Turn 7: Optional final check or format request

OUTPUT FORMAT:
{
  "conversation": [
    {
      "user": "<casual/vague/typo-filled query>",
      "assistant": "<natural, helpful response>"
    }
    // 5-7 turns
  ]
}

QUALITY CHECKLIST:
- [ ] 5-7 conversation turns
- [ ] 60-80% of user queries use casual/imperfect language
- [ ] Mix of typos, slang, abbreviations, vagueness
- [ ] Assistant always helpful and natural
- [ ] At least 1-2 clarification moments (Assistant asks or offers options)
- [ ] Assistant demonstrates context recovery (understands vague references)
- [ ] Natural conversation flow (not robotic or overly formal)
- [ ] Valid outer JSON format
- [ ] No mockery of user's casual language

DIVERSITY GUIDELINES:
- Vary the TYPE of laziness (typos vs vagueness vs slang) across conversations
- Don't make every conversation ultra-casual (some can be moderately casual)
- Mix serious business queries with casual tone
- Include both simple extraction and complex questions asked casually
- Show Assistant adapting to user's communication style

EXAMPLES:

Example 1 - Typos & Ultra-Short:
{
  "conversation": [
    {"user": "tổng bnhieu", 
     "assistant": "Tổng số lượng là 11,100 pcs."},
    {"user": "cái mã 500vs kia là j",
     "assistant": "Mã 500VS29 là sản phẩm Snack Bắp Karamucho vị cay, có số lượng 2,400 pcs."},
    {"user": "nhìu nhất đúg ko",
     "assistant": "Đúng rồi, đây là sản phẩm có số lượng cao nhất trong phiếu này."},
    {"user": "còn ncc",
     "assistant": "Nhà cung cấp là KOIKEYA VIETNAM CO., LTD."},
    {"user": "có email ko",
     "assistant": "Không có thông tin email trong phiếu này, chỉ có số điện thoại: +84(0)251 368 1368 và fax."}
  ]
}

Example 2 - Vague References:
{
  "conversation": [
    {"user": "cho xem tổng",
     "assistant": "Tổng số lượng tất cả sản phẩm là 11,100 pcs."},
    {"user": "cái đầu",
     "assistant": "Sản phẩm đầu tiên là Snack Bắp Karamucho vị cay (mã 500VS29), số lượng 2,400 pcs."},
    {"user": "vs cái thứ 2",
     "assistant": "Sản phẩm thứ 2 là Snack Bắp Karamucho vị phô mai (mã 517VS29), số lượng 600 pcs.\n\nSản phẩm đầu nhiều gấp 4 lần sản phẩm thứ 2."},
    {"user": "mấy cái kia",
     "assistant": "Bạn muốn hỏi về các sản phẩm còn lại trong danh sách không? Còn 6 sản phẩm nữa, số lượng từ 600 đến 1,500 pcs."},
    {"user": "ừ list ra",
     "assistant": "Các sản phẩm còn lại:\n\n3. Snack Bắp tương đen - 600 pcs\n4. Snack Bắp phô mai đậm đà - 600 pcs\n5. Snack Khoai tây cay - 1,500 pcs\n6. Snack Khoai tây muối biển - 600 pcs\n7. Snack Khoai tây BBQ - 1,500 pcs\n8. Snack Khoai tây kem chua - 1,500 pcs"}
  ]
}

Example 3 - Code-Mixing:
{
  "conversation": [
    {"user": "check total xem",
     "assistant": "Tổng số lượng là 11,100 pcs."},
    {"user": "list top 3 theo quantity",
     "assistant": "Top 3 sản phẩm theo số lượng:\n\n1. Snack Bắp Karamucho cay - 2,400 pcs\n2. Snack Khoai tây cay - 1,500 pcs\n2. Snack Khoai tây BBQ - 1,500 pcs\n2. Snack Khoai tây kem chua - 1,500 pcs\n\n(Có 3 sản phẩm cùng đứng thứ 2 với 1,500 pcs)"},
    {"user": "có issue gì trong phiếu ko",
     "assistant": "Phiếu này không có vấn đề về mặt số liệu. Tổng số lượng khớp chính xác. Chỉ thiếu thông tin email, nhưng không phải trường bắt buộc."},
    {"user": "oke export json đi",
     "assistant": "{\"so_phieu\": \"PXB25110053\", \"ngay\": \"03/11/2025\", \"nha_cung_cap\": \"KOIKEYA VIETNAM\", \"tong_so_luong\": 11100, \"don_vi\": \"pcs\", \"email\": null}"}
  ]
}

Example 4 - Ambiguous + Clarification:
{
  "conversation": [
    {"user": "bao nhiêu?",
     "assistant": "Bạn muốn biết tổng số lượng sản phẩm hay tổng số loại sản phẩm?"},
    {"user": "sl",
     "assistant": "Tổng số lượng là 11,100 pcs."},
    {"user": "đúng không",
     "assistant": "Để tôi kiểm tra lại: tổng các dòng từ 1 đến 8 = 2,400 + 600 + 600 + 600 + 1,500 + 600 + 1,500 + 1,500 = 11,100 pcs.\n\nĐúng rồi, số liệu khớp chính xác."},
    {"user": "thế còn",
     "assistant": "Bạn muốn kiểm tra thêm thông tin gì nữa? Tôi có thể xem nhà cung cấp, ngày tháng, hoặc chi tiết từng sản phẩm."},
    {"user": "thôi oke r",
     "assistant": "Ok, cần gì thêm cứ hỏi nhé!"}
  ]
}

Example 5 - Impatient Boss:
{
  "conversation": [
    {"user": "ê số phiếu là j",
     "assistant": "Số phiếu là PXB25110053."},
    {"user": "ngày mấy",
     "assistant": "Ngày 03/11/2025."},
    {"user": "có đúg hem, check lại",
     "assistant": "Tôi kiểm tra lại:\n- Tổng các dòng: 11,100 pcs ✓\n- Ngày tháng: 03/11/2025 ✓\n- Thông tin nhà cung cấp: đầy đủ ✓\n\nTất cả đều chính xác."},
    {"user": "mã 500 kia tên đầy đủ",
     "assistant": "Mã 500VS29: Snack Bắp Karamucho vị cay (KARAMUCHO CORN SNACK SPICY FLAVOR), số lượng 2,400 pcs."},
    {"user": "oke",
     "assistant": "Ok, có gì cần thêm cứ bảo!"}
  ]
}

CRITICAL RULES:
1. User queries must feel NATURAL, not artificially dumbed down
2. Mix different types of laziness (don't use only typos or only vagueness)
3. Assistant NEVER mocks or points out user's casual language
4. Assistant stays natural and helpful regardless of user's tone
5. Balance clarity and conciseness based on query complexity
6. Show Assistant's context recovery abilities
7. Respond naturally without being overly formal or robotic
8. Make conversations feel like real workplace chat, not training exercises
"""
    return prompt
 
def get_visual_spatial_correction_prompt() -> str:
   """Generate conversations focused on spatial awareness and graceful error correction."""
   prompt = """You are an expert in analyzing Vietnamese business documents with focus on VISUAL LAYOUT UNDERSTANDING and COLLABORATIVE ERROR CORRECTION.

TASK:
Analyze the provided document image and generate a realistic multi-turn conversation (5-8 turns) where:
1. Users ask about SPATIAL POSITIONS, VISUAL FEATURES, and LAYOUT
2. Users challenge or correct the assistant's answers (simulating OCR errors or misreadings)
3. Assistant demonstrates spatial awareness AND graceful error handling

CORE PRINCIPLES:

**Principle 1: See the Document as a 2D Space**
Don't just read text - understand WHERE things are, HOW they look, and WHAT surrounds them.

**Principle 2: Collaborate, Don't Defend**
When corrected, check again, admit mistakes gracefully, and explain ambiguities without being defensive.

---

PART A: VISUAL GROUNDING & SPATIAL AWARENESS (60% of conversation)

**1. POSITION QUERIES** (25-30%):

Users ask WHERE something is located:

a) **Corner/Edge Positioning**:
   - "Số phiếu nằm ở góc nào?"
   - "Logo ở đâu trong trang?"
   - "Chữ ký nằm bên nào?"
   
   Assistant MUST specify:
   - Corner: "góc trên bên phải", "góc dưới trái"
   - Edge: "ở đầu trang", "cuối trang", "bên trái giữa"
   - Relative: "phía trên bảng chi tiết", "ngay dưới tiêu đề"
   
   Example:
   User: "Số phiếu ở đâu?"
   Assistant: "Số phiếu PXB25110053 nằm ở góc trên bên phải của trang, phía trên thông tin nhà cung cấp."

b) **Section/Zone Positioning**:
   - "Thông tin nhà cung cấp ở phần nào?"
   - "Bảng sản phẩm nằm chỗ nào?"
   - "Footer có những gì?"
   
   Assistant describes document structure:
   - Header (đầu trang)
   - Body (thân tài liệu)
   - Footer (chân trang)
   - Sidebar (nếu có)
   
   Example:
   User: "Thông tin liên hệ nằm ở đâu?"
   Assistant: "Thông tin liên hệ của KOIKEYA nằm ở phần header bên trái, bao gồm địa chỉ, điện thoại và fax. Ngay dưới logo công ty."

c) **Relative Positioning**:
   - "Tổng cộng nằm so với bảng thế nào?"
   - "Ngày tháng ở trên hay dưới số phiếu?"
   - "Con dấu gần chỗ nào?"
   
   Assistant uses relational terms:
   - Trên/dưới, trái/phải
   - Gần/xa, bên cạnh
   - Trong/ngoài (bảng/khung)
   
   Example:
   User: "Tổng số lượng 11,100 ở đâu?"
   Assistant: "Số 11,100 pcs nằm ở dòng cuối cùng của bảng, bên phải cột 'Số lượng', được in đậm để dễ nhận biết."

**2. VISUAL FEATURE QUERIES** (20-25%):

Users ask about HOW things LOOK:

a) **Typography & Formatting**:
   - "Phần nào được in đậm?"
   - "Tiêu đề có khác màu không?"
   - "Chữ nào lớn nhất?"
   - "Font chữ có khác nhau không?"
   
   Assistant describes:
   - Bold/regular/italic
   - Font size (lớn/nhỏ/trung bình)
   - Color (if visible in document)
   - Underline/highlight
   
   Example:
   User: "Phần tiêu đề có được nhấn mạnh không?"
   Assistant: "Có, dòng 'PHIẾU XUẤT KHO' ở đầu trang được in đậm và cỡ chữ lớn hơn các phần khác, rất dễ nhận ra."

b) **Visual Elements**:
   - "Có logo không? Màu gì?"
   - "Có con dấu đỏ không?"
   - "Chữ ký ở đâu? Rõ không?"
   - "Có hình ảnh sản phẩm không?"
   - "Có mã vạch/QR code không?"
   
   Assistant describes what's visible:
   - Logos, stamps, signatures
   - Color (red stamp, blue logo)
   - Clarity (rõ nét, mờ, nhòe)
   - Presence/absence
   
   Example:
   User: "Có con dấu xác nhận không?"
   Assistant: "Trong ảnh này tôi không thấy con dấu đỏ. Có thể đây là bản scan trước khi ký duyệt, hoặc con dấu nằm ở trang khác."

c) **Layout Structure**:
   - "Tài liệu có mấy cột?"
   - "Bảng có bao nhiêu dòng?"
   - "Trang này có chia thành mấy phần?"
   - "Có khung viền không?"
   
   Assistant analyzes structure:
   - Number of columns/rows
   - Tables, boxes, borders
   - Sections and divisions
   
   Example:
   User: "Bảng sản phẩm có mấy cột?"
   Assistant: "Bảng có 4 cột: STT, Mã số, Tên hàng hóa, và Số lượng. Tất cả được căn chỉnh rõ ràng với đường kẻ ngang dọc."

**3. ZONE-SPECIFIC READING** (15-20%):

Users ask to read SPECIFIC AREAS:

a) **By Location**:
   - "Đọc cho tôi phần footer"
   - "Dòng chữ nhỏ ở dưới cùng là gì?"
   - "Góc trên trái có ghi gì?"
   - "Phần header bên phải nội dung gì?"
   
   Assistant reads only that zone:
   
   Example:
   User: "Đọc footer cho tôi"
   Assistant: "Phần footer ở dưới cùng trang ghi:\n'Địa chỉ: N3-2 St., Long Duc IP, Long Thanh, Dong Nai'\n'Tel: +84(0)251 368 1368  Fax: +84(0)251 368 1369'\nChữ nhỏ hơn so với phần chính của tài liệu."

b) **By Visual Cue**:
   - "Dòng chữ in đậm ở giữa trang là gì?"
   - "Phần được đánh dấu màu vàng nói gì?"
   - "Chữ lớn nhất trên trang ghi gì?"
   
   Example:
   User: "Số nào được viết lớn nhất?"
   Assistant: "Số 11,100 ở dòng tổng cộng được viết cỡ chữ lớn nhất và in đậm, nằm ở cuối bảng."

c) **By Surrounding Context**:
   - "Dòng ngay dưới logo là gì?"
   - "Số bên cạnh tên công ty là gì?"
   - "Text trong khung viền nội dung gì?"
   
   Example:
   User: "Ngay dưới tên công ty KOIKEYA có ghi gì?"
   Assistant: "Ngay dưới tên công ty có ghi địa chỉ đầy đủ: 'N3-2 St., Long Duc IP, Long Thanh, Dong Nai'."

**4. SPATIAL REASONING** (10-15%):

Users ask questions requiring spatial logic:

a) **Distance/Proximity**:
   - "Số phiếu và ngày tháng có gần nhau không?"
   - "Logo cách tiêu đề bao xa?"
   - "Chữ ký và con dấu có chồng lên nhau không?"
   
   Assistant judges spatial relationships:
   
   Example:
   User: "Số phiếu và ngày tháng có ở cùng vùng không?"
   Assistant: "Có, cả hai đều nằm ở phần header trên cùng trang, cách nhau chỉ vài dòng. Số phiếu bên phải, ngày tháng ở dòng đầu bảng bên trái."

b) **Alignment**:
   - "Các số ở cột số lượng có thẳng hàng không?"
   - "Tên sản phẩm có căn trái hay căn giữa?"
   - "Tổng cộng có được căn phải không?"
   
   Example:
   User: "Số lượng trong bảng có căn chỉnh không?"
   Assistant: "Có, tất cả số lượng trong cột 'Số lượng' được căn phải (right-aligned), giúp dễ dàng so sánh và cộng dồn."

c) **Hierarchy**:
   - "Thông tin nào nổi bật nhất trên trang?"
   - "Phần nào được ưu tiên về mặt thị giác?"
   - "Mắt nhìn vào sẽ thấy gì đầu tiên?"
   
   Example:
   User: "Nhìn vào trang này điều gì thu hút mắt nhất?"
   Assistant: "Tiêu đề 'PHIẾU XUẤT KHO' ở đầu trang là nổi bật nhất do cỡ chữ lớn và in đậm. Sau đó là số phiếu PXB25110053 ở góc phải và logo KOIKEYA."

---

PART B: SELF-CORRECTION & OCR ERROR HANDLING (40% of conversation)

**1. USER CHALLENGES ASSISTANT** (20-25%):

Simulate user correcting the assistant's reading:

a) **Number Confusion**:
   User: "Sai rồi, số này là 8 chứ không phải 3"
   
   Assistant responses:
   
   **If truly wrong** (Assistant made OCR error):
   "Để mình kiểm tra lại kỹ hơn... Bạn nói đúng, đây là số 8 (ở vị trí X, dòng Y). Mình đọc nhầm do chữ viết hơi nghiêng. Cảm ơn bạn đã chỉnh sửa!"
   
   **If ambiguous** (Could be either):
   "Mình hiểu, vùng chữ số này hơi mờ. Nhìn kỹ lại, có thể là số 8 như bạn nói. Nếu bạn chắc chắn là 8 thì mình ghi nhận."
   
   **If assistant was correct** (User misread):
   "Để mình xác nhận lại... Mình thấy ở vị trí này là số 3 (góc trên bên phải, dòng thứ 2). Có thể bạn đang nhìn số ở vị trí khác? Số 8 xuất hiện ở dòng 5 bên trái."

b) **Text Misreading**:
   User: "Không phải 'Long Duc' mà là 'Long Due', OCR sai rồi"
   
   Assistant: "Cảm ơn bạn đã để ý! Để mình nhìn kỹ lại phần địa chỉ... Chữ 'c' cuối từ 'Duc' hơi mờ và nhỏ nên dễ nhầm. Nếu tài liệu gốc ghi 'Due' thì đúng như bạn nói. Mình sẽ cập nhật thành 'Long Due' theo góp ý của bạn."

c) **Field Mix-up**:
   User: "Ê, đó là mã khách hàng chứ không phải số phiếu"
   
   Assistant: "Ôi, xin lỗi! Bạn nói đúng rồi. Mình nhầm lẫn giữa hai trường. Số phiếu thực tế là PXB25110053 ở góc phải, còn mã mà mình đọc lúc nãy là mã khách hàng ở vị trí khác. Cảm ơn bạn đã chỉ ra!"

**2. USER REQUESTS RE-CHECK** (10-15%):

User asks assistant to verify again:

a) **Vague Request**:
   User: "Đọc lại xem"
   
   Assistant: "Ok, để mình kiểm tra lại kỹ hơn... [re-reads the relevant section]. Mình confirm lại: [correct information]. Nếu vẫn thấy không khớp thì bạn chỉ cho mình vị trí cụ thể nhé."

b) **Specific Zone**:
   User: "Cái dòng địa chỉ OCR bị lỗi font, đọc lại đi"
   
   Assistant: "Ok, để mình nhìn kỹ lại phần địa chỉ ở [vị trí cụ thể]... Chữ hơi mờ và font đặc biệt nên dễ nhầm. Địa chỉ chính xác là: 'N3-2 St., Long Duc IP, Long Thanh, Dong Nai'. Bạn thấy khớp chưa?"

c) **Uncertain Reading**:
   User: "Chữ này nhòe quá, bạn chắc không?"
   
   Assistant: "Thực sự phần này chữ rất mờ. Mình đọc được là '[X]' nhưng độ chắc chắn chỉ khoảng 70%. Nếu bạn có tài liệu gốc rõ hơn hoặc biết ngữ cảnh thì có thể giúp mình xác nhận chính xác hơn được không?"

**3. ASSISTANT SELF-CORRECTION** (5-10%):

Assistant proactively catches own errors:

a) **Mid-conversation Correction**:
   Assistant: "Khoan, để mình kiểm tra lại... Mình vừa nói tổng là 11,000 nhưng nhìn lại kỹ thì là 11,100 mới đúng. Xin lỗi bạn về sự nhầm lẫn!"

b) **Clarifying Ambiguity**:
   Assistant: "Số này hơi khó đọc do bị mờ. Mình thấy giống số 5 nhưng cũng có thể là số 6. Bạn thấy rõ hơn mình không?"

c) **Acknowledging Limitations**:
   Assistant: "Phần chữ ký này mình không đọc được do viết tay và khá nhòe. Mình chỉ thấy có chữ ký ở góc dưới bên phải nhưng không thể OCR được tên người ký."

**ERROR HANDLING STRATEGIES:**

**When User Says "Wrong":**
1. Don't argue immediately
2. Check again the specific area mentioned
3. Options:
   - Admit mistake: "Bạn đúng rồi, mình đọc nhầm..."
   - Explain ambiguity: "Phần này hơi mờ, có thể là..."
   - Clarify location: "Mình đang nhìn vị trí X, bạn có thể chỉ rõ vị trí bạn thấy không?"

**Tone Guidelines:**
- Humble, not defensive
- Collaborative: "Cùng nhau kiểm tra lại nhé"
- Grateful: "Cảm ơn bạn đã chỉ ra"
- Natural: "Xin lỗi về sự nhầm lẫn"

**DON'T:**
- ✗ "Mình chắc chắn 100% là đúng"
- ✗ "Bạn nhìn sai rồi"
- ✗ "Không thể nào sai được"
- ✗ Ignore user's correction

**DO:**
- ✓ "Để mình kiểm tra lại..."
- ✓ "Bạn nói đúng, mình nhầm..."
- ✓ "Phần này hơi khó đọc, mình thấy là..."
- ✓ "Cảm ơn bạn đã góp ý"

---

CONVERSATION PATTERNS:

**Pattern A: Spatial Exploration (40%)**:
```
Turn 1: User asks WHERE something is
→ Assistant describes position

Turn 2: User asks about VISUAL FEATURE
→ Assistant describes appearance

Turn 3: User asks to read SPECIFIC ZONE
→ Assistant reads that area

Turn 4: User challenges a detail
→ Assistant checks, corrects/clarifies
```

**Pattern B: Correction-Heavy (35%)**:
Turn 1: Assistant provides info
Turn 2: User: "Sai rồi, đây là X chứ không phải Y"
→ Assistant re-checks, admits error gracefully
Turn 3: User: "Đọc lại chỗ này xem"
→ Assistant re-reads carefully
Turn 4: User: "Bạn chắc không?"
→ Assistant expresses confidence level

**Pattern C: Mixed Spatial + Correction (25%)**:
Turn 1: User asks about position
Turn 2: User asks about visual feature
Turn 3: User challenges assistant's reading
Turn 4: Assistant corrects, user asks zone-specific read
Turn 5: Collaborative verification

CONVERSATION LENGTH: 5-8 turns
- 3-4 turns: Spatial awareness questions
- 2-3 turns: Error correction scenarios
- 1 turn: Optional final verification

OUTPUT FORMAT:
{
  "conversation": [
    {
      "user": "<Question about position/visual OR correction challenge>",
      "assistant": "<Spatial description OR graceful error handling>"
    }
    // 5-8 turns
  ]
}

QUALITY CHECKLIST:
- [ ] 5-8 conversation turns
- [ ] 60% spatial awareness (position, visual features, zones)
- [ ] 40% error correction (challenges, re-checks, self-correction)
- [ ] Assistant describes positions using directional terms (góc trên phải, phía dưới, etc.)
- [ ] Assistant describes visual features (đậm, mờ, lớn, nhỏ, etc.)
- [ ] At least 1-2 user correction challenges
- [ ] Assistant responds gracefully to corrections (no defensiveness)
- [ ] Uses collaborative language ("để mình kiểm tra lại", "cảm ơn bạn")
- [ ] Shows spatial reasoning (relative positions, alignment, hierarchy)
- [ ] Valid outer JSON format

DIVERSITY GUIDELINES:
- Vary spatial questions: corners, zones, relative positions, visual features
- Vary correction types: numbers, text, fields, ambiguous characters
- Mix confident corrections with uncertain readings
- Show different error responses: admit wrong, explain ambiguity, clarify location
- Balance between "assistant was wrong" and "user misunderstood" scenarios
- Make spatial descriptions natural, not robotic

EXAMPLES:

Example 1 - Spatial Focus:
{
  "conversation": [
    {"user": "Số phiếu nằm ở đâu trong trang?",
     "assistant": "Số phiếu PXB25110053 nằm ở góc trên bên phải của trang, ngay phía trên thông tin nhà cung cấp KOIKEYA."},
    {"user": "Phần tiêu đề có được in đậm không?",
     "assistant": "Có, dòng 'PHIẾU XUẤT KHO' ở đầu trang được in đậm và cỡ chữ lớn hơn hẳn các phần khác, rất dễ nhận ra."},
    {"user": "Có con dấu đỏ không?",
     "assistant": "Trong ảnh này mình không thấy con dấu màu đỏ. Có thể đây là bản draft chưa ký duyệt, hoặc con dấu nằm ở trang khác của tài liệu."},
    {"user": "Tổng số lượng 11,100 nằm ở đâu?",
     "assistant": "Số 11,100 pcs nằm ở dòng cuối cùng của bảng, cột bên phải nhất (cột Số lượng), được in đậm để nổi bật."},
    {"user": "Đọc cho tôi phần footer",
     "assistant": "Phần footer ở dưới cùng trang ghi:\n'Địa chỉ: N3-2 St., Long Duc IP, Long Thanh, Dong Nai'\n'Tel: +84(0)251 368 1368  Fax: +84(0)251 368 1369'\n\nChữ nhỏ hơn phần chính, được căn trái."}
  ]
}

Example 2 - Correction Focus:
{
  "conversation": [
    {"user": "Tổng số lượng là bao nhiêu?",
     "assistant": "Tổng số lượng là 11,100 pcs."},
    {"user": "Sai rồi, em nhìn lại xem, đó là 11,000 chứ không phải 11,100",
     "assistant": "Để mình kiểm tra lại kỹ hơn... Mình nhìn ở dòng cuối cùng của bảng, cột Số lượng. Chữ số ghi rõ là '11,100' (có ba số 1 liên tiếp và hai số 0). Có thể bạn đang nhìn số khác hoặc phần chữ số '1' giữa hơi mờ nên nhìn nhầm thành '0'?"},
    {"user": "À ừ nhỉ, có ba số 1. Vậy mã sản phẩm đầu là gì?",
     "assistant": "Mã sản phẩm đầu tiên là 500VS29."},
    {"user": "Đọc lại xem, có phải 500VS29 không hay là 500VS39?",
     "assistant": "Để mình nhìn kỹ lại ký tự cuối... Đúng là số 2, không phải số 3. Mã chính xác là 500VS29. Chữ số '2' viết hơi tròn nên dễ nhầm thành '3', nhưng mình chắc chắn đây là số 2."},
    {"user": "Ok, cảm ơn nhé",
     "assistant": "Không có gì! Cảm ơn bạn đã kiểm tra cùng mình cho chắc chắn."}
  ]
}

Example 3 - Mixed Spatial + Correction:
{
  "conversation": [
    {"user": "Logo công ty ở đâu?",
     "assistant": "Logo KOIKEYA nằm ở góc trên bên trái của trang, phía trên thông tin địa chỉ công ty."},
    {"user": "Bảng sản phẩm có mấy cột?",
     "assistant": "Bảng có 4 cột: STT, Mã số, Tên hàng hóa, và Số lượng. Các cột được căn chỉnh rõ ràng với đường kẻ dọc."},
    {"user": "Số điện thoại là bao nhiêu?",
     "assistant": "Số điện thoại là +84(0)251 368 1368."},
    {"user": "Sai rồi, số cuối là 1369 chứ không phải 1368",
     "assistant": "Để mình kiểm tra lại... Ồ bạn nói đúng! Mình nhầm lẫn. Số +84(0)251 368 1368 là số điện thoại (Tel), còn +84(0)251 368 1369 là số fax (Fax). Mình đọc nhầm dòng. Cảm ơn bạn đã chỉ ra!"},
    {"user": "Vậy cả hai số đều có đúng không?",
     "assistant": "Đúng rồi. Cả hai đều có trong phần footer:\n- Tel (điện thoại): +84(0)251 368 1368\n- Fax: +84(0)251 368 1369\n\nHai số chỉ khác nhau ở chữ số cuối cùng (8 vs 9)."},
    {"user": "Hai số này nằm cách nhau bao xa?",
     "assistant": "Hai số nằm cùng một dòng ở footer, cách nhau khoảng vài từ. Tel ở bên trái, Fax ở bên phải, cùng phần thông tin liên hệ."}
  ]
}

Example 4 - OCR Ambiguity:
{
  "conversation": [
    {"user": "Địa chỉ của công ty là gì?",
     "assistant": "Địa chỉ là: N3-2 St., Long Duc IP, Long Thanh, Dong Nai."},
    {"user": "Cái 'Duc' đó viết đúng chưa? Chữ 'c' hơi mờ",
     "assistant": "Đúng, mình thấy chữ 'c' ở cuối từ 'Duc' hơi nhỏ và mờ một chút. Nhưng nhìn kỹ thì mình thấy đây là chữ 'c' không phải 'e' hay chữ khác. Nếu bạn có tài liệu gốc rõ hơn thì có thể double-check giúp mình được không?"},
    {"user": "Chữ đầu của 'Long' là L hay I?",
     "assistant": "Đó là chữ L (L hoa), không phải chữ I. Mặc dù trong một số font chữ L và I trông giống nhau, nhưng ngữ cảnh 'Long Duc IP' (khu công nghiệp) thì chắc chắn là chữ L."},
    {"user": "Số N3-2 có chắc không? Có phải N8 không?",
     "assistant": "Để mình nhìn kỹ lại... Chữ số sau chữ N hơi nghiêng nhưng mình thấy đây là số 3, không phải số 8. Số 3 có góc cạnh rõ, còn số 8 sẽ tròn hơn. Mình khá chắc chắn đây là N3-2."},
    {"user": "Ok cảm ơn. Ngày tháng là ngày mấy?",
     "assistant": "Ngày 03/11/2025, nằm ở dòng đầu của bảng sản phẩm."}
  ]
}

Example 5 - Self-Correction:
{
  "conversation": [
    {"user": "Sản phẩm đầu tiên có số lượng bao nhiêu?",
     "assistant": "Sản phẩm đầu tiên có số lượng 2,400 pcs."},
    {"user": "Và sản phẩm thứ hai?",
     "assistant": "Sản phẩm thứ hai có 600 pcs... Khoan, để mình confirm lại. Đúng rồi, sản phẩm thứ hai (mã 517VS29) có 600 pcs."},
    {"user": "Tổng cả hai sản phẩm là bao nhiêu?",
     "assistant": "Tổng của hai sản phẩm đầu: 2,400 + 600 = 3,000 pcs."},
    {"user": "Bạn chắc chắn không?",
     "assistant": "Để mình tính lại: 2,400 + 600... Đúng là 3,000 pcs. Mình đã kiểm tra lại phép cộng và các số từ bảng. Nếu bạn thấy khác thì cho mình biết nhé!"},
    {"user": "Ok không sao, tôi cũng tính ra 3,000",
     "assistant": "Ok, cảm ơn bạn đã xác nhận lại cùng mình!"}
  ]
}

CRITICAL RULES:
1. **Spatial descriptions must be specific**: Use góc/cạnh/trên/dưới/trái/phải/giữa
2. Visual descriptions must be observable: đậm/mờ/lớn/nhỏ/màu/font
3. Never defend wrong answers stubbornly: Always "để em kiểm tra lại"
4. Admit mistakes gracefully: "Bạn đúng rồi, em nhầm..."
5. Explain ambiguities honestly: "Phần này hơi mờ, có thể là..."
6. Collaborate, don't lecture: "Cùng kiểm tra nhé" not "Em chắc chắn"
7. Show gratitude for corrections: "Cảm ơn bạn đã chỉ ra"
8. Use location references when re-checking: "ở góc X, dòng Y"
9. Express confidence levels when uncertain: "Em khá chắc" vs "Em 100% chắc"
10. Proactively self-correct when catching errors mid-conversation
"""
   return prompt