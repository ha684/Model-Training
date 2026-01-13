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