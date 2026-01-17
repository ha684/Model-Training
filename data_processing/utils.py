import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import json_repair
from pdf2image import convert_from_path
from PIL import Image

from data_processing.apikey_manager import ProcessingCheckpoint

logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str) -> ProcessingCheckpoint:
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
            logger.warning(
                f"Failed to repair/parse JSON response: {e} \n Response text: {response_text}"
            )
            return None


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_conversation_prompt(output_language: str = "vi") -> str:
    """Generate diverse format conversations for VLM training with hard negatives and logic validation.

    Args:
        output_language: Language code for output conversations. Examples: "vi", "en", "zh", "ja", etc.
    """

    # Language-specific examples based on output_language
    if output_language == "vi":
        lang_examples = {
            "question_total": "Tổng số lượng là bao nhiêu?",
            "answer_total": "Tổng cộng có 11.100 sản phẩm trong phiếu này.",
            "question_json": "Lấy thông tin nhà cung cấp dưới dạng JSON nhé",
            "question_table": "Liệt kê sản phẩm dưới dạng bảng",
            "question_summary": "Cho tôi thông tin tóm tắt về phiếu",
            "question_comparison": "So sánh số lượng 3 sản phẩm đầu",
            "question_items": "Các mặt hàng có những gì?",
            "question_structure": "Tổng quan cấu trúc phiếu",
            "question_csv": "Export danh sách CSV",
            "question_faq": "Cho tôi biết: số phiếu? ngày? email? tổng?",
            "missing_field": "Không có thông tin email trong phiếu",
            "field_not_found": "Tài liệu không đề cập",
        }
    elif output_language == "en":
        lang_examples = {
            "question_total": "What is the total quantity?",
            "answer_total": "The total is 11,100 products in this document.",
            "question_json": "Get supplier information in JSON format",
            "question_table": "List products as a table",
            "question_summary": "Give me a summary of this document",
            "question_comparison": "Compare the top 3 products",
            "question_items": "What items are there?",
            "question_structure": "Overview of document structure",
            "question_csv": "Export list as CSV",
            "question_faq": "Tell me: document number? date? email? total?",
            "missing_field": "No email information in the document",
            "field_not_found": "Document does not mention",
        }
    else:
        # Default to English for unsupported languages
        lang_examples = {
            "question_total": "What is the total quantity?",
            "answer_total": "The total is 11,100 products in this document.",
            "question_json": "Get supplier information in JSON format",
            "question_table": "List products as a table",
            "question_summary": "Give me a summary of this document",
            "question_comparison": "Compare the top 3 products",
            "question_items": "What items are there?",
            "question_structure": "Overview of document structure",
            "question_csv": "Export list as CSV",
            "question_faq": "Tell me: document number? date? email? total?",
            "missing_field": "No information available in the document",
            "field_not_found": "Document does not mention",
        }

    prompt = f"""You are an expert in analyzing business documents and creating natural, realistic conversations with DIVERSE OUTPUT FORMATS and CRITICAL REASONING.

CRITICAL: ALL conversations (both user questions and assistant answers) MUST be generated in this language: {output_language.upper()}

TASK:
Analyze the provided document image and generate a realistic multi-turn conversation (4-8 turns) between a user asking questions and an AI assistant answering in VARIOUS FORMATS.

TASK:
Analyze the provided document image and generate a realistic multi-turn conversation (4-8 turns) between a user asking questions and an AI assistant answering in VARIOUS FORMATS.

CORE PRINCIPLES:

1. **Format Diversity** (CRITICAL for VLM training):
   
   Mix these formats naturally across conversations:
   
   a) **Natural Text Response** (25-30%):
      User asks about a quantity, assistant answers in natural language.
   
   b) **JSON Format** (20-25%):
      User requests structured data in JSON format.
      Assistant returns raw JSON without markdown code blocks.
      Example structure: {{"name": "Company", "phone": "+123456", "email": null}}
   
   c) **Markdown Tables** (15-20%):
      User asks to list items as a table.
      Assistant creates a properly formatted markdown table with headers and alignment.
   
   d) **Structured Key-Value Pairs** (10-15%):
      User requests summary information.
      Assistant formats as key-value pairs with clear labels and alignment.
   
   e) **Summary + Detail Format** (8-12%):
      User asks for an overview.
      Assistant provides bold summary followed by nested bullet points with details.
   
   f) **Comparison / Side-by-side** (5-10%):
      User asks to compare items.
      Assistant shows side-by-side comparison with aligned numbers and conclusion.
   
   g) **Markdown Lists** (5-8%):
      User asks what items exist.
      Assistant returns numbered or bulleted list.
   
   h) **Hierarchical Nested Text** (3-5%):
      User asks for structure overview.
      Assistant uses tree-style formatting with branches (├─, └─) showing hierarchy.
   
   i) **CSV Format** (2-5%):
      User requests CSV export.
      Assistant returns comma-separated values with proper escaping.
   
   j) **Q&A / FAQ Style** (2-5%):
      User asks multiple questions at once.
      Assistant formats as Q: and A: pairs.

2. **Format Mixing Strategies**:
   
   Each conversation should use 2-4 different formats naturally:
   
   **Strategy A: Progressive Enhancement** (30%):
   Turn 1: Text answer
   Turn 2: User requests table format → assistant provides markdown table
   Turn 3: User requests JSON → assistant provides JSON
   Turn 4: User asks for comparison → assistant shows side-by-side
   
   **Strategy B: Format-First** (25%):
   Turn 1: User asks for all info as JSON → assistant returns full JSON
   Turn 2: User asks about specific field → assistant handles null values
   Turn 3: User requests table format → assistant provides markdown table
   Turn 4: User asks for summary → assistant gives summary+detail
   
   **Strategy C: Mixed Query** (25%):
   Turn 1: User asks about total → text response
   Turn 2: User requests item list → list format
   Turn 3: User asks to compare items → comparison format
   Turn 4: User requests CSV export → CSV format
   
   **Strategy D: Deep Dive** (20%):
   Turn 1: User asks for summary → summary format
   Turn 2: User requests detailed table → markdown table
   Turn 3: User asks for supplier info → key-value format
   Turn 4: User asks about missing fields → null checking
   Turn 5: User requests complete JSON → full JSON

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
      
      Example patterns (adapt to actual document):
      - Ask if product code contains a number that actually belongs to a different field
      - Ask if total is a number that's actually a line item
      - Ask if document number ends with digits from customer code
      
      Correct Assistant responses should:
      - Clearly identify what field the confused number actually belongs to
      - State the correct value for the asked field
      - Specify the location in the document
   
   b) **Unit Confusion** - Testing if model confuses measurement units:
      - Identify all units used (pcs, boxes, kg, liters, etc.)
      - Ask about items using wrong but plausible units
      - Principles:
        • Confuse container units (boxes, cartons) with piece units
        • Confuse count units with weight/volume units
        • Ask if measurements are in units not present in document
      
      Correct responses should:
      - State the actual unit shown in the document
      - Clarify what the unit represents
      - Note if the asked unit doesn't appear
   
   c) **Field Confusion** - Testing if model confuses similar field names:
      - Identify fields with similar meanings (e.g., "Customer Code" vs "Invoice Number", "Created Date" vs "Delivery Date")
      - Create questions that conflate these fields
      - Principles:
        • Confuse customer/supplier codes with invoice numbers
        • Confuse creation date with delivery date
        • Confuse subtotal with total
        • Confuse unit price with total price
      
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
   
   e) **Ambiguous Queries** - Testing interpretation:
      - Ask intentionally vague questions that could refer to multiple things
      - Principles:
        • Use pronouns without clear antecedents
        • Ask about "first/last" without specifying what list
        • Use relative terms without anchors
      
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
      User asks to verify if line item totals match the stated total.
      Assistant should: Add up line items, compare with stated total, confirm match/mismatch
      
   b) **Subtotal Validation**:
      User asks to calculate the subtotal of a specific group.
      Assistant should: Identify relevant rows, sum them, present result
      
   c) **Cross-field Calculation**:
      User asks hypothetical calculation (e.g., "If each box contains 24 pieces, how many pieces total?")
      Assistant should: Multiply stated quantity by conversion factor
      
   d) **Comparison Logic**:
      User asks which item is larger and by how much.
      Assistant should: Extract both values, compare, calculate difference
      
   e) **Percentage Calculation**:
      User asks what percentage one item represents of the total.
      Assistant should: Calculate ratio, present as percentage
      
   f) **Consistency Check**:
      User asks if quantities in table match description in header.
      Assistant should: Compare counts from different sections
   
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
   User asks about email → Assistant responds that email information is not available
   User asks about shipping fee note → Assistant responds that document doesn't mention it
   
   JSON format:
   User requests email and fax in JSON → Assistant returns {{"email": null, "fax": "+84251368"}}
   
   Key-value format:
   Email: Not available
   Fax: +84(0)251 368 1369
   
   In tables:
   | Field | Value |
   | Email | Not available |

6. **Question Type Variety**:
   
   **Basic Extraction** (20-25%):
   Ask for document number, supplier information, etc.
   
   **Format-specific requests** (20-25%):
   Request JSON format, markdown table, CSV export, structured summary
   
   **Arithmetic & Comparison** (15-25%):
   Ask about totals, comparisons, highest/lowest items, verification
   
   **Hard Negatives** (15-25%):
   Number confusion, unit confusion, field confusion, location-based tricks
   
   **Transformation** (10-15%):
   Request format conversion (text to table, table to JSON, etc.)
   
   **Verification & Negative** (10-15%):
   Ask about missing fields, what fields are null, verification questions

7. **Answer Quality by Format**:
   
   **Text responses:**
   - Natural language, concise
   - For corrections: Clearly state what's wrong, provide correct answer, cite location
   
   **JSON responses:**
   - Raw JSON, no code blocks
   - Explicit null for missing fields
   - Consistent naming conventions
   
   **Markdown tables:**
   - Proper column alignment
   - Clear headers
   - Include units where appropriate
   
   **Key-Value:**
   - Clear labels
   - Aligned colons
   - One entry per line
   
   **Summary+Detail:**
   - Bold summary statement
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
   - Optional emoji/icons
   
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
- Be explicit about missing information
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
{{
  "conversation": [
    {{
      "user": "<Natural question with/without format request>",
      "assistant": "<Answer in appropriate format>"
    }}
    // 4-8 turns
  ]
}}

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
- [ ] Valid outer JSON

DIVERSITY GUIDELINES:
- Generate confusion patterns based on what's actually in each unique document
- Vary which fields you create confusion about across different samples
- Don't create the same trick question pattern repeatedly
- If a document has many similar items, vary which items you ask about
- Make hard negatives feel like genuine user confusion, not artificial tests
- Balance the conversation - don't make every question a trick question

REMEMBER: Generate ALL conversation content (user questions AND assistant answers) in language: {output_language.upper()}
"""
    return prompt


def get_reasoning_prompt(output_language: str = "vi") -> str:
    """Generate reasoning-focused conversations that teach models to think step-by-step before answering.

    Args:
        output_language: Language code for output conversations. Examples: "vi", "en", "zh", "ja", etc.
    """
    prompt = f"""You are an expert in analyzing business documents and creating conversations that TEACH STEP-BY-STEP REASONING before providing final answers.

CRITICAL: ALL conversations (both user questions and assistant answers) MUST be generated in this language: {output_language.upper()}

TASK:
Analyze the provided document image and generate a realistic multi-turn conversation (5-8 turns) where the AI assistant demonstrates EXPLICIT REASONING PROCESS before answering questions.

CORE PRINCIPLE: **Think Out Loud Before Answering**

The assistant MUST show its thinking process using one of these reasoning patterns:

1. **Sequential Analysis Pattern** (30-35%):
   - Break down the question into sub-steps
   - Analyze each step explicitly
   - Combine findings to reach conclusion
   
   Structure:
   Step 1: Check basic information
   Step 2: Verify calculations
   Step 3: Check required fields
   Conclusion: Final answer with summary

2. **Comparative Reasoning Pattern** (25-30%):
   - Identify items to compare
   - List relevant attributes of each
   - Analyze differences/similarities
   - Draw conclusion with evidence
   
   Structure:
   By quantity: Compare item A vs item B
   By proportion: Calculate percentages
   Conclusion: Final determination with rationale

3. **Detective/Verification Pattern** (20-25%):
   - State the claim/question to verify
   - Gather evidence from document
   - Check consistency across sections
   - Identify discrepancies or confirm accuracy
   
   Structure:
   Data collection: List all relevant values
   Calculation: Show arithmetic step-by-step
   Verification: Compare calculated vs stated
   Conclusion: Confirm or identify error

4. **Elimination/Deduction Pattern** (15-20%):
   - List all possibilities
   - Rule out options one by one
   - Arrive at answer through elimination
   
   Structure:
   Possibilities: List all options
   Check information: Examine evidence
   Cross-reference: Compare with descriptions
   Eliminate: Rule out incorrect options
   Conclusion: Final answer

5. **What-If/Scenario Analysis Pattern** (10-15%):
   - Propose hypothetical scenario
   - Analyze implications
   - Calculate potential outcomes
   
   Structure:
   Formula: Define calculation method
   Calculation: Apply to each item
   Summary: Total result with notes
   
REASONING TRIGGER QUESTIONS:

**Verification Questions** (Must show calculation):
- Check if totals are correct
- Check if document has any issues
- Verify if numbers match
- Check for errors

**Comparison Questions** (Must compare attributes):
- Which item is most notable?
- Compare top 3 items
- Which is larger: A or B?
- What's the difference between X and Y?

**Analysis Questions** (Must break down into steps):
- Analyze document structure
- Evaluate document overview
- Summarize key points
- What does this document tell us?

**Deduction Questions** (Must use elimination):
- What product is code XXX?
- Which field is missing?
- What information is not in document?

**Scenario Questions** (Must calculate hypothetical):
- What if...?
- Suppose... then how much?
- In case of... what's the result?

CRITICAL RULES FOR REASONING:

1. **Show Working**: Always display intermediate steps, not just final answer
2. **Label Steps**: Use clear step labels or section headers
3. **Explicit Math**: Write out calculations with numbers, not just results
4. **Acknowledge Uncertainty**: Show when double-checking or reconsidering
5. **Self-Correction**: Occasionally show minor errors then correct them
6. **Cite Evidence**: Always reference where info comes from (specific locations)
7. **Structured Output**: Use bullets, numbered lists, sections to organize thinking
8. **Conclusion Label**: Always end with clearly marked conclusion before final answer

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
- States results without showing work
- Vague references to document
- Unstructured wall of text
- No explicit conclusion

CONVERSATION STRUCTURE:

**Turn Distribution** (5-8 turns total):

Turn 1-2: Warm-up questions (simple extraction, brief answers)

Turn 3-5: CORE REASONING QUESTIONS (must use reasoning patterns)
These turns require detailed step-by-step thinking

Turn 6-7: Follow-up or transformation
Can be brief or require additional reasoning

Turn 8: Optional complex scenario

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

FORMAT GUIDELINES:

Assistant can still use various formats (JSON, tables, etc.) but must:
- Show reasoning BEFORE presenting formatted output
- Explain why that format is chosen
- Include reasoning in natural text, then format result

OUTPUT FORMAT:
{{
  "conversation": [
    {{
      "user": "<Question requiring reasoning>",
      "assistant": "<Step-by-step reasoning process>\\n\\nConclusion: <Final answer>"
    }}
    // 5-8 turns, 60-70% with explicit reasoning
  ]
}}

QUALITY CHECKLIST:
- [ ] 5-8 conversation turns
- [ ] At least 3 turns show explicit step-by-step reasoning
- [ ] Reasoning uses clear structure (bullets, numbering, sections)
- [ ] All calculations shown explicitly with numbers
- [ ] Conclusions clearly labeled
- [ ] Evidence cited with specific locations
- [ ] At least 2 different reasoning patterns used
- [ ] Includes 1-2 verification/checking questions
- [ ] Natural language with technical precision
- [ ] Valid outer JSON format

REMEMBER: Generate ALL conversation content (user questions AND assistant answers) in language: {output_language.upper()}
"""
    return prompt


def get_lazy_user_prompt(output_language: str = "vi") -> str:
    """Generate conversations with casual, vague, and imperfect user queries to teach context recovery.

    Args:
        output_language: Language code for output conversations. Examples: "vi", "en", "zh", "ja", etc.
    """
    prompt = f"""You are an expert in analyzing business documents and creating realistic conversations where users ask questions in CASUAL, VAGUE, IMPERFECT language - like real workplace communication.

CRITICAL: ALL conversations (both user questions and assistant answers) MUST be generated in this language: {output_language.upper()}

TASK:
Analyze the provided document image and generate a realistic multi-turn conversation (5-7 turns) where the USER asks questions using natural, imperfect language (like a busy colleague or boss), and the ASSISTANT responds naturally and helpfully while handling ambiguity.

CORE PRINCIPLE: **Real People Don't Talk Like Robots**

USER QUERY TYPES (Mix these naturally):

1. **Ultra-Short Queries** (25-30%):
   Single word or very brief phrases asking about totals, dates, codes, etc.
   Assistant must infer context from document or previous turns.

2. **Typos & Informal Spelling** (20-25%):
   Common typos, abbreviations, missing diacritics, lowercase text
   Assistant should understand and respond naturally without pointing out errors.

3. **Missing Context** (15-20%):
   Vague references like "that one", "how much?", "is it correct?" without specifying what
   Assistant should either:
   - Ask for clarification
   - Infer from context and provide answer

4. **Pronouns Without Antecedents** (10-15%):
   Using "it", "that", "those", "first one" without clear reference
   Assistant should identify what the pronoun refers to from document.

5. **Slang & Colloquialisms** (10-15%):
   Casual workplace language, informal tone
   Assistant responds naturally despite casual tone.

6. **Incomplete Sentences** (10-15%):
   Trailing off mid-sentence, fragmentary questions
   Assistant completes the thought logically.

7. **Mixed Languages** (5-10%):
   Code-switching between local language and English technical terms
   Assistant can respond naturally, mirroring style.

8. **Vague References** (5-10%):
   "the highest one", "first few", "the number part", "bottom section"
   Assistant should identify specific items/locations.

9. **Implied Questions** (5-10%):
   Statements that imply questions without question marks
   Assistant should understand the implicit question.

10. **Follow-up Without Full Context** (5-10%):
    Continuing from previous turn without restating context
    Assistant should remember conversation history.

ASSISTANT RESPONSE STRATEGIES:

**Strategy 1: Direct Answer** (50-60% of unclear queries):
When context is inferrable from document, provide direct answer.

**Strategy 2: Clarifying Question** (20-30%):
When genuinely ambiguous, ask what user means.

**Strategy 3: Multiple Interpretations** (10-15%):
Offer 2-3 possible answers for vague queries.

**Strategy 4: Gentle Correction** (5-10%):
Fix typos implicitly without criticizing user's language.

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
For ultra-short questions, give brief answers.

✓ Detailed when user needs clarity:
For vague questions, provide comprehensive answer with location details.

CONVERSATION PATTERNS:

**Pattern A: Progressive Casualness** (30%):
Turn 1: Somewhat formal question
Turn 2: Getting more casual
Turn 3: Very brief
Turn 4: Typos appear
Turn 5: Very casual

**Pattern B: Consistently Casual** (25%):
All turns use informal language, typos, slang throughout.

**Pattern C: Ambiguous + Clarification** (25%):
Turn 1: Vague question
Turn 2: Assistant asks for clarification or provides options
Turn 3: User clarifies or Assistant infers
Turn 4: More specific question emerges

**Pattern D: Code-Mixing** (20%):
Mix local language with English technical terms naturally.

CONVERSATION LENGTH: 5-7 turns
- Turn 1-2: Casual questions (may need clarification)
- Turn 3-4: Follow-ups (often vague, reference previous context)
- Turn 5-6: Specific requests or corrections
- Turn 7: Optional final check or format request

OUTPUT FORMAT:
{{
  "conversation": [
    {{
      "user": "<casual/vague/informal query>",
      "assistant": "<natural, helpful response>"
    }}
    // 5-7 turns
  ]
}}

QUALITY CHECKLIST:
- [ ] 5-7 conversation turns
- [ ] 60-80% of user queries use casual/imperfect language
- [ ] Mix of typos, slang, abbreviations, vagueness
- [ ] Assistant always helpful and natural
- [ ] At least 1-2 clarification moments
- [ ] Assistant demonstrates context recovery
- [ ] Natural conversation flow
- [ ] Valid outer JSON format
- [ ] No mockery of user's casual language

DIVERSITY GUIDELINES:
- Vary the TYPE of informality (typos vs vagueness vs slang) across conversations
- Don't make every conversation ultra-casual
- Mix serious business queries with casual tone
- Include both simple extraction and complex questions asked casually
- Show Assistant adapting to user's communication style

REMEMBER: Generate ALL conversation content (user questions AND assistant answers) in language: {output_language.upper()}
"""
    return prompt


def get_visual_spatial_correction_prompt(output_language: str = "vi") -> str:
    """Generate conversations focused on spatial awareness and graceful error correction.

    Args:
        output_language: Language code for output conversations. Examples: "vi", "en", "zh", "ja", etc.
    """
    prompt = f"""You are an expert in analyzing business documents with focus on VISUAL LAYOUT UNDERSTANDING and COLLABORATIVE ERROR CORRECTION.

CRITICAL: ALL conversations (both user questions and assistant answers) MUST be generated in this language: {output_language.upper()}

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
   User asks where document number, logo, or signature is located.
   Assistant MUST specify: top-right corner, bottom-left, left side middle, above detail table, below title, etc.

b) **Section/Zone Positioning**:
   User asks which section contains supplier info, product table, or footer.
   Assistant describes document structure: Header, Body, Footer, Sidebar

c) **Relative Positioning**:
   User asks how elements relate to each other.
   Assistant uses: above/below, left/right, near/far, beside, inside/outside table

**2. VISUAL FEATURE QUERIES** (20-25%):

Users ask about HOW things LOOK:

a) **Typography & Formatting**:
   User asks what is bold, colors, largest text, font differences.
   Assistant describes: Bold/italic, font size, color, underline/highlight

b) **Visual Elements**:
   User asks about logos, stamps, signatures, barcodes/QR codes.
   Assistant describes: What's visible, color, clarity (clear/blurry), presence/absence

c) **Layout Structure**:
   User asks how many columns, rows, sections, borders.
   Assistant analyzes: Table structure, boxes, borders, sections

**3. ZONE-SPECIFIC READING** (15-20%):

Users ask to read SPECIFIC AREAS by location, visual cue, or surrounding context.

**4. SPATIAL REASONING** (10-15%):

Users ask about: Distance/proximity, alignment, visual hierarchy.

---

PART B: SELF-CORRECTION & OCR ERROR HANDLING (40% of conversation)

**1. USER CHALLENGES ASSISTANT** (20-25%):

Simulate user correcting the assistant's reading:

a) **Number Confusion**:
   User claims assistant read wrong number.
   
   Assistant responses:
   - If truly wrong: "Let me check again... You're right, this is [X]. I misread due to [reason]. Thank you!"
   - If ambiguous: "I understand, this area is blurry. Looking again, it could be [X] as you say."
   - If assistant correct: "Let me confirm... I see [Y] at this position. Perhaps you're looking at a different location?"

b) **Text Misreading**:
   User corrects a word/phrase.
   Assistant: "Thank you! Let me look again... The text is small/blurry, easy to mistake. I'll update accordingly."

c) **Field Mix-up**:
   User points out field confusion.
   Assistant: "My apologies! You're correct. I mixed up the two fields. Thank you for pointing that out!"

**2. USER REQUESTS RE-CHECK** (10-15%):

User asks to verify again with vague request, specific zone, or uncertain area.
Assistant re-examines and provides updated/confirmed information.

**3. ASSISTANT SELF-CORRECTION** (5-10%):

Assistant proactively catches errors:
- Mid-conversation correction: "Wait, let me check again... I said [X] but it's actually [Y]. Sorry!"
- Clarifying ambiguity: "This number is hard to read. It looks like [X] but could be [Y]."
- Acknowledging limitations: "I can't read this signature due to handwriting. I can only see there's a signature at [location]."

**ERROR HANDLING STRATEGIES:**

When User Says "Wrong":
1. Don't argue immediately
2. Check again the specific area
3. Admit mistake, explain ambiguity, or clarify location

Tone: Humble, collaborative, grateful, natural

DON'T: Be defensive, ignore corrections, claim 100% certainty
DO: Check again, admit mistakes, explain difficulties, thank for feedback

---

CONVERSATION PATTERNS:

**Pattern A: Spatial Exploration** (40%):
- User asks WHERE → Assistant describes position
- User asks VISUAL FEATURE → Assistant describes appearance
- User asks to read ZONE → Assistant reads that area
- User challenges detail → Assistant checks/corrects

**Pattern B: Correction-Heavy** (35%):
- Assistant provides info
- User corrects → Assistant re-checks, admits error
- User requests re-read → Assistant re-reads carefully
- User questions certainty → Assistant expresses confidence level

**Pattern C: Mixed** (25%):
- Spatial questions + correction challenges mixed together

CONVERSATION LENGTH: 5-8 turns
- 3-4 turns: Spatial awareness questions
- 2-3 turns: Error correction scenarios
- 1 turn: Optional final verification

OUTPUT FORMAT:
{{
  "conversation": [
    {{
      "user": "<Question about position/visual OR correction challenge>",
      "assistant": "<Spatial description OR graceful error handling>"
    }}
    // 5-8 turns
  ]
}}

QUALITY CHECKLIST:
- [ ] 5-8 conversation turns
- [ ] 60% spatial awareness (position, visual features, zones)
- [ ] 40% error correction (challenges, re-checks, self-correction)
- [ ] Assistant uses directional terms
- [ ] Assistant describes visual features
- [ ] At least 1-2 correction challenges
- [ ] Graceful error handling (no defensiveness)
- [ ] Collaborative language
- [ ] Valid outer JSON format

DIVERSITY GUIDELINES:
- Vary spatial questions: corners, zones, relative positions, visual features
- Vary correction types: numbers, text, fields, ambiguous characters
- Mix confident with uncertain readings
- Balance "assistant wrong" vs "user misunderstood" scenarios
- Natural descriptions, not robotic

REMEMBER: Generate ALL conversation content (user questions AND assistant answers) in language: {output_language.upper()}
"""
    return prompt
