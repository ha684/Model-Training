# -*- coding: utf-8 -*-
"""
Configuration file for data extraction fields and prompts.
Modify the fields lists below to add/remove fields as needed.
"""

DEFAULT_FIELDS = [
    "Số phiếu",
    "Ngày chứng từ(mm/dd/yyyy)",
    "Ngày xuất(mm/dd/yyyy)",
    "Mã FG",
    "Đvt",
    "Số lượng(Thùng)",
    "Số lượng(Pcs)",
    "Kho nhận",
    "Địa chỉ",
    "Mã AR",
]

METADATA_FIELDS = [
    "Số phiếu",
    "Ngày chứng từ(mm/dd/yyyy)",
    "Ngày xuất(mm/dd/yyyy)",
    "Kho nhận",
    "Địa chỉ",
    "Mã AR",
]

TABLE_ROW_FIELDS = [f for f in DEFAULT_FIELDS if f not in METADATA_FIELDS]


def get_extraction_prompt() -> str:
    """
    Generate the extraction prompt for Vision Language Model API.
    The prompt includes all fields to extract and output format specification.
    """
    fields_list = "\n".join([f"  - {field}" for field in DEFAULT_FIELDS])
    metadata_fields_list = ", ".join([f'"{field}"' for field in METADATA_FIELDS])
    table_row_fields_list = ", ".join([f'"{field}"' for field in TABLE_ROW_FIELDS])

    prompt = f"""You are an expert in extracting structured data from delivery notes, cargo documents, and invoices.

Extract the following fields from the provided document image:

{fields_list}

OUTPUT REQUIREMENTS:
- Return data in JSON format with the following structure:
{{
  "metadata": {{
    {", ".join([f'"{f}": <value or null>' for f in METADATA_FIELDS])}
  }},
  "table_rows": [
    {{
      {", ".join([f'"{f}": <value or null>' for f in TABLE_ROW_FIELDS])}
    }}
  ]
}}

EXTRACTION RULES:
1. "metadata" contains document-level information: {metadata_fields_list}
2. "table_rows" contains table row data: {table_row_fields_list}
3. If there are multiple rows in the table, each row should be a separate object in the "table_rows" array
4. If a field value cannot be found or does not exist in the document, set it to null
5. For date fields, convert to mm/dd/yyyy format if possible. If the date format is unclear or cannot be determined, set to null
6. For numeric fields (quantities), extract only the number. If not found, set to null
7. Be extremely precise - only include values that are explicitly present in the document
8. Do not infer, guess, or make assumptions about missing data
9. Return ONLY valid JSON with no additional text, explanations, or markdown formatting
10. Ensure the JSON is properly formatted and can be parsed
11. Pay special attention to Vietnamese text fields - preserve exact spelling and diacritics
12. For code fields (Mã FG, Mã AR), extract the exact alphanumeric code as shown

IMPORTANT:
- Accuracy is critical. If you're unsure about a value, set it to null rather than guessing
- Double-check all extracted values against the source document
- Maintain data type consistency (strings for text, numbers for quantities)

Analyze and extract data from this document:"""

    return prompt


def get_fields_schema() -> dict:
    """
    Returns the schema definition for extracted data.
    Useful for validation and documentation.
    """
    return {
        "metadata_fields": METADATA_FIELDS,
        "table_row_fields": TABLE_ROW_FIELDS,
        "all_fields": DEFAULT_FIELDS,
    }
