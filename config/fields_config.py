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

    prompt = f"""You are an expert in extracting structured data from Vietnamese delivery notes, cargo documents, and invoices.

Extract the following fields from the provided document image:

{fields_list}

FIELD DEFINITIONS AND EXTRACTION RULES:

**METADATA FIELDS (Document-level information):**

1. **"Số phiếu"** (PO Number/Document Number):
   - This is typically labeled as "Số PGC" or similar at the top of the document
   - Example format: "471016736" or "470167389"
   - CRITICAL: Do NOT confuse this with numbers appearing after "Mã AR" field
   - The number after "Mã AR" (e.g., "141" in "A68/750___141") is a SUFFIX of Mã AR, NOT the Số phiếu
   - Location: Usually in the top-right area of the document

2. **"Ngày chứng từ(mm/dd/yyyy)"** (Document Date):
   - Convert to mm/dd/yyyy format
   - If format unclear or not found, set to null
   - Example: "01/04/2025"

3. **"Ngày xuất(mm/dd/yyyy)"** (Export Date):
   - Convert to mm/dd/yyyy format
   - May be the same as or different from Ngày chứng từ
   - If not found, set to null

4. **"Kho nhận"** (Receiving Warehouse):
   - The warehouse or customer name receiving the goods
   - Example: "DC KHO HẢI PHÒNG", "Kho Hà Nội"
   - Extract exact name as shown

5. **"Địa chỉ"** (Address):
   - Full address of receiving location
   - This field is OPTIONAL and may not exist in all documents
   - If no address is shown, set to null
   - Do not infer or make up an address

6. **"Mã AR"** (AR Code/Customer Code):
   - Customer or account receivable code
   - Example format: "A68/750" or with suffix "A68/750___141"
   - Extract the COMPLETE code including any suffix numbers after underscores
   - Location: Usually appears in a field labeled "Mã AR" or "Khách hàng"

**TABLE ROW FIELDS (Item-level information from table rows):**

7. **"Mã FG"** (Finished Goods Code):
   - Product code from the leftmost column of the table
   - Example formats: "PG1490", "PG1495", "PG7105", "FC9345", "PG8023", "PG8027", "PL2368"
   - Extract exact alphanumeric code as shown
   - Each row in the product table will have its own Mã FG

8. **"Đvt"** (Unit of Measurement):
   - Unit type for the product
   - Common values: "Hộp" (box), "Hũ" (jar), "Gói" (pack), "Chai" (bottle)
   - Extract the Vietnamese unit name exactly as shown

9. **"Số lượng(Thùng)"** (Quantity in Cartons/Containers):
   - Number of CARTONS/CONTAINERS (not individual pieces)
   - A "thùng" is a container that holds multiple pieces (pcs)
   - Think of it like a carton of 12 cans you see at supermarkets
   - Extract only the number for cartons shipped
   - If only pcs quantity is shown (no carton quantity), calculate: Số lượng(Thùng) = Số lượng(Pcs) / standard pack size
   - If neither is clear, set to null

10. **"Số lượng(Pcs)"** (Quantity in Pieces):
    - Total number of INDIVIDUAL PIECES
    - "Pcs" refers to individual units (like individual cans in a carton)
    - Extract the total piece count
    - Relationship: Số lượng(Pcs) = Số lượng(Thùng) × pieces per carton
    - Note: The document may not specify how many pcs are in each thùng (this is OK, just extract what's shown)

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

CRITICAL EXTRACTION RULES:
1. "metadata" contains document-level information: {metadata_fields_list}
2. "table_rows" contains table row data: {table_row_fields_list}
3. Each row in the product table should be a separate object in the "table_rows" array
4. **DO NOT confuse numbers after "Mã AR" as "Số phiếu"** - these are suffixes of Mã AR
5. If a field value cannot be found or does not exist, set it to null
6. For date fields, convert to mm/dd/yyyy format. If unclear, set to null
7. For numeric fields (quantities), extract only the number. If not found, set to null
8. Preserve exact Vietnamese spelling and diacritics (ă, â, đ, ê, ô, ơ, ư, etc.)
9. For code fields (Mã FG, Mã AR), extract the exact alphanumeric code as shown
10. Be extremely precise - only include values explicitly present in the document
11. Do not infer, guess, or make assumptions about missing data
12. Return ONLY valid JSON with no additional text, explanations, or markdown formatting

VALIDATION CHECKLIST:
- [ ] "Số phiếu" is from the document number field, NOT from after "Mã AR"
- [ ] "Mã AR" includes complete code with any suffixes
- [ ] All Vietnamese diacritics are preserved correctly
- [ ] Dates are in mm/dd/yyyy format or null
- [ ] Each table row is a separate object in "table_rows"
- [ ] "Địa chỉ" is null if not shown (don't guess)
- [ ] Quantities are numbers only (no text)
- [ ] Output is valid, parseable JSON

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
        "field_descriptions": {
            "Số phiếu": "PO/Document number (usually labeled Số PGC), NOT the number after Mã AR",
            "Ngày chứng từ(mm/dd/yyyy)": "Document date in mm/dd/yyyy format",
            "Ngày xuất(mm/dd/yyyy)": "Export/shipping date in mm/dd/yyyy format",
            "Mã FG": "Finished goods product code (e.g., PG1490, FC9345)",
            "Đvt": "Unit type (Hộp, Hũ, Gói, Chai, etc.)",
            "Số lượng(Thùng)": "Quantity in cartons/containers (each contains multiple pcs)",
            "Số lượng(Pcs)": "Quantity in individual pieces",
            "Kho nhận": "Receiving warehouse or customer name",
            "Địa chỉ": "Delivery address (may not exist in all documents)",
            "Mã AR": "Customer/AR code with any suffixes (e.g., A68/750___141)",
        }
    }