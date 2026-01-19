# config/fields_config.py
# -*- coding: utf-8 -*-
"""
Configuration file for data extraction fields and prompts.
Modify the fields lists below to add/remove fields as needed.
"""

# DEFAULT_FIELDS = [
#     "Số phiếu",
#     "Ngày chứng từ(dd/mm/yyyy)",
#     "Ngày xuất(dd/mm/yyyy)",
#     "Mã FG",
#     "Đvt",
#     "Số lượng",
#     "Kho nhận",
#     "Địa chỉ",
#     "Mã AR",
# ]

# METADATA_FIELDS = [
#     "Số phiếu",
#     "Kho nhận",
#     "Địa chỉ",
#     "Mã AR",
#     "Ngày chứng từ(dd/mm/yyyy)",
#     "Ngày xuất(dd/mm/yyyy)",
# ]

# TABLE_ROW_FIELDS = [f for f in DEFAULT_FIELDS if f not in METADATA_FIELDS]

DEFAULT_FIELDS = [
    "document_number",
    "document_date",
    "export_date",
    "product_code",
    "unit",
    "quantity",
    "receiving_warehouse",
    "address",
    "ar_code",
]

METADATA_FIELDS = [
    "document_number",
    "document_date",
    "export_date",
    "receiving_warehouse",
    "address",
    "ar_code",
]

TABLE_ROW_FIELDS = [f for f in DEFAULT_FIELDS if f not in METADATA_FIELDS]

# def get_extraction_prompt() -> str:
#     """
#     Generate the extraction prompt for Vision Language Model API.
#     The prompt includes all fields to extract and output format specification.
#     """
#     fields_list = "\n".join([f"  - {field}" for field in DEFAULT_FIELDS])
#     metadata_fields_list = ", ".join([f'"{field}"' for field in METADATA_FIELDS])
#     table_row_fields_list = ", ".join([f'"{field}"' for field in TABLE_ROW_FIELDS])

#     prompt = f"""You are an expert in extracting structured data from Vietnamese delivery notes, cargo documents, and invoices.

# Extract the following fields from the provided document image:

# {fields_list}

# FIELD DEFINITIONS AND EXTRACTION RULES:

# **METADATA FIELDS (Document-level information):**

# 1. "Số phiếu" (Document Number/PO Number):
#    - Look for labels such as: "Số PO", "PO/PO No", "S.PO/PO No.", "Số:", "Số PGC", "Number"
#    - Usually located in the top-right area of the document header
#    - Common formats: "PXB25110076", "2544051804257", "252.104.868.0263", "252-104-866-7210"
#    - Extract the number exactly as shown, preserving dots, dashes, and spaces
#    - DO NOT extract handwritten page numbers (e.g., "1/2", "2/2")
#    - DO NOT extract numbers that appear after "Mã AR" field (these are AR code suffixes, not document numbers)
#    - DO NOT extract dates or reference numbers from table rows
#    - If the document number is unclear or not found, set to null

# 2. "Ngày chứng từ(dd/mm/yyyy)" (Document Date):
#    - Look for labels: "Ngày/Date:", "Document date:", "Ngày chứng từ:"
#    - Located near the document number in the header
#    - If multiple dates exist, choose the one labeled as document date or issue date
#    - If unclear or not found, set to null

# 3. "Ngày xuất(dd/mm/yyyy)" (Export/Delivery Date):
#    - Look for labels: "Ngày giao hàng", "Delivery date:", "Ngày xuất:", "Ngày xuất kho"
#    - Convert to dd/mm/yyyy format
#    - This may be the same as or different from the document date
#    - If not explicitly shown, set to null

# 4. **"Kho nhận"** (Receiving Warehouse):
#    - The warehouse or customer name receiving the goods
#    - Example: "DC KHO HẢI PHÒNG", "Kho Hà Nội"
#    - Extract exact name as shown

# 5. **"Địa chỉ"** (Address):
#    - Full address of receiving location
#    - This field is OPTIONAL and may not exist in all documents
#    - If no address is shown, set to null
#    - Do not infer or make up an address

# 6. **"Mã AR"** (AR Code/Customer Code):
#    - Customer or account receivable code
#    - Example format: "A68/750" or with suffix "A68/750_141"
#    - Extract the COMPLETE code including any suffix numbers after underscores
#    - Location: Usually appears in a field labeled "Mã AR" or "Khách hàng"

# **TABLE ROW FIELDS (Item-level information from table rows):**

# 7. **"Mã FG"** (Finished Goods Code):
#    - Product code from the leftmost column of the table
#    - Example formats: "PG1490", "PG1495", "PG7105", "FC9345", "PG8023", "PG8027", "PL2368"
#    - Extract exact alphanumeric code as shown
#    - Each row in the product table will have its own Mã FG

# 8. **"Đvt"** (Unit of Measurement):
#    - Unit type for the product
#    - Common values: "Hộp" (box), "Hũ" (jar), "Gói" (pack), "Chai" (bottle)
#    - Extract the Vietnamese unit name exactly as shown

# 9. **"Số lượng(Thùng)"** (Quantity in Cartons/Containers):
#    - Number of CARTONS/CONTAINERS (not individual pieces)
#    - A "thùng" is a container that holds multiple pieces (pcs)
#    - Think of it like a carton of 12 cans you see at supermarkets
#    - Extract only the number for cartons shipped
#    - If only pcs quantity is shown (no carton quantity), calculate: Số lượng(Thùng) = Số lượng(Pcs) / standard pack size
#    - If neither is clear, set to null

# 10. **"Số lượng(Pcs)"** (Quantity in Pieces):
#     - Total number of INDIVIDUAL PIECES
#     - "Pcs" refers to individual units (like individual cans in a carton)
#     - Extract the total piece count
#     - Relationship: Số lượng(Pcs) = Số lượng(Thùng) × pieces per carton
#     - Note: The document may not specify how many pcs are in each thùng (this is OK, just extract what's shown)

# OUTPUT REQUIREMENTS:
# - Return data in JSON format with the following structure:
# {{
#   "metadata": {{
#     {", ".join([f'"{f}": <value or null>' for f in METADATA_FIELDS])}
#   }},
#   "table_rows": [
#     {{
#       {", ".join([f'"{f}": <value or null>' for f in TABLE_ROW_FIELDS])}
#     }}
#   ]
# }}

# CRITICAL DATA TYPE REQUIREMENTS:
# - "Số lượng" MUST be a number (integer or float), NOT a string
#   * Correct: {{"Số lượng": 168}} or {{"Số lượng": 168.5}}
#   * Wrong: {{"Số lượng": "168"}} or {{"Số lượng": "168.5"}}
# - All other fields can be strings or null
# - Dates must be strings in dd/mm/yyyy format

# CRITICAL EXTRACTION RULES:
# 1. "metadata" contains document-level information: {metadata_fields_list}
# 2. "table_rows" contains table row data: {table_row_fields_list}
# 3. Each row in the product table should be a separate object in the "table_rows" array
# 4. **DO NOT confuse numbers after "Mã AR" as "Số phiếu"** - these are suffixes of Mã AR
# 5. If a field value cannot be found or does not exist, set it to null
# 6. For date fields, convert to dd/mm/yyyy format. If unclear, set to null
# 7. For numeric fields (quantities), extract only the number. If not found, set to null
# 8. Preserve exact Vietnamese spelling and diacritics (ă, â, đ, ê, ô, ơ, ư, etc.)
# 9. For code fields (Mã FG, Mã AR), extract the exact alphanumeric code as shown
# 10. Be extremely precise - only include values explicitly present in the document
# 11. Do not infer, guess, or make assumptions about missing data
# 12. Return ONLY valid JSON with no additional text, explanations, or markdown formatting

# VALIDATION CHECKLIST:
# - [ ] "Số phiếu" is from the document number field, NOT from after "Mã AR"
# - [ ] "Mã AR" includes complete code with any suffixes
# - [ ] All Vietnamese diacritics are preserved correctly
# - [ ] Dates are in dd/mm/yyyy format or null
# - [ ] Each table row is a separate object in "table_rows"
# - [ ] "Địa chỉ" is null if not shown (don't guess)
# - [ ] Quantities are numbers only (no text)
# - [ ] Output is valid, parseable JSON
# """

   #  return prompt
 
def get_extraction_prompt() -> str:
    """Generate extraction prompt with inline field definitions."""
    
    prompt = f"""Extract structured data from this Vietnamese delivery document and return as JSON.

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

#EXTRACTION INSTRUCTIONS:

### METADATA FIELDS:

1. document_number
   - The main document/PO number, usually in top-right header area
   - Common formats: "PXB25110076", "2544051804257", "252.104.868.0263"
   - Extract exactly as shown with dots, dashes, spaces
   - NOT the handwritten page numbers like "1/2" or numbers after customer code

2. document_date
   - Document issue date in dd/mm/yyyy format
   - Extract exactly as shown on document
   - If multiple dates exist, choose the one labeled as issue/document date

3. export_date
   - Delivery or shipment date in dd/mm/yyyy format
   - Extract exactly as shown on document
   - May be same as or different from document_date

4. receiving_warehouse
   - Name of warehouse or customer receiving the goods
   - Examples: "DC KHO HẢI PHÒNG", "Kho Hà Nội"
   - Extract full name as shown

5. address
   - Full delivery address
   - Optional field - set null if not present on document
   - Do not guess or infer

6. ar_code
   - Customer account code
   - Examples: "A68/750", "A68/750_141"
   - Extract complete code including any suffix after underscore

### TABLE FIELDS:

7. product_code
   - Product identifier, typically leftmost column in table
   - Examples: "PG1490", "FC9345", "PL2368"
   - Extract exact alphanumeric code

8. unit
   - Unit of measurement for the product
   - Examples: "Hộp" (box), "Gói" (pack), "Chai" (bottle), "Thùng" (carton)
   - Keep text as-is

9. quantity
   - Numeric quantity value
   - CRITICAL: Must be number type, not string
   - Example: {{"quantity": 168}} NOT {{"quantity": "168"}}

IMPORTANT RULES:
- Return ONLY valid JSON with no markdown formatting or explanations
- quantity field MUST be number type (integer or float), all other fields are strings or null
- Dates are already in dd/mm/yyyy format - extract exactly as shown
- Each product row in the table creates one object in table_rows array
- Preserve Vietnamese diacritics in text values (ă, â, đ, ê, ô, ơ, ư, etc.)
- Set fields to null if not found - do not guess or make assumptions
- Document number is in the header, NOT the number that appears after customer code
- Include complete ar_code with suffix if present
"""
    
    return prompt