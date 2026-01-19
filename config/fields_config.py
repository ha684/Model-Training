# config/fields_config.py
# -*- coding: utf-8 -*-
"""
Configuration file for data extraction fields and prompts.
Modify the fields lists below to add/remove fields as needed.
"""

DEFAULT_FIELDS = [
    "Số phiếu",
    "Ngày chứng từ(dd/mm/yyyy)",
    "Ngày xuất(dd/mm/yyyy)",
    "Mã FG",
    "Đvt",
    "Số lượng",
    "Kho nhận",
    "Địa chỉ",
    "Mã AR",
]

METADATA_FIELDS = [
    "Số phiếu",
    "Kho nhận",
    "Địa chỉ",
    "Mã AR",
    "Ngày chứng từ(dd/mm/yyyy)",
    "Ngày xuất(dd/mm/yyyy)",
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

1. "Số phiếu" (Document Number/PO Number):
   - Look for labels such as: "Số PO", "PO/PO No", "S.PO/PO No.", "Số:", "Số PGC", "Number"
   - Usually located in the top-right area of the document header
   - Common formats: "PXB25110076", "2544051804257", "252.104.868.0263", "252-104-866-7210"
   - Extract the number exactly as shown, preserving dots, dashes, and spaces
   - DO NOT extract handwritten page numbers (e.g., "1/2", "2/2")
   - DO NOT extract numbers that appear after "Mã AR" field (these are AR code suffixes, not document numbers)
   - DO NOT extract dates or reference numbers from table rows
   - If the document number is unclear or not found, set to null

2. "Ngày chứng từ(dd/mm/yyyy)" (Document Date):
   - Look for labels: "Ngày/Date:", "Document date:", "Ngày chứng từ:"
   - Located near the document number in the header
   - If multiple dates exist, choose the one labeled as document date or issue date
   - If unclear or not found, set to null

3. "Ngày xuất(dd/mm/yyyy)" (Export/Delivery Date):
   - Look for labels: "Ngày giao hàng", "Delivery date:", "Ngày xuất:", "Ngày xuất kho"
   - Convert to dd/mm/yyyy format
   - This may be the same as or different from the document date
   - If not explicitly shown, set to null

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
   - Example format: "A68/750" or with suffix "A68/750_141"
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

CRITICAL DATA TYPE REQUIREMENTS:
- "Số lượng" MUST be a number (integer or float), NOT a string
  * Correct: {{"Số lượng": 168}} or {{"Số lượng": 168.5}}
  * Wrong: {{"Số lượng": "168"}} or {{"Số lượng": "168.5"}}
- All other fields can be strings or null
- Dates must be strings in dd/mm/yyyy format

CRITICAL EXTRACTION RULES:
1. "metadata" contains document-level information: {metadata_fields_list}
2. "table_rows" contains table row data: {table_row_fields_list}
3. Each row in the product table should be a separate object in the "table_rows" array
4. **DO NOT confuse numbers after "Mã AR" as "Số phiếu"** - these are suffixes of Mã AR
5. If a field value cannot be found or does not exist, set it to null
6. For date fields, convert to dd/mm/yyyy format. If unclear, set to null
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
- [ ] Dates are in dd/mm/yyyy format or null
- [ ] Each table row is a separate object in "table_rows"
- [ ] "Địa chỉ" is null if not shown (don't guess)
- [ ] Quantities are numbers only (no text)
- [ ] Output is valid, parseable JSON

Analyze and extract data from this document:"""

    return prompt
 
def get_extraction_prompt_evaluation() -> str:
    """
    Generate the extraction prompt for Vision Language Model API.
    The prompt includes all fields to extract and output format specification.
    """
    fields_list = "\n".join([f"  - {field}" for field in DEFAULT_FIELDS])
    metadata_fields_list = ", ".join([f'"{field}"' for field in METADATA_FIELDS])
    table_row_fields_list = ", ".join([f'"{field}"' for field in TABLE_ROW_FIELDS])

    prompt = f"""You are an expert in extracting structured data from Vietnamese delivery notes (phiếu giao hàng), cargo documents, and invoices.

Extract the following fields from the provided document image:

{fields_list}

FIELD DEFINITIONS AND EXTRACTION RULES:

METADATA FIELDS (Document-level information):

1. "Số phiếu" (Document Number/PO Number):
   - Look for labels such as: "Số PO", "PO/PO No", "S.PO/PO No.", "Số:", "Số PGC", "Number"
   - Usually located in the top-right area of the document header
   - Common formats: "PXB25110076", "2544051804257", "252.104.868.0263", "252-104-866-7210"
   - Extract the number exactly as shown, preserving dots, dashes, and spaces
   - DO NOT extract handwritten page numbers (e.g., "1/2", "2/2")
   - DO NOT extract numbers that appear after "Mã AR" field (these are AR code suffixes, not document numbers)
   - DO NOT extract dates or reference numbers from table rows
   - If the document number is unclear or not found, set to null

2. "Ngày chứng từ(dd/mm/yyyy)" (Document Date):
   - Look for labels: "Ngày/Date:", "Document date:", "Ngày chứng từ:"
   - Located near the document number in the header
   - Convert to dd/mm/yyyy format (e.g., "03/11/2025")
   - If multiple dates exist, choose the one labeled as document date or issue date
   - If unclear or not found, set to null

3. "Ngày xuất(dd/mm/yyyy)" (Export/Delivery Date):
   - Look for labels: "Ngày giao hàng", "Delivery date:", "Ngày xuất:", "Ngày xuất kho"
   - Convert to dd/mm/yyyy format
   - This may be the same as or different from the document date
   - If not explicitly shown, set to null

4. "Kho nhận" (Receiving Party/Customer Name):
   - Look for sections labeled: "Bên nhận hàng:", "Đơn vị nhận hàng:", "Tên người mua hàng:", "Customer's name:", "Đơn vị nhận:"
   - This field contains the COMPANY NAME or WAREHOUSE NAME of the receiving party
   - Extract the complete name as shown in the document
   - Examples: "CÔNG TY TNHH THIẾT BỊ CÔNG NGHIỆP VÀ DỊCH VỤ KỸ THUẬT THÁI SƠN", "CÔNG TY TNHH DỊCH VỤ EB", "kho SENKO", "DC KHO HẢI PHÒNG"
   - This is different from the detailed street address
   - If not found, set to null

5. "Địa chỉ" (Delivery Address):
   - Look for labels: "Địa chỉ:", "Địa chỉ nhận hàng:", "Delivery address:", "Đia chỉ:"
   - This field contains the FULL STREET ADDRESS with specific location details
   - Extract complete address including street numbers, street names, wards, districts, and cities
   - Examples: "Số 77 Kim Anh, Xã Nội Bài, TP.Hà Nội", "Số 163, Đường Phan Đăng Lưu, Phường Cầu Kiều, Thành phố Hồ Chí Minh, Việt Nam", "KM54+100, quốc lộ 5, khu 3, phường nhi châu, tp hải dương"
   - Note: This field is optional and may not exist in all documents
   - DO NOT confuse with company/warehouse names (those belong in "Kho nhận")
   - If no address field is present, set to null

6. "Mã AR" (AR Code/Customer Account Code):
   - Look for labels: "Mã AR:", "Khách hàng:", "Customer code:", "AR Code:"
   - Format typically includes alphanumeric characters with slashes or underscores
   - Examples: "A68/750", "AR5239_141", "A68/750_141"
   - Extract the COMPLETE code including any suffixes or numbers after underscores
   - The numbers after underscores are part of the AR code, NOT the document number
   - If not found, set to null

TABLE ROW FIELDS (Item-level information from product table):

7. "Mã FG" (Finished Goods Code/Product Code):
   - The product identifier typically found in the leftmost column of the product table
   - Common formats: "PG1490", "PG1495", "FGB0027", "FC9345", "PL2368", "PG8023"
   - Extract the exact alphanumeric code as shown
   - Each row in the product table represents one product with its own Mã FG

8. "Đvt" (Unit of Measurement):
   - Look for column headers: "Đvt", "ĐVT", "DVT", "Unit", "Đơn vị", "Đơn vị tính"
   - Common Vietnamese units include:
     * "Hộp" (box)
     * "Hũ" (jar)
     * "Gói" (pack/packet)
     * "Chai" (bottle)
     * "Thùng" (carton/crate)
     * "Kg" (kilogram)
     * "Lít" (liter)
   - Extract the unit name exactly as shown in the document
   - Preserve Vietnamese diacritical marks

9. "Số lượng" (Quantity):
   - Look for columns labeled: "Số lượng", "SL", "Quantity", "Qty", or similar
   - The table may have multiple quantity columns such as "Số lượng(Thùng)" for cartons and "Số lượng(Pcs)" for individual pieces
   - Extract the quantity value that corresponds to the product in that specific row
   - If the table shows both carton quantity and piece quantity, extract the value that represents the actual shipped quantity for that product
   - Common scenarios:
     * If only one quantity column exists, extract that value
     * If multiple quantity columns exist (Thùng and Pcs), extract the value from the primary quantity column used in the document
     * Look at the table structure and column headers to determine which quantity is being reported
   - Extract only the numeric value
   - If the quantity is unclear or not found, set to null

OUTPUT FORMAT:

Return data in JSON format with the following structure:

{{
  "metadata": {{
    {", ".join([f'"{f}": <value or null>' for f in METADATA_FIELDS])}
  }},
  "table_rows": [
    {{
      {", ".join([f'"{f}": <value or null>' for f in TABLE_ROW_FIELDS])}
    }},
    ... (one object per table row)
  ]
}}

CRITICAL EXTRACTION RULES:

1. "metadata" contains document-level information: {metadata_fields_list}
2. "table_rows" contains an array of objects, where each object represents one row from the product table
3. Each table row object contains: {table_row_fields_list}
4. Extract values exactly as they appear in the document
5. Preserve all formatting including dots, dashes, and spaces in numeric fields
6. Preserve all Vietnamese diacritical marks (ă, â, đ, ê, ô, ơ, ư, etc.)
7. For date fields, convert to dd/mm/yyyy format. If the format is unclear, set to null
8. For fields that cannot be found or do not exist in the document, set the value to null
9. DO NOT infer, guess, or make assumptions about missing data
10. DO NOT confuse "Kho nhận" (company/warehouse name) with "Địa chỉ" (street address)
11. DO NOT confuse "Số phiếu" (document number) with handwritten page numbers or AR code suffixes
12. Return ONLY valid JSON with no additional text, explanations, or markdown formatting

VALIDATION CHECKLIST:

Before returning your answer, verify that:
- "Số phiếu" is extracted from a labeled document number field, not from page numbers or AR codes
- "Số phiếu" preserves its original format with dots, dashes, or spaces
- "Kho nhận" contains a company or warehouse name, not a street address
- "Địa chỉ" contains a street address with location details, or is null if not present
- "Mã AR" includes the complete code with any suffixes
- All Vietnamese diacritics are correctly preserved
- Dates are in dd/mm/yyyy format or null
- Each product table row is a separate object in the "table_rows" array
- "Số lượng" for each row corresponds to the correct product quantity
- No values are invented or inferred
- Output is valid, parseable JSON

Extract data from this document now:"""

    return prompt