# config/fields_config_1.py
# -*- coding: utf-8 -*-
"""
Configuration file for data extraction - Validation Pass.
Uses slightly different phrasing to catch extraction inconsistencies.
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
    This is a validation pass with slightly different instruction ordering.
    """
    fields_list = "\n".join([f"  - {field}" for field in DEFAULT_FIELDS])
    metadata_fields_list = ", ".join([f'"{field}"' for field in METADATA_FIELDS])
    table_row_fields_list = ", ".join([f'"{field}"' for field in TABLE_ROW_FIELDS])

    prompt = f"""You are an expert in extracting structured data from Vietnamese delivery notes (phiếu giao hàng), cargo documents, and invoices.

Your task is to extract the following fields from the provided document image:

{fields_list}

EXTRACTION INSTRUCTIONS:

DOCUMENT-LEVEL FIELDS (Metadata):

1. **"Số phiếu"** (Document Number/PO Number):
   - Search for field labels: "Số PO", "PO/PO No", "S.PO/PO No.", "Số:", "Số PGC", "Number"
   - Typically found in the header section (top-right area)
   - Extract the complete number preserving dots, dashes, and spaces EXACTLY as shown
   - Common formats include: "PXB25110076", "2544051804257", "252.104.868.0263", "252-104-866-7210"
   - CRITICAL VERIFICATION:
     * Is this number next to or under a label like "Số PO" or "Number"? → YES = extract it
     * Is this a handwritten number like "1/2" or "2/2"? → NO = ignore it
     * Is this number appearing after "Mã AR:" field? → NO = it's part of AR code
     * Does this number look like a date (03/11/2025)? → NO = it's a date field
   - Common mistakes to avoid:
     * Extracting "252.104.868.0263" when you see "Mã AR: A68/750_252.104.868.0263" (the long number is NOT Số phiếu)
     * Extracting "1/2" from handwritten page indicators
   - If the document number cannot be clearly identified from a labeled field, set to null

2. **"Ngày chứng từ(dd/mm/yyyy)"** (Document Date):
   - Search for labels: "Ngày/Date:", "Document date:", "Ngày chứng từ:"
   - Usually positioned near the document number
   - Output format must be EXACTLY dd/mm/yyyy (e.g., "03/11/2025")
   - Conversion rules:
     * If you see "03/11/2025" → keep as "03/11/2025"
     * If you see "3/11/2025" → convert to "03/11/2025" (add leading zero)
     * If you see "20251103" or "2025-11-03" → convert to "03/11/2025"
   - CRITICAL: Do not confuse with "Ngày xuất" (delivery date)
   - If multiple dates exist, choose the one specifically labeled as document date or issue date
   - If the date is ambiguous or missing, set to null

3. **"Ngày xuất(dd/mm/yyyy)"** (Export/Delivery Date):
   - Search for labels: "Ngày giao hàng", "Delivery date:", "Ngày xuất:", "Ngày xuất kho"
   - Output format must be EXACTLY dd/mm/yyyy
   - Conversion rules:
     * If you see "03/11/2025" → keep as "03/11/2025"
     * If you see "3/11/2025" → convert to "03/11/2025" (add leading zero)
     * If you see "20251103" or "2025-11-03" → convert to "03/11/2025"
   - CRITICAL: This may be identical to or different from "Ngày chứng từ"
   - If both dates are the same, verify you're extracting from the correct field
   - If not explicitly stated, set to null

4. **"Kho nhận"** (Receiving Party/Customer):
   - Search for sections: "Bên nhận hàng:", "Đơn vị nhận hàng:", "Tên người mua hàng:", "Customer's name:", "Đơn vị nhận:"
   - This field represents the COMPANY NAME or WAREHOUSE NAME ONLY
   - Extract the full organizational name EXACTLY as displayed
   - Examples: "CÔNG TY TNHH THIẾT BỊ CÔNG NGHIỆP VÀ DỊCH VỤ KỸ THUẬT THÁI SƠN", "CÔNG TY TNHH DỊCH VỤ EB", "kho SENKO", "DC KHO HẢI PHÒNG"
   - CRITICAL VERIFICATION:
     * Does this value contain street names, numbers, wards, or districts? → NO = it's an address, not company name
     * Does this value start with "Số", "KM", or contain "Phường", "Quận"? → NO = it's an address
     * Is this the first line under "Bên nhận hàng" before the address? → YES = likely company name
   - Common mistakes:
     * Mixing company name with address: "CÔNG TY ABC, Số 123 Đường XYZ" → extract only "CÔNG TY ABC"
     * Confusing warehouse location with warehouse name
   - Preserve exact capitalization and spacing
   - This is distinct from the street address field
   - If missing, set to null

5. **"Địa chỉ"** (Delivery Address):
   - Search for labels: "Địa chỉ:", "Địa chỉ nhận hàng:", "Delivery address:", "Đia chỉ:"
   - This field contains the COMPLETE STREET ADDRESS including location details
   - Extract full address with street numbers, street names, wards, districts, cities EXACTLY as shown
   - Examples: "Số 77 Kim Anh, Xã Nội Bài, TP.Hà Nội", "Số 163, Đường Phan Đăng Lưu, Phường Cầu Kiều, Thành phố Hồ Chí Minh, Việt Nam", "KM54+100, quốc lộ 5, khu 3, phường nhi châu, tp hải dương"
   - CRITICAL VERIFICATION:
     * Does this field exist with a label "Địa chỉ"? → YES = extract it
     * Is this just a company name without street details? → NO = it belongs in "Kho nhận"
     * Does the document show no address field at all? → set to null
   - Common mistakes:
     * Inventing an address when none is shown
     * Copying the company name into this field
     * Extracting partial address (missing ward/district/city)
   - NOTE: This is an OPTIONAL field - many documents do not have it
   - Preserve exact spelling, capitalization, and punctuation
   - Do not confuse with company/warehouse names (those belong in "Kho nhận")
   - If no address field exists in the document, MUST set to null

6. **"Mã AR"** (AR Code/Customer Account Code):
   - Search for labels: "Mã AR:", "Khách hàng:", "Customer code:", "AR Code:"
   - Format typically contains alphanumeric characters with slashes or underscores
   - Examples: "A68/750", "AR5239_141", "A68/750_141"
   - CRITICAL: Extract the COMPLETE code including ALL suffix numbers after underscores or slashes
   - Common mistakes:
     * Extracting only "A68/750" when the full code is "A68/750_141" (missing _141)
     * Confusing numbers after "Mã AR" as document number (they belong to AR code)
     * Extracting "A68" and "750_141" separately (should be one complete code)
   - VERIFICATION:
     * If you see "Mã AR: A68/750_141" → extract "A68/750_141" (the ENTIRE code)
     * If you see multiple numbers/codes on the same line → extract the one closest to the label
   - The portion after underscores is PART OF the AR code, not a separate field
   - Preserve exact format with slashes, underscores, and spacing
   - If not found, set to null

ITEM-LEVEL FIELDS (Table Rows):

7. **"Mã FG"** (Finished Goods Code/Product Code):
   - Located in the leftmost column of the product table
   - Common formats: "PG1490", "PG1495", "FGB0027", "FC9345", "PL2368", "PG8023"
   - Extract the alphanumeric code EXACTLY as shown (case-sensitive)
   - CRITICAL: Each table row represents a distinct product with its own code
   - Common mistakes:
     * Extracting row numbers instead of product codes
     * Missing letters or numbers (e.g., "PG149" instead of "PG1490")
     * Confusing similar codes (e.g., "PG1490" vs "PG1495")
   - Preserve exact capitalization and spacing
   - If a row has no product code, set to null for that row

8. **"Đvt"** (Unit of Measurement):
   - Look for column headers: "Đvt", "ĐVT", "DVT", "Unit", "Đơn vị", "Đơn vị tính"
   - Common Vietnamese units:
     * "Hộp" (box)
     * "Hũ" (jar)
     * "Gói" (pack/packet)
     * "Chai" (bottle)
     * "Thùng" (carton/crate)
     * "Kg" (kilogram)
     * "Lít" (liter)
   - Extract unit name EXACTLY as displayed in the document
   - CRITICAL: Preserve all Vietnamese diacritical marks precisely
   - Common mistakes:
     * "Hop" instead of "Hộp" (missing tone mark)
     * "Hu" instead of "Hũ" (missing tone mark)
     * "Goi" instead of "Gói" (missing tone mark)
   - If the unit for a row is unclear or missing, set to null for that row

9. **"Số lượng"** (Quantity):
   - Search for columns: "Số lượng", "SL", "Quantity", "Qty", or similar
   - The table may contain multiple quantity columns (e.g., "Số lượng(Thùng)" for cartons and "Số lượng(Pcs)" for pieces)
   - Extract the quantity value corresponding to EACH SPECIFIC product in that row
   - Guidelines for extraction:
     * Single quantity column: extract that value
     * Multiple quantity columns: extract the primary quantity shown for that specific product
     * Review table structure and column headers carefully to determine the reported quantity
   - CRITICAL DATA TYPE:
     * Extract as a NUMBER (integer or float), NOT as a string
     * Correct: 168 or 168.5
     * Wrong: "168" or "168.5" (these are strings, not numbers)
   - CRITICAL VERIFICATION:
     * Extract ONLY numeric values (no text, no units)
     * Each row gets its OWN quantity (do not sum across rows)
     * Verify you're reading from the correct column for that product
   - Common mistakes:
     * Returning quantity as string instead of number
     * Extracting quantity from wrong column
     * Including units in the number (e.g., "100 Hộp" instead of 100)
     * Summing quantities instead of extracting individual row values
     * Confusing "Số lượng(Thùng)" with "Số lượng(Pcs)"
   - If quantity is unclear or missing for a row, set to null for that row

JSON OUTPUT FORMAT:

Return the data in this exact JSON structure:

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

CRITICAL DATA TYPE REQUIREMENTS:
- "Số lượng" MUST be a number (integer or float), NOT a string
  * Correct: {{"Số lượng": 168}} or {{"Số lượng": 168.5}}
  * Wrong: {{"Số lượng": "168"}} or {{"Số lượng": "168.5"}}
- All other fields can be strings or null
- Dates must be strings in dd/mm/yyyy format

EXTRACTION RULES:

1. The "metadata" object contains document-level fields: {metadata_fields_list}
2. The "table_rows" array contains objects representing product table rows
3. Each table row object includes: {table_row_fields_list}
4. Extract all values exactly as they appear in the document
5. Preserve formatting such as dots, dashes, and spaces in alphanumeric fields
6. Preserve all Vietnamese diacritical marks (ă, â, đ, ê, ô, ơ, ư, etc.)
7. Convert dates to dd/mm/yyyy format; if format is uncertain, set to null
8. For fields that are not found or do not exist in the document, use null
9. Do not infer, estimate, or assume values for missing data
10. Do not confuse "Kho nhận" (company/warehouse name) with "Địa chỉ" (street address)
11. Do not confuse "Số phiếu" (document number) with page numbers or AR code suffixes
12. Return only valid JSON without additional text, explanations, or markdown formatting

VERIFICATION CHECKLIST:

Before finalizing your response, verify:
- "Số phiếu" is extracted from a labeled field (not page numbers or AR codes)
- "Số phiếu" retains its original format with dots, dashes, or spaces
- "Kho nhận" contains a company or warehouse name (not a street address)
- "Địa chỉ" contains a street address with details, or is null if absent
- "Mã AR" includes the complete code with any suffixes
- All Vietnamese diacritics are correctly preserved
- Dates follow dd/mm/yyyy format or are null
- Each product table row is represented as a separate object in "table_rows"
- "Số lượng" for each row matches the correct product quantity
- No values are fabricated or assumed
- Output is valid, parseable JSON

Now extract the data from this document:"""

    return prompt