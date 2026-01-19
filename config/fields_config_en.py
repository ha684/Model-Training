# config/fields_config_en.py
# -*- coding: utf-8 -*-
"""
Configuration file for data extraction fields and prompts.
Modify the fields lists below to add/remove fields as needed.
"""

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
   - Examples: "A68/750", "A68/750_141", "AR1830"
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