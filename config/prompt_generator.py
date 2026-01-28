# config/prompt_generator.py
# -*- coding: utf-8 -*-
"""
Dynamic prompt generator that builds prompts based on field definitions.
Prompts automatically adjust when fields are added/removed.
"""

from typing import List

from .fields_definitions import (
    FIELD_DEFINITIONS,
    FieldDefinition,
    Language,
)

PROMPT_TEMPLATES = {
    "en": {
        "header": "Extract structured data from this Vietnamese delivery document and return as JSON.",
        "output_requirements": "OUTPUT REQUIREMENTS:",
        "return_format": "Return data in JSON format with the following structure:",
        "metadata_label": "metadata",
        "table_rows_label": "table_rows",
        "value_placeholder": "<value or null>",
        "extraction_instructions": "#EXTRACTION INSTRUCTIONS:",
        "metadata_section": "### METADATA FIELDS:",
        "table_section": "### TABLE FIELDS:",
        "important_rules": "IMPORTANT RULES:",
        "rules": [
            "Return ONLY valid JSON with no markdown formatting or explanations",
            "{quantity_field} field MUST be number type (integer or float), all other fields are strings or null",
            "Dates are already in dd/mm/yyyy format - extract exactly as shown",
            "Each product row in the table creates one object in {table_rows} array",
            "Preserve Vietnamese diacritics in text values (ă, â, đ, ê, ô, ơ, ư, etc.)",
            "Set fields to null if not found - do not guess or make assumptions",
            "Document number is in the header, NOT the number that appears after customer code",
            "Include complete ar_code with suffix if present",
        ],
    },
    "ja": {
        "header": "このベトナム語の出庫伝票から構造化データを抽出し、JSON形式で返してください。",
        "output_requirements": "出力要件:",
        "return_format": "以下の構造でJSON形式でデータを返す:",
        "metadata_label": "metadata",
        "table_rows_label": "table_rows",
        "value_placeholder": "<値またはnull>",
        "extraction_instructions": "#抽出手順:",
        "metadata_section": "### メタデータフィールド:",
        "table_section": "### テーブルフィールド:",
        "important_rules": "重要なルール:",
        "rules": [
            "マークダウン形式や説明なしで有効なJSONのみを返す",
            "{quantity_field}フィールドは数値型(整数または浮動小数点数)である必要があります。他のすべてのフィールドは文字列またはnull",
            "日付はすでにdd/mm/yyyy形式 - 表示通りに正確に抽出",
            "テーブル内の各製品行が{table_rows}配列内の1つのオブジェクトを作成",
            "テキスト値内のベトナム語の発音記号を保持(ă, â, đ, ê, ô, ơ, ư など)",
            "見つからない場合はフィールドをnullに設定 - 推測や仮定をしない",
            "伝票番号はヘッダーにあります。顧客コードの後に表示される番号ではありません",
            "存在する場合は接尾辞を含む完全なARコードを含める",
        ],
    },
    "vi": {
        "header": "Trích xuất dữ liệu có cấu trúc từ phiếu xuất kho tiếng Việt này và trả về dưới dạng JSON.",
        "output_requirements": "YÊU CẦU ĐẦU RA:",
        "return_format": "Trả về dữ liệu dưới định dạng JSON với cấu trúc sau:",
        "metadata_label": "metadata",
        "table_rows_label": "table_rows",
        "value_placeholder": "<giá trị hoặc null>",
        "extraction_instructions": "#HƯỚNG DẪN TRÍCH XUẤT:",
        "metadata_section": "### CÁC TRƯỜNG METADATA:",
        "table_section": "### CÁC TRƯỜNG TRONG BẢNG:",
        "important_rules": "QUY TẮC QUAN TRỌNG:",
        "rules": [
            "CHỈ trả về JSON hợp lệ, không có markdown formatting hay giải thích",
            "Trường {quantity_field} BẮT BUỘC là kiểu số (integer hoặc float), các trường khác là chuỗi hoặc null",
            "Ngày đã ở định dạng dd/mm/yyyy - trích xuất chính xác như hiển thị",
            "Mỗi dòng sản phẩm trong bảng tạo một object trong mảng {table_rows}",
            "Giữ nguyên dấu tiếng Việt trong giá trị văn bản (ă, â, đ, ê, ô, ơ, ư, v.v.)",
            "Đặt trường là null nếu không tìm thấy - không đoán hoặc giả định",
            "Số phiếu ở phần header, KHÔNG phải số xuất hiện sau mã khách hàng",
            "Bao gồm Mã AR đầy đủ với hậu tố nếu có",
        ],
    },
}


def _generate_json_structure(
    metadata_fields: List[str], table_fields: List[str], template: dict
) -> str:
    """Generate JSON structure example."""
    parts = []
    
    if metadata_fields:
        metadata_items = ", ".join(
            [f'"{f}": {template["value_placeholder"]}' for f in metadata_fields]
        )
        parts.append(f'  "{template["metadata_label"]}": [\n    {{\n      {metadata_items}\n    }}\n  ]')
    else:
        parts.append(f'  "{template["metadata_label"]}": []')

    table_items = ", ".join(
        [f'"{f}": {template["value_placeholder"]}' for f in table_fields]
    )
    parts.append(f'  "{template["table_rows_label"]}": [\n    {{\n      {table_items}\n    }}\n  ]')
    
    return "{\n" + ",\n".join(parts) + "\n}"


def _generate_field_descriptions(
    fields: List[FieldDefinition], language: Language
) -> str:
    """Generate field descriptions without numbering."""
    descriptions = []
    for field in fields:
        field_name = field.get_name(language)
        field_desc = field.get_description(language)
        descriptions.append(f"{field_name}\n{field_desc}")

    return "\n\n".join(descriptions)


def generate_extraction_prompt(
    language: Language = "en", custom_fields: List[FieldDefinition] = None
) -> str:
    """
    Generate extraction prompt dynamically based on language and field definitions.

    Args:
        language: Language code ("en", "ja", or "vi")
        custom_fields: Optional custom field list. If None, uses FIELD_DEFINITIONS

    Returns:
        Generated prompt string

    Example:
        >>> prompt_en = generate_extraction_prompt("en")
        >>> prompt_ja = generate_extraction_prompt("ja")
        >>> prompt_vi = generate_extraction_prompt("vi")
    """
    fields = custom_fields if custom_fields is not None else FIELD_DEFINITIONS

    template = PROMPT_TEMPLATES[language]

    metadata_fields_defs = [f for f in fields if f.field_type == "metadata"]
    table_fields_defs = [f for f in fields if f.field_type == "table"]

    metadata_field_names = [f.get_name(language) for f in metadata_fields_defs]
    table_field_names = [f.get_name(language) for f in table_fields_defs]

    quantity_field_def = next((f for f in fields if f.key == "quantity"), None)
    quantity_field_name = (
        quantity_field_def.get_name(language) if quantity_field_def else "quantity"
    )

    sections = []

    # Header
    sections.append(template["header"])
    sections.append("")

    sections.append(template["output_requirements"])
    sections.append(f"- {template['return_format']}")
    sections.append(
        _generate_json_structure(metadata_field_names, table_field_names, template)
    )
    sections.append("")

    sections.append(template["extraction_instructions"])
    sections.append("")

    if metadata_fields_defs:
        sections.append(template["metadata_section"])
        sections.append("")
        sections.append(
            _generate_field_descriptions(metadata_fields_defs, language)
        )
        sections.append("")

    sections.append(template["table_section"])
    sections.append("")
    sections.append(
        _generate_field_descriptions(table_fields_defs, language)
    )
    sections.append("")

    sections.append(template["important_rules"])
    for rule in template["rules"]:
        if "{quantity_field}" in rule and not any(f.key == "quantity" for f in fields):
            continue
        # Skip rules that reference table_rows if no table fields
        if "{table_rows}" in rule and not table_fields_defs:
            continue
        
        formatted_rule = rule.format(
            quantity_field=quantity_field_name, table_rows=template["table_rows_label"]
        )
        sections.append(f"- {formatted_rule}")

    return "\n".join(sections)