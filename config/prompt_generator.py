# config/prompt_generator.py
# -*- coding: utf-8 -*-
"""
Dynamic prompt generator that builds prompts based on field definitions.
Prompts automatically adjust when fields are added/removed.
"""

import random
from typing import List, Optional

from .fields_definitions import (
    FIELD_DEFINITIONS,
    FieldDefinition,
    Language,
    get_field_names,
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
    metadata_items = ", ".join(
        [f'"{f}": {template["value_placeholder"]}' for f in metadata_fields]
    )
    table_items = ", ".join(
        [f'"{f}": {template["value_placeholder"]}' for f in table_fields]
    )

    return f"""{{
  "{template["metadata_label"]}": {{
    {metadata_items}
  }},
  "{template["table_rows_label"]}": [
    {{
      {table_items}
    }}
  ]
}}"""


def _generate_field_descriptions(
    fields: List[FieldDefinition], language: Language, start_index: int = 1
) -> str:
    """Generate numbered field descriptions."""
    descriptions = []
    for i, field in enumerate(fields, start=start_index):
        field_name = field.get_name(language)
        field_desc = field.get_description(language)
        descriptions.append(f"{i}. {field_name}\n{field_desc}")

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
    # Use custom fields or default
    fields = custom_fields if custom_fields is not None else FIELD_DEFINITIONS

    # Get template for language
    template = PROMPT_TEMPLATES[language]

    # Get metadata and table fields
    metadata_fields_defs = [f for f in fields if f.field_type == "metadata"]
    table_fields_defs = [f for f in fields if f.field_type == "table"]

    metadata_field_names = [f.get_name(language) for f in metadata_fields_defs]
    table_field_names = [f.get_name(language) for f in table_fields_defs]

    # Find quantity field name for rules
    quantity_field_def = next((f for f in fields if f.key == "quantity"), None)
    quantity_field_name = (
        quantity_field_def.get_name(language) if quantity_field_def else "quantity"
    )

    # Build prompt sections
    sections = []

    # Header
    sections.append(template["header"])
    sections.append("")

    # Output requirements
    sections.append(template["output_requirements"])
    sections.append(f"- {template['return_format']}")
    sections.append(
        _generate_json_structure(metadata_field_names, table_field_names, template)
    )
    sections.append("")

    # Extraction instructions
    sections.append(template["extraction_instructions"])
    sections.append("")

    # Metadata fields
    sections.append(template["metadata_section"])
    sections.append("")
    sections.append(
        _generate_field_descriptions(metadata_fields_defs, language, start_index=1)
    )
    sections.append("")

    # Table fields
    sections.append(template["table_section"])
    sections.append("")
    table_start_index = len(metadata_fields_defs) + 1
    sections.append(
        _generate_field_descriptions(
            table_fields_defs, language, start_index=table_start_index
        )
    )
    sections.append("")

    # Important rules
    sections.append(template["important_rules"])
    for rule in template["rules"]:
        formatted_rule = rule.format(
            quantity_field=quantity_field_name, table_rows=template["table_rows_label"]
        )
        sections.append(f"- {formatted_rule}")

    return "\n".join(sections)


def get_metadata_fields(language: Language = "en") -> List[str]:
    """Get list of metadata field names for a language."""
    return get_field_names(language, field_type="metadata")


def get_table_fields(language: Language = "en") -> List[str]:
    """Get list of table field names for a language."""
    return get_field_names(language, field_type="table")


def get_all_fields(language: Language = "en") -> List[str]:
    """Get all field names for a language."""
    return get_field_names(language)


# Convenience functions for backward compatibility
def get_extraction_prompt_en() -> str:
    """Generate English extraction prompt."""
    return generate_extraction_prompt("en")


def get_extraction_prompt_ja() -> str:
    """Generate Japanese extraction prompt."""
    return generate_extraction_prompt("ja")


def get_extraction_prompt_vi() -> str:
    """Generate Vietnamese extraction prompt."""
    return generate_extraction_prompt("vi")


def random_select_fields(
    num_fields: Optional[int] = None,
    num_metadata: Optional[int] = None,
    num_table: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[FieldDefinition]:
    """
    Randomly select a subset of fields.

    Args:
        num_fields: Total number of fields to select (if specified, overrides num_metadata and num_table)
        num_metadata: Number of metadata fields to select
        num_table: Number of table fields to select
        seed: Random seed for reproducibility

    Returns:
        List of randomly selected field definitions

    Examples:
        # Select random 5 fields total
        fields = random_select_fields(num_fields=5)

        # Select 3 metadata + 2 table fields
        fields = random_select_fields(num_metadata=3, num_table=2)

        # With seed for reproducibility
        fields = random_select_fields(num_fields=4, seed=42)
    """
    if seed is not None:
        random.seed(seed)

    metadata_fields = [f for f in FIELD_DEFINITIONS if f.field_type == "metadata"]
    table_fields = [f for f in FIELD_DEFINITIONS if f.field_type == "table"]

    if num_fields is not None:
        # Select random num_fields from all fields
        all_fields = FIELD_DEFINITIONS.copy()
        random.shuffle(all_fields)
        return all_fields[:num_fields]

    # Select specific number from each type
    selected = []

    if num_metadata is not None:
        num_metadata = min(num_metadata, len(metadata_fields))
        selected.extend(random.sample(metadata_fields, num_metadata))
    else:
        selected.extend(metadata_fields)

    if num_table is not None:
        num_table = min(num_table, len(table_fields))
        selected.extend(random.sample(table_fields, num_table))
    else:
        selected.extend(table_fields)

    return selected


def random_remove_fields(
    num_to_remove: int,
    seed: Optional[int] = None,
) -> List[FieldDefinition]:
    """
    Randomly remove a number of fields from FIELD_DEFINITIONS.

    Args:
        num_to_remove: Number of fields to remove
        seed: Random seed for reproducibility

    Returns:
        Remaining field definitions after removal

    Example:
        # Remove 4 random fields, keep the rest
        fields = random_remove_fields(4)
        prompt = generate_extraction_prompt("en", custom_fields=fields)
    """
    if seed is not None:
        random.seed(seed)

    all_fields = FIELD_DEFINITIONS.copy()
    num_to_remove = min(num_to_remove, len(all_fields))

    # Randomly select fields to keep
    num_to_keep = len(all_fields) - num_to_remove
    return random.sample(all_fields, num_to_keep)


def generate_random_prompt(
    language: Language = "en",
    num_fields: Optional[int] = None,
    num_metadata: Optional[int] = None,
    num_table: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple[str, List[FieldDefinition]]:
    """
    Generate prompt with randomly selected fields.

    Args:
        language: Language code ("en", "ja", or "vi")
        num_fields: Total number of fields to select
        num_metadata: Number of metadata fields to select
        num_table: Number of table fields to select
        seed: Random seed for reproducibility

    Returns:
        Tuple of (generated_prompt, selected_fields)

    Examples:
        # Generate prompt with only 5 random fields
        prompt, fields = generate_random_prompt("en", num_fields=5)

        # Generate prompt with 2 metadata + 2 table fields
        prompt, fields = generate_random_prompt("vi", num_metadata=2, num_table=2)

        # With seed for reproducibility
        prompt, fields = generate_random_prompt("ja", num_fields=4, seed=42)
    """
    selected_fields = random_select_fields(
        num_fields=num_fields,
        num_metadata=num_metadata,
        num_table=num_table,
        seed=seed,
    )

    prompt = generate_extraction_prompt(language, custom_fields=selected_fields)

    return prompt, selected_fields
