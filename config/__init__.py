# -*- coding: utf-8 -*-
"""Config package for data extraction pipeline."""

try:
    from .fields_config_en import (
        DEFAULT_FIELDS,
        METADATA_FIELDS,
        TABLE_ROW_FIELDS,
        get_extraction_prompt,
    )
except ImportError:
    DEFAULT_FIELDS = []
    METADATA_FIELDS = []
    TABLE_ROW_FIELDS = []

    def get_extraction_prompt():
        from .prompt_generator import generate_extraction_prompt

        return generate_extraction_prompt("en")


from .fields_definitions import (
    FIELD_DEFINITIONS,
    FieldDefinition,
    FieldType,
    Language,
    get_field_names,
    get_fields_by_type,
)
from .prompt_generator import (
    generate_extraction_prompt,
    generate_random_prompt,
    get_all_fields,
    get_extraction_prompt_en,
    get_extraction_prompt_ja,
    get_extraction_prompt_vi,
    get_metadata_fields,
    get_table_fields,
    random_remove_fields,
    random_select_fields,
)

__all__ = [
    # Legacy exports
    "DEFAULT_FIELDS",
    "METADATA_FIELDS",
    "TABLE_ROW_FIELDS",
    "get_extraction_prompt",
    # New dynamic system
    "generate_extraction_prompt",
    "get_extraction_prompt_en",
    "get_extraction_prompt_ja",
    "get_extraction_prompt_vi",
    "get_metadata_fields",
    "get_table_fields",
    "get_all_fields",
    # Random field selection
    "random_select_fields",
    "random_remove_fields",
    "generate_random_prompt",
    # Field definitions
    "FIELD_DEFINITIONS",
    "FieldDefinition",
    "Language",
    "FieldType",
    "get_fields_by_type",
    "get_field_names",
]
