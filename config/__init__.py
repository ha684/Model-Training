# -*- coding: utf-8 -*-
"""Config package for data extraction pipeline."""

from .fields_config import (
    DEFAULT_FIELDS,
    METADATA_FIELDS,
    TABLE_ROW_FIELDS,
    get_extraction_prompt,
    get_fields_schema,
)

__all__ = [
    "DEFAULT_FIELDS",
    "METADATA_FIELDS", 
    "TABLE_ROW_FIELDS",
    "get_extraction_prompt",
    "get_fields_schema",
]

