# -*- coding: utf-8 -*-
"""Config package for data extraction pipeline."""

from .fields_definitions import (
    FIELD_DEFINITIONS,
    FieldDefinition,
    FieldType,
    Language,
)
from .prompt_generator import (
    generate_extraction_prompt,
)

__all__ = [
    "generate_extraction_prompt",
    "FIELD_DEFINITIONS",
    "FieldDefinition",
    "Language",
    "FieldType",
]
