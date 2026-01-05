from data_processing.apikey_manager import ProcessingCheckpoint
from PIL import Image
from pdf2image import convert_from_path
import json
import os
from pathlib import Path
from datetime import datetime
import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_path: str) -> ProcessingCheckpoint:
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            checkpoint = ProcessingCheckpoint.from_dict(data)
            logger.info(f"Loaded checkpoint: {checkpoint.total_processed} items processed")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
    return ProcessingCheckpoint()


def save_checkpoint(checkpoint: ProcessingCheckpoint, checkpoint_path: str) -> None:
    checkpoint.last_updated = datetime.now().isoformat()
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)


def append_result(result: dict, output_path: str) -> None:
    results = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except (json.JSONDecodeError, Exception):
            results = []

    results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def get_supported_files(input_dir: str) -> list:
    supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    files = []

    input_path = Path(input_dir)
    if input_path.is_file():
        if input_path.suffix.lower() in supported_extensions:
            files.append(str(input_path))
    else:
        for ext in supported_extensions:
            files.extend([str(p) for p in input_path.rglob(f"*{ext}")])
            files.extend([str(p) for p in input_path.rglob(f"*{ext.upper()}")])

    files = sorted(list(set(files)))
    logger.info(f"Found {len(files)} supported file(s)")
    return files


def load_image(file_path: str) -> Image.Image:
    return Image.open(file_path)


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list:
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        logger.info(f"Converted PDF '{pdf_path}' to {len(images)} page(s)")
        return images
    except Exception as e:
        logger.error(f"Failed to convert PDF '{pdf_path}': {e}")
        return []


def parse_gemini_response(response_text: str) -> Optional[dict]:
    text = response_text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return None

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")