# -*- coding: utf-8 -*-
"""
Prepare and push dataset to Hugging Face Hub.

This script converts the extraction results to a Hugging Face dataset
with two columns: image (PIL Image loaded from file paths) and text (JSON string of metadata + table_rows).

The script uses a memory-efficient approach by saving images to disk temporarily
and processing them in a streaming fashion to avoid MemoryError issues.

Usage:
    python prepare_hf_dataset.py --results ./output/results.json --dataset-name your-username/dataset-name
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Generator, List, Optional

from datasets import Dataset, DatasetDict, Features, Value
from datasets import Image as HFImage
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

os.environ["HF_TOKEN"] = "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"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_results(results_path: str) -> List[Dict]:
    """Load extraction results from JSON file."""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} results from {results_path}")
    return results


def extract_page_from_pdf(
    pdf_path: str, page_number: int, dpi: int = 200
) -> Optional[Image.Image]:
    """Extract a specific page from PDF as PIL Image."""
    try:
        # pdf2image uses 1-based indexing
        images = convert_from_path(
            pdf_path, dpi=dpi, first_page=page_number, last_page=page_number
        )
        if images:
            return images[0]
        return None
    except Exception as e:
        logger.error(f"Failed to extract page {page_number} from {pdf_path}: {e}")
        return None


def create_text_annotation(metadata: List[Dict], table_rows: List[Dict]) -> str:
    """Create text annotation from metadata and table_rows in JSON format."""
    annotation = {"metadata": metadata, "table_rows": table_rows}
    return json.dumps(annotation, ensure_ascii=False, indent=2)


def prepare_dataset_rows(
    results: List[Dict], temp_image_dir: Path
) -> Generator[Dict, None, None]:
    """Convert results to dataset rows with image paths and text columns.

    This generator yields rows one at a time to avoid loading all images into memory.
    Images are saved to disk temporarily and their paths are stored.
    """
    # Cache for PDF images to avoid re-loading the same PDF multiple times
    pdf_cache = {}

    temp_image_dir.mkdir(parents=True, exist_ok=True)
    image_counter = 0

    for result in tqdm(results, desc="Processing results"):
        source_file = result.get("source_file")
        source_type = result.get("source_type")
        page_number = result.get("page_number", 1)
        metadata = result.get("metadata", [])
        table_rows = result.get("table_rows", [])

        # Skip if there's an error in the result
        if "error" in result:
            logger.warning(
                f"Skipping result with error: {source_file} page {page_number}"
            )
            continue

        # Get image
        image = None
        if source_type == "pdf":
            # Check cache first
            cache_key = f"{source_file}_{page_number}"
            if cache_key in pdf_cache:
                image = pdf_cache[cache_key]
            else:
                image = extract_page_from_pdf(source_file, page_number)
                if image:
                    pdf_cache[cache_key] = image
        elif source_type == "image":
            try:
                image = Image.open(source_file)
            except Exception as e:
                logger.error(f"Failed to load image {source_file}: {e}")

        if image is None:
            logger.warning(f"Failed to get image for {source_file} page {page_number}")
            continue

        # Save image to temporary directory
        image_counter += 1
        image_filename = f"image_{image_counter:06d}.png"
        image_path = temp_image_dir / image_filename

        try:
            # Convert to RGB if necessary (some images might be in other modes)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(image_path, format="PNG", optimize=True)
            # Close the image to free memory immediately
            image.close()
            # Remove from cache if it was cached (to free memory)
            if source_type == "pdf" and cache_key in pdf_cache:
                del pdf_cache[cache_key]
        except Exception as e:
            logger.error(f"Failed to save image {image_path}: {e}")
            continue

        # Create text annotation
        text = create_text_annotation(metadata, table_rows)

        # Yield row with image path instead of PIL Image object
        yield {
            "image": str(image_path),
            "text": text,
            "source_file": source_file,
            "page_number": page_number,
        }

    logger.info(f"Prepared {image_counter} dataset rows")


def create_huggingface_dataset(
    dataset_generator: Generator[Dict, None, None],
) -> Dataset:
    """Create Hugging Face Dataset from generator of rows.

    Uses from_generator to process rows one at a time, avoiding memory issues.
    """
    # Define features
    features = Features(
        {
            "image": HFImage(),
            "text": Value("string"),
            "source_file": Value("string"),
            "page_number": Value("int32"),
        }
    )

    # Create dataset from generator - this processes rows one at a time
    dataset = Dataset.from_generator(dataset_generator, features=features)
    logger.info(f"Created Hugging Face dataset with {len(dataset)} examples")

    return dataset


def split_dataset(dataset: Dataset, train_ratio: float = 0.9) -> DatasetDict:
    """Split dataset into train and validation sets."""
    # Calculate split size
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)

    # Split
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, total_size))

    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

    logger.info(
        f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation"
    )
    return dataset_dict


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare and push dataset to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create dataset locally only
  python prepare_hf_dataset.py --results ./output/results.json --output-dir ./hf_dataset
  
  # Create and push to Hugging Face
  python prepare_hf_dataset.py --results ./output/results.json --dataset-name username/dataset-name
  
  # Create with custom split ratio
  python prepare_hf_dataset.py --results ./output/results.json --dataset-name username/dataset-name --train-ratio 0.8
        """,
    )

    parser.add_argument(
        "-r",
        "--results",
        required=True,
        help="Path to results.json file",
    )
    parser.add_argument(
        "-d",
        "--dataset-name",
        help="Hugging Face dataset name (e.g., username/dataset-name). If not provided, only saves locally.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="./hf_dataset",
        help="Local output directory to save the dataset (default: ./hf_dataset)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train/validation split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Don't split into train/val, keep as single dataset",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private on Hugging Face",
    )

    args = parser.parse_args()

    # Load results
    results = load_results(args.results)

    if not results:
        logger.error("No results found in the JSON file!")
        return

    # Create temporary directory for images
    temp_image_dir = Path(tempfile.mkdtemp(prefix="hf_dataset_images_"))
    logger.info(f"Using temporary directory for images: {temp_image_dir}")

    try:
        # Prepare dataset rows generator
        dataset_generator = prepare_dataset_rows(results, temp_image_dir)

        # Create dataset from generator
        dataset = create_huggingface_dataset(dataset_generator)

        if len(dataset) == 0:
            logger.error("No valid dataset rows created!")
            return

        # Split dataset if requested
        if args.no_split:
            final_dataset = dataset
        else:
            final_dataset = split_dataset(dataset, args.train_ratio)

        # Save locally
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving dataset to {output_dir}")

        if isinstance(final_dataset, DatasetDict):
            final_dataset.save_to_disk(str(output_dir))
        else:
            final_dataset.save_to_disk(str(output_dir))

        logger.info("✓ Dataset saved locally")

        # Push to Hugging Face if dataset name provided
        if args.dataset_name:
            logger.info(f"Pushing dataset to Hugging Face: {args.dataset_name}")
            try:
                if isinstance(final_dataset, DatasetDict):
                    final_dataset.push_to_hub(args.dataset_name, private=args.private)
                else:
                    final_dataset.push_to_hub(args.dataset_name, private=args.private)
                logger.info("✓ Dataset pushed to Hugging Face successfully!")
                logger.info(
                    f"View at: https://huggingface.co/datasets/{args.dataset_name}"
                )
            except Exception as e:
                logger.error(f"Failed to push to Hugging Face: {e}")
                logger.info("Dataset is still saved locally.")
        else:
            logger.info("Skipping Hugging Face push (no dataset name provided)")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Summary:")
        if isinstance(final_dataset, DatasetDict):
            logger.info(f"  Train: {len(final_dataset['train'])} examples")
            logger.info(f"  Validation: {len(final_dataset['validation'])} examples")
        else:
            logger.info(f"  Total: {len(final_dataset)} examples")
        logger.info(f"  Local path: {output_dir}")
        if args.dataset_name:
            logger.info(f"  HF dataset: {args.dataset_name}")
        logger.info("=" * 60)

    finally:
        # Clean up temporary image directory
        if temp_image_dir.exists():
            logger.info(f"Cleaning up temporary directory: {temp_image_dir}")
            try:
                shutil.rmtree(temp_image_dir)
                logger.info("✓ Temporary directory cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")


if __name__ == "__main__":
    main()
