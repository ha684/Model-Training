# -*- coding: utf-8 -*-
"""
Prepare and push dataset to Hugging Face Hub.

This script converts the extraction results to a Hugging Face dataset
with two columns: image (PIL Image) and text (JSON string of metadata + table_rows).

Usage:
    python prepare_hf_dataset.py --results ./output/results.json --dataset-name your-username/dataset-name
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict, Features, Value
from datasets import Image as HFImage
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    pdf_path: str, page_number: int, output_dir: str, dpi: int = 200
) -> Optional[str]:
    """Extract a specific page from PDF and save to disk.

    Returns the path to the saved image file, or None on failure.
    """
    try:
        # Create a unique filename based on source path and page number
        hash_input = f"{pdf_path}_{page_number}"
        file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        output_path = os.path.join(output_dir, f"page_{file_hash}.png")

        # Skip if already extracted
        if os.path.exists(output_path):
            return output_path

        # pdf2image uses 1-based indexing
        images = convert_from_path(
            pdf_path, dpi=dpi, first_page=page_number, last_page=page_number
        )
        if images:
            images[0].save(output_path, format="PNG")
            # Explicitly close/delete the PIL image to free memory
            images[0].close()
            del images
            return output_path
        return None
    except Exception as e:
        logger.error(f"Failed to extract page {page_number} from {pdf_path}: {e}")
        return None


def create_text_annotation(metadata: List[Dict], table_rows: List[Dict]) -> str:
    """Create text annotation from metadata and table_rows in JSON format."""
    annotation = {"metadata": metadata, "table_rows": table_rows}
    return json.dumps(annotation, ensure_ascii=False, indent=2)


def prepare_dataset_rows(results: List[Dict], image_cache_dir: str) -> List[Dict]:
    """Convert results to dataset rows with image paths and text columns.

    Images are saved to disk to avoid memory issues with large datasets.
    """
    dataset_rows = []

    # Ensure image cache directory exists
    os.makedirs(image_cache_dir, exist_ok=True)
    logger.info(f"Saving extracted images to: {image_cache_dir}")

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

        # Get image path (images are saved to disk to avoid memory issues)
        image_path = None
        if source_type == "pdf":
            image_path = extract_page_from_pdf(
                source_file, page_number, image_cache_dir
            )
        elif source_type == "image":
            # For image sources, copy to cache dir to ensure consistent handling
            try:
                if os.path.exists(source_file):
                    hash_input = f"{source_file}"
                    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
                    ext = Path(source_file).suffix or ".png"
                    cached_path = os.path.join(image_cache_dir, f"img_{file_hash}{ext}")

                    if not os.path.exists(cached_path):
                        # Open, convert to RGB if needed, and save
                        with Image.open(source_file) as img:
                            if img.mode in ("RGBA", "P"):
                                img = img.convert("RGB")
                            img.save(cached_path, format="PNG")
                    image_path = cached_path
            except Exception as e:
                logger.error(f"Failed to process image {source_file}: {e}")

        if image_path is None:
            logger.warning(f"Failed to get image for {source_file} page {page_number}")
            continue

        # Create text annotation
        text = create_text_annotation(metadata, table_rows)

        # Add to dataset (using file path instead of PIL Image object)
        dataset_rows.append(
            {
                "image": image_path,
                "text": text,
                "source_file": source_file,
                "page_number": page_number,
            }
        )

    logger.info(f"Prepared {len(dataset_rows)} dataset rows")
    return dataset_rows


def create_huggingface_dataset(dataset_rows: List[Dict]) -> Dataset:
    """Create Hugging Face Dataset from prepared rows."""
    # Define features
    features = Features(
        {
            "image": HFImage(),
            "text": Value("string"),
            "source_file": Value("string"),
            "page_number": Value("int32"),
        }
    )

    # Create dataset
    dataset = Dataset.from_list(dataset_rows, features=features)
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
    parser.add_argument(
        "--image-cache-dir",
        default=None,
        help="Directory to cache extracted images (default: <output-dir>/images)",
    )

    args = parser.parse_args()

    # Load results
    results = load_results(args.results)

    if not results:
        logger.error("No results found in the JSON file!")
        return

    # Set up image cache directory
    output_dir = Path(args.output_dir)
    image_cache_dir = args.image_cache_dir or str(output_dir / "images")

    # Prepare dataset rows (saves images to disk to avoid memory issues)
    dataset_rows = prepare_dataset_rows(results, image_cache_dir)

    if not dataset_rows:
        logger.error("No valid dataset rows created!")
        return

    # Create dataset
    dataset = create_huggingface_dataset(dataset_rows)

    # Split dataset if requested
    if args.no_split:
        final_dataset = dataset
    else:
        final_dataset = split_dataset(dataset, args.train_ratio)

    # Save locally
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
            logger.info(f"View at: https://huggingface.co/datasets/{args.dataset_name}")
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


if __name__ == "__main__":
    main()
