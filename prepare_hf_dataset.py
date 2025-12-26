# -*- coding: utf-8 -*-
"""
Prepare and push dataset to Hugging Face Hub.

This script converts the extraction results to a Hugging Face dataset
with two columns: image (PIL Image) and text (JSON string of metadata + table_rows).

Usage:
    python prepare_hf_dataset.py --results ./output/results.json --dataset-name your-username/dataset-name
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
import os

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


def extract_page_from_pdf(pdf_path: str, page_number: int, dpi: int = 200) -> Optional[Image.Image]:
    """Extract a specific page from PDF as PIL Image."""
    try:
        # pdf2image uses 1-based indexing
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_number,
            last_page=page_number
        )
        if images:
            return images[0]
        return None
    except Exception as e:
        logger.error(f"Failed to extract page {page_number} from {pdf_path}: {e}")
        return None


def create_text_annotation(metadata: List[Dict], table_rows: List[Dict]) -> str:
    """Create text annotation from metadata and table_rows in JSON format."""
    annotation = {
        "metadata": metadata,
        "table_rows": table_rows
    }
    return json.dumps(annotation, ensure_ascii=False, indent=2)


def prepare_dataset_rows(results: List[Dict]) -> List[Dict]:
    """Convert results to dataset rows with image and text columns."""
    dataset_rows = []
    
    # Cache for PDF images to avoid re-loading the same PDF multiple times
    pdf_cache = {}
    
    for result in tqdm(results, desc="Processing results"):
        source_file = result.get("source_file")
        source_type = result.get("source_type")
        page_number = result.get("page_number", 1)
        metadata = result.get("metadata", [])
        table_rows = result.get("table_rows", [])
        
        # Skip if there's an error in the result
        if "error" in result:
            logger.warning(f"Skipping result with error: {source_file} page {page_number}")
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
        
        # Create text annotation
        text = create_text_annotation(metadata, table_rows)
        
        # Add to dataset
        dataset_rows.append({
            "image": image,
            "text": text,
            "source_file": source_file,
            "page_number": page_number
        })
    
    logger.info(f"Prepared {len(dataset_rows)} dataset rows")
    return dataset_rows


def create_huggingface_dataset(dataset_rows: List[Dict]) -> Dataset:
    """Create Hugging Face Dataset from prepared rows."""
    # Define features
    features = Features({
        "image": HFImage(),
        "text": Value("string"),
        "source_file": Value("string"),
        "page_number": Value("int32")
    })
    
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
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation")
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
    
    # Prepare dataset rows
    dataset_rows = prepare_dataset_rows(results)
    
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
                final_dataset.push_to_hub(
                    args.dataset_name,
                    private=args.private
                )
            else:
                final_dataset.push_to_hub(
                    args.dataset_name,
                    private=args.private
                )
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

