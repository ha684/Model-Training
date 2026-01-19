# -*- coding: utf-8 -*-
"""
Augment dataset by randomly removing fields from metadata and table rows.
Creates augmented versions with different field combinations for data augmentation.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DATASET_VIETNAMESE = "ha684/senko-V1"
DATASET_ENGLISH = "ha684/senko-V1-english"
DATASET_JAPANESE = "ha684/senko-V1-japanese"

OUTPUT_DATASET = "ha684/senko-V1-augmented-fields"

ALL_METADATA_FIELDS = [
    "document_number",
    "document_date",
    "export_date",
    "receiving_warehouse",
    "address",
    "ar_code",
]

TABLE_OPTIONAL_FIELDS = ["unit", "quantity"]


def random_remove_metadata_fields(
    metadata: Dict[str, Any],
    min_keep: int = 2,
    max_keep: int = 5,
    seed: int = None,
) -> Dict[str, Any]:
    """
    Randomly remove some metadata fields.

    Args:
        metadata: Original metadata dict
        min_keep: Minimum number of fields to keep
        max_keep: Maximum number of fields to keep
        seed: Random seed

    Returns:
        Metadata with randomly removed fields
    """
    if seed is not None:
        random.seed(seed)

    # Get current fields that have values
    available_fields = [k for k in metadata.keys() if metadata[k] is not None]

    if not available_fields:
        return metadata

    # Decide how many fields to keep
    num_keep = random.randint(
        min(min_keep, len(available_fields)), min(max_keep, len(available_fields))
    )

    # Randomly select fields to keep
    fields_to_keep = random.sample(available_fields, num_keep)

    # Create new metadata with only selected fields
    new_metadata = {
        k: metadata[k] if k in fields_to_keep else None for k in metadata.keys()
    }

    return new_metadata


def random_remove_table_field(
    table_rows: List[Dict[str, Any]],
    seed: int = None,
) -> List[Dict[str, Any]]:
    """
    Randomly remove either 'unit' or 'quantity' from table rows.
    Always keeps 'product_code'.

    Args:
        table_rows: List of table row dicts
        seed: Random seed

    Returns:
        Table rows with randomly removed field
    """
    if seed is not None:
        random.seed(seed)

    if not table_rows:
        return table_rows

    # Randomly choose to remove 'unit' or 'quantity'
    field_to_remove = random.choice(TABLE_OPTIONAL_FIELDS)

    # Create new table rows without the chosen field
    new_table_rows = []
    for row in table_rows:
        new_row = row.copy()
        if field_to_remove in new_row:
            new_row[field_to_remove] = None
        new_table_rows.append(new_row)

    return new_table_rows


def augment_sample(
    sample: Dict[str, Any],
    min_metadata: int = 2,
    max_metadata: int = 5,
    remove_table_field: bool = True,
) -> Dict[str, Any]:
    """
    Augment a single sample by randomly removing fields.

    Args:
        sample: Original sample
        min_metadata: Min metadata fields to keep
        max_metadata: Max metadata fields to keep
        remove_table_field: Whether to remove a table field

    Returns:
        Augmented sample
    """
    # Parse text field (structured_output)
    text_data = sample.get("text")
    if isinstance(text_data, str):
        structured_output = json.loads(text_data)
    else:
        structured_output = text_data

    # Get metadata dict (it's a list with 1 dict)
    metadata_list = structured_output["metadata"]
    if isinstance(metadata_list, list) and len(metadata_list) > 0:
        metadata_dict = metadata_list[0]
    else:
        metadata_dict = metadata_list  # Fallback if already a dict

    # Augment metadata
    new_metadata_dict = random_remove_metadata_fields(
        metadata_dict,
        min_keep=min_metadata,
        max_keep=max_metadata,
    )

    # Augment table rows
    new_table_rows = structured_output["table_rows"]
    if remove_table_field:
        new_table_rows = random_remove_table_field(new_table_rows)

    # Create new structured output (wrap metadata back in list)
    new_structured_output = {
        "metadata": [new_metadata_dict],  # Wrap back in list
        "table_rows": new_table_rows,
    }

    # Create augmented sample
    augmented = sample.copy()
    augmented["text"] = json.dumps(new_structured_output, ensure_ascii=False)

    # Add augmentation flag
    augmented["augmented"] = True
    augmented["augmentation_type"] = "random_field_removal"

    return augmented


def augment_dataset(
    dataset: Dataset,
    min_metadata: int = 2,
    max_metadata: int = 5,
    remove_table_field: bool = True,
) -> Dataset:
    """
    Augment entire dataset.

    Args:
        dataset: Input dataset
        min_metadata: Min metadata fields to keep
        max_metadata: Max metadata fields to keep
        remove_table_field: Whether to remove table field

    Returns:
        Augmented dataset
    """
    logger.info(f"Augmenting dataset with {len(dataset)} samples...")

    augmented_samples = []
    for sample in tqdm(dataset, desc="Augmenting"):
        augmented = augment_sample(
            sample,
            min_metadata=min_metadata,
            max_metadata=max_metadata,
            remove_table_field=remove_table_field,
        )
        augmented_samples.append(augmented)

    # Convert back to dataset
    augmented_dataset = Dataset.from_list(augmented_samples)

    logger.info(f"Augmented dataset size: {len(augmented_dataset)}")
    return augmented_dataset


def load_and_augment_multilingual():
    """Load Vietnamese, English, Japanese datasets and create augmented versions."""

    logger.info("=" * 80)
    logger.info("Loading source datasets...")
    logger.info("=" * 80)

    # Load datasets (train + validation)
    logger.info(f"Loading Vietnamese dataset: {DATASET_VIETNAMESE}")
    ds_vi_train = load_dataset(DATASET_VIETNAMESE, split="train")
    ds_vi_val = load_dataset(DATASET_VIETNAMESE, split="validation")

    logger.info(f"Loading English dataset: {DATASET_ENGLISH}")
    ds_en_train = load_dataset(DATASET_ENGLISH, split="train")
    ds_en_val = load_dataset(DATASET_ENGLISH, split="validation")

    logger.info(f"Loading Japanese dataset: {DATASET_JAPANESE}")
    ds_ja_train = load_dataset(DATASET_JAPANESE, split="train")
    ds_ja_val = load_dataset(DATASET_JAPANESE, split="validation")

    logger.info("\nDataset sizes:")
    logger.info(f"  Vietnamese: train={len(ds_vi_train)}, val={len(ds_vi_val)}")
    logger.info(f"  English: train={len(ds_en_train)}, val={len(ds_en_val)}")
    logger.info(f"  Japanese: train={len(ds_ja_train)}, val={len(ds_ja_val)}")

    # Merge train + validation for each language
    logger.info("\n" + "=" * 80)
    logger.info("Merging train + validation splits...")
    logger.info("=" * 80)

    ds_vi = concatenate_datasets([ds_vi_train, ds_vi_val])
    ds_en = concatenate_datasets([ds_en_train, ds_en_val])
    ds_ja = concatenate_datasets([ds_ja_train, ds_ja_val])

    logger.info(f"  Vietnamese total: {len(ds_vi)}")
    logger.info(f"  English total: {len(ds_en)}")
    logger.info(f"  Japanese total: {len(ds_ja)}")

    # Augment each dataset
    logger.info("\n" + "=" * 80)
    logger.info("Augmenting datasets...")
    logger.info("=" * 80)

    logger.info("\nAugmenting Vietnamese dataset...")
    ds_vi_aug = augment_dataset(
        ds_vi, min_metadata=2, max_metadata=5, remove_table_field=True
    )

    logger.info("\nAugmenting English dataset...")
    ds_en_aug = augment_dataset(
        ds_en, min_metadata=2, max_metadata=5, remove_table_field=True
    )

    logger.info("\nAugmenting Japanese dataset...")
    ds_ja_aug = augment_dataset(
        ds_ja, min_metadata=2, max_metadata=5, remove_table_field=True
    )

    # Create dataset dict with splits
    logger.info("\n" + "=" * 80)
    logger.info("Creating dataset dict...")
    logger.info("=" * 80)

    dataset_dict = DatasetDict(
        {
            "vietnamese": ds_vi_aug,
            "english": ds_en_aug,
            "japanese": ds_ja_aug,
        }
    )

    logger.info("\nFinal dataset splits:")
    for split_name, split_ds in dataset_dict.items():
        logger.info(f"  {split_name}: {len(split_ds)} samples")

    # Show example
    logger.info("\n" + "=" * 80)
    logger.info("Example augmented sample (Vietnamese):")
    logger.info("=" * 80)
    example = dataset_dict["vietnamese"][0]
    logger.info(f"Image: {example.get('image')}")
    logger.info(f"Source file: {example.get('source_file')}")
    logger.info(f"Page number: {example.get('page_number')}")

    structured = json.loads(example["text"])
    logger.info("\nMetadata fields:")
    metadata = (
        structured["metadata"][0]
        if isinstance(structured["metadata"], list)
        else structured["metadata"]
    )
    for k, v in metadata.items():
        logger.info(f"  {k}: {v}")

    logger.info("\nTable rows (first 2):")
    for i, row in enumerate(structured["table_rows"][:2]):
        logger.info(f"  Row {i}: {row}")

    logger.info("\nAugmentation info:")
    logger.info(f"  Augmented: {example.get('augmented')}")
    logger.info(f"  Type: {example.get('augmentation_type')}")

    return dataset_dict


def push_to_hub(dataset_dict: DatasetDict, repo_id: str):
    """Push dataset to HuggingFace Hub."""
    logger.info("\n" + "=" * 80)
    logger.info(f"Pushing to HuggingFace Hub: {repo_id}")
    logger.info("=" * 80)

    dataset_dict.push_to_hub(
        repo_id,
        private=False,
        token=os.environ.get("HF_TOKEN"),
    )

    logger.info(f"✅ Successfully pushed to {repo_id}")
    logger.info(f"   View at: https://huggingface.co/datasets/{repo_id}")


def main():
    """Main augmentation pipeline."""
    logger.info("=" * 80)
    logger.info("FIELD AUGMENTATION PIPELINE")
    logger.info("=" * 80)

    # Load and augment
    dataset_dict = load_and_augment_multilingual()

    # Push to hub
    push_to_hub(dataset_dict, OUTPUT_DATASET)

    logger.info("\n" + "=" * 80)
    logger.info("✅ Augmentation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
