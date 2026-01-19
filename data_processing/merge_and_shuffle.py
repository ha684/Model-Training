# -*- coding: utf-8 -*-

import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import albumentations as A
import numpy as np
from dotenv import load_dotenv

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
)
from datasets import Image as HFImage
from PIL import Image
import cv2

load_dotenv()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["HF_TOKEN"] = "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"

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION - EDIT THESE AS NEEDED
# ==========================================
# Multilingual datasets
DATASET_VIETNAMESE = "ha684/senko-V1" # Vietnamese
DATASET_ENGLISH = "ha684/senko-V1-english"
DATASET_JAPANESE = "ha684/senko-V1-japanese"

CONVERSATION_DATASET = "ha684/VQA-Gemini3FlashPreview-V1.1"
VINTERN_DATASET = "5CD-AI/Vintern-1B-Benchmark-gpt4o-score"
LEGAL_DATASET = "ha684/legal_cleaned"

OUTPUT_DATASET_NAME = "ha684/vl-vi-en-ja-conv-mix-v1.0"
OUTPUT_DIR = "./merged_multilang_dataset"
PRIVATE = True
TEST_SIZE = 150
SEED = 42

# ==========================================
# GLOBAL FEATURES
# ==========================================
GLOBAL_FEATURES = Features(
    {
        "image": HFImage(),
        "text": Value("string"),
        "source_file": Value("string"),
        "page_number": Value("int32"),
        "is_multiturn": Value("bool"),
        "language": Value("string"),  # NEW: track language
    }
)

# Augmentation pipelines
doc_aug_pipeline_light = A.Compose(
    [
        A.ShiftScaleRotate(
            shift_limit=0.01,
            scale_limit=0.015,
            rotate_limit=1.5,
            border_mode=cv2.BORDER_REPLICATE, 
            p=0.4
        ),
        A.OneOf(
            [
                A.GaussNoise(std_range=(5.0/255, 15.0/255), mean_range=(0, 0), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.ImageCompression(quality_range=(50, 95), p=1.0),
                A.Downscale(
                    scale_range=(0.7, 0.95),
                    interpolation_pair={
                        "downscale": cv2.INTER_LINEAR,
                        "upscale": cv2.INTER_LINEAR
                    }, 
                    p=0.5
                ),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                A.ToGray(p=0.5),
            ],
            p=0.4,
        ),
        A.OneOf(
            [
                A.Morphological(scale=(1, 2), operation='dilation', p=1.0),
                A.Morphological(scale=(1, 2), operation='erosion', p=1.0),
            ],
            p=0.2
        ),
    ]
)

doc_aug_pipeline_minimal = A.Compose(
    [
        A.OneOf(
            [
                A.GaussNoise(std_range=(3.0/255, 8.0/255), mean_range=(0, 0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            ],
            p=0.3,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),
    ]
)


def apply_augment(pil_img, pipeline='light'):
    """Apply albumentations to PIL Image"""
    img_arr = np.array(pil_img)
    
    if pipeline == 'minimal':
        augmented = doc_aug_pipeline_minimal(image=img_arr)["image"]
    elif pipeline == 'light':
        augmented = doc_aug_pipeline_light(image=img_arr)["image"]
    else:
        augmented = img_arr
    
    return Image.fromarray(augmented)


def get_data_type(source_file: str) -> str:
    """Identify data type based on source_file pattern"""
    source_file = str(source_file)
    if "\\" in source_file or "/" in source_file:
        filename = Path(source_file).name
    else:
        filename = source_file

    if filename.endswith(".pdf"):
        base_name = filename[:-4]
    else:
        base_name = filename

    if base_name.startswith("PGH"):
        return "KO"
    elif base_name.startswith("DN"):
        return "SK"
    else:
        return "LT"


def is_valid_text(text_value):
    """Check if text value is valid"""
    if text_value is None:
        return False
    if isinstance(text_value, str):
        return len(text_value.strip()) > 0
    if isinstance(text_value, list):
        return len(text_value) > 0
    return True


def filter_invalid_texts(dataset, dataset_name):
    """Filter out rows with invalid text values"""
    original_size = len(dataset)
    valid_indices = [i for i in range(len(dataset)) if is_valid_text(dataset[i]["text"])]
    filtered_dataset = dataset.select(valid_indices)
    
    removed_count = original_size - len(filtered_dataset)
    if removed_count > 0:
        logger.warning(f"[{dataset_name}] Removed {removed_count} rows with invalid text")
        logger.info(f"[{dataset_name}] Remaining: {len(filtered_dataset)}/{original_size}")
    else:
        logger.info(f"[{dataset_name}] All {original_size} rows have valid text")
    
    return filtered_dataset


# ==========================================
# TRANSFORM FUNCTIONS
# ==========================================

def transform_production_with_aug(batch, language="vi"):
    """
    Transform production data with augmentation
    Input: Batch size N
    Output: Batch size N (augmented once)
    Since we have 3 languages (VI, EN, JA), augmenting each once = x3 total
    """
    out_imgs = []
    out_texts = []
    out_sources = []
    out_pages = []
    out_multiturn = []
    out_languages = []

    page_numbers = batch.get("page_number", [1] * len(batch["text"]))

    for img, txt, src, page in zip(
        batch["image"], 
        batch["text"], 
        batch["source_file"],
        page_numbers
    ):
        if img.mode != "RGB":
            original_img = img.convert("RGB")
        else:
            original_img = img

        # Apply augmentation once
        aug_img = apply_augment(original_img, pipeline='light')
        
        out_imgs.append(aug_img)
        out_texts.append(txt)
        out_sources.append(src)
        out_pages.append(page)
        out_multiturn.append(False)
        out_languages.append(language)

    return {
        "image": out_imgs,
        "text": out_texts,
        "source_file": out_sources,
        "page_number": out_pages,
        "is_multiturn": out_multiturn,
        "language": out_languages,
    }


def transform_production(batch, language="vi"):
    """Transform production data without augmentation"""
    imgs = []
    for img in batch["image"]:
        if img.mode != "RGB":
            imgs.append(img.convert("RGB"))
        else:
            imgs.append(img)

    page_numbers = batch.get("page_number", [1] * len(batch["text"]))

    return {
        "image": imgs,
        "text": batch["text"],
        "source_file": batch["source_file"],
        "page_number": page_numbers,
        "is_multiturn": [False] * len(batch["text"]),
        "language": [language] * len(batch["text"]),
    }


def transform_conversation_with_aug(batch):
    """Apply MINIMAL augmentation to conversation dataset"""
    out_imgs = []
    out_texts = []
    out_sources = []
    out_pages = []
    out_multiturn = []
    out_languages = []

    page_numbers = batch.get("page_number", [1] * len(batch["text"]))

    for img, txt, src, page in zip(
        batch["image"], 
        batch["text"], 
        batch["source_file"], 
        page_numbers
    ):
        if img.mode != "RGB":
            original_img = img.convert("RGB")
        else:
            original_img = img

        final_img = apply_augment(original_img, pipeline='minimal')

        out_imgs.append(final_img)
        out_texts.append(txt)
        out_sources.append(src)
        out_pages.append(page)
        out_multiturn.append(True)
        out_languages.append("vi")  # Conversation is Vietnamese

    return {
        "image": out_imgs,
        "text": out_texts,
        "source_file": out_sources,
        "page_number": out_pages,
        "is_multiturn": out_multiturn,
        "language": out_languages,
    }


def transform_vintern(batch):
    imgs = []
    for img in batch["image"]:
        if img.mode != "RGB":
            imgs.append(img.convert("RGB"))
        else:
            imgs.append(img)

    processed_texts = []
    for res_str in batch["result"]:
        try:
            data = json.loads(res_str) if isinstance(res_str, str) else res_str
            conv = []
            for qa in data:
                conv.append({"role": "user", "content": qa.get("question", "")})
                conv.append({"role": "assistant", "content": qa.get("label", "")})
            processed_texts.append(json.dumps(conv, ensure_ascii=False))
        except:
            processed_texts.append("[]")

    n = len(batch["image"])
    return {
        "image": imgs,
        "text": processed_texts,
        "source_file": [f"vintern_{i}" for i in range(n)],
        "page_number": [1] * n,
        "is_multiturn": [True] * n,
        "language": ["vi"] * n,
    }


def transform_legal(batch):
    imgs = []
    for img in batch["image"]:
        if img.mode != "RGB":
            imgs.append(img.convert("RGB"))
        else:
            imgs.append(img)

    processed_texts = []
    for t in batch["text"]:
        conv = [
            {"role": "user", "content": "Free OCR to extract text from the image"},
            {"role": "assistant", "content": t},
        ]
        processed_texts.append(json.dumps(conv, ensure_ascii=False))

    n = len(batch["image"])
    return {
        "image": imgs,
        "text": processed_texts,
        "source_file": [f"legal_{i}" for i in range(n)],
        "page_number": [1] * n,
        "is_multiturn": [True] * n,
        "language": ["vi"] * n,
    }


# ==========================================
# MAIN LOGIC
# ==========================================

def load_and_process_production_dataset(dataset_name, language_code, test_indices_set=None):
    """
    Load and process a production dataset for a specific language.
    If test_indices_set is provided, use those indices for test split.
    """
    logger.info(f"Loading {language_code.upper()} dataset: {dataset_name}")
    
    ds = load_dataset(dataset_name)
    ds_full = concatenate_datasets([ds["train"], ds["validation"]])
    logger.info(f"  Loaded: {len(ds_full)} samples")
    
    # Filter invalid texts
    ds_full = filter_invalid_texts(ds_full, f"{language_code.upper()}")
    
    # If test_indices_set is provided (for en/ja), use them
    if test_indices_set is not None:
        # Use the same test indices as Vietnamese
        test_indices = list(test_indices_set)
        train_indices = [i for i in range(len(ds_full)) if i not in test_indices_set]
        
        logger.info(f"  Using shared test indices: {len(test_indices)} test, {len(train_indices)} train")
    else:
        # This is Vietnamese - create test split
        logger.info("  Grouping by type (LT, KO, SK)...")
        type_groups = defaultdict(list)
        
        for idx in range(len(ds_full)):
            source_file = ds_full[idx]["source_file"]
            data_type = get_data_type(source_file)
            type_groups[data_type].append(idx)
        
        logger.info("  Data type distribution:")
        for dtype, indices in type_groups.items():
            logger.info(f"    {dtype}: {len(indices)} samples")
        
        # Sample test set proportionally
        test_indices = []
        samples_per_type = TEST_SIZE // 3
        
        collected = {}
        for dtype in ["LT", "KO", "SK"]:
            if dtype in type_groups:
                indices = type_groups[dtype].copy()
                random.shuffle(indices)
                
                n_available = len(indices)
                n_eval = min(samples_per_type, max(1, int(n_available * 0.2)))
                
                collected[dtype] = indices[:n_eval]
                test_indices.extend(collected[dtype])
                logger.info(f"    {dtype}: {n_eval} for eval, {n_available - n_eval} for train")
        
        # Fill shortage from LT
        shortage = TEST_SIZE - len(test_indices)
        if shortage > 0 and "LT" in type_groups:
            logger.info(f"    Short by {shortage}, filling from LT...")
            used_lt = set(collected.get("LT", []))
            available_lt = [idx for idx in type_groups["LT"] if idx not in used_lt]
            random.shuffle(available_lt)
            additional = available_lt[:shortage]
            test_indices.extend(additional)
            logger.info(f"    Added {len(additional)} more from LT")
        
        train_indices = [i for i in range(len(ds_full)) if i not in test_indices]
        logger.info(f"  Final split: {len(test_indices)} test, {len(train_indices)} train")
    
    ds_test = ds_full.select(test_indices)
    ds_train = ds_full.select(train_indices)
    
    # Transform
    logger.info(f"  Transforming {language_code.upper()} data...")
    common_cols = ["image", "text", "source_file", "page_number", "is_multiturn", "language"]
    
    ds_train_transformed = ds_train.map(
        lambda batch: transform_production_with_aug(batch, language=language_code),
        batched=True,
        batch_size=20,
        remove_columns=[c for c in ds_train.column_names if c not in common_cols],
    ).cast(GLOBAL_FEATURES)
    
    ds_test_transformed = ds_test.map(
        lambda batch: transform_production(batch, language=language_code),
        batched=True,
        batch_size=100,
        remove_columns=[c for c in ds_test.column_names if c not in common_cols],
    ).cast(GLOBAL_FEATURES)
    
    logger.info(f"  {language_code.upper()} done: {len(ds_train_transformed)} train (augmented x1), {len(ds_test_transformed)} test")
    
    return ds_train_transformed, ds_test_transformed, set(test_indices)


def main():
    random.seed(SEED)
    output_dir = Path(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 1. LOAD ALL 3 PRODUCTION DATASETS
    # ==========================================
    logger.info("=" * 80)
    logger.info("LOADING MULTILINGUAL PRODUCTION DATA")
    logger.info("=" * 80)
    
    # Load Vietnamese first to get test indices
    ds_vi_train, ds_vi_test, vi_test_indices = load_and_process_production_dataset(
        DATASET_VIETNAMESE, "vi", test_indices_set=None
    )
    
    # Load English using same test indices
    ds_en_train, ds_en_test, _ = load_and_process_production_dataset(
        DATASET_ENGLISH, "en", test_indices_set=vi_test_indices
    )
    
    # Load Japanese using same test indices
    ds_ja_train, ds_ja_test, _ = load_and_process_production_dataset(
        DATASET_JAPANESE, "ja", test_indices_set=vi_test_indices
    )
    
    # Combine all production data
    logger.info("\nCombining multilingual production data...")
    ds_prod_train_all = concatenate_datasets([ds_vi_train, ds_en_train, ds_ja_train])
    ds_prod_test_all = concatenate_datasets([ds_vi_test, ds_en_test, ds_ja_test])
    
    logger.info(f"Combined Production Train: {len(ds_prod_train_all)} (VI: {len(ds_vi_train)}, EN: {len(ds_en_train)}, JA: {len(ds_ja_train)})")
    logger.info(f"Combined Production Test: {len(ds_prod_test_all)} (VI: {len(ds_vi_test)}, EN: {len(ds_en_test)}, JA: {len(ds_ja_test)})")
    logger.info("Note: Each language augmented once = x3 diversity total")

    # ==========================================
    # 2. LOAD CONVERSATION DATA
    # ==========================================
    logger.info("=" * 80)
    logger.info("LOADING CONVERSATION DATA")
    logger.info("=" * 80)
    logger.info(f"Dataset: {CONVERSATION_DATASET}")
    
    ds_conv = load_dataset(CONVERSATION_DATASET, split="train")
    logger.info(f"Loaded: {len(ds_conv)} samples")
    ds_conv = filter_invalid_texts(ds_conv, "Conversation")
    
    logger.info("Transforming conversation data with augmentation...")
    common_cols = ["image", "text", "source_file", "page_number", "is_multiturn", "language"]
    ds_conv_train = ds_conv.map(
        transform_conversation_with_aug,
        batched=True,
        batch_size=20,
        remove_columns=[c for c in ds_conv.column_names if c not in common_cols],
    ).cast(GLOBAL_FEATURES)

    # ==========================================
    # 3. LOAD AUXILIARY DATASETS
    # ==========================================
    logger.info("=" * 80)
    logger.info("LOADING AUXILIARY DATA")
    logger.info("=" * 80)

    # Vintern
    logger.info(f"Loading Vintern: {VINTERN_DATASET}")
    try:
        ds_vintern_1 = load_dataset(VINTERN_DATASET, split="OpenViVQA_dev")
        ds_vintern_2 = load_dataset(VINTERN_DATASET, split="ViTextVQA_dev")
        ds_vintern = concatenate_datasets([ds_vintern_1, ds_vintern_2])
        logger.info(f"Loaded: {len(ds_vintern)} samples")

        ds_vintern = ds_vintern.map(
            transform_vintern,
            batched=True,
            batch_size=100,
            remove_columns=[c for c in ds_vintern.column_names if c not in common_cols],
        ).cast(GLOBAL_FEATURES)
    except Exception as e:
        logger.error(f"Error loading Vintern: {e}")
        ds_vintern = Dataset.from_dict(
            {"image": [], "text": [], "source_file": [], "page_number": [], "is_multiturn": [], "language": []},
            features=GLOBAL_FEATURES,
        )

    # Legal
    logger.info(f"Loading Legal: {LEGAL_DATASET}")
    try:
        ds_legal = load_dataset(LEGAL_DATASET, split="train")
        logger.info(f"Loaded: {len(ds_legal)} samples")

        ds_legal = ds_legal.map(
            transform_legal,
            batched=True,
            batch_size=100,
            remove_columns=[c for c in ds_legal.column_names if c not in common_cols],
        ).cast(GLOBAL_FEATURES)
    except Exception as e:
        logger.error(f"Error loading Legal: {e}")
        ds_legal = Dataset.from_dict(
            {"image": [], "text": [], "source_file": [], "page_number": [], "is_multiturn": [], "language": []},
            features=GLOBAL_FEATURES,
        )

    # ==========================================
    # 4. BALANCE DATASETS
    # ==========================================
    logger.info("=" * 80)
    logger.info("BALANCING DATASETS")
    logger.info("=" * 80)

    n_prod_train = len(ds_prod_train_all)
    n_conv_train = len(ds_conv_train)
    n_vintern = len(ds_vintern)
    n_legal = len(ds_legal)

    logger.info(f"Production count (all languages, with aug): {n_prod_train}")
    logger.info(f"Conversation count (with aug): {n_conv_train}")
    logger.info(f"Vintern count: {n_vintern}")
    logger.info(f"Legal count: {n_legal}")

    # Target: Production + Conversation = 70%, Others = 30%
    base_count = n_prod_train + n_conv_train
    target_total = int(base_count / 0.7)
    target_others_total = target_total - base_count

    logger.info(f"Base count (Production + Conversation): {base_count}")
    logger.info(f"Target Total: ~{target_total}")
    logger.info(f"Target 'Others' Quota (30%): {target_others_total}")

    target_each = target_others_total // 2
    logger.info(f"Target per auxiliary dataset (15% each): {target_each}")

    # Sample Vintern
    take_vintern = min(n_vintern, target_each)
    if take_vintern > 0:
        indices_vintern = list(range(n_vintern))
        random.shuffle(indices_vintern)
        ds_vintern_sampled = ds_vintern.select(indices_vintern[:take_vintern])
        logger.info(f"Sampled Vintern: {take_vintern}/{n_vintern}")
    else:
        ds_vintern_sampled = Dataset.from_dict(
            {"image": [], "text": [], "source_file": [], "page_number": [], "is_multiturn": [], "language": []},
            features=GLOBAL_FEATURES,
        )

    # Sample Legal
    take_legal = min(n_legal, target_each)
    if take_legal > 0:
        indices_legal = list(range(n_legal))
        random.shuffle(indices_legal)
        ds_legal_sampled = ds_legal.select(indices_legal[:take_legal])
        logger.info(f"Sampled Legal: {take_legal}/{n_legal}")
    else:
        ds_legal_sampled = Dataset.from_dict(
            {"image": [], "text": [], "source_file": [], "page_number": [], "is_multiturn": [], "language": []},
            features=GLOBAL_FEATURES,
        )

    # ==========================================
    # 5. COMBINE ALL
    # ==========================================
    logger.info("Combining all datasets...")
    ds_train_final = concatenate_datasets([
        ds_prod_train_all,
        ds_conv_train,
        ds_vintern_sampled,
        ds_legal_sampled,
    ])

    # Calculate percentages
    actual_total = len(ds_train_final)
    prod_pct = (n_prod_train / actual_total) * 100
    conv_pct = (n_conv_train / actual_total) * 100
    vintern_pct = (take_vintern / actual_total) * 100
    legal_pct = (take_legal / actual_total) * 100

    logger.info("=" * 80)
    logger.info("FINAL DISTRIBUTION")
    logger.info("=" * 80)
    logger.info(f"  Production (VI+EN+JA): {n_prod_train} ({prod_pct:.1f}%)")
    logger.info(f"    - Vietnamese: {len(ds_vi_train)}")
    logger.info(f"    - English: {len(ds_en_train)}")
    logger.info(f"    - Japanese: {len(ds_ja_train)}")
    logger.info(f"  Conversation: {n_conv_train} ({conv_pct:.1f}%)")
    logger.info(f"  Vintern: {take_vintern} ({vintern_pct:.1f}%)")
    logger.info(f"  Legal: {take_legal} ({legal_pct:.1f}%)")
    logger.info(f"  Production + Conversation: {n_prod_train + n_conv_train} ({prod_pct + conv_pct:.1f}%)")
    logger.info(f"  Vintern + Legal: {take_vintern + take_legal} ({vintern_pct + legal_pct:.1f}%)")

    # ==========================================
    # 6. SHUFFLE & SAVE
    # ==========================================
    logger.info("=" * 80)
    logger.info("FINALIZING")
    logger.info("=" * 80)
    
    ds_train_final = ds_train_final.shuffle(seed=SEED)
    ds_prod_test_all = ds_prod_test_all.shuffle(seed=SEED)

    final_dict = DatasetDict({"train": ds_train_final, "test": ds_prod_test_all})

    logger.info(f"Final Train Size: {len(ds_train_final)}")
    logger.info(f"Final Test Size: {len(ds_prod_test_all)}")

    logger.info(f"Pushing to Hub: {OUTPUT_DATASET_NAME}")
    final_dict.push_to_hub(OUTPUT_DATASET_NAME, private=PRIVATE)

    logger.info("=" * 80)
    logger.info("DONE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()