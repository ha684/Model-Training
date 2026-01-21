# -*- coding: utf-8 -*-

import hashlib
import json
import logging
import multiprocessing
import os
import random
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
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

# ==========================================
# SETUP & CONFIG
# ==========================================
load_dotenv()

# OPTIMIZATION: Use all available cores minus 1 for system stability
NUM_PROC = max(1, multiprocessing.cpu_count() - 1)

# OPTIMIZATION: Disable OpenCV threading to avoid conflicts with multiprocessing
cv2.setNumThreads(0)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
# Ensure your token is correct in your .env or set here
os.environ["HF_TOKEN"] = "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" 

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_VIETNAMESE = "ha684/senko-V1"
DATASET_ENGLISH = "ha684/senko-V1-english"
DATASET_JAPANESE = "ha684/senko-V1-japanese"
DATASET_AUGMENTED = "ha684/senko-V1-augmented-fields"
CONVERSATION_DATASET = "ha684/VQA-Gemini3FlashPreview-V1.1"
VINTERN_DATASET = "5CD-AI/Vintern-1B-Benchmark-gpt4o-score"
LEGAL_DATASET = "ha684/legal_cleaned"

OUTPUT_DATASET_NAME = "ha684/vl-vi-en-ja-conv-mix-v1.3"
OUTPUT_DIR = "./merged_multilang_dataset"
PRIVATE = False
TEST_SIZE_PER_LANG = 100  # 100 images per language = 400 total unique test images
SEED = 42

GLOBAL_FEATURES = Features(
    {
        "image": HFImage(),
        "text": Value("string"),
        "source_file": Value("string"),
        "page_number": Value("int32"),
        "is_multiturn": Value("bool"),
        "language": Value("string"),
    }
)

# ==========================================
# AUGMENTATION PIPELINES
# ==========================================
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

# ==========================================
# HELPER FUNCTIONS
# ==========================================

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


def create_multilingual_test_split(ds_vi, ds_en, ds_ja, ds_aug, test_size_per_lang=100, seed=42):
    """
    Create test split with DIFFERENT images for each language/dataset
    Returns: test_indices_dict and train_indices list
    
    Strategy:
    - VI gets 100 images from LT/KO/SK
    - EN gets 100 DIFFERENT images from LT/KO/SK
    - JA gets 100 DIFFERENT images from LT/KO/SK
    - Augmented gets 100 DIFFERENT images from LT/KO/SK
    - Total: 400 unique test images
    - Train: All remaining images
    """
    random.seed(seed)
    
    logger.info("=" * 80)
    logger.info("CREATING MULTILINGUAL TEST SPLIT")
    logger.info(f"Test size per language: {test_size_per_lang}")
    logger.info("=" * 80)
    
    # Group VI by type (we use VI as reference since all datasets share same images)
    type_groups = defaultdict(list)
    for idx in range(len(ds_vi)):
        source_file = ds_vi[idx]["source_file"]
        data_type = get_data_type(source_file)
        type_groups[data_type].append(idx)
    
    logger.info(f"Type distribution in dataset:")
    for dtype in ["LT", "KO", "SK"]:
        logger.info(f"  {dtype}: {len(type_groups.get(dtype, []))} images")
    
    # Calculate samples per type for each language (evenly distributed)
    samples_per_type = test_size_per_lang // 3
    
    all_test_indices = set()
    test_indices_per_lang = {
        "vi": [],
        "en": [],
        "ja": [],
        "aug": []
    }
    
    # For each language, select different images from each type
    for lang in ["vi", "en", "ja", "aug"]:
        logger.info(f"\nSelecting test indices for {lang.upper()}...")
        lang_test_indices = []
        collected = {}
        
        for dtype in ["LT", "KO", "SK"]:
            if dtype in type_groups:
                # Get indices NOT yet used in any test set
                available_indices = [idx for idx in type_groups[dtype] if idx not in all_test_indices]
                random.shuffle(available_indices)
                
                n_available = len(available_indices)
                n_eval = min(samples_per_type, n_available)
                
                if n_eval == 0:
                    logger.warning(f"  {dtype}: No available indices for {lang}")
                    collected[dtype] = []
                else:
                    collected[dtype] = available_indices[:n_eval]
                    lang_test_indices.extend(collected[dtype])
                    logger.info(f"  {dtype}: selected {n_eval} images (available: {n_available})")
        
        # Fill shortage from LT if needed
        shortage = test_size_per_lang - len(lang_test_indices)
        if shortage > 0 and "LT" in type_groups:
            used_lt = set(collected.get("LT", []))
            available_lt = [idx for idx in type_groups["LT"] 
                           if idx not in used_lt and idx not in all_test_indices]
            random.shuffle(available_lt)
            additional = available_lt[:shortage]
            lang_test_indices.extend(additional)
            logger.info(f"  Filled {len(additional)} additional from LT to reach {test_size_per_lang}")
        
        test_indices_per_lang[lang] = lang_test_indices
        all_test_indices.update(lang_test_indices)
        logger.info(f"  ✓ {lang.upper()} test set: {len(lang_test_indices)} images")
    
    # Create train indices (all indices NOT in test)
    total_size = len(ds_vi)
    all_train_indices = [i for i in range(total_size) if i not in all_test_indices]
    
    logger.info("=" * 80)
    logger.info("TEST/TRAIN SPLIT SUMMARY")
    logger.info(f"  Total images in dataset: {total_size}")
    logger.info(f"  Total UNIQUE test images: {len(all_test_indices)}")
    logger.info(f"  Total train images: {len(all_train_indices)}")
    logger.info(f"  ")
    logger.info(f"  Test breakdown:")
    for lang, indices in test_indices_per_lang.items():
        logger.info(f"    {lang.upper()}: {len(indices)} images")
    logger.info(f"  ")
    logger.info(f"  ✓ No overlap: Test images are EXCLUDED from train set")
    logger.info("=" * 80)
    
    return test_indices_per_lang, all_train_indices


# ==========================================
# TRANSFORM FUNCTIONS
# ==========================================

def transform_production_with_aug(batch, language="vi"):
    """Transform production data with augmentation"""
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


def transform_augmented_production(batch, language="vi"):
    """
    Transform augmented dataset (already has field removal)
    Apply image augmentation only
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

        # Apply image augmentation
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
        out_languages.append("vi")

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
# MAIN LOADING LOGIC
# ==========================================

def main():
    random.seed(SEED)
    output_dir = Path(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Detected {multiprocessing.cpu_count()} CPUs, using {NUM_PROC} workers.")

    # ==========================================
    # 1. LOAD ALL RAW DATASETS
    # ==========================================
    logger.info("=" * 80)
    logger.info("STEP 1: LOADING RAW DATASETS")
    logger.info("=" * 80)
    
    # Load VI
    logger.info(f"Loading Vietnamese: {DATASET_VIETNAMESE}")
    ds_vi_raw = load_dataset(DATASET_VIETNAMESE)
    ds_vi_full = concatenate_datasets([ds_vi_raw["train"], ds_vi_raw["validation"]])
    ds_vi_full = filter_invalid_texts(ds_vi_full, "Vietnamese")
    
    # Load EN
    logger.info(f"Loading English: {DATASET_ENGLISH}")
    ds_en_raw = load_dataset(DATASET_ENGLISH)
    ds_en_full = concatenate_datasets([ds_en_raw["train"], ds_en_raw["validation"]])
    ds_en_full = filter_invalid_texts(ds_en_full, "English")
    
    # Load JA
    logger.info(f"Loading Japanese: {DATASET_JAPANESE}")
    ds_ja_raw = load_dataset(DATASET_JAPANESE)
    ds_ja_full = concatenate_datasets([ds_ja_raw["train"], ds_ja_raw["validation"]])
    ds_ja_full = filter_invalid_texts(ds_ja_full, "Japanese")
    
    # Load Augmented (all 3 languages)
    logger.info(f"Loading Augmented: {DATASET_AUGMENTED}")
    ds_aug_raw = load_dataset(DATASET_AUGMENTED)
    ds_aug_vi_full = filter_invalid_texts(ds_aug_raw["vietnamese"], "Augmented-VI")
    ds_aug_en_full = filter_invalid_texts(ds_aug_raw["english"], "Augmented-EN")
    ds_aug_ja_full = filter_invalid_texts(ds_aug_raw["japanese"], "Augmented-JA")
    
    # ==========================================
    # 2. CREATE TEST/TRAIN SPLIT
    # ==========================================
    test_indices_per_lang, train_indices = create_multilingual_test_split(
        ds_vi_full, ds_en_full, ds_ja_full, ds_aug_vi_full,
        test_size_per_lang=TEST_SIZE_PER_LANG,
        seed=SEED
    )
    
    # ==========================================
    # 3. SPLIT AND TRANSFORM EACH DATASET
    # ==========================================
    logger.info("=" * 80)
    logger.info("STEP 2: TRANSFORMING PRODUCTION DATASETS")
    logger.info("=" * 80)
    
    common_cols = ["image", "text", "source_file", "page_number", "is_multiturn", "language"]
    
    # ===== VIETNAMESE =====
    logger.info("\n[1/4] Processing Vietnamese...")
    ds_vi_test = ds_vi_full.select(test_indices_per_lang["vi"])
    ds_vi_train = ds_vi_full.select(train_indices)
    
    logger.info(f"  Test: {len(ds_vi_test)} samples (no augmentation)")
    ds_vi_test_transformed = ds_vi_test.map(
        lambda batch: transform_production(batch, language="vi"),
        batched=True,
        batch_size=100,
        remove_columns=[c for c in ds_vi_test.column_names if c not in common_cols],
        num_proc=NUM_PROC,
        features=GLOBAL_FEATURES,
    )
    
    logger.info(f"  Train: {len(ds_vi_train)} samples (with augmentation)")
    ds_vi_train_transformed = ds_vi_train.map(
        lambda batch: transform_production_with_aug(batch, language="vi"),
        batched=True,
        batch_size=50,
        remove_columns=[c for c in ds_vi_train.column_names if c not in common_cols],
        num_proc=NUM_PROC,
        features=GLOBAL_FEATURES,
    )
    logger.info(f"  ✓ Vietnamese done")
    
    # ===== ENGLISH =====
    logger.info("\n[2/4] Processing English...")
    ds_en_test = ds_en_full.select(test_indices_per_lang["en"])
    ds_en_train = ds_en_full.select(train_indices)
    
    logger.info(f"  Test: {len(ds_en_test)} samples (no augmentation)")
    ds_en_test_transformed = ds_en_test.map(
        lambda batch: transform_production(batch, language="en"),
        batched=True,
        batch_size=100,
        remove_columns=[c for c in ds_en_test.column_names if c not in common_cols],
        num_proc=NUM_PROC,
        features=GLOBAL_FEATURES,
    )
    
    logger.info(f"  Train: {len(ds_en_train)} samples (with augmentation)")
    ds_en_train_transformed = ds_en_train.map(
        lambda batch: transform_production_with_aug(batch, language="en"),
        batched=True,
        batch_size=50,
        remove_columns=[c for c in ds_en_train.column_names if c not in common_cols],
        num_proc=NUM_PROC,
        features=GLOBAL_FEATURES,
    )
    logger.info(f"  ✓ English done")
    
    # ===== JAPANESE =====
    logger.info("\n[3/4] Processing Japanese...")
    ds_ja_test = ds_ja_full.select(test_indices_per_lang["ja"])
    ds_ja_train = ds_ja_full.select(train_indices)
    
    logger.info(f"  Test: {len(ds_ja_test)} samples (no augmentation)")
    ds_ja_test_transformed = ds_ja_test.map(
        lambda batch: transform_production(batch, language="ja"),
        batched=True,
        batch_size=100,
        remove_columns=[c for c in ds_ja_test.column_names if c not in common_cols],
        num_proc=NUM_PROC,
        features=GLOBAL_FEATURES,
    )
    
    logger.info(f"  Train: {len(ds_ja_train)} samples (with augmentation)")
    ds_ja_train_transformed = ds_ja_train.map(
        lambda batch: transform_production_with_aug(batch, language="ja"),
        batched=True,
        batch_size=50,
        remove_columns=[c for c in ds_ja_train.column_names if c not in common_cols],
        num_proc=NUM_PROC,
        features=GLOBAL_FEATURES,
    )
    logger.info(f"  ✓ Japanese done")
    
    # ===== AUGMENTED (VI) =====
    logger.info("\n[4/4] Processing Augmented Vietnamese...")
    ds_aug_vi_test = ds_aug_vi_full.select(test_indices_per_lang["aug"])
    ds_aug_vi_train = ds_aug_vi_full.select(train_indices)
    
    logger.info(f"  Test: {len(ds_aug_vi_test)} samples (with image augmentation)")
    ds_aug_vi_test_transformed = ds_aug_vi_test.map(
        lambda batch: transform_augmented_production(batch, language="vi"),
        batched=True,
        batch_size=100,
        remove_columns=[c for c in ds_aug_vi_test.column_names if c not in common_cols],
        num_proc=NUM_PROC,
        features=GLOBAL_FEATURES,
    )
    
    logger.info(f"  Train: {len(ds_aug_vi_train)} samples (with image augmentation)")
    ds_aug_vi_train_transformed = ds_aug_vi_train.map(
        lambda batch: transform_augmented_production(batch, language="vi"),
        batched=True,
        batch_size=50,
        remove_columns=[c for c in ds_aug_vi_train.column_names if c not in common_cols],
        num_proc=NUM_PROC,
        features=GLOBAL_FEATURES,
    )
    logger.info(f"  ✓ Augmented Vietnamese done")
    
    # ===== AUGMENTED (EN & JA) - TRAIN ONLY =====
    logger.info("\nProcessing Augmented English (train only)...")
    ds_aug_en_train = ds_aug_en_full.select(train_indices)
    ds_aug_en_train_transformed = ds_aug_en_train.map(
        lambda batch: transform_augmented_production(batch, language="en"),
        batched=True,
        batch_size=50,
        remove_columns=[c for c in ds_aug_en_train.column_names if c not in common_cols],
        num_proc=NUM_PROC,
        features=GLOBAL_FEATURES,
    )
    logger.info(f"  ✓ Augmented English: {len(ds_aug_en_train_transformed)} train samples")
    
    logger.info("\nProcessing Augmented Japanese (train only)...")
    ds_aug_ja_train = ds_aug_ja_full.select(train_indices)
    ds_aug_ja_train_transformed = ds_aug_ja_train.map(
        lambda batch: transform_augmented_production(batch, language="ja"),
        batched=True,
        batch_size=50,
        remove_columns=[c for c in ds_aug_ja_train.column_names if c not in common_cols],
        num_proc=NUM_PROC,
        features=GLOBAL_FEATURES,
    )
    logger.info(f"  ✓ Augmented Japanese: {len(ds_aug_ja_train_transformed)} train samples")
    
    # ==========================================
    # 4. COMBINE PRODUCTION TRAIN AND TEST
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: COMBINING PRODUCTION DATASETS")
    logger.info("=" * 80)
    
    ds_test_all = concatenate_datasets([
        ds_vi_test_transformed,
        ds_en_test_transformed,
        ds_ja_test_transformed,
        ds_aug_vi_test_transformed,
    ])
    logger.info(f"✓ Combined test set: {len(ds_test_all)} samples (400 unique images)")
    
    ds_train_production = concatenate_datasets([
        ds_vi_train_transformed,
        ds_en_train_transformed,
        ds_ja_train_transformed,
        ds_aug_vi_train_transformed,
        ds_aug_en_train_transformed,
        ds_aug_ja_train_transformed,
    ])
    logger.info(f"✓ Combined production train: {len(ds_train_production)} samples")
    
    # ==========================================
    # 5. LOAD CONVERSATION DATA
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: LOADING CONVERSATION DATA")
    logger.info("=" * 80)
    
    ds_conv = load_dataset(CONVERSATION_DATASET, split="train")
    ds_conv = filter_invalid_texts(ds_conv, "Conversation")
    
    logger.info(f"Transforming conversation data (CPUs: {NUM_PROC})...")
    ds_conv_train = ds_conv.map(
        transform_conversation_with_aug,
        batched=True,
        batch_size=50,
        remove_columns=[c for c in ds_conv.column_names if c not in common_cols],
        num_proc=NUM_PROC,
        features=GLOBAL_FEATURES,
    )
    logger.info(f"✓ Conversation dataset: {len(ds_conv_train)} samples")

    # ==========================================
    # 6. LOAD AUXILIARY DATASETS
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: LOADING AUXILIARY DATA")
    logger.info("=" * 80)
    # Vintern
    logger.info(f"Loading Vintern: {VINTERN_DATASET}")
    try:
        ds_vintern_1 = load_dataset(VINTERN_DATASET, split="OpenViVQA_dev")
        ds_vintern_2 = load_dataset(VINTERN_DATASET, split="ViTextVQA_dev")
        ds_vintern = concatenate_datasets([ds_vintern_1, ds_vintern_2])
        
        ds_vintern = ds_vintern.map(
            transform_vintern,
            batched=True,
            batch_size=100,
            remove_columns=[c for c in ds_vintern.column_names if c not in common_cols],
            num_proc=NUM_PROC,
            features=GLOBAL_FEATURES,
        )
        logger.info(f"✓ Vintern dataset: {len(ds_vintern)} samples")
    except Exception as e:
        logger.error(f"Error loading Vintern: {e}")
        ds_vintern = Dataset.from_dict(
            {"image": [], "text": [], "source_file": [], "page_number": [], "is_multiturn": [], "language": []},
            features=GLOBAL_FEATURES,
        )
        logger.warning("Using empty Vintern dataset")

    # Legal
    logger.info(f"Loading Legal: {LEGAL_DATASET}")
    try:
        ds_legal = load_dataset(LEGAL_DATASET, split="train")
        
        ds_legal = ds_legal.map(
            transform_legal,
            batched=True,
            batch_size=100,
            remove_columns=[c for c in ds_legal.column_names if c not in common_cols],
            num_proc=NUM_PROC,
            features=GLOBAL_FEATURES,
        )
        logger.info(f"✓ Legal dataset: {len(ds_legal)} samples")
    except Exception as e:
        logger.error(f"Error loading Legal: {e}")
        ds_legal = Dataset.from_dict(
            {"image": [], "text": [], "source_file": [], "page_number": [], "is_multiturn": [], "language": []},
            features=GLOBAL_FEATURES,
        )
        logger.warning("Using empty Legal dataset")

    # ==========================================
    # 7. BALANCE DATASETS
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: BALANCING DATASETS")
    logger.info("=" * 80)

    n_prod_train = len(ds_train_production)
    n_conv_train = len(ds_conv_train)
    n_vintern = len(ds_vintern)
    n_legal = len(ds_legal)

    logger.info(f"Current distribution:")
    logger.info(f"  Production (VI+EN+JA+Aug): {n_prod_train}")
    logger.info(f"  Conversation: {n_conv_train}")
    logger.info(f"  Vintern (available): {n_vintern}")
    logger.info(f"  Legal (available): {n_legal}")

    base_count = n_prod_train + n_conv_train
    target_total = int(base_count / 0.8)
    target_others_total = target_total - base_count
    target_each = target_others_total // 2

    logger.info(f"\nBalancing strategy:")
    logger.info(f"  Base count (Prod + Conv): {base_count}")
    logger.info(f"  Target total (80% of final): {target_total}")
    logger.info(f"  Target for others (Vintern + Legal): {target_others_total}")
    logger.info(f"  Target each: {target_each}")

    # Sample Vintern
    take_vintern = min(n_vintern, target_each)
    if take_vintern > 0 and n_vintern > 0:
        indices_vintern = list(range(n_vintern))
        random.shuffle(indices_vintern)
        ds_vintern_sampled = ds_vintern.select(indices_vintern[:take_vintern])
        logger.info(f"  ✓ Sampled Vintern: {take_vintern}/{n_vintern}")
    else:
        ds_vintern_sampled = ds_vintern.select(range(0))
        logger.info(f"  ✓ Vintern: 0 samples")

    # Sample Legal
    take_legal = min(n_legal, target_each)
    if take_legal > 0 and n_legal > 0:
        indices_legal = list(range(n_legal))
        random.shuffle(indices_legal)
        ds_legal_sampled = ds_legal.select(indices_legal[:take_legal])
        logger.info(f"  ✓ Sampled Legal: {take_legal}/{n_legal}")
    else:
        ds_legal_sampled = ds_legal.select(range(0))
        logger.info(f"  ✓ Legal: 0 samples")

    # ==========================================
    # 8. COMBINE ALL & SAVE
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: FINAL COMBINATION & UPLOAD")
    logger.info("=" * 80)

    logger.info("Combining all training datasets...")
    ds_train_final = concatenate_datasets([
        ds_train_production,
        ds_conv_train,
        ds_vintern_sampled,
        ds_legal_sampled,
    ])

    logger.info("\n" + "=" * 80)
    logger.info("FINAL DATASET SUMMARY")
    logger.info("=" * 80)
    logger.info(f"TRAIN SET: {len(ds_train_final)} samples")
    logger.info(f"  - Production (VI+EN+JA+Aug): {len(ds_train_production)}")
    logger.info(f"  - Conversation: {len(ds_conv_train)}")
    logger.info(f"  - Vintern: {len(ds_vintern_sampled)}")
    logger.info(f"  - Legal: {len(ds_legal_sampled)}")
    logger.info(f"")
    logger.info(f"TEST SET: {len(ds_test_all)} samples (400 UNIQUE images)")
    logger.info(f"  - VI: {len(ds_vi_test_transformed)}")
    logger.info(f"  - EN: {len(ds_en_test_transformed)}")
    logger.info(f"  - JA: {len(ds_ja_test_transformed)}")
    logger.info(f"  - Augmented: {len(ds_aug_vi_test_transformed)}")
    logger.info("=" * 80)

    logger.info("\nShuffling datasets...")
    ds_train_final = ds_train_final.shuffle(seed=SEED)
    ds_test_all = ds_test_all.shuffle(seed=SEED)

    final_dict = DatasetDict({"train": ds_train_final, "test": ds_test_all})

    logger.info(f"\nPushing to Hugging Face Hub: {OUTPUT_DATASET_NAME}")
    logger.info(f"Private: {PRIVATE}")
    final_dict.push_to_hub(OUTPUT_DATASET_NAME, private=PRIVATE)

    logger.info("\n" + "=" * 80)
    logger.info("✓ PROCESS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()