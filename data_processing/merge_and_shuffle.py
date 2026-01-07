# -*- coding: utf-8 -*-
"""
Optimized Merge Script: Fixed 'int32 vs int64' mismatch error.
Strategy: 
  1. Production: Upsample 3x.
  2. Others: Take ALL Local Conversations.
  3. Others: Split the remaining quota equally between Vintern and Legal.
"""

import argparse
import hashlib
import json
import logging
import os
import random
from pathlib import Path
import albumentations as A
import numpy as np
# Cấu hình để tối ưu I/O trên Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1" 

from datasets import (
    Dataset, 
    DatasetDict, 
    Features, 
    Value, 
    load_dataset, 
    concatenate_datasets, 
    Image as HFImage
)
from pdf2image import convert_from_path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==========================================
# DEFINITIONS (GLOBAL SCHEMA)
# ==========================================
GLOBAL_FEATURES = Features({
    "image": HFImage(),
    "text": Value("string"),
    "source_file": Value("string"),
    "page_number": Value("int32"),
    "is_multiturn": Value("bool"),
})

# ==========================================
# CÁC HÀM XỬ LÝ ẢNH LOCAL
# ==========================================
doc_aug_pipeline = A.Compose([
    # 1. Giả lập chất lượng ảnh kém (Noise, Blur, Compression)
    A.OneOf([
        A.GaussNoise(std_range=(0.04, 0.20), mean_range=(0.0, 0.0), p=0.5),  # std_range thay vì var_limit
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.ImageCompression(quality_range=(50, 85), p=0.5),  # quality_range thay vì quality_lower/upper
    ], p=0.5),
    
    # 2. Giả lập ánh sáng/màu mực (Sáng tối, Tương phản)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    
    # 3. Giả lập scan lệch (chỉ xoay rất nhẹ +/- 5 độ, fill viền bằng màu trắng 255)
    A.SafeRotate(limit=5, border_mode=0, fill=255, p=0.3),  # fill thay vì value/fill_value
])

def apply_augment(pil_img):
    """Hàm helper để áp dụng albumentations lên PIL Image"""
    # Convert PIL -> Numpy
    img_arr = np.array(pil_img)
    
    # Apply Augmentation
    augmented = doc_aug_pipeline(image=img_arr)["image"]
    
    # Convert back Numpy -> PIL
    return Image.fromarray(augmented)

def extract_page_from_pdf(pdf_path: str, page_number: int, output_dir: str, dpi: int = 200) -> str:
    try:
        hash_input = f"{pdf_path}_{page_number}"
        file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        output_path = os.path.join(output_dir, f"page_{file_hash}.png")

        if os.path.exists(output_path):
            return output_path

        images = convert_from_path(pdf_path, dpi=dpi, first_page=page_number, last_page=page_number)
        if images:
            img = images[0]
            img.save(output_path, format="PNG")
            img.close()
            return output_path
        return None
    except Exception as e:
        logger.error(f"Failed to extract page {page_number} from {pdf_path}: {e}")
        return None

def copy_image_to_cache(source_file: str, image_cache_dir: str) -> str:
    try:
        if os.path.exists(source_file):
            hash_input = f"{source_file}"
            file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
            ext = Path(source_file).suffix or ".png"
            cached_path = os.path.join(image_cache_dir, f"img_{file_hash}{ext}")

            if not os.path.exists(cached_path):
                with Image.open(source_file) as img:
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.save(cached_path, format="PNG")
            return cached_path
    except Exception as e:
        logger.error(f"Failed to process image {source_file}: {e}")
    return None

# ==========================================
# MAPPERS
# ==========================================

def transform_production(batch):
    imgs = []
    for img in batch["image"]:
        if img.mode != "RGB":
            imgs.append(img.convert("RGB"))
        else:
            imgs.append(img)
    
    n = len(batch["text"])
    return {
        "image": imgs,
        "text": batch["text"],
        "source_file": [f"prod_hf_{i}" for i in range(n)], 
        "page_number": [1] * n,
        "is_multiturn": [False] * n
    }

def transform_production_with_aug(batch):
    """
    Input: Batch size N
    Output: Batch size N * 3 (1 Original + 2 Augmented)
    """
    out_imgs = []
    out_texts = []
    out_sources = []
    out_pages = []
    out_multiturn = []

    # Iterate qua từng mẫu trong batch
    for img, txt, src in zip(batch["image"], batch["text"], range(len(batch["image"]))):
        
        # 1. Đảm bảo ảnh gốc là RGB
        if img.mode != "RGB":
            original_img = img.convert("RGB")
        else:
            original_img = img

        # --- COPY 1: ORIGINAL (Giữ nguyên vẹn để mô hình học cái chuẩn) ---
        out_imgs.append(original_img)
        out_texts.append(txt)
        out_sources.append(f"prod_orig_{src}")
        out_pages.append(1)
        out_multiturn.append(False)

        # --- COPY 2: AUGMENTATION 1 (Biến thể 1) ---
        aug_img_1 = apply_augment(original_img)
        out_imgs.append(aug_img_1)
        out_texts.append(txt) # Text giữ nguyên, chỉ đổi ảnh
        out_sources.append(f"prod_aug1_{src}")
        out_pages.append(1)
        out_multiturn.append(False)

        # --- COPY 3: AUGMENTATION 2 (Biến thể 2) ---
        aug_img_2 = apply_augment(original_img)
        out_imgs.append(aug_img_2)
        out_texts.append(txt)
        out_sources.append(f"prod_aug2_{src}")
        out_pages.append(1)
        out_multiturn.append(False)
    
    return {
        "image": out_imgs,
        "text": out_texts,
        "source_file": out_sources,
        "page_number": out_pages,
        "is_multiturn": out_multiturn
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
        "is_multiturn": [True] * n
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
            {"role": "assistant", "content": t}
        ]
        processed_texts.append(json.dumps(conv, ensure_ascii=False))
        
    n = len(batch["image"])
    return {
        "image": imgs,
        "text": processed_texts,
        "source_file": [f"legal_{i}" for i in range(n)],
        "page_number": [1] * n,
        "is_multiturn": [True] * n
    }

# ==========================================
# GENERATOR
# ==========================================

def local_data_generator(conversations_path, image_cache_dir):
    with open(conversations_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    
    logger.info(f"Streaming {len(conversations)} local conversations...")
    
    for idx, conv in enumerate(conversations):
        source_file = conv.get("source_file")
        source_type = conv.get("source_type")
        page_number = conv.get("page_number", 1)
        conversation_data = conv.get("conversation", [])
        
        image_path = None
        if source_type == "pdf":
            image_path = extract_page_from_pdf(source_file, page_number, image_cache_dir)
        elif source_type == "image":
            image_path = copy_image_to_cache(source_file, image_cache_dir)
            
        if image_path:
            text = json.dumps(conversation_data, ensure_ascii=False)
            yield {
                "image": image_path, 
                "text": text,
                "source_file": str(source_file),
                "page_number": page_number,
                "is_multiturn": True
            }

# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--output-dir", default="./merged_dataset")
    parser.add_argument("--production-ratio", type=float, default=0.65)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--image-cache-dir", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    image_cache_dir = args.image_cache_dir or str(output_dir / "images")
    os.makedirs(image_cache_dir, exist_ok=True)

    common_cols = ["image", "text", "source_file", "page_number", "is_multiturn"]

    # 1. LOAD & TRANSFORM PRODUCTION
    logger.info("--- Processing Production Data ---")
    ds_prod = load_dataset("ha684/senko2")
    ds_prod_full = concatenate_datasets([ds_prod["train"], ds_prod["validation"]])
    
    full_indices = list(range(len(ds_prod_full)))
    random.shuffle(full_indices)
    test_indices = full_indices[:args.test_size]
    train_indices = full_indices[args.test_size:]
    
    ds_prod_test = ds_prod_full.select(test_indices)
    ds_prod_train = ds_prod_full.select(train_indices)
    
    logger.info("Mapping and Casting production data...")
    ds_prod_train = ds_prod_train.map(
        transform_production_with_aug,
        batched=True, 
        batch_size=20,
        remove_columns=[c for c in ds_prod_train.column_names if c not in common_cols]
    ).cast(GLOBAL_FEATURES)

    ds_prod_test = ds_prod_test.map(
        transform_production, 
        batched=True, 
        batch_size=100, 
        remove_columns=[c for c in ds_prod_test.column_names if c not in common_cols]
    ).cast(GLOBAL_FEATURES)

    # 2. LOAD & TRANSFORM OTHER DATASETS
    logger.info("--- Processing Auxiliary Data ---")
    
    # 2a. Vintern
    logger.info("Loading Vintern...")
    try:
        ds_vintern_1 = load_dataset("5CD-AI/Vintern-1B-Benchmark-gpt4o-score", split="OpenViVQA_dev")
        ds_vintern_2 = load_dataset("5CD-AI/Vintern-1B-Benchmark-gpt4o-score", split="ViTextVQA_dev")
        ds_vintern = concatenate_datasets([ds_vintern_1, ds_vintern_2])
        
        ds_vintern = ds_vintern.map(
            transform_vintern, 
            batched=True, 
            batch_size=100,
            remove_columns=[c for c in ds_vintern.column_names if c not in common_cols]
        ).cast(GLOBAL_FEATURES)
    except Exception as e:
        logger.error(f"Error loading Vintern: {e}")
        ds_vintern = Dataset.from_dict(
            {"image": [], "text": [], "source_file": [], "page_number": [], "is_multiturn": []}, 
            features=GLOBAL_FEATURES
        )
    
    # 2b. Legal Cleaned
    logger.info("Loading Legal Cleaned...")
    try:
        ds_legal = load_dataset("ha684/legal_cleaned", split="train")
        ds_legal = ds_legal.map(
            transform_legal, 
            batched=True, 
            batch_size=100,
            remove_columns=[c for c in ds_legal.column_names if c not in common_cols]
        ).cast(GLOBAL_FEATURES)
    except Exception as e:
        logger.error(f"Error loading Legal Cleaned: {e}")
        ds_legal = Dataset.from_dict(
            {"image": [], "text": [], "source_file": [], "page_number": [], "is_multiturn": []}, 
            features=GLOBAL_FEATURES
        )

    # 2c. Local Conversations
    logger.info("Processing Local Conversations...")
    features_local_gen = Features({
        "image": Value("string"), 
        "text": Value("string"),
        "source_file": Value("string"),
        "page_number": Value("int32"),
        "is_multiturn": Value("bool"),
    })
    
    ds_local = Dataset.from_generator(
        lambda: local_data_generator(args.conversations, image_cache_dir),
        features=features_local_gen
    )
    ds_local = ds_local.cast(GLOBAL_FEATURES)

    # 3. MERGE & BALANCE (CUSTOM STRATEGY)
    logger.info("--- Balancing Datasets ---")
    
    n_prod_train = len(ds_prod_train)
    n_local = len(ds_local)
    
    logger.info(f"Production count: {n_prod_train}")
    logger.info(f"Local Conversations count: {n_local}")
    
    # B. CALCULATE QUOTA FOR OTHERS
    # n_prod_train / total = 0.65 -> total = n_prod_train / 0.65
    target_total = int(n_prod_train / args.production_ratio)
    target_others_total = target_total - n_prod_train
    
    logger.info(f"Target Total: ~{target_total}")
    logger.info(f"Target 'Others' Quota: {target_others_total}")
    
    # C. FILL QUOTA: PRIORITY LOCAL -> SPLIT REMAINDER (VINTERN + LEGAL)
    if n_local >= target_others_total:
        logger.info("Local conversations fill the entire quota!")
        ds_others_balanced = ds_local.shuffle(seed=args.seed).select(range(target_others_total))
    else:
        logger.info("Taking ALL Local Conversations.")
        remaining_slots = target_others_total - n_local
        
        # Split remaining between Vintern and Legal
        target_each = remaining_slots // 2
        logger.info(f"Remaining slots: {remaining_slots} -> Vintern: {target_each}, Legal: {target_each}")
        
        # Sample Vintern
        n_vintern = len(ds_vintern)
        take_vintern = min(n_vintern, target_each)
        indices_vintern = list(range(n_vintern))
        random.shuffle(indices_vintern)
        ds_vintern_sampled = ds_vintern.select(indices_vintern[:take_vintern])
        
        # Sample Legal
        n_legal = len(ds_legal)
        take_legal = min(n_legal, target_each)
        indices_legal = list(range(n_legal))
        random.shuffle(indices_legal)
        ds_legal_sampled = ds_legal.select(indices_legal[:take_legal])
        
        # Combine Others
        ds_others_balanced = concatenate_datasets([ds_local, ds_vintern_sampled, ds_legal_sampled])

    # D. FINAL MERGE
    ds_train_final = concatenate_datasets([ds_prod_train, ds_others_balanced])

    # 4. SHUFFLE & SAVE
    logger.info("--- Finalizing ---")
    ds_train_final = ds_train_final.shuffle(seed=args.seed)
    
    final_dict = DatasetDict({
        "train": ds_train_final,
        "test": ds_prod_test
    })
    
    logger.info(f"Final Train Size: {len(ds_train_final)}")
    logger.info(f"Final Test Size: {len(ds_prod_test)}")
    
    logger.info(f"Saving to disk at {output_dir}...")
    final_dict.save_to_disk(str(output_dir))
    
    if args.dataset_name:
        logger.info(f"Pushing to Hub: {args.dataset_name}")
        final_dict.push_to_hub(args.dataset_name, private=args.private)
        
    logger.info("Done!")

if __name__ == "__main__":
    main()