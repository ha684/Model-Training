# pro_doc_aug.py
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from pdf2image import convert_from_path


def build_optimized_ocr_pipeline(seed: int | None = None) -> A.Compose:
    if seed is not None:
        np.random.seed(seed)

    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.02,
                rotate_limit=2,
                border_mode=cv2.BORDER_REPLICATE, 
                p=0.5
            ),

            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.10), mean_range=(0.0, 0.0), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.6,
            ),

            A.OneOf(
                [
                    A.ImageCompression(quality_range=(30, 85), p=1.0),
                    A.Downscale(
                        scale_range=(0.5, 0.9), 
                        interpolation_pair={"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_LINEAR}, 
                        p=1.0
                    ),
                ],
                p=0.8,
            ),

            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.ToGray(p=1.0), 
                ],
                p=0.5,
            ),
            
            A.OneOf(
                [
                    A.Morphological(scale=(2, 3), operation='dilation', p=1.0),
                    A.Morphological(scale=(2, 3), operation='erosion', p=1.0),
                ],
                p=0.3
            ),
        ]
    )

def pil_to_bgr_uint8(pil_img) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_to_png(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=str, required=True, help="Path to input PDF")
    ap.add_argument("--out", type=str, required=True, help="Output folder")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI (e.g. 150/200/300)")
    ap.add_argument("--max_pages", type=int, default=0, help="0 = all pages, else limit")
    ap.add_argument("--num_aug", type=int, default=3, help="How many augmented versions per page")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)
    out_orig = out_dir / "orig"
    out_aug = out_dir / "aug"

    pipeline = build_optimized_ocr_pipeline(seed=args.seed)

    pages = convert_from_path(str(pdf_path), dpi=args.dpi)
    if args.max_pages and args.max_pages > 0:
        pages = pages[: args.max_pages]

    for i, pil_page in enumerate(pages, start=1):
        bgr = pil_to_bgr_uint8(pil_page)

        orig_path = out_orig / f"page_{i:04d}.png"
        bgr_to_png(orig_path, bgr)

        for k in range(1, args.num_aug + 1):
            augmented = pipeline(image=bgr)["image"]
            aug_path = out_aug / f"page_{i:04d}_aug_{k:02d}.png"
            bgr_to_png(aug_path, augmented)

        print(f"Saved page {i}: {orig_path.name} + {args.num_aug} aug(s)")

    print(f"\nDone.\nOriginals: {out_orig}\nAugmented: {out_aug}")


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "0")
    main()
