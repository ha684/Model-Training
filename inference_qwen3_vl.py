#!/usr/bin/env python3
"""
Inference script for fine-tuned Qwen3-VL model
Usage:
    python inference_qwen3_vl.py --model_path lora_model --image_path path/to/image.jpg
    python inference_qwen3_vl.py --model_path lora_model --test_dataset
"""

import argparse
from typing import Any, Tuple

from datasets import load_dataset
from PIL import Image
from unsloth import FastVisionModel

try:
    from config.fields_config import get_extraction_prompt
except ImportError:
    # Inline fallback
    def get_extraction_prompt() -> str:
        return """You are an expert in extracting structured data from delivery notes, cargo documents, and invoices.

Extract the following fields from the provided document image:
  - Số phiếu
  - Ngày chứng từ(mm/dd/yyyy)
  - Ngày xuất(mm/dd/yyyy)
  - Mã FG
  - Đvt
  - Số lượng(Thùng)
  - Số lượng(Pcs)
  - Kho nhận
  - Địa chỉ
  - Mã AR

OUTPUT REQUIREMENTS:
- Return data in JSON format with metadata and table_rows structure
- Set null for missing values
- Be precise and accurate

Analyze and extract data from this document:"""


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Qwen3-VL")

    parser.add_argument(
        "--model_path",
        type=str,
        default="lora_model",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--image_path", type=str, default=None, help="Path to image file"
    )
    parser.add_argument(
        "--test_dataset",
        action="store_true",
        help="Test on validation dataset",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="ha684/senko", help="Dataset name"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to test from dataset",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.5, help="Generation temperature"
    )
    parser.add_argument(
        "--min_p", type=float, default=0.1, help="Min-p sampling parameter"
    )

    return parser.parse_args()


def load_model(model_path: str) -> Tuple[Any, Any]:
    """Load fine-tuned model and tokenizer"""
    print(f"🔧 Loading model from: {model_path}")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_path,
        load_in_4bit=True,
    )

    FastVisionModel.for_inference(model)
    print("✅ Model loaded successfully!")

    return model, tokenizer


def inference_single_image(
    model: Any,
    tokenizer: Any,
    image_path: str,
    instruction: str,
    max_new_tokens: int = 1024,
    temperature: float = 1.5,
    min_p: float = 0.1,
):
    """Run inference on a single image"""
    print(f"\n📸 Processing image: {image_path}")

    # Load image
    image = Image.open(image_path)

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]

    # Tokenize
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        images=image,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
        padding=True,
        max_length=8192,
    ).to("cuda")

    # Generate
    print("🤖 Generating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        min_p=min_p,
    )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    print("\n" + "=" * 70)
    print("📄 EXTRACTED DATA:")
    print("=" * 70)
    print(response)
    print("=" * 70)

    return response


def inference_dataset(
    model: Any,
    tokenizer: Any,
    dataset_name: str,
    num_samples: int,
    instruction: str,
    max_new_tokens: int = 1024,
    temperature: float = 1.5,
    min_p: float = 0.1,
):
    """Run inference on dataset samples"""
    print(f"\n📊 Loading validation dataset: {dataset_name}")

    val_dataset = load_dataset(dataset_name, split="validation")

    # Limit number of samples
    num_samples = min(num_samples, len(val_dataset))
    print(f"📝 Testing on {num_samples} samples")

    for idx in range(num_samples):
        sample = val_dataset[idx]
        image = sample["image"]
        ground_truth = sample["text"]

        print("\n" + "=" * 70)
        print(f"Sample {idx + 1}/{num_samples}")
        print("=" * 70)

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": instruction}],
            }
        ]

        # Tokenize
        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = tokenizer(
            images=image,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
            padding=True,
            max_length=8192,
        ).to("cuda")

        # Generate
        print("🤖 Generating response...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            min_p=min_p,
        )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        print("\n📄 PREDICTED:")
        print("-" * 70)
        print(response)
        print("\n📌 GROUND TRUTH:")
        print("-" * 70)
        print(ground_truth)
        print("=" * 70)

        # Simple accuracy check (optional)
        if response.strip() == ground_truth.strip():
            print("✅ Exact match!")
        else:
            print("⚠️  Different from ground truth")


def main():
    args = parse_args()

    # Get extraction prompt
    instruction = get_extraction_prompt()

    # Load model
    model, tokenizer = load_model(args.model_path)

    if args.test_dataset:
        # Test on validation dataset
        inference_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            num_samples=args.num_samples,
            instruction=instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
        )
    elif args.image_path:
        # Single image inference
        inference_single_image(
            model=model,
            tokenizer=tokenizer,
            image_path=args.image_path,
            instruction=instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
        )
    else:
        print("❌ Please provide either --image_path or --test_dataset")
        return

    print("\n✅ Inference completed!")


if __name__ == "__main__":
    main()

