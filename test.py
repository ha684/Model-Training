#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để test inference trên 1 ảnh từ dataset
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import base64
import json
import os
from io import BytesIO

import aiohttp
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from config.fields_config_vn import get_extraction_prompt_vi

load_dotenv()

# Config
HF_TOKEN = os.getenv("hf_token")
base_url = os.getenv("base_url", "https://8fad34849342.ngrok-free.app")
VLLM_API_URL = f"{base_url}/v1/chat/completions"
MODEL_NAME = "ha684/qwen3-vl-2b-grpo-V1.1"
DATASET_NAME = "ha684/vl-vi-conv-mix-v1.1"
TIMEOUT = 180


def image_to_base64(image) -> str:
    """Convert PIL Image to base64 string."""
    if isinstance(image, dict):
        if 'bytes' in image:
            return base64.b64encode(image['bytes']).decode('utf-8')
        image = image.get('image', image)
    
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    if isinstance(image, bytes):
        return base64.b64encode(image).decode('utf-8')
    
    raise ValueError(f"Unsupported image type: {type(image)}")


async def call_vllm_api(image, prompt: str) -> str:
    """Call vLLM API với image và prompt."""
    image_base64 = image_to_base64(image)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 1.5,
        "min_p": 0.1,
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('api_key', '')}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                VLLM_API_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT),
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"❌ API call failed: {e}")
            return None


def parse_model_output(output: str) -> dict:
    """Parse model output JSON."""
    if output is None:
        return None

    try:
        cleaned = output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print(f"Output preview: {output[:200]}...")
        return None


async def test_single_image(index: int = 0):
    """Test với 1 ảnh từ dataset."""
    print(f"📦 Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(
        "parquet",
        data_files={"test": f"hf://datasets/{DATASET_NAME}/data/test-*.parquet"},
        split="test"
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} examples")
    
    # Lấy example
    example = dataset[index]
    image = example["image"]
    ground_truth = example["text"]
    source_file = example.get("source_file", f"example_{index}")
    page_number = example.get("page_number", "N/A")
    
    print(f"\n{'='*80}")
    print(f"📄 Testing image #{index}")
    print(f"   Source: {source_file}")
    print(f"   Page: {page_number}")
    print(f"{'='*80}\n")
    
    # Parse ground truth
    try:
        gt_data = json.loads(ground_truth)
        print("📋 Ground Truth:")
        print(json.dumps(gt_data, indent=2, ensure_ascii=False))
    except:
        print("❌ Failed to parse ground truth")
        gt_data = None
    
    # Get prompt và gọi API
    prompt = get_extraction_prompt_vi()

    print(f"\n🚀 Calling API...")
    
    model_output = await call_vllm_api(image, prompt)
    
    if model_output:
        print(f"\n📤 Raw Model Output:")
        print(model_output)
        print(f"\n{'='*80}\n")
        
        # Parse output
        pred_data = parse_model_output(model_output)
        
        if pred_data:
            print("✅ Parsed Model Prediction:")
            print(json.dumps(pred_data, indent=2, ensure_ascii=False))
            
            # So sánh nhanh
            if gt_data:
                print(f"\n{'='*80}")
                print("📊 Quick Comparison:")
                print(f"{'='*80}")
                
                # Metadata
                gt_meta = gt_data.get("metadata", {})
                pred_meta = pred_data.get("metadata", {})
                
                if isinstance(gt_meta, list):
                    gt_meta = gt_meta[0] if gt_meta else {}
                if isinstance(pred_meta, list):
                    pred_meta = pred_meta[0] if pred_meta else {}
                
                print("\n🏷️  Metadata:")
                for field in gt_meta.keys():
                    gt_val = gt_meta.get(field)
                    pred_val = pred_meta.get(field)
                    match = "✅" if str(gt_val).strip().lower() == str(pred_val).strip().lower() else "❌"
                    print(f"  {match} {field}:")
                    print(f"     GT:   {gt_val}")
                    print(f"     Pred: {pred_val}")
                
                # Table rows
                gt_rows = gt_data.get("table_rows", [])
                pred_rows = pred_data.get("table_rows", [])
                
                print(f"\n📊 Table Rows:")
                print(f"  GT rows: {len(gt_rows)}")
                print(f"  Predicted rows: {len(pred_rows)}")
        else:
            print("❌ Failed to parse model output as JSON")
    else:
        print("❌ No response from API")
    
    # Save image để xem
    if isinstance(image, Image.Image):
        output_path = f"test_image_{index}.png"
        image.save(output_path)
        print(f"\n💾 Image saved to: {output_path}")


def main():
    """Main function."""

    
    asyncio.run(test_single_image(0))


if __name__ == "__main__":
    main()