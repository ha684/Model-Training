"""
Script để fix và process dataset ha684/senko-V1.2
Xử lý các sample multiturn bị lỗi và push lại lên HuggingFace
"""

import json
from datasets import load_dataset, Dataset, Features, Image as HFImage, Value
from typing import Dict, List, Any
from tqdm import tqdm
import os

def analyze_sample(sample: Dict) -> Dict[str, Any]:
    """Phân tích sample để detect issues"""
    issues = []
    
    multiturn = sample.get("is_multiturn", False)
    text_data = sample.get("text", "")
    
    if multiturn:
        try:
            if isinstance(text_data, str):
                if text_data.strip() in ["[]", ""]:
                    issues.append("empty_conversation")
                else:
                    parsed = json.loads(text_data)
                    if not parsed or len(parsed) == 0:
                        issues.append("empty_conversation")
                    else:
                        # Check if has user role
                        has_user = any(
                            "user" in item or 
                            item.get("role") == "user" or 
                            item.get("from") == "human"
                            for item in parsed
                        )
                        if not has_user:
                            issues.append("no_user_role")
            elif isinstance(text_data, list):
                if len(text_data) == 0:
                    issues.append("empty_conversation")
        except json.JSONDecodeError:
            issues.append("invalid_json")
    
    return {
        "is_multiturn": multiturn,
        "issues": issues,
        "text_length": len(str(text_data))
    }

def fix_sample(sample: Dict) -> Dict:
    """Fix sample bị lỗi"""
    multiturn = sample.get("is_multiturn", False)
    text_data = sample.get("text", "")
    
    if not multiturn:
        return sample  # Single turn không cần fix
    
    # Parse text data
    if isinstance(text_data, str):
        if text_data.strip() in ["[]", ""]:
            conversation = []
        else:
            try:
                conversation = json.loads(text_data)
            except json.JSONDecodeError:
                conversation = []
    elif isinstance(text_data, list):
        conversation = text_data
    else:
        conversation = []
    
    # Nếu conversation rỗng hoặc không có user role, convert sang single turn
    has_user = False
    if conversation:
        has_user = any(
            "user" in item or 
            item.get("role") == "user" or 
            item.get("from") == "human"
            for item in conversation
        )
    
    if not conversation or not has_user:
        # Convert to single turn với default text
        sample_fixed = sample.copy()
        sample_fixed["is_multiturn"] = False
        
        # Nếu có assistant response, dùng nó, không thì dùng placeholder
        assistant_text = ""
        if conversation:
            for item in conversation:
                if "assistant" in item:
                    assistant_text = item["assistant"]
                    break
                elif item.get("role") == "assistant" or item.get("from") == "gpt":
                    assistant_text = item.get("content", item.get("value", ""))
                    break
        
        if not assistant_text:
            assistant_text = "Please analyze this image."
        
        sample_fixed["text"] = assistant_text
        return sample_fixed
    
    return sample

def process_dataset(dataset_name: str = "ha684/senko-V1.2", output_name: str = None, print_bad_samples: bool = True, max_print: int = 10):
    """Process và fix toàn bộ dataset"""
    
    print(f"📥 Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    print(f"\n📊 Dataset info:")
    print(f"  Splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")
    
    # Analyze issues
    print("\n🔍 Analyzing dataset for issues...")
    split_to_process = "train" if "train" in dataset else list(dataset.keys())[0]
    
    issues_count = {
        "empty_conversation": 0,
        "no_user_role": 0,
        "invalid_json": 0,
        "total_multiturn": 0
    }
    
    bad_samples = []  # Store bad samples for printing
    
    for idx, sample in enumerate(tqdm(dataset[split_to_process], desc="Analyzing")):
        analysis = analyze_sample(sample)
        if analysis["is_multiturn"]:
            issues_count["total_multiturn"] += 1
        
        if analysis["issues"]:
            bad_samples.append({
                "index": idx,
                "issues": analysis["issues"],
                "sample": sample
            })
        
        for issue in analysis["issues"]:
            issues_count[issue] = issues_count.get(issue, 0) + 1
    
    print("\n📋 Issues found:")
    print(f"  Total multiturn samples: {issues_count['total_multiturn']}")
    print(f"  Empty conversations: {issues_count['empty_conversation']}")
    print(f"  No user role: {issues_count['no_user_role']}")
    print(f"  Invalid JSON: {issues_count['invalid_json']}")
    
    # Print bad samples
    if print_bad_samples and bad_samples:
        print("\n" + "="*80)
        print(f"🔍 PRINTING BAD SAMPLES (showing first {max_print} of {len(bad_samples)} total)")
        print("="*80)
        
        for i, bad_sample in enumerate(bad_samples[:max_print]):
            print(f"\n{'─'*80}")
            print(f"❌ BAD SAMPLE #{i+1} - Index: {bad_sample['index']}")
            print(f"Issues: {', '.join(bad_sample['issues'])}")
            print(f"{'─'*80}")
            
            sample = bad_sample['sample']
            
            # Print sample info
            print(f"Source File: {sample.get('source_file', 'N/A')}")
            print(f"Page Number: {sample.get('page_number', 'N/A')}")
            print(f"Is Multiturn: {sample.get('is_multiturn', 'N/A')}")
            
            # Print text content
            text_data = sample.get('text', '')
            print(f"\nText Content (length: {len(str(text_data))} chars):")
            print(f"  Raw: {repr(text_data)[:200]}...")
            
            # Try to parse and pretty print
            if isinstance(text_data, str) and text_data.strip() and text_data.strip() != "[]":
                try:
                    parsed = json.loads(text_data)
                    print(f"  Parsed JSON:")
                    print(f"    {json.dumps(parsed, indent=4, ensure_ascii=False)[:500]}...")
                except:
                    print(f"  (Cannot parse as JSON)")
            
            # Print image info
            image = sample.get('image')
            if image:
                if hasattr(image, 'size'):
                    print(f"\nImage: {image.size} pixels, mode: {image.mode}")
                else:
                    print(f"\nImage: {type(image)}")
        
        print("\n" + "="*80)
        if len(bad_samples) > max_print:
            print(f"... and {len(bad_samples) - max_print} more bad samples not shown")
        print("="*80 + "\n")
    
    # Fix dataset
    print("\n🔧 Fixing dataset...")
    fixed_datasets = {}
    
    for split_name, split_data in dataset.items():
        print(f"\n  Processing split: {split_name}")
        fixed_samples = []
        
        for sample in tqdm(split_data, desc=f"Fixing {split_name}"):
            fixed_sample = fix_sample(sample)
            fixed_samples.append(fixed_sample)
        
        # Create new dataset with same features
        fixed_datasets[split_name] = Dataset.from_list(
            fixed_samples,
            features=split_data.features
        )
    
    # Verify fixes
    print("\n✅ Verifying fixes...")
    for split_name, split_data in fixed_datasets.items():
        error_count = 0
        still_bad_samples = []
        
        for idx, sample in enumerate(tqdm(split_data, desc=f"Verifying {split_name}")):
            analysis = analyze_sample(sample)
            if analysis["issues"]:
                error_count += 1
                still_bad_samples.append({
                    "index": idx,
                    "issues": analysis["issues"],
                    "sample": sample
                })
        
        print(f"  {split_name}: {error_count} samples still have issues")
        
        # Print samples that still have issues after fixing
        if still_bad_samples and print_bad_samples:
            print(f"\n  ⚠️  Showing first {min(3, len(still_bad_samples))} samples that still have issues:")
            for i, bad in enumerate(still_bad_samples[:3]):
                print(f"\n    Index {bad['index']}: {bad['issues']}")
                print(f"    Source: {bad['sample'].get('source_file', 'N/A')}")
                print(f"    Text: {repr(bad['sample'].get('text', ''))[:100]}...")
    
    # Push to hub
    if output_name:
        print(f"\n📤 Pushing to HuggingFace Hub: {output_name}")
        for split_name, split_data in fixed_datasets.items():
            split_data.push_to_hub(
                output_name,
                split=split_name,
                private=False
            )
        print("✅ Upload complete!")
    else:
        print("\n💾 Saving locally...")
        for split_name, split_data in fixed_datasets.items():
            output_dir = f"./fixed_senko_{split_name}"
            split_data.save_to_disk(output_dir)
            print(f"  Saved {split_name} to {output_dir}")
    
    return fixed_datasets

def test_conversion(dataset, instruction: str = "Describe this image in detail.", num_samples: int = 5):
    """Test conversion function với fixed dataset"""
    from PIL import Image as PILImage
    
    def convert_to_conversation(sample: Dict, instruction: str) -> Dict:
        multiturn = sample["is_multiturn"]
        image = sample["image"]

        if not multiturn:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "image": image},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["text"]}],
                },
            ]
            return {"messages": conversation}

        text_data = sample["text"]
        raw_data = json.loads(text_data) if isinstance(text_data, str) else text_data

        conversation = []

        for item in raw_data:
            if "user" in item and "assistant" in item:
                conversation.append({"role": "user", "content": item["user"]})
                conversation.append({"role": "assistant", "content": item["assistant"]})
            elif "role" in item or "from" in item:
                role = item.get("role", item.get("from", "")).lower()
                content = item.get("content", item.get("value", ""))

                if role == "human":
                    role = "user"
                if role == "gpt":
                    role = "assistant"

                conversation.append({"role": role, "content": content})

        # Insert image vào user message đầu tiên
        image_added = False
        for message in conversation:
            if message["role"] == "user":
                current_content = message["content"]

                if isinstance(current_content, str):
                    message["content"] = [
                        {"type": "text", "text": current_content},
                        {"type": "image", "image": image},
                    ]
                elif isinstance(current_content, list):
                    if not any(x.get("type") == "image" for x in current_content):
                        message["content"].append({"type": "image", "image": image})

                image_added = True
                break

        if not image_added:
            raise ValueError("No user role found in multiturn conversation")

        return {"messages": conversation}
    
    print(f"\n🧪 Testing conversion with {num_samples} samples...")
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    
    success = 0
    failed = 0
    
    for i in range(min(num_samples, len(dataset[split_name]))):
        sample = dataset[split_name][i]
        try:
            result = convert_to_conversation(sample, instruction)
            success += 1
            print(f"  ✅ Sample {i}: OK")
        except Exception as e:
            failed += 1
            print(f"  ❌ Sample {i}: {e}")
    
    print(f"\n📊 Test results: {success} success, {failed} failed")

if __name__ == "__main__":
    # Sử dụng:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("hf_token")
    # 1. Process và fix dataset - IN RA BAD SAMPLES
    fixed_datasets = process_dataset(
        dataset_name="ha684/senko-V1.2",
        output_name=None,  # Thay bằng username của bạn
        print_bad_samples=True,  # Print ra samples bị lỗi
        max_print=20  # Số lượng bad samples tối đa sẽ in ra (default: 10)
    )
    
    # 2. Test conversion
    test_conversion(fixed_datasets, num_samples=10)
    
    # Nếu chỉ muốn analyze và xem bad samples mà không fix:
    # fixed_datasets = process_dataset(
    #     dataset_name="ha684/senko-V1.2",
    #     output_name=None,  # Không push/save
    #     print_bad_samples=True,
    #     max_print=50  # In nhiều hơn để review
    # )