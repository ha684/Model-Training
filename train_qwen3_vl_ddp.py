#!/usr/bin/env python3
"""
Fine-tune Qwen3-VL Vision Language Model with Multi-GPU support using OpenSloth + Unsloth
Usage:
    # Single GPU
    python train_qwen3_vl_ddp.py

    # Multi-GPU (2 GPUs)
    torchrun --nproc_per_node=2 train_qwen3_vl_ddp.py

    # Multi-GPU (4 GPUs) with custom settings
    torchrun --nproc_per_node=4 train_qwen3_vl_ddp.py --batch_size 1 --epochs 3
"""

import argparse
import os
from typing import Any, Dict, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

os.environ["HF_TOKEN"] = "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"

# Import opensloth DDP patches if available
try:
    from closed_sloth.patching.ddp_patch import (
        ddp_patch,
        patch_optimize_sft_trainer_batch_samples,
    )

    OPENSLOTH_AVAILABLE = False
except ImportError:
    OPENSLOTH_AVAILABLE = False
    print("⚠️  OpenSloth not available. Multi-GPU support will be limited.")

# Import config if available, otherwise use inline config
try:
    from config.fields_config import get_extraction_prompt
except ImportError:
    print("⚠️  Config not available. Using inline config.")
    DEFAULT_FIELDS = [
        "Số phiếu",
        "Ngày chứng từ(mm/dd/yyyy)",
        "Ngày xuất(mm/dd/yyyy)",
        "Mã FG",
        "Đvt",
        "Số lượng(Thùng)",
        "Số lượng(Pcs)",
        "Kho nhận",
        "Địa chỉ",
        "Mã AR",
    ]

    METADATA_FIELDS = [
        "Số phiếu",
        "Ngày chứng từ(mm/dd/yyyy)",
        "Ngày xuất(mm/dd/yyyy)",
        "Kho nhận",
        "Địa chỉ",
        "Mã AR",
    ]

    TABLE_ROW_FIELDS = [f for f in DEFAULT_FIELDS if f not in METADATA_FIELDS]

    def get_extraction_prompt() -> str:
        fields_list = "\n".join([f"  - {field}" for field in DEFAULT_FIELDS])
        metadata_fields_list = ", ".join([f'"{field}"' for field in METADATA_FIELDS])
        table_row_fields_list = ", ".join([f'"{field}"' for field in TABLE_ROW_FIELDS])

        prompt = f"""You are an expert in extracting structured data from delivery notes, cargo documents, and invoices.

Extract the following fields from the provided document image:

{fields_list}

OUTPUT REQUIREMENTS:
- Return data in JSON format with the following structure:
{{
  "metadata": {{
    {", ".join([f'"{f}": <value or null>' for f in METADATA_FIELDS])}
  }},
  "table_rows": [
    {{
      {", ".join([f'"{f}": <value or null>' for f in TABLE_ROW_FIELDS])}
    }}
  ]
}}

EXTRACTION RULES:
1. "metadata" contains document-level information: {metadata_fields_list}
2. "table_rows" contains table row data: {table_row_fields_list}
3. If there are multiple rows in the table, each row should be a separate object in the "table_rows" array
4. If a field value cannot be found or does not exist in the document, set it to null
5. For date fields, convert to mm/dd/yyyy format if possible. If the date format is unclear or cannot be determined, set to null
6. For numeric fields (quantities), extract only the number. If not found, set to null
7. Be extremely precise - only include values that are explicitly present in the document
8. Do not infer, guess, or make assumptions about missing data
9. Return ONLY valid JSON with no additional text, explanations, or markdown formatting
10. Ensure the JSON is properly formatted and can be parsed
11. Pay special attention to Vietnamese text fields - preserve exact spelling and diacritics
12. For code fields (Mã FG, Mã AR), extract the exact alphanumeric code as shown

IMPORTANT:
- Accuracy is critical. If you're unsure about a value, set it to null rather than guessing
- Double-check all extracted values against the source document
- Maintain data type consistency (strings for text, numbers for quantities)

Analyze and extract data from this document:"""
        return prompt


def get_gpu_precision_settings() -> Dict[str, bool]:
    """
    Detect GPU capability and return appropriate precision settings.

    Returns:
        Dict with 'bf16' and 'fp16' settings based on GPU capability
    """
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}

    # Check if GPU supports bf16 (Ampere or newer, compute capability >= 8.0)
    gpu_capability = torch.cuda.get_device_capability()
    major, minor = gpu_capability

    # Ampere (A100, RTX 30xx) and newer support bf16
    supports_bf16 = major >= 8

    if supports_bf16:
        return {"bf16": True, "fp16": False}
    else:
        # Use fp16 for older GPUs (Pascal, Volta, Turing)
        return {"bf16": False, "fp16": True}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL with multi-GPU support"
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=8192, help="Maximum sequence length"
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ha684/senko1",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HuggingFace API token"
    )

    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha", type=int, default=128, help="LoRA alpha parameter"
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")

    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per device train batch size",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2,
        help="Per device eval batch size",
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps")
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=10, help="Evaluation interval"
    )
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging interval")

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen3_vl_lora",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the model after training",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="lora_model",
        help="Path to save the fine-tuned model",
    )

    # Other configurations
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "tensorboard", "wandb"],
        help="Reporting tool",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument(
        "--no_patches",
        action="store_true",
        help="Disable OpenSloth DDP patches",
    )

    return parser.parse_args()


def init_model(args: argparse.Namespace, local_rank: int) -> Tuple[Any, Any]:
    """
    Initialize Qwen3-VL model and tokenizer with proper device mapping for multi-GPU.

    Args:
        args: Command line arguments
        local_rank: Local rank for DDP (0 for single GPU)

    Returns:
        Tuple of (model, tokenizer)
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Device mapping: use specific GPU for multi-GPU, auto for single GPU
    if world_size > 1:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    print(f"🔧 Loading model: {args.model_name}")
    print(f"   Device map: {device_map}")

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        device_map=device_map,
    )

    # Fix tokenizer EOS token for SFTTrainer compatibility
    # For Qwen3VLProcessor, the actual tokenizer is nested inside
    print(f"   Tokenizer type: {type(tokenizer).__name__}")

    # Get the actual tokenizer if this is a processor
    actual_tokenizer = tokenizer
    if hasattr(tokenizer, "tokenizer"):
        actual_tokenizer = tokenizer.tokenizer
        print(f"   Found nested tokenizer: {type(actual_tokenizer).__name__}")

    # Check current EOS token
    current_eos = getattr(actual_tokenizer, "eos_token", None)
    print(f"   Current EOS token: {current_eos}")

    # Fix if EOS token is invalid
    if not current_eos or current_eos == "<EOS_TOKEN>":
        print("   Fixing EOS token...")

        # Get vocabulary
        try:
            vocab = actual_tokenizer.get_vocab()

            # Try common Qwen EOS tokens
            for candidate in [
                "<|im_end|>",
                "<|endoftext|>",
                "</s>",
                "<|end|>",
                "<eos>",
            ]:
                if candidate in vocab:
                    actual_tokenizer.eos_token = candidate
                    # Also set eos_token_id
                    actual_tokenizer.eos_token_id = vocab[candidate]
                    print(
                        f"   ✓ Set EOS token to: {candidate} (id: {vocab[candidate]})"
                    )
                    break

            # If we set it on the nested tokenizer, also set on the processor
            if hasattr(tokenizer, "tokenizer") and actual_tokenizer.eos_token:
                tokenizer.eos_token = actual_tokenizer.eos_token

        except Exception as e:
            print(f"   ⚠️  Failed to fix EOS token: {e}")
    else:
        print(f"   ✓ EOS token is valid: {current_eos}")

    print("🔧 Configuring LoRA...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Don't finetune vision encoder
        finetune_language_layers=True,  # Finetune language model
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def convert_to_conversation(sample: Dict, instruction: str) -> Dict:
    """
    Convert a dataset sample to conversation format for vision-language training.

    Args:
        sample: Dataset sample with 'image' and 'text' fields
        instruction: System instruction/prompt

    Returns:
        Dictionary with 'messages' field in conversation format
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["text"]}],
        },
    ]
    return {"messages": conversation}


class LazyVisionDataset(TorchDataset):
    """
    Lazy loading dataset wrapper that converts samples to conversation format on-demand.

    This avoids loading all images into memory at once, preventing OOM errors
    with large image datasets.
    """

    def __init__(self, hf_dataset, instruction: str):
        """
        Args:
            hf_dataset: HuggingFace dataset with 'image' and 'text' columns
            instruction: System instruction/prompt for extraction
        """
        self.dataset = hf_dataset
        self.instruction = instruction

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load only this sample's image on-demand
        sample = self.dataset[idx]
        return convert_to_conversation(sample, self.instruction)


def prepare_datasets(args: argparse.Namespace, local_rank: int) -> Tuple[Any, Any]:
    """
    Load and prepare datasets from HuggingFace with lazy loading.

    Uses LazyVisionDataset to load images on-demand during training,
    avoiding OOM errors from loading all images into memory at once.

    Args:
        args: Command line arguments
        local_rank: Local rank for DDP

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Only print on rank 0 to avoid duplicate logs
    if local_rank == 0:
        print(f"📊 Loading dataset: {args.dataset_name}")

    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Load datasets (lazy - images stored in Arrow format, not decoded yet)
    train_dataset = load_dataset(args.dataset_name, split="train")
    eval_dataset = load_dataset(args.dataset_name, split="validation")

    if local_rank == 0:
        print(f"✅ Found {len(train_dataset)} training samples")
        print(f"✅ Found {len(eval_dataset)} validation samples")

    # Get extraction prompt
    instruction = get_extraction_prompt()

    # Wrap with lazy loading - images loaded one at a time during training
    # This prevents OOM by not loading all images into memory at once
    lazy_train = LazyVisionDataset(train_dataset, instruction)
    lazy_eval = LazyVisionDataset(eval_dataset, instruction)

    if local_rank == 0:
        print("✅ Datasets wrapped with lazy loading (images loaded on-demand)")

    return lazy_train, lazy_eval


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    args: argparse.Namespace,
    world_size: int,
) -> SFTTrainer:
    """
    Build SFTTrainer with proper configuration.

    Args:
        model: The model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        args: Command line arguments
        world_size: Number of GPUs

    Returns:
        Configured SFTTrainer
    """
    # Detect GPU capability and set appropriate precision
    precision_settings = get_gpu_precision_settings()

    training_config = {
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "optim": "adamw_8bit",
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "linear",
        "seed": args.seed,
        "output_dir": args.output_dir,
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": args.max_seq_length,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "epoch",
        # Add precision settings based on GPU capability
        **precision_settings,
    }

    # Add DDP specific config for multi-GPU
    if world_size > 1:
        training_config["ddp_find_unused_parameters"] = False

    training_args = SFTConfig(**training_config)

    # Calculate effective batch size
    effective_batch = args.batch_size * args.grad_accum * world_size

    # Determine precision mode for display
    precision_mode = (
        "bf16"
        if precision_settings.get("bf16")
        else ("fp16" if precision_settings.get("fp16") else "fp32")
    )

    print("\n📊 Training configuration:")
    print(f"   - World size (GPUs): {world_size}")
    print(f"   - Precision: {precision_mode}")
    print(f"   - Per device batch size: {args.batch_size}")
    print(f"   - Gradient accumulation: {args.grad_accum}")
    print(f"   - Effective batch size: {effective_batch}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Max sequence length: {args.max_seq_length}")
    print()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # In TRL 0.12.0+, use processing_class instead of tokenizer
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    return trainer


def main():
    """Main training function"""
    args = parse_args()

    # Get distributed training info
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    # Apply DDP patches if using multi-GPU and patches not disabled
    if world_size > 1 and not args.no_patches:
        if OPENSLOTH_AVAILABLE:
            print("🔧 Applying OpenSloth DDP patches...")
            ddp_patch()
            patch_optimize_sft_trainer_batch_samples()
        else:
            print("⚠️  OpenSloth not available. Running without optimization patches.")

    # Print info only on rank 0
    if rank == 0:
        print("=" * 70)
        print("🦥 Qwen3-VL Fine-tuning with Multi-GPU Support")
        print("=" * 70)
        print(f"🌍 World size: {world_size} GPU(s)")
        print(f"📍 Local rank: {local_rank}")
        print(f"🎯 Model: {args.model_name}")
        print(f"📦 Dataset: {args.dataset_name}")
        print(f"📂 Output directory: {args.output_dir}")
        print("=" * 70)
        print()

    # Initialize model
    if rank == 0:
        print("🚀 Initializing model and tokenizer...")
    model, tokenizer = init_model(args, local_rank)

    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(args, local_rank)

    # Build trainer
    if rank == 0:
        print("🔨 Building trainer...")
    trainer = build_trainer(
        model, tokenizer, train_dataset, eval_dataset, args, world_size
    )

    # Train
    if rank == 0:
        print("🎯 Starting training...")
        print()

    # Use FastVisionModel.for_training to enable training mode
    FastVisionModel.for_training(model)

    # Train using unsloth's optimized trainer if available
    try:
        from unsloth import unsloth_train

        trainer_stats = unsloth_train(trainer)
    except ImportError:
        print("⚠️  unsloth not available. Using trl's trainer.")
        trainer_stats = trainer.train()

    if rank == 0:
        print()
        print("✅ Training completed!")
        print(f"📊 Training stats: {trainer_stats}")

    # Save model if requested (only on rank 0)
    if args.save_model and rank == 0:
        print(f"\n💾 Saving model to {args.model_save_path}...")
        model.save_pretrained(args.model_save_path)
        tokenizer.save_pretrained(args.model_save_path)
        print("✅ Model saved successfully!")

    if rank == 0:
        print("\n" + "=" * 70)
        print("🎉 All done! Happy training! 🦥⚡")
        print("=" * 70)


if __name__ == "__main__":
    main()
