# qwen3_vl/training_utils.py
"""
Shared utilities for training Qwen3-VL models
Used by both SFT and GRPO/RL training scripts
"""

import argparse
import os
from typing import Any, Dict, Tuple

import mlflow
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from unsloth import FastVisionModel


def get_gpu_precision_settings() -> Dict[str, bool]:
    """
    Detect GPU capability and return optimal precision settings
    Returns bf16 for Ampere+ (compute >= 8.0), fp16 for older GPUs
    """
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}

    gpu_capability = torch.cuda.get_device_capability()
    major, minor = gpu_capability

    supports_bf16 = major >= 8

    if supports_bf16:
        return {"bf16": True, "fp16": False}
    else:
        return {"bf16": False, "fp16": True}


def parse_common_args(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """
    Add common arguments shared between SFT and GRPO training
    
    Args:
        parser: Existing ArgumentParser instance. If None, creates new one.
    
    Returns:
        ArgumentParser with common arguments added
    """
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=8192, help="Maximum sequence length"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ha684/senko1",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HuggingFace API token"
    )

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha", type=int, default=128, help="LoRA alpha parameter"
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per device train batch size",
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps (overrides epochs if > 0)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps")
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging interval")

    # Output arguments
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

    # Reporting arguments
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "tensorboard", "wandb", "mlflow"],
        help="Reporting tool",
    )
    parser.add_argument(
        "--mlflow_uri", type=str, default="http://localhost:5000", help="MLflow URI"
    )
    parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        default="qwen3_vl_finetune",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--mlflow_run_name",
        type=str,
        default="qwen3_vl_finetune",
        help="MLflow run name",
    )

    # Misc arguments
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Repository ID on HF (e.g., username/qwen-finetuned)",
    )
    parser.add_argument(
        "--metric_file", type=str, default=None, help="File path to save the final metric"
    )

    return parser


def setup_mlflow(args: argparse.Namespace, local_rank: int = 0) -> None:
    """
    Initialize MLflow tracking if report_to == 'mlflow'
    Only runs on rank 0 for distributed training
    
    Args:
        args: Parsed arguments containing mlflow settings
        local_rank: Local rank for distributed training (default: 0)
    """
    if args.report_to != "mlflow" or local_rank != 0:
        return

    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment_name

    if args.mlflow_run_name:
        os.environ["MLFLOW_RUN_NAME"] = args.mlflow_run_name

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)
    print(f"✅ [Init] MLflow Configured: {args.mlflow_uri} | Exp: {args.mlflow_experiment_name}")


def init_model(
    args: argparse.Namespace,
    local_rank: int,
    load_in_4bit: bool = False,
    fast_inference: bool = False,
) -> Tuple[Any, Any]:
    """
    Initialize Qwen3-VL model with LoRA adapters
    
    Args:
        args: Parsed arguments containing model settings
        local_rank: Local rank for distributed training
        load_in_4bit: Whether to load model in 4-bit quantization (default: False for SFT, True for GRPO)
        fast_inference: Enable vLLM fast inference for GRPO (default: False)
    
    Returns:
        Tuple of (model, tokenizer/processor)
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    print(f"🔧 Loading model: {args.model_name}")
    print(f"   Device map: {device_map}")
    print(f"   4-bit mode: {load_in_4bit}")

    # For GRPO: use 4-bit + fast_inference
    # For SFT: use 16-bit + gradient checkpointing
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_16bit=not load_in_4bit,  # 16bit if not 4bit
        use_gradient_checkpointing="unsloth" if not load_in_4bit else None,
        fast_inference=fast_inference,
        device_map=device_map,
        gpu_memory_utilization=0.8 if fast_inference else None,
    )

    # Sync special tokens for processor
    actual_tokenizer = tokenizer
    if hasattr(tokenizer, "tokenizer"):
        actual_tokenizer = tokenizer.tokenizer
        print(f"   Found nested tokenizer: {type(actual_tokenizer).__name__}")

    if hasattr(tokenizer, "tokenizer"):
        for attr in [
            "eos_token",
            "eos_token_id",
            "bos_token",
            "bos_token_id",
            "pad_token",
            "pad_token_id",
        ]:
            if hasattr(actual_tokenizer, attr):
                value = getattr(actual_tokenizer, attr)
                if value is not None:
                    setattr(tokenizer, attr, value)
        print("   Synced special tokens to processor")
        print(f"   Processor eos_token: {getattr(tokenizer, 'eos_token', None)}")

    print("🔧 Configuring LoRA...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
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


def create_custom_optimizer_and_scheduler(
    model: Any,
    args: argparse.Namespace,
    num_training_steps: int,
) -> Tuple[AdamW, Any]:
    """
    Create optimizer with differential learning rates and cosine scheduler
    
    Vision layers: 0.01x base LR
    Projector layers: 1.0x base LR
    LLM layers: 1.0x base LR
    
    Args:
        model: The model to optimize
        args: Parsed arguments containing lr and weight_decay
        num_training_steps: Total training steps for scheduler
    
    Returns:
        Tuple of (optimizer, scheduler)
    """
    vision_params = []
    proj_params = []
    llm_params = []

    print("🔍 Grouping parameters for Differential LR...")

    LR_LLM = args.learning_rate
    LR_PROJ = args.learning_rate * 1.0
    LR_VISION = args.learning_rate * 0.01

    count_v, count_p, count_l = 0, 0, 0

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if any(
            keyword in n
            for keyword in [
                "visual",
                "vision",
                "vision_tower",
                "vision_model",
                "image_encoder",
                "visual_encoder",
            ]
        ):
            vision_params.append(p)
            count_v += 1

        elif any(
            keyword in n
            for keyword in [
                "merger",
                "projector",
                "mm_projector",
                "multi_modal_projector",
                "vision_projector",
            ]
        ):
            proj_params.append(p)
            count_p += 1

        else:
            llm_params.append(p)
            count_l += 1

    print(f"  - Vision params ({LR_VISION}): {count_v} tensors")
    print(f"  - Projector params ({LR_PROJ}): {count_p} tensors")
    print(f"  - LLM params ({LR_LLM}): {count_l} tensors")

    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total_grouped = count_v + count_p + count_l
    if total_trainable != total_grouped:
        print(f"⚠️  WARNING: {total_trainable - total_grouped} params not grouped!")

    optimizer = AdamW(
        [
            {"params": vision_params, "lr": LR_VISION},
            {"params": proj_params, "lr": LR_PROJ},
            {"params": llm_params, "lr": LR_LLM},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler
