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
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import argparse
import os
from typing import Any, Dict, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from torch.optim import AdamW
from transformers import EarlyStoppingCallback, get_cosine_schedule_with_warmup
from trl import SFTConfig, SFTTrainer


from config.fields_config import get_extraction_prompt


def get_gpu_precision_settings() -> Dict[str, bool]:
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}

    gpu_capability = torch.cuda.get_device_capability()
    major, minor = gpu_capability

    supports_bf16 = major >= 8

    if supports_bf16:
        return {"bf16": True, "fp16": False}
    else:
        return {"bf16": False, "fp16": True}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL with multi-GPU support"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=8192, help="Maximum sequence length"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ha684/senko1",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HuggingFace API token"
    )

    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha", type=int, default=128, help="LoRA alpha parameter"
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")

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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

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

def create_custom_optimizer_and_scheduler(model, args, num_training_steps):
    vision_params = []
    proj_params = []
    llm_params = []

    print("🔍 Grouping parameters for Differential LR...")
    
    LR_LLM = args.learning_rate
    LR_PROJ = args.learning_rate * 1.0
    LR_VISION = args.learning_rate * 0.1

    count_v, count_p, count_l = 0, 0, 0

    for n, p in model.named_parameters():
        print(n)
        if not p.requires_grad:
            continue
            
        if "merger" in n or "projector" in n or "mm_projector" in n or "mlp" in n:
            proj_params.append(p)
            count_p += 1
        
        elif "visual" in n or "vision" in n:
            vision_params.append(p)
            count_v += 1
            
        else:
            llm_params.append(p)
            count_l += 1

    print(f"  - LLM params ({LR_LLM}): {count_l} tensors")
    print(f"  - Projector params ({LR_PROJ}): {count_p} tensors")
    print(f"  - Vision params ({LR_VISION}): {count_v} tensors")

    optimizer = AdamW(
        [
            {"params": llm_params, "lr": LR_LLM},
            {"params": proj_params, "lr": LR_PROJ},
            {"params": vision_params, "lr": LR_VISION},
        ],
        weight_decay=args.weight_decay,
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler

def convert_to_conversation(sample: Dict, instruction: str) -> Dict:
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
    def __init__(self, hf_dataset, instruction: str):
        self.dataset = hf_dataset
        self.instruction = instruction

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return convert_to_conversation(sample, self.instruction)


def prepare_datasets(args: argparse.Namespace, local_rank: int) -> Tuple[Any, Any]:
    if local_rank == 0:
        print(f"📊 Loading dataset: {args.dataset_name}")

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    train_dataset = load_dataset(args.dataset_name, split="train")
    eval_dataset = load_dataset(args.dataset_name, split="validation")

    if local_rank == 0:
        print(f"✅ Found {len(train_dataset)} training samples")
        print(f"✅ Found {len(eval_dataset)} validation samples")

    instruction = get_extraction_prompt()

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
        "save_strategy": "steps",
        "save_steps": args.eval_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        **precision_settings,
    }

    if world_size > 1:
        training_config["ddp_find_unused_parameters"] = False

    training_args = SFTConfig(**training_config)

    effective_batch = args.batch_size * args.grad_accum * world_size

    precision_mode = (
        "bf16"
        if precision_settings.get("bf16")
        else ("fp16" if precision_settings.get("fp16") else "fp32")
    )
    
    optimizer, scheduler = create_custom_optimizer_and_scheduler(model, args, effective_batch)
    
    print("\n📊 Training configuration:")
    print(f"   - World size (GPUs): {world_size}")
    print(f"   - Precision: {precision_mode}")
    print(f"   - Per device batch size: {args.batch_size}")
    print(f"   - Gradient accumulation: {args.grad_accum}")
    print(f"   - Effective batch size: {effective_batch}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Max sequence length: {args.max_seq_length}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5, early_stopping_threshold=0.01
            )
        ],
        optimizers=(optimizer, scheduler),
    )

    return trainer


def main():
    args = parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

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

    if rank == 0:
        print("🚀 Initializing model and tokenizer...")
    model, tokenizer = init_model(args, local_rank)

    train_dataset, eval_dataset = prepare_datasets(args, local_rank)

    if rank == 0:
        print("🔨 Building trainer...")
    trainer = build_trainer(
        model, tokenizer, train_dataset, eval_dataset, args, world_size
    )

    if rank == 0:
        print("🎯 Starting training...")
        print()

    FastVisionModel.for_training(model)

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
