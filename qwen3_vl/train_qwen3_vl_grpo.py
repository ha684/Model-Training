# qwen3_vl/train_qwen3_vl_grpo.py
#!/usr/bin/env python3
"""
Fine-tune Qwen3-VL Vision Language Model using GRPO/GSPO Reinforcement Learning
Usage:
    # Single GPU
    python train_qwen3_vl_grpo.py

    # Multi-GPU (2 GPUs)
    torchrun --nproc_per_node=2 train_qwen3_vl_grpo.py

    # Multi-GPU with custom settings
    torchrun --nproc_per_node=4 train_qwen3_vl_grpo.py --batch_size 1 --epochs 0.5
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from unsloth import FastVisionModel
import mlflow
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoProcessor
from trl import GRPOConfig, GRPOTrainer

from config.fields_config_1 import get_extraction_prompt
from qwen3_vl.utils import LazyVisionGRPODataset

from qwen3_vl.training_utils import (
    get_gpu_precision_settings,
    init_model,
    parse_common_args,
    setup_mlflow,
)
from qwen3_vl.reward_functions import (
    format_reward_func,
    content_accuracy_reward_func,
)

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

    
def parse_args():
    """Parse command line arguments for GRPO training"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL with GRPO reinforcement learning"
    )

    parser = parse_common_args(parser)

    parser.add_argument(
        "--num_generations",
        type=int,
        default=2,
        help="Number of completions to generate per prompt (decrease if OOM)",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="Maximum prompt length for GRPO",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=1024,
        help="Maximum completion length for GRPO",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.1, help="Max gradient norm for clipping"
    )
    parser.add_argument(
        "--save_steps", type=int, default=60, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--log_completions",
        action="store_true",
        help="Log full completions to wandb/mlflow",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="Adam beta1 parameter"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.99, help="Adam beta2 parameter"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1, help="Warmup ratio (overrides warmup_steps)"
    )
    parser.add_argument(
        "--use_gspo",
        action="store_true",
        help="Enable GSPO (Generalized Sequential Policy Optimization)",
    )

    return parser.parse_args()

def prepare_datasets(args: argparse.Namespace, local_rank: int) -> Tuple[Any, Any]:
    if local_rank == 0:
        print(f"📊 Loading dataset: {args.dataset_name}")

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    train_dataset = load_dataset(args.dataset_name, split="train", cache_dir=args.dataset_cache_dir)
    eval_dataset = load_dataset(args.dataset_name, split="validation", cache_dir=args.dataset_cache_dir)

    if local_rank == 0:
        print(f"✅ Found {len(train_dataset)} training samples")
        print(f"✅ Found {len(eval_dataset)} validation samples")

    instruction = get_extraction_prompt()
    processor = AutoProcessor.from_pretrained(args.model_name)
    lazy_train = LazyVisionGRPODataset(train_dataset, instruction,processor)
    lazy_eval = LazyVisionGRPODataset(eval_dataset, instruction,processor)

    if local_rank == 0:
        print("✅ Datasets wrapped with lazy loading (images loaded on-demand)")

    return lazy_train, lazy_eval


def build_grpo_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    args: argparse.Namespace,
    world_size: int,
    local_rank: int = 0,
):
    """Build GRPOTrainer with reward functions"""
    precision_settings = get_gpu_precision_settings()

    training_config = {
        "learning_rate": args.learning_rate,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch_fused",
        "logging_steps": args.logging_steps,
        "log_completions": args.log_completions,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "save_steps": args.save_steps,
        "max_grad_norm": args.max_grad_norm,
        "report_to": args.report_to,
        "output_dir": args.output_dir,
        "run_name": args.mlflow_run_name,
        "seed": args.seed,
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "remove_unused_columns": False,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "temperature": 1.5,
        "min_p": 0.1,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        **precision_settings,
    }

    if args.max_steps > 0:
        training_config["max_steps"] = args.max_steps
    else:
        training_config["num_train_epochs"] = args.epochs

    if args.use_gspo:
        training_config["importance_sampling_level"] = "sequence"
        training_config["mask_truncated_completions"] = False
        training_config["loss_type"] = "dr_grpo"
        if local_rank == 0:
            print("✅ GSPO (Generalized Sequential Policy Optimization) enabled")

    if world_size > 1:
        training_config["ddp_find_unused_parameters"] = False

    training_args = GRPOConfig(**training_config)

    effective_batch = args.batch_size * args.grad_accum * world_size

    precision_mode = (
        "bf16"
        if precision_settings.get("bf16")
        else ("fp16" if precision_settings.get("fp16") else "fp32")
    )

    print("\n📊 GRPO Training configuration:")
    print(f"   - World size (GPUs): {world_size}")
    print(f"   - Precision: {precision_mode}")
    print(f"   - Per device batch size: {args.batch_size}")
    print(f"   - Gradient accumulation: {args.grad_accum}")
    print(f"   - Effective batch size: {effective_batch}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Num generations: {args.num_generations}")
    print(f"   - Max prompt length: {args.max_prompt_length}")
    print(f"   - Max completion length: {args.max_completion_length}")
    if args.max_steps > 0:
        print(f"   - Max steps: {args.max_steps}")
    else:
        print(f"   - Epochs: {args.epochs}")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward_func,
            content_accuracy_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    return trainer


def main():
    args = parse_args()
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    setup_mlflow(args, local_rank=rank)

    if rank == 0:
        print("=" * 70)
        print("🦥 Qwen3-VL GRPO Fine-tuning with Multi-GPU Support")
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
    model, tokenizer = init_model(args, local_rank, load_in_4bit=False)

    if rank == 0:
        print("📊 Preparing GRPO dataset...")
    train_dataset, eval_dataset = prepare_datasets(args, local_rank)

    if rank == 0:
        print("🔨 Building GRPO trainer...")
        
    FastVisionModel.for_training(model)
    trainer = build_grpo_trainer(model, tokenizer, train_dataset, eval_dataset, args, world_size, local_rank=rank)

    if rank == 0:
        print("🎯 Starting GRPO training...")
        print()

    trainer_stats = trainer.train()

    if rank == 0:
        print()
        print("✅ Training completed!")
        print(f"📊 Training stats: {trainer_stats}")

        if args.metric_file:
            final_loss = trainer_stats.training_loss if hasattr(trainer_stats, 'training_loss') else 0.0
            with open(args.metric_file, "w") as f:
                json.dump({"training_loss": final_loss}, f)

    if args.save_model and rank == 0:
        print(f"\n💾 Saving model to {args.model_save_path}...")
        model.save_pretrained(args.model_save_path)
        tokenizer.save_pretrained(args.model_save_path)
        print("✅ Model saved successfully!")

    # Push to Hub
    if args.push_to_hub and args.hub_model_id and rank == 0:
        print(f"🚀 Pushing model to Hugging Face Hub: {args.hub_model_id}...")
        try:
            model.push_to_hub(args.hub_model_id, use_temp_dir=True)
            tokenizer.push_to_hub(args.hub_model_id, use_temp_dir=True)

            if args.report_to == "mlflow":
                mlflow.log_text(
                    f"https://huggingface.co/{args.hub_model_id}", "hf_model_url.txt"
                )

            print("✅ Push to Hub completed!")
        except Exception as e:
            print(f"⚠️ Failed to push to hub: {e}")

    if args.report_to == "mlflow" and rank == 0:
        print("📤 Uploading extra artifacts to MLflow...")

        def upload_and_cleanup():
            if args.save_model:
                mlflow.log_artifacts(args.model_save_path, artifact_path="lora_adapter")

            print("✅ Artifacts uploaded successfully!")

            print("🧹 Cleaning up local disk to save space...")
            try:
                if os.path.exists(args.output_dir):
                    shutil.rmtree(args.output_dir)
                    print(f"   - Deleted: {args.output_dir}")

                if args.save_model and os.path.exists(args.model_save_path):
                    shutil.rmtree(args.model_save_path)
                    print(f"   - Deleted: {args.model_save_path}")

            except Exception as e:
                print(f"⚠️ Error during cleanup: {e}")

        if mlflow.active_run():
            print(f"🔗 Attaching to CURRENT active run: {mlflow.active_run().info.run_id}")
            upload_and_cleanup()
            mlflow.end_run()
        else:
            last_run = mlflow.last_active_run()
            if last_run:
                print(f"🔄 Re-opening LAST run: {last_run.info.run_id}")
                with mlflow.start_run(run_id=last_run.info.run_id):
                    upload_and_cleanup()
            else:
                print("⚠️ Could not find any MLflow run to upload artifacts.")

    if rank == 0:
        print("\n" + "=" * 70)
        print("🎉 All done! Happy training! 🦥⚡")
        print("=" * 70)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
