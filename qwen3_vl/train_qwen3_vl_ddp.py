# qwen3-vl/train_qwen3_vl_ddp.py
#!/usr/bin/env python3
"""
Fine-tune Qwen3-VL Vision Language Model with Multi-GPU support using Supervised Fine-Tuning (SFT)
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
import sys
from pathlib import Path
from typing import Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
import mlflow
import torch.distributed as dist
from datasets import load_dataset
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from config.fields_config_1 import get_extraction_prompt
from qwen3_vl.training_utils import (
    create_custom_optimizer_and_scheduler,
    get_gpu_precision_settings,
    init_model,
    parse_common_args,
    setup_mlflow,
)
from qwen3_vl.utils import LazyVisionDataset

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def parse_args():
    """Parse command line arguments for SFT training"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL with multi-GPU support using SFT"
    )

    parser = parse_common_args(parser)
    return parser.parse_args()


def prepare_datasets(args: argparse.Namespace, local_rank: int) -> Tuple[Any, Any]:
    if local_rank == 0:
        print(f"📊 Loading dataset: {args.dataset_name}")

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    train_dataset = load_dataset(args.dataset_name, split="train")
    eval_dataset = load_dataset(args.dataset_name, split="test")

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
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "optim": "adamw_torch_fused",
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
        "save_total_limit": 1,
        "run_name": args.mlflow_run_name,
        **precision_settings,
    }

    if args.max_steps > 0:
        training_config["max_steps"] = args.max_steps
    else:
        training_config["num_train_epochs"] = args.epochs

    if world_size > 1:
        training_config["ddp_find_unused_parameters"] = False

    training_args = SFTConfig(**training_config)

    effective_batch = args.batch_size * args.grad_accum * world_size

    precision_mode = (
        "bf16"
        if precision_settings.get("bf16")
        else ("fp16" if precision_settings.get("fp16") else "fp32")
    )

    optimizer, scheduler = create_custom_optimizer_and_scheduler(
        model, args, effective_batch
    )

    print("\n📊 Training configuration:")
    print(f"   - World size (GPUs): {world_size}")
    print(f"   - Precision: {precision_mode}")
    print(f"   - Per device batch size: {args.batch_size}")
    print(f"   - Gradient accumulation: {args.grad_accum}")
    print(f"   - Effective batch size: {effective_batch}")
    print(f"   - Learning rate: {args.learning_rate}")
    if args.max_steps > 0:
        print(f"   - Max steps: {args.max_steps}")
    else:
        print(f"   - Epochs: {args.epochs}")
    print(f"   - Max sequence length: {args.max_seq_length}")
    # compute_metrics_fn = partial(compute_metrics_cer, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        # compute_metrics=compute_metrics_fn,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=1e-4
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
    setup_mlflow(args, local_rank=rank)

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
    model, tokenizer = init_model(
        args, local_rank, load_in_4bit=False
    )

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
        trainer.args.report_to = []

    metrics = trainer.evaluate()

    if rank == 0:
        final_eval_loss = metrics.get("eval_loss", float("inf"))

        print(f"📉 Final Eval Loss: {final_eval_loss}")

        if args.metric_file:
            import json

            with open(args.metric_file, "w") as f:
                json.dump({"eval_loss": final_eval_loss}, f)

        print()
        print("✅ Training completed!")
        print(f"📊 Training stats: {trainer_stats}")

    if args.save_model and rank == 0:
        print(f"\n💾 Saving model to {args.model_save_path}...")
        model.save_pretrained(args.model_save_path)
        tokenizer.save_pretrained(args.model_save_path)
        print("✅ Model saved successfully!")

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
            if "metrics" in locals():
                print("📊 Logging eval metrics directly to run...")
                mlflow.log_metrics(metrics)

            if args.save_model:
                mlflow.log_artifacts(args.model_save_path, artifact_path="lora_adapter")

            # if os.path.exists(args.output_dir):
            #     mlflow.log_artifacts(args.output_dir, artifact_path="checkpoints")

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
            print(
                f"🔗 Attaching to CURRENT active run: {mlflow.active_run().info.run_id}"
            )
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
