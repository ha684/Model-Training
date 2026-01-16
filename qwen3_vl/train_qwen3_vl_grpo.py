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

sys.path.insert(0, str(Path(__file__).parent.parent))

from unsloth import FastVisionModel
import mlflow
import torch.distributed as dist
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from qwen3_vl.reward_functions import (
    REASONING_END,
    REASONING_START,
    SOLUTION_END,
    SOLUTION_START,
    correctness_reward_func,
    formatting_reward_func,
)
from qwen3_vl.training_utils import (
    get_gpu_precision_settings,
    init_model,
    parse_common_args,
    setup_mlflow,
)

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def parse_args():
    """Parse command line arguments for GRPO training"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL with GRPO reinforcement learning"
    )

    # Add common arguments (shared with SFT)
    parser = parse_common_args(parser)

    # GRPO-specific arguments
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

    # Override defaults for GRPO
    parser.set_defaults(
        learning_rate=5e-6,
        batch_size=1,
        grad_accum=1,
        epochs=0.5,
        output_dir="outputs/qwen3_vl_grpo",
        mlflow_experiment_name="qwen3_vl_grpo",
        mlflow_run_name="qwen3_vl_grpo",
    )

    return parser.parse_args()


def prepare_grpo_dataset(args: argparse.Namespace, tokenizer, local_rank: int):
    """
    Prepare dataset for GRPO training
    
    Uses MathVista testmini split (filtered for numeric answers)
    Formats prompts to include reasoning + solution delimiters
    
    Args:
        args: Parsed arguments
        tokenizer: Model tokenizer/processor
        local_rank: Local rank for distributed training
    
    Returns:
        Prepared dataset ready for GRPOTrainer
    """
    if local_rank == 0:
        print(f"📊 Loading dataset: {args.dataset_name}")

    # Load dataset
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    dataset = load_dataset(args.dataset_name, split="train")

    if local_rank == 0:
        print(f"✅ Found {len(dataset)} samples")

    # Filter for numeric answers (if using MathVista)
    def is_numeric_answer(example):
        try:
            float(example["answer"])
            return True
        except:
            return False

    if "answer" in dataset.column_names:
        dataset = dataset.filter(is_numeric_answer)
        if local_rank == 0:
            print(f"✅ Filtered to {len(dataset)} numeric answer samples")

    # Resize images to (512, 512) for faster training
    def resize_images(example):
        image = example.get("decoded_image", example.get("image"))
        if image:
            image = image.resize((512, 512))
            example["image"] = image
        return example

    dataset = dataset.map(resize_images)

    # Convert to RGB
    def convert_to_rgb(example):
        image = example.get("image")
        if image and image.mode != "RGB":
            image = image.convert("RGB")
            example["image"] = image
        return example

    dataset = dataset.map(convert_to_rgb)

    # Format conversations
    def make_conversation(example):
        text_content = (
            f"{example['question']}. Also first provide your reasoning or working out"
            f" on how you would go about solving the question between {REASONING_START} and {REASONING_END}"
            f" and then your final answer between {SOLUTION_START} and (put a single float here) {SOLUTION_END}"
        )

        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_content},
                ],
            },
        ]
        return {
            "prompt": prompt,
            "image": example.get("image"),
            "answer": example.get("answer"),
        }

    train_dataset = dataset.map(make_conversation)

    # Fix column names for image
    if "decoded_image" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("decoded_image")

    # Apply chat template
    train_dataset = train_dataset.map(
        lambda example: {
            "prompt": tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=True,  # Must add for generation
            )
        }
    )

    if local_rank == 0:
        print("✅ Dataset formatted for GRPO training")
        print(f"   Sample prompt: {train_dataset[0]['prompt'][:200]}...")

    return train_dataset


def build_grpo_trainer(
    model,
    tokenizer,
    train_dataset,
    args: argparse.Namespace,
    world_size: int,
    local_rank: int = 0,
):
    """
    Build GRPOTrainer with reward functions
    
    Args:
        model: The model to train
        tokenizer: Model tokenizer/processor
        train_dataset: Prepared dataset
        args: Parsed arguments
        world_size: Number of GPUs
        local_rank: Local rank for distributed training
    
    Returns:
        GRPOTrainer instance
    """
    precision_settings = get_gpu_precision_settings()

    training_config = {
        "learning_rate": args.learning_rate,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "logging_steps": args.logging_steps,
        "log_completions": args.log_completions,
        "per_device_train_batch_size": args.batch_size,
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
        **precision_settings,
    }

    # Add epochs or max_steps
    if args.max_steps > 0:
        training_config["max_steps"] = args.max_steps
    else:
        training_config["num_train_epochs"] = args.epochs

    # Enable GSPO if requested
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
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[
            formatting_reward_func,
            correctness_reward_func,
        ],
        train_dataset=train_dataset,
    )

    return trainer


def main():
    args = parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    # Setup MLflow tracking
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
    # For GRPO: use 4-bit quantization + fast inference
    model, tokenizer = init_model(args, local_rank, load_in_4bit=True, fast_inference=False)

    if rank == 0:
        print("📊 Preparing GRPO dataset...")
    train_dataset = prepare_grpo_dataset(args, tokenizer, local_rank)

    if rank == 0:
        print("🔨 Building GRPO trainer...")
    trainer = build_grpo_trainer(model, tokenizer, train_dataset, args, world_size, local_rank=rank)

    if rank == 0:
        print("🎯 Starting GRPO training...")
        print()

    # Enable training mode
    FastVisionModel.for_training(model)

    # Train
    try:
        from unsloth import unsloth_train

        trainer_stats = unsloth_train(trainer)
    except ImportError:
        print("⚠️  unsloth_train not available. Using trl's trainer.")
        trainer_stats = trainer.train()

    if rank == 0:
        print()
        print("✅ Training completed!")
        print(f"📊 Training stats: {trainer_stats}")

        # Save final metrics
        if args.metric_file:
            final_loss = trainer_stats.training_loss if hasattr(trainer_stats, 'training_loss') else 0.0
            with open(args.metric_file, "w") as f:
                json.dump({"training_loss": final_loss}, f)

    # Save model
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

    # Upload to MLflow
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
