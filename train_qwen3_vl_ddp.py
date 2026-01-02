# train_qwen3_vl_ddp.py
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

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
import mlflow
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import Dataset as TorchDataset
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
    parser.add_argument("--metric_file", type=str, default=None, help="File path to save the final metric")
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
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        use_gradient_checkpointing="unsloth",
        device_map=device_map,
        fast_inference=True,
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
        if not p.requires_grad:
            continue

        if any(keyword in n for keyword in [
            "visual", "vision", "vision_tower", "vision_model", 
            "image_encoder", "visual_encoder"
        ]):
            vision_params.append(p)
            count_v += 1
        
        elif any(keyword in n for keyword in [
            "merger", "projector", "mm_projector", 
            "multi_modal_projector", "vision_projector"
        ]):
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
        "save_total_limit": 2,
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
    
    if args.report_to == "mlflow":
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment_name
        
        if args.mlflow_run_name:
            os.environ["MLFLOW_RUN_NAME"] = args.mlflow_run_name
        
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        print(f"✅ [Init] MLflow Configured: {args.mlflow_uri} | Exp: {args.mlflow_experiment_name}")
            
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
        metrics = trainer.evaluate()
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
        
        last_run = mlflow.last_active_run()
        if last_run:
            with mlflow.start_run(run_id=last_run.info.run_id):
                if args.save_model:
                    mlflow.log_artifacts(args.model_save_path, artifact_path="lora_adapter")
                
                mlflow.log_artifacts(args.output_dir, artifact_path="checkpoints")
                print("✅ Artifacts uploaded to MLflow run:", last_run.info.run_id)
        else:
            print("⚠️ Could not find active MLflow run to upload artifacts.")
            
    if rank == 0:
        print("\n" + "=" * 70)
        print("🎉 All done! Happy training! 🦥⚡")
        print("=" * 70)


if __name__ == "__main__":
    main()
