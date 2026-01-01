# hpo_runner.py
import argparse
import random
import subprocess
import time
import math

SEARCH_SPACE = {
    "learning_rate": (1e-5, 5e-4, "log"), 
    "lora_r": [16, 32, 64, 128],
    "lora_alpha": [16, 32, 64, 128],
    "lora_dropout": (0.0, 0.1),     
    "grad_accum": [1,2,4,8],
    "weight_decay": (0.0, 0.1),
    "warmup_steps": (0, 20, "int"),
}

def get_random_value(key, value):
    if isinstance(value, list):
        return random.choice(value)
    
    if isinstance(value, tuple):
        if len(value) == 3 and value[2] == "log":
            min_val, max_val = value[0], value[1]
            log_min, log_max = math.log10(min_val), math.log10(max_val)
            return 10 ** random.uniform(log_min, log_max)
        
        elif len(value) == 3 and value[2] == "int":
            return random.randint(value[0], value[1])

        else:
            return random.uniform(value[0], value[1])
            
    return value

def get_random_config():
    return {k: get_random_value(k, v) for k, v in SEARCH_SPACE.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_trials", type=int, default=10, help="Số lượng thử nghiệm"
    )
    parser.add_argument(
        "--ngrok_url", type=str, required=True, help="URL ngrok trỏ về MLflow local"
    )
    parser.add_argument("--num_gpus", type=int, default=2, help="Số GPU sử dụng")
    parser.add_argument(
        "--dataset_name", type=str, default="ha684/senko1", help="Dataset name"
    )
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token")
    args = parser.parse_args()

    print(f"\n🎯 Starting Random Search HPO with {args.num_trials} trials")
    print(f"   - GPUs: {args.num_gpus}")
    print(f"   - Dataset: {args.dataset_name}")
    print(f"   - MLflow URI: {args.ngrok_url}")
    print("=" * 80)

    for i in range(args.num_trials):
        config = get_random_config()
        run_name = f"rs_trial_{i + 1}_lr{config['learning_rate']:.0e}_r{config['lora_r']}_steps100"

        print(f"\n{'=' * 20} STARTING TRIAL {i + 1}/{args.num_trials} {'=' * 20}")
        print(f"⚙️ Config: {config}")

        cmd = [
            "torchrun",
            f"--nproc_per_node={args.num_gpus}",
            "train_qwen3_vl_ddp.py",
            "--model_name",
            "Qwen/Qwen3-VL-2B-Instruct",
            "--report_to",
            "mlflow",
            "--learning_rate",
            str(config["learning_rate"]),
            "--lora_r",
            str(config["lora_r"]),
            "--lora_alpha",
            str(config["lora_alpha"]),
            "--lora_dropout",
            str(config["lora_dropout"]),
            "--batch_size",
            "4",
            "--grad_accum",
            str(config["grad_accum"]),
            "--weight_decay",
            str(config["weight_decay"]),
            "--warmup_steps",
            str(config["warmup_steps"]),
            "--max_steps",
            "50",
            "--eval_steps",
            "10",
            "--logging_steps",
            "1",
            "--output_dir",
            f"outputs/hpo/{run_name}",
            "--mlflow_uri",
            args.ngrok_url,
            "--mlflow_experiment_name",
            "qwen3_vl_random_search",
            "--mlflow_run_name",
            run_name,
            "--dataset_name",
            args.dataset_name,
        ]

        if args.hf_token:
            cmd.extend(["--hf_token", args.hf_token])

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Trial {i + 1} finished successfully.")
        except subprocess.CalledProcessError:
            print(
                f"❌ Trial {i + 1} FAILED (likely OOM or training error). Skipping to next..."
            )
            continue
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user. Stopping HPO...")
            break

        time.sleep(5)

    print("\n" + "=" * 80)
    print("🏁 HPO Random Search Completed!")
    print(f"📊 Check MLflow at {args.ngrok_url} to analyze results")
    print("💡 Sort by 'eval_loss' to find the best model")
    print("=" * 80)


if __name__ == "__main__":
    main()
