import argparse
import subprocess
import os
import json
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """
    Hàm này định nghĩa 1 lần chạy thử nghiệm (1 trial)
    """
    # 1. Định nghĩa không gian tìm kiếm bằng Optuna (Thay thế SEARCH_SPACE cũ)
    # Optuna tự động gợi ý giá trị trong khoảng này
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    lora_r = trial.suggest_categorical("lora_r", [16, 32, 64, 128])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64, 128])
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)
    grad_accum = trial.suggest_categorical("grad_accum", [1, 2, 4, 8])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    # warmup_steps = trial.suggest_int("warmup_steps", 0, 20)

    # Tạo tên run
    run_name = f"optuna_trial_{trial.number}_lr{learning_rate:.1e}_r{lora_r}_alpha{lora_alpha}"
    
    # File tạm để chứa kết quả loss từ training script
    metric_filename = f"metric_trial_{trial.number}.json"

    print(f"\n{'=' * 20} STARTING TRIAL {trial.number} {'=' * 20}")
    print(f"⚙️ Params: {trial.params}")

    # 2. Xây dựng câu lệnh (Thêm --metric_file)
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        "train_qwen3_vl_ddp.py",
        "--model_name", "Qwen/Qwen3-VL-2B-Instruct",
        "--report_to", "mlflow",
        "--learning_rate", str(learning_rate),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
        "--batch_size", "2",
        "--grad_accum", str(grad_accum),
        "--weight_decay", str(weight_decay),
        "--warmup_steps", "5",
        "--max_steps", "30", 
        "--eval_steps", "5",
        "--logging_steps", "1",
        "--output_dir", f"outputs/hpo/{run_name}",
        "--mlflow_uri", args.ngrok_url,
        # Lưu ý: OptunaCallback sẽ tự log params, nên trong script con 
        # ta có thể log vào experiment con hoặc để chung
        "--mlflow_experiment_name", "qwen3_vl_optuna_details", 
        "--mlflow_run_name", run_name,
        "--dataset_name", args.dataset_name,
        "--metric_file", metric_filename,
    ]

    if args.hf_token:
        cmd.extend(["--hf_token", args.hf_token])

    # 3. Chạy training script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Trial {trial.number} FAILED.")
        # Nếu fail, trả về loss thật lớn để Optuna biết đường tránh
        return float("inf") 

    # 4. Đọc kết quả từ file
    eval_loss = float("inf")
    if os.path.exists(metric_filename):
        with open(metric_filename, "r") as f:
            data = json.load(f)
            eval_loss = data.get("eval_loss", float("inf"))
        # Dọn dẹp file tạm
        os.remove(metric_filename)
    else:
        print("⚠️ Metric file not found!")

    print(f"🏁 Trial {trial.number} finished with Loss: {eval_loss}")
    
    # Optuna cần giá trị trả về để tối ưu (Minimize loss)
    return eval_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--ngrok_url", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--dataset_name", type=str, default="ha684/senko1")
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    # Tích hợp MLflow để theo dõi quá trình Optuna chọn tham số
    mlflow_cb = MLflowCallback(
        tracking_uri=args.ngrok_url,
        metric_name="eval_loss",
        mlflow_kwargs={
            "experiment_name": "qwen3_vl_optuna_coordinator" # Experiment cha
        }
    )

    print(f"🎯 Starting Optuna HPO...")

    # Tạo Study: Hướng là MINIMIZE (vì chúng ta muốn giảm Loss)
    study = optuna.create_study(
        study_name="qwen3_vl_optimization",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner() # Tùy chọn
    )

    # Bắt đầu tối ưu
    # Dùng lambda để truyền `args` vào hàm objective
    study.optimize(lambda trial: objective(trial, args), n_trials=args.num_trials, callbacks=[mlflow_cb])

    print("\n" + "=" * 80)
    print("🏆 Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")
    print("=" * 80)

if __name__ == "__main__":
    main()