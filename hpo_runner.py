import argparse
import json
import os
import subprocess

import optuna
from optuna.integration.mlflow import MLflowCallback


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """
    Hàm này định nghĩa 1 lần chạy thử nghiệm (1 trial)
    """
    # 1. Định nghĩa không gian tìm kiếm bằng Optuna (Thay thế SEARCH_SPACE cũ)
    # Optuna tự động gợi ý giá trị trong khoảng này
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    lora_r = trial.suggest_categorical("lora_r", [32, 64, 128])
    lora_alpha = trial.suggest_categorical("lora_alpha", [64, 128, 256])
    # lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)
    grad_accum = trial.suggest_categorical("grad_accum", [1, 2])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    # warmup_steps = trial.suggest_int("warmup_steps", 0, 20)

    # Tạo tên run
    run_name = (
        f"optuna_trial_{trial.number}_lr{learning_rate:.1e}_r{lora_r}_alpha{lora_alpha}"
    )

    # File tạm để chứa kết quả loss từ training script
    metric_filename = f"metric_trial_{trial.number}.json"

    print(f"\n{'=' * 20} STARTING TRIAL {trial.number} {'=' * 20}")
    print(f"⚙️ Params: {trial.params}")

    # 2. Xây dựng câu lệnh (Thêm --metric_file)
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        "train_qwen3_vl_ddp.py",
        "--model_name",
        "Qwen/Qwen3-VL-2B-Instruct",
        "--report_to",
        "mlflow",
        "--learning_rate",
        str(learning_rate),
        "--lora_r",
        str(lora_r),
        "--lora_alpha",
        str(lora_alpha),
        "--lora_dropout",
        "0",
        "--batch_size",
        "2",
        "--grad_accum",
        str(grad_accum),
        "--weight_decay",
        str(weight_decay),
        "--warmup_steps",
        "5",
        "--max_steps",
        "30",
        "--eval_steps",
        "10",
        "--logging_steps",
        "1",
        "--output_dir",
        f"outputs/hpo/{run_name}",
        "--mlflow_uri",
        args.ngrok_url,
        # Lưu ý: OptunaCallback sẽ tự log params, nên trong script con
        # ta có thể log vào experiment con hoặc để chung
        "--mlflow_experiment_name",
        "qwen3_vl_optuna_details",
        "--mlflow_run_name",
        run_name,
        "--dataset_name",
        args.dataset_name,
        "--metric_file",
        metric_filename,
    ]

    if args.hf_token:
        cmd.extend(["--hf_token", args.hf_token])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Trial {trial.number} FAILED.")
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

    return eval_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--ngrok_url", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--dataset_name", type=str, default="ha684/senko1")
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    db_storage = "sqlite:///optuna_history.db"
    study_name = "qwen3_vl_optimization"

    mlflow_cb = MLflowCallback(
        tracking_uri=args.ngrok_url,
        metric_name="eval_loss",
        mlflow_kwargs={"experiment_name": "qwen3_vl_optuna_coordinator"},
    )

    print(f"🎯 Starting Optuna HPO with persistent storage: {db_storage}")

    study = optuna.create_study(
        study_name=study_name,
        storage=db_storage,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )

    completed_trials = len(
        [
            t
            for t in study.trials
            if t.state
            in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        ]
    )

    print(f"📊 Status: {completed_trials}/{args.num_trials} trials completed.")

    if completed_trials >= args.num_trials:
        print("✅ Study already finished! No new trials needed.")
    else:
        remaining_trials = args.num_trials - completed_trials
        print(f"🚀 Resuming... Running {remaining_trials} more trials.")

        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=remaining_trials,  # Chỉ chạy số lượng còn thiếu
            callbacks=[mlflow_cb],
        )

    print("\n" + "=" * 80)
    print("🏆 Best trial:")
    if study.best_trial:
        print(f"  Value: {study.best_value}")
        print(f"  Params: {study.best_params}")
    else:
        print("  No successful trials yet.")
    print("=" * 80)


if __name__ == "__main__":
    main()
