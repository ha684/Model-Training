import argparse
import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import optuna
import random
env = os.environ.copy()
env["MASTER_ADDR"] = "127.0.0.1"
env["MASTER_PORT"] = str(random.randint(20000, 40000))

# Bạn cần: pip install mlflow
import mlflow

os.environ["MLFLOW_TRACKING_USERNAME"]="robotpro682004"
os.environ["MLFLOW_TRACKING_PASSWORD"]="59d3b474fda55ef9e051969b30b778bc42763e2d"

@dataclass
class TrialResult:
    final_eval_loss: float
    last_step: int
    state: str  # "COMPLETE" | "PRUNED" | "FAIL"


def _safe_float(x, default=float("inf")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _read_latest_metric_from_jsonl(metric_file: Path) -> Tuple[Optional[int], Optional[float]]:
    """
    Đọc dòng cuối cùng hợp lệ từ file JSONL (mỗi dòng 1 JSON object).
    Format gợi ý cho training script ghi:
      {"step": 10, "eval_loss": 2.345}
      {"step": 20, "eval_loss": 2.123}
    """
    if not metric_file.exists():
        return None, None

    try:
        # Đọc nhanh: lấy vài KB cuối file để tìm dòng cuối
        with metric_file.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # đọc tối đa 64KB cuối file
            read_size = min(size, 65536)
            f.seek(-read_size, os.SEEK_END)
            tail = f.read().decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        # duyệt từ cuối lên để tìm JSON hợp lệ có step & eval_loss
        for ln in reversed(lines):
            try:
                obj = json.loads(ln)
                step = obj.get("step", None)
                loss = obj.get("eval_loss", None)
                if step is None or loss is None:
                    continue
                return int(step), float(loss)
            except Exception:
                continue
        return None, None
    except Exception:
        return None, None


def _terminate_process_tree(proc: subprocess.Popen, kill_after_sec: float = 8.0) -> None:
    """
    Terminate subprocess + process group (Linux-friendly).
    """
    if proc.poll() is not None:
        return

    try:
        # Nếu start_new_session=True thì pid là pgid
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    t0 = time.time()
    while time.time() - t0 < kill_after_sec:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    # ---- 1) Search space (bạn có thể mở rộng thêm sau)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lora_r = trial.suggest_categorical("lora_r", [32, 64, 128])
    lora_alpha = trial.suggest_categorical("lora_alpha", [64, 128, 256])
    grad_accum = trial.suggest_categorical("grad_accum", [1, 2])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    # ---- 2) MLflow: tạo 1 run cho mỗi trial (coordinator)
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    run_name = f"trial_{trial.number:04d}_lr{learning_rate:.1e}_r{lora_r}_a{lora_alpha}_ga{grad_accum}_wd{weight_decay:.3f}"
    run_tags = {
        "optuna.study_name": args.study_name,
        "optuna.trial_number": str(trial.number),
        "optuna.sampler": type(trial.study.sampler).__name__,
        "optuna.pruner": type(trial.study.pruner).__name__ if trial.study.pruner else "None",
        "dataset_name": args.dataset_name,
        "model_name": "Qwen/Qwen3-VL-2B-Instruct",
    }

    # ---- 3) Chuẩn bị metric file unique + output dir
    trial_id = uuid.uuid4().hex[:8]
    out_dir = Path(args.output_root) / args.study_name / f"trial_{trial.number:04d}_{trial_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_file = out_dir / "metrics.jsonl"  # training script phải append json lines vào đây

    # ---- 4) Chạy training subprocess và poll metrics để pruning hoạt động thật
    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id

        # log params + tags ngay đầu run
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "grad_accum": grad_accum,
                "weight_decay": weight_decay,
                "max_steps": args.max_steps,
                "eval_steps": args.eval_steps,
                "logging_steps": args.logging_steps,
                "batch_size": args.batch_size,
                "num_gpus": args.num_gpus,
            }
        )
        mlflow.set_tags(run_tags)

        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = args.mlflow_uri
        env["MLFLOW_RUN_ID"] = run_id
        env["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment

        cmd = [
            "torchrun",
            f"--nproc_per_node={args.num_gpus}",
            f"--master_port={29500 + (trial.number % 1000)}",
            args.train_script,
            "--model_name",
            "Qwen/Qwen3-VL-8B-Instruct",
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
            str(args.batch_size),
            "--grad_accum",
            str(grad_accum),
            "--weight_decay",
            str(weight_decay),
            "--warmup_steps",
            str(args.warmup_steps),
            "--max_steps",
            str(args.max_steps),
            "--eval_steps",
            str(args.eval_steps),
            "--logging_steps",
            str(args.logging_steps),
            "--output_dir",
            str(out_dir / "outputs"),
            "--mlflow_uri",
            args.mlflow_uri,
            "--mlflow_experiment_name",
            args.mlflow_experiment,
            "--mlflow_run_name",
            run_name,
            "--dataset_name",
            args.dataset_name,
            "--metric_file",
            str(metric_file),
        ]

        if args.hf_token:
            cmd.extend(["--hf_token", args.hf_token])

        print(f"\n{'=' * 20} STARTING TRIAL {trial.number} {'=' * 20}")
        print(f"⚙️ Params: {trial.params}")
        print(f"📁 Output: {out_dir}")
        print(f"🧾 Metric file: {metric_file}")

        proc = subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,  # để kill process group khi prune
        )

        last_reported_step = -1
        last_loss = float("inf")
        t_start = time.time()

        try:
            while True:
                rc = proc.poll()
                # poll metrics để report cho Optuna -> pruning hoạt động
                step, loss = _read_latest_metric_from_jsonl(metric_file)
                if step is not None and loss is not None and step > last_reported_step:
                    last_reported_step = step
                    last_loss = loss

                    # report intermediate metric
                    trial.report(loss, step=step)
                    mlflow.log_metric("eval_loss", loss, step=step)

                    # prune?
                    if trial.should_prune():
                        mlflow.set_tag("optuna.state", "PRUNED")
                        mlflow.set_tag("optuna.pruned_at_step", str(step))
                        _terminate_process_tree(proc)
                        raise optuna.TrialPruned(f"Pruned at step={step} with eval_loss={loss}")

                if rc is not None:
                    # process ended
                    break

                # tránh busy-wait
                time.sleep(args.poll_interval)

            # process ended: check return code
            if proc.returncode != 0:
                mlflow.set_tag("optuna.state", "FAIL")
                mlflow.set_tag("train.returncode", str(proc.returncode))
                print(f"❌ Trial {trial.number} FAILED with return code {proc.returncode}")
                return float("inf")

            # finalize: lấy metric cuối cùng
            final_step, final_loss = _read_latest_metric_from_jsonl(metric_file)
            if final_loss is None:
                # fallback: nếu training script chỉ ghi JSON 1 lần (không phải jsonl)
                # bạn có thể sửa training script theo jsonl để prune hoạt động tốt.
                mlflow.set_tag("optuna.state", "FAIL")
                print("⚠️ No metric found in metrics file. Returning inf.")
                return float("inf")

            # log final
            mlflow.log_metric("final_eval_loss", float(final_loss))
            mlflow.set_tag("optuna.state", "COMPLETE")
            mlflow.set_tag("optuna.final_step", str(final_step if final_step is not None else -1))
            mlflow.log_metric("trial_wall_time_sec", time.time() - t_start)

            print(f"🏁 Trial {trial.number} finished with Loss: {final_loss}")
            return float(final_loss)

        except optuna.TrialPruned:
            # Optuna sẽ xử lý state PRUNED
            raise
        except KeyboardInterrupt:
            mlflow.set_tag("optuna.state", "INTERRUPTED")
            _terminate_process_tree(proc)
            raise
        except Exception as e:
            mlflow.set_tag("optuna.state", "FAIL")
            mlflow.set_tag("exception", repr(e))
            _terminate_process_tree(proc)
            print(f"❌ Trial {trial.number} ERROR: {e!r}")
            return float("inf")


def main():
    parser = argparse.ArgumentParser()

    # Optuna
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="qwen3_vl_optimization")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_history.db")
    parser.add_argument("--seed", type=int, default=42)

    # MLflow
    parser.add_argument("--mlflow_uri", type=str, required=True)
    parser.add_argument("--mlflow_experiment", type=str, default="qwen3_vl_optuna")

    # Training
    parser.add_argument("--train_script", type=str, default="qwen3-vl/train_qwen3_vl_ddp.py")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--dataset_name", type=str, default="ha684/senko1")
    parser.add_argument("--hf_token", type=str, default=None)

    # Training knobs
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=1)

    # Coordinator runtime
    parser.add_argument("--output_root", type=str, default="outputs/hpo")
    parser.add_argument("--poll_interval", type=float, default=1.0)

    args = parser.parse_args()

    # Sampler + Pruner (đúng bài: có pruning thật nhờ trial.report/should_prune)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # cho vài trial chạy đủ để có median baseline
        n_warmup_steps=1,    # step được tính theo "eval step" mà bạn report
        interval_steps=1,
    )

    print(f"🎯 Starting Optuna HPO with storage: {args.storage}")
    print(f"🧪 Study: {args.study_name}")
    print(f"📌 MLflow: {args.mlflow_uri} / experiment={args.mlflow_experiment}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    # Resume-friendly
    completed_or_pruned = [
        t for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    ]
    done = len(completed_or_pruned)

    print(f"📊 Status: {done}/{args.num_trials} trials completed/pruned.")

    if done >= args.num_trials:
        print("✅ Study already finished! No new trials needed.")
    else:
        remaining = args.num_trials - done
        print(f"🚀 Running {remaining} more trials...")

        study.optimize(lambda tr: objective(tr, args), n_trials=remaining)

    print("\n" + "=" * 80)
    print("🏆 Best trial:")
    if study.best_trial is not None:
        print(f"  Value: {study.best_value}")
        print(f"  Params: {study.best_params}")
        print(f"  Trial#: {study.best_trial.number}")
    else:
        print("  No successful trials yet.")
    print("=" * 80)


if __name__ == "__main__":
    main()
