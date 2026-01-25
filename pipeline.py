from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

from prefect import flow, get_run_logger

from pipeline.tasks import (
    _build_train_command,
    _render_repo,
    _write_json,
    deploy_vllm,
    evaluate_grpo_gate,
    evaluate_sft_gate,
    load_config,
    promote_alias_pointer,
    resolve_run_context,
    run_command,
    run_eval,
    send_notification,
    stop_process,
    wait_for_health,
    wait_hf_model_ready,
)
from dotenv import load_dotenv

load_dotenv()

def _build_eval_paths(run_dir: Path) -> Dict[str, str]:
    return {
        "sft": str(run_dir / "sft_eval_report.json"),
        "grpo": str(run_dir / "grpo_eval_report.json"),
    }


def _make_env(token_env: str, token_value: str) -> Dict[str, str]:
    env = os.environ.copy()
    if token_value:
        env[token_env] = token_value
    env["PYTHON_EXECUTABLE"] = sys.executable
    return env


def _summarize_slack(decision: Dict[str, Any], run_id: str) -> str:
    lines = [f"Run `{run_id}` summary:"]
    sft = decision.get("sft")
    if sft:
        lines.append(
            f"- SFT: {'PASS' if sft.get('passed') else 'FAIL'} "
            f"(winrate={sft['metrics'].get('winrate')}, "
            f"format_pass_rate={sft['metrics'].get('format_pass_rate')}, "
            f"p95_latency_ms={sft['metrics'].get('p95_latency_ms')})"
        )
    grpo = decision.get("grpo")
    if grpo:
        lines.append(
            f"- GRPO: {'PASS' if grpo.get('passed') else 'FAIL'} "
            f"(winrate={grpo['metrics'].get('winrate')}, "
            f"format_pass_rate={grpo['metrics'].get('format_pass_rate')}, "
            f"p95_latency_ms={grpo['metrics'].get('p95_latency_ms')})"
        )
    promoted = decision.get("promotion")
    if promoted:
        lines.append(
            f"- Promoted: {promoted.get('repo')} @ {promoted.get('revision')}"
        )
    return "\n".join(lines)


@flow(name="qwen3_vl_pipeline")
def pipeline_flow(config_path: str) -> None:
    logger = get_run_logger()
    config = load_config(config_path)
    run_ctx = resolve_run_context(config)

    run_id = run_ctx["run_id"]
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    hf_cfg = config.get("hf", {})
    token_env = hf_cfg.get("token_env", "HF_TOKEN")
    hf_token = os.getenv(token_env, "")
    org = hf_cfg.get("org_or_user")

    eval_cfg = config.get("eval", {})
    serve_cfg = config.get("serve", {})
    gate_cfg = config.get("gate", {})
    quant_cfg = config.get("quantize", {})
    notify_cfg = config.get("notify", {})

    if not org:
        raise ValueError("hf.org_or_user is required in config")
    if not eval_cfg.get("dataset_path"):
        raise ValueError("eval.dataset_path is required in config")

    sft_repo = _render_repo(hf_cfg.get("sft_repo_template"), run_id)
    grpo_repo = _render_repo(hf_cfg.get("grpo_repo_template"), run_id)
    sft_model_id = f"{org}/{sft_repo}"
    grpo_model_id = f"{org}/{grpo_repo}"
    alias_repo = f"{org}/{hf_cfg.get('prod_alias_repo')}"

    meta = {
        "run": run_ctx,
        "hf": {
            "sft_repo": sft_model_id,
            "grpo_repo": grpo_model_id,
            "alias_repo": alias_repo,
        },
        "config": config,
    }
    _write_json(run_dir / "meta.json", meta)

    env = _make_env(token_env, hf_token)

    eval_paths = _build_eval_paths(run_dir)
    decision: Dict[str, Any] = {"run_id": run_id, "timestamp": run_ctx["timestamp"]}

    train_sft_cfg = dict(config.get("train_sft", {}))
    train_sft_cfg.setdefault("seed", run_ctx["seed"])
    if train_sft_cfg.get("mlflow_run_name") == "auto":
        train_sft_cfg["mlflow_run_name"] = run_id
    sft_cmd = _build_train_command(
        script=train_sft_cfg.get("script", "bash scripts/train_multi_gpu.sh"),
        params=train_sft_cfg,
        hub_model_id=sft_model_id,
        hf_token=hf_token,
    )
    run_command(
        name="train_sft",
        cmd=sft_cmd,
        run_dir=str(run_dir),
        env=env,
        mask_values=[hf_token],
    )
    sft_revision = wait_hf_model_ready(sft_model_id, hf_token)

    sft_deploy = deploy_vllm(sft_model_id, serve_cfg, str(run_dir))
    try:
        served_model = wait_for_health(
            sft_deploy["base_url"],
            serve_cfg.get("health_endpoint", "/v1/models"),
            serve_cfg.get("startup_timeout_sec", 900),
        )
        run_eval(
            base_url=sft_deploy["base_url"],
            model_name=served_model,
            dataset_path=eval_cfg.get("dataset_path"),
            out_path=eval_paths["sft"],
            run_id=run_id,
            stage="sft",
            model_repo=sft_model_id,
            model_revision=sft_revision,
            script=eval_cfg.get("script"),
            max_samples=eval_cfg.get("max_samples"),
            concurrency=eval_cfg.get("concurrency"),
            run_dir=str(run_dir),
        )
    finally:
        stop_process(sft_deploy["process"])

    decision["sft"] = evaluate_sft_gate(
        eval_paths["sft"],
        config["baseline"]["sft_baseline_report"],
        gate_cfg.get("sft", {}),
    )
    _write_json(run_dir / "decision.json", decision)

    if not decision["sft"]["passed"]:
        logger.info("SFT gate failed, stopping pipeline")
        send_notification(_summarize_slack(decision, run_id), notify_cfg)
        return

    train_grpo_cfg = dict(config.get("train_grpo", {}))
    train_grpo_cfg.setdefault("seed", run_ctx["seed"])
    if not train_grpo_cfg.get("model_name"):
        train_grpo_cfg["model_name"] = sft_model_id
    if train_grpo_cfg.get("mlflow_run_name") == "auto":
        train_grpo_cfg["mlflow_run_name"] = run_id

    grpo_cmd = _build_train_command(
        script=train_grpo_cfg.get("script", "bash scripts/train_grpo.sh"),
        params=train_grpo_cfg,
        hub_model_id=grpo_model_id,
        hf_token=hf_token,
    )
    run_command(
        name="train_grpo",
        cmd=grpo_cmd,
        run_dir=str(run_dir),
        env=env,
        mask_values=[hf_token],
    )
    grpo_revision = wait_hf_model_ready(grpo_model_id, hf_token)

    grpo_deploy = deploy_vllm(grpo_model_id, serve_cfg, str(run_dir))
    try:
        served_model = wait_for_health(
            grpo_deploy["base_url"],
            serve_cfg.get("health_endpoint", "/v1/models"),
            serve_cfg.get("startup_timeout_sec", 900),
        )
        run_eval(
            base_url=grpo_deploy["base_url"],
            model_name=served_model,
            dataset_path=eval_cfg.get("dataset_path"),
            out_path=eval_paths["grpo"],
            run_id=run_id,
            stage="grpo",
            model_repo=grpo_model_id,
            model_revision=grpo_revision,
            script=eval_cfg.get("script"),
            max_samples=eval_cfg.get("max_samples"),
            concurrency=eval_cfg.get("concurrency"),
            run_dir=str(run_dir),
        )
    finally:
        stop_process(grpo_deploy["process"])

    decision["grpo"] = evaluate_grpo_gate(
        eval_paths["grpo"],
        eval_paths["sft"],
        config["baseline"]["grpo_baseline_report"],
        gate_cfg.get("grpo", {}),
    )
    _write_json(run_dir / "decision.json", decision)

    if not decision["grpo"]["passed"]:
        logger.info("GRPO gate failed, stopping pipeline")
        send_notification(_summarize_slack(decision, run_id), notify_cfg)
        return

    pointer_path = promote_alias_pointer(
        alias_repo=alias_repo,
        target_repo=grpo_model_id,
        revision=grpo_revision,
        token=hf_token,
        run_id=run_id,
    )
    decision["promotion"] = {
        "repo": grpo_model_id,
        "revision": grpo_revision,
        "pointer_path": pointer_path,
    }

    if quant_cfg.get("enabled"):
        quant_script = quant_cfg.get("script")
        quant_cmd = shlex.split(quant_script)
        if quant_cfg.get("method"):
            quant_cmd.extend(["--method", str(quant_cfg.get("method"))])
        run_command(
            name="quantize",
            cmd=quant_cmd,
            run_dir=str(run_dir),
            env=env,
            mask_values=[hf_token],
        )

    _write_json(run_dir / "decision.json", decision)
    meta.update(
        {
            "artifacts": {
                "sft_report": eval_paths["sft"],
                "grpo_report": eval_paths["grpo"],
                "decision": str(run_dir / "decision.json"),
            },
            "revisions": {
                "sft": sft_revision,
                "grpo": grpo_revision,
            },
            "ports": {
                "sft": sft_deploy["port"],
                "grpo": grpo_deploy["port"],
            },
        }
    )
    _write_json(run_dir / "meta.json", meta)

    send_notification(_summarize_slack(decision, run_id), notify_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL orchestration pipeline")
    parser.add_argument(
        "--config",
        default="configs/run.yaml",
        help="Path to pipeline YAML config (default: configs/run.yaml)",
    )
    args = parser.parse_args()
    pipeline_flow(args.config)


if __name__ == "__main__":
    main()
