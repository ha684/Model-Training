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
    pause_between_tasks,
    promote_alias_pointer,
    resolve_run_context,
    run_command,
    run_eval,
    send_notification,
    stop_process,
    wait_for_health,
    wait_hf_model_ready,
    _kill_existing_vllm_processes,
    _cleanup_gpu_memory,
)
from dotenv import load_dotenv

load_dotenv()

def _build_eval_paths(run_dir: Path) -> Dict[str, str]:
    return {
        "sft": str(run_dir / "sft_eval_report.json"),
        "grpo": str(run_dir / "grpo_eval_report.json"),
        "sft_full": str(run_dir / "sft_eval_report_full_summary.json"),
        "grpo_full": str(run_dir / "grpo_eval_report_full_summary.json"),
    }


def _make_env(token_env: str, token_value: str) -> Dict[str, str]:
    env = os.environ.copy()
    if token_value:
        env[token_env] = token_value
    env["PYTHON_EXECUTABLE"] = sys.executable
    return env


def _load_eval_report(report_path: str) -> Dict[str, Any]:
    """Load full evaluation report including detailed metrics."""
    import json
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_report_summary(report: Dict[str, Any], stage_name: str, passed: bool) -> str:
    """Format detailed evaluation metrics for notification."""
    lines = [f"\n--- {stage_name.upper()}: {'✓ PASS' if passed else '✗ FAIL'} ---"]
    
    # Check if we have full_summary
    full_summary = report.get("full_summary", {})
    
    if full_summary:
        # Display detailed metrics from full summary
        total = full_summary.get("total_examples", 0)
        evaluated = full_summary.get("evaluated_examples", 0)
        successful = full_summary.get("successful_parses", 0)
        
        lines.append(f"Examples: {evaluated}/{total} evaluated")
        lines.append(f"Successful Parses: {successful}/{evaluated} ({successful/evaluated*100:.1f}%)" if evaluated > 0 else "Successful Parses: 0/0")
        
        # Overall metadata accuracy
        overall_meta = full_summary.get("overall_metadata_accuracy")
        if overall_meta is not None:
            lines.append(f"Overall Metadata Accuracy: {overall_meta:.1f}%")
        
        # Table rows accuracy
        table_acc = full_summary.get("table_rows_accuracy", 0)
        lines.append(f"Table Rows Accuracy: {table_acc:.1f}%")
        
        # Table structure detection
        structure_acc = full_summary.get("table_structure_accuracy", 0)
        if structure_acc > 0:
            lines.append(f"Table Structure Detection: {structure_acc:.1f}%")
        
        # Per-language breakdown
        lang_breakdown = full_summary.get("language_breakdown", {})
        if lang_breakdown:
            lines.append("\nPer-Language Results:")
            for lang in sorted(lang_breakdown.keys()):
                stats = lang_breakdown[lang]
                lang_total = stats.get("total", 0)
                lang_successful = stats.get("successful", 0)
                lang_meta_acc = stats.get("metadata_accuracy", {})
                lang_rows_acc = stats.get("table_rows_accuracy", 0)
                
                if lang_total > 0:
                    success_pct = lang_successful / lang_total * 100
                    overall_meta_lang = sum(lang_meta_acc.values()) / len(lang_meta_acc) if lang_meta_acc else 0
                    lines.append(
                        f"  • {lang.upper()}: {lang_successful}/{lang_total} ({success_pct:.1f}%) "
                        f"| Meta: {overall_meta_lang:.1f}% | Rows: {lang_rows_acc:.1f}%"
                    )
    else:
        # Fallback to basic metrics
        winrate = report.get("winrate", 0)
        accuracy = report.get("accuracy", winrate)
        lines.append(f"Overall Accuracy: {accuracy * 100:.1f}%")
        
        format_pass_rate = report.get("format_pass_rate", 0)
        lines.append(f"Format Pass Rate: {format_pass_rate * 100:.1f}%")
    
    return "\n".join(lines)


def _summarize_slack(decision: Dict[str, Any], run_id: str, eval_paths: Dict[str, str]) -> str:
    """Generate detailed notification summary with evaluation metrics."""
    lines = [f"🚀 Run `{run_id}` Summary"]
    lines.append("=" * 50)
    
    sft = decision.get("sft")
    if sft:
        try:
            # Try to load full summary first, fallback to basic report
            sft_report = None
            try:
                sft_report = _load_eval_report(eval_paths.get("sft_full", ""))
            except Exception:
                sft_report = _load_eval_report(eval_paths.get("sft", ""))
            
            sft_summary = _format_report_summary(sft_report, "SFT", sft.get("passed", False))
            lines.append(sft_summary)
            
            # Add delta info if available
            deltas = sft.get("deltas", {})
            if deltas.get("overall_metadata_accuracy_delta") is not None:
                lines.append(f"Δ Metadata vs Baseline: {deltas['overall_metadata_accuracy_delta']:+.1f}%")
            if deltas.get("table_rows_accuracy_delta") is not None:
                lines.append(f"Δ Rows vs Baseline: {deltas['table_rows_accuracy_delta']:+.1f}%")
        except Exception as e:
            # Fallback to basic metrics
            lines.append(
                f"\n--- SFT: {'✓ PASS' if sft.get('passed') else '✗ FAIL'} ---\n"
                f"Metadata Accuracy: {sft['metrics'].get('overall_metadata_accuracy', 0):.1f}%\n"
                f"Rows Accuracy: {sft['metrics'].get('table_rows_accuracy', 0):.1f}%"
            )
    
    grpo = decision.get("grpo")
    if grpo:
        try:
            # Try to load full summary first, fallback to basic report
            grpo_report = None
            try:
                grpo_report = _load_eval_report(eval_paths.get("grpo_full", ""))
            except Exception:
                grpo_report = _load_eval_report(eval_paths.get("grpo", ""))
            
            grpo_summary = _format_report_summary(grpo_report, "GRPO", grpo.get("passed", False))
            lines.append(grpo_summary)
            
            # Add delta info
            deltas = grpo.get("deltas", {})
            if deltas.get("grpo_vs_sft_metadata_delta") is not None:
                lines.append(f"Δ Metadata vs SFT: {deltas['grpo_vs_sft_metadata_delta']:+.1f}%")
            if deltas.get("grpo_vs_baseline_metadata_delta") is not None:
                lines.append(f"Δ Metadata vs Baseline: {deltas['grpo_vs_baseline_metadata_delta']:+.1f}%")
        except Exception as e:
            # Fallback to basic metrics
            lines.append(
                f"\n--- GRPO: {'✓ PASS' if grpo.get('passed') else '✗ FAIL'} ---\n"
                f"Metadata Accuracy: {grpo['metrics'].get('overall_metadata_accuracy', 0):.1f}%\n"
                f"Rows Accuracy: {grpo['metrics'].get('table_rows_accuracy', 0):.1f}%"
            )
    
    promoted = decision.get("promotion")
    if promoted:
        lines.append(f"\n🎉 Promoted: {promoted.get('repo')} @ {promoted.get('revision')[:7]}")
    
    return "\n".join(lines)


@flow(name="qwen3_vl_pipeline", persist_result=True)
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
        run_id=run_id,
        env=env,
        mask_values=[hf_token],
    )
    sft_revision = wait_hf_model_ready(sft_model_id, hf_token)

    pause_between_tasks("Allowing resources to stabilize after SFT training", 10)

    sft_deploy = deploy_vllm(sft_model_id, serve_cfg, str(run_dir))
    try:
        # FIXED: Pass run_id to ensure fresh health check every run
        served_model = wait_for_health(
            sft_deploy["base_url"],
            serve_cfg.get("health_endpoint", "/v1/models"),
            serve_cfg.get("startup_timeout_sec", 900),
            stage="sft",
            run_id=run_id,
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

    pause_between_tasks("Allowing cleanup after stopping vLLM", 15)

    decision["sft"] = evaluate_sft_gate(
        eval_paths["sft"],
        config["baseline"]["sft_baseline_report"],
        gate_cfg.get("sft", {}),
    )
    _write_json(run_dir / "decision.json", decision)

    if not decision["sft"]["passed"]:
        logger.info("SFT gate failed, stopping pipeline")
        send_notification(_summarize_slack(decision, run_id, eval_paths), notify_cfg)
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
        run_id=run_id,
        env=env,
        mask_values=[hf_token],
    )
    
    grpo_revision = wait_hf_model_ready(grpo_model_id, hf_token)
    
    logger.info("Cleaning up before GRPO deployment...")
    _kill_existing_vllm_processes()
    _cleanup_gpu_memory()
    pause_between_tasks("Allowing resources to stabilize after GRPO training", 15)

    grpo_deploy = deploy_vllm(grpo_model_id, serve_cfg, str(run_dir))
    try:
        served_model = wait_for_health(
            grpo_deploy["base_url"],
            serve_cfg.get("health_endpoint", "/v1/models"),
            serve_cfg.get("startup_timeout_sec", 900),
            stage="grpo",
            run_id=run_id,
        )
        
        pause_between_tasks("Ensuring vLLM is fully ready before GRPO evaluation", 5)
        
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

    pause_between_tasks("Allowing cleanup after stopping vLLM", 5)

    decision["grpo"] = evaluate_grpo_gate(
        eval_paths["grpo"],
        eval_paths["sft"],
        config["baseline"]["grpo_baseline_report"],
        gate_cfg.get("grpo", {}),
    )
    _write_json(run_dir / "decision.json", decision)

    if not decision["grpo"]["passed"]:
        logger.info("GRPO gate failed, stopping pipeline")
        send_notification(_summarize_slack(decision, run_id, eval_paths), notify_cfg)
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
            run_id=run_id,
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

    send_notification(_summarize_slack(decision, run_id, eval_paths), notify_cfg)


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