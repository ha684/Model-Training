#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI wrapper for evaluation to avoid manual edits.
"""

from __future__ import annotations

print("Evaluation CLI starting...", flush=True)

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

print("Loading datasets library...", flush=True)
from datasets import load_dataset

print("Loading evaluation module...", flush=True)
from qwen3_vl import evaluation

print("Imports complete!", flush=True)


def load_eval_dataset(dataset_path: str, dataset_split: str):
    if dataset_path.startswith("hf://"):
        return load_dataset("parquet", data_files=dataset_path, split=dataset_split)

    path = Path(dataset_path)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            return load_dataset("json", data_files=str(path), split=dataset_split)
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split=dataset_split)
        raise ValueError(f"Unsupported dataset format: {suffix}")

    return load_dataset(dataset_path, split=dataset_split)


def build_report(
    summary: Dict[str, Any],
    model_repo: str,
    model_revision: str,
    run_id: str,
    stage: str,
) -> Dict[str, Any]:
    overall_metadata = summary.get("overall_metadata_accuracy")
    accuracy = overall_metadata / 100.0 if overall_metadata is not None else 0.0
    winrate = accuracy
    p95_latency_ms = summary.get("p95_latency_ms")
    format_pass_rate = summary.get("format_pass_rate", 0.0)

    return {
        "winrate": winrate,
        "accuracy": accuracy,
        "p95_latency_ms": p95_latency_ms,
        "format_pass_rate": format_pass_rate,
        "model_repo": model_repo,
        "model_revision": model_revision,
        "run_id": run_id,
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    print("\n" + "=" * 70, flush=True)
    print("EVALUATION CLI - Starting", flush=True)
    print("=" * 70, flush=True)
    
    parser = argparse.ArgumentParser(description="Evaluation CLI wrapper")
    parser.add_argument("--base_url", required=True, help="vLLM base URL")
    parser.add_argument("--model", required=True, help="Model name for requests")
    parser.add_argument("--dataset", required=True, help="Dataset path")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--model_repo", default=None)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--run_id", default=os.getenv("RUN_ID", ""))
    parser.add_argument("--stage", default=os.getenv("RUN_STAGE", ""))
    args = parser.parse_args()

    print("\nConfiguration:", flush=True)
    print(f"  Base URL: {args.base_url}", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Dataset: {args.dataset}", flush=True)
    print(f"  Output: {args.out}", flush=True)
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}", flush=True)
    if args.concurrency:
        print(f"  Concurrency: {args.concurrency}", flush=True)
    print(flush=True)

    print(f"Loading dataset from {args.dataset}...", flush=True)
    dataset = load_eval_dataset(args.dataset, args.dataset_split)
    print(f"Dataset loaded: {len(dataset)} examples", flush=True)

    print("Configuring evaluation runtime...", flush=True)
    evaluation.configure_runtime(
        base_url_override=args.base_url,
        model_name_override=args.model,
        max_concurrent_requests=args.concurrency,
    )

    print("\nStarting evaluation...", flush=True)
    output = evaluation.run_evaluation(
        dataset=dataset,
        max_samples=args.max_samples,
        max_concurrent_requests=args.concurrency,
    )
    print("\nEvaluation complete!", flush=True)

    print("Building report...", flush=True)
    report = build_report(
        output["summary"],
        model_repo=args.model_repo or args.model,
        model_revision=args.model_revision or "",
        run_id=args.run_id,
        stage=args.stage,
    )

    print(f"Saving results to {args.out}...", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Also save full summary with detailed metrics
    full_summary_path = out_path.parent / f"{out_path.stem}_full_summary.json"
    full_report = {**report, "full_summary": output["summary"]}
    with full_summary_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"Full summary saved to {full_summary_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("EVALUATION COMPLETE - Results:", flush=True)
    print("=" * 70, flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
