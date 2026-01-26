from __future__ import annotations

import json
import hashlib
import threading
import os
import shlex
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import urllib.request
import urllib.error

import yaml
from huggingface_hub import HfApi, create_repo
from prefect import get_run_logger, task


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return _read_json(path)
    except FileNotFoundError:
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _mask_values(text: str, secrets: Iterable[str]) -> str:
    masked = text
    for secret in secrets:
        if secret:
            masked = masked.replace(secret, "***")
    return masked


def _format_command(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _append_command_log(
    run_dir: Path, cmd: List[str], secret_values: Iterable[str]
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    command_text = _mask_values(_format_command(cmd), secret_values)
    with (run_dir / "commands.log").open("a", encoding="utf-8") as f:
        f.write(f"{command_text}\n")


def _select_port(start: int, end: int, host: str) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found in range {start}-{end}")


@task(persist_result=True)
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@task(persist_result=True)
def resolve_run_context(config: Dict[str, Any]) -> Dict[str, Any]:
    run_cfg = config.get("run", {})
    timestamp = run_cfg.get("timestamp", "auto")
    git_commit = run_cfg.get("git_commit", "auto")

    if timestamp == "auto":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if git_commit == "auto":
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            git_commit = result.stdout.strip() or "nogit"
        except Exception:
            git_commit = "nogit"

    run_id = run_cfg.get("run_id", "auto")
    if run_id == "auto":
        run_id = f"{timestamp}-{git_commit}"

    return {
        "run_name": run_cfg.get("name", "qwen3_vl_pipeline"),
        "seed": run_cfg.get("seed", 3407),
        "timestamp": timestamp,
        "git_commit": git_commit,
        "run_id": run_id,
    }


def _render_repo(template: str, run_id: str) -> str:
    return template.format(run_id=run_id)


def _build_train_command(
    script: str,
    params: Dict[str, Any],
    hub_model_id: str,
    hf_token: Optional[str],
) -> List[str]:
    cmd = shlex.split(script)
    arg_map = {
        "num_gpus": "--num_gpus",
        "model_name": "--model_name",
        "max_seq_length": "--max_seq_length",
        "dataset_name": "--dataset_name",
        "lora_r": "--lora_r",
        "lora_alpha": "--lora_alpha",
        "lora_dropout": "--lora_dropout",
        "batch_size": "--batch_size",
        "eval_batch_size": "--eval_batch_size",
        "grad_accum": "--grad_accum",
        "epochs": "--epochs",
        "max_steps": "--max_steps",
        "learning_rate": "--learning_rate",
        "warmup_steps": "--warmup_steps",
        "weight_decay": "--weight_decay",
        "eval_steps": "--eval_steps",
        "logging_steps": "--logging_steps",
        "output_dir": "--output_dir",
        "model_save_path": "--model_save_path",
        "report_to": "--report_to",
        "mlflow_uri": "--mlflow_uri",
        "mlflow_experiment_name": "--mlflow_experiment_name",
        "mlflow_run_name": "--mlflow_run_name",
        "seed": "--seed",
        "metric_file": "--metric_file",
        "dataset_cache_dir": "--dataset_cache_dir",
        "num_generations": "--num_generations",
        "max_prompt_length": "--max_prompt_length",
        "max_completion_length": "--max_completion_length",
        "max_grad_norm": "--max_grad_norm",
        "adam_beta1": "--adam_beta1",
        "adam_beta2": "--adam_beta2",
        "use_gspo": "--use_gspo",
    }

    for key, flag in arg_map.items():
        if key not in params or params[key] is None:
            continue
        value = params[key]
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    if params.get("save_model"):
        cmd.append("--save_model")

    if params.get("push_to_hub"):
        cmd.append("--push_to_hub")

    cmd.extend(["--hub_model_id", hub_model_id])

    if hf_token:
        cmd.extend(["--hf_token", hf_token])

    return cmd


def _execute_command(
    name: str,
    cmd: List[str],
    run_dir: str,
    env: Optional[Dict[str, str]] = None,
    mask_values: Optional[List[str]] = None,
) -> None:
    logger = get_run_logger()
    run_path = Path(run_dir)
    mask_values = mask_values or []
    _append_command_log(run_path, cmd, mask_values)
    logger.info("Running %s", name)
    logger.info(_mask_values(_format_command(cmd), mask_values))

    # Ensure unbuffered output
    exec_env = os.environ.copy() if env is None else env.copy()
    exec_env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffering
        env=exec_env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info("%s", line.rstrip())
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{name} failed with exit code {return_code}")


def _command_cache_key(task_run_context: Any, parameters: Dict[str, Any]) -> str:
    payload = {
        "run_id": parameters.get("run_id"),
        "name": parameters.get("name"),
        "cmd": parameters.get("cmd"),
        "run_dir": parameters.get("run_dir"),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@task(retries=1, retry_delay_seconds=60, persist_result=True, cache_key_fn=_command_cache_key)
def run_command(
    name: str,
    cmd: List[str],
    run_dir: str,
    run_id: str,
    env: Optional[Dict[str, str]] = None,
    mask_values: Optional[List[str]] = None,
) -> None:
    _execute_command(name, cmd, run_dir, env=env, mask_values=mask_values)


@task(retries=20, retry_delay_seconds=30, persist_result=True)
def wait_hf_model_ready(repo_id: str, token: Optional[str]) -> str:
    logger = get_run_logger()
    api = HfApi(token=token)
    logger.info("Checking model availability: %s", repo_id)
    info = api.model_info(repo_id)
    return info.sha

def _tail_file(path: Path, n_lines: int = 200) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except Exception as e:
        return f"<failed to read log tail: {e}>"


def _cleanup_gpu_memory() -> None:
    """Attempt to clear GPU memory cache."""
    logger = get_run_logger()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cache cleared")
    except Exception as e:
        logger.warning("Failed to clear GPU cache: %s", e)


def _kill_existing_vllm_processes() -> None:
    """Kill any existing vLLM processes to free up resources."""
    logger = get_run_logger()
    try:
        result = subprocess.run(
            ["pkill", "-f", "vllm.entrypoints"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Killed existing vLLM processes")
            time.sleep(3)  # Wait for processes to fully terminate
        else:
            logger.info("No existing vLLM processes found")
    except Exception as e:
        logger.warning("Failed to check/kill vLLM processes: %s", e)
    
def _stream_process_output(
    process: subprocess.Popen, log_path: Path, logger: Optional[Any] = None
) -> None:
    if logger is None:
        import logging

        logger = logging.getLogger("prefect")
    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8") as log_file:
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            logger.info("vLLM | %s", line.rstrip())


@task(persist_result=False)
def deploy_vllm(repo_id: str, serve_cfg: Dict[str, Any], run_dir: str) -> Dict[str, Any]:
    logger = get_run_logger()

    # Cleanup before deploying
    logger.info("Preparing to deploy vLLM...")
    _kill_existing_vllm_processes()
    _cleanup_gpu_memory()
    time.sleep(2)  # Give system time to stabilize

    host = serve_cfg.get("host", "127.0.0.1")
    port_start, port_end = serve_cfg.get("port_range", [8001, 8099])
    port = _select_port(int(port_start), int(port_end), host)
    base_url = f"http://{host}:{port}"

    log_path = Path(run_dir) / "vllm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "vllm", "serve", repo_id,
        "--tensor-parallel-size", str(serve_cfg.get("tensor_parallel_size", 1)),
        "--pipeline-parallel-size", str(serve_cfg.get("pipeline_parallel_size", 1)),
        "--trust-remote-code",
        "--max-model-len", str(serve_cfg.get("max_model_len", 8192)),
        "--gpu-memory-utilization", str(serve_cfg.get("gpu_memory_utilization", 0.9)),
        "--host", host,
        "--port", str(port),
    ]

    attention_backend = serve_cfg.get("attention_backend")
    if attention_backend:
        cmd.extend(["--attention-config.backend", attention_backend])

    env = os.environ.copy()
    hf_token = os.getenv(serve_cfg.get("token_env", "HF_TOKEN"), "")
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env.setdefault("HF_TOKEN", hf_token)

    env["HF_HUB_ENABLE_HF_TRANSFER"] = str(serve_cfg.get("hf_hub_enable_hf_transfer", 0))

    logger.info("Starting vLLM on %s", base_url)
    cmd_str = ' '.join(shlex.quote(str(arg)) for arg in cmd)
    logger.info("Command: %s", _mask_values(cmd_str, [hf_token]))

    _append_command_log(Path(run_dir), cmd, [])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    stream_thread = threading.Thread(
        target=_stream_process_output, args=(process, log_path, logger), daemon=True
    )
    stream_thread.start()

    time.sleep(2)
    rc = process.poll()
    if rc is not None:
        tail = _tail_file(log_path, 200)
        raise RuntimeError(
            f"vLLM exited immediately (rc={rc}). Tail of log:\n{tail}"
        )

    return {"process": process, "base_url": base_url, "port": port, "log_path": str(log_path)}

def _http_get(url: str, timeout: int = 10) -> Tuple[int, str]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.getcode(), body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body


@task(persist_result=True)
def wait_for_health(base_url: str, health_endpoint: str, timeout_sec: int) -> Optional[str]:
    logger = get_run_logger()
    deadline = time.time() + timeout_sec
    url = f"{base_url}{health_endpoint}"
    last_error = None
    attempt = 0

    logger.info("Waiting for vLLM health check at %s (timeout: %ds)", url, timeout_sec)

    while time.time() < deadline:
        attempt += 1
        remaining = int(deadline - time.time())
        
        try:
            code, body = _http_get(url, timeout=10)

            if health_endpoint.rstrip("/") == "/health":
                if code == 200:
                    logger.info("✓ vLLM alive (GET %s -> 200)", url)
                    return None
                last_error = f"GET {url} -> HTTP {code} body={body[:200]!r}"
                logger.info("Health check attempt %d: not ready yet (HTTP %d), retrying in 5s... (%ds remaining)", 
                           attempt, code, remaining)

            else:
                if code != 200:
                    last_error = f"GET {url} -> HTTP {code} body={body[:200]!r}"
                    logger.info("Health check attempt %d: not ready yet (HTTP %d), retrying in 5s... (%ds remaining)", 
                               attempt, code, remaining)
                else:
                    payload = json.loads(body) if body.strip() else {}
                    data = payload.get("data", [])
                    if data:
                        model_id = data[0].get("id", "")
                        logger.info("✓ vLLM ready with model id %s (attempt %d)", model_id, attempt)
                        return model_id
                    last_error = f"No model data yet: {payload}"
                    logger.info("Health check attempt %d: model not loaded yet, retrying in 5s... (%ds remaining)", 
                               attempt, remaining)

        except Exception as exc:
            last_error = str(exc)
            logger.info("Health check attempt %d: connection failed (%s), retrying in 5s... (%ds remaining)", 
                       attempt, str(exc)[:100], remaining)

        time.sleep(5)

    raise TimeoutError(f"vLLM health check failed for {url}: {last_error}")

@task(persist_result=False)
def stop_process(process: subprocess.Popen) -> None:
    logger = get_run_logger()
    if process.poll() is not None:
        return
    logger.info("Stopping vLLM process")
    process.terminate()
    try:
        process.wait(timeout=30)
        logger.info("vLLM process stopped successfully")
    except subprocess.TimeoutExpired:
        logger.warning("vLLM did not stop, killing")
        process.kill()
    
    # Extra cleanup after stopping
    time.sleep(2)
    _cleanup_gpu_memory()


@task
def pause_between_tasks(reason: str, seconds: int = 5) -> None:
    """Add a pause between tasks to allow resource cleanup."""
    logger = get_run_logger()
    logger.info("Pausing for %ds: %s", seconds, reason)
    time.sleep(seconds)
    logger.info("Resuming pipeline")


@task(persist_result=True)
def run_eval(
    base_url: str,
    model_name: str,
    dataset_path: str,
    out_path: str,
    run_id: str,
    stage: str,
    model_repo: str,
    model_revision: str,
    script: Optional[str] = None,
    max_samples: Optional[int] = None,
    concurrency: Optional[int] = None,
    run_dir: Optional[str] = None,
) -> None:
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("Starting evaluation for stage: %s", stage)
    logger.info("Base URL: %s", base_url)
    logger.info("Model: %s", model_name)
    logger.info("Dataset: %s", dataset_path)
    if max_samples:
        logger.info("Max samples: %d", max_samples)
    if concurrency:
        logger.info("Concurrency: %d", concurrency)
    logger.info("=" * 60)
    
    if script:
        cmd = shlex.split(script)
    else:
        cmd = [
            os.environ.get("PYTHON_EXECUTABLE", "python"),
            "-u",  # Unbuffered output
            "-m",
            "qwen3_vl.evaluation_cli",
        ]
    cmd.extend(
        [
            "--base_url",
            base_url,
            "--model",
            model_name,
            "--dataset",
            dataset_path,
            "--out",
            out_path,
            "--run_id",
            run_id,
            "--stage",
            stage,
            "--model_repo",
            model_repo,
            "--model_revision",
            model_revision,
        ]
    )
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])
    if concurrency is not None:
        cmd.extend(["--concurrency", str(concurrency)])

    _execute_command(name=f"eval-{stage}", cmd=cmd, run_dir=run_dir or ".")


def _evaluate_gate(
    checks: List[Tuple[str, bool]],
    require_all: bool,
) -> Tuple[bool, List[str]]:
    failures = [name for name, passed in checks if not passed]
    if require_all:
        return len(failures) == 0, failures
    return len(failures) < len(checks), failures


@task(persist_result=True)
def evaluate_sft_gate(
    sft_report_path: str,
    baseline_report_path: str,
    gate_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    sft_report = _read_json(Path(sft_report_path))
    baseline = _read_json_optional(Path(baseline_report_path))
    baseline_missing = baseline is None
    if baseline_missing:
        logger = get_run_logger()
        logger.warning(
            "Baseline report not found at %s; using current SFT as baseline",
            baseline_report_path,
        )
        baseline = sft_report

    winrate_delta = sft_report["winrate"] - baseline["winrate"]
    min_winrate = gate_cfg.get("min_winrate_delta_vs_sft_baseline", 0)
    min_format = gate_cfg.get("min_format_pass_rate", 0)
    max_latency = gate_cfg.get("max_p95_latency_ms", 0)
    checks = [
        (
            "winrate_delta_vs_sft_baseline",
            winrate_delta >= min_winrate if winrate_delta is not None else False,
        ),
        (
            "format_pass_rate",
            sft_report["format_pass_rate"] >= min_format
            if sft_report.get("format_pass_rate") is not None
            else False,
        ),
        (
            "p95_latency_ms",
            sft_report["p95_latency_ms"] <= max_latency
            if sft_report.get("p95_latency_ms") is not None
            else False,
        ),
    ]

    passed, failures = _evaluate_gate(checks, gate_cfg.get("require_all", True))
    return {
        "passed": passed,
        "failures": failures,
        "deltas": {"winrate_delta": winrate_delta},
        "baseline_missing": baseline_missing,
        "thresholds": gate_cfg,
        "metrics": {
            "winrate": sft_report["winrate"],
            "format_pass_rate": sft_report["format_pass_rate"],
            "p95_latency_ms": sft_report["p95_latency_ms"],
        },
    }


@task(persist_result=True)
def evaluate_grpo_gate(
    grpo_report_path: str,
    sft_report_path: str,
    baseline_report_path: str,
    gate_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    grpo_report = _read_json(Path(grpo_report_path))
    sft_report = _read_json(Path(sft_report_path))
    baseline = _read_json(Path(baseline_report_path))

    grpo_vs_sft_delta = grpo_report["winrate"] - sft_report["winrate"]
    grpo_vs_baseline_delta = grpo_report["winrate"] - baseline["winrate"]

    min_vs_sft = gate_cfg.get("min_winrate_delta_vs_sft_new", 0)
    min_vs_baseline = gate_cfg.get("min_winrate_delta_vs_grpo_baseline", 0)
    min_format = gate_cfg.get("min_format_pass_rate", 0)
    max_latency = gate_cfg.get("max_p95_latency_ms", 0)
    checks = [
        (
            "winrate_delta_vs_sft_new",
            grpo_vs_sft_delta >= min_vs_sft if grpo_vs_sft_delta is not None else False,
        ),
        (
            "winrate_delta_vs_grpo_baseline",
            grpo_vs_baseline_delta >= min_vs_baseline
            if grpo_vs_baseline_delta is not None
            else False,
        ),
        (
            "format_pass_rate",
            grpo_report["format_pass_rate"] >= min_format
            if grpo_report.get("format_pass_rate") is not None
            else False,
        ),
        (
            "p95_latency_ms",
            grpo_report["p95_latency_ms"] <= max_latency
            if grpo_report.get("p95_latency_ms") is not None
            else False,
        ),
    ]

    passed, failures = _evaluate_gate(checks, gate_cfg.get("require_all", True))
    return {
        "passed": passed,
        "failures": failures,
        "deltas": {
            "grpo_vs_sft_delta": grpo_vs_sft_delta,
            "grpo_vs_baseline_delta": grpo_vs_baseline_delta,
        },
        "thresholds": gate_cfg,
        "metrics": {
            "winrate": grpo_report["winrate"],
            "format_pass_rate": grpo_report["format_pass_rate"],
            "p95_latency_ms": grpo_report["p95_latency_ms"],
        },
    }


@task(persist_result=True)
def promote_alias_pointer(
    alias_repo: str,
    target_repo: str,
    revision: str,
    token: Optional[str],
    run_id: str,
) -> str:
    logger = get_run_logger()
    create_repo(alias_repo, repo_type="model", exist_ok=True, token=token)
    pointer = {
        "repo_id": target_repo,
        "revision": revision,
        "run_id": run_id,
        "timestamp": _now_utc(),
    }
    pointer_path = Path("runs") / run_id / "MODEL_POINTER.json"
    _write_json(pointer_path, pointer)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(pointer_path),
        path_in_repo="MODEL_POINTER.json",
        repo_id=alias_repo,
        repo_type="model",
        commit_message=f"Update pointer for {run_id}",
    )
    logger.info("Promoted %s at %s", target_repo, revision)
    return "MODEL_POINTER.json"


def _send_email(message: str, config: Dict[str, Any]) -> None:
    """Send email notification using SMTP."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_host = config.get("smtp_host", "smtp.gmail.com")
    smtp_port = config.get("smtp_port", 587)
    smtp_user = os.getenv(config.get("smtp_user_env", "SMTP_USER"))
    smtp_password = os.getenv(config.get("smtp_password_env", "SMTP_PASSWORD"))
    from_email = config.get("from_email", smtp_user)
    to_emails = config.get("to_emails", [])

    if not smtp_user or not smtp_password or not to_emails:
        return

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = ", ".join(to_emails)
    msg["Subject"] = config.get("subject", "Pipeline Notification")
    msg.attach(MIMEText(message, "plain"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def _send_google_chat(message: str, webhook_url: str) -> None:
    """Send Google Chat notification."""
    import urllib.request

    payload = json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10):
        return


def _send_slack(message: str, webhook_url: str) -> None:
    """Send Slack notification."""
    import urllib.request

    payload = json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10):
        return


@task(persist_result=True)
def send_notification(message: str, notify_cfg: Dict[str, Any]) -> None:
    """Send notification via configured backend (email, google_chat, slack)."""
    logger = get_run_logger()
    backend = notify_cfg.get("backend", "slack")

    try:
        if backend == "email":
            _send_email(message, notify_cfg.get("email", {}))
            logger.info("Email notification sent")
        elif backend == "google_chat":
            webhook_url = os.getenv(
                notify_cfg.get("google_chat_webhook_env", "GOOGLE_CHAT_WEBHOOK_URL")
            )
            if webhook_url:
                _send_google_chat(message, webhook_url)
                logger.info("Google Chat notification sent")
        elif backend == "slack":
            webhook_url = os.getenv(
                notify_cfg.get("slack_webhook_env", "SLACK_WEBHOOK_URL")
            )
            if webhook_url:
                _send_slack(message, webhook_url)
                logger.info("Slack notification sent")
        else:
            logger.warning("Unknown notification backend: %s", backend)
    except Exception as exc:
        logger.warning("Notification failed: %s", exc)

