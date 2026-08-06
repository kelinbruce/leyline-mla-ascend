#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the complete validation-only DeepSeek-V2-Lite flow on one 310P host."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

LEYLINE_CONFIG = (
    '{"kv_connector":"LeylineConnector","kv_role":"kv_both",'
    '"kv_connector_module_path":'
    '"vllm_ascend.distributed.kv_transfer.leyline.connector",'
    '"kv_load_failure_policy":"recompute"}'
)


def build_server_command(args: argparse.Namespace, *, connector: bool) -> list[str]:
    command = [
        args.vllm_executable,
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        "4",
        "--decode-context-parallel-size",
        "1",
        "--prefill-context-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--block-size",
        "128",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        str(args.max_model_len),
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enforce-eager",
        "--enable-per-request-metrics",
    ]
    if args.model_revision != "local":
        command.extend(("--revision", args.model_revision))
    if args.tokenizer_revision != "local":
        command.extend(("--tokenizer-revision", args.tokenizer_revision))
    if args.tokenizer != args.model:
        command.extend(("--tokenizer", args.tokenizer))
    if connector:
        command.extend(("--kv-transfer-config", LEYLINE_CONFIG))
    return command


def runner_config(args: argparse.Namespace, *, connector: bool) -> dict[str, Any]:
    endpoint = f"http://127.0.0.1:{args.port}"
    arms = (
        {
            "full": endpoint,
            "honest_edited": endpoint,
            "patched_disabled": endpoint,
            "vanilla_apc": endpoint,
            "leyline": endpoint,
        }
        if connector
        else {"cache_off": endpoint}
    )
    return {
        "model": args.served_model_name,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": (
            None if args.tokenizer_revision == "local" else args.tokenizer_revision
        ),
        "trust_remote_code": True,
        "max_tokens": args.max_tokens,
        "prompt_format": "raw",
        "evaluation": {"mode": "reference_prefix", "reference_tokens": 1},
        "arms": arms,
        "npu_memory_command": ["npu-smi", "info"],
    }


def merged_report_passed(report: dict[str, Any]) -> bool:
    admissible = [
        case for case in report.get("cases", []) if case.get("category") == "admissible"
    ]
    return bool(
        admissible
        and all(
            case.get("gates", {}).get("admitted")
            and case.get("gates", {}).get("leyline_accepted")
            and isinstance(
                case.get("arms", {}).get("cache_off", {}).get("output_token_ids"),
                list,
            )
            for case in admissible
        )
    )


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, env=env, check=True)


def _wait_for_server(process: subprocess.Popen[Any], port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"vLLM server exited before readiness with code {returncode}")
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM server did not become ready within {timeout} seconds")


def _stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _run_service_stage(
    args: argparse.Namespace,
    output_dir: Path,
    environment_path: Path,
    *,
    connector: bool,
) -> Path:
    stage = "leyline" if connector else "no_connector"
    config_path = output_dir / f"runner_config.{stage}.json"
    report_path = output_dir / f"correctness.{stage}.json"
    log_path = output_dir / f"server.{stage}.log"
    config_path.write_text(json.dumps(runner_config(args, connector=connector), indent=2) + "\n")
    command = build_server_command(args, connector=connector)
    env = {**os.environ, "ASCEND_RT_VISIBLE_DEVICES": args.devices}
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_server(process, args.port, args.startup_timeout)
            _run(
                [
                    sys.executable,
                    str(args.repo / "benchmarks/leyline/run_validation.py"),
                    "--config",
                    str(config_path),
                    "--environment",
                    str(environment_path),
                    "--output",
                    str(report_path),
                ],
                env=env,
            )
        finally:
            _stop_server(process)
    return report_path


def _run_probes(args: argparse.Namespace, output_dir: Path) -> list[Path]:
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if len(devices) != 4:
        raise ValueError("--devices must identify exactly four 310P devices")
    paths = []
    for rank, device in enumerate(devices):
        path = output_dir / f"310p-probe-rank{rank}.json"
        env = {**os.environ, "ASCEND_RT_VISIBLE_DEVICES": device}
        _run(
            [
                sys.executable,
                str(args.repo / "benchmarks/leyline/probe_310p_mla_ops.py"),
                "--model",
                args.model,
                "--model-revision",
                args.model_revision,
                "--tokenizer-revision",
                args.tokenizer_revision,
                "--tensor-parallel-size",
                "4",
                "--rank",
                str(rank),
                "--vllm-repo",
                str(args.vllm_repo),
                "--output",
                str(path),
            ],
            env=env,
        )
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--vllm-repo", type=Path, default=repo.parent / "vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--served-model-name", default="deepseek-v2-lite")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--vllm-executable", default="vllm")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/leyline-310p-validation"))
    parser.add_argument("--skip-probes", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.vllm_repo = args.vllm_repo.resolve()
    args.tokenizer = args.tokenizer or args.model
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = args.repo / "benchmarks/leyline/runtime_config.310p.example.json"
    environment_path = output_dir / "environment.json"

    if not args.skip_probes:
        _run_probes(args, output_dir)
    _run(
        [
            sys.executable,
            str(args.repo / "benchmarks/leyline/collect_environment.py"),
            "--vllm-ascend",
            str(args.repo),
            "--vllm",
            str(args.vllm_repo),
            "--model",
            args.model,
            "--model-revision",
            args.model_revision,
            "--tokenizer",
            args.tokenizer,
            "--tokenizer-revision",
            args.tokenizer_revision,
            "--runtime-config",
            str(runtime_config),
            "--output",
            str(environment_path),
        ]
    )
    no_connector = _run_service_stage(
        args,
        output_dir,
        environment_path,
        connector=False,
    )
    leyline = _run_service_stage(
        args,
        output_dir,
        environment_path,
        connector=True,
    )
    merged_path = output_dir / "correctness.310p.json"
    _run(
        [
            sys.executable,
            str(args.repo / "benchmarks/leyline/merge_reports.py"),
            str(no_connector),
            str(leyline),
            "--output",
            str(merged_path),
        ]
    )
    report = json.loads(merged_path.read_text())
    summary = {
        "passed": merged_report_passed(report),
        "report": str(merged_path),
        "environment": str(environment_path),
        "probe_reports": [str(path) for path in sorted(output_dir.glob("310p-probe-rank*.json"))],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
