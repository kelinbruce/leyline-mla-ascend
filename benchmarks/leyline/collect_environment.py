#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Collect the exact software, model, runtime, and 910B environment manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": args, "error": str(exc)}
    return {
        "command": args,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def git_manifest(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "commit": command(["git", "rev-parse", "HEAD"], path),
        "status": command(["git", "status", "--short"], path),
    }


def distribution_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_manifest() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"error": str(exc)}
    result: dict[str, Any] = {"version": torch.__version__}
    npu = getattr(torch, "npu", None)
    if npu is not None:
        try:
            count = npu.device_count()
            result["npu_count"] = count
            result["npu_names"] = [npu.get_device_name(index) for index in range(count)]
        except Exception as exc:  # Hardware discovery must not erase the rest of the manifest.
            result["npu_error"] = str(exc)
    return result


def cann_version_files() -> dict[str, str]:
    root = Path("/usr/local/Ascend/ascend-toolkit/latest")
    result = {}
    for name in ("version.info", "version.cfg", "x86_64-linux/ascend_toolkit_install.info"):
        path = root / name
        try:
            result[str(path)] = path.read_text(errors="replace").strip()
        except OSError:
            continue
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--vllm-ascend", type=Path, default=repo)
    parser.add_argument("--vllm", type=Path, default=repo.parent / "vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {
        "schema_version": 1,
        "created_unix": time.time(),
        "platform": platform.platform(),
        "python": {"version": sys.version, "executable": sys.executable},
        "repositories": {
            "vllm": git_manifest(args.vllm),
            "vllm_ascend": git_manifest(args.vllm_ascend),
        },
        "packages": distribution_versions(
            ["vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "triton-ascend"]
        ),
        "torch": torch_manifest(),
        "model": {
            "name": args.model,
            "revision": args.model_revision,
            "tokenizer": args.tokenizer,
            "tokenizer_revision": args.tokenizer_revision,
        },
        "runtime_config": json.loads(args.runtime_config.read_text()),
        "npu_smi": command(["npu-smi", "info"]),
        "topology": command(["npu-smi", "info", "-t", "topo"]),
        "cann_installation": {
            "listing": command(["ls", "-1", "/usr/local/Ascend/ascend-toolkit/latest"]),
            "version_files": cann_version_files(),
            "environment": {
                name: os.environ.get(name)
                for name in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES")
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
