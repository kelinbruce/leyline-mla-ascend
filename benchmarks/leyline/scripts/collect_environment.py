#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Collect the exact software, model, runtime, and 910B environment manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


IDENTITY_FILES = (
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "model.safetensors.index.json",
)
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


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


def module_provenance(names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
            result[name] = {"module_file": str(path) if path else None}
        except Exception as exc:
            result[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def cann_version_files() -> dict[str, str]:
    configured = os.environ.get("ASCEND_HOME_PATH")
    roots = [Path(configured)] if configured else []
    roots.append(Path("/usr/local/Ascend/ascend-toolkit/latest"))
    result = {}
    names = (
        "version.info",
        "version.cfg",
        "ascend_toolkit_install.info",
        "x86_64-linux/ascend_toolkit_install.info",
        "aarch64-linux/ascend_toolkit_install.info",
    )
    for root in dict.fromkeys(path.resolve() for path in roots if path.exists()):
        for name in names:
            path = root / name
            try:
                result[str(path)] = path.read_text(errors="replace").strip()
            except OSError:
                continue
    return result


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_json(path: Path) -> dict[str, Any] | None:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    keys = (
        "model_type",
        "architectures",
        "torch_dtype",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "rope_scaling",
        "chat_template",
        "tokenizer_class",
    )
    return {key: document[key] for key in keys if key in document}


def artifact_manifest(location: str, revision: str) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(location).expanduser()
    manifest: dict[str, Any] = {
        "requested": location,
        "revision": revision,
        "local": path.exists(),
    }
    if not path.exists():
        manifest["hashing_seconds"] = time.perf_counter() - started
        return manifest
    resolved = path.resolve()
    manifest["resolved_path"] = str(resolved)
    if resolved.is_file():
        files = [resolved]
        root = resolved.parent
    else:
        root = resolved
        files = [root / name for name in IDENTITY_FILES if (root / name).is_file()]
        files.extend(
            item
            for item in sorted(root.iterdir())
            if item.is_file() and item.suffix in WEIGHT_SUFFIXES
        )
    unique_files = list(dict.fromkeys(files))
    manifest["artifacts"] = [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in unique_files
    ]
    manifest["selected_configuration"] = {
        path.name: selected
        for path in unique_files
        if path.suffix == ".json" and (selected := _selected_json(path)) is not None
    }
    tokenizer_config = manifest["selected_configuration"].get("tokenizer_config.json", {})
    manifest["chat_template_present"] = bool(tokenizer_config.get("chat_template"))
    manifest["hashing_seconds"] = time.perf_counter() - started
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
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
        "schema_version": 2,
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
        "imported_modules": module_provenance(["vllm", "vllm_ascend"]),
        "torch": torch_manifest(),
        "model": {
            "name": args.model,
            "revision": args.model_revision,
            "tokenizer": args.tokenizer,
            "tokenizer_revision": args.tokenizer_revision,
            "checkpoint_identity": artifact_manifest(args.model, args.model_revision),
            "tokenizer_identity": artifact_manifest(args.tokenizer, args.tokenizer_revision),
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
