#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run controlled Leyline semantic and performance validation arms."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class PromptPlan:
    full: list[int]
    edited: list[int]
    delete_start: int
    delete_end: int


REFERENCE_PREFIX = "reference_prefix"
STRUCTURED_JSON = "structured_json"
_EDIT_START = "<|leyline_delete_start_7f1d42|>"
_EDIT_END = "<|leyline_delete_end_7f1d42|>"


def _encode(tokenizer: Any, text: str, *, special: bool = False) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=special))


def _case_text(case: dict[str, Any], removed: str | None) -> tuple[str, str, str]:
    removed_text = (removed if removed is not None else case["removed"]) * case.get("removed_repeat", 1)
    surviving_text = case["surviving"] * case.get("surviving_repeat", 1)
    return case["prefix"], removed_text, surviving_text + case["query"]


def _chat_template_parts(tokenizer: Any, prefix: str, removed: str, suffix: str) -> tuple[str, str, str]:
    content = prefix + removed + suffix
    if _EDIT_START in content or _EDIT_END in content:
        raise ValueError("workload contains a reserved Leyline edit marker")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prefix + _EDIT_START + removed + _EDIT_END + suffix}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return text")
    before, separator, tail = rendered.partition(_EDIT_START)
    if not separator:
        raise ValueError("chat template did not preserve the Leyline start marker")
    rendered_removed, separator, after = tail.partition(_EDIT_END)
    if not separator:
        raise ValueError("chat template did not preserve the Leyline end marker")
    return before, rendered_removed, after


def build_prompt_plan(
    tokenizer: Any,
    case: dict[str, Any],
    removed: str | None = None,
    *,
    prompt_format: str = "raw",
) -> PromptPlan:
    prefix_text, removed_text, suffix_text = _case_text(case, removed)
    if prompt_format == "raw":
        prefix = _encode(tokenizer, prefix_text, special=True)
        removed_tokens = _encode(tokenizer, removed_text)
        suffix = _encode(tokenizer, suffix_text)
    elif prompt_format == "chat_template":
        rendered_prefix, rendered_removed, rendered_suffix = _chat_template_parts(
            tokenizer, prefix_text, removed_text, suffix_text
        )
        prefix = _encode(tokenizer, rendered_prefix)
        removed_tokens = _encode(tokenizer, rendered_removed)
        suffix = _encode(tokenizer, rendered_suffix)
    else:
        raise ValueError(f"unsupported prompt_format: {prompt_format!r}")
    full = prefix + removed_tokens + suffix
    edited = prefix + suffix
    return PromptPlan(full, edited, len(prefix), len(prefix) + len(removed_tokens))


def extract_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _completion_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/v1/completions") else endpoint + "/v1/completions"


def request_completion(
    endpoint: str,
    model: str,
    prompt: list[int],
    *,
    api_key: str | None,
    max_tokens: int,
    kv_transfer_params: dict[str, Any] | None = None,
    cache_salt: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 0,
        # The harness has already applied BOS/chat formatting and must keep
        # token indices identical to the declared deletion span.
        "add_special_tokens": False,
        "return_token_ids": True,
    }
    if kv_transfer_params is not None:
        payload["kv_transfer_params"] = kv_transfer_params
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        _completion_url(endpoint),
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=600) as response:
            body = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"completion request failed ({exc.code}): {detail}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    choice = body["choices"][0]
    text = choice["text"]
    return {
        "text": text,
        "output_token_ids": choice.get("token_ids"),
        "structured": extract_json(text),
        "usage": body.get("usage"),
        "metrics": body.get("metrics"),
        "kv_transfer_params": body.get("kv_transfer_params"),
        "client_latency_ms": elapsed_ms,
    }


def _directive(action: str, session_id: str, plan: PromptPlan | None = None) -> dict[str, Any]:
    leyline: dict[str, Any] = {"version": 1, "action": action, "session_id": session_id}
    if action == "amortize":
        assert plan is not None
        leyline["delete"] = {"start": plan.delete_start, "end": plan.delete_end}
    return {"leyline": leyline}


def _arm_prompt(arm: str, plan: PromptPlan) -> list[int]:
    return plan.full if arm in {"full", "cache_off"} else plan.edited


def _evaluation_config(config: dict[str, Any]) -> tuple[str, int]:
    evaluation = config.get("evaluation", {})
    mode = evaluation.get("mode", REFERENCE_PREFIX)
    if mode not in {REFERENCE_PREFIX, STRUCTURED_JSON}:
        raise ValueError(f"unsupported evaluation mode: {mode!r}")
    reference_tokens = int(evaluation.get("reference_tokens", 1))
    if reference_tokens < 1:
        raise ValueError("evaluation.reference_tokens must be at least 1")
    return mode, reference_tokens


def _annotate_match(
    result: dict[str, Any],
    *,
    mode: str,
    oracle: dict[str, Any],
    reference: list[int] | None,
    reference_tokens: int,
) -> bool:
    if mode == STRUCTURED_JSON:
        matched = result["structured"] == oracle
        result["matches_oracle"] = matched
        return matched

    output = result.get("output_token_ids")
    ready = isinstance(reference, list) and len(reference) >= reference_tokens
    matched = bool(
        ready
        and isinstance(output, list)
        and len(output) >= reference_tokens
        and output[:reference_tokens] == reference[:reference_tokens]
    )
    result["matches_reference"] = matched
    return matched


def leyline_execution_evidence(result: dict[str, Any] | None) -> dict[str, bool]:
    """Return strict evidence that Leyline transformed cache instead of recomputing."""

    if result is None:
        return {
            "output_token_ids_present": False,
            "applied": False,
            "transformed_tokens_positive": False,
            "no_fallback": False,
            "valid": False,
        }
    token_ids = result.get("output_token_ids")
    metadata = (result.get("kv_transfer_params") or {}).get("leyline")
    metadata = metadata if isinstance(metadata, dict) else {}
    evidence = {
        "output_token_ids_present": isinstance(token_ids, list) and bool(token_ids),
        "applied": metadata.get("applied") is True,
        "transformed_tokens_positive": isinstance(metadata.get("transformed_tokens"), int)
        and metadata["transformed_tokens"] > 0,
        "no_fallback": "fallback_reason" in metadata and metadata["fallback_reason"] is None,
    }
    evidence["valid"] = all(evidence.values())
    return evidence


def evaluate_case_results(
    case: dict[str, Any],
    arms: dict[str, Any],
    counterfactuals: list[dict[str, Any]],
    *,
    mode: str,
    reference_tokens: int,
) -> dict[str, Any]:
    oracle = case["oracle"]
    full_result = arms.get("full")
    reference = full_result.get("output_token_ids") if full_result is not None else None
    matches: dict[str, bool] = {}
    for arm, result in arms.items():
        matches[arm] = _annotate_match(
            result,
            mode=mode,
            oracle=oracle,
            reference=reference,
            reference_tokens=reference_tokens,
        )
    counterfactual_matches = [
        _annotate_match(
            result,
            mode=mode,
            oracle=oracle,
            reference=reference,
            reference_tokens=reference_tokens,
        )
        for result in counterfactuals
    ]

    if mode == STRUCTURED_JSON:
        full_ready = matches.get("full")
        target = "declared_structured_oracle"
    else:
        full_ready = isinstance(reference, list) and len(reference) >= reference_tokens
        target = f"first_{reference_tokens}_generated_token_ids_from_full"
    full_ok = bool(full_ready)
    honest_ok = matches.get("honest_edited")
    counterfactual_ok = all(counterfactual_matches)
    declared_admissible = case["category"] in {"admissible", "counterfactual_admissible"}
    admitted = bool(declared_admissible and full_ok and honest_ok and counterfactual_ok)
    semantic_admitted = admitted if mode == STRUCTURED_JSON else False
    reference_admitted = admitted if mode == REFERENCE_PREFIX else False
    leyline_evidence = leyline_execution_evidence(arms.get("leyline"))
    leyline_matches = matches.get("leyline", False)
    return {
        "mode": mode,
        "match_target": target,
        "reference_tokens": reference_tokens if mode == REFERENCE_PREFIX else None,
        "semantic_oracle_validated": mode == STRUCTURED_JSON,
        "gates": {
            "full_matches": full_ready,
            "honest_edited_matches": honest_ok,
            "counterfactuals_match": counterfactual_ok,
            "admitted": admitted,
            "semantic_admitted": semantic_admitted,
            "reference_admitted": reference_admitted,
            "leyline_matches": leyline_matches,
            "leyline_execution_valid": leyline_evidence["valid"],
            "leyline_accepted": bool(leyline_matches and leyline_evidence["valid"]),
        },
        "leyline_evidence": leyline_evidence,
    }


def run_case(case: dict[str, Any], tokenizer: Any, config: dict[str, Any]) -> dict[str, Any]:
    endpoints = config["arms"]
    model = config["model"]
    api_key = config.get("api_key")
    max_tokens = int(config.get("max_tokens", 64))
    prompt_format = config.get("prompt_format", "raw")
    evaluation_mode, reference_tokens = _evaluation_config(config)
    plan = build_prompt_plan(tokenizer, case, prompt_format=prompt_format)
    oracle = case["oracle"]
    arms: dict[str, Any] = {}

    for arm in ("cache_off", "full", "vanilla_apc", "honest_edited", "patched_disabled"):
        if arm not in endpoints:
            continue
        result = request_completion(
            endpoints[arm],
            model,
            _arm_prompt(arm, plan),
            api_key=api_key,
            max_tokens=max_tokens,
            cache_salt=uuid.uuid4().hex,
        )
        arms[arm] = result

    if "leyline" in endpoints:
        session_id = f"{case['id']}-{uuid.uuid4().hex}"
        cache_salt = uuid.uuid4().hex
        record = request_completion(
            endpoints["leyline"],
            model,
            plan.full,
            api_key=api_key,
            max_tokens=max_tokens,
            kv_transfer_params=_directive("record", session_id),
            cache_salt=cache_salt,
        )
        result = request_completion(
            endpoints["leyline"],
            model,
            plan.edited,
            api_key=api_key,
            max_tokens=max_tokens,
            kv_transfer_params=_directive("amortize", session_id, plan),
            cache_salt=cache_salt,
        )
        result["record"] = record
        arms["leyline"] = result

    counterfactuals = []
    for index, removed in enumerate(case.get("counterfactual_removed", [])) if "full" in endpoints else []:
        variant = build_prompt_plan(tokenizer, case, removed, prompt_format=prompt_format)
        result = request_completion(
            endpoints["full"],
            model,
            variant.full,
            api_key=api_key,
            max_tokens=max_tokens,
            cache_salt=uuid.uuid4().hex,
        )
        result["variant"] = index
        counterfactuals.append(result)

    evaluation = evaluate_case_results(
        case,
        arms,
        counterfactuals,
        mode=evaluation_mode,
        reference_tokens=reference_tokens,
    )
    return {
        "id": case["id"],
        "category": case["category"],
        "prompt_tokens": {
            "full": len(plan.full),
            "edited": len(plan.edited),
            "deleted": plan.delete_end - plan.delete_start,
        },
        "oracle": oracle,
        "evaluation": {
            key: value for key, value in evaluation.items() if key not in {"gates", "leyline_evidence"}
        },
        "arms": arms,
        "counterfactuals": counterfactuals,
        "gates": evaluation["gates"],
        "leyline_evidence": evaluation["leyline_evidence"],
    }


class NpuMemorySampler:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.samples_mib: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> NpuMemorySampler:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(self.command, capture_output=True, text=True, timeout=5, check=False)
                pairs = re.findall(r"(\d+)\s*/\s*(\d+)\s*(?:MiB|MB)", result.stdout)
                self.samples_mib.extend(int(used) for used, _ in pairs)
            except (OSError, subprocess.SubprocessError):
                return
            self._stop.wait(0.2)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_performance(results: list[dict[str, Any]], wall_seconds: float, memory: list[int]) -> dict[str, Any]:
    ttft = [
        float(item["metrics"]["time_to_first_token_ms"])
        for item in results
        if item.get("metrics") and item["metrics"].get("time_to_first_token_ms") is not None
    ]
    latency = [float(item["client_latency_ms"]) for item in results]
    output_tokens = sum(int((item.get("usage") or {}).get("completion_tokens") or 0) for item in results)
    leyline = [
        (item.get("kv_transfer_params") or {}).get("leyline", {})
        for item in results
        if (item.get("kv_transfer_params") or {}).get("leyline")
    ]
    return {
        "requests": len(results),
        "wall_seconds": wall_seconds,
        "output_tokens_per_second": output_tokens / wall_seconds if wall_seconds else None,
        "ttft_ms": {"p50": percentile(ttft, 0.5), "p95": percentile(ttft, 0.95), "p99": percentile(ttft, 0.99)},
        "client_latency_ms": {
            "mean": statistics.fmean(latency) if latency else None,
            "p95": percentile(latency, 0.95),
        },
        "leyline": {
            "transformed_tokens": sum(int(item.get("transformed_tokens", 0)) for item in leyline),
            "normal_prefill_tokens": sum(int(item.get("normal_prefill_tokens", 0)) for item in leyline),
            "transform_duration_ms": sum(float(item.get("transform_duration_ms", 0)) for item in leyline),
            "applied_requests": sum(bool(item.get("applied")) for item in leyline),
            "fallback_reasons": [item.get("fallback_reason") for item in leyline if item.get("fallback_reason")],
        },
        "max_observed_npu_memory_mib": max(memory) if memory else None,
    }


def run_performance(
    case: dict[str, Any], tokenizer: Any, config: dict[str, Any], concurrencies: list[int], repetitions: int
) -> list[dict[str, Any]]:
    plan = build_prompt_plan(tokenizer, case, prompt_format=config.get("prompt_format", "raw"))
    api_key = config.get("api_key")
    model = config["model"]
    max_tokens = int(config.get("max_tokens", 64))
    memory_command = config.get("npu_memory_command", ["npu-smi", "info"])
    summaries = []
    for arm, endpoint in config["arms"].items():
        for concurrency in concurrencies:
            jobs: list[tuple[list[int], dict[str, Any] | None, str]] = []
            if arm == "leyline":
                for _ in range(concurrency * repetitions):
                    session_id = f"perf-{uuid.uuid4().hex}"
                    cache_salt = uuid.uuid4().hex
                    request_completion(
                        endpoint,
                        model,
                        plan.full,
                        api_key=api_key,
                        max_tokens=1,
                        kv_transfer_params=_directive("record", session_id),
                        cache_salt=cache_salt,
                    )
                    jobs.append((plan.edited, _directive("amortize", session_id, plan), cache_salt))
            elif arm == "vanilla_apc":
                for _ in range(concurrency * repetitions):
                    cache_salt = uuid.uuid4().hex
                    request_completion(
                        endpoint,
                        model,
                        plan.edited,
                        api_key=api_key,
                        max_tokens=1,
                        cache_salt=cache_salt,
                    )
                    jobs.append((plan.edited, None, cache_salt))
            else:
                jobs = [
                    (_arm_prompt(arm, plan), None, uuid.uuid4().hex)
                    for _ in range(concurrency * repetitions)
                ]

            with NpuMemorySampler(memory_command) as sampler:
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(
                            request_completion,
                            endpoint,
                            model,
                            prompt,
                            api_key=api_key,
                            max_tokens=max_tokens,
                            kv_transfer_params=params,
                            cache_salt=cache_salt,
                        )
                        for prompt, params, cache_salt in jobs
                    ]
                    results = [future.result() for future in futures]
                wall_seconds = time.perf_counter() - started
            summaries.append(
                {
                    "arm": arm,
                    "concurrency": concurrency,
                    **summarize_performance(results, wall_seconds, sampler.samples_mib),
                }
            )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, default=here / "workloads.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    workloads = json.loads(args.workloads.read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"],
        revision=config.get("tokenizer_revision"),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    cases = [run_case(case, tokenizer, config) for case in workloads["cases"]]
    report: dict[str, Any] = {
        "schema_version": 2,
        "created_unix": time.time(),
        "config": {key: value for key, value in config.items() if key != "api_key"},
        "cases": cases,
    }
    if args.environment:
        report["environment"] = json.loads(args.environment.read_text())
    if args.performance:
        admissible = next(case for case in workloads["cases"] if case["category"] == "admissible")
        concurrencies = [int(value) for value in args.concurrency.split(",")]
        report["performance"] = run_performance(
            admissible, tokenizer, config, concurrencies, args.repetitions
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
