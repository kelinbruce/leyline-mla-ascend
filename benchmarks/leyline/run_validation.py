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
RAW = "raw"
CHAT_TEMPLATE = "chat_template"


def _encode(tokenizer: Any, text: str, *, special: bool = False) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=special))


def _case_text(case: dict[str, Any], removed: str | None = None) -> tuple[str, str]:
    removed_text = (removed if removed is not None else case["removed"]) * case.get("removed_repeat", 1)
    surviving_text = case["surviving"] * case.get("surviving_repeat", 1)
    return case["prefix"] + removed_text + surviving_text + case["query"], (
        case["prefix"] + surviving_text + case["query"]
    )


def _exact_deletion(full: list[int], edited: list[int]) -> tuple[int, int]:
    deleted = len(full) - len(edited)
    if deleted <= 0:
        raise ValueError("full prompt must contain at least one deleted token")
    start = 0
    while start < len(edited) and full[start] == edited[start]:
        start += 1
    end = start + deleted
    if full[:start] + full[end:] != edited:
        raise ValueError("full and edited prompts are not related by one exact token deletion")
    return start, end


def build_prompt_plan(
    tokenizer: Any,
    case: dict[str, Any],
    removed: str | None = None,
    *,
    prompt_format: str = RAW,
) -> PromptPlan:
    full_text, edited_text = _case_text(case, removed)
    if prompt_format == RAW:
        # Encode the canonical complete strings. This detects token-boundary
        # replacement instead of silently treating it as a deletion.
        full = _encode(tokenizer, full_text, special=True)
        edited = _encode(tokenizer, edited_text, special=True)
    elif prompt_format == CHAT_TEMPLATE:
        if not getattr(tokenizer, "chat_template", None) and not hasattr(
            tokenizer, "apply_chat_template"
        ):
            raise ValueError("chat_template prompt format requires a tokenizer chat template")
        messages_full = [{"role": "user", "content": full_text}]
        messages_edited = [{"role": "user", "content": edited_text}]
        try:
            rendered_full = tokenizer.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=True
            )
            rendered_edited = tokenizer.apply_chat_template(
                messages_edited, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, ValueError) as exc:
            raise ValueError("tokenizer has no usable chat template") from exc
        if not isinstance(rendered_full, str) or not isinstance(rendered_edited, str):
            raise TypeError("chat template must return strings when tokenize=false")
        full = _encode(tokenizer, rendered_full)
        edited = _encode(tokenizer, rendered_edited)
    else:
        raise ValueError(f"unsupported prompt format: {prompt_format!r}")
    delete_start, delete_end = _exact_deletion(full, edited)
    return PromptPlan(full, edited, delete_start, delete_end)


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
    top_logprobs: int = 0,
    request_id: str | None = None,
) -> dict[str, Any]:
    request_id = request_id or f"leyline-validation-{uuid.uuid4().hex}"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 0,
        "add_special_tokens": False,
        "return_token_ids": True,
        "request_id": request_id,
    }
    if top_logprobs > 0:
        payload["logprobs"] = top_logprobs
        payload["return_tokens_as_token_ids"] = True
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
        "request_id": request_id,
        "response_id": body.get("id"),
        "text": text,
        "output_token_ids": choice.get("token_ids"),
        "first_token_scores": _first_token_scores(choice),
        "structured": extract_json(text),
        "usage": body.get("usage"),
        "metrics": body.get("metrics"),
        "kv_transfer_params": body.get("kv_transfer_params"),
        "client_latency_ms": elapsed_ms,
    }


def _token_id(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith("token_id:"):
        try:
            return int(value.removeprefix("token_id:"))
        except ValueError:
            return None
    return None


def _first_token_scores(choice: dict[str, Any]) -> dict[str, Any] | None:
    logprobs = choice.get("logprobs")
    token_ids = choice.get("token_ids")
    if not isinstance(logprobs, dict) or not isinstance(token_ids, list) or not token_ids:
        return None
    token_logprobs = logprobs.get("token_logprobs") or []
    top = logprobs.get("top_logprobs") or []
    first_top = top[0] if top and isinstance(top[0], dict) else {}
    candidates = [
        {"token_id": token_id, "logprob": float(score)}
        for token, score in first_top.items()
        if (token_id := _token_id(token)) is not None
    ]
    candidates.sort(key=lambda item: item["logprob"], reverse=True)
    selected_score = token_logprobs[0] if token_logprobs else None
    return {
        "evidence_type": "api_logprobs",
        "raw_logits": False,
        "selected_token_id": int(token_ids[0]),
        "selected_logprob": float(selected_score) if selected_score is not None else None,
        "candidates": candidates,
        "top1_top2_margin": (
            candidates[0]["logprob"] - candidates[1]["logprob"]
            if len(candidates) >= 2
            else None
        ),
    }


def _directive(action: str, session_id: str, plan: PromptPlan | None = None) -> dict[str, Any]:
    leyline: dict[str, Any] = {"version": 1, "action": action, "session_id": session_id}
    if action == "amortize":
        assert plan is not None
        leyline["delete"] = {"start": plan.delete_start, "end": plan.delete_end}
    return {"leyline": leyline}


def _arm_prompt(arm: str, plan: PromptPlan) -> list[int]:
    return plan.full if arm in {"full", "cache_off"} else plan.edited


def evaluation_config(config: dict[str, Any]) -> tuple[str, str, int]:
    evaluation = config.get("evaluation", {})
    mode = evaluation.get("mode", REFERENCE_PREFIX)
    prompt_format = config.get("prompt_format", RAW)
    reference_tokens = int(evaluation.get("reference_tokens", 1))
    if mode not in {REFERENCE_PREFIX, STRUCTURED_JSON}:
        raise ValueError(f"unsupported evaluation mode: {mode!r}")
    if prompt_format not in {RAW, CHAT_TEMPLATE}:
        raise ValueError(f"unsupported prompt format: {prompt_format!r}")
    if mode == STRUCTURED_JSON and prompt_format != CHAT_TEMPLATE:
        raise ValueError("structured_json evaluation requires prompt_format=chat_template")
    if mode == REFERENCE_PREFIX and prompt_format != RAW:
        raise ValueError("reference_prefix evaluation requires prompt_format=raw")
    if reference_tokens < 1:
        raise ValueError("evaluation.reference_tokens must be at least 1")
    return mode, prompt_format, reference_tokens


def _annotate_match(
    result: dict[str, Any],
    *,
    mode: str,
    oracle: dict[str, Any],
    reference: list[int] | None,
    reference_tokens: int,
) -> bool:
    if mode == STRUCTURED_JSON:
        matched = result.get("structured") == oracle
        result["matches_oracle"] = matched
        return matched
    output = result.get("output_token_ids")
    matched = bool(
        isinstance(reference, list)
        and len(reference) >= reference_tokens
        and isinstance(output, list)
        and len(output) >= reference_tokens
        and output[:reference_tokens] == reference[:reference_tokens]
    )
    result["matches_reference"] = matched
    return matched


def leyline_execution_evidence(result: dict[str, Any] | None) -> dict[str, bool]:
    metadata = ((result or {}).get("kv_transfer_params") or {}).get("leyline") or {}
    evidence = {
        "output_token_ids_present": bool((result or {}).get("output_token_ids")),
        "recorded": metadata.get("recorded") is True,
        "applied": metadata.get("applied") is True,
        "transformed_tokens_positive": isinstance(metadata.get("transformed_tokens"), int)
        and metadata["transformed_tokens"] > 0,
        "no_fallback": "fallback_reason" in metadata and metadata["fallback_reason"] is None,
    }
    evidence["valid"] = all(evidence.values())
    return evidence


def _common_prefix(left: list[int], right: list[int]) -> int:
    return next(
        (i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        min(len(left), len(right)),
    )


def pairwise_diagnostic(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    left_ids = (left or {}).get("output_token_ids")
    right_ids = (right or {}).get("output_token_ids")
    if not isinstance(left_ids, list) or not isinstance(right_ids, list):
        return {"available": False}
    prefix = _common_prefix(left_ids, right_ids)
    left_candidates = {
        item["token_id"] for item in (((left or {}).get("first_token_scores") or {}).get("candidates") or [])
    }
    right_candidates = {
        item["token_id"] for item in (((right or {}).get("first_token_scores") or {}).get("candidates") or [])
    }
    return {
        "available": True,
        "first_token_agreement": bool(left_ids and right_ids and left_ids[0] == right_ids[0]),
        "common_prefix_tokens": prefix,
        "first_divergence": None
        if prefix == len(left_ids) == len(right_ids)
        else {
            "index": prefix,
            "left": left_ids[prefix] if prefix < len(left_ids) else None,
            "right": right_ids[prefix] if prefix < len(right_ids) else None,
        },
        "topk_overlap_token_ids": sorted(left_candidates & right_candidates),
        "left_top1_top2_margin": ((left or {}).get("first_token_scores") or {}).get("top1_top2_margin"),
        "right_top1_top2_margin": ((right or {}).get("first_token_scores") or {}).get("top1_top2_margin"),
        "evidence_type": "api_logprobs" if left_candidates or right_candidates else "generated_token_ids",
    }


def evaluate_case_results(
    case: dict[str, Any],
    arms: dict[str, Any],
    counterfactuals: list[dict[str, Any]],
    *,
    mode: str,
    reference_tokens: int,
    preflight_passed: bool = True,
) -> dict[str, Any]:
    reference = arms.get("full", {}).get("output_token_ids")
    matches = {
        arm: _annotate_match(
            result,
            mode=mode,
            oracle=case["oracle"],
            reference=reference,
            reference_tokens=reference_tokens,
        )
        for arm, result in arms.items()
    }
    counterfactual_matches = [
        _annotate_match(
            result,
            mode=mode,
            oracle=case["oracle"],
            reference=reference,
            reference_tokens=reference_tokens,
        )
        for result in counterfactuals
    ]
    full_ok = matches.get("full", False) if mode == STRUCTURED_JSON else bool(
        isinstance(reference, list) and len(reference) >= reference_tokens
    )
    honest_ok = matches.get("honest_edited", False)
    counterfactual_ok = all(counterfactual_matches)
    category = case["category"]
    declared = category in {"admissible", "counterfactual_admissible"}
    category_baseline_ok = (
        full_ok and honest_ok and counterfactual_ok if declared else full_ok
    )
    admitted = bool(declared and category_baseline_ok and (preflight_passed or mode == REFERENCE_PREFIX))
    execution = leyline_execution_evidence(arms.get("leyline"))
    leyline_matches = matches.get("leyline", False)
    return {
        "mode": mode,
        "match_target": (
            "declared_structured_oracle"
            if mode == STRUCTURED_JSON
            else f"first_{reference_tokens}_generated_token_ids_from_full"
        ),
        "reference_tokens": reference_tokens if mode == REFERENCE_PREFIX else None,
        "semantic_oracle_validated": mode == STRUCTURED_JSON and preflight_passed,
        "gates": {
            "full_matches": full_ok,
            "honest_edited_matches": honest_ok,
            "counterfactuals_match": counterfactual_ok,
            "category_baseline_matches": category_baseline_ok,
            "admitted": admitted,
            "semantic_admitted": admitted if mode == STRUCTURED_JSON else False,
            "reference_admitted": admitted if mode == REFERENCE_PREFIX else False,
            "diagnostic_only": not declared or not admitted,
            "leyline_matches": leyline_matches,
            "leyline_execution_valid": execution["valid"],
            "leyline_accepted": bool(admitted and leyline_matches and execution["valid"]),
        },
        "leyline_execution": execution,
        "pairwise": {
            "full_honest_edited": pairwise_diagnostic(arms.get("full"), arms.get("honest_edited")),
            "full_leyline": pairwise_diagnostic(arms.get("full"), arms.get("leyline")),
            "honest_edited_leyline": pairwise_diagnostic(arms.get("honest_edited"), arms.get("leyline")),
        },
    }


def run_case(
    case: dict[str, Any],
    tokenizer: Any,
    config: dict[str, Any],
    *,
    preflight_passed: bool = True,
) -> dict[str, Any]:
    endpoints = config["arms"]
    model = config["model"]
    api_key = config.get("api_key")
    max_tokens = int(config.get("max_tokens", 64))
    mode, prompt_format, reference_tokens = evaluation_config(config)
    top_logprobs = int(config.get("diagnostics", {}).get("top_logprobs", 0))
    plan = build_prompt_plan(tokenizer, case, prompt_format=prompt_format)
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
            top_logprobs=top_logprobs,
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
            top_logprobs=top_logprobs,
        )
        result = request_completion(
            endpoints["leyline"],
            model,
            plan.edited,
            api_key=api_key,
            max_tokens=max_tokens,
            kv_transfer_params=_directive("amortize", session_id, plan),
            cache_salt=cache_salt,
            top_logprobs=top_logprobs,
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
            top_logprobs=top_logprobs,
        )
        result["variant"] = index
        counterfactuals.append(result)

    evaluation = evaluate_case_results(
        case,
        arms,
        counterfactuals,
        mode=mode,
        reference_tokens=reference_tokens,
        preflight_passed=preflight_passed,
    )
    return {
        "id": case["id"],
        "category": case["category"],
        "prompt_tokens": {
            "full": len(plan.full),
            "edited": len(plan.edited),
            "deleted": plan.delete_end - plan.delete_start,
        },
        "oracle": case["oracle"],
        "evaluation": {
            "mode": evaluation["mode"],
            "match_target": evaluation["match_target"],
            "reference_tokens": evaluation["reference_tokens"],
            "semantic_oracle_validated": evaluation["semantic_oracle_validated"],
            "prompt_format": prompt_format,
            "preflight_passed": preflight_passed,
        },
        "arms": arms,
        "counterfactuals": counterfactuals,
        "gates": evaluation["gates"],
        "leyline_execution": evaluation["leyline_execution"],
        "pairwise": evaluation["pairwise"],
    }


class NpuMemorySampler:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.samples_mib: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "NpuMemorySampler":
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
    _, prompt_format, _ = evaluation_config(config)
    plan = build_prompt_plan(tokenizer, case, prompt_format=prompt_format)
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


def run_preflight(tokenizer: Any, config: dict[str, Any]) -> dict[str, Any]:
    mode, prompt_format, _ = evaluation_config(config)
    if mode == REFERENCE_PREFIX:
        return {"required": False, "passed": True, "skipped_reason": "reference_prefix_mode"}
    settings = config.get("preflight", {})
    prompt = settings.get("prompt", 'Return only this JSON: {"ok":true}')
    oracle = settings.get("oracle", {"ok": True})
    endpoint = settings.get("endpoint") or config.get("arms", {}).get("full")
    if not endpoint:
        return {"required": True, "passed": False, "error": "full/preflight endpoint is missing"}
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        if not isinstance(rendered, str):
            raise TypeError("chat template did not return a string")
        prompt_ids = _encode(tokenizer, rendered)
        result = request_completion(
            endpoint,
            config["model"],
            prompt_ids,
            api_key=config.get("api_key"),
            max_tokens=int(settings.get("max_tokens", 16)),
            cache_salt=uuid.uuid4().hex,
            top_logprobs=int(config.get("diagnostics", {}).get("top_logprobs", 0)),
        )
    except Exception as exc:
        return {
            "required": True,
            "passed": False,
            "prompt_format": prompt_format,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "required": True,
        "passed": result["structured"] == oracle,
        "prompt_format": prompt_format,
        "oracle": oracle,
        "prompt_token_ids": prompt_ids,
        "result": result,
    }


def performance_gate(
    cases: list[dict[str, Any]], config: dict[str, Any], preflight: dict[str, Any]
) -> dict[str, Any]:
    blockers: list[str] = []
    if not preflight.get("passed"):
        blockers.append("baseline_preflight_failed")
    admissible = [case for case in cases if case["category"] in {"admissible", "counterfactual_admissible"}]
    if not admissible or not all(case["gates"]["admitted"] for case in admissible):
        blockers.append("selected_baseline_not_admitted")
    if not admissible or not all(case["gates"]["leyline_execution_valid"] for case in admissible):
        blockers.append("leyline_execution_invalid")
    if not admissible or not all(case["gates"]["leyline_accepted"] for case in admissible):
        blockers.append("leyline_correctness_not_accepted")
    prerequisites = config.get("performance_prerequisites", {})
    if prerequisites.get("numerical_passed") is not True:
        blockers.append("numerical_gate_failed_or_missing")
    if prerequisites.get("rollback_passed") is not True:
        blockers.append("rollback_gate_failed_or_missing")
    capture_enabled = bool(config.get("diagnostics", {}).get("device_capture_enabled"))
    if capture_enabled:
        blockers.append("device_capture_must_be_disabled")
    if config.get("diagnostics", {}).get("raw_logits_capture_enabled"):
        blockers.append("raw_logits_capture_must_be_disabled")
    return {"passed": not blockers, "blockers": blockers}


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
    preflight = run_preflight(tokenizer, config)
    allow_diagnostics = bool(config.get("diagnostics", {}).get("continue_after_preflight_failure", True))
    if not preflight["passed"] and not allow_diagnostics:
        cases: list[dict[str, Any]] = []
    else:
        cases = [
            run_case(case, tokenizer, config, preflight_passed=bool(preflight["passed"]))
            for case in workloads["cases"]
        ]
    report: dict[str, Any] = {
        "schema_version": 2,
        "created_unix": time.time(),
        "config": {key: value for key, value in config.items() if key != "api_key"},
        "evaluation_contract": {
            "mode": evaluation_config(config)[0],
            "prompt_format": evaluation_config(config)[1],
            "score_evidence": "api_logprobs",
            "raw_logits_captured": bool(
                config.get("diagnostics", {}).get("raw_logits_capture_enabled", False)
            ),
        },
        "preflight": preflight,
        "cases": cases,
    }
    if args.environment:
        report["environment"] = json.loads(args.environment.read_text())
        report["checkpoint_identity"] = report["environment"].get("model")
    if args.performance:
        gate = performance_gate(cases, config, preflight)
        report["performance_gate"] = gate
        if gate["passed"]:
            admissible = next(case for case in workloads["cases"] if case["category"] == "admissible")
            concurrencies = [int(value) for value in args.concurrency.split(",")]
            report["performance"] = run_performance(
                admissible, tokenizer, config, concurrencies, args.repetitions
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
