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
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.leyline.evidence import (  # noqa: E402
    cache_comparison_errors,
    classify_case,
    ensure_run_id,
    environment_blockers,
    report_identity,
    retained_context_summary,
    rollback_report_errors,
    validation_request_id,
)


@dataclass(frozen=True)
class PromptPlan:
    full: list[int]
    edited: list[int]
    delete_start: int
    delete_end: int
    target_token_ids: tuple[int, ...] = ()
    surviving_filler_repeat: int = 0


@dataclass(frozen=True)
class TransformFeasibility:
    full_tokens: int
    edited_tokens: int
    block_size: int
    complete_source_blocks: tuple[int, ...]
    local_computed_tokens: int
    max_target_tokens: int
    reusable_target_start: int
    reusable_target_end: int
    predicted_transformed_tokens: int
    predicted_shifted_tokens: int
    blocking_source_block: int | None
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "full_tokens": self.full_tokens,
            "edited_tokens": self.edited_tokens,
            "block_size": self.block_size,
            "complete_source_blocks": list(self.complete_source_blocks),
            "complete_source_block_count": len(self.complete_source_blocks),
            "local_computed_tokens": self.local_computed_tokens,
            "max_target_tokens": self.max_target_tokens,
            "reusable_target_start": self.reusable_target_start,
            "reusable_target_end": self.reusable_target_end,
            "predicted_transformed_tokens": self.predicted_transformed_tokens,
            "predicted_shifted_tokens": self.predicted_shifted_tokens,
            "blocking_source_block": self.blocking_source_block,
            "reason": self.reason,
        }


REFERENCE_PREFIX = "reference_prefix"
STRUCTURED_JSON = "structured_json"
COMPLETION_TARGET = "completion_target"
RAW = "raw"
CHAT_TEMPLATE = "chat_template"


def _encode(tokenizer: Any, text: str, *, special: bool = False) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=special))


def _case_text(case: dict[str, Any], removed: str | None = None) -> tuple[str, str]:
    removed_text = (removed if removed is not None else case["removed"]) * case.get("removed_repeat", 1)
    surviving_filler = case.get("surviving_filler_unit", "") * int(
        case.get("_surviving_filler_repeat", 0)
    )
    surviving_text = surviving_filler + case["surviving"] * case.get("surviving_repeat", 1)
    return case["prefix"] + removed_text + surviving_text + case["query"], (
        case["prefix"] + surviving_text + case["query"]
    )


def plan_transform_feasibility(
    plan: PromptPlan,
    *,
    block_size: int,
    local_computed_tokens: int = 0,
) -> TransformFeasibility:
    """Predict the connector's longest reusable complete-block prefix."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if local_computed_tokens < 0 or local_computed_tokens % block_size:
        raise ValueError("local_computed_tokens must be non-negative and block aligned")
    complete_source_blocks = tuple(range(len(plan.full) // block_size))
    max_target_tokens = max(len(plan.edited) - 1, 0)
    candidate_end = min(len(plan.edited), max_target_tokens)
    candidate_end -= candidate_end % block_size
    reusable_end = local_computed_tokens
    blocking_source_block: int | None = None
    resident = set(complete_source_blocks)
    delete_length = plan.delete_end - plan.delete_start
    for target_end in range(
        local_computed_tokens + block_size,
        candidate_end + 1,
        block_size,
    ):
        target_start = target_end - block_size
        required = {
            (position if position < plan.delete_start else position + delete_length)
            // block_size
            for position in range(target_start, target_end)
        }
        missing = sorted(required - resident)
        if missing:
            blocking_source_block = missing[0]
            break
        reusable_end = target_end
    transformed = max(reusable_end - local_computed_tokens, 0)
    shifted = max(reusable_end - max(local_computed_tokens, plan.delete_start), 0)
    if not complete_source_blocks:
        reason = "missing_source_blocks"
    elif transformed:
        reason = None
    elif blocking_source_block is not None:
        reason = "mapped_source_block_not_resident"
    else:
        reason = "no_reusable_target_blocks"
    return TransformFeasibility(
        full_tokens=len(plan.full),
        edited_tokens=len(plan.edited),
        block_size=block_size,
        complete_source_blocks=complete_source_blocks,
        local_computed_tokens=local_computed_tokens,
        max_target_tokens=candidate_end,
        reusable_target_start=local_computed_tokens,
        reusable_target_end=reusable_end,
        predicted_transformed_tokens=transformed,
        predicted_shifted_tokens=shifted,
        blocking_source_block=blocking_source_block,
        reason=reason,
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


def _fit_removed_repeat(
    tokenizer: Any,
    case: dict[str, Any],
    requested_tokens: int,
    surviving_filler_repeat: int = 0,
) -> int:
    if requested_tokens < 1:
        raise ValueError("position_coverage.delete_tokens must be positive")
    if not case.get("filler_unit"):
        return int(case.get("removed_repeat", 1))

    def achieved(repeat: int) -> int:
        candidate = {
            **case,
            "removed": case["filler_unit"],
            "removed_repeat": repeat,
            "_surviving_filler_repeat": surviving_filler_repeat,
        }
        full_text, edited_text = _case_text(candidate)
        full = _encode(tokenizer, full_text, special=True)
        edited = _encode(tokenizer, edited_text, special=True)
        start, end = _exact_deletion(full, edited)
        return end - start

    low, high = 1, max(2, requested_tokens)
    while achieved(high) < requested_tokens and high < requested_tokens * 4 + 64:
        high *= 2
    while low <= high:
        middle = (low + high) // 2
        value = achieved(middle)
        if value == requested_tokens:
            return middle
        if value < requested_tokens:
            low = middle + 1
        else:
            high = middle - 1
    nearest = sorted(
        (
            (abs(achieved(repeat) - requested_tokens), repeat, achieved(repeat))
            for repeat in range(max(1, high - 4), low + 5)
        ),
        key=lambda item: item[0],
    )[0]
    raise ValueError(
        f"cannot construct exactly {requested_tokens} deleted tokens from filler_unit; "
        f"nearest repeat={nearest[1]} produces {nearest[2]}"
    )


def _build_prompt_plan_once(
    tokenizer: Any,
    case: dict[str, Any],
    removed: str | None = None,
    *,
    prompt_format: str = RAW,
    surviving_filler_repeat: int = 0,
) -> PromptPlan:
    working_case = case
    requested_delete = case.get("position_coverage", {}).get("delete_tokens")
    if removed is None and requested_delete is not None and case.get("filler_unit"):
        repeat = _fit_removed_repeat(
            tokenizer,
            case,
            int(requested_delete),
            surviving_filler_repeat,
        )
        working_case = {**case, "removed": case["filler_unit"], "removed_repeat": repeat}
    if surviving_filler_repeat:
        working_case = {
            **working_case,
            "_surviving_filler_repeat": surviving_filler_repeat,
        }
    full_text, edited_text = _case_text(working_case, removed)
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
    target_token_ids: tuple[int, ...] = ()
    expected_completion = case.get("expected_completion")
    if expected_completion is not None:
        if not isinstance(expected_completion, str) or not expected_completion.strip():
            raise ValueError("expected_completion must contain non-whitespace text")
        if prompt_format != RAW:
            raise ValueError("expected_completion is supported only for raw completion prompts")
        full_with_target = _encode(tokenizer, full_text + expected_completion, special=True)
        edited_with_target = _encode(tokenizer, edited_text + expected_completion, special=True)
        if full_with_target[: len(full)] != full or edited_with_target[: len(edited)] != edited:
            raise ValueError("expected_completion replaces tokens at the prompt boundary")
        full_suffix = tuple(full_with_target[len(full) :])
        edited_suffix = tuple(edited_with_target[len(edited) :])
        if not full_suffix or full_suffix != edited_suffix:
            raise ValueError("expected_completion has unstable full/edited tokenization")
        target_token_ids = full_suffix
    if requested_delete is not None and delete_end - delete_start != int(requested_delete):
        raise ValueError(
            f"requested {requested_delete} deleted tokens, achieved {delete_end - delete_start}"
        )
    return PromptPlan(
        full,
        edited,
        delete_start,
        delete_end,
        target_token_ids,
        surviving_filler_repeat,
    )


def build_prompt_plan(
    tokenizer: Any,
    case: dict[str, Any],
    removed: str | None = None,
    *,
    prompt_format: str = RAW,
    block_size: int = 128,
) -> PromptPlan:
    """Build an exact deletion plan and minimally expand surviving filler."""

    filler_unit = case.get("surviving_filler_unit")
    if not filler_unit:
        return _build_prompt_plan_once(
            tokenizer,
            case,
            removed,
            prompt_format=prompt_format,
        )
    minimum = int(case.get("minimum_transform_tokens", block_size))
    if minimum < 0 or minimum % block_size:
        raise ValueError("minimum_transform_tokens must be non-negative and block aligned")
    maximum_repeat = int(case.get("surviving_filler_max_repeat", 512))
    if maximum_repeat < 1:
        raise ValueError("surviving_filler_max_repeat must be at least 1")
    last: TransformFeasibility | None = None
    for repeat in range(1, maximum_repeat + 1):
        plan = _build_prompt_plan_once(
            tokenizer,
            case,
            removed,
            prompt_format=prompt_format,
            surviving_filler_repeat=repeat,
        )
        last = plan_transform_feasibility(plan, block_size=block_size)
        if (
            last.predicted_transformed_tokens >= minimum
            and last.predicted_shifted_tokens > 0
        ):
            return plan
    assert last is not None
    raise ValueError(
        "surviving filler cannot satisfy transform feasibility within "
        f"{maximum_repeat} repeats: {json.dumps(last.as_dict(), sort_keys=True)}"
    )


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
    if mode not in {REFERENCE_PREFIX, STRUCTURED_JSON, COMPLETION_TARGET}:
        raise ValueError(f"unsupported evaluation mode: {mode!r}")
    if prompt_format not in {RAW, CHAT_TEMPLATE}:
        raise ValueError(f"unsupported prompt format: {prompt_format!r}")
    if mode == STRUCTURED_JSON and prompt_format != CHAT_TEMPLATE:
        raise ValueError("structured_json evaluation requires prompt_format=chat_template")
    if mode in {REFERENCE_PREFIX, COMPLETION_TARGET} and prompt_format != RAW:
        raise ValueError(f"{mode} evaluation requires prompt_format=raw")
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
    target_token_ids: tuple[int, ...] = (),
) -> bool:
    if mode == STRUCTURED_JSON:
        matched = result.get("structured") == oracle
        result["matches_oracle"] = matched
        return matched
    output = result.get("output_token_ids")
    if mode == COMPLETION_TARGET:
        matched = bool(
            target_token_ids
            and isinstance(output, list)
            and len(output) >= len(target_token_ids)
            and tuple(output[: len(target_token_ids)]) == target_token_ids
        )
        result["matches_completion_target"] = matched
        return matched
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
    record_metadata = ((((result or {}).get("record") or {}).get("kv_transfer_params") or {}).get("leyline") or {})
    evidence = {
        "output_token_ids_present": bool((result or {}).get("output_token_ids")),
        "recorded": record_metadata.get("recorded") is True,
        "applied": metadata.get("applied") is True,
        "transformed_tokens_positive": isinstance(metadata.get("transformed_tokens"), int)
        and metadata["transformed_tokens"] > 0,
        "no_fallback": "fallback_reason" in metadata and metadata["fallback_reason"] is None,
        "transform_complete": metadata.get("transform_complete") is True,
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
    target_token_ids: tuple[int, ...] = (),
) -> dict[str, Any]:
    reference = arms.get("full", {}).get("output_token_ids")
    matches = {
        arm: _annotate_match(
            result,
            mode=mode,
            oracle=case.get("oracle", {}),
            reference=reference,
            reference_tokens=reference_tokens,
            target_token_ids=target_token_ids,
        )
        for arm, result in arms.items()
    }
    counterfactual_matches = [
        _annotate_match(
            result,
            mode=mode,
            oracle=case.get("oracle", {}),
            reference=reference,
            reference_tokens=reference_tokens,
            target_token_ids=target_token_ids,
        )
        for result in counterfactuals
    ]
    full_ok = (
        matches.get("full", False)
        if mode in {STRUCTURED_JSON, COMPLETION_TARGET}
        else bool(isinstance(reference, list) and len(reference) >= reference_tokens)
    )
    honest_ok = matches.get("honest_edited", False)
    counterfactual_ok = all(counterfactual_matches)
    category = case["category"]
    declared = category in {"admissible", "counterfactual_admissible"}
    category_baseline_ok = (
        full_ok and honest_ok and counterfactual_ok if declared else full_ok
    )
    admitted = bool(
        declared
        and category_baseline_ok
        and mode != REFERENCE_PREFIX
        and (preflight_passed or mode == COMPLETION_TARGET)
    )
    execution = leyline_execution_evidence(arms.get("leyline"))
    leyline_matches = matches.get("leyline", False)
    return {
        "mode": mode,
        "match_target": (
            "declared_structured_oracle"
            if mode == STRUCTURED_JSON
            else (
                "declared_completion_target"
                if mode == COMPLETION_TARGET
                else f"first_{reference_tokens}_generated_token_ids_from_full"
            )
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
            "completion_admitted": admitted if mode == COMPLETION_TARGET else False,
            "reference_admitted": False,
            "reference_diagnostic_matches": category_baseline_ok if mode == REFERENCE_PREFIX else False,
            "diagnostic_only": not declared or not admitted,
            "leyline_matches": leyline_matches,
            "leyline_execution_valid": execution["valid"],
            "leyline_accepted": bool(
                mode != REFERENCE_PREFIX and admitted and leyline_matches and execution["valid"]
            ),
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
    repetition: int = 0,
) -> dict[str, Any]:
    endpoints = config["arms"]
    model = config["model"]
    api_key = config.get("api_key")
    max_tokens = int(config.get("max_tokens", 64))
    mode, prompt_format, reference_tokens = evaluation_config(config)
    top_logprobs = int(config.get("diagnostics", {}).get("top_logprobs", 0))
    block_size = int(config.get("block_size", 128))
    plan = build_prompt_plan(
        tokenizer,
        case,
        prompt_format=prompt_format,
        block_size=block_size,
    )
    feasibility = plan_transform_feasibility(plan, block_size=block_size)
    arms: dict[str, Any] = {}
    run_id = str(config.get("run_id") or "adhoc")

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
            request_id=validation_request_id(run_id, case["id"], arm, repetition),
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
            request_id=validation_request_id(
                run_id, case["id"], "leyline-record", repetition
            ),
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
            request_id=validation_request_id(run_id, case["id"], "leyline", repetition),
        )
        result["record"] = record
        arms["leyline"] = result

    counterfactuals = []
    counterfactual_arm = "full" if "full" in endpoints else (
        "cache_off" if "cache_off" in endpoints else None
    )
    for index, removed in (
        enumerate(case.get("counterfactual_removed", []))
        if counterfactual_arm is not None
        else []
    ):
        variant = build_prompt_plan(
            tokenizer,
            case,
            removed,
            prompt_format=prompt_format,
            block_size=block_size,
        )
        result = request_completion(
            endpoints[counterfactual_arm],
            model,
            variant.full,
            api_key=api_key,
            max_tokens=max_tokens,
            cache_salt=uuid.uuid4().hex,
            top_logprobs=top_logprobs,
            request_id=validation_request_id(
                run_id,
                case["id"],
                counterfactual_arm,
                repetition,
                variant=f"counterfactual-{index}",
            ),
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
        target_token_ids=plan.target_token_ids,
    )
    transform_metadata = ((arms.get("leyline", {}).get("kv_transfer_params") or {}).get("leyline") or {})
    target_start = int(transform_metadata.get("local_apc_tokens", 0) or 0)
    transformed_tokens = int(transform_metadata.get("transformed_tokens", 0) or 0)
    return {
        "id": case["id"],
        "repetition": repetition,
        "category": case["category"],
        "family": case.get("family"),
        "diagnostic_family": case.get("diagnostic_family"),
        "claim_type": case.get("claim_type"),
        "prompt_tokens": {
            "full": len(plan.full),
            "edited": len(plan.edited),
            "deleted": plan.delete_end - plan.delete_start,
            "delete_start": plan.delete_start,
            "delete_end": plan.delete_end,
        },
        "oracle": case.get("oracle"),
        "expected_completion": case.get("expected_completion"),
        "target_token_ids": list(plan.target_token_ids),
        "workload_feasibility": {
            **feasibility.as_dict(),
            "surviving_filler_repeat": plan.surviving_filler_repeat,
        },
        "position_coverage": case.get("position_coverage"),
        "leyline_transform": {
            "target_start": target_start,
            "target_end": target_start + transformed_tokens,
            "transformed_tokens": transformed_tokens,
            "destination_block_start": target_start // block_size,
            "destination_block_end": (
                (target_start + transformed_tokens + block_size - 1) // block_size
                if transformed_tokens
                else target_start // block_size
            ),
            "normal_prefill_tokens": transform_metadata.get("normal_prefill_tokens"),
            "delete_delta": plan.delete_end - plan.delete_start,
        },
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


def _smoke_capture_evidence(
    result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = config.get("diagnostics", {})
    required = bool(
        config.get("smoke_gate", {}).get(
            "require_device_capture",
            diagnostics.get("device_capture_enabled", False),
        )
    )
    if not required:
        return {"required": False, "passed": True, "manifests": []}
    capture_dir_value = diagnostics.get("device_capture_dir")
    if not capture_dir_value:
        return {
            "required": True,
            "passed": False,
            "reason": "device_capture_dir_missing",
            "manifests": [],
        }
    capture_dir = Path(capture_dir_value).expanduser()
    request_id = result.get("arms", {}).get("leyline", {}).get("request_id")
    manifests: list[dict[str, Any]] = []
    if capture_dir.is_dir():
        for path in sorted(capture_dir.glob("*.manifest.json")):
            manifest = json.loads(path.read_text())
            matching = [
                item
                for item in manifest.get("captures", [])
                if (
                    item.get("request_id") == request_id
                    or (
                        request_id
                        and isinstance(item.get("request_id"), str)
                        and (
                            item["request_id"] == f"cmpl-{request_id}"
                            or item["request_id"].startswith(
                                f"cmpl-{request_id}-"
                            )
                        )
                    )
                )
            ]
            if matching:
                manifests.append(
                    {
                        "path": str(path),
                        "rank": manifest.get("rank"),
                        "complete_for_rank": manifest.get("complete_for_rank") is True,
                        "matching_captures": len(matching),
                    }
                )
    metadata = (
        (result.get("arms", {}).get("leyline", {}).get("kv_transfer_params") or {})
        .get("leyline", {})
    )
    expected_ranks = int(metadata.get("expected_ranks", 0) or 0)
    observed_ranks = {
        item["rank"] for item in manifests if item["complete_for_rank"]
    }
    passed = bool(
        request_id
        and expected_ranks > 0
        and len(observed_ranks) == expected_ranks
    )
    return {
        "required": True,
        "passed": passed,
        "request_id": request_id,
        "expected_ranks": expected_ranks,
        "observed_complete_ranks": sorted(observed_ranks),
        "manifests": manifests,
        "reason": None if passed else "capture_manifest_incomplete_or_missing",
    }


def evaluate_transform_smoke(
    result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    execution = result.get("leyline_execution", {})
    metadata = (
        (result.get("arms", {}).get("leyline", {}).get("kv_transfer_params") or {})
        .get("leyline", {})
    )
    block_size = int(config.get("block_size", 128))
    expected_layers = int(metadata.get("expected_layers", 0) or 0)
    transformed_layers = int(metadata.get("transformed_layers", 0) or 0)
    expected_ranks = int(metadata.get("expected_ranks", 0) or 0)
    successful_ranks = int(metadata.get("successful_ranks", 0) or 0)
    transformed_tokens = int(metadata.get("transformed_tokens", 0) or 0)
    capture = _smoke_capture_evidence(result, config)
    conditions = {
        "recorded": execution.get("recorded") is True,
        "applied": execution.get("applied") is True,
        "positive_block_aligned_transform": (
            transformed_tokens > 0 and transformed_tokens % block_size == 0
        ),
        "no_fallback": execution.get("no_fallback") is True,
        "transform_complete": execution.get("transform_complete") is True,
        "layer_complete": (
            expected_layers > 0 and transformed_layers == expected_layers
        ),
        "rank_complete": (
            expected_ranks > 0 and successful_ranks == expected_ranks
        ),
        "capture_complete": capture["passed"],
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "metadata": {
            "expected_layers": expected_layers,
            "transformed_layers": transformed_layers,
            "expected_ranks": expected_ranks,
            "successful_ranks": successful_ranks,
            "transformed_tokens": transformed_tokens,
            "fallback_reason": metadata.get("fallback_reason"),
        },
        "capture": capture,
    }


def run_transform_smoke(
    case: dict[str, Any],
    tokenizer: Any,
    config: dict[str, Any],
    *,
    preflight_passed: bool,
) -> dict[str, Any]:
    result = run_case(
        case,
        tokenizer,
        config,
        preflight_passed=preflight_passed,
    )
    evaluation = evaluate_transform_smoke(result, config)
    return {"case_id": case["id"], **evaluation, "result": result}


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
    plan = build_prompt_plan(
        tokenizer,
        case,
        prompt_format=prompt_format,
        block_size=int(config.get("block_size", 128)),
    )
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
    if mode in {REFERENCE_PREFIX, COMPLETION_TARGET}:
        return {"required": False, "passed": True, "skipped_reason": f"{mode}_mode"}
    settings = config.get("preflight", {})
    prompt = settings.get("prompt", 'Return only this JSON: {"ok":true}')
    oracle = settings.get("oracle", {"ok": True})
    endpoint = config.get("arms", {}).get("full")
    if not endpoint:
        return {"required": True, "passed": False, "error": "full/preflight endpoint is missing"}
    if settings.get("endpoint") not in {None, endpoint}:
        return {
            "required": True,
            "passed": False,
            "error": "preflight endpoint must be the configured full endpoint",
        }
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
    parser.add_argument("--workloads", type=Path, default=here / "workloads.base.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--numerical-report", type=Path)
    parser.add_argument("--logit-report", type=Path)
    parser.add_argument("--rollback-report", type=Path)
    parser.add_argument("--diagnostic-plan", type=Path)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def validate_workload_corpus(workloads: dict[str, Any], config: dict[str, Any]) -> None:
    mode, _, _ = evaluation_config(config)
    version = workloads.get("version")
    cases = workloads.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("workload corpus must contain cases")
    identifiers = [case.get("id") for case in cases]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("every workload case requires a non-empty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("workload case ids must be unique")
    if version == 1:
        if not config.get("legacy_diagnostic", False) or mode != REFERENCE_PREFIX:
            raise ValueError(
                "version-1 workloads require legacy_diagnostic=true with reference_prefix mode"
            )
        return
    if version != 2 or not isinstance(workloads.get("corpus_id"), str):
        raise ValueError("qualification workloads require version=2 and corpus_id")
    block_size = int(config.get("block_size", 128))
    for case in cases:
        if not case.get("family") or not case.get("claim_type"):
            raise ValueError(f"case {case['id']} requires family and claim_type")
        if (
            case.get("claim_type") == "retained_context_diagnostic"
            and not case.get("diagnostic_family")
        ):
            raise ValueError(
                f"retained-context case {case['id']} requires diagnostic_family"
            )
        expectation = case.get("execution_expectation")
        if expectation not in {"required", "diagnostic"}:
            raise ValueError(
                f"case {case['id']} requires execution_expectation=required|diagnostic"
            )
        minimum_transform_tokens = case.get("minimum_transform_tokens")
        if (
            not isinstance(minimum_transform_tokens, int)
            or minimum_transform_tokens < 0
            or minimum_transform_tokens % block_size
        ):
            raise ValueError(
                f"case {case['id']} minimum_transform_tokens must be non-negative "
                f"and aligned to block_size={block_size}"
            )
        if expectation == "required" and minimum_transform_tokens < block_size:
            raise ValueError(
                f"case {case['id']} requires at least one complete transform block"
            )
        if case.get("evaluation_mode") != mode:
            raise ValueError(
                f"case {case['id']} evaluation_mode must match configured mode {mode}"
            )
        if mode == COMPLETION_TARGET:
            target = case.get("expected_completion")
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"case {case['id']} requires expected_completion")
        if mode == STRUCTURED_JSON:
            oracle = case.get("oracle")
            if not isinstance(oracle, dict) or not oracle:
                raise ValueError(f"case {case['id']} requires a non-empty structured oracle")
        if case["family"] in {"counterfactual", "structured_counterfactual"} and len(
            case.get("counterfactual_removed", [])
        ) < 3:
            raise ValueError(
                f"counterfactual case {case['id']} requires at least three variants"
            )
    family_counts: dict[str, int] = {}
    for case in cases:
        family = case["family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    if mode == COMPLETION_TARGET:
        minimums = {
            "admissible": 6,
            "position_stress": 4,
            "counterfactual": 2,
            "mechanism_diagnostic": 4,
            "negative_control": 2,
        }
        missing = {
            family: required - family_counts.get(family, 0)
            for family, required in minimums.items()
            if family_counts.get(family, 0) < required
        }
        if len(cases) < 16 or missing:
            raise ValueError(f"base corpus family minimums are not met: {missing}")
        diagnostic_families = {
            case.get("diagnostic_family")
            for case in cases
            if case.get("claim_type") == "retained_context_diagnostic"
        }
        diagnostic_families.discard(None)
        if len(diagnostic_families) < 3:
            raise ValueError(
                "base corpus requires retained-context candidates across at least three families"
            )
    elif mode == STRUCTURED_JSON and len(cases) < 6:
        raise ValueError("Chat corpus requires at least six cases")


def workload_feasibility_report(
    cases: list[dict[str, Any]],
    tokenizer: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Plan every canonical/counterfactual prompt before network requests."""

    _, prompt_format, _ = evaluation_config(config)
    block_size = int(config.get("block_size", 128))
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        expectation = case.get("execution_expectation", "required")
        if expectation not in {"required", "diagnostic"}:
            raise ValueError(
                f"case {case['id']} has unsupported execution_expectation {expectation!r}"
            )
        minimum = int(
            case.get(
                "minimum_transform_tokens",
                block_size if expectation == "required" else 0,
            )
        )
        variants: list[tuple[str, str | None]] = [("canonical", None)]
        variants.extend(
            (f"counterfactual-{index}", removed)
            for index, removed in enumerate(case.get("counterfactual_removed", []))
        )
        for variant_id, removed in variants:
            try:
                plan = build_prompt_plan(
                    tokenizer,
                    case,
                    removed,
                    prompt_format=prompt_format,
                    block_size=block_size,
                )
            except (TypeError, ValueError) as exc:
                entry = {
                    "case_id": case["id"],
                    "variant": variant_id,
                    "execution_expectation": expectation,
                    "minimum_transform_tokens": minimum,
                    "surviving_filler_repeat": None,
                    "passed": False,
                    "predicted_transformed_tokens": 0,
                    "blocking_source_block": None,
                    "reason": "prompt_planning_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                entries.append(entry)
                failures.append(entry)
                continue
            feasibility = plan_transform_feasibility(plan, block_size=block_size)
            passed = (
                expectation == "diagnostic"
                or (
                    feasibility.predicted_transformed_tokens >= minimum
                    and feasibility.predicted_shifted_tokens > 0
                )
            )
            admission_reason = feasibility.reason
            if (
                expectation == "required"
                and feasibility.predicted_transformed_tokens >= minimum
                and feasibility.predicted_shifted_tokens == 0
            ):
                admission_reason = "delete_outside_reusable_range"
            entry = {
                "case_id": case["id"],
                "variant": variant_id,
                "execution_expectation": expectation,
                "minimum_transform_tokens": minimum,
                "surviving_filler_repeat": plan.surviving_filler_repeat,
                "passed": passed,
                **feasibility.as_dict(),
                "reason": admission_reason,
            }
            entries.append(entry)
            if not passed:
                failures.append(entry)
    return {"passed": not failures, "entries": entries, "failures": failures}


def validate_workload_feasibility(
    cases: list[dict[str, Any]],
    tokenizer: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    report = workload_feasibility_report(cases, tokenizer, config)
    if report["failures"]:
        summary = [
            {
                "case_id": item["case_id"],
                "variant": item["variant"],
                "reason": item["reason"],
                "predicted_transformed_tokens": item["predicted_transformed_tokens"],
                "predicted_shifted_tokens": item.get("predicted_shifted_tokens", 0),
                "minimum_transform_tokens": item["minimum_transform_tokens"],
                "blocking_source_block": item["blocking_source_block"],
                "error": item.get("error"),
            }
            for item in report["failures"]
        ]
        raise ValueError(
            "Leyline workload feasibility failed before requests: "
            f"{json.dumps(summary, sort_keys=True)}"
        )
    return report


def _stability_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [
        (
            run["gates"].get("full_matches"),
            run["gates"].get("honest_edited_matches"),
            run["gates"].get("leyline_matches"),
            run["gates"].get("leyline_execution_valid"),
        )
        for run in runs
    ]
    execution = [
        ((run.get("arms", {}).get("leyline", {}).get("kv_transfer_params") or {}).get("leyline") or {})
        for run in runs
    ]
    transformed = [item.get("transformed_tokens") for item in execution]
    complete = [item.get("transform_complete") for item in execution]
    fallback = [item.get("fallback_reason") for item in execution]
    execution_stable = (
        len(set(decisions)) == 1
        and len(set(transformed)) == 1
        and len(set(complete)) == 1
        and len(set(fallback)) == 1
    )
    generation_identities = [
        (
            tuple(
                (arm, tuple(result.get("output_token_ids") or ()))
                for arm, result in sorted(run.get("arms", {}).items())
            ),
            tuple(
                tuple(result.get("output_token_ids") or ())
                for result in run.get("counterfactuals", [])
            ),
        )
        for run in runs
    ]
    generation_stable = len(set(generation_identities)) == 1
    return {
        "repetitions": len(runs),
        # Schema-v2 compatibility: stable historically meant connector/gate
        # stability, not byte-for-byte generation stability.
        "stable": execution_stable,
        "execution_stable": execution_stable,
        "generation_stable": generation_stable,
        "gate_decisions": [list(item) for item in decisions],
        "transformed_tokens": transformed,
        "transform_complete": complete,
        "fallback_reason": fallback,
        "generation_output_token_ids": [
            {
                "arms": {
                    arm: result.get("output_token_ids")
                    for arm, result in sorted(run.get("arms", {}).items())
                },
                "counterfactuals": [
                    result.get("output_token_ids")
                    for result in run.get("counterfactuals", [])
                ],
            }
            for run in runs
        ],
    }


def run_case_repetitions(
    case: dict[str, Any],
    tokenizer: Any,
    config: dict[str, Any],
    *,
    preflight_passed: bool,
) -> dict[str, Any]:
    repetitions = int(config.get("correctness_repetitions", 1))
    if repetitions < 1:
        raise ValueError("correctness_repetitions must be at least 1")
    runs = [
        run_case(
            case,
            tokenizer,
            config,
            preflight_passed=preflight_passed,
            repetition=repetition,
        )
        for repetition in range(repetitions)
    ]
    result = dict(runs[0])
    result["repetitions"] = runs
    result["stability"] = _stability_summary(runs)
    durations = [
        float(
            (((run.get("arms", {}).get("leyline", {}).get("kv_transfer_params") or {}).get("leyline") or {}).get(
                "transform_duration_ms", 0.0
            ))
        )
        for run in runs
    ]
    result["transform_timing"] = {
        "cold_first_ms": durations[0] if durations else None,
        "warm_ms": durations[1:],
        "warm_mean_ms": statistics.fmean(durations[1:]) if len(durations) > 1 else None,
    }
    if not result["stability"]["stable"]:
        result["gates"] = {**result["gates"], "admitted": False, "leyline_accepted": False}
    return result


def expand_workload_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for case in cases:
        expanded.append(case)
        coverage = case.get("position_coverage") or {}
        for delete_tokens in coverage.get("delete_variants", []):
            expanded.append(
                {
                    **case,
                    "id": f"{case['id']}-delta-{delete_tokens}",
                    "position_coverage": {**coverage, "delete_tokens": delete_tokens, "delete_variants": []},
                    "notes": f"{case.get('notes', '')} Expanded deletion variant {delete_tokens}.",
                }
            )
    return expanded


def classify_case_result(
    case: dict[str, Any],
    *,
    environment_errors: list[str],
    numerical_state: str,
    rollback_state: str,
    baseline_state: str,
) -> str:
    return classify_case(
        case,
        environment_state="failed" if environment_errors else "passed",
        numerical_state=numerical_state,
        rollback_state=rollback_state,
        baseline_state=baseline_state,
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    diagnostic_plan = (
        json.loads(args.diagnostic_plan.read_text()) if args.diagnostic_plan else None
    )
    if diagnostic_plan is not None:
        if diagnostic_plan.get("schema_version") != 1:
            raise ValueError("unsupported divergence diagnostic plan schema")
        config["run_id"] = diagnostic_plan["target_run_id"]
        config["smoke_gate"] = {"enabled": False}
        selected_arms = {"full", "honest_edited", "leyline"}
        config["arms"] = {
            arm: endpoint
            for arm, endpoint in (config.get("arms") or {}).items()
            if arm in selected_arms
        }
    ensure_run_id(config)
    workloads = json.loads(args.workloads.read_text())
    validate_workload_corpus(workloads, config)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"],
        revision=config.get("tokenizer_revision"),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    expanded_cases = expand_workload_cases(workloads["cases"])
    if diagnostic_plan is not None:
        selected = set(diagnostic_plan.get("case_ids") or [])
        expanded_cases = [case for case in expanded_cases if case["id"] in selected]
        if not expanded_cases:
            raise ValueError("diagnostic plan selected no workload cases")
    if config.get("legacy_diagnostic", False):
        feasibility: dict[str, Any] = {
            "passed": None,
            "skipped_reason": "legacy_diagnostic_mode",
            "entries": [],
            "failures": [],
        }
    else:
        feasibility = validate_workload_feasibility(
            expanded_cases,
            tokenizer,
            config,
        )
    preflight = run_preflight(tokenizer, config)
    allow_diagnostics = bool(config.get("diagnostics", {}).get("continue_after_preflight_failure", True))
    smoke_settings = config.get("smoke_gate", {})
    smoke: dict[str, Any] = {
        "enabled": bool(smoke_settings.get("enabled", False)),
        "passed": None,
        "skipped_reason": "disabled",
    }
    if smoke["enabled"] and (preflight["passed"] or allow_diagnostics):
        smoke_case_id = smoke_settings.get("case_id")
        smoke_case = next(
            (case for case in expanded_cases if case["id"] == smoke_case_id),
            None,
        )
        if smoke_case is None:
            raise ValueError(
                f"smoke_gate.case_id {smoke_case_id!r} is not present in the expanded corpus"
            )
        smoke = {
            "enabled": True,
            "skipped_reason": None,
            **run_transform_smoke(
                smoke_case,
                tokenizer,
                config,
                preflight_passed=bool(preflight["passed"]),
            ),
        }
    if not preflight["passed"] and not allow_diagnostics:
        cases: list[dict[str, Any]] = []
    elif smoke["enabled"] and smoke["passed"] is not True:
        cases = []
    else:
        cases = [
            run_case_repetitions(
                case,
                tokenizer,
                config,
                preflight_passed=bool(preflight["passed"]),
            )
            for case in expanded_cases
        ]
    report: dict[str, Any] = {
        "schema_version": 2,
        "created_unix": time.time(),
        "corpus_id": workloads.get("corpus_id", "legacy-v1"),
        "workload_version": workloads.get("version"),
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
        "workload_feasibility": feasibility,
        "smoke_gate": smoke,
        "cases": cases,
    }
    if diagnostic_plan is not None:
        report["diagnostic_plan"] = diagnostic_plan
    numerical_report = (
        json.loads(args.numerical_report.read_text()) if args.numerical_report else None
    )
    if numerical_report is not None:
        report["numerical_validation"] = numerical_report
    if args.logit_report:
        report["raw_logit_validation"] = json.loads(args.logit_report.read_text())
    rollback_report = (
        json.loads(args.rollback_report.read_text()) if args.rollback_report else None
    )
    if rollback_report is not None:
        report["rollback_validation"] = rollback_report
    environment = json.loads(args.environment.read_text()) if args.environment else None
    if environment:
        report["environment"] = environment
        report["checkpoint_identity"] = report["environment"].get("model")
    report["evidence_identity"] = report_identity(report)
    report["retained_context_diagnostics"] = retained_context_summary(report)
    blockers = environment_blockers(environment)
    prerequisites = config.get("performance_prerequisites", {})
    numerical_passed = bool(
        numerical_report is not None and not cache_comparison_errors(numerical_report)
    )
    rollback_passed = bool(
        rollback_report is not None and not rollback_report_errors(rollback_report)
    )
    numerical_state = (
        "missing"
        if numerical_report is None
        else ("passed" if numerical_passed else "failed")
    )
    rollback_state = (
        "missing"
        if rollback_report is None
        else ("passed" if rollback_passed else "failed")
    )
    effective_config = {
        **config,
        "performance_prerequisites": {
            **prerequisites,
            "numerical_passed": numerical_passed,
            "rollback_passed": rollback_passed,
        },
    }
    report["qualification"] = {
        "environment_blockers": blockers,
        "numerical_passed": numerical_passed,
        "rollback_passed": rollback_passed,
        "case_classifications": {
            case["id"]: classify_case_result(
                case,
                environment_errors=blockers,
                numerical_state=numerical_state,
                rollback_state=rollback_state,
                baseline_state="missing",
            )
            for case in cases
        },
    }
    if args.performance:
        gate = performance_gate(cases, effective_config, preflight)
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
