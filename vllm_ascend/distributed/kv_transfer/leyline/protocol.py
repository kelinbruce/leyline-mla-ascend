# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Request-level protocol for the Leyline v0 connector."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class LeylineAction(StrEnum):
    RECORD = "record"
    AMORTIZE = "amortize"


class LeylineFallbackReason(StrEnum):
    INVALID_DIRECTIVE = "invalid_directive"
    INVALID_EDIT = "invalid_edit"
    TOKEN_MISMATCH = "token_mismatch"
    MISSING_SOURCE = "missing_source"
    INCOMPATIBLE_IDENTITY = "incompatible_identity"
    UNSUPPORTED_RUNTIME = "unsupported_runtime"
    SOURCE_BLOCK_MISSING = "source_block_missing"
    NO_REUSABLE_BLOCKS = "no_reusable_blocks"
    TRANSFORM_FAILED = "transform_failed"


class LeylineDirectiveError(ValueError):
    def __init__(self, reason: LeylineFallbackReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class DeleteSpan:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class LeylineFaultInjection:
    rank: int
    layer: int
    stage: str = "after_layer_write"


@dataclass(frozen=True)
class LeylineDirective:
    version: int
    action: LeylineAction
    session_id: str
    delete: DeleteSpan | None = None
    fault_injection: LeylineFaultInjection | None = None


def _require_int(value: Any, field: str) -> int:
    # bool is an int subclass but must not be accepted as a token offset.
    if isinstance(value, bool) or not isinstance(value, int):
        raise LeylineDirectiveError(
            LeylineFallbackReason.INVALID_DIRECTIVE,
            f"leyline.{field} must be an integer",
        )
    return value


def parse_leyline_directive(
    kv_transfer_params: dict[str, Any] | None,
) -> LeylineDirective | None:
    """Parse the namespaced connector directive.

    Absence is not an error: requests without a ``leyline`` object use normal
    vLLM cache behavior. Malformed opt-in directives raise a typed error so the
    connector can expose a stable fallback reason without logging prompt data.
    """

    if not kv_transfer_params or "leyline" not in kv_transfer_params:
        return None

    raw = kv_transfer_params["leyline"]
    if not isinstance(raw, dict):
        raise LeylineDirectiveError(
            LeylineFallbackReason.INVALID_DIRECTIVE,
            "kv_transfer_params.leyline must be an object",
        )

    version = _require_int(raw.get("version"), "version")
    if version != 1:
        raise LeylineDirectiveError(
            LeylineFallbackReason.INVALID_DIRECTIVE,
            f"unsupported leyline protocol version: {version}",
        )

    try:
        action = LeylineAction(raw.get("action"))
    except (TypeError, ValueError) as exc:
        raise LeylineDirectiveError(
            LeylineFallbackReason.INVALID_DIRECTIVE,
            "leyline.action must be 'record' or 'amortize'",
        ) from exc

    session_id = raw.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise LeylineDirectiveError(
            LeylineFallbackReason.INVALID_DIRECTIVE,
            "leyline.session_id must be a non-empty string",
        )

    raw_delete = raw.get("delete")
    raw_fault = raw.get("fault_injection")
    if action is LeylineAction.RECORD:
        if raw_delete is not None or raw_fault is not None:
            raise LeylineDirectiveError(
                LeylineFallbackReason.INVALID_DIRECTIVE,
                "record directives must not contain delete or fault injection",
            )
        return LeylineDirective(version, action, session_id)

    if not isinstance(raw_delete, dict):
        raise LeylineDirectiveError(
            LeylineFallbackReason.INVALID_DIRECTIVE,
            "amortize directives require a delete object",
        )
    start = _require_int(raw_delete.get("start"), "delete.start")
    end = _require_int(raw_delete.get("end"), "delete.end")
    if start < 0 or end <= start:
        raise LeylineDirectiveError(
            LeylineFallbackReason.INVALID_EDIT,
            "delete offsets must satisfy 0 <= start < end",
        )
    fault = None
    if raw_fault is not None:
        if not isinstance(raw_fault, dict):
            raise LeylineDirectiveError(
                LeylineFallbackReason.INVALID_DIRECTIVE,
                "leyline.fault_injection must be an object",
            )
        rank = _require_int(raw_fault.get("rank"), "fault_injection.rank")
        layer = _require_int(raw_fault.get("layer"), "fault_injection.layer")
        stage = raw_fault.get("stage")
        if rank < 0 or layer < 0 or stage != "after_layer_write":
            raise LeylineDirectiveError(
                LeylineFallbackReason.INVALID_DIRECTIVE,
                "fault injection requires non-negative rank/layer and stage=after_layer_write",
            )
        fault = LeylineFaultInjection(rank=rank, layer=layer, stage=stage)
    return LeylineDirective(version, action, session_id, DeleteSpan(start, end), fault)
