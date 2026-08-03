# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Leyline MLA cache amortization primitives.

The connector is deliberately not imported here.  Keeping the protocol and
mapping helpers free of vLLM and torch imports makes their correctness tests
runnable on development machines without an Ascend software stack.
"""

from .mapping import (
    DeletionMapping,
    SlotMapping,
    build_slot_mapping,
    find_reusable_target_end,
    validate_deletion,
)
from .protocol import (
    DeleteSpan,
    LeylineAction,
    LeylineDirective,
    LeylineDirectiveError,
    LeylineFallbackReason,
    parse_leyline_directive,
)

__all__ = [
    "DeleteSpan",
    "DeletionMapping",
    "LeylineAction",
    "LeylineDirective",
    "LeylineDirectiveError",
    "LeylineFallbackReason",
    "SlotMapping",
    "build_slot_mapping",
    "find_reusable_target_end",
    "parse_leyline_directive",
    "validate_deletion",
]
