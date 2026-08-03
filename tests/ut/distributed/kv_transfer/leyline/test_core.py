# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.leyline.mapping import (
    build_slot_mapping,
    find_reusable_target_end,
    validate_deletion,
)
from vllm_ascend.distributed.kv_transfer.leyline.protocol import (
    DeleteSpan,
    LeylineAction,
    LeylineDirectiveError,
    LeylineFallbackReason,
    parse_leyline_directive,
)
from vllm_ascend.distributed.kv_transfer.leyline.reference import (
    deepseek_yarn_inv_freq,
    rotate_kpe_half_split,
    unit_delta_cos_sin,
)


def test_parse_record_and_amortize_directives():
    record = parse_leyline_directive(
        {"leyline": {"version": 1, "action": "record", "session_id": "agent-1"}}
    )
    assert record is not None
    assert record.action is LeylineAction.RECORD
    assert record.delete is None

    amortize = parse_leyline_directive(
        {
            "leyline": {
                "version": 1,
                "action": "amortize",
                "session_id": "agent-1",
                "delete": {"start": 4, "end": 9},
            }
        }
    )
    assert amortize is not None
    assert amortize.action is LeylineAction.AMORTIZE
    assert amortize.delete == DeleteSpan(4, 9)


@pytest.mark.parametrize(
    "params,reason",
    [
        ({"leyline": []}, LeylineFallbackReason.INVALID_DIRECTIVE),
        ({"leyline": {"version": 2, "action": "record", "session_id": "s"}}, LeylineFallbackReason.INVALID_DIRECTIVE),
        (
            {"leyline": {"version": 1, "action": "amortize", "session_id": "s", "delete": {"start": 3, "end": 3}}},
            LeylineFallbackReason.INVALID_EDIT,
        ),
    ],
)
def test_invalid_directives_have_stable_reasons(params, reason):
    with pytest.raises(LeylineDirectiveError) as exc:
        parse_leyline_directive(params)
    assert exc.value.reason is reason


def test_validate_deletion_allows_new_tail_and_rejects_mismatch():
    source = list(range(20))
    edited = source[:5] + source[9:] + [100, 101]
    mapping = validate_deletion(source, edited, DeleteSpan(5, 9))
    assert mapping.edited_source_length == 16
    assert mapping.source_position(4) == 4
    assert mapping.source_position(5) == 9

    with pytest.raises(LeylineDirectiveError) as exc:
        validate_deletion(source, edited[:8] + [999] + edited[9:], DeleteSpan(5, 9))
    assert exc.value.reason is LeylineFallbackReason.TOKEN_MISMATCH


def test_reusable_end_stops_before_missing_source_block_and_partial_tail():
    block_size = 4
    source = list(range(19))
    edited = source[:3] + source[6:]
    deletion = validate_deletion(source, edited, DeleteSpan(3, 6))

    assert find_reusable_target_end(deletion, 0, len(edited), {0, 1, 2, 3}, block_size) == 12
    assert find_reusable_target_end(deletion, 0, len(edited), {0, 1, 2, 3, 4}, block_size) == 16


def test_slot_mapping_repacks_rows_across_blocks():
    block_size = 4
    source = list(range(16))
    edited = source[:3] + source[6:]
    deletion = validate_deletion(source, edited, DeleteSpan(3, 6))
    mapping = build_slot_mapping(
        deletion,
        target_start=0,
        target_end=12,
        source_block_ids={0: 10, 1: 11, 2: 12, 3: 13},
        destination_block_ids=[20, 21, 22],
        block_size=block_size,
    )
    assert mapping.old_positions[:5] == (0, 1, 2, 6, 7)
    assert mapping.source_slots[:5] == (40, 41, 42, 46, 47)
    assert mapping.destination_slots[:5] == (80, 81, 82, 83, 84)


@pytest.mark.parametrize("delta", [0, 1, 17, 127, 128, 129, 1024])
def test_unit_delta_rotation_matches_direct_rotation(delta):
    inv_freq = deepseek_yarn_inv_freq(64, 10000.0, 40.0, 4096)
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((1, 1, 64), dtype=np.float32)
    mscale = np.float32(1.17)
    old_position = 8192
    old_cos, old_sin = unit_delta_cos_sin([0], [old_position], inv_freq)
    old_kpe = rotate_kpe_half_split(raw, old_cos, old_sin) * mscale

    delta_cos, delta_sin = unit_delta_cos_sin([old_position], [old_position - delta], inv_freq)
    transformed = rotate_kpe_half_split(old_kpe, delta_cos, delta_sin)
    direct_cos, direct_sin = unit_delta_cos_sin([0], [old_position - delta], inv_freq)
    expected = rotate_kpe_half_split(raw, direct_cos, direct_sin) * mscale

    np.testing.assert_allclose(transformed, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.linalg.norm(transformed), np.linalg.norm(old_kpe), rtol=2e-5, atol=2e-5)
