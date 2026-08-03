# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Token and physical-slot mapping for one Leyline deletion."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from .protocol import DeleteSpan, LeylineDirectiveError, LeylineFallbackReason


@dataclass(frozen=True)
class DeletionMapping:
    source_length: int
    edited_source_length: int
    delete: DeleteSpan

    def source_position(self, target_position: int) -> int:
        if not 0 <= target_position < self.edited_source_length:
            raise IndexError(f"target position out of edited source: {target_position}")
        if target_position < self.delete.start:
            return target_position
        return target_position + self.delete.length


@dataclass(frozen=True)
class SlotMapping:
    source_slots: tuple[int, ...]
    destination_slots: tuple[int, ...]
    old_positions: tuple[int, ...]
    new_positions: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.source_slots)


def validate_deletion(
    source_tokens: Sequence[int],
    edited_tokens: Sequence[int],
    delete: DeleteSpan,
) -> DeletionMapping:
    """Validate an edited request and return its constant-shift mapping.

    Extra tokens after the edited source are allowed. They represent the new
    turn and are normally prefetched after the transformed reusable prefix.
    """

    if delete.start < 0 or delete.end <= delete.start or delete.end > len(source_tokens):
        raise LeylineDirectiveError(
            LeylineFallbackReason.INVALID_EDIT,
            "delete span is outside the recorded source token sequence",
        )

    expected = tuple(source_tokens[: delete.start]) + tuple(source_tokens[delete.end :])
    if len(edited_tokens) < len(expected) or tuple(edited_tokens[: len(expected)]) != expected:
        raise LeylineDirectiveError(
            LeylineFallbackReason.TOKEN_MISMATCH,
            "edited prompt does not equal the recorded source with the declared span removed",
        )
    return DeletionMapping(len(source_tokens), len(expected), delete)


def find_reusable_target_end(
    deletion: DeletionMapping,
    local_computed_tokens: int,
    max_target_tokens: int,
    resident_source_blocks: Collection[int],
    block_size: int,
) -> int:
    """Return the longest full target-block boundary that can be transformed."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if local_computed_tokens < 0 or local_computed_tokens % block_size:
        raise ValueError("local_computed_tokens must be block aligned")

    candidate_end = min(deletion.edited_source_length, max_target_tokens)
    candidate_end -= candidate_end % block_size
    reusable_end = local_computed_tokens
    for target_end in range(local_computed_tokens + block_size, candidate_end + 1, block_size):
        target_start = target_end - block_size
        required = {
            deletion.source_position(position) // block_size
            for position in range(target_start, target_end)
        }
        if not required.issubset(resident_source_blocks):
            break
        reusable_end = target_end
    return reusable_end


def build_slot_mapping(
    deletion: DeletionMapping,
    target_start: int,
    target_end: int,
    source_block_ids: Mapping[int, int],
    destination_block_ids: Sequence[int],
    block_size: int,
) -> SlotMapping:
    """Build flat physical source/destination slots for a transformed range."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if target_start < 0 or target_end < target_start:
        raise ValueError("invalid target range")
    if target_start % block_size or target_end % block_size:
        raise ValueError("target range must be block aligned")
    if target_end > deletion.edited_source_length:
        raise ValueError("target range exceeds edited source")

    source_slots: list[int] = []
    destination_slots: list[int] = []
    old_positions: list[int] = []
    new_positions: list[int] = []
    for new_position in range(target_start, target_end):
        old_position = deletion.source_position(new_position)
        source_block_index, source_offset = divmod(old_position, block_size)
        destination_block_index, destination_offset = divmod(new_position, block_size)
        try:
            source_block_id = source_block_ids[source_block_index]
        except KeyError as exc:
            raise KeyError(f"missing source block index {source_block_index}") from exc
        if destination_block_index >= len(destination_block_ids):
            raise IndexError(f"missing destination block index {destination_block_index}")

        source_slots.append(source_block_id * block_size + source_offset)
        destination_slots.append(destination_block_ids[destination_block_index] * block_size + destination_offset)
        old_positions.append(old_position)
        new_positions.append(new_position)

    return SlotMapping(
        tuple(source_slots),
        tuple(destination_slots),
        tuple(old_positions),
        tuple(new_positions),
    )
