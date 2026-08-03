# Leyline MLA cache amortization (experimental)

Leyline reuses DeepSeek MLA cache blocks after a caller-declared, semantically admissible deletion. It is experimental, opt-in, and is not a secure forgetting mechanism.

## v0 request schema

The connector reads a namespaced object from the existing `kv_transfer_params` request field.

Record the completed source conversation under a scheduler-local session identifier:

```json
{
  "kv_transfer_params": {
    "leyline": {
      "version": 1,
      "action": "record",
      "session_id": "agent-session-42"
    }
  }
}
```

Request cache amortization after deleting one token-index span:

```json
{
  "kv_transfer_params": {
    "leyline": {
      "version": 1,
      "action": "amortize",
      "session_id": "agent-session-42",
      "delete": {"start": 1024, "end": 1536}
    }
  }
}
```

Offsets refer to tokens in the recorded source sequence. The edited prompt must equal that source with `[start, end)` removed; tokens for a new turn may follow the edited source. v0 supports one non-empty deletion only. Non-empty replacement and multiple edits use normal prefill.

## Fallback reasons

| Reason | Meaning |
|---|---|
| `invalid_directive` | The opt-in object is malformed or uses an unsupported protocol version. |
| `invalid_edit` | The deletion is empty, reversed, or outside the recorded source. |
| `token_mismatch` | Tokens outside the deletion do not match the recorded source. |
| `missing_source` | The session was not recorded or has expired. |
| `incompatible_identity` | Model, tokenizer, RoPE, LoRA, cache layout, or tenant identity differs. |
| `unsupported_runtime` | The request uses a runtime mode outside the v0 support matrix. |
| `source_block_missing` | A required source APC block is no longer resident. |
| `no_reusable_blocks` | No complete destination block remains worth transforming. |
| `transform_failed` | A worker could not complete the cache transformation. |

Any fallback leaves ordinary vLLM-Ascend prefill behavior in effect. Metrics and logs must report only the reason and token/block counts, never prompt contents.
