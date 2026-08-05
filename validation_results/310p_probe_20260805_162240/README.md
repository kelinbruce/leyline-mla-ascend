# Ascend 310P Leyline MLA Probe Results

## Summary

- Overall result: **failed**
- Device topology: physical NPU 4, 5, 6, 7 mapped to logical rank 0, 1, 2, 3
- All four ranks produced consistent probe results.
- Leyline cache gather/scatter and Kpe rotation passed.
- The normal 310P MLA cache-write, prefill-attention and paged-decode paths failed because the current operator definitions do not register Ascend 310P support.

## Environment

- Device: `Ascend310P3`
- PyTorch: `2.10.0+cpu`
- torch_npu: `2.10.0.post1.dev20260620`
- vLLM package: `0.23.0+empty`
- vLLM-Ascend package: `None`
- Leyline checkout: `8e787206eaedc7dcb5ee205161066f39d463fa37`
- vLLM checkout record: `72506c98349d6bcd32b4e33eec7b5513453c1502`

## Probe Matrix

| Probe | Rank 0 | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|---|
| `required_operator_presence` | PASS | PASS | PASS | PASS |
| `basic_fp16_matmul` | PASS | PASS | PASS | PASS |
| `fp32_cos_sin` | FAIL | FAIL | FAIL | FAIL |
| `fp16_cache_gather_scatter` | PASS | PASS | PASS | PASS |
| `fp16_leyline_rotation` | PASS | PASS | PASS | PASS |
| `fp16_mla_cache_write` | FAIL | FAIL | FAIL | FAIL |
| `fp16_mla_prefill_attention` | FAIL | FAIL | FAIL | FAIL |
| `fp16_mla_paged_decode` | FAIL | FAIL | FAIL | FAIL |
| `fp16_mla_chunked_cache_load` | FAIL | FAIL | FAIL | FAIL |

## Failure Details

### Logical rank 0

#### `fp32_cos_sin`

- Error type: `AssertionError`

```text
Tensor-likes are not close!

Mismatched elements: 66 / 4096 (1.6%)
Greatest absolute difference: 5.2094459533691406e-05 at index (419,) (up to 2e-05 allowed)
Greatest relative difference: 5.2094459533691406e-05 at index (419,) (up to 2e-05 allowed)
```

#### `fp16_mla_cache_write`

- Error type: `RuntimeError`

```text
npu_kv_rmsnorm_rope_cache:../third_party/op-plugin/op_plugin/ops/opapi/KvRmsNormRopeCacheNpuOpApi.cpp:88 NPU function error: call aclnnKvRmsNormRopeCache failed, error code is 161001
[ERROR] 2026-08-05-16:04:44 (PID:22287, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 22287] 2026-08-05-16:04:44.378.535 Execution_Error(EZ1009): Failed to execute operator KvRmsNormRopeCache_0. Reason: SoC version ascend310p verification failed. This SoC is not configured through the AddConfig API of the OpDef class.
TraceBack (most recent call last):
        Check nnopExecutor != nullptr failed
        Assert ((executor) != nullptr) failed
```

#### `fp16_mla_prefill_attention`

- Error type: `RuntimeError`

```text
npu_fused_infer_attention_score_symint:../third_party/op-plugin/op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp:425 NPU function error: call aclnnFusedInferAttentionScoreV3 failed, error code is 161001
[ERROR] 2026-08-05-16:04:44 (PID:22287, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 22287] 2026-08-05-16:04:44.382.715 Execution_Error(EZ1009): Failed to execute operator FusedInferAttentionScore_0. Reason: SoC version ascend310p verification failed. This SoC is not configured through the AddConfig API of the OpDef class.
TraceBack (most recent call last):
        Check nnopExecutor != nullptr failed
```

#### `fp16_mla_paged_decode`

- Error type: `RuntimeError`

```text
npu_fused_infer_attention_score_v2_symint:../third_party/op-plugin/op_plugin/ops/opapi/FusedInferAttentionScoreV2KernelNpuOpApi.cpp:459 NPU function error: call aclnnFusedInferAttentionScoreV4 failed, error code is 161001
[ERROR] 2026-08-05-16:04:46 (PID:22287, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 22287] 2026-08-05-16:04:46.565.948 Execution_Error(EZ1009): Failed to execute operator FusedInferAttentionScore_0. Reason: SoC version ascend310p verification failed. This SoC is not configured through the AddConfig API of the OpDef class.
TraceBack (most recent call last):
        Check nnopExecutor != nullptr failed
```

#### `fp16_mla_chunked_cache_load`

- Error type: `RuntimeError`

```text
The Inner error is reported as above. The process exits for this inner error, and the current working operator name is AtbPagedCacheLoad.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, please set the environment variable ASCEND_LAUNCH_BLOCKING=1.
Note: ASCEND_LAUNCH_BLOCKING=1 will force ops to run in synchronous mode, resulting in performance degradation. Please unset ASCEND_LAUNCH_BLOCKING in time after debugging.
[ERROR] 2026-08-05-16:04:47 (PID:22287, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 22287] 2026-08-05-16:04:20.485.292 Invalid_Argument(EH0012): aclrtAllocatorGetByStreamImpl failed, Parameter stream is invalid. Reason: The stream is not registered with any allocator.
```

### Logical rank 1

#### `fp32_cos_sin`

- Error type: `AssertionError`

```text
Tensor-likes are not close!

Mismatched elements: 66 / 4096 (1.6%)
Greatest absolute difference: 5.2094459533691406e-05 at index (419,) (up to 2e-05 allowed)
Greatest relative difference: 5.2094459533691406e-05 at index (419,) (up to 2e-05 allowed)
```

#### `fp16_mla_cache_write`

- Error type: `RuntimeError`

```text
npu_kv_rmsnorm_rope_cache:../third_party/op-plugin/op_plugin/ops/opapi/KvRmsNormRopeCacheNpuOpApi.cpp:88 NPU function error: call aclnnKvRmsNormRopeCache failed, error code is 161001
[ERROR] 2026-08-05-16:05:33 (PID:27599, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 27599] 2026-08-05-16:05:33.683.012 Execution_Error(EZ1009): Failed to execute operator KvRmsNormRopeCache_0. Reason: SoC version ascend310p verification failed. This SoC is not configured through the AddConfig API of the OpDef class.
TraceBack (most recent call last):
        Check nnopExecutor != nullptr failed
        Assert ((executor) != nullptr) failed
```

#### `fp16_mla_prefill_attention`

- Error type: `RuntimeError`

```text
npu_fused_infer_attention_score_symint:../third_party/op-plugin/op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp:425 NPU function error: call aclnnFusedInferAttentionScoreV3 failed, error code is 161001
[ERROR] 2026-08-05-16:05:33 (PID:27599, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 27599] 2026-08-05-16:05:33.687.166 Execution_Error(EZ1009): Failed to execute operator FusedInferAttentionScore_0. Reason: SoC version ascend310p verification failed. This SoC is not configured through the AddConfig API of the OpDef class.
TraceBack (most recent call last):
        Check nnopExecutor != nullptr failed
```

#### `fp16_mla_paged_decode`

- Error type: `RuntimeError`

```text
npu_fused_infer_attention_score_v2_symint:../third_party/op-plugin/op_plugin/ops/opapi/FusedInferAttentionScoreV2KernelNpuOpApi.cpp:459 NPU function error: call aclnnFusedInferAttentionScoreV4 failed, error code is 161001
[ERROR] 2026-08-05-16:05:35 (PID:27599, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 27599] 2026-08-05-16:05:35.870.832 Execution_Error(EZ1009): Failed to execute operator FusedInferAttentionScore_0. Reason: SoC version ascend310p verification failed. This SoC is not configured through the AddConfig API of the OpDef class.
TraceBack (most recent call last):
        Check nnopExecutor != nullptr failed
```

#### `fp16_mla_chunked_cache_load`

- Error type: `RuntimeError`

```text
The Inner error is reported as above. The process exits for this inner error, and the current working operator name is AtbPagedCacheLoad.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, please set the environment variable ASCEND_LAUNCH_BLOCKING=1.
Note: ASCEND_LAUNCH_BLOCKING=1 will force ops to run in synchronous mode, resulting in performance degradation. Please unset ASCEND_LAUNCH_BLOCKING in time after debugging.
[ERROR] 2026-08-05-16:05:36 (PID:27599, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 27599] 2026-08-05-16:05:10.469.446 Invalid_Argument(EH0012): aclrtAllocatorGetByStreamImpl failed, Parameter stream is invalid. Reason: The stream is not registered with any allocator.
```

### Logical rank 2

#### `fp32_cos_sin`

- Error type: `AssertionError`

```text
Tensor-likes are not close!

Mismatched elements: 66 / 4096 (1.6%)
Greatest absolute difference: 5.2094459533691406e-05 at index (419,) (up to 2e-05 allowed)
Greatest relative difference: 5.2094459533691406e-05 at index (419,) (up to 2e-05 allowed)
```

#### `fp16_mla_cache_write`

- Error type: `RuntimeError`

```text
npu_kv_rmsnorm_rope_cache:../third_party/op-plugin/op_plugin/ops/opapi/KvRmsNormRopeCacheNpuOpApi.cpp:88 NPU function error: call aclnnKvRmsNormRopeCache failed, error code is 161001
[ERROR] 2026-08-05-16:06:23 (PID:32909, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 32909] 2026-08-05-16:06:23.880.755 Execution_Error(EZ1009): Failed to execute operator KvRmsNormRopeCache_0. Reason: SoC version ascend310p verification failed. This SoC is not configured through the AddConfig API of the OpDef class.
TraceBack (most recent call last):
        Check nnopExecutor != nullptr failed
        Assert ((executor) != nullptr) failed
```

#### `fp16_mla_prefill_attention`

- Error type: `RuntimeError`

```text
npu_fused_infer_attention_score_symint:../third_party/op-plugin/op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp:425 NPU function error: call aclnnFusedInferAttentionScoreV3 failed, error code is 161001
[ERROR] 2026-08-05-16:06:23 (PID:32909, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 32909] 2026-08-05-16:06:23.886.265 Execution_Error(EZ1009): Failed to execute operator FusedInferAttentionScore_0. Reason: SoC version ascend310p verification failed. This SoC is not configured through the AddConfig API of the OpDef class.
TraceBack (most recent call last):
        Check nnopExecutor != nullptr failed
```

#### `fp16_mla_paged_decode`

- Error type: `RuntimeError`

```text
npu_fused_infer_attention_score_v2_symint:../third_party/op-plugin/op_plugin/ops/opapi/FusedInferAttentionScoreV2KernelNpuOpApi.cpp:459 NPU function error: call aclnnFusedInferAttentionScoreV4 failed, error code is 161001
[ERROR] 2026-08-05-16:06:25 (PID:32909, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 32909] 2026-08-05-16:06:25.920.487 Execution_Error(EZ1009): Failed to execute operator FusedInferAttentionScore_0. Reason: SoC version ascend310p verification failed. This SoC is not configured through the AddConfig API of the OpDef class.
TraceBack (most recent call last):
        Check nnopExecutor != nullptr failed
```

#### `fp16_mla_chunked_cache_load`

- Error type: `RuntimeError`

```text
The Inner error is reported as above. The process exits for this inner error, and the current working operator name is AtbPagedCacheLoad.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, please set the environment variable ASCEND_LAUNCH_BLOCKING=1.
Note: ASCEND_LAUNCH_BLOCKING=1 will force ops to run in synchronous mode, resulting in performance degradation. Please unset ASCEND_LAUNCH_BLOCKING in time after debugging.
[ERROR] 2026-08-05-16:06:26 (PID:32909, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 32909] 2026-08-05-16:06:00.554.099 Invalid_Argument(EH0012): aclrtAllocatorGetByStreamImpl failed, Parameter stream is invalid. Reason: The stream is not registered with any allocator.
```

### Logical rank 3

#### `fp32_cos_sin`

- Error type: `AssertionError`

```text
Tensor-likes are not close!

Mismatched elements: 66 / 4096 (1.6%)
Greatest absolute difference: 5.2094459533691406e-05 at index (419,) (up to 2e-05 allowed)
Greatest relative difference: 5.2094459533691406e-05 at index (419,) (up to 2e-05 allowed)
```

#### `fp16_mla_cache_write`

- Error type: `RuntimeError`

```text
npu_kv_rmsnorm_rope_cache:../third_party/op-plugin/op_plugin/ops/opapi/KvRmsNormRopeCacheNpuOpApi.cpp:88 NPU function error: call aclnnKvRmsNormRopeCache failed, error code is 161001
[ERROR] 2026-08-05-16:07:13 (PID:38221, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 38221] 2026-08-05-16:07:13.804.380 Execution_Error(EZ1009): Failed to execute operator KvRmsNormRopeCache_0. Reason: SoC version ascend310p verification failed. This SoC is not configured through the AddConfig API of the OpDef class.
TraceBack (most recent call last):
        Check nnopExecutor != nullptr failed
        Assert ((executor) != nullptr) failed
```

#### `fp16_mla_prefill_attention`

- Error type: `RuntimeError`

```text
npu_fused_infer_attention_score_symint:../third_party/op-plugin/op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp:425 NPU function error: call aclnnFusedInferAttentionScoreV3 failed, error code is 161001
[ERROR] 2026-08-05-16:07:13 (PID:38221, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 38221] 2026-08-05-16:07:13.809.030 Execution_Error(EZ1009): Failed to execute operator FusedInferAttentionScore_0. Reason: SoC version ascend310p verification failed. This SoC is not configured through the AddConfig API of the OpDef class.
TraceBack (most recent call last):
        Check nnopExecutor != nullptr failed
```

#### `fp16_mla_paged_decode`

- Error type: `RuntimeError`

```text
npu_fused_infer_attention_score_v2_symint:../third_party/op-plugin/op_plugin/ops/opapi/FusedInferAttentionScoreV2KernelNpuOpApi.cpp:459 NPU function error: call aclnnFusedInferAttentionScoreV4 failed, error code is 161001
[ERROR] 2026-08-05-16:07:15 (PID:38221, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 38221] 2026-08-05-16:07:15.851.885 Execution_Error(EZ1009): Failed to execute operator FusedInferAttentionScore_0. Reason: SoC version ascend310p verification failed. This SoC is not configured through the AddConfig API of the OpDef class.
TraceBack (most recent call last):
        Check nnopExecutor != nullptr failed
```

#### `fp16_mla_chunked_cache_load`

- Error type: `RuntimeError`

```text
The Inner error is reported as above. The process exits for this inner error, and the current working operator name is AtbPagedCacheLoad.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, please set the environment variable ASCEND_LAUNCH_BLOCKING=1.
Note: ASCEND_LAUNCH_BLOCKING=1 will force ops to run in synchronous mode, resulting in performance degradation. Please unset ASCEND_LAUNCH_BLOCKING in time after debugging.
[ERROR] 2026-08-05-16:07:16 (PID:38221, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 38221] 2026-08-05-16:06:50.731.665 Invalid_Argument(EH0012): aclrtAllocatorGetByStreamImpl failed, Parameter stream is invalid. Reason: The stream is not registered with any allocator.
```

## Initial Assessment

The following Leyline-specific primitives passed:

- `fp16_cache_gather_scatter`
- `fp16_leyline_rotation`

The primary blocker is the normal MLA runtime on Ascend 310P:

- `KvRmsNormRopeCache` rejects the Ascend 310P SoC.
- `FusedInferAttentionScoreV3/V4` rejects the Ascend 310P SoC.
- `AtbPagedCacheLoad` requires independent synchronous debugging to determine whether it is an independent layout/stream issue or a secondary error.
- The FP32 trigonometric tolerance requires calibration for Ascend 310P.

These results do not indicate failure of the Leyline cKV-copy/Kpe-rotation mechanism. They indicate that the underlying Ascend 310P MLA backend is not yet hardware-qualified.
