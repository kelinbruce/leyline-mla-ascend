export REPO=/opt/z00927893/project/1_leyline/7_leyline/leyline-mla-ascend
export MODEL=/opt/foundation_model/DeepSeek-V2-Lite

export LEYLINE_RUN_ID=schema_v3_20260810_064824
export LEYLINE_RUN_DIR="$REPO/results/leyline/$LEYLINE_RUN_ID"

export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_BATCH_INVARIANT=0
export GLOO_SOCKET_IFNAME=lo
export VLLM_ENGINE_READY_TIMEOUT_S=3600

export VLLM_ASCEND_LEYLINE_CAPTURE_DIR="$LEYLINE_RUN_DIR/cache-captures"
export VLLM_ASCEND_LEYLINE_CAPTURE_MAX_ROWS=64
export VLLM_ASCEND_LEYLINE_CAPTURE_REQUIRED_DELTAS=0,1,127,128,129,1024

export VLLM_ASCEND_LEYLINE_RAW_LOGITS_DIR="$LEYLINE_RUN_DIR/raw-logits"
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_ARMS=full,honest_edited,leyline

export VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_STEP=32
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_FILES=4096
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_BYTES=8589934592

export VLLM_ASCEND_LEYLINE_RAW_LOGITS_RUN_ID="$(
  jq -r '.target_run_id' \
    "$LEYLINE_RUN_DIR/divergence-plan.json"
)"

export VLLM_ASCEND_LEYLINE_RAW_LOGITS_STEPS="$(
  jq -r '.capture_steps | join(",")' \
    "$LEYLINE_RUN_DIR/divergence-plan.json"
)"

export VLLM_ASCEND_LEYLINE_RAW_LOGITS_CASES="$(
  jq -r '.case_ids | join(",")' \
    "$LEYLINE_RUN_DIR/divergence-plan.json"
)"
