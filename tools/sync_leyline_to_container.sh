#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 CONTAINER [TARGET_REPO]" >&2
  exit 2
fi

container_name=$1
target_repo=${2:-/vllm-workspace/vllm-ascend}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_repo=$(cd -- "${script_dir}/.." && pwd)

if [[ -z ${container_name} || ${target_repo} != /* ]]; then
  echo "container must be non-empty and target repo must be absolute" >&2
  exit 2
fi

docker inspect "${container_name}" >/dev/null
docker exec "${container_name}" test -d "${target_repo}/vllm_ascend/distributed/kv_transfer"
docker exec "${container_name}" mkdir -p \
  "${target_repo}/tests/ut/distributed/kv_transfer" \
  "${target_repo}/benchmarks" \
  "${target_repo}/docs/source/user_guide/feature_guide"

docker cp "${source_repo}/vllm_ascend/distributed/kv_transfer/leyline" \
  "${container_name}:${target_repo}/vllm_ascend/distributed/kv_transfer/"
docker cp "${source_repo}/vllm_ascend/distributed/kv_transfer/__init__.py" \
  "${container_name}:${target_repo}/vllm_ascend/distributed/kv_transfer/__init__.py"
docker cp "${source_repo}/vllm_ascend/ops/leyline_mla.py" \
  "${container_name}:${target_repo}/vllm_ascend/ops/leyline_mla.py"
docker cp "${source_repo}/tests/ut/distributed/kv_transfer/leyline" \
  "${container_name}:${target_repo}/tests/ut/distributed/kv_transfer/"
docker cp "${source_repo}/benchmarks/leyline" \
  "${container_name}:${target_repo}/benchmarks/"
docker cp "${source_repo}/docs/source/user_guide/feature_guide/leyline_mla.md" \
  "${container_name}:${target_repo}/docs/source/user_guide/feature_guide/leyline_mla.md"

docker exec "${container_name}" git -C "${target_repo}" status --short
docker exec "${container_name}" git -C "${target_repo}" rev-parse HEAD
docker exec "${container_name}" git -C /vllm-workspace/vllm rev-parse HEAD
docker exec "${container_name}" python3 -c \
  'import importlib.metadata as m; names=("vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "triton-ascend"); print("\n".join(f"{name}={m.version(name)}" for name in names))'
docker exec "${container_name}" npu-smi info
