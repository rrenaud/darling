#!/usr/bin/env bash

set -euo pipefail

# lightweight guard so we fail fast if vllm is missing
if ! command -v vllm >/dev/null 2>&1; then
  echo "Error: vllm CLI not found in PATH. Install vLLM before running this script." >&2
  exit 1
fi

MODEL_ID_INPUT="${1:-${MODEL_ID:-Qwen/Qwen3-4B}}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TENSOR_PARALLEL:-1}"
GPU_MEM="${VLLM_GPU_MEM_UTIL:-0.9}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
SERVE_NAME="${VLLM_SERVED_MODEL_NAME:-$(basename "${MODEL_ID_INPUT}")}"
MODEL_ID="${MODEL_ID_INPUT}"

if [[ -n "${VLLM_MAX_SEQ_LEN:-}" ]]; then
  MAX_LEN="${VLLM_MAX_SEQ_LEN}"
fi

DOWNLOAD_ROOT="${VLLM_DOWNLOAD_ROOT:-${HOME}/.cache/hf_models}"
REVISION="${VLLM_REVISION:-}"

# Derive the actual path that vLLM should load. If we detect a remote HF model id,
# snapshot it locally to avoid partial directories triggering config errors.
USE_SNAPSHOT=false
if [[ "${VLLM_SKIP_DOWNLOAD:-0}" != "1" ]]; then
  if [[ ! -d "${MODEL_ID_INPUT}" || ! -f "${MODEL_ID_INPUT}/config.json" ]]; then
    if [[ "${MODEL_ID_INPUT}" == */* ]]; then
      SAFE_NAME="${MODEL_ID_INPUT//\//__}"
      SNAPSHOT_DIR="${DOWNLOAD_ROOT}/${SAFE_NAME}"
      MODEL_ID="${SNAPSHOT_DIR}"
      if [[ ! -f "${SNAPSHOT_DIR}/config.json" ]]; then
        if ! command -v huggingface-cli >/dev/null 2>&1; then
          echo "Error: huggingface-cli not found. Install it with 'pip install huggingface_hub'." >&2
          exit 1
        fi
        mkdir -p "${DOWNLOAD_ROOT}"
        echo "Downloading ${MODEL_ID_INPUT} from Hugging Face into ${SNAPSHOT_DIR}"
        HF_ARGS=(download "${MODEL_ID_INPUT}" --local-dir "${SNAPSHOT_DIR}" --local-dir-use-symlinks False --repo-type model)
        if [[ -n "${REVISION}" ]]; then
          HF_ARGS+=(--revision "${REVISION}")
        fi
        set -x
        huggingface-cli "${HF_ARGS[@]}"
        set +x
      fi
      USE_SNAPSHOT=true
    fi
  else
    MODEL_ID="${MODEL_ID_INPUT}"
  fi
else
  MODEL_ID="${MODEL_ID_INPUT}"
fi

if [[ ! -f "${MODEL_ID}/config.json" ]]; then
  echo "Error: Unable to locate config.json under ${MODEL_ID}. Provide a valid local path or allow downloading from Hugging Face." >&2
  exit 1
fi

if [[ "${USE_SNAPSHOT}" == "true" ]]; then
  echo "Serving snapshot from ${MODEL_ID}"
fi

ARGS=(
  serve "${MODEL_ID}"
  --served-model-name "${SERVE_NAME}"
  --tensor-parallel "${TP}"
  --host "${HOST}"
  --port "${PORT}"
  --gpu-memory-utilization "${GPU_MEM}"
  --max-model-len "${MAX_LEN}"
  --trust-remote-code
)

if [[ -n "${VLLM_DTYPE:-}" ]]; then
  ARGS+=(--dtype "${VLLM_DTYPE}")
fi

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  # Support passing additional command-line switches through env var.
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${VLLM_EXTRA_ARGS})
  ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "Launching vLLM server for ${MODEL_ID} on ${HOST}:${PORT} (tensor parallel ${TP})"
set -x
vllm "${ARGS[@]}"
