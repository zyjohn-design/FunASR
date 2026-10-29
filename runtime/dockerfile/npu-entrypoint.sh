#!/usr/bin/env bash
# Start Fun-ASR-Nano services on Ascend NPU.
#   FUNASR_SERVICE=all       (default) OpenAI/HTTP server + realtime WebSocket
#   FUNASR_SERVICE=openai    serve_vllm.py only (port FUNASR_HTTP_PORT)
#   FUNASR_SERVICE=realtime  realtime_ws only (port FUNASR_WS_PORT)
# Any other first argument is executed as a command (e.g. bash, npu-smi info).
# Leading "-" arguments are appended to the single selected server (openai or
# realtime); in "all" mode use FUNASR_OPENAI_ARGS / FUNASR_REALTIME_ARGS instead.
set -euo pipefail

if [[ $# -gt 0 && "$1" != -* ]]; then
  exec "$@"
fi

service="${FUNASR_SERVICE:-all}"
model="${FUNASR_MODEL:-FunAudioLLM/Fun-ASR-Nano-2512}"
hub="${FUNASR_HUB:-ms}"
device="${FUNASR_DEVICE:-npu:0}"
dtype="${FUNASR_DTYPE:-bf16}"
http_port="${FUNASR_HTTP_PORT:-8899}"
ws_port="${FUNASR_WS_PORT:-10095}"

# Both engines share one NPU in "all" mode, so split HBM between them.
if [[ "${service}" == "all" ]]; then
  default_mem=0.35
else
  default_mem=0.8
fi
http_mem="${FUNASR_HTTP_MEM_UTIL:-${default_mem}}"
ws_mem="${FUNASR_WS_MEM_UTIL:-${default_mem}}"

cd /workspace/FunASR

openai_cmd=(
  python3 examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py
  --host 0.0.0.0 --port "${http_port}"
  --model "${model}" --hub "${hub}" --device "${device}" --dtype "${dtype}"
  --gpu-memory-utilization "${http_mem}"
  --spk-model "${FUNASR_SPK_MODEL:-}"
)
realtime_cmd=(
  python3 -m funasr.bin.realtime_ws
  --port "${ws_port}"
  --model "${model}" --hub "${hub}" --device "${device}" --dtype "${dtype}"
  --gpu-memory-utilization "${ws_mem}"
  --hotword-file "${FUNASR_HOTWORD_FILE:-}"
)
if [[ -n "${FUNASR_LANGUAGE:-}" ]]; then
  realtime_cmd+=(--language "${FUNASR_LANGUAGE}")
fi

# Extra flags, whitespace-split (e.g. FUNASR_REALTIME_ARGS="--enable-spk").
read -r -a openai_extra <<< "${FUNASR_OPENAI_ARGS:-}"
read -r -a realtime_extra <<< "${FUNASR_REALTIME_ARGS:-}"
openai_cmd+=("${openai_extra[@]}")
realtime_cmd+=("${realtime_extra[@]}")

case "${service}" in
  openai)
    exec "${openai_cmd[@]}" "$@"
    ;;
  realtime)
    exec "${realtime_cmd[@]}" "$@"
    ;;
  all)
    if [[ $# -gt 0 ]]; then
      echo "FUNASR_SERVICE=all: pass flags via FUNASR_OPENAI_ARGS / FUNASR_REALTIME_ARGS" >&2
      exit 2
    fi
    "${openai_cmd[@]}" &
    openai_pid=$!
    "${realtime_cmd[@]}" &
    realtime_pid=$!
    trap 'kill -TERM "${openai_pid}" "${realtime_pid}" 2>/dev/null || true' TERM INT
    # Exit (and let the orchestrator restart us) as soon as either server dies.
    set +e
    wait -n "${openai_pid}" "${realtime_pid}"
    status=$?
    kill -TERM "${openai_pid}" "${realtime_pid}" 2>/dev/null
    wait
    exit "${status}"
    ;;
  *)
    echo "Unknown FUNASR_SERVICE=${service} (expected all|openai|realtime)" >&2
    exit 2
    ;;
esac
