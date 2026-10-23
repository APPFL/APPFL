#!/bin/bash
#SBATCH --job-name=adko_h_vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# Serves one open model for every ADKO-H agent in every run. The agents are separate
# contexts against one server, not separate models, so one GPU plans a federation -- which
# is the only version of this that still makes sense at a hundred agents.
#
# The node's address goes into $ADKO_LLM_ENDPOINT_FILE; the run jobs read it and wait for it.
# Add your partition and account (-p, -A) when you submit.
#
#   export ADKO_H=/path/to/APPFL ADKO_DATA=/path/to/adko_h_data
#   sbatch -p <gpu-partition> -A <account> examples/decentralized/adko_h/serve.sbatch
#
# One-time, on a login node (compute nodes may have no internet):
#   conda create -y -n adko_vllm python=3.12 && conda run -n adko_vllm pip install "vllm>=0.31"
#   conda run -n adko_vllm hf download Qwen/Qwen3.8-27B-FP8
set -eu

MODEL=${ADKO_LLM_MODEL:-Qwen/Qwen3.8-27B-FP8}
VLLM_ENV=${ADKO_VLLM_ENV:-adko_vllm}
PORT=${ADKO_LLM_PORT:-8000}
PARSER=${ADKO_LLM_PARSER-qwen3}
MAXLEN=${ADKO_LLM_MAXLEN:-32768}
UTIL=${ADKO_LLM_UTIL:-0.90}
DATA=${ADKO_DATA:?set ADKO_DATA to a writable directory}
ENDPOINT_FILE=${ADKO_LLM_ENDPOINT_FILE:-$DATA/llm_endpoint.txt}

HOST=$(hostname -s)
mkdir -p "$(dirname "$ENDPOINT_FILE")"
rm -f "$ENDPOINT_FILE"

ARGS=(--host 0.0.0.0 --port "$PORT" --tensor-parallel-size "${ADKO_LLM_TP:-1}"
      --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN"
      --served-model-name "$MODEL")
[ -n "$PARSER" ] && ARGS+=(--reasoning-parser "$PARSER")

echo "serving $MODEL on $HOST:$PORT"
conda run -n "$VLLM_ENV" --no-capture-output vllm serve "$MODEL" "${ARGS[@]}" &
SERVER=$!

# Publish the address only once the server answers, so a run job never races it.
for _ in $(seq 1 180); do
  if curl -fsS "http://$HOST:$PORT/v1/models" > /dev/null 2>&1; then
    echo "http://$HOST:$PORT/v1" > "$ENDPOINT_FILE"
    echo "endpoint published: $(cat "$ENDPOINT_FILE")"
    break
  fi
  kill -0 "$SERVER" 2>/dev/null || { echo "server died before answering"; exit 1; }
  sleep 10
done
[ -s "$ENDPOINT_FILE" ] || { echo "server did not answer in 30 minutes"; kill "$SERVER"; exit 1; }

trap 'rm -f "$ENDPOINT_FILE"; kill "$SERVER" 2>/dev/null || true' EXIT
wait "$SERVER"
