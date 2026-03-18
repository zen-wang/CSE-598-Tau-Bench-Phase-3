#!/bin/bash
#SBATCH --partition=gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --qos=class_gaudi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:hl225:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=tau_retail20
#SBATCH --output=/scratch/%u/taubench_logs/slurm-%j.out
#SBATCH --error=/scratch/%u/taubench_logs/slurm-%j.err

set -euo pipefail

# ---- args (defaults) ----
DOMAIN=${1:-airline}
STRATEGY=${2:-act}     # act | react | tool-calling (whatever tau-bench expects)
START_IDX=${3:-0}
END_IDX=${4:-50}

# ---- knobs ----
MAX_LEN=16384
DTYPE="bfloat16"
GPU_UTIL_USER=0.70
GPU_UTIL_AGENT=0.55
MAX_CONCURRENCY=1
NUM_TRIALS=5

# ---- models ----
USER_MODEL="Qwen/Qwen3-32B"
AGENT_MODEL="Qwen/Qwen3-4B"

# These are what tau-bench will call in --model/--user-model
USER_SERVED="user32b"
AGENT_SERVED="agent4b"

# ---- container ----
CONTAINER="/data/sse/gaudi/containers/vllm-gaudi.sif"

# ---- env ----
module load mamba/latest
source activate vllm312   # <-- your env (312)
# --- FORCE the correct python from your env (don’t trust PATH on SOL) ---
PY="/home/dvallur2/.conda/envs/vllm312/bin/python"
export PATH="/home/dvallur2/.conda/envs/vllm312/bin:$PATH"
hash -r

echo "[DEBUG] which python = $(which python)"
echo "[DEBUG] PY = $PY"
$PY -c "import sys; print('[DEBUG] sys.executable=', sys.executable)"
$PY -c "import litellm; print('[DEBUG] litellm ok')"
unset http_proxy https_proxy
export NO_PROXY="localhost,127.0.0.1"

# ---- scratch/cache ----
TB_SCRATCH="/scratch/$USER/taubench_logs/${SLURM_JOB_ID}"
HABANA_LOGS_HOST="$TB_SCRATCH/habana_logs"
mkdir -p "$TB_SCRATCH"/{logs,tmp,habana_logs,results}
chmod 700 "$HABANA_LOGS_HOST"

export HF_HOME="/scratch/$USER/.cache/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export VLLM_CACHE_ROOT="/scratch/$USER/.cache/vllm"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT"

# pass cache + habana logs into container
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export APPTAINERENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export APPTAINERENV_HABANA_LOGS="/var/log/habana_logs"
export APPTAINERENV_HABANA_LOGS_DIR="/var/log/habana_logs"
export APPTAINERENV_PT_HPU_LAZY_MODE=1

# ---- ports ----
USER_PORT=$((9000 + SLURM_JOB_ID % 1000))
AGENT_PORT=$((9500 + SLURM_JOB_ID % 1000))

USER_LOG="$TB_SCRATCH/logs/${SLURM_JOB_ID}_vllm_user.log"
AGENT_LOG="$TB_SCRATCH/logs/${SLURM_JOB_ID}_vllm_agent.log"
RUN_LOG="$TB_SCRATCH/logs/${SLURM_JOB_ID}_taubench.log"

USER_PID=""
AGENT_PID=""

cleanup() {
  echo "[CLEANUP] stopping servers..."
  if [ -n "${USER_PID:-}" ]; then kill "$USER_PID" 2>/dev/null || true; sleep 2; kill -9 "$USER_PID" 2>/dev/null || true; fi
  if [ -n "${AGENT_PID:-}" ]; then kill "$AGENT_PID" 2>/dev/null || true; sleep 2; kill -9 "$AGENT_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

start_user_server() {
  echo "[INFO] Starting USER server on HPUs 0,1 port $USER_PORT served-name=$USER_SERVED"
  HABANA_VISIBLE_DEVICES="0,1" \
  APPTAINERENV_HABANA_VISIBLE_DEVICES="0,1" \
  apptainer exec \
    --bind /scratch:/scratch \
    --bind /data:/data \
    --bind "$HABANA_LOGS_HOST:/var/log/habana_logs" \
    "$CONTAINER" \
    vllm serve "$USER_MODEL" \
      --served-model-name "$USER_SERVED" \
      --device hpu \
      --host 127.0.0.1 \
      --port "$USER_PORT" \
      --dtype "$DTYPE" \
      --block-size 128 \
      --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization "$GPU_UTIL_USER" \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      > "$USER_LOG" 2>&1 &
  USER_PID=$!
}

start_agent_server() {
  echo "[INFO] Starting AGENT server on HPUs 2,3 port $AGENT_PORT served-name=$AGENT_SERVED"
  HABANA_VISIBLE_DEVICES="2,3" \
  APPTAINERENV_HABANA_VISIBLE_DEVICES="2,3" \
  apptainer exec \
    --bind /scratch:/scratch \
    --bind /data:/data \
    --bind "$HABANA_LOGS_HOST:/var/log/habana_logs" \
    "$CONTAINER" \
    vllm serve "$AGENT_MODEL" \
      --served-model-name "$AGENT_SERVED" \
      --device hpu \
      --host 127.0.0.1 \
      --port "$AGENT_PORT" \
      --dtype "$DTYPE" \
      --block-size 128 \
      --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization "$GPU_UTIL_AGENT" \
      --tensor-parallel-size 2 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes \
      --trust-remote-code \
      > "$AGENT_LOG" 2>&1 &
  AGENT_PID=$!
}

wait_ready() {
  local port="$1"
  local name="$2"
  for i in {1..240}; do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[INFO] $name ready on port $port"
      return 0
    fi
    sleep 5
  done
  echo "[ERROR] $name not ready. Tail log:"
  if [ "$name" = "user" ]; then tail -n 120 "$USER_LOG" || true; else tail -n 120 "$AGENT_LOG" || true; fi
  return 1
}

start_user_server
start_agent_server

wait_ready "$USER_PORT" "user" || exit 1
wait_ready "$AGENT_PORT" "agent" || exit 1

# ---- point tau-bench to the two servers ----
export USER_MODEL_API_BASE="http://127.0.0.1:${USER_PORT}/v1"
export USER_MODEL_API_KEY="EMPTY"
export OPENAI_API_BASE="http://127.0.0.1:${AGENT_PORT}/v1"
export OPENAI_API_KEY="EMPTY"
export LITELLM_REQUEST_TIMEOUT=900
export OPENAI_TIMEOUT=900

cd "$HOME/tau-bench"

OUT_DIR="$TB_SCRATCH/results/${DOMAIN}_${STRATEGY}_agent_${AGENT_SERVED}_user_${USER_SERVED}_${START_IDX}_${END_IDX}"
mkdir -p "$OUT_DIR"

echo "[INFO] Running tau-bench: domain=$DOMAIN strategy=$STRATEGY tasks=$START_IDX..$END_IDX num_trials=$NUM_TRIALS max_concurrency=$MAX_CONCURRENCY max_len=$MAX_LEN"
$PY -u run.py \
  --agent-strategy "$STRATEGY" \
  --env "$DOMAIN" \
  --model "$AGENT_SERVED" \
  --model-provider openai \
  --user-model "$USER_SERVED" \
  --user-model-provider openai \
  --user-strategy llm \
  --max-concurrency "$MAX_CONCURRENCY" \
  --num-trials "$NUM_TRIALS" \
  --temperature 0.7 \
  --start-index "$START_IDX" \
  --end-index "$END_IDX" \
  --log-dir "$OUT_DIR" \
  3>&1 | tee "$RUN_LOG"
