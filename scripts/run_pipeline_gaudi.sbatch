#!/bin/bash
#SBATCH --partition=gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --qos=class_gaudi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:hl225:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=tau-pipeline-gaudi
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err

###############################################################################
# Gaudi pipeline runner — derived from run_benchmark.sbatch (A100) with
# infrastructure from sbatch_gaudi_2srv_retail20.sbatch.
#
# Usage:
#   sbatch scripts/run_pipeline_gaudi.sbatch                            # defaults: 4b, retail, react, 5 trials
#   sbatch scripts/run_pipeline_gaudi.sbatch 14b airline act 5          # 14b agent, airline, act, 5 trials
#   sbatch scripts/run_pipeline_gaudi.sbatch 4b retail react 5 "0"     # smoke test (1 task)
#   sbatch scripts/run_pipeline_gaudi.sbatch 4b retail react 5 "0 1 2" # tasks 0,1,2
#
# To run BASELINE (no pipeline):
#   Set BASELINE=1 before sbatch, or uncomment the BASELINE=1 line below.
#
# Check status:   squeue -u $USER
# Check output:   tail -f results/slurm-<jobid>.out
# Cancel:         scancel <jobid>
###############################################################################

set -euo pipefail

# --- Configurable parameters (override via CLI args) ---
AGENT_SIZE="${1:-4b}"          # 4b, 8b, 14b, or 32b
ENV="${2:-retail}"             # retail or airline
STRATEGY="${3:-react}"         # react, act, or tool-calling
NUM_TRIALS="${4:-5}"           # number of trials per task
TASK_IDS="${5:-}"              # optional: "0" or "0 1 2" (empty = all tasks)
BASELINE="${BASELINE:-0}"      # set BASELINE=1 to run without pipeline

# --- Gaudi-specific knobs ---
CONTAINER="/data/sse/gaudi/containers/vllm-gaudi.sif"
DTYPE="bfloat16"
MAX_LEN=40960                  # Divyesh confirmed 40960 works on Gaudi with bf16 (matches A100)
# NOTE: Gaudi does NOT reliably support AWQ quantization — use non-AWQ (bf16) models.
# NOTE: Gaudi does NOT support --enforce-eager or --enable-prefix-caching.

# Conda env — must have litellm, flask, requests, and pipeline deps installed.
# The teammate's env is "vllm312"; create your own or use a shared one.
CONDA_ENV="${CONDA_ENV:-tau-bench}"

# --- Models (non-AWQ bf16 for Gaudi HPUs — AWQ is NVIDIA-only) ---
USER_HF="Qwen/Qwen3-32B"
AGENT_MODEL="agent-${AGENT_SIZE}"

# GPU memory + concurrency per agent size.
# Each model gets 2 HPUs (2 x 96GB = 192GB).
# Concurrency from Divyesh's tested settings; max-num-seqs limits vLLM scheduler.
case "$AGENT_SIZE" in
    4b)  AGENT_HF="Qwen/Qwen3-4B";  AGENT_GPU=0.55; USER_GPU=0.70; CONCURRENCY=3 ;;
    8b)  AGENT_HF="Qwen/Qwen3-8B";  AGENT_GPU=0.55; USER_GPU=0.70; CONCURRENCY=3 ;;
    14b) AGENT_HF="Qwen/Qwen3-14B"; AGENT_GPU=0.60; USER_GPU=0.70; CONCURRENCY=2 ;;
    32b) AGENT_HF="Qwen/Qwen3-32B"; AGENT_GPU=0.70; USER_GPU=0.70; CONCURRENCY=1 ;;
    *)   echo "Unknown agent size: $AGENT_SIZE"; exit 1 ;;
esac
MAX_NUM_SEQS_USER=8
MAX_NUM_SEQS_AGENT=16

# --- Dynamic ports (avoid collision between concurrent SLURM jobs) ---
USER_PORT=$((9000 + SLURM_JOB_ID % 1000))
AGENT_PORT=$((9500 + SLURM_JOB_ID % 1000))
PROXY_PORT=$((8500 + SLURM_JOB_ID % 1000))

# --- TOKEN_BUDGET must fit within MAX_LEN ---
# MAX_LEN=16384 → TOKEN_BUDGET ~12000 (leaves ~4k for generation + tokenizer mismatch)
# MAX_LEN=40960 → TOKEN_BUDGET 30000 (A100 default)
if [ "$MAX_LEN" -le 16384 ]; then
    TOKEN_BUDGET=12000
    EMERGENCY_BUDGET=14000
else
    TOKEN_BUDGET=30000
    EMERGENCY_BUDGET=35000
fi

# --- Project directory ---
PROJECT_DIR="/home/wwang360/CSE598/Phase3"

# --- Log setup ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$PROJECT_DIR/results"

echo "============================================"
echo "GAUDI Pipeline Run"
echo "Agent: ${AGENT_HF} (gpu=${AGENT_GPU})"
echo "User sim: ${USER_HF} (gpu=${USER_GPU})"
echo "Env: ${ENV}, Strategy: ${STRATEGY}"
echo "Trials: ${NUM_TRIALS}, Concurrency: ${CONCURRENCY}, Tasks: ${TASK_IDS:-all}"
echo "MAX_LEN: ${MAX_LEN}, TOKEN_BUDGET: ${TOKEN_BUDGET}"
echo "Baseline: ${BASELINE}"
echo "Ports: user=${USER_PORT} agent=${AGENT_PORT} proxy=${PROXY_PORT}"
echo "============================================"

# --- Environment setup ---
cd "$PROJECT_DIR"
# Hardcode conda env path — source activate is unreliable on Sol SLURM nodes
# (mamba/latest hijacks python before activate takes effect)
CONDA_PREFIX="/home/wwang360/.conda/envs/${CONDA_ENV}"
PY="${CONDA_PREFIX}/bin/python"
export PATH="${CONDA_PREFIX}/bin:$PATH"
hash -r

echo "[DEBUG] which python = $(which python)"
echo "[DEBUG] PY = $PY"
$PY -c "import sys; print('[DEBUG] sys.executable=', sys.executable)"
$PY -c "import litellm; print('[DEBUG] litellm ok')"

unset http_proxy https_proxy
export NO_PROXY="localhost,127.0.0.1"
export HF_HOME="/scratch/$USER/.cache/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export VLLM_CACHE_ROOT="/scratch/$USER/.cache/vllm"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT"

export OPENAI_API_KEY="dummy"
export OPENAI_API_BASE="http://127.0.0.1:${PROXY_PORT}/v1"
export LITELLM_REQUEST_TIMEOUT=1200
export OPENAI_TIMEOUT=1200

# Pipeline TOKEN_BUDGET for this MAX_LEN
export TOKEN_BUDGET="${TOKEN_BUDGET}"
export EMERGENCY_BUDGET="${EMERGENCY_BUDGET}"
export MAX_CONTEXT_TOKENS="${MAX_LEN}"

# Proxy env vars (dynamic ports)
export VLLM_AGENT_URL="http://127.0.0.1:${AGENT_PORT}"
export VLLM_USER_URL="http://127.0.0.1:${USER_PORT}"
export PROXY_PORT="${PROXY_PORT}"

# --- Habana / apptainer env ---
TB_SCRATCH="/scratch/$USER/taubench_logs/${SLURM_JOB_ID}"
HABANA_LOGS_HOST="$TB_SCRATCH/habana_logs"
mkdir -p "$TB_SCRATCH"/{logs,habana_logs}
chmod 700 "$HABANA_LOGS_HOST"

export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export APPTAINERENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export APPTAINERENV_HABANA_LOGS="/var/log/habana_logs"
export APPTAINERENV_HABANA_LOGS_DIR="/var/log/habana_logs"
export APPTAINERENV_PT_HPU_LAZY_MODE=1

# --- Logs ---
USER_LOG="$TB_SCRATCH/logs/vllm_user.log"
AGENT_LOG="$TB_SCRATCH/logs/vllm_agent.log"

# --- Cleanup ---
USER_PID=""
AGENT_PID=""
PROXY_PID=""

cleanup() {
    echo "[CLEANUP] stopping servers..."
    [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2>/dev/null || true
    if [ -n "${USER_PID:-}" ]; then kill "$USER_PID" 2>/dev/null || true; sleep 2; kill -9 "$USER_PID" 2>/dev/null || true; fi
    if [ -n "${AGENT_PID:-}" ]; then kill "$AGENT_PID" 2>/dev/null || true; sleep 2; kill -9 "$AGENT_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

# --- Start vLLM servers via apptainer (HPUs 0,1 for user; 2,3 for agent) ---
echo "Starting user simulator (${USER_HF}) on HPUs 0,1 port ${USER_PORT}..."
HABANA_VISIBLE_DEVICES="0,1" \
APPTAINERENV_HABANA_VISIBLE_DEVICES="0,1" \
apptainer exec \
    --bind /scratch:/scratch \
    --bind /data:/data \
    --bind "$HABANA_LOGS_HOST:/var/log/habana_logs" \
    "$CONTAINER" \
    vllm serve "$USER_HF" \
        --served-model-name user-32b \
        --device hpu \
        --host 127.0.0.1 \
        --port "$USER_PORT" \
        --dtype "$DTYPE" \
        --block-size 128 \
        --max-model-len "$MAX_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS_USER" \
        --gpu-memory-utilization "$USER_GPU" \
        --tensor-parallel-size 2 \
        --trust-remote-code \
        > "$USER_LOG" 2>&1 &
USER_PID=$!

echo "Starting agent (${AGENT_HF}) on HPUs 2,3 port ${AGENT_PORT}..."
HABANA_VISIBLE_DEVICES="2,3" \
APPTAINERENV_HABANA_VISIBLE_DEVICES="2,3" \
apptainer exec \
    --bind /scratch:/scratch \
    --bind /data:/data \
    --bind "$HABANA_LOGS_HOST:/var/log/habana_logs" \
    "$CONTAINER" \
    vllm serve "$AGENT_HF" \
        --served-model-name "$AGENT_MODEL" \
        --device hpu \
        --host 127.0.0.1 \
        --port "$AGENT_PORT" \
        --dtype "$DTYPE" \
        --block-size 128 \
        --max-model-len "$MAX_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS_AGENT" \
        --gpu-memory-utilization "$AGENT_GPU" \
        --tensor-parallel-size 2 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --trust-remote-code \
        > "$AGENT_LOG" 2>&1 &
AGENT_PID=$!

# --- Wait for servers to be ready ---
wait_ready() {
    local port="$1"
    local name="$2"
    local log="$3"
    for i in $(seq 1 240); do
        if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            echo "[INFO] $name ready on port $port"
            return 0
        fi
        sleep 5
    done
    echo "[ERROR] $name not ready after 20 minutes. Tail log:"
    tail -n 120 "$log" || true
    return 1
}

wait_ready "$USER_PORT" "user-sim" "$USER_LOG" || exit 1
wait_ready "$AGENT_PORT" "agent" "$AGENT_LOG" || exit 1

# --- Start proxy ---
echo "Starting proxy on port ${PROXY_PORT}..."
$PY "$PROJECT_DIR/proxy.py" &
PROXY_PID=$!
sleep 2

# --- Verify connectivity ---
echo "Verifying connectivity..."
curl -sf "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${AGENT_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" \
    > /dev/null 2>&1 && echo "  Agent: OK" || echo "  Agent: FAILED"

curl -sf "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"user-32b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
    > /dev/null 2>&1 && echo "  User sim: OK" || echo "  User sim: FAILED"

# --- Run benchmark ---
echo ""
echo "Starting benchmark..."
echo "============================================"

TASK_FLAG=""
if [ -n "${TASK_IDS}" ]; then
    TASK_FLAG="--task-ids ${TASK_IDS}"
fi

BASELINE_FLAG=""
if [ "${BASELINE}" = "1" ]; then
    BASELINE_FLAG="--baseline"
fi

$PY "$PROJECT_DIR/src/run_eval.py" \
    --env "$ENV" \
    --agent-strategy "$STRATEGY" \
    --model "$AGENT_MODEL" \
    --model-provider openai \
    --user-model user-32b \
    --user-model-provider openai \
    --num-trials "$NUM_TRIALS" \
    --max-concurrency "$CONCURRENCY" \
    --temperature 0.7 \
    $TASK_FLAG \
    $BASELINE_FLAG

BENCH_EXIT=$?

echo ""
echo "============================================"
echo "Benchmark complete (exit code: $BENCH_EXIT)."
cleanup
exit $BENCH_EXIT
