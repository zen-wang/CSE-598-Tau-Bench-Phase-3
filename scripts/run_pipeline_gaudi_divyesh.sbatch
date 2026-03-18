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
#SBATCH --job-name=tau_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- Configurable parameters ---
AGENT_SIZE="${1:-4b}"
ENV="${2:-retail}"
STRATEGY="${3:-react}"
NUM_TRIALS="${4:-5}"
START_INDEX="${5:-}"
END_INDEX="${6:-}"

# --- Derived settings ---
AGENT_MODEL="agent-${AGENT_SIZE}"

# Dynamic ports based on SLURM job ID to avoid collisions between jobs
PORT_BASE=$(( 10000 + (SLURM_JOB_ID % 5000) * 10 ))
export AGENT_PORT=$PORT_BASE
export USER_PORT=$((PORT_BASE + 1))
export PROXY_PORT=$((PORT_BASE + 2))
export PORT=$PROXY_PORT
export AGENT_SIZE   # needed by proxy.py to build ROUTES correctly

# Full paths
MAMBA_ENV=/home/dpate481/.conda/envs/tau-bench-habana
VLLM=${MAMBA_ENV}/bin/vllm
PYTHON=${MAMBA_ENV}/bin/python
export PATH="${MAMBA_ENV}/bin:$PATH"
export LD_LIBRARY_PATH="${MAMBA_ENV}/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/usr/local/lib/python3.12/site-packages:/usr/local/lib64/python3.12/site-packages:$PYTHONPATH

# Memory utilization and tensor parallelism per agent size
case "$AGENT_SIZE" in
    4b)  AGENT_MEM=0.20; USER_MEM=0.60; CONCURRENCY=3; AGENT_TP=1; USER_TP=2 ;;
    8b)  AGENT_MEM=0.25; USER_MEM=0.60; CONCURRENCY=3; AGENT_TP=1; USER_TP=2 ;;
    14b) AGENT_MEM=0.35; USER_MEM=0.55; CONCURRENCY=2; AGENT_TP=2; USER_TP=2 ;;
    32b) AGENT_MEM=0.45; USER_MEM=0.45; CONCURRENCY=1; AGENT_TP=2; USER_TP=2 ;;
    *)   echo "Unknown agent size: $AGENT_SIZE"; exit 1 ;;
esac

# --- Directory and log setup ---
mkdir -p logs results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/${STRATEGY}-agent-${AGENT_SIZE}-${ENV}-${START_INDEX}-${END_INDEX}-${TIMESTAMP}.log"

# --- Move to working directory ---
cd /scratch/dpate481/Phase3 || { echo "ERROR: Could not cd to /scratch/dpate481/Phase3"; exit 1; }

# --- Diagnostics ---
echo "=== DIAGNOSTICS ==="
echo "Date:        $(date)"
echo "Node:        $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "AGENT_PORT:  ${AGENT_PORT}"
echo "USER_PORT:   ${USER_PORT}"
echo "PROXY_PORT:  ${PROXY_PORT}"
echo "AGENT_SIZE:  ${AGENT_SIZE}"
echo "ENV:         ${ENV}"
echo "STRATEGY:    ${STRATEGY}"
echo "START_INDEX: ${START_INDEX}"
echo "END_INDEX:   ${END_INDEX}"
$PYTHON -c "import habana_frameworks.torch as ht; print('Habana torch OK')" || echo "WARNING: habana_frameworks not available"
$PYTHON -c "import torch; print('torch:', torch.__version__)"
echo "=== END DIAGNOSTICS ==="

# --- Gaudi / HPU environment variables ---
export HABANA_VISIBLE_DEVICES=0,1,2,3
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_SKIP_WARMUP=false
export VLLM_TARGET_DEVICE=hpu
export HF_HOME=/scratch/$USER/huggingface_shared
export OPENAI_API_KEY="dummy"
export OPENAI_API_BASE="http://localhost:${PROXY_PORT}/v1"
export LITELLM_REQUEST_TIMEOUT=1200

# --- Cleanup handler ---
cleanup() {
    echo "Cleaning up background processes..."
    kill $(jobs -p) 2>/dev/null
    wait 2>/dev/null
}
trap cleanup SIGTERM SIGINT

# =============================================================================
# Start user simulator (Qwen3-32B) on HPUs 2-3
# =============================================================================
echo "[$(date)] Starting user simulator on port ${USER_PORT}..."
HABANA_VISIBLE_DEVICES=2,3 $VLLM serve Qwen/Qwen3-32B \
    --served-model-name user-32b \
    --port ${USER_PORT} \
    --device hpu \
    --tensor-parallel-size ${USER_TP} \
    --gpu-memory-utilization ${USER_MEM} \
    --max-model-len 40960 \
    --max-num-seqs 8 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    >> "${LOGFILE}.user" 2>&1 &

USER_PID=$!

echo "Waiting for user simulator to be ready (up to 30 min)..."
USER_READY=0
for i in $(seq 1 360); do
    if curl -s http://localhost:${USER_PORT}/health > /dev/null 2>&1; then
        echo "[$(date)] User simulator ready after $((i*5))s"
        USER_READY=1
        break
    fi
    if ! kill -0 $USER_PID 2>/dev/null; then
        echo "ERROR: User simulator process died."
        cat "${LOGFILE}.user"
        exit 1
    fi
    echo "  attempt $i/360..."
    sleep 5
done

if [ $USER_READY -eq 0 ]; then
    echo "ERROR: User simulator failed to start within 30 minutes."
    cat "${LOGFILE}.user"
    exit 1
fi

# =============================================================================
# Start agent model on HPUs 0-1
# =============================================================================
echo "[$(date)] Starting agent (Qwen3-${AGENT_SIZE^^}) on port ${AGENT_PORT}..."
HABANA_VISIBLE_DEVICES=0,1 $VLLM serve Qwen/Qwen3-${AGENT_SIZE^^} \
    --served-model-name ${AGENT_MODEL} \
    --port ${AGENT_PORT} \
    --device hpu \
    --tensor-parallel-size ${AGENT_TP} \
    --gpu-memory-utilization ${AGENT_MEM} \
    --max-model-len 40960 \
    --max-num-seqs 16 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    >> "${LOGFILE}.agent" 2>&1 &

AGENT_PID=$!

echo "Waiting for agent to be ready (up to 30 min)..."
AGENT_READY=0
for i in $(seq 1 360); do
    if curl -s http://localhost:${AGENT_PORT}/health > /dev/null 2>&1; then
        echo "[$(date)] Agent ready after $((i*5))s"
        AGENT_READY=1
        break
    fi
    if ! kill -0 $AGENT_PID 2>/dev/null; then
        echo "ERROR: Agent process died."
        cat "${LOGFILE}.agent"
        exit 1
    fi
    echo "  attempt $i/360..."
    sleep 5
done

if [ $AGENT_READY -eq 0 ]; then
    echo "ERROR: Agent failed to start within 30 minutes."
    cat "${LOGFILE}.agent"
    exit 1
fi

# =============================================================================
# Start LiteLLM proxy
# =============================================================================
echo "[$(date)] Starting proxy on port ${PROXY_PORT}..."
$PYTHON proxy.py >> "${LOGFILE}.proxy" 2>&1 &
PROXY_PID=$!
sleep 5

if ! kill -0 $PROXY_PID 2>/dev/null; then
    echo "ERROR: Proxy failed to start."
    cat "${LOGFILE}.proxy"
    exit 1
fi
echo "[$(date)] Proxy is running."

# =============================================================================
# Run benchmark
# =============================================================================
INDEX_FLAGS=""
if [ -n "${START_INDEX}" ]; then
    INDEX_FLAGS="--start-index ${START_INDEX}"
fi
if [ -n "${END_INDEX}" ]; then
    INDEX_FLAGS="${INDEX_FLAGS} --end-index ${END_INDEX}"
fi

echo "[$(date)] Starting tau-bench evaluation..."
echo "  ENV:         ${ENV}"
echo "  STRATEGY:    ${STRATEGY}"
echo "  AGENT:       ${AGENT_MODEL} (size=${AGENT_SIZE})"
echo "  USER:        user-32b"
echo "  TRIALS:      ${NUM_TRIALS}"
echo "  CONCURRENCY: ${CONCURRENCY}"
echo "  INDEX_FLAGS: ${INDEX_FLAGS}"
echo "  LOG:         ${LOGFILE}"

$PYTHON src/run_eval.py \
    --env ${ENV} \
    --agent-strategy ${STRATEGY} \
    --model ${AGENT_MODEL} \
    --model-provider openai \
    --user-model user-32b \
    --user-model-provider openai \
    --num-trials ${NUM_TRIALS} \
    --max-concurrency ${CONCURRENCY} \
    ${INDEX_FLAGS} \
    2>&1 | tee -a "${LOGFILE}"

BENCH_EXIT=${PIPESTATUS[0]}
echo "[$(date)] Benchmark finished with exit code ${BENCH_EXIT}"
cleanup
exit $BENCH_EXIT
