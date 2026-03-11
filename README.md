# Phase 3 — Augmented Pipeline for tau-bench

A 5-module pipeline that wraps the [tau-bench](https://github.com/sierra-research/tau-bench) benchmark to improve language agent performance. The pipeline does **not** modify any tau-bench code — it wraps the conversation loop via a `PipelineAgent` class.

## Prerequisites

- NVIDIA GPU (tested on A100 80GB)
- Conda
- CUDA toolkit

## Setup

### 1. Clone with submodule

```bash
git clone --recurse-submodules <repo-url>
cd Phase3
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Create conda environment

```bash
conda create -n tau-bench python=3.10 -y
conda activate tau-bench
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set HuggingFace cache (optional, recommended for shared clusters)

```bash
export HF_HOME=/scratch/$USER/huggingface_shared
```

## Running the Pipeline

We use **tmux** to run 4 processes in one terminal: two vLLM servers, the routing proxy, and the benchmark itself.

### Step 1: Start tmux with 4 panes

```bash
# Create a new tmux session
tmux new -s pipeline

# Split into 4 panes (run these inside tmux):
# Ctrl+b %     → split vertical
# Ctrl+b "     → split horizontal
# Ctrl+b o     → switch between panes
```

Or create all 4 panes in one command:

```bash
tmux new-session -s pipeline \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; \
  split-window -v \;
```

### Step 2: In EVERY pane, load the environment

```bash
module load cuda-13.0.1-gcc-12.1.0
conda activate tau-bench
export HF_HOME=/scratch/$USER/huggingface_shared
```

### Step 3: Start services (one per pane)

**Pane 1 — User Simulator (32B)**

```bash
# <your vllm serve command for user-32b here>
```

**Pane 2 — Agent Model**

```bash
# <your vllm serve command for agent here>
```

Wait for both servers to print `Uvicorn running on ...` before proceeding.

**Pane 3 — Routing Proxy**

```bash
python proxy.py
```

The proxy listens on port 9000 and routes requests by model name to the correct vLLM server. You must update the `ROUTES` dict in `proxy.py` if you change model names.

**Pane 4 — Run Benchmark**

```bash
export OPENAI_API_KEY="dummy"
export OPENAI_API_BASE="http://localhost:9000/v1"

# <your python src/run_eval.py command here>
```

### Step 4: Verify connectivity (optional, in pane 4)

```bash
curl -s http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<agent-model-name>","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

## Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | — | `retail` or `airline` |
| `--agent-strategy` | — | `react`, `act`, or `tool-calling` |
| `--model` | — | Agent model name (must match vLLM `--served-model-name`) |
| `--user-model` | — | User simulator model name |
| `--model-provider` | — | `openai` |
| `--user-model-provider` | — | `openai` |
| `--num-trials` | 1 | Number of trials per task |
| `--max-concurrency` | 1 | Parallel task execution |
| `--task-ids` | all | Specific task IDs to run (e.g., `0 1 2`) |
| `--baseline` | off | Disable all pipeline modules (pure baseline) |
| `--log-dir` | `results` | Output directory |

## GPU Memory Guide (A100 80GB)

| Agent Model | Agent `--gpu-memory-utilization` | 32B User Sim | Free |
|-------------|--------------------------------|-------------|------|
| 4B-AWQ | 0.15 | 0.55 | ~30% |
| 8B-AWQ | 0.20 | 0.55 | ~25% |
| 14B-AWQ | 0.30 | 0.55 | ~15% |
| 32B-AWQ | 0.45 | 0.45 | ~10% |

Rule: sum of both `--gpu-memory-utilization` values should stay below 0.80.

## Project Structure

```
Phase3/
├── README.md
├── requirements.txt
├── proxy.py                  # Model routing proxy (port 9000)
├── tau-bench/                # Git submodule (sierra-research/tau-bench)
├── src/
│   ├── run_eval.py           # Evaluation runner
│   ├── compare_results.py    # Result comparison tool
│   └── pipeline/
│       ├── pipeline_agent.py     # Core wrapper — PipelineAgent(Agent)
│       ├── task_planner.py       # Module 1: LLM step decomposition
│       ├── context_injector.py   # Module 2: Policy injection
│       ├── state_tracker.py      # Module 3: Conversation state tracking
│       ├── action_gate.py        # Module 4: Pre-action checks + retry
│       └── completion_checker.py # Module 5: Post-task audit (logging only)
├── src/policies/
│   ├── retail_policies.py
│   └── airline_policies.py
└── results/                  # Output JSON files
```

## Comparing Results

```bash
# Compare two result files
python src/compare_results.py pair results/baseline.json results/pipeline.json

# Summarize all results in a directory
python src/compare_results.py dir results/ --csv results/summary.csv
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `The model X does not exist` (404) | Use the proxy (port 9000), not vLLM directly. Check `ROUTES` in `proxy.py` matches your model names. |
| `ContextWindowExceededError` | Use `--max-model-len 40960` for vLLM. Do not exceed 40960 for Qwen3. |
| `Engine core initialization failed` | GPU already occupied. Run `nvidia-smi` and `pkill -f vllm` first. |
| Timeout errors at concurrency > 1 | Make sure proxy.py has `threaded=True` (already set). |
