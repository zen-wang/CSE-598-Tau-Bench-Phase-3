# Runtime Performance Review: Job 48860544 (14B Airline Baselines)

## Job Overview

| Metric | Value |
|---|---|
| **Wall clock** | **4h 41min** (22:25 → 03:06 MST, Mar 12-13 2026) |
| **GPU** | A100 80GB (0.55 user-sim + 0.30 agent) |
| **Concurrency** | 2 workers |
| **Trials** | 1 of 5 requested (all 3 strategies crashed before trial 2) |

## Strategy Timing Breakdown

| Strategy | Duration | Tasks | Min/task | Pass Rate | Steps | Steps/task |
|---|---|---|---|---|---|---|
| **react** | 1h 14m | 37 | 2.0 | 21.6% (8/37) | 217 | 5.9 |
| **act** | 1h 23m | 22 | 3.8 | 13.6% (3/22) | 220 | 10.0 |
| **tool-calling** | 2h 02m | 43 | 2.8 | 18.6% (8/43) | 404 | 9.6 |
| **Startup** | 1m 44s | — | — | — | — | — |

## Root Cause: Only 1 Trial Completed

All 3 strategies exited with code 1 after completing only trial 0. The crash is a **litellm/httpx client lifecycle bug** — `"Cannot send a request, as the client has been closed"` during `LLMUserSimulationEnv.__init__()`. The httpx client gets garbage-collected between task runs. This is non-deterministic: react crashed after task 37, act after 22, tool-calling after 43.

**Impact**: Only 1 of 5 requested trials ran. Getting 5 trials would require ~23.5 hours at current throughput — far exceeding any reasonable SLURM allocation.

## Where the Time Went

### 1. Stall/Loop Tasks (Biggest Waste)

| Strategy | Stall Tasks | Steps Burned | % of Total Steps |
|---|---|---|---|
| react | 3 think-loops (T20, T24, T11) | 66 | 30.4% |
| act | 6 respond-loops (T7, T17, T21, T8, T11, T12) | 101 | 46% |
| tool-calling | 4 stalls (T0, T4, T9, T38) | 120 | 30% |

**287 steps wasted on loops/stalls across all strategies** — zero reward from any of them.

### 2. Act's Respond Problem

Act had a **56% respond ratio** (123/220 steps are "respond" not tool calls) vs react's ~27%. Without the Thought/Action/Observation structure, the 14B model falls into conversation loops. Act averaged 3.8 min/task (1.9x slower than react) due to this.

### 3. Tool-Calling's Speed Advantage

Tool-calling completed the most tasks (43) because:
- **Consecutive tool-call streaks** skip user-sim inference — only 63% of steps needed user-sim calls (saving ~149 LLM calls)
- **Fast exits** (10 tasks in 1-3 steps) free concurrency slots
- Total LLM calls: 659 vs ~808 if every step needed both models (19% reduction)

## Token Throughput (Healthy — Not the Bottleneck)

| Model | Prompt tok/s | Gen tok/s | Prefix Cache Hit |
|---|---|---|---|
| Agent (14B) | 390-1023 | 16-157 | **95.5%** |
| User sim (32B) | 19-817 | 22-127 | 65-86% |

No GPU memory pressure, no OOM, no request queuing. The A100 handles both models comfortably.

## Key Bottlenecks & Improvement Opportunities

| Bottleneck | Impact | Fix |
|---|---|---|
| **litellm client crash** | Only 1/5 trials complete | Fix httpx client lifecycle (recreate client per task, or catch & retry) |
| **Stall/loop tasks** | 287 wasted steps (34% of all compute) | Pipeline's action gate (inaction check + auth-stall escalation) directly addresses this |
| **Act respond loops** | 1.9x slower per task | Pipeline's action gate blocks respond loops after 3+ idle steps |
| **Immediate transfers** | 8+5+5=18 tasks across strategies | Pipeline's hallucination check + "ask don't transfer" policy |
| **Single trial** | No statistical confidence | Fix crash → 5 trials, or increase SLURM time limit |
| **Low concurrency** | 2 workers underutilizes A100 | Could try concurrency=3-4 (vLLM showed no queuing) |

## What the Pipeline Can Address

From the failure analysis:

| Failure Type | Count (all strategies) | Pipeline Module |
|---|---|---|
| Stall/loops | ~13 tasks, 287 steps | Action Gate (inaction + auth-stall) |
| Immediate/premature transfer | ~18 tasks | Action Gate (hallucination check) + Context Injector |
| Wrong params | ~31 tasks | Task Planner (decomposition) + Context Injector (policy) |
| Extra write actions | ~10 tasks | Action Gate (confirmation gate) |

**The stall/loop problem is the highest-leverage runtime fix** — eliminating those 287 wasted steps would cut ~30% of compute time and allow more tasks to complete within the time budget.

## Recommendations for Next Runs

1. **Fix the litellm crash** — investigate httpx client reuse across tasks; may need to set `LITELLM_SET_VERBOSE=True` or recreate the client per strategy
2. **Increase SLURM time** to 8h if running all 3 strategies with 5 trials
3. **Try concurrency=3** — vLLM showed zero queuing at concurrency=2
4. **Run pipeline vs baseline** — the pipeline's action gate directly targets the stall/loop waste that consumed 34% of compute

## Detailed Strategy Analyses

### React (37 tasks, 8 pass)

**Passing tasks**: 12, 15, 17, 21, 29, 30, 36, 37

**Failure breakdown (29 failures):**

| Category | Count | Steps Used | % of Runtime |
|---|---|---|---|
| IMMEDIATE_TRANSFER | 8 | 8 | 3.7% |
| THINK_LOOP | 3 | 66 | 30.4% |
| WRONG_ACTIONS | 7 | 59 | 27.2% |
| EXTRA_ACTIONS | 4 | 39 | 18.0% |
| PREMATURE_TRANSFER | 6 | 20 | 9.2% |
| TIMEOUT | 1 | 0 | 0.0% |

Top runtime drivers: Task 24 (30 steps, think loop), Task 20 (23 steps, think loop), Task 9 (15 steps, wrong actions). These 3 tasks consumed 31.3% of all react LLM calls.

### Act (22 tasks, 3 pass)

**Passing tasks**: 6, 15, 18

**Failure breakdown (19 failures):**

| Category | Count | Tasks |
|---|---|---|
| PREMATURE_TRANSFER | 5 | 2, 5, 16, 20, 22 |
| WRONG_ACTION | 5 | 0, 1, 3, 9, 10, 13, 14 |
| CONVERSATION_STALL | 4 | 8, 11, 12, 21 |
| MAX_STEPS | 2 | 7, 17 |
| IMMEDIATE_FAIL | 1 | 4 |

Top runtime drivers: Task 7 (30 steps, respond loop), Task 17 (30 steps, echo loop), Task 21 (21 steps, stall). These 4 highest-step tasks consumed 46% of all act steps.

### Tool-Calling (43 tasks, 8 pass)

**Passing tasks**: 6, 11, 18, 20, 29, 34, 37, 40

**Failure breakdown (35 failures):**

| Category | Count | Tasks |
|---|---|---|
| WRONG_PARAMS | 19 | 1,2,10,12,14,17,21,22,23,24,25,26,27,30,32,35,36,39,42 |
| IMMEDIATE_TRANSFER | 5 | 5,16,19,33,43 |
| LATE_TRANSFER | 3 | 3,8,13 |
| STALL_NO_TOOLS | 2 | 4,38 |
| STALL_WITH_TOOLS | 2 | 0,9 |
| INCOMPLETE | 2 | 7,31 |
| CONNECTION_ERROR | 1 | 15 |
| NO_TOOLS | 1 | 28 |

Top runtime drivers: 4 stall tasks (T0, T4, T9, T38) consumed 120/404 steps (30%). WRONG_PARAMS dominates failures at 54% (19/35).

## Cross-Strategy Observations

- **No single task passed across all 3 strategies** — results dominated by user-sim variance
- Task 15 passed in both react and act; Task 29 passed in both react and tool-calling
- Only 2 react passes required substantive work (Task 30: multi-cancel, Task 36: lookup+transfer); the rest were trivial transfers
- Tool-calling's WRONG_PARAMS dominance (54%) confirms task planner as highest-leverage pipeline module
