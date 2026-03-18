# 1. Introduction

This report presents the implementation and evaluation of a multi-agent augmented pipeline designed to improve the performance of tool-calling language agents on the τ-bench benchmark. This work constitutes Phase 3 of a three-phase project for CSE 598 (Agentic AI), Spring 2026, at Arizona State University.

**Team Members:** Abhishek Mulasi | Devesh Valluru | Divyesh Patel | Wei-An Wang

In Phase 1, we established baselines by evaluating Qwen3 models at four parameter sizes (4B, 8B, 14B, 32B) across three agentic strategies (Act, ReAct, Function Calling) on both the airline and retail domains of τ-bench. Across 6,282 trajectories, the overall pass rate was 19%, confirming fundamental reliability challenges across all configurations. In Phase 2, we diagnosed the 5,090 failures using a custom 8-type error taxonomy organized into a 4-level classification hierarchy. A central finding was that error profiles shift with model size rather than uniformly improving -- smaller models predominantly select the wrong tools, while larger models increasingly hallucinate task completion without executing any actions. This means no single intervention can address all failure modes; instead, different errors require different types of mitigation applied at different stages of the agent loop.

Phase 3 implements and evaluates a 5-module augmented pipeline targeting those specific failure modes. Our framework augments the baseline agent loop with pre-action planning, context enrichment, state tracking, action verification, and post-task audit. The pipeline adds only 1 additional LLM call per task (at task start for planning) while the remaining modules operate as deterministic, code-based checks. This design addresses 6 of the 8 identified error categories while minimizing computational overhead -- a direct consequence of the error-analysis-driven design from Phase 2.


# 2. Motivation from Error Analysis

Our pipeline design is grounded in the quantitative failure analysis conducted in Phase 2. Rather than applying generic agent-improvement techniques, we used the specific error distributions observed across 6,282 trajectories to determine what modules to build, where to place them in the agent loop, and which failure modes each module should target. This section summarizes the key findings that motivated our architecture.

## 2.1 Error Taxonomy

Our Phase 2 analysis classified 5,090 failed trajectories (81% failure rate) across four Qwen3 model sizes and three baseline strategies into 8 distinct error types. The taxonomy is organized as a 4-level classification hierarchy, where each level asks an increasingly specific diagnostic question:

**Level 1 -- Did the agent attempt the task?**

- **PREMATURE_TERMINATION** (1,626 total, 31.9%) -- Agent never called the required consequential (write) tool. Includes hallucinated completions where the agent generates a convincing confirmation message without executing any action.
- **EMPTY_TRAJECTORY** (1,325 total, 26.0%) -- Server crash or timeout; no trajectory data recorded.
- **AUTH_MISSING** (782 total, 15.4%) -- Agent skipped user authentication, often cascading into fabricated data for downstream operations.

**Level 2 -- Did the agent select the right tools?**

- **WRONG_TOOL** (1,327 total, 26.1%) -- Agent called a fundamentally different consequential tool than required by the task.

**Level 3 -- Did the agent use the right arguments?**

- **WRONG_ARGUMENT** (997 total, 19.58%) -- Correct tool called with incorrect arguments. This is a combined category encompassing 6 sub-types: wrong order ID (253), wrong item selection (231), wrong replacement item (221), wrong payment method (118), other argument mismatches (162), and wrong reason/justification (12).
- **PARTIAL_FULFILLMENT** (40 total, 0.8%) -- Correct tool and correct arguments, but incomplete item coverage (e.g., returned 2 of 3 requested items).

**Level 4 -- Did the agent follow proper process?**

- **MISSING_LOOKUP** (1,045 total, 20.5%) -- Fewer read/lookup tool calls than ground truth expected, indicating the agent skipped information-gathering steps.
- **NO_CONFIRMATION** (626 total, 12.3%) -- Agent executed a consequential tool without obtaining explicit user confirmation, violating customer service protocol.

[INSERT Figure 1: Aggregate Error Distribution bar chart from Phase 2]

*Figure 1: Aggregate error distribution across all models and baselines (excluding EMPTY_TRAJECTORY). PREMATURE_TERMINATION is the most frequent error type at 31.9%, followed by WRONG_TOOL at 26.1%.*

## 2.2 The Error Profile Shift

The most consequential finding from Phase 2 is that model scaling changes *which* errors dominate, rather than simply reducing error rates. This insight directly shaped our multi-module architecture.

The 4B model's failures are dominated by **WRONG_TOOL** (43.4%), indicating that the smallest model frequently cannot select the correct API endpoint from the available tool set. The 8B model resolves much of the tool selection problem but introduces a new dominant failure: **PREMATURE_TERMINATION** rises sharply to 69.3%, driven by the model stalling or abandoning tasks mid-conversation. The 14B and 32B models partially address both issues but converge on a persistent failure pattern: PREMATURE_TERMINATION accounts for 37.0% (14B) and 35% (32B) of failures, with manual inspection revealing that approximately 70% of these cases involve the agent *hallucinating* task completion -- generating convincing confirmation messages (e.g., "Your order has been cancelled, refund of $1,229.82 will be processed") without ever calling any tool.

This shift has a direct architectural implication: pre-action interventions such as input reformulation can help smaller models choose the right tools, but they cannot detect post-hoc hallucinations from larger models. A single-point intervention is therefore insufficient. Our framework addresses both failure regimes through a combination of pre-action planning (to prevent stalling) and post-action verification (to catch hallucinated completions).

[INSERT Figure 2: Error Profile Shift stacked bar chart from Phase 2]

*Figure 2: Error profile shift across model sizes showing the transition from tool selection errors (4B) to execution hallucination (8B/14B/32B). Each bar represents the failure distribution for one model size.*

## 2.3 Cause-Oriented Grouping

For actionability, we group the 8 error types into 6 cause-oriented categories. Each category maps to a potential multi-agent intervention, providing a direct link between observed failures and pipeline design decisions.

*Table 1: Cause-oriented error categories (aggregate across all models).*

| Cause Category | Error Types | Count | % Failures |
|---|---|---|---|
| Execution Failure | PREMATURE_TERMINATION + EMPTY_TRAJECTORY | 2,951 | 57.9% |
| Tool Selection Error | WRONG_TOOL | 1,327 | 26.07% |
| Argument Error | WRONG_ARGUMENT + subtypes | 997 | 19.58% |
| Authentication Bypass | AUTH_MISSING | 782 | 15.3% |
| Incomplete Resolution | MISSING_LOOKUP + PARTIAL_FULFILLMENT | 1,085 | 21.3% |
| Protocol Violation | NO_CONFIRMATION + WRONG_ARGUMENT | 626 | 12.29% |

*Note: Percentages exceed 100% because individual trajectories may exhibit multiple error types. WRONG_ARGUMENT appears in both Argument Error and Protocol Violation as some argument failures are also process violations.*

Execution Failure alone accounts for 57.9% of all failures, making it the highest-priority target. However, because this category spans two distinct root causes -- agents that never start (stalling) and agents that claim to finish without acting (hallucination) -- it requires two different mechanisms: proactive task decomposition to prevent stalling, and retrospective action verification to detect hallucinated completions.

Tool Selection Error and Argument Error together account for 45.6% of failures and share a common theme: the agent has the right intent but selects the wrong operation or passes incorrect parameters. These errors are addressable through context enrichment (narrowing the tool set, surfacing relevant policies) and argument validation (comparing tool call parameters against prior lookup results).

Authentication Bypass and Protocol Violation represent process errors where the agent reaches the right outcome through an improper procedure. These are addressable through prompt-level interventions that inject mandatory process steps (authentication, confirmation) into the agent's context.

## 2.4 Error-to-Module Mapping

The cause-oriented analysis leads directly to our module design. Each pipeline module targets a specific subset of error types through a defined mechanism. Table 2 shows this mapping along with the estimated addressable failure count derived from our Phase 2 analysis.

*Table 2: Mapping from error types to pipeline modules with addressable failure counts.*

| Error Type | Target Module | Count | Correction Mechanism |
|---|---|---|---|
| PREMATURE_TERMINATION (never started) | Task Planner | ~427 | Explicit step checklist prevents agent from stalling at task onset |
| PREMATURE_TERMINATION (hallucinated) | Action Gate | ~640 | Detects mismatch between claimed completion and actual tool execution |
| WRONG_TOOL | Context Injector | 669 | Narrows tool list to task-relevant subset, reducing selection confusion |
| WRONG_ARGUMENT | Action Gate | 798 | Validates argument values against results from prior lookup calls |
| AUTH_MISSING | Context Injector | 422 | Injects mandatory authentication step into agent prompt |
| MISSING_LOOKUP | Completion Check | 567 | Audits executed lookups against steps planned by Task Planner |
| NO_CONFIRMATION | Context Injector | 276 | Injects explicit confirmation requirement into agent prompt |
| PARTIAL_FULFILLMENT | Completion Check | 32 | Detects incomplete checklist items and resumes execution |

This mapping covers 6 of the 8 error types. The two types not directly addressed are EMPTY_TRAJECTORY (infrastructure failures unrelated to agent behavior) and a portion of WRONG_TOOL errors that stem from fundamental model capability limitations rather than context deficiency. The total addressable failure count across all modules is approximately 3,831 trajectories, representing 75.3% of the 5,090 failures observed in Phase 2.

The key design principle reflected in this mapping is *separation of concerns*: no single module attempts to address all error types. Instead, each module operates at a specific stage of the agent loop (pre-action, during execution, or post-action) and targets a specific failure mechanism. This modular design allows each component to be implemented with the simplest effective technique -- the Task Planner uses a single LLM call at task start, the Action Gate operates within the existing agent loop using code-based detection augmented by LLM verification only when a violation is detected, and the Context Injector and Completion Check are implemented as fully deterministic code-based checks requiring zero additional inference.


# 3. Architecture and Method

## 3.1 Overview

Our pipeline wraps the standard τ-bench agent loop with five modules that operate at different stages of the conversation. A key design constraint is that the pipeline makes **no modifications to the τ-bench codebase itself**. Instead, `PipelineAgent` inherits from τ-bench's `Agent` base class and overrides the `solve()` method, providing a drop-in replacement that augments the agent loop without altering the benchmark's environment, evaluation logic, or user simulator. This ensures that all performance differences are attributable to the pipeline, not to changes in the evaluation framework.

The five modules execute in the following order within each task:

```
                          TASK START
                              |
                    [1] Task Planner  (1 LLM call)
                              |
                    [2] Context Injector  (0 LLM calls)
                              |
                     +--------+--------+
                     |  CONVERSATION   |
                     |     LOOP        |
                     |                 |
                     |  LLM generates  |
                     |  proposed action|
                     |        |        |
                     |  [3] State      |
                     |      Tracker    |
                     |   (0 LLM calls) |
                     |        |        |
                     |  [4] Action     |
                     |      Gate       |
                     |  (0-1 LLM calls |
                     |   per trigger)  |
                     |        |        |
                     |   env.step()    |
                     |        |        |
                     |  loop until done|
                     +--------+--------+
                              |
                    [5] Completion Checker  (0 LLM calls)
                              |
                          TASK END
```

*Figure 3: Pipeline flow showing module placement relative to the agent conversation loop. Modules 1-2 run once at task start (pre-loop), Module 3-4 run each turn within the loop (per-step), and Module 5 runs once after the loop ends (post-loop).*

The pipeline's computational overhead is deliberately minimal. Only the Task Planner requires an LLM call (once per task at startup). The Action Gate may trigger an additional LLM call for regeneration, but only when a violation is detected -- in the common case where no violation occurs, it operates as a zero-cost code-based check. The remaining three modules (Context Injector, State Tracker, Completion Checker) are fully deterministic and require zero inference.

## 3.2 Module 1: Task Planner

**What it does.** The Task Planner decomposes the user's initial message into a concrete, ordered checklist of 3--6 actionable steps. For example, given "I want to cancel order #W12345," the planner produces steps such as: (1) Authenticate user via email or name+zip, (2) Look up order #W12345 details, (3) Verify order is pending, (4) Present cancellation details and get confirmation, (5) Cancel order with reason.

**When it runs.** Once per task, immediately after the environment is reset and the first user message is received. This is a pre-loop operation.

**What errors it targets.** PREMATURE_TERMINATION (stalling variant, ~427 failures). By providing an explicit step sequence, the planner prevents the agent from stalling at task onset -- a failure mode where the agent responds conversationally without initiating any tool calls.

**Computational cost.** 1 LLM call per task. The planner uses the same agent model with `max_tokens=2048` and `enable_thinking=False` to avoid reasoning trace overhead.

**Key implementation details.** The planner uses domain-specific system prompts (one for retail, one for airline) that include canonical step templates (e.g., "always authenticate first"). Output is parsed as a JSON array with fallback to numbered-line splitting. A validation pass rejects steps that reference hallucinated tool names (e.g., `make_reservation` instead of `book_reservation`) by checking against a whitelist of known tool names and identifiers. Steps containing XML-like tags or exceeding 200 characters are discarded to guard against malformed model output (the Qwen3 think-tag problem, see Section 4.3). If parsing fails entirely, the pipeline continues without a checklist -- the remaining modules operate independently.

## 3.3 Module 2: Context Injector

**What it does.** The Context Injector builds an augmented system prompt by appending three categories of additional context after the wiki (domain policy document): (1) matched policy excerpts relevant to the user's request, (2) authentication and confirmation reminders, and (3) the task planner's checklist.

**When it runs.** Once per task, after the Task Planner completes and before the conversation loop begins. This is a pre-loop operation.

**What errors it targets.** WRONG_TOOL (669 failures via policy-guided tool selection), AUTH_MISSING (422 failures via authentication reminders), and NO_CONFIRMATION (276 failures via confirmation reminders).

**Computational cost.** 0 LLM calls. The module is entirely deterministic.

**Key implementation details.** Policy matching is keyword-based: the user's first message is scanned for verb keywords (cancel, return, exchange, modify, book, etc.) and matched against a domain-specific policy map that returns relevant policy excerpts. Matches are capped at 3 excerpts, with verb keywords prioritized over noun keywords, to limit prompt bloat that could confuse smaller models. The prompt layout places the wiki first (preserving vLLM prefix caching across turns), followed by matched policies, then tools and instructions, and finally reminders and the checklist at the end -- exploiting recency bias in small models so that authentication and confirmation requirements are positioned closest to the generation boundary. For the tool-calling strategy, tools are passed via the API's `tools` parameter rather than embedded in the prompt, following τ-bench's baseline convention.

## 3.4 Module 3: State Tracker

**What it does.** The State Tracker maintains a structured record of the conversation's evolving state across turns. It tracks: authentication status (whether the user has been identified), entity IDs (order IDs, reservation IDs, user IDs, item IDs, payment method IDs), tool call history (categorized as consequential, authentication, or read-only), user confirmations (with negation-aware detection and recency gating), and step counts.

**When it runs.** Every turn within the conversation loop, after the LLM generates an action and after the environment returns a response. It is updated in three places: (1) from the initial user message at task start, (2) from each action the agent proposes, and (3) from each environment observation.

**What errors it targets.** The State Tracker does not directly correct errors. Instead, it provides the structured state data that enables the Action Gate (Module 4) and Completion Checker (Module 5) to detect violations.

**Computational cost.** 0 LLM calls. All state updates are deterministic regex-based extractions and dictionary lookups.

**Key implementation details.** The tracker distinguishes between domain-specific tool categories: retail consequential tools (e.g., `cancel_pending_order`, `exchange_delivered_order_items`) and airline consequential tools (e.g., `cancel_reservation`, `update_reservation_flights`). Confirmation detection is negation-aware: before accepting a keyword match (e.g., "yes", "confirm", "proceed"), it checks the preceding 15 characters for negation prefixes ("not ", "don't ", "no, ", etc.). Confirmations are recency-gated to a window of 2 user messages, preventing stale confirmations from permanently disabling the confirmation check. The initial user message is excluded from confirmation detection to avoid false positives from phrases like "Sure, I'd like to cancel..." in the task description.

## 3.5 Module 4: Action Gate

**What it does.** The Action Gate sits between the LLM's proposed action and the environment's `step()` method. It runs five code-based checks on every proposed action and, if any check fails, appends a correction message to the conversation and asks the LLM to regenerate its action (up to `max_retries=2` attempts). The five checks are:

1. **Hallucinated Completion** -- Detects when the agent sends a `respond` action containing completion phrases (e.g., "has been cancelled," "successfully processed") but no consequential tool call has been made. Also catches claims of success after a failed consequential call.
2. **Inaction / Auth Stall** -- Detects when the agent has taken 3+ steps without any tool calls (inaction) or 6+ steps without authentication (auth stall). On auth stall, instructs the agent to transfer to a human agent rather than continuing to ask for credentials indefinitely.
3. **Auth Gate** -- Blocks consequential tool calls when the user has not been authenticated.
4. **Confirmation Gate** -- Blocks consequential tool calls when the user has not given explicit confirmation within the last 2 messages.
5. **Argument Validation** -- Checks that consequential tool calls include all required parameters (e.g., `cancel_pending_order` requires `order_id` and `reason`).

**When it runs.** Every turn within the conversation loop, after the LLM generates an action and before `env.step()` executes it.

**What errors it targets.** PREMATURE_TERMINATION (hallucinated variant, ~640 failures), AUTH_MISSING (422 failures), NO_CONFIRMATION (276 failures), WRONG_ARGUMENT (798 failures, via missing-parameter detection).

**Computational cost.** 0 LLM calls when all checks pass (the common case). 1 LLM call per retry when a violation is detected and regeneration is triggered. Maximum of `max_retries` (default 2) regeneration calls per turn. In practice, most turns pass all checks with zero additional inference.

**Key implementation details.** Correction messages are prefixed with "SYSTEM NOTICE" -- a marker used by the token-aware context manager (Section 3.7) to deprioritize correction exchanges during truncation, since they are less informative than real conversation turns. For the tool-calling strategy, corrections are formatted as `role="tool"` messages referencing the rejected tool call's ID, maintaining valid message ordering for the LLM API. Regeneration uses `build_llm_context()` to truncate the updated history (including the correction) before calling the LLM, preventing context window overflows during retries. A `Timeout` exception during regeneration causes the gate to pass the original action through rather than crashing the entire task.

## 3.6 Module 5: Completion Checker

**What it does.** The Completion Checker performs a post-conversation audit by comparing the Task Planner's checklist against the State Tracker's accumulated data. It produces an audit log that records which checklist steps were completed, which were missed, whether authentication was performed before consequential actions, whether user confirmation was obtained, and whether information lookups preceded write operations.

**When it runs.** Once per task, after the conversation loop ends. This is a post-loop operation.

**What errors it targets.** MISSING_LOOKUP (567 failures) and PARTIAL_FULFILLMENT (32 failures). The audit log identifies tasks where the agent skipped required information-gathering steps or completed only a subset of requested operations.

**Computational cost.** 0 LLM calls. The module uses heuristic keyword matching to determine step completion.

**Key implementation details.** Step completion is determined by keyword-based heuristics: authentication steps are marked complete if the state tracker recorded successful authentication, lookup steps are marked complete if at least one read tool was called, confirmation steps are marked complete if the user provided confirmation, and execution steps are marked complete if at least one consequential tool was called. The audit log is stored in the trajectory's `info` metadata for post-hoc analysis but does not affect the task's reward -- the Completion Checker is a diagnostic tool, not an active intervention. This design avoids the risk of the checker incorrectly overriding a correct outcome based on imperfect heuristics.

## 3.7 Token-Aware Context Management

The pipeline manages conversation histories that can grow to tens of thousands of tokens over multi-step tasks. Rather than allowing the context to overflow (which causes `ContextWindowExceededError` crashes), the pipeline implements a token-aware truncation system called `build_llm_context()`.

**Dual-history architecture.** The pipeline maintains two separate conversation histories: `full_history`, which is the complete, untruncated record of all messages (used for final trajectory output and by the Action Gate for state inspection), and `llm_context`, which is a truncated copy created fresh before each LLM call. This separation ensures that truncation decisions are based on the full conversation and that diagnostic data is never lost.

**Turn-aware truncation.** Messages are grouped into logical turns (assistant message + tool/user response pairs) before truncation. Turns are kept or dropped as atomic units, preventing orphaned tool responses or broken role alternation that would cause LLM API errors.

**Correction deprioritization.** During truncation, messages containing the "SYSTEM NOTICE" marker (Action Gate corrections) are separated from real conversation turns and dropped first. Correction exchanges are less informative for the agent's ongoing task and often contain repeated instructions, making them low-priority candidates for retention.

**Facts buffer.** When messages are dropped, a regex-based extractor scans them for key identifiers (order IDs matching `#W\d+`, user IDs, reservation IDs, item IDs, payment method IDs) and injects a compact "Previously retrieved information" summary into the truncated context as a system message. This prevents the agent from losing access to IDs it looked up earlier in the conversation.

**Budget configuration.** The token budget defaults to 30,000 tokens for the A100 deployment (against Qwen3's 40,960-token context window), leaving a 27% margin to account for tokenizer mismatch between litellm's tiktoken-based counter and Qwen3's actual tokenizer. An emergency budget of 35,000 tokens is used when the system prompt alone exceeds the primary budget. Both values are configurable via environment variables (`TOKEN_BUDGET`, `EMERGENCY_BUDGET`) for deployments with different `max-model-len` settings (e.g., 12,000 tokens on Intel Gaudi with `max-model-len=16384`).


## 4. Implementation Challenges and Bug Analysis

Building a multi-module inference pipeline around a third-party benchmark on shared HPC infrastructure exposed a range of systems-level, model-level, and library-level challenges. This section documents the infrastructure constraints we navigated, the 16 bugs we discovered and resolved through systematic debugging, and the speed optimizations that made full-scale evaluation feasible.

### 4.1 Infrastructure Challenges

Our pipeline runs two vLLM instances concurrently on a single NVIDIA A100 80GB GPU: one serving the Qwen3-32B-AWQ user simulator and one serving the Qwen3-14B-AWQ agent. This dual-model deployment on shared hardware introduced four interconnected constraints.

**GPU memory budgeting.** Each vLLM instance requires an explicit `--gpu-memory-utilization` allocation at startup. The combined allocation must remain at or below 80% of the GPU's total memory to leave headroom for CUDA kernels, temporary activations, and KV cache growth. For the 14B agent configuration, we allocated 30% to the agent and 55% to the user simulator, leaving a 15% buffer. Early runs with a 4B agent used allocations that summed to 95%, triggering out-of-memory errors during KV cache expansion (Bug #8).

**Sequential model loading.** When both vLLM instances were launched simultaneously, the second model observed negative available KV cache memory because the first model had not yet stabilized its GPU memory footprint. This race condition caused initialization failures under concurrent startup (Bug #9). We resolved this by loading the user simulator first, waiting for its `/health` endpoint to return HTTP 200, and only then launching the agent server.

**Model routing proxy.** The τ-bench framework supports only a single `OPENAI_API_BASE` endpoint for all model requests. Since our agent and user simulator run on separate vLLM ports, we implemented a lightweight Flask proxy on port 9000 that inspects the `model` field in each incoming request and routes it to the appropriate vLLM server. The proxy initially ran in single-threaded mode, which caused request timeouts when concurrent evaluation workers issued simultaneous requests (Bug #3). Adding `threaded=True` resolved this.

**SLURM job management.** Running multi-hour evaluation jobs on the ASU Sol cluster required robust process lifecycle management. Our batch scripts needed to reliably start both vLLM servers, the proxy, and the evaluation runner, then clean up all background processes on job completion or failure. Early versions used `trap EXIT` for cleanup, which interacted poorly with SLURM's signal handling; we switched to explicit `trap SIGTERM SIGINT` with post-benchmark cleanup (Bug #14).

### 4.2 Bugs Found and Fixed

We discovered and fixed 16 bugs during implementation, testing, and evaluation. Several of these bugs were invisible during static code review and only manifested at runtime with the Qwen3 model family's specific output characteristics. The tables below summarize all bugs by category.

**Pipeline Bugs (Bugs 1--7)**

| # | Severity | File | Bug Description | Fix |
|---|----------|------|-----------------|-----|
| 1 | Critical | `task_planner.py` | Qwen3's `<think>` tags not stripped from planner output; bracket extraction gated behind a backtick check that Qwen3 never triggers | Strip `<think>` tags via regex; always attempt bracket extraction regardless of backtick presence |
| 2 | Critical | `context_injector.py` | No sanity check on planner output -- garbage checklists containing `<think>` tags were injected verbatim into the system prompt | Reject checklist steps that start with `<`, end with `>`, or exceed 200 characters |
| 3 | High | `proxy.py` | Single-threaded Flask server caused request timeouts at concurrency > 1 | Added `threaded=True` to `app.run()` |
| 4 | High | `pipeline_agent.py`, `action_gate.py` | `message.content.split("Action:")` crashes with `AttributeError` when `content` is `None` | Guarded with `(message.content or "")` |
| 5 | High | `retail_policies.py`, `airline_policies.py` | No instruction telling the agent to ask the user for credentials rather than transferring to a human agent | Added an "ask, don't transfer" rule to the general reminders section |
| 6 | High | `pipeline_agent.py` | Agent `<think>` tags not stripped from conversation history, causing context window bloat of 600--1,100 tokens per message and eventual `ContextWindowExceededError` | Strip `<think>` tags from assistant messages before appending to history |
| 7 | High | `task_planner.py` | Unclosed `<think>` tags not handled -- model spent its entire token budget reasoning without producing a closing tag or any checklist content | Added regex handling for unclosed tags; increased `max_tokens` from 300 to 1,024 |

**Batch Script Bugs (Bugs 8--9)**

| # | Severity | File | Bug Description | Fix |
|---|----------|------|-----------------|-----|
| 8 | High | `run_benchmark.sbatch` | GPU memory over-allocated for 4B agent (0.25 + 0.70 = 0.95, exceeding the 0.80 budget), causing OOM on KV cache allocation | Reduced to 0.15 agent / 0.55 user simulator (0.70 total) |
| 9 | High | `run_benchmark.sbatch` | Concurrent vLLM server initialization caused GPU memory contention -- second model saw negative available KV cache | Implemented sequential loading with health-check gating |

**Post-Benchmark Improvements (Bugs 10--14)**

| # | Severity | File | Bug Description | Fix |
|---|----------|------|-----------------|-----|
| 10 | High | `task_planner.py` | Planner produced empty checklists 72.7% of the time on 8B model because `<think>` reasoning consumed the entire token budget | Increased `max_tokens` to 2,048; disabled thinking via `enable_thinking: False`; added explicit no-think prompt instruction |
| 11 | High | `action_gate.py` | `litellm.Timeout` during gate regeneration crashed the entire task run | Catch timeout exceptions; pass the original action through instead of failing |
| 12 | High | `action_gate.py` | Agent stalled indefinitely on authentication loops, consuming 20--30 steps with zero reward | After 6+ steps with no tool calls and no authentication, instruct the agent to transfer to a human |
| 13 | Medium | `completion_checker.py` | Recency-gated `has_confirmation()` produced false positives in post-task audit | Switched to total confirmation count for post-hoc auditing |
| 14 | Medium | Batch scripts | `trap EXIT` interacted poorly with SLURM signal handling, causing inconsistent cleanup | Changed to `trap SIGTERM SIGINT` with explicit post-benchmark cleanup |

**Infrastructure Bugs (Bugs 15--16)**

| # | Severity | File | Bug Description | Fix |
|---|----------|------|-----------------|-----|
| 15 | High | Batch scripts | Tool-calling strategy produced 0% pass rate because vLLM requires `--enable-auto-tool-choice --tool-call-parser hermes` flags for structured function calling | Added the required flags to the agent vLLM startup command |
| 16 | High | `run_eval.py` | `litellm`'s internal `HTTPHandler` has a one-hour TTL; upon expiry, its `__del__` method closes a shared `httpx.Client` still in use by a cached OpenAI SDK client, causing `"Cannot send a request, as the client has been closed"` errors | Set a persistent `litellm.client_session` to prevent garbage collection; wrapped `get_env()` in try/except for resilience |

Bug #16 was particularly consequential: it limited our initial baseline runs to a single trial per strategy (102 of 750 requested task-runs, or 13.6% completion), wasting approximately 5 hours of GPU time.

### 4.3 Case Study: The Qwen3 Think-Tag Problem

The most instructive class of bugs centered on Qwen3's default behavior of wrapping all output in `<think>...</think>` reasoning tags. This single model-level behavior triggered a cascade of failures across four pipeline modules, and its resolution required coordinated changes throughout the codebase.

**The problem.** By default, Qwen3 models emit a `<think>` tag at the start of every response, reason within that block, close it with `</think>`, and only then produce the user-visible output. This behavior was not documented in the model card at the time of our development. For our pipeline, this meant that every LLM call -- the task planner, the agent itself, the action gate's regeneration, and the user simulator -- produced output prefixed with reasoning traces that downstream components did not expect.

**Manifestation 1: Empty checklists (Bug #1, #10).** The task planner calls the agent model with a maximum token budget. When the model spent 80--100% of that budget inside the `<think>` block reasoning about the task, it either produced a truncated response with an unclosed tag (Bug #7) or left no remaining tokens for the JSON checklist. On the 8B model, 72.7% of planner calls returned empty checklists.

**Manifestation 2: Garbage injection (Bug #2).** When the planner did produce output, the `<think>` tags were embedded in the checklist text. The context injector, which appends the checklist to the system prompt, injected raw XML-like tags into the prompt. This confused the agent model into treating the injected text as instructions to output more `<think>` tags, creating a feedback loop.

**Manifestation 3: Context window overflow (Bug #6).** Each agent response contained 600--1,100 tokens of `<think>` reasoning. Over a 10--20 step conversation, this added 6,000--22,000 tokens of reasoning traces to the history, rapidly exhausting the 40,960-token context window.

**Manifestation 4: User simulator contamination.** The 32B user simulator exhibited the same behavior. Its `<think>` tags, if left in the conversation history, consumed additional context budget and occasionally confused the agent's parsing of user messages.

**The solution.** We applied a defense-in-depth strategy:

1. **Disable thinking at the API level.** We set `enable_thinking=False` via litellm's `extra_body` parameter on all nine `completion()` calls in the pipeline (four in the pipeline agent, four in the action gate, and one in the task planner). This instructs the model not to emit reasoning traces.
2. **Regex stripping as defense-in-depth.** Even with thinking disabled, we strip `<think>...</think>` blocks (including unclosed tags) from all assistant and user messages before they enter the conversation history. This guards against model non-compliance or API-level flag failures.
3. **Checklist validation.** The context injector rejects any checklist step containing XML-like tags, providing a final safety net against malformed planner output.

This experience underscores that deploying open-weight models in a pipeline setting requires defensive handling of model-specific output conventions. A behavior that is merely verbose in a single-turn chat application can become a cascading failure in a multi-module pipeline where each component's output feeds into the next.

### 4.4 Speed Optimizations

Runtime analysis of our initial 14B baseline run (Job 48860544) revealed that 4 hours and 41 minutes of a 10-hour GPU allocation produced only 102 task-runs across three strategies. Three optimizations, informed by this analysis, substantially improved throughput.

| Optimization | Scope | Rationale | Impact |
|---|---|---|---|
| `enable_thinking=False` on all 9 `completion()` calls | `pipeline_agent.py` (4 calls), `action_gate.py` (4 calls), `task_planner.py` (1 call) | Reasoning traces consumed 40--60% of generated tokens with no benefit -- the action gate uses code-based detection with fixed correction strings, not LLM reasoning | ~40--60% reduction in tokens generated per LLM call |
| Reduce `max_num_steps` from 30 to 20 | `run_eval.py`, `pipeline_agent.py` | Stalled tasks burned 20--30 steps for zero reward; passing tasks averaged fewer than 10 steps | Eliminated 10+ wasted steps per stalling task |
| Increase concurrency from 2 to 3 for 14B | `run_baseline.sbatch`, `run_all_baselines.sbatch` | vLLM telemetry showed zero request queuing at concurrency 2; the A100's KV cache had 2.5x spare capacity | ~33% improvement in task throughput |

The first optimization was the most impactful: by eliminating think-tag generation across all LLM calls, we reduced both the token count per call and the cumulative context growth rate. Combined with the stall-step reduction and increased concurrency, these changes brought the per-strategy runtime for a 50-task, 5-trial airline evaluation within a single SLURM job's time budget.


## 5. Experimental Setup

### Hardware and Infrastructure

All experiments were conducted on the ASU Sol Supercomputing cluster using a single NVIDIA A100 80GB GPU (CUDA 13.0, Driver 580.95.05). We use the `tau-bench` conda environment with PyTorch and vLLM for inference.

### Models

We employ a dual-model architecture where the agent and user simulator run as separate vLLM instances on the same GPU:

- **Agent Model**: Qwen3-4B-AWQ, with 8B, 14B, and 32B AWQ variants for model-size scaling experiments. The agent model is varied across experiments while the user simulator remains fixed, isolating the effect of agent capability on task performance.
- **User Simulator**: Qwen3-32B-AWQ, held constant across all experiments to ensure fair comparison. The user simulator follows τ-bench's persona-driven dialogue protocol, where each task specifies user attributes, goals, and behavioral constraints.

### Inference Engine

We use vLLM (Kwon et al., 2023) with the following configuration: prefix caching enabled for KV-cache reuse across turns, maximum model length of 40,960 tokens, and enforce-eager mode to avoid CUDA graph overhead. Thinking/reasoning mode is disabled (`enable_thinking=False`) on all LLM calls to reduce token overhead -- our pipeline modules use code-based detection rather than chain-of-thought reasoning.

### Benchmark

We evaluate on τ-bench (Yao et al., 2024), a benchmark for tool-agent-user interaction in real-world customer service domains:

- **Retail domain**: 115 tasks covering customer service scenarios including order lookups, refunds, exchanges, and account modifications.
- **Airline domain**: 50 tasks covering flight modifications, cancellations, rebookings, and policy-sensitive requests.

Each task defines a user persona with specific attributes, a goal, and ground-truth database state changes that constitute success.

### Strategies

We evaluate three agent interaction strategies:

- **ReAct**: The agent produces interleaved Thought and Action steps, reasoning explicitly before each tool call or response.
- **ACT**: The agent produces Actions without explicit reasoning traces.
- **Tool-Calling (Function Calling)**: The agent uses structured function-call syntax to invoke tools, with the LLM generating tool names and arguments as structured output.

### Evaluation Metric

We report Pass^k for k = 1 through 5, where each task is run for 5 independent trials. Pass^k measures the fraction of tasks where all k randomly-selected trials succeed, providing a stricter measure of reliability than single-trial accuracy. Pass^1 corresponds to standard single-trial success rate, while Pass^5 requires perfect consistency across all trials.

### Dual-Model Routing

Since τ-bench supports only a single `OPENAI_API_BASE` endpoint, we implement a lightweight Flask proxy on port 9000 that inspects the `model` field in each request and routes to the correct vLLM instance (port 8000 for the agent, port 8001 for the user simulator). Both vLLM instances are loaded sequentially to avoid GPU memory contention during model initialization.

### GPU Memory Allocation

The combined `--gpu-memory-utilization` across both vLLM instances is kept at or below 0.80 to reserve headroom for CUDA kernels and transient allocations.

*Table 3: GPU memory allocation for dual-model deployment on a single A100 80GB.*

| Agent Model | Agent GPU Util. | User Sim GPU Util. | Reserved |
|---|---|---|---|
| Qwen3-4B-AWQ | 0.15 | 0.55 | ~30% |
| Qwen3-8B-AWQ | 0.20 | 0.55 | ~25% |
| Qwen3-14B-AWQ | 0.30 | 0.55 | ~15% |
| Qwen3-32B-AWQ | 0.45 | 0.45 | ~10% |


# 6. Results

## 6.1 Main Results

Table 4 presents the complete Pass^1 results across all evaluated configurations. We report results for four model sizes (4B, 8B, 14B, 32B), two domains (airline, retail), and three strategies (act, react, tool-calling) where available. Baseline and pipeline results are shown side by side with the absolute delta.

*Table 4: Complete Pass^1 results across all configurations. B = baseline, P = pipeline. Delta is P minus B. Entries marked "---" indicate missing data (configuration not run or insufficient results).*

| Model | Domain | Strategy | B Pass^1 | P Pass^1 | Delta | B Tasks | P Tasks | B Trials | P Trials |
|-------|--------|----------|----------|----------|-------|---------|---------|----------|----------|
| **32B** | Airline | act | 0.260 | 0.364 | **+0.104** | 50 | 50 | 5 | 5 |
| **32B** | Airline | react | 0.230 | 0.404 | **+0.174** | 50 | 50 | 2 | 5 |
| **32B** | Airline | tool-calling | 0.248 | 0.298 | **+0.050** | 50 | 50 | 5 | 5 |
| **32B** | Retail | act | 0.179 | 0.245 | **+0.066** | 115 | 58 | 5 | 5 |
| **32B** | Retail | react | 0.351 | 0.335 | -0.016 | 114 | 58 | 3 | 5 |
| **32B** | Retail | tool-calling | 0.239 | 0.155 | -0.084 | 114 | 58 | 5 | 4 |
| **14B** | Airline | act | 0.244 | 0.304 | **+0.060** | 50 | 50 | 5 | 5 |
| **14B** | Airline | react | 0.244 | 0.208 | -0.036 | 50 | 50 | 5 | 5 |
| **14B** | Airline | tool-calling | 0.192 | ---** | --- | 50 | --- | 5 | --- |
| **14B** | Retail | tool-calling | 0.189 | 0.182 | -0.007 | 115 | 115 | 5 | 5 |
| **4B** | Airline | act | 0.333 | 0.233 | -0.100 | 50 | 50 | 5* | 5 |
| **4B** | Airline | react | 0.311 | 0.212 | -0.099 | 50 | 50 | 5* | 5 |
| **4B** | Airline | tool-calling | 0.000 | 0.224 | ---*** | 25 | 50 | 5 | 5 |
| **4B** | Retail | act | --- | 0.067 | --- | --- | 115 | --- | 5 |
| **4B** | Retail | react | --- | 0.118 | --- | --- | 115 | --- | 5 |
| **4B** | Retail | tool-calling | --- | 0.149 | --- | --- | 115 | --- | 5 |
| **8B** | Retail | tool-calling | --- | 0.056 | --- | --- | 36 | --- | 1 |
| **8B** | Retail | react | --- | 0.000 | --- | --- | 1 | --- | 1 |

*\* 4B baselines had 5 trials requested but some tasks incomplete (23 of 50 for act, 10 of 50 for react).*
*\*\* 14B airline tool-calling pipeline was not run due to GPU time constraints.*
*\*\*\* 4B airline tool-calling delta is omitted because the baseline covered only tasks 25--49 (25 of 50 tasks), making the comparison unreliable.*

[INSERT Figure 4: Grouped bar chart of Pass^1 baseline vs. pipeline for 32B and 14B configurations]

## 6.2 Headline Findings

The pipeline's strongest improvements occur at the 32B model scale in the airline domain:

**32B Airline react: +0.174 absolute improvement** (0.230 to 0.404). This is the largest single improvement across all configurations, though the baseline used only 2 trials (compared to 5 for the pipeline), making this comparison less reliable than the act and tool-calling results. The react strategy at 32B produces verbose reasoning traces that, in the baseline, frequently lead to hallucinated completions. The Action Gate's hallucination detection check directly targets this failure mode.

**32B Airline act: +0.104 absolute improvement** (0.260 to 0.364). The act strategy, which produces actions without explicit reasoning, benefits from the Context Injector's policy excerpts and the Task Planner's checklist, which provide the structured guidance that the act strategy lacks by design.

**14B Airline act: +0.060 absolute improvement** (0.244 to 0.304). At the 14B scale, the act strategy again shows the largest improvement, consistent with the pattern that action-only strategies benefit most from pipeline-provided structure.

**32B configurations improve across 4 of 6 settings.** The pipeline improves Pass^1 in airline act (+0.104), airline react (+0.174), airline tool-calling (+0.050), and retail act (+0.066). Retail react (-0.016) and retail tool-calling (-0.084) show regressions, and the retail results are based on partial runs (58 of 115 tasks) that may not be representative.

## 6.3 Reliability Improvement (Pass^k Analysis)

The pipeline's most significant contribution is not in single-trial accuracy (Pass^1) but in multi-trial reliability (Pass^k for k > 1). Pass^k measures the probability that *all* k randomly-selected trials succeed, making it an exponentially stricter metric as k increases.

*Table 5: Pass^k comparison for key configurations. Higher k represents stricter reliability requirements.*

| Configuration | k | Baseline | Pipeline | Delta |
|---------------|---|----------|----------|-------|
| 32B Airline act | 1 | 0.260 | 0.364 | +0.104 |
| 32B Airline act | 2 | 0.127 | 0.248 | +0.121 |
| 32B Airline act | 3 | 0.024 | 0.192 | **+0.168** |
| 32B Airline act | 4 | 0.000 | 0.152 | **+0.152** |
| 32B Airline act | 5 | 0.000 | 0.120 | **+0.120** |
| 32B Airline react | 1 | 0.230 | 0.404 | +0.174 |
| 32B Airline react | 5 | N/A (2 trials) | 0.160 | --- |
| 32B Airline tool-calling | 1 | 0.248 | 0.298 | +0.050 |
| 32B Airline tool-calling | 5 | 0.000 | 0.040 | **+0.040** |
| 14B Airline act | 1 | 0.244 | 0.304 | +0.060 |
| 14B Airline act | 2 | 0.162 | 0.216 | +0.054 |
| 14B Airline act | 3 | 0.132 | 0.180 | +0.048 |
| 14B Airline act | 4 | 0.112 | 0.156 | +0.044 |
| 14B Airline act | 5 | 0.100 | 0.140 | +0.040 |
| 32B Retail react | 1 | 0.351 | 0.335 | -0.016 |
| 32B Retail react | 3 | 0.026 | 0.107 | **+0.081** |
| 14B Retail tool-calling | 1 | 0.189 | 0.182 | -0.007 |
| 14B Retail tool-calling | 2 | 0.062 | 0.071 | +0.009 |
| 14B Retail tool-calling | 3 | 0.024 | 0.035 | +0.011 |

The most striking result is at **32B Airline act Pass^5**: the baseline achieves **0.000** (no task succeeded in all 5 trials), while the pipeline achieves **0.120** (12% of tasks succeeded in all 5 trials). This means the baseline 32B act agent never produces a task result that is reliably reproducible, while the pipeline enables consistent success on a meaningful fraction of tasks. The same pattern holds for 32B Airline tool-calling, where baseline Pass^5 is 0.000 and pipeline Pass^5 is 0.040.

Notably, even configurations where Pass^1 is flat or slightly negative show improved Pass^k at higher k. The 32B Retail react configuration has a Pass^1 regression of -0.016, but its Pass^3 improves from 0.026 to 0.107 -- a 4.1x improvement. Similarly, 14B Retail tool-calling has a Pass^1 delta of -0.007 but improves at both Pass^2 (+0.009) and Pass^3 (+0.011). This pattern indicates that the pipeline reduces variance across trials rather than simply increasing mean performance, making agent behavior more predictable and reliable.

[INSERT Figure 5: Line plot of Pass^k (k=1..5) for 32B Airline act, baseline vs. pipeline]

## 6.4 Where the Pipeline Underperforms

The pipeline does not uniformly improve performance. Three categories of regression are observed:

**4B act and react (Airline): -0.100 and -0.099.** The 4B model shows consistent regression under the act and react strategies. At this model scale, the dominant failure mode is WRONG_TOOL (43.4% of failures in Phase 2), which the pipeline addresses primarily through the Context Injector's policy excerpts. However, the additional context injected by the pipeline (policy excerpts, reminders, checklist) may overwhelm the 4B model's limited context processing capacity. The augmented system prompt is approximately 1,500--2,500 tokens longer than the baseline prompt, and the 4B model's attention mechanism may not effectively integrate this additional information. Furthermore, the Action Gate's correction messages add further prompt complexity that the 4B model struggles to follow. Notably, 4B baselines had some incomplete tasks (23 of 50 tasks missing one trial for act; 10 of 50 for react), and the tool-calling baseline covered only tasks 25--49, making precise comparison difficult. (*Denotes 5 trials requested but some tasks incomplete.)

**32B Retail tool-calling: -0.084.** This regression is notable given the strong 32B airline results. However, the pipeline's retail evaluation covered only 58 of 115 tasks (a partial run), making this comparison less reliable than the airline results where both baseline and pipeline cover all 50 tasks. The retail domain's higher task complexity -- exchanges involving specific item selections, multi-step address modifications, and complex return policies -- may interact differently with the pipeline's policy injection, particularly if injected policy excerpts conflict with the specific task requirements.

**14B Airline react: -0.036.** A modest regression that may fall within noise given τ-bench's high user simulator variance. The react strategy already provides explicit reasoning traces, so the pipeline's Task Planner checklist may be partially redundant, adding prompt length without proportional benefit.

## 6.5 8B and Partial Results

The 8B model was evaluated only in a very limited capacity: retail tool-calling produced Pass^1 = 0.056 across 36 tasks with a single trial, and retail react produced 0 correct answers on a single task. These results are insufficient for meaningful analysis but are included for completeness.

The 8B model's limited evaluation reflects a prioritization decision: given constrained GPU time, we focused evaluation resources on the 4B, 14B, and 32B configurations where Phase 2's error analysis provided the strongest hypotheses about pipeline effectiveness. The 8B model's dominant failure mode (PREMATURE_TERMINATION at 69.3%, driven by stalling) is theoretically addressable by the Task Planner's checklist, but this hypothesis remains untested at scale.


# 7. Analysis and Discussion

## 7.1 Why 32B Benefits Most

The 32B model shows the strongest pipeline improvements across both domains and multiple strategies. We attribute this to the alignment between the 32B model's dominant failure mode and the pipeline's primary intervention mechanism.

Phase 2 identified that 32B failures are dominated by PREMATURE_TERMINATION at 35%, with approximately 70% of these cases involving hallucinated completion -- the agent generating convincing success messages without executing any tool calls. The Action Gate's hallucinated completion check (Check 1) directly targets this failure mode by detecting the combination of completion-indicating phrases and zero consequential tool calls. At the 32B scale, the model has sufficient capability to understand and follow the correction instructions when the gate intervenes, making the retry mechanism effective.

The 32B model also benefits more from the Context Injector because it can effectively process the additional context. Larger models have been shown to better integrate information from longer prompts, meaning the injected policy excerpts and reminders are more likely to influence the 32B model's decisions in the intended direction. By contrast, the same information may degrade performance in smaller models that lack the attention capacity to reconcile the additional instructions with the existing prompt (as observed with 4B).

## 7.2 Why 4B Regresses on Act/React

The 4B model's consistent regression under act (-0.100) and react (-0.099) strategies represents the pipeline's most significant failure case. Two complementary explanations account for this result.

First, the pipeline adds meaningful overhead to the system prompt. The Context Injector appends policy excerpts (300--800 tokens), authentication and confirmation reminders (~200 tokens), and the task checklist (100--300 tokens). For the 4B model, whose effective context processing capacity is more limited, this additional prompt content may dilute the core instructions rather than augment them. The 4B model's WRONG_TOOL error rate of 43.4% in Phase 2 suggests fundamental limitations in tool selection that cannot be resolved by prompt augmentation alone.

Second, the Action Gate's correction mechanism assumes the model can interpret and act on structured correction instructions. When the gate detects a violation and appends a message like "HALLUCINATED COMPLETION: You claimed the task is complete but no consequential tool call was made," the 4B model may not reliably parse and follow this instruction, leading to further confusion and potentially worse actions on retry. The correction-and-retry loop may thus introduce additional failure modes that outweigh the violations it catches.

These findings suggest that pipeline interventions should be model-size-aware: smaller models may benefit from simpler interventions (shorter prompts, fewer modules) while larger models can absorb the full pipeline's complexity.

## 7.3 Airline vs. Retail Domain Differences

The pipeline shows consistently stronger improvements in the airline domain compared to retail:

- **Airline:** 5 of 7 comparable configurations improve (32B act +0.104, 32B react +0.174, 32B tool-calling +0.050, 14B act +0.060; only 14B react -0.036 regresses).
- **Retail:** 2 of 4 comparable configurations improve (32B act +0.066; 32B react -0.016, 32B tool-calling -0.084, 14B tool-calling -0.007).

We identify two structural factors that explain this asymmetry.

**Task complexity.** Airline tasks involve a smaller number of distinct operations (book, cancel, modify flights, modify baggage, send certificate) with relatively clear eligibility rules (24-hour cancellation window, cabin class restrictions, insurance status). The pipeline's policy injection is well-suited to these structured rules. Retail tasks span a broader range of operations with more complex argument requirements -- exchanges require specific item ID matching, returns require payment method selection, and address modifications require seven separate fields. The pipeline's argument validation (Action Gate Check 5) catches missing parameters but cannot validate argument *values*, leaving the more nuanced retail errors unaddressed.

**Task count asymmetry.** The airline domain has 50 tasks; the retail domain has 115 tasks. Our partial retail runs (58 of 115 tasks for 32B pipeline) cover only half the task distribution, potentially overrepresenting easier or harder task subsets. The airline results, where both baseline and pipeline evaluate all 50 tasks, provide more reliable comparisons.

## 7.4 Strategy-Level Analysis

The three strategies interact differently with the pipeline:

**Act benefits most consistently.** The act strategy, which produces actions without explicit reasoning, shows the strongest pipeline improvements at both 32B (+0.104 airline) and 14B (+0.060 airline). Without reasoning traces, the act strategy's baseline performance is more susceptible to hallucinated completions and stalling -- failure modes that the Action Gate and Task Planner directly address. The pipeline effectively compensates for the reasoning that the act strategy omits.

**React shows the largest peak improvement but higher variance.** The 32B airline react result (+0.174) is the single strongest improvement, but 14B airline react shows a regression (-0.036). The react strategy already provides structured reasoning, so the pipeline's value depends on whether the model's reasoning is accurate. When the model's reasoning is good but occasionally leads to hallucinated conclusions (32B), the Action Gate catches these. When the model's reasoning is marginal (14B), the additional prompt complexity from the pipeline may degrade the reasoning quality itself.

**Tool-calling shows moderate, mixed results.** The tool-calling strategy uses structured function-call syntax, which provides some of the same guardrails as the pipeline (e.g., argument structure is enforced by the API). This overlap may explain why tool-calling shows smaller improvements than act/react at 32B (+0.050 airline) and regressions in some retail configurations (-0.084 at 32B retail). The pipeline's corrections for tool-calling also require a specific message format (role="tool" with the rejected tool_call_id) that adds implementation complexity and may interact unpredictably with the model's function-calling behavior.

## 7.5 Reliability vs. Accuracy Tradeoff

The Pass^k analysis in Section 6.3 reveals a pattern that is arguably the pipeline's most important contribution: even when Pass^1 is flat or slightly negative, higher-k pass rates often improve substantially. This suggests the pipeline operates primarily as a **variance reducer** rather than an accuracy booster.

The mechanism is straightforward. The Action Gate's checks are deterministic: a hallucinated completion is always caught, an unauthenticated consequential call is always blocked, and a missing-parameter tool call always triggers a correction. These checks eliminate specific failure modes that, in the baseline, occur stochastically -- sometimes the model hallucinates, sometimes it does not. By removing these stochastic failure modes, the pipeline reduces the per-trial failure probability, which compounds multiplicatively in the Pass^k metric.

Consider the 32B Airline act configuration: baseline Pass^1 = 0.260 (any single trial has a 26% chance of success), but baseline Pass^5 = 0.000 (no task succeeds on all 5 trials). Pipeline Pass^1 = 0.364 (36.4% per trial), and pipeline Pass^5 = 0.120 (12% of tasks succeed on all 5 trials). The per-trial success rate improved by 40% in relative terms (from 26.0% to 36.4%), but the all-trials reliability improved from zero to a meaningful value. This is because the pipeline eliminated specific failure modes that previously caused intermittent failures on otherwise-solvable tasks.

This reliability improvement has practical significance: in production deployments, agents must perform consistently, not just well on average. The pipeline's contribution to reliability may be more valuable than its contribution to single-trial accuracy.

## 7.6 Limitations

Several limitations constrain the conclusions that can be drawn from this evaluation.

**Incomplete experimental coverage.** Not all configurations have matching baseline and pipeline data. The 4B baselines cover all 50 airline tasks but with some tasks missing one trial (23 incomplete for act, 10 for react); the tool-calling baseline covered only tasks 25--49. No 4B retail baselines exist. The 32B retail pipeline covered only 58 of 115 tasks. The 8B model was evaluated only minimally. Missing configurations (14B airline tool-calling pipeline, 14B retail act/react pipeline, 4B retail baselines) leave gaps in the comparison matrix. These gaps reflect GPU time constraints on shared HPC infrastructure rather than design choices.

**User simulator variance.** τ-bench uses an LLM-based user simulator that introduces stochastic variation across trials. The same task can produce different user behaviors (different phrasing, different willingness to provide information), making trial-to-trial variation inherent to the benchmark. No single task passed all 3 strategies across our runs, highlighting the high variance introduced by user simulation. This variance means that some observed deltas (particularly small ones like 14B retail tool-calling at -0.007) may not be statistically significant.

**Single GPU constraint.** Running both the agent and user simulator on a single A100 80GB GPU required aggressive memory management (Section 4.1) and limited the maximum model size. The 32B agent was paired with a 32B user simulator, consuming 90% of GPU memory and leaving minimal headroom. This constraint prevented us from evaluating larger models or using unquantized weights, both of which could affect pipeline effectiveness.

**Pipeline overhead on small models.** The pipeline's fixed overhead (additional prompt tokens, correction messages, task planner LLM call) represents a larger relative cost for smaller models. The 4B regressions suggest that future work should explore model-size-adaptive pipeline configurations that reduce overhead for smaller models.

**No ablation study.** We report results for the full pipeline only and do not isolate the contribution of individual modules. An ablation study (disabling one module at a time) would clarify which modules provide the most value and whether any modules actively harm performance at specific model scales. This is an important direction for future work but was not feasible within the available GPU budget. However, the trajectory highlights document (Deliverable 4) provides qualitative evidence of individual module contributions through side-by-side trajectory analysis, identifying the Action Gate as the most impactful module across all five examined cases.


## 8. Related Work

**IRMA** (Ding et al., 2025) introduces a three-module input-reformulation framework consisting of Memory, Constraints, and Tool Suggestion modules that reformulate agent inputs to reduce errors in τ-bench. Their ablation study demonstrates that the Constraints module provides the largest individual performance gain. However, IRMA is explicitly designed as a "verification-loop-free" system -- it does not check whether the agent actually executed the tools it planned to use. Our error analysis reveals that hallucinated completion (the agent claiming success without performing the required actions) is the dominant failure mode for larger models, particularly at the 32B scale. This class of errors lies outside IRMA's design scope. Furthermore, IRMA adds three LLM calls per agent turn (approximately 30-45 calls per task), whereas our pipeline adds only a single LLM call per task for task planning, with all other modules operating through code-based heuristics.

**Agent-R** (Yuan et al., 2025) employs Monte Carlo Tree Search to train language model agents to reflect on and recover from errors through iterative self-training. While this approach yields strong results, it requires generating training data from error trajectories and fine-tuning model parameters. This makes Agent-R impractical in our setting, where we evaluate fixed, quantized Qwen3 models without any parameter modification -- our pipeline must operate as a purely inference-time intervention.

**CORRECT** (Yu et al., 2025) proposes a training-free error recognition framework that distills error patterns into cached schemata for multi-agent systems. Their approach of encoding error patterns into lightweight lookup structures shares a design philosophy with our Action Gate module, which uses pattern-matching rules to detect unsafe or incorrect tool calls. The key difference is granularity: CORRECT operates at the inter-agent communication level, while our Action Gate verifies individual tool calls against domain-specific policy constraints.

**ReIn** (Kim et al., 2026) introduces Reasoning Inception, a test-time intervention that injects external reasoning into an agent's decision-making process to enable error recovery. This parameter-free, inference-time approach is similar in spirit to our Task Planner and Context Injector modules, which inject structured task decompositions and policy-relevant context into the agent's prompt. However, ReIn focuses primarily on recovering from user-induced conversational errors, whereas our pipeline targets agent-side failures in tool selection, parameter construction, and premature task termination.

**Our differentiation.** Unlike prior work, our pipeline is designed around a data-driven error taxonomy derived from automated analysis of 5,090 failed trajectories across four model sizes and three interaction strategies. Each of the five pipeline modules targets specific, quantified error categories -- we can trace every intervention back to the failure pattern that motivated it. This error-analysis-driven methodology ensures that no module is speculative; each has a measurable mandate grounded in observed failure distributions. Additionally, our architecture is deliberately lightweight: only the Task Planner requires an LLM call (once per task), while the remaining four modules operate through deterministic code, making the pipeline suitable for resource-constrained single-GPU deployment.


---


## 9. Individual Contributions

**Wei-An Wang.** Designed and implemented the complete 5-module pipeline architecture (`pipeline_agent.py`, `task_planner.py`, `context_injector.py`, `state_tracker.py`, `action_gate.py`, `completion_checker.py`; approximately 2,400 lines of Python). Built the model routing proxy for dual-vLLM operation on a single GPU. Developed the `run_eval.py` evaluation runner and `compare_results.py` analysis tooling. Identified and resolved 16 bugs through systematic debugging, including critical issues in think-tag handling, context injection validation, and GPU memory allocation. Conducted detailed tool-calling failure analysis (35 failures across 7 categories) and runtime performance profiling. Applied inference-time speed optimizations (thinking mode disabled, step limits, concurrency tuning). Authored the Phase 3 report. In prior phases: designed the overall framework architecture, authored the Phase 2 multi-agent pipeline proposal, and surveyed related work.

**Divyesh Patel.** [Phase 3 contributions to be added: ran 4B pipeline benchmarks across all strategies and domains; additional contributions to be specified.] In prior phases: led the error analysis infrastructure effort, developed automated classification scripts (`error_analysis_folder.py`, `error_sampler.py`), and conducted manual review of 32B PREMATURE_TERMINATION and AUTH_MISSING failure samples.

**Abhishek Mulasi.** [Phase 3 contributions to be added.] In prior phases: contributed to error taxonomy development and trajectory sampling, reviewed WRONG_TOOL and MISSING_LOOKUP failure patterns, and assisted with cause-oriented error grouping.

**Devesh Valluru.** [Phase 3 contributions to be added: running 32B benchmark experiments; additional contributions to be specified.] In prior phases: ran Phase 1 baseline benchmarks on the Sol and Intel Gaudi environments, and debugged infrastructure challenges including HuggingFace rate limiting and GPU memory contention.


---


## 10. References

[1] Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2024). τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. *arXiv preprint*. https://github.com/sierra-research/tau-bench

[2] Ding, Y., Chen, X., Li, Z., & Wang, H. (2025). IRMA: Input Reformulation for Multi-Agent Tool Calling. *arXiv preprint arXiv:2508.20931*.

[3] Yuan, Z., Li, X., Zhang, Y., & Liu, T. (2025). Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training. *arXiv preprint*.

[4] Yu, Z., Wang, R., Chen, L., & Zhang, M. (2025). CORRECT: A Training-Free Error Recognition Framework for Multi-Agent Systems. *arXiv preprint*.

[5] Kim, J., Park, S., & Lee, H. (2026). ReIn: Reasoning Inception for Test-Time Error Recovery in Language Agents. *arXiv preprint*.

[6] Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP '23)*.

[7] BerriAI. (2024). LiteLLM: Call All LLM APIs Using the OpenAI Format. https://github.com/BerriAI/litellm

[8] Alibaba Cloud. (2025). Qwen3 Technical Report. https://qwenlm.github.io/blog/qwen3/
