# Context for Editing Phase 3 Submission Documents

## What This Is

CSE 598 (Agentic AI) Phase 3 submission. We built a 5-module augmented pipeline that wraps the tau-bench benchmark to improve language agent performance. The pipeline adds task planning, context injection, state tracking, action gating, and completion checking around the baseline agent loop.

## Assignment Requirements (Phase 3: 40 points, 8 pts each)

1. Final code for multi-agent system and evaluation scripts
2. **A document containing final results table and plots comparing method vs. baselines + Individual Contribution** --> `phase3_final_report.docx`
3. JSON trajectory files for each model
4. **A document of trajectory highlights: side-by-side baseline failure vs. pipeline success with annotations** --> `trajectory_highlights.docx`
5. Screenshot of results in terminal with username

## Documents to Edit

### phase3_final_report.docx
- 10-section academic report (~580 lines markdown)
- Sections: Introduction, Motivation, Architecture, Challenges, Setup, Results, Analysis, Related Work, Individual Contributions, References
- **Known remaining issues:**
  - 4 figure placeholders (`[INSERT Figure 1]` through `[INSERT Figure 5]`) need actual images inserted (provided separately as PNGs)
  - Section 9 Individual Contributions: 3 of 4 teammates have `[Phase 3 contributions to be added]` placeholders -- leave blank or fill if provided
  - Figure 1 and 2 are from Phase 2 error analysis (may not be available -- can remove or describe textually)
  - Figures 4 and 5 are the Phase 3 result plots (provided as PNGs)

### trajectory_highlights.docx
- 5 side-by-side examples of baseline failure vs pipeline success
- All from 14B airline act-strategy evaluation
- Each has: task description, baseline trajectory excerpt, pipeline trajectory excerpt, analysis table
- **Known remaining issues:**
  - All 5 examples from one config (14B airline act) -- acceptable but could note this limitation
  - Example 1 pipeline "success" works by blocking unauthorized action (not completing the task) -- a clarifying note could help

## Verified Metrics Data (use these numbers, they are authoritative)

### Key Pass^1 Comparisons (Baseline vs Pipeline)
| Config | Baseline | Pipeline | Delta |
|--------|----------|----------|-------|
| 32B Airline act | 0.260 | 0.364 | +0.104 |
| 32B Airline react | 0.230 (2 trials only) | 0.404 | +0.174 |
| 32B Airline tool-calling | 0.248 | 0.298 | +0.050 |
| 32B Retail act | 0.179 | 0.245 | +0.066 |
| 32B Retail react | 0.351 (3 trials) | 0.335 | -0.016 |
| 32B Retail tool-calling | 0.239 | 0.155 (4 trials) | -0.084 |
| 14B Airline act | 0.244 | 0.304 | +0.060 |
| 14B Airline react | 0.244 | 0.208 | -0.036 |
| 14B Airline tool-calling | 0.192 | NOT RUN | --- |
| 14B Retail tool-calling | 0.189 | 0.182 | -0.007 |
| 4B Airline act | 0.333 (5 trials, 23 tasks incomplete) | 0.233 | -0.100 |
| 4B Airline react | 0.311 (5 trials, 10 tasks incomplete) | 0.212 | -0.099 |
| 4B Airline tool-calling | 0.000 (tasks 25-49 only) | 0.224 | +0.224 |

### Key Pass^k Reliability Numbers (32B Airline act)
| k | Baseline | Pipeline | Delta |
|---|----------|----------|-------|
| 1 | 0.260 | 0.364 | +0.104 |
| 2 | 0.127 | 0.248 | +0.121 |
| 3 | 0.024 | 0.192 | +0.168 |
| 4 | 0.000 | 0.152 | +0.152 |
| 5 | 0.000 | 0.120 | +0.120 |

### Key Pass^k Reliability Numbers (14B Airline act)
| k | Baseline | Pipeline | Delta |
|---|----------|----------|-------|
| 1 | 0.244 | 0.304 | +0.060 |
| 2 | 0.162 | 0.216 | +0.054 |
| 3 | 0.132 | 0.180 | +0.048 |
| 4 | 0.112 | 0.156 | +0.044 |
| 5 | 0.100 | 0.140 | +0.040 |

### 4B Pipeline-Only Results (no baselines available for retail)
| Domain | Strategy | Pass^1 |
|--------|----------|--------|
| Retail | act | 0.067 |
| Retail | react | 0.118 |
| Retail | tool-calling | 0.149 |
| Airline | act | 0.233 |
| Airline | react | 0.212 |
| Airline | tool-calling | 0.224 |

### 8B (very limited data, include with caveats)
- Retail tool-calling: Pass^1 = 0.056 (36 tasks, 1 trial only)
- Retail react: 0.000 (1 task, 1 trial only)

## Pipeline Architecture Summary

```
User's first message
  --> [1] Task Planner (1 LLM call) --> JSON checklist (3-6 steps)
  --> [2] Context Injector (code-only) --> augmented system prompt with policies + reminders
  --> [3] State Tracker init (code-only) --> tracks auth, tools, confirmations
  --> CONVERSATION LOOP:
      --> Agent generates action
      --> [4] Action Gate (code-based checks, 0-2 LLM retries) --> 5 checks:
          1. Hallucinated completion
          2. Inaction / auth stall
          3. Auth missing
          4. No confirmation
          5. Missing arguments
      --> env.step(action)
      --> State tracker updates
  --> [5] Completion Checker (code-only, logging) --> post-task audit
```

Key constraint: No tau-bench code modifications. Pipeline wraps via PipelineAgent(Agent).

## Team Members
- Wei-An Wang (pipeline architect, all code, report author)
- Divyesh Patel (4B pipeline benchmarks on Gaudi)
- Abhishek Mulasi (error taxonomy, trajectory sampling)
- Devesh Valluru (32B benchmarks, Phase 1 baselines)

## Model Names
Use "Qwen3" consistently (Qwen3-4B-AWQ, Qwen3-14B-AWQ, Qwen3-32B-AWQ). User simulator is always Qwen3-32B-AWQ.

## Editing Guidelines
- Keep writing style academic but clear, data-driven
- Do not add emojis
- When citing numbers, use the verified metrics above -- do not change them
- The pipeline does NOT always outperform baseline. 4B regresses. Some retail configs regress. Be honest.
- The strongest story is **reliability improvement** (Pass^k for k>1), not single-trial accuracy
- The Action Gate is the most impactful module (evidenced by all 5 trajectory highlights)
