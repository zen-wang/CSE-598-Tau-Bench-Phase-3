"""
Evaluation runner for the pipeline agent.
Modeled on tau_bench/run.py with additional flags for enabling/disabling
pipeline modules and a --baseline mode.
"""

import os
import sys
import json
import random
import argparse
import traceback
import multiprocessing
from math import comb
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add tau-bench to path
_tau_bench_path = os.path.join(os.path.dirname(__file__), "..", "tau-bench")
if _tau_bench_path not in sys.path:
    sys.path.insert(0, os.path.abspath(_tau_bench_path))

from tau_bench.envs import get_env
from tau_bench.types import EnvRunResult
from tau_bench.envs.user import UserStrategy
from litellm import provider_list
import litellm
import httpx

from src.pipeline.pipeline_agent import PipelineAgent

# Fix litellm Bug #16: HTTPHandler TTL expiry closes shared httpx.Client after 1 hour,
# crashing all subsequent requests. Setting client_session bypasses the TTL cache entirely.
litellm.client_session = httpx.Client(timeout=httpx.Timeout(timeout=600.0, connect=5.0))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pipeline agent evaluation on tau-bench"
    )

    # --- Standard tau-bench args (same as tau-bench/run.py) ---
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="The model to use for the agent (e.g. agent-14b)",
    )
    parser.add_argument(
        "--model-provider", type=str, choices=provider_list, required=True,
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--user-model", type=str, default="gpt-4o",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider", type=str, choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--agent-strategy", type=str, default="tool-calling",
        choices=["tool-calling", "act", "react"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--task-split", type=str, default="test",
        choices=["train", "test", "dev"],
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument(
        "--task-ids", type=int, nargs="+",
        help="(Optional) run only the tasks with the given IDs",
    )
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument(
        "--user-strategy", type=str, default="llm",
        choices=[item.value for item in UserStrategy],
    )
    parser.add_argument("--max-num-steps", type=int, default=20)

    # --- Pipeline-specific args ---
    parser.add_argument(
        "--baseline", action="store_true",
        help="Disable all pipeline modules (pure baseline replication)",
    )
    parser.add_argument(
        "--enable-planner", type=int, default=1, choices=[0, 1],
        help="Enable Task Planner module (default: 1)",
    )
    parser.add_argument(
        "--enable-context-injector", type=int, default=1, choices=[0, 1],
        help="Enable Context Injector module (default: 1)",
    )
    parser.add_argument(
        "--enable-action-gate", type=int, default=1, choices=[0, 1],
        help="Enable Action Gate module (default: 1)",
    )
    parser.add_argument(
        "--enable-completion-checker", type=int, default=1, choices=[0, 1],
        help="Enable Completion Checker module (default: 1)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="Max retries per Action Gate check (default: 2)",
    )

    args = parser.parse_args()

    # --baseline overrides all module flags to off
    if args.baseline:
        args.enable_planner = 0
        args.enable_context_injector = 0
        args.enable_action_gate = 0
        args.enable_completion_checker = 0

    return args


def run(args) -> List[EnvRunResult]:
    random.seed(args.seed)

    # Build checkpoint path
    mode = "baseline" if args.baseline else "pipeline"
    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = (
        f"{args.log_dir}/{args.agent_strategy}-{args.model.split('/')[-1]}"
        f"-{args.env}-{mode}_{time_str}.json"
    )
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Create environment (for tools_info and wiki)
    print(f"Loading user with strategy: {args.user_strategy}")
    env = get_env(
        args.env,
        user_strategy=args.user_strategy,
        user_model=args.user_model,
        user_provider=args.user_model_provider,
        task_split=args.task_split,
    )

    # Create pipeline agent
    agent = PipelineAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=args.model,
        provider=args.model_provider,
        agent_strategy=args.agent_strategy,
        domain=args.env,
        temperature=args.temperature,
        enable_planner=bool(args.enable_planner),
        enable_context_injector=bool(args.enable_context_injector),
        enable_action_gate=bool(args.enable_action_gate),
        enable_completion_checker=bool(args.enable_completion_checker),
        max_retries_per_gate=args.max_retries,
    )

    end_index = (
        len(env.tasks) if args.end_index == -1
        else min(args.end_index, len(env.tasks))
    )

    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()

    if args.task_ids and len(args.task_ids) > 0:
        print(f"Running tasks {args.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {args.start_index} to {end_index} "
            f"(checkpoint path: {ckpt_path})"
        )
    print(
        f"Mode: {mode} | Strategy: {args.agent_strategy} | "
        f"Model: {args.model} | Domain: {args.env}"
    )
    if not args.baseline:
        print(
            f"Modules: planner={bool(args.enable_planner)} "
            f"context={bool(args.enable_context_injector)} "
            f"gate={bool(args.enable_action_gate)} "
            f"checker={bool(args.enable_completion_checker)}"
        )

    for i in range(args.num_trials):
        if args.task_ids and len(args.task_ids) > 0:
            idxs = args.task_ids
        else:
            idxs = list(range(args.start_index, end_index))
        if args.shuffle:
            random.shuffle(idxs)

        def _run(idx: int) -> EnvRunResult:
            print(f"Running task {idx} (trial {i})")
            try:
                # Each thread gets its own env instance
                isolated_env = get_env(
                    args.env,
                    user_strategy=args.user_strategy,
                    user_model=args.user_model,
                    task_split=args.task_split,
                    user_provider=args.user_model_provider,
                    task_index=idx,
                )

                res = agent.solve(
                    env=isolated_env,
                    task_index=idx,
                    max_num_steps=args.max_num_steps,
                )
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                result.info.get("error", ""),
            )
            print("-----")
            # Incremental checkpoint save
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)
            return result

        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            res = list(executor.map(_run, idxs))
            results.extend(res)

    display_metrics(results)

    # Final save (overwrites incremental to ensure clean JSON)
    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    return results


def display_metrics(results: List[EnvRunResult]) -> None:
    """Copied verbatim from tau_bench/run.py for identical metric reporting."""
    if not results:
        print("No results to display.")
        return

    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")


if __name__ == "__main__":
    args = parse_args()
    run(args)
