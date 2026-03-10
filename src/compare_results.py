"""
Compare baseline vs pipeline results.
Loads JSON result files and produces comparison tables with Pass^k metrics.
"""

import os
import sys
import json
import argparse
import csv
from math import comb
from typing import Dict, List, Any, Optional, Tuple


def load_results(path: str) -> List[Dict[str, Any]]:
    """Load a JSON result file."""
    with open(path, "r") as f:
        return json.load(f)


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute average reward and Pass^k metrics."""
    if not results:
        return {"avg_reward": 0.0, "pass_k": {}, "num_tasks": 0, "num_trials": 0}

    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set(r["trial"] for r in results))
    rewards = [r["reward"] for r in results]
    avg_reward = sum(rewards) / len(rewards)

    c_per_task: Dict[int, int] = {}
    for r in results:
        tid = r["task_id"]
        if tid not in c_per_task:
            c_per_task[tid] = 0
        if is_successful(r["reward"]):
            c_per_task[tid] += 1

    pass_hat_ks = {}
    for k in range(1, num_trials + 1):
        total = 0
        for c in c_per_task.values():
            total += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = total / len(c_per_task)

    return {
        "avg_reward": avg_reward,
        "pass_k": pass_hat_ks,
        "num_tasks": len(c_per_task),
        "num_trials": num_trials,
        "total_results": len(results),
        "successes": sum(1 for r in rewards if is_successful(r)),
        "failures": sum(1 for r in rewards if not is_successful(r)),
    }


def extract_config_from_filename(filename: str) -> Dict[str, str]:
    """Try to extract strategy, model, domain, mode from filename."""
    name = os.path.basename(filename).replace(".json", "")
    parts = name.split("-")

    config = {"strategy": "unknown", "model": "unknown", "domain": "unknown", "mode": "unknown"}

    # Expected format: {strategy}-{model}-{domain}-{mode}_{timestamp}
    if len(parts) >= 4:
        config["strategy"] = parts[0]
        config["model"] = parts[1]
        config["domain"] = parts[2]
        mode_part = parts[3]
        config["mode"] = mode_part.split("_")[0] if "_" in mode_part else mode_part

    return config


def compare_files(baseline_path: str, pipeline_path: str) -> None:
    """Compare two result files and print metrics."""
    baseline_results = load_results(baseline_path)
    pipeline_results = load_results(pipeline_path)

    baseline_metrics = compute_metrics(baseline_results)
    pipeline_metrics = compute_metrics(pipeline_results)

    baseline_config = extract_config_from_filename(baseline_path)
    pipeline_config = extract_config_from_filename(pipeline_path)

    print("=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70)

    print(f"\nBaseline: {os.path.basename(baseline_path)}")
    print(f"Pipeline: {os.path.basename(pipeline_path)}")

    print(f"\n{'Metric':<30} {'Baseline':>15} {'Pipeline':>15} {'Delta':>10}")
    print("-" * 70)

    # Average reward
    b_avg = baseline_metrics["avg_reward"]
    p_avg = pipeline_metrics["avg_reward"]
    delta = p_avg - b_avg
    print(f"{'Average Reward':<30} {b_avg:>15.4f} {p_avg:>15.4f} {delta:>+10.4f}")

    # Pass rate
    b_pass = baseline_metrics["successes"] / max(baseline_metrics["total_results"], 1)
    p_pass = pipeline_metrics["successes"] / max(pipeline_metrics["total_results"], 1)
    delta = p_pass - b_pass
    print(f"{'Pass Rate':<30} {b_pass:>14.1%} {p_pass:>14.1%} {delta:>+9.1%}")

    # Successes / Failures
    print(f"{'Successes':<30} {baseline_metrics['successes']:>15} {pipeline_metrics['successes']:>15}")
    print(f"{'Failures':<30} {baseline_metrics['failures']:>15} {pipeline_metrics['failures']:>15}")
    print(f"{'Total Results':<30} {baseline_metrics['total_results']:>15} {pipeline_metrics['total_results']:>15}")

    # Pass^k
    print(f"\n{'Pass^k Metrics':}")
    print("-" * 70)
    max_k = max(
        max(baseline_metrics["pass_k"].keys(), default=0),
        max(pipeline_metrics["pass_k"].keys(), default=0),
    )
    for k in range(1, max_k + 1):
        b_pk = baseline_metrics["pass_k"].get(k, 0)
        p_pk = pipeline_metrics["pass_k"].get(k, 0)
        delta = p_pk - b_pk
        print(f"  {'Pass^' + str(k):<28} {b_pk:>15.4f} {p_pk:>15.4f} {delta:>+10.4f}")

    # Per-task comparison
    print(f"\n{'Per-Task Analysis':}")
    print("-" * 70)
    baseline_by_task = {}
    for r in baseline_results:
        tid = r["task_id"]
        if tid not in baseline_by_task:
            baseline_by_task[tid] = []
        baseline_by_task[tid].append(r["reward"])

    pipeline_by_task = {}
    for r in pipeline_results:
        tid = r["task_id"]
        if tid not in pipeline_by_task:
            pipeline_by_task[tid] = []
        pipeline_by_task[tid].append(r["reward"])

    improved = []
    regressed = []
    unchanged = []

    all_tasks = sorted(set(baseline_by_task.keys()) | set(pipeline_by_task.keys()))
    for tid in all_tasks:
        b_rewards = baseline_by_task.get(tid, [0.0])
        p_rewards = pipeline_by_task.get(tid, [0.0])
        b_avg_t = sum(b_rewards) / len(b_rewards)
        p_avg_t = sum(p_rewards) / len(p_rewards)

        if p_avg_t > b_avg_t + 1e-6:
            improved.append(tid)
        elif p_avg_t < b_avg_t - 1e-6:
            regressed.append(tid)
        else:
            unchanged.append(tid)

    print(f"  Improved tasks:   {len(improved)} {improved[:20]}{'...' if len(improved) > 20 else ''}")
    print(f"  Regressed tasks:  {len(regressed)} {regressed[:20]}{'...' if len(regressed) > 20 else ''}")
    print(f"  Unchanged tasks:  {len(unchanged)}")

    print("=" * 70)


def compare_directory(results_dir: str, output_csv: Optional[str] = None) -> None:
    """Compare all result files in a directory, grouping by config."""
    files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    if not files:
        print(f"No JSON files found in {results_dir}")
        return

    rows = []
    for f in sorted(files):
        path = os.path.join(results_dir, f)
        results = load_results(path)
        metrics = compute_metrics(results)
        config = extract_config_from_filename(f)

        row = {
            "file": f,
            "strategy": config["strategy"],
            "model": config["model"],
            "domain": config["domain"],
            "mode": config["mode"],
            "avg_reward": metrics["avg_reward"],
            "pass_rate": metrics["successes"] / max(metrics["total_results"], 1),
            "successes": metrics["successes"],
            "failures": metrics["failures"],
            "total": metrics["total_results"],
            "num_tasks": metrics["num_tasks"],
            "num_trials": metrics["num_trials"],
        }
        # Add pass^k
        for k, v in metrics["pass_k"].items():
            row[f"pass^{k}"] = v

        rows.append(row)

    # Print table
    print("=" * 120)
    print("ALL RESULTS SUMMARY")
    print("=" * 120)
    header = f"{'Strategy':<12} {'Model':<12} {'Domain':<10} {'Mode':<10} {'Pass Rate':>10} {'Avg Reward':>12} {'Pass^1':>8} {'Total':>6}"
    print(header)
    print("-" * 120)
    for row in rows:
        print(
            f"{row['strategy']:<12} {row['model']:<12} {row['domain']:<10} "
            f"{row['mode']:<10} {row['pass_rate']:>9.1%} {row['avg_reward']:>12.4f} "
            f"{row.get('pass^1', 0):>8.4f} {row['total']:>6}"
        )
    print("=" * 120)

    # Export CSV
    if output_csv:
        fieldnames = list(rows[0].keys()) if rows else []
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n📄 CSV exported to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs pipeline results")
    subparsers = parser.add_subparsers(dest="command")

    # Compare two files
    pair = subparsers.add_parser("pair", help="Compare two result files")
    pair.add_argument("baseline", help="Path to baseline result JSON")
    pair.add_argument("pipeline", help="Path to pipeline result JSON")

    # Compare all files in directory
    dir_cmd = subparsers.add_parser("dir", help="Summarize all results in a directory")
    dir_cmd.add_argument("results_dir", help="Path to results directory")
    dir_cmd.add_argument("--csv", help="Export to CSV file")

    args = parser.parse_args()

    if args.command == "pair":
        compare_files(args.baseline, args.pipeline)
    elif args.command == "dir":
        compare_directory(args.results_dir, args.csv)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
