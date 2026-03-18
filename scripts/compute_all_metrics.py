#!/usr/bin/env python3
"""
Compute Pass^k (k=1..5) metrics for ALL available baseline and pipeline result JSONs.
Outputs a consolidated Markdown table and a JSON data file.
"""

import json
import os
import sys
from math import comb
from collections import Counter
from typing import Dict, List, Any, Tuple

BASE = "/home/wwang360/CSE598/Phase3"
RESULTS_DIR = os.path.join(BASE, "results")
PHASE1_DIR = os.path.join(BASE, "Phase1-result")
OUTPUT_MD = os.path.join(BASE, "docs/report/all_metrics.md")
OUTPUT_JSON = os.path.join(BASE, "docs/report/all_metrics.json")


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def merge_jsons(paths: List[str]) -> List[Dict[str, Any]]:
    """Load and concatenate multiple JSON result files."""
    merged = []
    for p in paths:
        merged.extend(load_json(p))
    return merged


def list_jsons(directory: str) -> List[str]:
    """List all JSON files in a directory, sorted."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".json")
    )


def compute_metrics(results: List[Dict[str, Any]], max_k: int = 5) -> Dict[str, Any]:
    """Compute average reward and Pass^k metrics.

    For tasks with fewer trials than num_trials (the max), we use the actual
    number of trials for that task in the combination calculation.

    We compute Pass^k up to min(max_k, num_trials).
    """
    if not results:
        return {
            "avg_reward": 0.0,
            "pass_k": {},
            "num_tasks": 0,
            "num_trials": 0,
            "total_results": 0,
            "successes": 0,
        }

    def is_successful(reward: float) -> bool:
        return abs(reward - 1.0) < 1e-6

    # Group by task_id
    task_data: Dict[int, List[Dict]] = {}
    for r in results:
        tid = r["task_id"]
        if tid not in task_data:
            task_data[tid] = []
        task_data[tid].append(r)

    # Determine num_trials as the max trials per task
    trials_per_task = {tid: len(records) for tid, records in task_data.items()}
    num_trials = max(trials_per_task.values())

    # Count successes per task
    c_per_task: Dict[int, int] = {}
    for tid, records in task_data.items():
        c_per_task[tid] = sum(1 for r in records if is_successful(r["reward"]))

    # Rewards
    rewards = [r["reward"] for r in results]
    avg_reward = sum(rewards) / len(rewards)
    successes = sum(1 for r in rewards if is_successful(r))

    # Pass^k: for each task, use its actual trial count
    # pass_hat_k = (1/num_tasks) * sum(comb(c,k)/comb(n_i,k) for each task)
    # where n_i is the number of trials for task i
    # If n_i < k, that task contributes 0 (comb(n_i, k) = 0, skip to avoid div-by-zero)
    pass_hat_ks = {}
    num_tasks = len(c_per_task)
    for k in range(1, min(max_k, num_trials) + 1):
        total = 0.0
        contributing_tasks = 0
        for tid in c_per_task:
            n_i = trials_per_task[tid]
            c = c_per_task[tid]
            denom = comb(n_i, k)
            if denom > 0:
                total += comb(c, k) / denom
                contributing_tasks += 1
        # Average over ALL tasks (including those with n_i < k, which contribute 0)
        pass_hat_ks[k] = total / num_tasks if num_tasks > 0 else 0.0

    # Check if any tasks have incomplete trials
    incomplete_tasks = sum(1 for n in trials_per_task.values() if n < num_trials)

    return {
        "avg_reward": avg_reward,
        "pass_k": pass_hat_ks,
        "num_tasks": num_tasks,
        "num_trials": num_trials,
        "total_results": len(results),
        "successes": successes,
        "incomplete_tasks": incomplete_tasks,
    }


# ── Define all result entries ──────────────────────────────────────────

def build_entries() -> List[Dict[str, Any]]:
    """Build list of (label, model, domain, strategy, mode, results) entries."""
    entries = []

    def add(model, domain, strategy, mode, results, note=""):
        metrics = compute_metrics(results)
        entries.append({
            "model": model,
            "domain": domain,
            "strategy": strategy,
            "mode": mode,
            "metrics": metrics,
            "note": note,
        })

    # ── Phase 3 Pipeline Results ───────────────────────────────────

    # 4B Pipeline
    for domain, ddir in [("Airline", "Airline"), ("Retail", "Retail")]:
        for strategy, fname in [
            ("act", f"{domain.lower()}_act_pipeline.json"),
            ("react", f"{domain.lower()}_react_pipeline.json"),
            ("tool-calling", f"{domain.lower()}_tool-calling_pipeline.json"),
        ]:
            path = os.path.join(RESULTS_DIR, "4B", ddir, fname)
            if os.path.exists(path):
                add("4B", domain, strategy, "pipeline", load_json(path))

    # 8B Pipeline (Retail only)
    path_8b_react = os.path.join(RESULTS_DIR, "8B/Retail/react-agent-8b-retail-pipeline_0317021306.json")
    if os.path.exists(path_8b_react):
        data = load_json(path_8b_react)
        add("8B", "Retail", "react", "pipeline", data, note=f"only {len(data)} result(s)")

    # 8B tool-calling: file1 (36 results) is superset of file2 (16 results, overlapping tasks)
    path_8b_tc1 = os.path.join(RESULTS_DIR, "8B/Retail/tool-calling-agent-8b-retail-pipeline_0317025424.json")
    if os.path.exists(path_8b_tc1):
        data = load_json(path_8b_tc1)
        add("8B", "Retail", "tool-calling", "pipeline", data, note="1 trial per task")

    # 14B Pipeline
    p14_act = os.path.join(RESULTS_DIR, "14B/Airline/act-agent-14b-airline-pipeline_0317001211.json")
    if os.path.exists(p14_act):
        add("14B", "Airline", "act", "pipeline", load_json(p14_act))

    p14_react = os.path.join(RESULTS_DIR, "14B/Airline/react-agent-14b-airline-pipeline_0316182451.json")
    if os.path.exists(p14_react):
        add("14B", "Airline", "react", "pipeline", load_json(p14_react))

    p14_tc_retail = os.path.join(RESULTS_DIR, "14B/Retail/tool-calling-agent-14b-retail-pipeline_0317063330.json")
    if os.path.exists(p14_tc_retail):
        add("14B", "Retail", "tool-calling", "pipeline", load_json(p14_tc_retail))

    # 32B Pipeline
    p32_files = {
        ("Airline", "act"): "act-agent-32b-airline-pipeline_0314112238.json",
        ("Airline", "react"): "react-agent-32b-airline-pipeline_0314135037.json",
        ("Airline", "tool-calling"): "tool-calling-agent-32b-airline-pipeline_0314112234.json",
        ("Retail", "act"): "act-agent-32b-retail-pipeline_0314112152.json",
        ("Retail", "react"): "react-agent-32b-retail-pipeline_0314134943.json",
        ("Retail", "tool-calling"): "tool-calling-agent-32b-retail-pipeline_0314112344.json",
    }
    for (domain, strategy), fname in p32_files.items():
        path = os.path.join(RESULTS_DIR, "32B", domain, fname)
        if os.path.exists(path):
            add("32B", domain, strategy, "pipeline", load_json(path))

    # ── Phase 1 Baseline Results ───────────────────────────────────

    # 4B Baseline (airline only, directories have leading space)
    for strategy, dirname in [
        ("act", " airline_act_4b"),
        ("react", " airline_react_4b"),
        ("tool-calling", " airline_toolcalling_4b"),
    ]:
        dpath = os.path.join(PHASE1_DIR, "4B", dirname)
        jsons = list_jsons(dpath)
        if jsons:
            data = merge_jsons(jsons)
            note = ""
            if strategy == "tool-calling":
                note = "partial: tasks 25-49 only"
            add("4B", "Airline", strategy, "baseline", data, note=note)

    # 14B Baseline
    for domain in ["Airline", "Retail"]:
        for strategy in ["act", "react", "tool-calling"]:
            fname = f"{domain.lower()}_14b_{strategy}.json"
            path = os.path.join(PHASE1_DIR, "14B", fname)
            if os.path.exists(path):
                add("14B", domain, strategy, "baseline", load_json(path))

    # 32B Baseline (split files, need merging)
    for domain in ["airline", "retail"]:
        for strategy in ["act", "react", "tool-calling"]:
            dirname = f"{domain}_{strategy}_32b"
            dpath = os.path.join(PHASE1_DIR, "32B", dirname)
            jsons = list_jsons(dpath)
            if jsons:
                data = merge_jsons(jsons)
                add("32B", domain.capitalize(), strategy, "baseline", data)

    return entries


def model_sort_key(model: str) -> int:
    """Sort models by size."""
    sizes = {"4B": 4, "8B": 8, "14B": 14, "32B": 32}
    return sizes.get(model, 999)


def strategy_sort_key(strategy: str) -> int:
    order = {"act": 0, "react": 1, "tool-calling": 2}
    return order.get(strategy, 99)


def mode_sort_key(mode: str) -> int:
    return 0 if mode == "baseline" else 1


def generate_markdown(entries: List[Dict[str, Any]]) -> str:
    """Generate the Markdown report."""
    lines = []
    lines.append("# Pass^k Metrics — All Baselines and Pipeline Results")
    lines.append("")
    lines.append("Generated by `scripts/compute_all_metrics.py`")
    lines.append("")

    # Sort entries
    entries.sort(key=lambda e: (
        model_sort_key(e["model"]),
        e["domain"],
        strategy_sort_key(e["strategy"]),
        mode_sort_key(e["mode"]),
    ))

    # Main table
    lines.append("## Consolidated Results Table")
    lines.append("")
    header = "| Model | Domain | Strategy | Mode | Tasks | Trials | Pass^1 | Pass^2 | Pass^3 | Pass^4 | Pass^5 | Avg Reward | Notes |"
    sep =    "|-------|--------|----------|------|------:|-------:|-------:|-------:|-------:|-------:|-------:|-----------:|-------|"
    lines.append(header)
    lines.append(sep)

    for e in entries:
        m = e["metrics"]
        pk = m["pass_k"]

        def fmt_pk(k):
            if k in pk:
                return f"{pk[k]:.4f}"
            return "—"

        notes = []
        if e.get("note"):
            notes.append(e["note"])
        if m["incomplete_tasks"] > 0:
            notes.append(f"{m['incomplete_tasks']} tasks incomplete")
        if m["num_trials"] < 5:
            notes.append(f"only {m['num_trials']} trial(s)")
        note_str = "; ".join(notes)

        row = (
            f"| {e['model']} | {e['domain']} | {e['strategy']} | {e['mode']} "
            f"| {m['num_tasks']} | {m['num_trials']} "
            f"| {fmt_pk(1)} | {fmt_pk(2)} | {fmt_pk(3)} | {fmt_pk(4)} | {fmt_pk(5)} "
            f"| {m['avg_reward']:.4f} | {note_str} |"
        )
        lines.append(row)

    lines.append("")

    # Summary by model
    lines.append("## Summary by Model Size")
    lines.append("")
    for model in ["4B", "8B", "14B", "32B"]:
        model_entries = [e for e in entries if e["model"] == model]
        if not model_entries:
            continue
        lines.append(f"### {model}")
        lines.append("")

        # Paired comparisons (baseline vs pipeline)
        pairs = {}
        for e in model_entries:
            key = (e["domain"], e["strategy"])
            if key not in pairs:
                pairs[key] = {}
            pairs[key][e["mode"]] = e

        has_pairs = any("baseline" in v and "pipeline" in v for v in pairs.values())
        if has_pairs:
            lines.append("| Domain | Strategy | Mode | Pass^1 | Pass^2 | Pass^3 | Avg Reward |")
            lines.append("|--------|----------|------|-------:|-------:|-------:|-----------:|")
            for (domain, strategy) in sorted(pairs.keys(), key=lambda x: (x[0], strategy_sort_key(x[1]))):
                p = pairs[(domain, strategy)]
                for mode in ["baseline", "pipeline"]:
                    if mode in p:
                        m = p[mode]["metrics"]
                        pk = m["pass_k"]
                        lines.append(
                            f"| {domain} | {strategy} | {mode} "
                            f"| {pk.get(1, 0):.4f} | {pk.get(2, 0):.4f} | {pk.get(3, 0):.4f} "
                            f"| {m['avg_reward']:.4f} |"
                        )
                # Delta row if both exist
                if "baseline" in p and "pipeline" in p:
                    bm = p["baseline"]["metrics"]
                    pm = p["pipeline"]["metrics"]
                    d1 = pm["pass_k"].get(1, 0) - bm["pass_k"].get(1, 0)
                    d2 = pm["pass_k"].get(2, 0) - bm["pass_k"].get(2, 0)
                    d3 = pm["pass_k"].get(3, 0) - bm["pass_k"].get(3, 0)
                    dr = pm["avg_reward"] - bm["avg_reward"]
                    lines.append(
                        f"| | | **delta** "
                        f"| **{d1:+.4f}** | **{d2:+.4f}** | **{d3:+.4f}** "
                        f"| **{dr:+.4f}** |"
                    )
            lines.append("")
        else:
            # No pairs, just list
            lines.append("| Domain | Strategy | Mode | Pass^1 | Avg Reward | Notes |")
            lines.append("|--------|----------|------|-------:|-----------:|-------|")
            for e in model_entries:
                m = e["metrics"]
                pk = m["pass_k"]
                note = e.get("note", "")
                lines.append(
                    f"| {e['domain']} | {e['strategy']} | {e['mode']} "
                    f"| {pk.get(1, 0):.4f} | {m['avg_reward']:.4f} | {note} |"
                )
            lines.append("")

    # Notes section
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Pass^k formula**: `pass_hat_k = (1/num_tasks) * sum(C(c_i,k) / C(n_i,k))` where `c_i` = successes for task i, `n_i` = trials for task i")
    lines.append("- A result is successful if `|reward - 1.0| < 1e-6`")
    lines.append("- Tasks with fewer than k trials contribute 0 to Pass^k")
    lines.append("- 4B baseline: Airline only (no Retail baseline available); tool-calling covers tasks 25-49 only")
    lines.append("- 8B: Pipeline results only, no baselines; react has only 1 result")
    lines.append("- 32B baseline airline_react: only 1-2 trials per task (partial run)")
    lines.append("- 32B pipeline retail: only 58 of 115 tasks completed")
    lines.append("- Trials column shows the maximum number of trials observed; some tasks may have fewer (see Notes column)")
    lines.append("")

    return "\n".join(lines)


def generate_json_data(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate JSON-serializable data for the plot generator."""
    rows = []
    for e in entries:
        m = e["metrics"]
        row = {
            "model": e["model"],
            "domain": e["domain"],
            "strategy": e["strategy"],
            "mode": e["mode"],
            "num_tasks": m["num_tasks"],
            "num_trials": m["num_trials"],
            "total_results": m["total_results"],
            "successes": m["successes"],
            "avg_reward": round(m["avg_reward"], 6),
            "incomplete_tasks": m["incomplete_tasks"],
            "note": e.get("note", ""),
        }
        for k in range(1, 6):
            row[f"pass_k_{k}"] = round(m["pass_k"].get(k, None) or 0, 6) if k in m["pass_k"] else None
        rows.append(row)
    return rows


def main():
    print("Building entries...")
    entries = build_entries()
    print(f"Found {len(entries)} result sets")

    # Sort
    entries.sort(key=lambda e: (
        model_sort_key(e["model"]),
        e["domain"],
        strategy_sort_key(e["strategy"]),
        mode_sort_key(e["mode"]),
    ))

    # Print summary to stdout
    print(f"\n{'Model':<6} {'Domain':<8} {'Strategy':<14} {'Mode':<10} {'Tasks':>5} {'Trials':>6} {'Pass^1':>8} {'AvgR':>8}")
    print("-" * 75)
    for e in entries:
        m = e["metrics"]
        pk1 = m["pass_k"].get(1, 0)
        print(f"{e['model']:<6} {e['domain']:<8} {e['strategy']:<14} {e['mode']:<10} {m['num_tasks']:>5} {m['num_trials']:>6} {pk1:>8.4f} {m['avg_reward']:>8.4f}")

    # Generate outputs
    md = generate_markdown(entries)
    with open(OUTPUT_MD, "w") as f:
        f.write(md)
    print(f"\nMarkdown saved to {OUTPUT_MD}")

    json_data = generate_json_data(entries)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
