#!/usr/bin/env python3
"""Auto-discover and display Pass^k results from all JSON files.

Scans both results/ (Phase3) and Phase1-result/ directories recursively.
Reads raw result JSONs directly — no pre-computed metrics file needed.

Usage:
    python scripts/show_results_summary.py                  # all models
    python scripts/show_results_summary.py --model 4B       # only 4B
    python scripts/show_results_summary.py --pass-k 1       # Pass^1 only
    python scripts/show_results_summary.py --pass-k 1 3 5   # Pass^1, ^3, ^5
    python scripts/show_results_summary.py -v               # show discovered files
"""
import argparse, json, os, re, sys
from math import comb
from collections import defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT, "results")
PHASE1_DIR = os.path.join(PROJECT, "Phase1-result")

SKIP_FILES = {"all_metrics.json"}


def find_all_jsons(*roots):
    """Recursively find all .json files in given directories."""
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for f in filenames:
                if f.endswith(".json") and f not in SKIP_FILES:
                    found.append(os.path.join(dirpath, f))
    return sorted(found)


def is_result_file(data):
    """Check if loaded JSON is a tau-bench result list."""
    if not isinstance(data, list) or len(data) == 0:
        return False
    first = data[0]
    return isinstance(first, dict) and "task_id" in first and "reward" in first


def parse_config(filepath):
    """Extract (model, domain, strategy, mode) from filepath. Returns None on failure."""
    filename = os.path.basename(filepath)
    fn_lower = filename.lower()
    path_lower = filepath.lower().replace("\\", "/")
    parts = path_lower.split("/")

    # --- Mode ---
    if "phase1-result" in path_lower:
        mode = "baseline"
    elif "baseline" in fn_lower:
        mode = "baseline"
    elif "pipeline" in fn_lower:
        mode = "pipeline"
    else:
        # Phase1 files without explicit mode are baselines
        mode = "baseline"

    # --- Strategy (from filename first, then directory) ---
    strategy = None
    # Must check tool-calling before act (since 'act' is substring of 'react')
    if fn_lower.startswith("tool-calling-") or "_tool-calling_" in fn_lower or "_tool-calling." in fn_lower:
        strategy = "tool-calling"
    elif fn_lower.startswith("react-") or "_react_" in fn_lower or "_react." in fn_lower:
        strategy = "react"
    elif fn_lower.startswith("act-") or "_act_" in fn_lower or "_act." in fn_lower:
        strategy = "act"

    if not strategy:
        # Try directory names
        for p in parts:
            if "tool-calling" in p or "toolcalling" in p:
                strategy = "tool-calling"
                break
            # Only match exact directory names to avoid 'react' matching 'react-agent-...'
        if not strategy:
            for p in parts:
                p_stripped = p.strip().lower()
                if p_stripped == "react":
                    strategy = "react"
                    break
                elif p_stripped == "act":
                    strategy = "act"
                    break

    # --- Model size ---
    model = None
    for p in parts:
        p_stripped = p.strip().lower()
        if p_stripped in ("4b", "8b", "14b", "32b"):
            model = p_stripped.upper()
            break
    if not model:
        # From filename: agent-8b, agent4b, qwen32b, _14b_
        m = re.search(r"(?:agent-?)(\d+b)", fn_lower)
        if m:
            model = m.group(1).upper()
        else:
            m = re.search(r"qwen(\d+b)", fn_lower)
            if m:
                model = m.group(1).upper()
            else:
                m = re.search(r"[_-](\d+b)[_.\-]", fn_lower)
                if m:
                    model = m.group(1).upper()

    # --- Domain ---
    domain = None
    # Check path parts + filename for airline/retail
    for p in parts + [fn_lower]:
        if "airline" in p:
            domain = "Airline"
            break
        elif "retail" in p:
            domain = "Retail"
            break

    if not all([model, domain, strategy, mode]):
        return None

    return (model, domain, strategy, mode)


def compute_pass_k(results, max_k=5):
    """Compute Pass^k metrics from result dicts."""
    if not results:
        return {"num_tasks": 0, "num_trials": 0, "total_results": 0,
                "successes": 0, "avg_reward": 0.0, "incomplete_tasks": 0, "pass_k": {}}

    # Group by task_id
    task_data = defaultdict(list)
    for r in results:
        task_data[r["task_id"]].append(r)

    trials_per_task = {tid: len(recs) for tid, recs in task_data.items()}
    successes_per_task = {tid: sum(1 for r in recs if abs(r["reward"] - 1.0) < 1e-6)
                         for tid, recs in task_data.items()}

    num_tasks = len(task_data)
    num_trials = max(trials_per_task.values())
    total_results = len(results)
    total_successes = sum(successes_per_task.values())
    avg_reward = sum(r["reward"] for r in results) / total_results
    incomplete = sum(1 for n in trials_per_task.values() if n < num_trials)

    pass_k = {}
    for k in range(1, min(max_k, num_trials) + 1):
        total = 0.0
        for tid in task_data:
            n_i = trials_per_task[tid]
            c = successes_per_task[tid]
            denom = comb(n_i, k)
            if denom > 0:
                total += comb(c, k) / denom
        pass_k[k] = total / num_tasks if num_tasks > 0 else 0.0

    return {
        "num_tasks": num_tasks,
        "num_trials": num_trials,
        "total_results": total_results,
        "successes": total_successes,
        "avg_reward": avg_reward,
        "incomplete_tasks": incomplete,
        "pass_k": pass_k,
    }


def dedup_results(results):
    """Deduplicate by (task_id, trial), keeping the last occurrence."""
    seen = {}
    for r in results:
        key = (r["task_id"], r.get("trial", 0))
        seen[key] = r
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description="Auto-discover and display Pass^k results.")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model size (e.g. 4B, 8B, 14B, 32B)")
    parser.add_argument("--pass-k", type=int, nargs="+", default=None,
                        help="Show only specific Pass^k columns (e.g. --pass-k 1 or --pass-k 1 3 5)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show which files were discovered per config")
    args = parser.parse_args()

    k_values = args.pass_k or [1, 2, 3, 4, 5]
    for k in k_values:
        if k < 1 or k > 5:
            print(f"Error: --pass-k values must be between 1 and 5, got {k}")
            return 1

    # Discover all JSON files
    all_files = find_all_jsons(RESULTS_DIR, PHASE1_DIR)

    # Parse and group by config
    config_files = defaultdict(list)
    skipped = []
    for fpath in all_files:
        config = parse_config(fpath)
        if config:
            config_files[config].append(fpath)
        else:
            skipped.append(fpath)

    if args.verbose:
        print(f"Discovered {len(all_files)} JSON files across {len(config_files)} configs\n")
        for config in sorted(config_files.keys()):
            files = config_files[config]
            model, domain, strategy, mode = config
            print(f"  {model:<5} {domain:<9} {strategy:<14} {mode:<10} ({len(files)} file(s))")
            for f in files:
                print(f"    {os.path.relpath(f, PROJECT)}")
        if skipped:
            print(f"\n  Skipped ({len(skipped)} files — could not parse config):")
            for f in skipped:
                print(f"    {os.path.relpath(f, PROJECT)}")
        print()

    # Load, merge, deduplicate, compute metrics
    rows = []
    load_errors = []
    for config in sorted(config_files.keys()):
        files = config_files[config]
        model, domain, strategy, mode = config

        if args.model and model.upper() != args.model.upper():
            continue

        all_results = []
        for fpath in files:
            try:
                data = json.load(open(fpath))
                if is_result_file(data):
                    all_results.extend(data)
                else:
                    load_errors.append((fpath, "not a result file"))
            except (json.JSONDecodeError, IOError) as e:
                load_errors.append((fpath, str(e)))

        if not all_results:
            continue

        all_results = dedup_results(all_results)
        metrics = compute_pass_k(all_results, max_k=max(k_values))

        rows.append({
            "model": model,
            "domain": domain,
            "strategy": strategy,
            "mode": mode,
            "metrics": metrics,
            "num_files": len(files),
        })

    if args.verbose and load_errors:
        print(f"Load errors ({len(load_errors)}):")
        for fpath, err in load_errors:
            print(f"  {os.path.relpath(fpath, PROJECT)}: {err}")
        print()

    if not rows:
        print(f"No results found{' for model ' + args.model if args.model else ''}.")
        return 1

    # Sort
    model_order = {"4B": 0, "8B": 1, "14B": 2, "32B": 3}
    strategy_order = {"act": 0, "react": 1, "tool-calling": 2}
    mode_order = {"baseline": 0, "pipeline": 1}
    rows.sort(key=lambda r: (
        model_order.get(r["model"], 99),
        r["domain"],
        strategy_order.get(r["strategy"], 99),
        mode_order.get(r["mode"], 99),
    ))

    # --- Main table ---
    def fmt(v):
        return f"{v:.4f}" if v is not None else "   -  "

    title = "CSE 598 Phase 3 — Pipeline vs Baseline Results Summary"
    if args.model:
        title += f" ({args.model.upper()})"

    k_header = "  ".join(f"{'Pass^' + str(k):>7}" for k in k_values)
    width = 62 + len(k_values) * 9
    print("=" * width)
    print(title)
    print("=" * width)
    print(f"{'Model':<6} {'Domain':<9} {'Strategy':<14} {'Mode':<10} {'Tasks':>5} {'Trials':>6} {'Reslt':>5}  {k_header}")
    print("-" * width)

    for row in rows:
        m = row["metrics"]
        k_cols = "  ".join(
            f"{fmt(m['pass_k'].get(k)):>7}" for k in k_values
        )
        print(f"{row['model']:<6} {row['domain']:<9} {row['strategy']:<14} {row['mode']:<10} "
              f"{m['num_tasks']:>5} {m['num_trials']:>6} {m['total_results']:>5}  {k_cols}")

    print("=" * width)

    # --- Improvement table ---
    pairs = {}
    for row in rows:
        key = (row["model"], row["domain"], row["strategy"])
        pairs.setdefault(key, {})[row["mode"]] = row

    has_comparison = any("baseline" in m and "pipeline" in m for m in pairs.values())
    if has_comparison:
        print(f"\nKey Comparisons (Pipeline - Baseline, Pass^1):")
        print("-" * 65)
        for key, modes in sorted(pairs.items(),
                                  key=lambda x: (model_order.get(x[0][0], 99), x[0][1],
                                                 strategy_order.get(x[0][2], 99))):
            if "baseline" in modes and "pipeline" in modes:
                b = modes["baseline"]["metrics"]["pass_k"].get(1, 0)
                p = modes["pipeline"]["metrics"]["pass_k"].get(1, 0)
                delta = p - b
                sign = "+" if delta > 0 else ""
                model, domain, strategy = key
                print(f"  {model:<5} {domain:<9} {strategy:<14}  {b:.4f} -> {p:.4f}  ({sign}{delta:.4f})")
        print("-" * 65)

    pipeline_only = sorted(
        [k for k, m in pairs.items() if "pipeline" in m and "baseline" not in m],
        key=lambda x: (model_order.get(x[0], 99), x[1], strategy_order.get(x[2], 99)),
    )
    if pipeline_only:
        print(f"\nPipeline Only (no baseline):")
        print("-" * 65)
        for key in pipeline_only:
            row = pairs[key]["pipeline"]
            model, domain, strategy = key
            print(f"  {model:<5} {domain:<9} {strategy:<14}  Pass^1: {fmt(row['metrics']['pass_k'].get(1))}")
        print("-" * 65)

    baseline_only = sorted(
        [k for k, m in pairs.items() if "baseline" in m and "pipeline" not in m],
        key=lambda x: (model_order.get(x[0], 99), x[1], strategy_order.get(x[2], 99)),
    )
    if baseline_only:
        print(f"\nBaseline Only (no pipeline):")
        print("-" * 65)
        for key in baseline_only:
            row = pairs[key]["baseline"]
            model, domain, strategy = key
            print(f"  {model:<5} {domain:<9} {strategy:<14}  Pass^1: {fmt(row['metrics']['pass_k'].get(1))}")
        print("-" * 65)

    # --- Coverage ---
    target_models = sorted(set(args.model.upper() for _ in [1]) if args.model else
                           {"4B", "8B", "14B", "32B"})
    domains = ["Airline", "Retail"]
    strategies = ["act", "react", "tool-calling"]
    all_modes = ["baseline", "pipeline"]

    existing = {(r["model"], r["domain"], r["strategy"], r["mode"]) for r in rows}
    total_results = sum(r["metrics"]["total_results"] for r in rows)

    missing = []
    for mdl in target_models:
        for dom in domains:
            for strat in strategies:
                for md in all_modes:
                    if (mdl, dom, strat, md) not in existing:
                        missing.append((mdl, dom, strat, md))

    full_tasks = {"Airline": 50, "Retail": 115}
    incomplete = []
    for row in rows:
        issues = []
        expected = full_tasks.get(row["domain"], 0)
        m = row["metrics"]
        if m["num_tasks"] < expected:
            issues.append(f"{m['num_tasks']}/{expected} tasks")
        if m["num_trials"] < 5:
            issues.append(f"{m['num_trials']}/5 trials")
        if issues:
            incomplete.append((row["model"], row["domain"], row["strategy"], row["mode"],
                               ", ".join(issues)))

    print(f"\nCoverage: {len(rows)} configs, {total_results} total result entries")
    print(f"          {len(missing)} missing, {len(incomplete)} incomplete")

    if missing:
        print(f"\nMissing Configurations ({len(missing)}):")
        print("-" * 60)
        for mdl, dom, strat, md in sorted(missing):
            print(f"  {mdl:<5} {dom:<9} {strat:<14} {md}")
        print("-" * 60)

    if incomplete:
        print(f"\nIncomplete Configurations ({len(incomplete)}):")
        print("-" * 60)
        for mdl, dom, strat, md, issue in sorted(incomplete):
            print(f"  {mdl:<5} {dom:<9} {strat:<14} {md:<10} ({issue})")
        print("-" * 60)

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
