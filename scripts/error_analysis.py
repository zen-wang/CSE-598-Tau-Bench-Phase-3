#!/usr/bin/env python3
"""
τ-bench Error Analysis Script
==============================
Compares agent actions in trajectory JSONs against ground-truth actions
to classify failure types for Phase 2 error analysis.

Usage:
    python error_analysis.py                          # scans current directory
    python error_analysis.py --input results_dir/     # scans a specific directory
    python error_analysis.py --verbose                # detailed per-task breakdown
"""

import json
import re
import os
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path


# ── Consequential (write) tools that modify the database ──
CONSEQUENTIAL_TOOLS = {
    "cancel_pending_order",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "modify_user_address",
    "book_reservation",
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_passengers",
    "update_reservation_baggages",
    "send_certificate",
    "transfer_to_human_agents",
}

AUTH_TOOLS = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
}

READ_TOOLS = {
    "get_order_details",
    "get_user_details",
    "get_product_details",
    "get_reservation_details",
    "get_flight_details",
    "list_all_product_types",
    "search_direct_flight",
    "search_onestop_flight",
    "calculate",
    "think",
}


# ============================================================
# 1. PARSING
# ============================================================

def extract_agent_actions_from_traj(traj):
    actions = []
    for msg in traj:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        if not content:
            for tc in (msg.get("tool_calls") or []):
                func = tc.get("function", {})
                name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if name and name != "respond":
                    actions.append({"name": name, "kwargs": args})
            continue

        # ReAct / ACT JSON action
        for pattern in [
            r'Action:\s*(\{[^{}]*"name"[^{}]*\})',
            r'Action:\s*(\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\s*\})',
        ]:
            for match in re.finditer(pattern, content, re.DOTALL):
                try:
                    aj = json.loads(match.group(1))
                    name = aj.get("name", "")
                    if name and name != "respond":
                        actions.append({"name": name, "kwargs": aj.get("arguments", {})})
                except json.JSONDecodeError:
                    pass

        # Function calling style
        for match in re.finditer(
            r"Function\(arguments=['\"](.+?)['\"],\s*name=['\"](\w+)['\"]\)", content
        ):
            try:
                args = json.loads(match.group(1).replace("'", '"'))
                name = match.group(2)
                if name and name != "respond":
                    actions.append({"name": name, "kwargs": args})
            except (json.JSONDecodeError, IndexError):
                pass

        # Direct JSON
        for match in re.finditer(
            r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{[^}]*\})\}', content
        ):
            try:
                name = match.group(1)
                args = json.loads(match.group(2))
                if name and name != "respond":
                    action = {"name": name, "kwargs": args}
                    if action not in actions:
                        actions.append(action)
            except json.JSONDecodeError:
                pass

    return actions


def extract_gt_actions(task_info):
    if not task_info:
        return []
    ri = task_info.get("reward_info")
    if ri:
        gt = ri.get("actions", [])
        if gt:
            return gt
    tk = task_info.get("task")
    if tk:
        gt = tk.get("actions", [])
        if gt:
            return gt
    return task_info.get("actions", []) or []


def check_user_confirmation_before_action(traj, action_name):
    for i, msg in enumerate(traj):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        if action_name in content:
            for j in range(i - 1, -1, -1):
                prev = traj[j]
                if prev.get("role") == "user":
                    pc = (prev.get("content") or "").lower()
                    if pc.startswith("api output"):
                        continue
                    return any(w in pc for w in ["yes", "confirm", "proceed", "go ahead", "sure"])
                elif prev.get("role") == "assistant":
                    pa = (prev.get("content") or "").lower()
                    if any(p in pa for p in [
                        "confirm", "would you like to proceed", "shall i proceed",
                        "do you want to", "would you like me to", "please confirm"
                    ]):
                        continue
                    break
    return False


# ============================================================
# 2. COMPARISON & CLASSIFICATION
# ============================================================

def normalize_kwargs(kwargs):
    out = {}
    for k, v in (kwargs or {}).items():
        if isinstance(v, list):
            out[k] = sorted([str(x) for x in v])
        else:
            out[k] = str(v)
    return out


def classify_errors(gt_actions, agent_actions, traj):
    errors = []

    gt_cons = [a for a in gt_actions if a["name"] in CONSEQUENTIAL_TOOLS]
    ag_cons = [a for a in agent_actions if a["name"] in CONSEQUENTIAL_TOOLS]
    gt_auth = [a for a in gt_actions if a["name"] in AUTH_TOOLS]
    ag_auth = [a for a in agent_actions if a["name"] in AUTH_TOOLS]
    gt_reads = [a for a in gt_actions if a["name"] in READ_TOOLS]
    ag_reads = [a for a in agent_actions if a["name"] in READ_TOOLS]

    # AUTH_MISSING
    if gt_auth and not ag_auth:
        errors.append({"type": "AUTH_MISSING", "desc": "Skipped user authentication"})

    # PREMATURE_TERMINATION
    if gt_cons and not ag_cons:
        errors.append({"type": "PREMATURE_TERMINATION", "desc": "Never called required consequential tool"})
        return errors

    # WRONG_TOOL
    gt_cnames = [a["name"] for a in gt_cons]
    ag_cnames = [a["name"] for a in ag_cons]
    if set(gt_cnames) != set(ag_cnames):
        for gn in gt_cnames:
            if gn not in ag_cnames:
                wrong = [n for n in ag_cnames if n not in gt_cnames]
                errors.append({"type": "WRONG_TOOL", "desc": f"Expected '{gn}', got {wrong or 'nothing'}"})

    # ARGUMENT COMPARISON
    for gt_a in gt_cons:
        matching = [a for a in ag_cons if a["name"] == gt_a["name"]]
        if not matching:
            continue
        ag_a = matching[0]
        gt_kw = normalize_kwargs(gt_a.get("kwargs", {}))
        ag_kw = normalize_kwargs(ag_a.get("kwargs", {}))

        for key in gt_kw:
            if key not in ag_kw:
                errors.append({"type": "MISSING_ARGUMENT", "desc": f"Missing arg '{key}' in '{gt_a['name']}'"})
            elif gt_kw[key] != ag_kw[key]:
                if key == "item_ids":
                    gt_i = gt_kw[key] if isinstance(gt_kw[key], list) else [gt_kw[key]]
                    ag_i = ag_kw[key] if isinstance(ag_kw[key], list) else [ag_kw[key]]
                    if set(ag_i) < set(gt_i):
                        errors.append({"type": "PARTIAL_FULFILLMENT", "desc": f"{len(ag_i)}/{len(gt_i)} items"})
                    else:
                        errors.append({"type": "WRONG_ITEMS", "desc": f"Expected {gt_kw[key]}, got {ag_kw[key]}"})
                elif key == "new_item_ids":
                    errors.append({"type": "WRONG_REPLACEMENT", "desc": f"Expected {gt_kw[key]}, got {ag_kw[key]}"})
                elif key == "payment_method_id":
                    errors.append({"type": "WRONG_PAYMENT", "desc": f"Expected {gt_kw[key]}, got {ag_kw[key]}"})
                elif key == "order_id":
                    errors.append({"type": "WRONG_ORDER", "desc": f"Expected {gt_kw[key]}, got {ag_kw[key]}"})
                elif key == "reason":
                    errors.append({"type": "WRONG_REASON", "desc": f"Expected {gt_kw[key]}, got {ag_kw[key]}"})
                else:
                    errors.append({"type": "WRONG_ARGUMENT", "desc": f"'{key}': expected {gt_kw[key]}, got {ag_kw[key]}"})

    # MISSING_LOOKUP
    gt_rc = Counter(a["name"] for a in gt_reads)
    ag_rc = Counter(a["name"] for a in ag_reads)
    for tool, gc in gt_rc.items():
        ac = ag_rc.get(tool, 0)
        if ac < gc:
            errors.append({"type": "MISSING_LOOKUP", "desc": f"Skipped {gc - ac} call(s) to '{tool}'"})

    # NO_CONFIRMATION
    for ag_a in ag_cons:
        if ag_a["name"] in CONSEQUENTIAL_TOOLS:
            if not check_user_confirmation_before_action(traj, ag_a["name"]):
                errors.append({"type": "NO_CONFIRMATION", "desc": f"'{ag_a['name']}' without user confirmation"})

    # UNKNOWN
    if not errors:
        errors.append({"type": "UNKNOWN_NEEDS_MANUAL_REVIEW", "desc": "Actions look correct — needs manual check"})

    return errors


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

def load_tasks(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ⚠ Parse error in {filepath}: {e}")
        return []

    if isinstance(data, list):
        tasks = data
    elif isinstance(data, dict):
        if "task_id" in data and "traj" in data:
            tasks = [data]
        else:
            tasks = []
            for key in ["results", "data", "tasks", "episodes"]:
                if key in data and isinstance(data[key], list):
                    tasks = data[key]
                    break
            if not tasks:
                for v in data.values():
                    if isinstance(v, list):
                        tasks = v
                        break
    else:
        tasks = []
    return [t for t in tasks if isinstance(t, dict)]


def detect_missing_trials(tasks, expected_trials=5):
    task_trials = defaultdict(set)
    for t in tasks:
        tid = t.get("task_id")
        trial = t.get("trial")
        if tid is not None and trial is not None:
            task_trials[tid].add(trial)

    expected = set(range(expected_trials))
    missing = {}
    for tid in sorted(task_trials):
        diff = expected - task_trials[tid]
        if diff:
            missing[tid] = sorted(diff)
    return missing


def analyze_file(filepath, verbose=False, expected_trials=5):
    fname = os.path.basename(filepath)
    print(f"\n{'='*70}")
    print(f"  FILE: {fname}")
    print(f"{'='*70}")

    tasks = load_tasks(filepath)
    valid = [t for t in tasks if "traj" in t]
    print(f"  Loaded: {len(tasks)} trajectories, {len(valid)} with traj data")

    if not valid:
        print("  No valid tasks. Skipping.\n")
        return None

    # ── Missing trials ──
    missing_trials = detect_missing_trials(valid, expected_trials)

    # ── Analyze ──
    results = []
    for task in valid:
        task_id = task.get("task_id", "unknown")
        trial = task.get("trial", 0)
        info = task.get("info") or {}
        traj = task.get("traj") or []

        reward = task.get("reward")
        if reward is None:
            ri = info.get("reward_info") or {}
            reward = ri.get("reward", -1)

        r = {"task_id": task_id, "trial": trial, "reward": reward, "errors": [], "error_types": []}

        if reward == 1.0:
            r["status"] = "SUCCESS"
        elif not traj:
            r["status"] = "FAILED"
            r["errors"] = [{"type": "EMPTY_TRAJECTORY", "desc": "No trajectory data"}]
            r["error_types"] = ["EMPTY_TRAJECTORY"]
        else:
            gt_actions = extract_gt_actions(info)
            agent_actions = extract_agent_actions_from_traj(traj)
            errors = classify_errors(gt_actions, agent_actions, traj)
            r["status"] = "FAILED"
            r["errors"] = errors
            r["error_types"] = list(set(e["type"] for e in errors))

        results.append(r)

    total = len(results)
    successes = sum(1 for r in results if r["status"] == "SUCCESS")
    failures = sum(1 for r in results if r["status"] == "FAILED")

    # ── Summary ──
    print(f"\n  SUMMARY")
    print(f"  {'─'*45}")
    print(f"  Total trajectories:  {total}")
    print(f"  Successes:           {successes} ({100*successes/total:.1f}%)")
    print(f"  Failures:            {failures} ({100*failures/total:.1f}%)")

    # ── Missing trials ──
    total_missing = sum(len(v) for v in missing_trials.values())
    print(f"  Missing trials:      {total_missing} across {len(missing_trials)} tasks")
    if missing_trials:
        mt_str = ", ".join(f"task {tid}(trial {','.join(str(t) for t in ts)})"
                           for tid, ts in list(missing_trials.items())[:20])
        if len(missing_trials) > 20:
            mt_str += f" ... and {len(missing_trials)-20} more"
        print(f"    → {mt_str}")

    # ── Error distribution ──
    error_counter = Counter()
    error_task_list = defaultdict(list)

    for r in results:
        if r["status"] == "FAILED":
            for et in r["error_types"]:
                error_counter[et] += 1
                error_task_list[et].append((r["task_id"], r["trial"]))

    if failures > 0 and error_counter:
        print(f"\n  ERROR DISTRIBUTION ({failures} failed)")
        print(f"  {'─'*45}")
        ml = max(len(et) for et in error_counter)
        for et, cnt in error_counter.most_common():
            pct = 100 * cnt / failures
            bar = "█" * int(pct / 2)
            print(f"  {et:<{ml}}  {cnt:>4} ({pct:5.1f}%)  {bar}")

        # ── Print affected task_id & trial per error type ──
        print(f"\n  AFFECTED TASKS PER ERROR TYPE")
        print(f"  {'─'*45}")
        for et, _ in error_counter.most_common():
            tl = error_task_list[et]
            by_task = defaultdict(list)
            for tid, tr in tl:
                by_task[tid].append(tr)

            ids_str = ", ".join(
                f"{tid}(t{','.join(str(t) for t in sorted(trs))})"
                for tid, trs in sorted(by_task.items())
            )
            print(f"\n  [{et}] — {len(tl)} occurrences")
            print(f"    {ids_str}")

    # ── Verbose detail ──
    if verbose:
        print(f"\n  DETAILED FAILURES")
        print(f"  {'─'*45}")
        for r in results:
            if r["status"] != "FAILED":
                continue
            print(f"\n  task_id={r['task_id']}, trial={r['trial']}, reward={r['reward']}")
            for e in r["errors"]:
                print(f"    [{e['type']}] {e['desc']}")

    return {
        "file": fname,
        "total": total,
        "successes": successes,
        "failures": failures,
        "error_distribution": dict(error_counter.most_common()),
    }


def main():
    parser = argparse.ArgumentParser(description="τ-bench Error Analysis")
    parser.add_argument("--input", "-i", default=".",
                        help="Directory or file (default: current dir)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Detailed per-task breakdown")
    parser.add_argument("--trials", "-t", type=int, default=5,
                        help="Expected trials per task (default: 5)")
    args = parser.parse_args()

    p = Path(args.input)
    if p.is_file() and p.suffix == ".json":
        json_files = [p]
    elif p.is_dir():
        json_files = sorted(p.glob("*.json"))
    else:
        print(f"Error: '{args.input}' not found")
        sys.exit(1)

    if not json_files:
        print(f"No .json files found in '{args.input}'")
        sys.exit(1)

    print(f"\nFound {len(json_files)} JSON file(s) in '{args.input}'")

    all_reports = []
    for jf in json_files:
        report = analyze_file(str(jf), verbose=args.verbose, expected_trials=args.trials)
        if report:
            all_reports.append(report)

    # ── Cross-file summary ──
    if len(all_reports) > 1:
        print(f"\n{'='*70}")
        print(f"  CROSS-FILE SUMMARY")
        print(f"{'='*70}")

        t_all = sum(r["total"] for r in all_reports)
        s_all = sum(r["successes"] for r in all_reports)
        f_all = sum(r["failures"] for r in all_reports)

        print(f"  Files:    {len(all_reports)}")
        print(f"  Total:    {t_all}")
        print(f"  Success:  {s_all} ({100*s_all/t_all:.1f}%)")
        print(f"  Failed:   {f_all} ({100*f_all/t_all:.1f}%)")

        agg = Counter()
        for r in all_reports:
            for et, c in r["error_distribution"].items():
                agg[et] += c

        if agg and f_all > 0:
            print(f"\n  AGGREGATE ERRORS")
            print(f"  {'─'*45}")
            ml = max(len(et) for et in agg)
            for et, cnt in agg.most_common():
                pct = 100 * cnt / f_all
                bar = "█" * int(pct / 2)
                print(f"  {et:<{ml}}  {cnt:>4} ({pct:5.1f}%)  {bar}")

    print()


if __name__ == "__main__":
    main()