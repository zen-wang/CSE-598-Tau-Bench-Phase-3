#!/usr/bin/env python3
"""
τ-bench Folder-Based Error Analysis Script
============================================
Scans a selected subfolder (e.g., 14B, 32B, 4B) under a base directory,
groups JSON result files by domain+method, combines chunked results,
and produces per-group reports plus an ultimate combined report.

Usage:
    python error_analysis_folder.py                        # interactive folder selection
    python error_analysis_folder.py --base /path/to/jsons  # specify base directory
    python error_analysis_folder.py --verbose              # detailed per-task breakdown
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
# 3. FILE LOADING & GROUPING
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


def infer_group_key(filepath):
    """
    Infer (model_version, domain, method) from a file path.
    Searches the full path (all directory components + filename) for clues.
    Examples:
      14B/airline_14b_act.json                        -> (14B, airline, act)
      32B/retail_react_32b/react-qwen32b-..._range... -> (32B, retail, react)
      4B/act-agent4b-0.7_range_0-25_user...           -> (4B, unknown, act)
    """
    fname = os.path.basename(filepath).lower()
    # Use ALL path components for matching (handles nested dirs)
    full_path = filepath.lower()
    parent = os.path.basename(os.path.dirname(filepath)).lower()

    # Determine model version — search entire path for patterns like "14b", "32b", "4b"
    model = "unknown"
    # Check path components (directory names) first for exact folder matches
    parts = Path(filepath).parts
    for part in parts:
        part_lower = part.lower()
        if re.fullmatch(r'\d+b', part_lower):
            model = part.upper()
            break
    # If not found as a directory, try extracting from filename
    if model == "unknown":
        m = re.search(r'(\d+)[bB]', fname)
        if m:
            model = m.group(1) + "B"

    # Determine domain
    domain = "unknown"
    for d in ["airline", "retail"]:
        if d in fname or d in parent:
            domain = d
            break

    # Determine method
    method = "unknown"
    # Check for tool-calling first (before checking "act" which is substring)
    for candidate in ["tool-calling", "tool_calling", "react", "act"]:
        if candidate in fname or candidate in parent:
            method = candidate.replace("_", "-")
            break

    return (model, domain, method)


def collect_json_files(folder_path):
    """Recursively collect all .json files, skipping non-result files."""
    json_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in sorted(files):
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))
    return json_files


def group_files(json_files):
    """Group JSON files by (domain, method), combining chunks."""
    groups = defaultdict(list)
    for fp in json_files:
        key = infer_group_key(fp)
        groups[key].append(fp)
    return dict(groups)


# ============================================================
# 4. ANALYSIS
# ============================================================

def analyze_tasks(tasks, label, source_files=None, verbose=False, expected_trials=5, silent=False, show_affected_tasks=True):
    """Analyze a list of tasks (possibly combined from multiple chunk files).
    If silent=True, skip printing individual report (for --ultimate-only mode).
    Returns a report dict."""
    valid = [t for t in tasks if "traj" in t]

    if not silent:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        if source_files:
            print(f"  Source files:")
            for sf in source_files:
                print(f"    - {sf}")
        print(f"  Loaded: {len(tasks)} trajectories, {len(valid)} with traj data")

    if not valid:
        if not silent:
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
    if not silent:
        print(f"\n  SUMMARY")
        print(f"  {'─'*45}")
        print(f"  Total trajectories:  {total}")
        print(f"  Successes:           {successes} ({100*successes/total:.1f}%)")
        print(f"  Failures:            {failures} ({100*failures/total:.1f}%)")

    # ── Missing trials ──
    total_missing = sum(len(v) for v in missing_trials.values())
    if not silent:
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

    if not silent and failures > 0 and error_counter:
        print(f"\n  ERROR DISTRIBUTION ({failures} failed)")
        print(f"  {'─'*45}")
        ml = max(len(et) for et in error_counter)
        for et, cnt in error_counter.most_common():
            pct = 100 * cnt / failures
            bar = "█" * int(pct / 2)
            print(f"  {et:<{ml}}  {cnt:>4} ({pct:5.1f}%)  {bar}")

        # ── Affected tasks per error type ──
        if show_affected_tasks:
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
    if not silent and verbose:
        print(f"\n  DETAILED FAILURES")
        print(f"  {'─'*45}")
        for r in results:
            if r["status"] != "FAILED":
                continue
            print(f"\n  task_id={r['task_id']}, trial={r['trial']}, reward={r['reward']}")
            for e in r["errors"]:
                print(f"    [{e['type']}] {e['desc']}")

    return {
        "label": label,
        "source_files": source_files or [],
        "total": total,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / total if total > 0 else 0,
        "error_distribution": dict(error_counter.most_common()),
        "error_task_list": {et: lst for et, lst in error_task_list.items()},
    }


def print_ultimate_report(all_reports, show_affected_tasks=True):
    """Print a combined ultimate report across all groups."""
    print(f"\n{'#'*70}")
    print(f"{'#'*70}")
    print(f"  ULTIMATE COMBINED REPORT (All Groups)")
    print(f"{'#'*70}")
    print(f"{'#'*70}")

    t_all = sum(r["total"] for r in all_reports)
    s_all = sum(r["successes"] for r in all_reports)
    f_all = sum(r["failures"] for r in all_reports)

    print(f"\n  SUMMARY")
    print(f"  {'─'*45}")
    print(f"  Groups analyzed:     {len(all_reports)}")
    print(f"  Total trajectories:  {t_all}")
    print(f"  Successes:           {s_all} ({100*s_all/t_all:.1f}%)" if t_all > 0 else "  Successes:           0")
    print(f"  Failures:            {f_all} ({100*f_all/t_all:.1f}%)" if t_all > 0 else "  Failures:            0")

    # ── Per-group breakdown table ──
    print(f"\n  PER-GROUP BREAKDOWN")
    print(f"  {'─'*65}")
    header = f"  {'Group':<35} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Rate':>8}"
    print(header)
    print(f"  {'─'*65}")
    for r in all_reports:
        rate = f"{100*r['success_rate']:.1f}%"
        print(f"  {r['label']:<35} {r['total']:>6} {r['successes']:>6} {r['failures']:>6} {rate:>8}")
        if r.get("source_files"):
            for sf in r["source_files"]:
                print(f"    ↳ {sf}")
    print(f"  {'─'*65}")
    total_rate = f"{100*s_all/t_all:.1f}%" if t_all > 0 else "N/A"
    print(f"  {'TOTAL':<35} {t_all:>6} {s_all:>6} {f_all:>6} {total_rate:>8}")

    # ── Aggregate error distribution ──
    agg_counter = Counter()
    agg_task_list = defaultdict(list)
    for r in all_reports:
        for et, c in r["error_distribution"].items():
            agg_counter[et] += c
        for et, lst in r["error_task_list"].items():
            agg_task_list[et].extend(lst)

    if agg_counter and f_all > 0:
        print(f"\n  AGGREGATE ERROR DISTRIBUTION ({f_all} failed)")
        print(f"  {'─'*45}")
        ml = max(len(et) for et in agg_counter)
        for et, cnt in agg_counter.most_common():
            pct = 100 * cnt / f_all
            bar = "█" * int(pct / 2)
            print(f"  {et:<{ml}}  {cnt:>4} ({pct:5.1f}%)  {bar}")

        # ── Affected tasks per error type ──
        if show_affected_tasks:
            print(f"\n  AFFECTED TASKS PER ERROR TYPE")
            print(f"  {'─'*45}")
            for et, _ in agg_counter.most_common():
                tl = agg_task_list[et]
                by_task = defaultdict(list)
                for tid, tr in tl:
                    by_task[tid].append(tr)

                ids_str = ", ".join(
                    f"{tid}(t{','.join(str(t) for t in sorted(trs))})"
                    for tid, trs in sorted(by_task.items())
                )
                # Truncate if too long
                if len(ids_str) > 300:
                    ids_str = ids_str[:300] + " ..."
                print(f"\n  [{et}] — {len(tl)} occurrences across {len(by_task)} unique tasks")
                print(f"    {ids_str}")

    print()


# ============================================================
# 5. MAIN
# ============================================================

def select_folder(base_path):
    """List subfolders and let user pick one interactively."""
    base = Path(base_path)
    if not base.is_dir():
        print(f"Error: '{base_path}' is not a directory.")
        sys.exit(1)

    # List subfolders (model versions like 14B, 32B, 4B, 8B)
    subfolders = sorted([
        d for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not subfolders:
        print(f"No subfolders found in '{base_path}'.")
        sys.exit(1)

    print(f"\nAvailable model folders in '{base_path}':")
    print(f"{'─'*40}")
    for i, sf in enumerate(subfolders, 1):
        # Count json files recursively
        json_count = sum(1 for _ in sf.rglob("*.json"))
        print(f"  [{i}] {sf.name:<20} ({json_count} JSON files)")
    print(f"  [0] ALL folders")
    print(f"{'─'*40}")

    while True:
        try:
            choice = input("\nSelect folder number (or 0 for all): ").strip()
            choice = int(choice)
            if choice == 0:
                return [str(sf) for sf in subfolders]
            elif 1 <= choice <= len(subfolders):
                return [str(subfolders[choice - 1])]
            else:
                print(f"Please enter 0-{len(subfolders)}")
        except (ValueError, EOFError):
            print(f"Please enter a valid number (0-{len(subfolders)})")


def main():
    parser = argparse.ArgumentParser(description="τ-bench Folder Error Analysis")
    parser.add_argument("--base", "-b", default=".",
                        help="Base directory containing model version folders (default: current dir)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Detailed per-task breakdown")
    parser.add_argument("--trials", "-t", type=int, default=5,
                        help="Expected trials per task (default: 5)")
    parser.add_argument("--ultimate-only", "-u", action="store_true",
                        help="Only print the ultimate combined report (skip individual group reports)")
    parser.add_argument("--no-affected-tasks", "-n", action="store_true",
                        help="Hide the AFFECTED TASKS PER ERROR TYPE section from reports")
    args = parser.parse_args()

    # Select which folder(s) to analyze
    selected_folders = select_folder(args.base)
    ultimate_only = args.ultimate_only

    if ultimate_only:
        print("\nRunning in ultimate-only mode (individual reports suppressed)...")

    all_reports = []

    for folder in selected_folders:
        folder_name = os.path.basename(folder)
        if not ultimate_only:
            print(f"\n{'#'*70}")
            print(f"  ANALYZING: {folder_name}")
            print(f"{'#'*70}")

        # Collect and group JSON files
        json_files = collect_json_files(folder)

        # Filter out non-result files (like PDFs that got .json extension, or report files)
        json_files = [f for f in json_files if not any(
            skip in os.path.basename(f).lower()
            for skip in ["report", "phase", "cse"]
        )]

        if not json_files:
            print(f"  No JSON result files found in '{folder}'.")
            continue

        if not ultimate_only:
            print(f"  Found {len(json_files)} JSON file(s)")

        groups = group_files(json_files)
        if not ultimate_only:
            print(f"  Grouped into {len(groups)} model+domain+method combinations:")
            for (model, domain, method), files in sorted(groups.items()):
                fnames = [os.path.basename(f) for f in files]
                print(f"    • {model}/{domain}/{method}: {len(files)} file(s)")
                for fn in fnames:
                    print(f"        - {fn}")

        # Analyze each group
        for (model, domain, method), files in sorted(groups.items()):
            label = f"{folder_name} | {model} | {domain} | {method}"

            # Track source file names (relative paths from the folder)
            source_files = [os.path.relpath(fp, folder) for fp in files]

            # Combine tasks from all chunk files in this group
            combined_tasks = []
            for fp in files:
                tasks = load_tasks(fp)
                if not ultimate_only:
                    print(f"  Loading {os.path.basename(fp)}: {len(tasks)} tasks")
                combined_tasks.extend(tasks)

            # Deduplicate by (task_id, trial) in case of overlapping chunks
            seen = set()
            deduped = []
            for t in combined_tasks:
                key = (t.get("task_id"), t.get("trial"))
                if key not in seen:
                    seen.add(key)
                    deduped.append(t)
                else:
                    if not ultimate_only:
                        print(f"  ⚠ Duplicate skipped: task_id={key[0]}, trial={key[1]}")

            report = analyze_tasks(
                deduped, label,
                source_files=source_files,
                verbose=args.verbose,
                expected_trials=args.trials,
                silent=ultimate_only,
                show_affected_tasks=not args.no_affected_tasks
            )
            if report:
                all_reports.append(report)
                if ultimate_only:
                    print(f"  ✓ {label} ({report['total']} trajectories, {report['success_rate']*100:.1f}% pass)")

    # ── Ultimate combined report ──
    if all_reports:
        print_ultimate_report(all_reports, show_affected_tasks=not args.no_affected_tasks)
    else:
        print("\nNo valid results found to produce a report.")


if __name__ == "__main__":
    main()