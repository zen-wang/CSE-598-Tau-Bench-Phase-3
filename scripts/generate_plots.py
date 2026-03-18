#!/usr/bin/env python3
"""Generate publication-quality plots for Phase 3 report."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "docs", "report", "all_metrics.json")
FIG_DIR = os.path.join(PROJECT_DIR, "docs", "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')

# Colorblind-friendly palette (Wong 2011)
C_BLUE = '#0072B2'
C_ORANGE = '#E69F00'
C_GREEN = '#009E73'
C_RED = '#D55E00'
C_PURPLE = '#CC79A7'
C_CYAN = '#56B4E9'
C_YELLOW = '#F0E442'

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(DATA_PATH) as f:
    data = json.load(f)


def lookup(model, domain, strategy, mode):
    """Return the first matching record or None."""
    for d in data:
        if (d['model'] == model and d['domain'] == domain
                and d['strategy'] == strategy and d['mode'] == mode):
            return d
    return None


def safe_val(record, key):
    """Return numeric value or None."""
    if record is None:
        return None
    v = record.get(key)
    if v is None:
        return None
    return float(v)


# ===================================================================
# Plot 1: Pass^1 Baseline vs Pipeline by Model Size
# ===================================================================
def plot1():
    print("Generating Plot 1: fig3_pass1_baseline_vs_pipeline.png")

    # Define which configs to average per model
    configs = {
        '4B': [('Airline', 'act'), ('Airline', 'react')],
        '14B': [('Airline', 'act'), ('Airline', 'react'), ('Retail', 'tool-calling')],
        '32B': [('Airline', 'act'), ('Airline', 'react'), ('Airline', 'tool-calling'),
                ('Retail', 'act'), ('Retail', 'react'), ('Retail', 'tool-calling')],
    }

    models = ['4B', '14B', '32B']
    baseline_means = []
    pipeline_means = []
    baseline_stds = []
    pipeline_stds = []

    for model in models:
        b_vals = []
        p_vals = []
        for domain, strategy in configs[model]:
            bv = safe_val(lookup(model, domain, strategy, 'baseline'), 'pass_k_1')
            pv = safe_val(lookup(model, domain, strategy, 'pipeline'), 'pass_k_1')
            if bv is not None and pv is not None:
                b_vals.append(bv)
                p_vals.append(pv)
        baseline_means.append(np.mean(b_vals) if b_vals else 0)
        pipeline_means.append(np.mean(p_vals) if p_vals else 0)
        baseline_stds.append(np.std(b_vals, ddof=0) if len(b_vals) > 1 else 0)
        pipeline_stds.append(np.std(p_vals, ddof=0) if len(p_vals) > 1 else 0)

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, baseline_means, width, yerr=baseline_stds,
                   label='Baseline', color=C_BLUE, edgecolor='white',
                   capsize=4, zorder=3)
    bars2 = ax.bar(x + width / 2, pipeline_means, width, yerr=pipeline_stds,
                   label='Pipeline', color=C_ORANGE, edgecolor='white',
                   capsize=4, zorder=3)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f'{h:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f'{h:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('Pass^1', fontsize=12)
    ax.set_title('Pass^1: Baseline vs. Pipeline by Model Size', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, max(max(baseline_means), max(pipeline_means)) * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', visible=True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_pass1_baseline_vs_pipeline.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# Plot 2: Pass^k Reliability Curves
# ===================================================================
def plot2():
    print("Generating Plot 2: fig4_pass_k_reliability.png")

    # Average across airline configs for each model+mode
    ks = [1, 2, 3, 4, 5]
    k_keys = [f'pass_k_{k}' for k in ks]

    series = {
        '32B Pipeline': ('32B', 'pipeline'),
        '32B Baseline': ('32B', 'baseline'),
        '14B Pipeline': ('14B', 'pipeline'),
        '14B Baseline': ('14B', 'baseline'),
    }

    # Airline strategies to average over
    strategies = ['act', 'react', 'tool-calling']

    colors = {'32B': C_BLUE, '14B': C_RED}
    styles = {'pipeline': '-', 'baseline': '--'}
    markers = {'pipeline': 'o', 'baseline': 's'}

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, (model, mode) in series.items():
        vals_per_k = []
        for key in k_keys:
            vals = []
            for strat in strategies:
                rec = lookup(model, 'Airline', strat, mode)
                v = safe_val(rec, key)
                if v is not None:
                    vals.append(v)
            vals_per_k.append(np.mean(vals) if vals else None)

        # Trim trailing Nones
        plot_ks = []
        plot_vals = []
        for k, v in zip(ks, vals_per_k):
            if v is not None:
                plot_ks.append(k)
                plot_vals.append(v)

        if not plot_vals:
            print(f"  WARNING: No data for {label}, skipping line")
            continue

        ax.plot(plot_ks, plot_vals,
                linestyle=styles[mode], marker=markers[mode],
                color=colors[model], linewidth=2.2, markersize=7,
                label=label, zorder=3)

    ax.set_xlabel('k (number of required successes)', fontsize=12)
    ax.set_ylabel('Pass^k', fontsize=12)
    ax.set_title('Reliability Curves: Pass^k Across Trial Requirements', fontsize=14)
    ax.set_xticks(ks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(bottom=0)
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', visible=True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_pass_k_reliability.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# Plot 3: Strategy Comparison on 32B Airline
# ===================================================================
def plot3():
    print("Generating Plot 3: fig5_strategy_comparison.png")

    strategies = ['act', 'react', 'tool-calling']
    strat_labels = ['Act', 'ReAct', 'Tool-Calling']
    b_vals = []
    p_vals = []

    for strat in strategies:
        bv = safe_val(lookup('32B', 'Airline', strat, 'baseline'), 'pass_k_1')
        pv = safe_val(lookup('32B', 'Airline', strat, 'pipeline'), 'pass_k_1')
        b_vals.append(bv if bv is not None else 0)
        p_vals.append(pv if pv is not None else 0)

    x = np.arange(len(strategies))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, b_vals, width,
                   label='Baseline', color=C_BLUE, edgecolor='white', zorder=3)
    bars2 = ax.bar(x + width / 2, p_vals, width,
                   label='Pipeline', color=C_ORANGE, edgecolor='white', zorder=3)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f'{h:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f'{h:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Delta annotations
    for i in range(len(strategies)):
        delta = p_vals[i] - b_vals[i]
        mid_x = x[i]
        top_y = max(b_vals[i], p_vals[i]) + 0.035
        sign = '+' if delta >= 0 else ''
        color = C_GREEN if delta >= 0 else C_RED
        ax.text(mid_x, top_y, f'{sign}{delta:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=color)

    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('Pass^1', fontsize=12)
    ax.set_title('Strategy-Level Improvement: 32B Airline Domain', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, max(max(b_vals), max(p_vals)) * 1.35)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', visible=True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig5_strategy_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# Plot 4: Domain Comparison for 32B
# ===================================================================
def plot4():
    print("Generating Plot 4: fig6_domain_comparison.png")

    strategies = ['act', 'react', 'tool-calling']
    strat_labels = ['Act', 'ReAct', 'Tool-Calling']

    air_b, air_p, ret_b, ret_p = [], [], [], []
    for strat in strategies:
        air_b.append(safe_val(lookup('32B', 'Airline', strat, 'baseline'), 'pass_k_1') or 0)
        air_p.append(safe_val(lookup('32B', 'Airline', strat, 'pipeline'), 'pass_k_1') or 0)
        ret_b.append(safe_val(lookup('32B', 'Retail', strat, 'baseline'), 'pass_k_1') or 0)
        ret_p.append(safe_val(lookup('32B', 'Retail', strat, 'pipeline'), 'pass_k_1') or 0)

    x = np.arange(len(strategies))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_ab = ax.bar(x + offsets[0] * width, air_b, width,
                     label='Airline Baseline', color=C_BLUE, edgecolor='white', zorder=3)
    bars_ap = ax.bar(x + offsets[1] * width, air_p, width,
                     label='Airline Pipeline', color=C_CYAN, edgecolor='white', zorder=3)
    bars_rb = ax.bar(x + offsets[2] * width, ret_b, width,
                     label='Retail Baseline', color=C_RED, edgecolor='white', zorder=3)
    bars_rp = ax.bar(x + offsets[3] * width, ret_p, width,
                     label='Retail Pipeline', color=C_ORANGE, edgecolor='white', zorder=3)

    for bars in [bars_ab, bars_ap, bars_rb, bars_rp]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                        f'{h:.1%}', ha='center', va='bottom', fontsize=8.5,
                        fontweight='bold', rotation=0)

    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('Pass^1', fontsize=12)
    ax.set_title('Domain Comparison: 32B Pipeline Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=12)
    ax.legend(fontsize=10, loc='upper right', ncol=2)
    ax.set_ylim(0, max(max(air_b), max(air_p), max(ret_b), max(ret_p)) * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', visible=True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig6_domain_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# Plot 5: 4B Pipeline Retail Performance
# ===================================================================
def plot5():
    print("Generating Plot 5: fig7_4b_pipeline_retail.png")

    strategies = ['act', 'react', 'tool-calling']
    strat_labels = ['Act', 'ReAct', 'Tool-Calling']
    vals = []

    for strat in strategies:
        v = safe_val(lookup('4B', 'Retail', strat, 'pipeline'), 'pass_k_1')
        if v is None:
            print(f"  WARNING: Missing 4B Retail {strat} pipeline data")
            v = 0
        vals.append(v)

    x = np.arange(len(strategies))
    width = 0.45

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, vals, width, color=[C_BLUE, C_ORANGE, C_GREEN],
                  edgecolor='white', zorder=3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f'{h:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('Pass^1', fontsize=12)
    ax.set_title('4B Pipeline Performance: Retail Domain', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=12)
    ax.set_ylim(0, max(vals) * 1.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', visible=True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig7_4b_pipeline_retail.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    print(f"Loading data from: {DATA_PATH}")
    print(f"Saving figures to: {FIG_DIR}\n")

    plot1()
    plot2()
    plot3()
    plot4()
    plot5()

    print("\nAll plots generated successfully.")
