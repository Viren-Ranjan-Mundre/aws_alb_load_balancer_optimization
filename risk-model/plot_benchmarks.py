"""
Generates 4 publication-quality charts from the 6-phase benchmark output
(run_aws_benchmarks.py) into plots/:

  Chart 1: Bar chart -- mean error rate across all 6 runs
  Chart 2: Grouped bar chart -- latency percentiles (p50/p95/p99) across all 6 runs
  Chart 3: Dual-axis bar chart -- throughput (req/s) vs avg cost per request ($/req)
  Chart 4: Multi-panel time series -- per-backend weight shifts during chaos,
           Optimizer Iteration 3 vs Round Robin

Run this AFTER run_aws_benchmarks.py has produced its benchmark_*.json /
results_opt*.json / weight_timeline_*.jsonl files.

Usage:
    python plot_benchmarks.py
    python plot_benchmarks.py --outdir plots
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless -- no display needed, just save PNGs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
sns.set_theme(style="whitegrid", context="talk")

RUNS = [
    ("round_robin", "Round Robin", "benchmark_round_robin.json"),
    ("lor", "LOR", "benchmark_lor.json"),
    ("default", "Default", "benchmark_default.json"),
    ("opt1", "Optimizer\nIter 1", "results_opt1.json"),
    ("opt2", "Optimizer\nIter 2", "results_opt2.json"),
    ("opt3", "Optimizer\nIter 3", "results_opt3.json"),
]

PALETTE = {
    "round_robin": "#8c8c8c",
    "lor": "#5b9bd5",
    "default": "#bfa14a",
    "opt1": "#a1d99b",
    "opt2": "#41ab5d",
    "opt3": "#00441b",
}


def load_runs():
    """Returns {key: measured_stats dict} for whichever run files exist,
    printing a clear warning for any that are missing rather than
    crashing -- a partial benchmark (e.g. baselines done, iterations not
    yet run) should still produce what charts it can."""
    data = {}
    for key, label, fname in RUNS:
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print(f"WARNING: {fname} not found -- '{label}' will be skipped in the charts. "
                  f"Run run_aws_benchmarks.py first (or the missing phase specifically).")
            continue
        with open(path) as f:
            payload = json.load(f)
        data[key] = payload.get("measured_stats", {})
    if not data:
        print("ERROR: no benchmark files found at all. Run run_aws_benchmarks.py first.")
        raise SystemExit(1)
    return data


def present_runs(data):
    """RUNS filtered down to only the ones we actually have data for,
    preserving the canonical left-to-right order."""
    return [(k, label) for k, label, _ in RUNS if k in data]


def chart_error_rate(data, outdir):
    runs = present_runs(data)
    labels = [l for _, l in runs]
    values = [((data[k].get("error_rate") or 0) * 100) for k, _ in runs]
    colors = [PALETTE[k] for k, _ in runs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Mean error rate (%)")
    ax.set_title("Error Rate Across All 6 Runs\n(same real load + randomized chaos conditions)")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}%",
                ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    path = os.path.join(outdir, "chart1_error_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def chart_latency_percentiles(data, outdir):
    runs = present_runs(data)
    labels = [l for _, l in runs]
    metrics = [("p50_latency_s", "p50"), ("p95_latency_s", "p95"), ("p99_latency_s", "p99")]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, (key, mlabel) in enumerate(metrics):
        values = [((data[k].get(key) or 0) * 1000) for k, _ in runs]  # ms
        ax.bar(x + (i - 1) * width, values, width, label=mlabel)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Percentiles Across All 6 Runs")
    ax.legend(title="Percentile")
    fig.tight_layout()
    path = os.path.join(outdir, "chart2_latency_percentiles.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def chart_throughput_cost(data, outdir):
    runs = present_runs(data)
    labels = [l for _, l in runs]
    throughput = [(data[k].get("throughput_rps") or 0) for k, _ in runs]
    cost = [(data[k].get("cost_per_request") or 0) for k, _ in runs]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(12, 6.5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, throughput, width, color="#4e79a7", label="Throughput (req/s)")
    bars2 = ax2.bar(x + width / 2, cost, width, color="#e15759", label="Avg cost / request ($)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Throughput (req/s)", color="#4e79a7")
    ax2.set_ylabel("Avg cost per request ($, relative units)", color="#e15759")
    ax1.tick_params(axis="y", labelcolor="#4e79a7")
    ax2.tick_params(axis="y", labelcolor="#e15759")
    ax1.set_title("Throughput vs. Cost per Request Across All 6 Runs", pad=40)

    lines = [bars1, bars2]
    labels_ = [l.get_label() for l in lines]
    ax1.legend(lines, labels_, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    fig.tight_layout()
    path = os.path.join(outdir, "chart3_throughput_vs_cost.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _load_timeline(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def chart_weight_shifts(outdir):
    rr_path = os.path.join(HERE, "weight_timeline_round_robin.jsonl")
    opt3_path = os.path.join(HERE, "weight_timeline_opt3.jsonl")
    rr_rows = _load_timeline(rr_path)
    opt3_rows = _load_timeline(opt3_path)

    if not rr_rows and not opt3_rows:
        print("WARNING: no weight_timeline_*.jsonl files found -- skipping Chart 4. "
              "These are written by run_aws_benchmarks.py during the round_robin "
              "baseline and the Optimizer Iteration 3 phases.")
        return

    backends = ["backend1", "backend2", "backend3"]
    colors = {"backend1": "#4e79a7", "backend2": "#e15759", "backend3": "#59a14f"}

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=False)

    for ax, rows, title, weight_key in [
        (axes[0], rr_rows, "Round Robin — traffic share over time (flat by definition)", "round_robin_weights_estimate"),
        (axes[1], opt3_rows, "Optimizer Iteration 3 — live weight shifts during chaos", "optimizer_weights"),
    ]:
        if not rows:
            ax.text(0.5, 0.5, "No data — run run_aws_benchmarks.py first", ha="center", va="center")
            ax.set_title(title)
            continue
        t = [r["t"] for r in rows]
        for b in backends:
            y = [((r.get(weight_key) or {}).get(b)) for r in rows]
            ax.plot(t, y, label=b, color=colors[b], linewidth=2)

        # shade regions where ground truth says ANY backend was in chaos,
        # so weight shifts can be visually correlated with real chaos events
        chaos_active_any = []
        for r in rows:
            truth = r.get("chaos_ground_truth") or {}
            chaos_active_any.append(any(v.get("chaos_active") for v in truth.values() if isinstance(v, dict)))
        in_span = False
        span_start = None
        for i, active in enumerate(chaos_active_any):
            if active and not in_span:
                span_start = t[i]
                in_span = True
            elif not active and in_span:
                ax.axvspan(span_start, t[i], color="red", alpha=0.08)
                in_span = False
        if in_span:
            ax.axvspan(span_start, t[-1], color="red", alpha=0.08)

        ax.set_title(title)
        ax.set_xlabel("Seconds into run")
        ax.set_ylabel("Traffic weight")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", ncol=3)

    fig.suptitle("Backend Weight Shifts During Chaos: Round Robin vs. Optimizer Iteration 3\n"
                  "(shaded red = at least one backend in a real chaos episode)", y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, "chart4_weight_shifts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="plots")
    args = parser.parse_args()

    outdir = os.path.join(HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    data = load_runs()
    chart_error_rate(data, outdir)
    chart_latency_percentiles(data, outdir)
    chart_throughput_cost(data, outdir)
    chart_weight_shifts(outdir)

    print(f"\nAll available charts saved to {outdir}/")


if __name__ == "__main__":
    main()
