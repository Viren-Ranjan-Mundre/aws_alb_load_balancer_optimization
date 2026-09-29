"""
PHASE 3 (final step) of the evaluation pipeline: pull together all three
real measured benchmarks and produce one full comparison report covering
every metric the router tracks -- error rate, latency percentiles,
throughput, SLA violations, and real cost -- not just error rate and
average latency.

Run this AFTER collect_round_robin_data.py, collect_lor_data.py, and
evaluate_optimizer.py have all produced their benchmark_*.json files.

Usage:
    python compare_all.py
"""
import json
import os

METRICS = [
    # (key, label, direction) -- direction is "lower" or "higher" for which
    # way counts as an improvement; "info" metrics are shown but not scored
    ("error_rate", "Error rate", "lower"),
    ("avg_latency_s", "Avg latency (s)", "lower"),
    ("p50_latency_s", "p50 latency (s)", "lower"),
    ("p95_latency_s", "p95 latency (s)", "lower"),
    ("p99_latency_s", "p99 latency (s)", "lower"),
    ("throughput_rps", "Throughput (req/s)", "higher"),
    ("sla_violation_rate", "SLA violation rate", "lower"),
    ("cost_per_request", "Cost per request", "lower"),
    ("requests", "Total requests", "info"),
    ("total_cost", "Total cost", "info"),
]

BENCHMARK_FILES = {
    "round_robin": "benchmark_round_robin.json",
    "lor": "benchmark_lor.json",
    "optimizer": "benchmark_optimizer.json",
}


def pct_improvement(baseline_value, optimizer_value, direction):
    if direction == "info" or baseline_value in (None, 0) or optimizer_value is None:
        return None
    if direction == "lower":
        return round((baseline_value - optimizer_value) / baseline_value * 100, 1)
    return round((optimizer_value - baseline_value) / baseline_value * 100, 1)


def load_benchmarks():
    benchmarks = {}
    missing = []
    for mode, filename in BENCHMARK_FILES.items():
        if os.path.exists(filename):
            with open(filename) as f:
                benchmarks[mode] = json.load(f).get("measured_stats", {})
        else:
            missing.append(filename)
    return benchmarks, missing


def main():
    benchmarks, missing = load_benchmarks()
    if missing:
        print("Missing benchmark file(s), run the corresponding script(s) first:")
        for m in missing:
            print(f"  - {m}")
        if "optimizer" not in benchmarks and ("round_robin" not in benchmarks or "lor" not in benchmarks):
            return
        print("\nContinuing with what's available...\n")

    report = {"benchmarks": benchmarks, "vs_round_robin": {}, "vs_lor": {}}

    print("=" * 72)
    print(f"{'Metric':<24}{'Round Robin':>16}{'LOR':>16}{'Optimizer':>16}")
    print("=" * 72)
    for key, label, direction in METRICS:
        rr = benchmarks.get("round_robin", {}).get(key)
        lor = benchmarks.get("lor", {}).get(key)
        opt = benchmarks.get("optimizer", {}).get(key)

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v) if v is not None else "—"

        print(f"{label:<24}{fmt(rr):>16}{fmt(lor):>16}{fmt(opt):>16}")

        if direction != "info":
            report["vs_round_robin"][key] = pct_improvement(rr, opt, direction)
            report["vs_lor"][key] = pct_improvement(lor, opt, direction)
    print("=" * 72)

    print("\nBackend distribution (which backends actually served traffic):")
    for mode, mode_label in [("round_robin", "Round Robin"), ("lor", "LOR"), ("optimizer", "Optimizer")]:
        dist = benchmarks.get(mode, {}).get("backend_distribution")
        if dist:
            total = sum(dist.values()) or 1
            pretty = ", ".join(f"{b}: {c} ({c/total*100:.0f}%)" for b, c in dist.items())
            print(f"  {mode_label:<14}{pretty}")
    report["backend_distribution"] = {
        mode: benchmarks.get(mode, {}).get("backend_distribution")
        for mode in ("round_robin", "lor", "optimizer")
    }

    print("\nOptimizer improvement vs. Round Robin:")
    for key, label, direction in METRICS:
        if direction == "info":
            continue
        v = report["vs_round_robin"].get(key)
        if v is not None:
            print(f"  {label}: {'+' if v >= 0 else ''}{v}%")

    print("\nOptimizer improvement vs. LOR:")
    for key, label, direction in METRICS:
        if direction == "info":
            continue
        v = report["vs_lor"].get(key)
        if v is not None:
            print(f"  {label}: {'+' if v >= 0 else ''}{v}%")

    with open("comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved full comparison to comparison_report.json")


if __name__ == "__main__":
    main()
