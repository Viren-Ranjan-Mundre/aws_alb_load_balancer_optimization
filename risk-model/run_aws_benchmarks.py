"""
Master runner for the full 6-phase evaluation protocol:

  Baselines (identical load/chaos conditions each time):
    1. Round Robin            -> data_round_robin.csv, benchmark_round_robin.json
    2. Least Outstanding Req. -> data_lor.csv,          benchmark_lor.json
    3. Default (static 1/3)   -> data_default.csv,      benchmark_default.json

  Optimizer active-learning iterations (RL calibration agent stays active
  throughout all three -- see engine/rl_agent.py):
    4. Iteration 1: train on baseline data (RR+LOR+Default)
                    -> data_opt1.csv, results_opt1.json
    5. Iteration 2: train on baseline + opt1 (cumulative)
                    -> data_opt2.csv, results_opt2.json
    6. Iteration 3: train on baseline + opt1 + opt2 (cumulative)
                    -> data_opt3.csv, results_opt3.json

This does NOT touch engine/optimizer.py's objective function, trade-off
parameters, or SciPy solver -- it only drives training (generate_and_train.py
--datasets), data collection (loadgen.py), and container lifecycle
(docker compose, to load each freshly retrained model). Between each
optimizer iteration, `docker compose up -d --build --force-recreate engine`
is what actually loads the newly trained risk_model.pkl -- this is the
same step evaluate_optimizer.py's own docstring already documents as
required, just automated here instead of manual.

Usage:
    python run_aws_benchmarks.py
    python run_aws_benchmarks.py --baseline-seconds 180 --opt-seconds 180
    python run_aws_benchmarks.py --load-generator locust --users 100 --spawn-rate 20
    python run_aws_benchmarks.py --skip-baselines   # reuse existing data_round_robin.csv etc.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import requests

import loadgen

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
# Real AWS deployment: set these to your EC2 addresses. Defaults assume
# docker-compose port-mapping on localhost, for local testing.
ENGINE_URL = os.environ.get("ENGINE_URL", "http://localhost:8010")
ROUTER_URL = os.environ.get("ROUTER_URL", "http://localhost:9000")
BACKEND_URLS = {
    "backend1": os.environ.get("BACKEND1_URL", "http://localhost:8001"),
    "backend2": os.environ.get("BACKEND2_URL", "http://localhost:8002"),
    "backend3": os.environ.get("BACKEND3_URL", "http://localhost:8003"),
}

WEIGHT_TIMELINE_ROUND_ROBIN = os.path.join(HERE, "weight_timeline_round_robin.jsonl")
WEIGHT_TIMELINE_OPT3 = os.path.join(HERE, "weight_timeline_opt3.jsonl")


# --------------------------------------------------------------------------
# Stack lifecycle
# --------------------------------------------------------------------------

def check_stack_reachable():
    for name, url in [("router", ROUTER_URL), ("engine", ENGINE_URL), *BACKEND_URLS.items()]:
        try:
            requests.get(f"{url}/health", timeout=3)
        except Exception:
            print(f"ERROR: can't reach {name} at {url}.")
            print("Is `docker compose up --build` running (in this folder's parent, "
                  "or already started separately)?")
            sys.exit(1)
    print("Stack reachable: router, engine, and all 3 backends are up.\n")


def recreate_engine():
    """Reloads the just-trained risk_model.pkl into the running engine by
    recreating that one container -- the documented way this project
    already picks up a retrained model (see evaluate_optimizer.py's
    docstring). Router/backends are untouched and keep running."""
    print("Recreating engine container to load the newly trained model...")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "--build", "--force-recreate", "engine"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print("ERROR: failed to recreate the engine container. Is Docker running "
              "and is this being run where docker-compose.yml's parent directory "
              "is reachable?")
        sys.exit(1)
    # give the engine a few seconds to actually come back up and do its
    # first telemetry poll before anything starts routing against it
    for _ in range(30):
        try:
            requests.get(f"{ENGINE_URL}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    else:
        print("WARNING: engine didn't come back healthy within 30s after recreate.")
    time.sleep(3)
    print("Engine recreated and healthy.\n")


# --------------------------------------------------------------------------
# Weight-timeline logging (for plot_benchmarks.py's Chart 4)
# --------------------------------------------------------------------------

def log_weight_timeline(path: str, duration_s: int, poll_interval_s: float = 2.0):
    """Polls the engine's /status every poll_interval_s and appends
    {timestamp, optimizer_weights, round_robin_weights_estimate} to a
    JSONL file. Run concurrently with a phase's load/chaos so
    plot_benchmarks.py can draw a real time series of how weights moved
    during actual chaos events, not just an aggregate end-of-run number."""
    import threading

    stop = threading.Event()

    def _loop():
        with open(path, "w") as f:
            start = time.time()
            while not stop.is_set() and (time.time() - start) < duration_s:
                try:
                    status = requests.get(f"{ENGINE_URL}/status", timeout=3).json()
                    f.write(json.dumps({
                        "t": time.time() - start,
                        "optimizer_weights": status.get("optimizer_weights"),
                        "round_robin_weights_estimate": status.get("round_robin_weights_estimate"),
                        "chaos_ground_truth": status.get("chaos_ground_truth"),
                    }) + "\n")
                    f.flush()
                except Exception:
                    pass
                time.sleep(poll_interval_s)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return stop, t


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------

def phase_baselines(args):
    print("=" * 80)
    print("PHASE 1: BASELINES (round_robin, lor, default)")
    print("=" * 80)

    have_all = all(
        os.path.exists(os.path.join(HERE, f))
        for f in ("data_round_robin.csv", "data_lor.csv", "data_default.csv")
    )
    if args.skip_baselines and have_all:
        print("--skip-baselines set and all three baseline CSVs already exist -- reusing them.\n")
        return

    # Round robin, with weight-timeline logging running alongside it (its
    # own "timeline" is a flat 1/3 by definition, but logging it for real
    # rather than assuming keeps Chart 4 honest about what was measured).
    stop, t = log_weight_timeline(WEIGHT_TIMELINE_ROUND_ROBIN, args.baseline_seconds)
    loadgen.run_collection_phase(
        mode="round_robin", seconds=args.baseline_seconds, generator=args.load_generator,
        users=args.users, spawn_rate=args.spawn_rate,
        training_csv=os.path.join(HERE, "data_round_robin.csv"),
        benchmark_json=os.path.join(HERE, "benchmark_round_robin.json"),
    )
    stop.set()
    t.join(timeout=5)

    loadgen.run_collection_phase(
        mode="lor", seconds=args.baseline_seconds, generator=args.load_generator,
        users=args.users, spawn_rate=args.spawn_rate,
        training_csv=os.path.join(HERE, "data_lor.csv"),
        benchmark_json=os.path.join(HERE, "benchmark_lor.json"),
    )

    loadgen.run_collection_phase(
        mode="default", seconds=args.baseline_seconds, generator=args.load_generator,
        users=args.users, spawn_rate=args.spawn_rate,
        training_csv=os.path.join(HERE, "data_default.csv"),
        benchmark_json=os.path.join(HERE, "benchmark_default.json"),
    )
    print()


def train(datasets, label):
    print(f"--- Training ({label}): datasets={datasets} ---")
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "generate_and_train.py"), "--datasets", *datasets],
        cwd=HERE, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print(f"Training failed for {label} -- aborting benchmark run.")
        sys.exit(1)
    print()


def phase_optimizer_iteration(iteration: int, cumulative_datasets: list, seconds: int,
                               generator: str, users: int, spawn_rate: int, log_timeline: bool):
    print("=" * 80)
    print(f"PHASE: OPTIMIZER ACTIVE-LEARNING ITERATION {iteration}")
    print("=" * 80)

    train(cumulative_datasets, f"iteration {iteration}")
    recreate_engine()

    data_csv = os.path.join(HERE, f"data_opt{iteration}.csv")
    results_json = os.path.join(HERE, f"results_opt{iteration}.json")

    timeline_stop = timeline_thread = None
    if log_timeline:
        timeline_stop, timeline_thread = log_weight_timeline(WEIGHT_TIMELINE_OPT3, seconds)

    benchmark = loadgen.run_collection_phase(
        mode="optimizer", seconds=seconds, generator=generator,
        users=users, spawn_rate=spawn_rate,
        training_csv=data_csv, benchmark_json=results_json,
    )

    if timeline_stop:
        timeline_stop.set()
        timeline_thread.join(timeout=5)

    print(f"Iteration {iteration} complete. Measured: {benchmark.get('measured_stats', {})}\n")
    return benchmark


# --------------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------------

def print_final_summary():
    print("=" * 80)
    print("ALL 6 PHASES COMPLETE")
    print("=" * 80)
    files = [
        "benchmark_round_robin.json", "benchmark_lor.json", "benchmark_default.json",
        "results_opt1.json", "results_opt2.json", "results_opt3.json",
    ]
    header = f"{'phase':<14}{'requests':>10}{'err rate':>10}{'p95':>9}{'p99':>9}{'rps':>8}{'cost/req':>10}"
    print(header)
    for fname in files:
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print(f"{fname:<14}{'missing':>10}")
            continue
        with open(path) as f:
            data = json.load(f)
        s = data.get("measured_stats", {})
        label = fname.replace("benchmark_", "").replace("results_", "").replace(".json", "")

        def fp(x, suffix=""):
            return f"{x*100:.2f}%" if suffix == "%" and x is not None else (f"{x:.3f}{suffix}" if x is not None else "—")

        print(f"{label:<14}{s.get('requests', 0):>10}"
              f"{fp(s.get('error_rate'), '%'):>10}"
              f"{fp(s.get('p95_latency_s'), 's'):>9}"
              f"{fp(s.get('p99_latency_s'), 's'):>9}"
              f"{(s.get('throughput_rps') or 0):>8.1f}"
              f"{fp(s.get('cost_per_request')):>10}")
    print("=" * 80)
    print("\nRun: python compare_all.py   for the detailed round_robin/lor/optimizer comparison")
    print("Run: python plot_benchmarks.py   to generate the 4 benchmark charts into plots/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-seconds", type=int, default=180,
                         help="Duration of each of the 3 baseline phases")
    parser.add_argument("--opt-seconds", type=int, default=180,
                         help="Duration of each of the 3 optimizer iteration phases "
                              "(the prompt's '3-minute load test' -- 180s -- is the default)")
    parser.add_argument("--load-generator", choices=["internal", "locust"], default="internal")
    parser.add_argument("--users", type=int, default=50, help="Locust virtual users (ignored for internal)")
    parser.add_argument("--spawn-rate", type=int, default=10, help="Locust spawn rate (ignored for internal)")
    parser.add_argument("--skip-baselines", action="store_true",
                         help="Reuse existing data_round_robin.csv / data_lor.csv / data_default.csv "
                              "if all three are already present, instead of re-collecting them.")
    args = parser.parse_args()

    check_stack_reachable()
    os.chdir(HERE)  # generate_and_train.py, loadgen.py etc. all resolve paths relative to here

    phase_baselines(args)

    phase_optimizer_iteration(1, ["baseline"], args.opt_seconds,
                               args.load_generator, args.users, args.spawn_rate, log_timeline=False)
    phase_optimizer_iteration(2, ["baseline", "opt1"], args.opt_seconds,
                               args.load_generator, args.users, args.spawn_rate, log_timeline=False)
    phase_optimizer_iteration(3, ["baseline", "opt1", "opt2"], args.opt_seconds,
                               args.load_generator, args.users, args.spawn_rate, log_timeline=True)

    print_final_summary()


if __name__ == "__main__":
    main()
