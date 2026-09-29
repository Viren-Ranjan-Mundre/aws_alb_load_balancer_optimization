"""
Shared building blocks for the evaluation pipeline
(collect_round_robin_data.py, collect_lor_data.py, evaluate_optimizer.py).

Keeping these in one place means every phase of the pipeline injects chaos
the same way and controls the engine/router the same way, so the resulting
comparison is apples-to-apples.

Two load generators are supported, chosen with --load-generator:

  internal (default) -- a lightweight background thread hitting
    GET /route directly in a loop. No extra dependencies, works everywhere,
    good for quick iteration.

  locust -- runs your actual loadtest/locustfile.py headless, for exactly
    --seconds seconds, with --users virtual users ramping at --spawn-rate
    per second. This is the real load-testing tool doing the driving,
    with its own request distribution/think-time behavior, and its own
    CSV stats written alongside ours for cross-checking.

Either way, this module also runs the chaos injection concurrently and
exposes the same mode/stats control helpers, so switching load generators
doesn't change anything else about how a phase runs.
"""
import os
import random
import subprocess
import sys
import threading
import time

import requests

# Real AWS deployment: this script talks to the backends' /metrics and
# /chaos endpoints directly (for training-data collection and chaos
# ground truth) and to the router/engine for routing and mode control.
# Set BACKEND1_URL/BACKEND2_URL/BACKEND3_URL, ROUTER_URL, ENGINE_URL to
# your real EC2 addresses -- defaults assume docker-compose port-mapping
# on localhost, for local testing.
BACKEND_URLS = {
    "backend1": os.environ.get("BACKEND1_URL", "http://localhost:8001"),
    "backend2": os.environ.get("BACKEND2_URL", "http://localhost:8002"),
    "backend3": os.environ.get("BACKEND3_URL", "http://localhost:8003"),
}
ROUTER_URL = os.environ.get("ROUTER_URL", "http://localhost:9000")
ENGINE_URL = os.environ.get("ENGINE_URL", "http://localhost:8010")

CHAOS_TYPES = ["latency", "errors", "cpu", "combo"]
SEVERITIES = ["mild", "moderate", "severe"]

LOCUSTFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loadtest", "locustfile.py")

_stop_load = threading.Event()
_stop_chaos = threading.Event()


def parse_metrics(text: str) -> dict:
    vals = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, v = line.split()
        vals[k] = float(v)
    return vals


def set_mode(mode: str):
    try:
        requests.post(f"{ENGINE_URL}/mode", params={"mode": mode}, timeout=5)
        print(f"Engine mode set to '{mode}'.")
    except Exception as e:
        print(f"WARNING: could not reach engine at {ENGINE_URL} to set mode '{mode}': {e}")
        print("Continuing anyway -- make sure `docker compose up` is running.")


def reset_stats():
    try:
        requests.post(f"{ROUTER_URL}/stats/reset", timeout=5)
    except Exception as e:
        print(f"WARNING: could not reset router stats: {e}")


def get_stats() -> dict:
    return requests.get(f"{ROUTER_URL}/stats", timeout=5).json()


def clear_all_chaos():
    for url in BACKEND_URLS.values():
        try:
            requests.post(f"{url}/chaos/set", timeout=3)
        except Exception:
            pass


# --- Chaos injection ---------------------------------------------------------
def chaos_loop():
    """Same randomized multi-backend, multi-type, multi-severity chaos
    distribution as chaos/orchestrator.py, run directly against the
    backends so this works even without that container running."""
    while not _stop_chaos.is_set():
        time.sleep(random.uniform(6, 15))
        if _stop_chaos.is_set():
            break
        n = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        targets = random.sample(list(BACKEND_URLS.keys()), k=n)
        for name in targets:
            chaos_type = random.choice(CHAOS_TYPES)
            severity = random.choices(SEVERITIES, weights=[0.5, 0.3, 0.2])[0]
            try:
                requests.post(
                    f"{BACKEND_URLS[name]}/chaos/set",
                    params={"type": chaos_type, "severity": severity},
                    timeout=3,
                )
            except Exception:
                pass
        duration = random.uniform(6, 20)
        time.sleep(duration)
        for name in targets:
            try:
                requests.post(f"{BACKEND_URLS[name]}/chaos/set", timeout=3)
            except Exception:
                pass


def start_chaos() -> threading.Thread:
    _stop_chaos.clear()
    t = threading.Thread(target=chaos_loop, daemon=True)
    t.start()
    return t


def stop_chaos(thread: threading.Thread):
    _stop_chaos.set()
    thread.join(timeout=5)
    clear_all_chaos()


# --- Load generation ----------------------------------------------------------
def _internal_load_loop():
    while not _stop_load.is_set():
        try:
            requests.get(f"{ROUTER_URL}/route", timeout=5)
        except Exception:
            pass
        time.sleep(random.uniform(0.02, 0.08))


class LoadHandle:
    """Uniform handle for either load generator so calling code doesn't
    need to know which one is running underneath."""

    def __init__(self, kind, thread=None, proc=None):
        self.kind = kind
        self.thread = thread
        self.proc = proc

    def stop(self):
        """Call once your own sampling loop has run for the intended
        duration. For 'internal', signals the thread and joins. For
        'locust', Locust should already be finishing on its own (it was
        given --run-time == the same duration) -- this just waits briefly
        and force-terminates if it hasn't exited yet."""
        if self.kind == "internal":
            _stop_load.set()
            if self.thread:
                self.thread.join(timeout=5)
        else:
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Locust hadn't exited on its own past its --run-time window -- terminating it.")
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except Exception:
                    self.proc.kill()


def start_load(generator: str, seconds: int, users: int, spawn_rate: int, csv_prefix: str) -> "LoadHandle":
    """generator: 'internal' or 'locust'. Both are duration-controlled by
    `seconds` -- for locust this is passed straight through as --run-time,
    so you get the same knob regardless of which generator you pick."""
    if generator == "locust":
        cmd = [
            sys.executable, "-m", "locust",
            "-f", LOCUSTFILE,
            "--host", ROUTER_URL,
            "--headless",
            "-u", str(users),
            "-r", str(spawn_rate),
            "--run-time", f"{seconds}s",
            "--csv", csv_prefix,
            "--only-summary",
        ]
        print("Starting Locust:", " ".join(cmd))
        proc = subprocess.Popen(cmd)
        return LoadHandle("locust", proc=proc)

    _stop_load.clear()
    t = threading.Thread(target=_internal_load_loop, daemon=True)
    t.start()
    print(f"Started internal load generator (no Locust) for {seconds}s.")
    return LoadHandle("internal", thread=t)


def run_collection_phase(mode: str, seconds: int, generator: str, users: int, spawn_rate: int,
                          training_csv: str, benchmark_json: str):
    """Runs one full data-collection phase for a given router mode: sets
    the mode, resets stats, drives real load + randomized chaos for
    `seconds`, samples real backend telemetry for training data, and saves
    a real measured performance benchmark. Shared by
    collect_round_robin_data.py and collect_lor_data.py so both baselines
    are collected identically apart from which mode is active."""
    import csv
    import json

    set_mode(mode)
    reset_stats()

    print(f"Collecting for {seconds}s in {mode.upper()} mode "
          f"(load generator: {generator}), with randomized chaos across all backends...")

    load_handle = start_load(generator, seconds, users, spawn_rate, csv_prefix=mode)
    chaos_thread = start_chaos()

    rows = []
    start = time.time()
    while time.time() - start < seconds:
        for name, url in BACKEND_URLS.items():
            try:
                m = parse_metrics(requests.get(f"{url}/metrics", timeout=2).text)
                truth = requests.get(f"{url}/chaos/status", timeout=2).json()
            except Exception as e:
                print("skip:", e)
                continue
            label = 1 if truth.get("chaos_active") else 0
            rows.append([
                m.get("backend_p95_latency", 0),
                m.get("backend_error_rate", 0),
                m.get("backend_cpu", 0),
                label,
            ])
        time.sleep(1)

    load_handle.stop()
    stop_chaos(chaos_thread)

    with open(training_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["latency", "error_rate", "cpu", "label"])
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r[-1] == 1)
    print(f"Wrote {len(rows)} rows to {training_csv} ({n_pos} chaos-labeled, {len(rows) - n_pos} healthy).")

    try:
        measured = get_stats().get(mode, {})
    except Exception as e:
        print(f"WARNING: could not fetch real router stats: {e}")
        measured = {}

    benchmark = {
        "collected_at": time.time(),
        "duration_s": seconds,
        "mode": mode,
        "load_generator": generator,
        "measured_stats": measured,
    }
    with open(benchmark_json, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"\nSaved {benchmark_json}:")
    print(json.dumps(benchmark, indent=2))
    return benchmark
