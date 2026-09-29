"""
Chaos orchestrator — the "real world" failure generator.

Instead of a human curling backend2 every time, this service continuously
and independently decides, for each backend, whether it should be healthy
or degraded right now — picking a random chaos type, random severity, and
random duration, for a random subset of backends (could be one, two, or
all three at once, or none).

This is what makes the demo (and the training data collected from it)
resemble production reality: failures are not scripted to one instance,
they overlap, they vary in kind and severity, and they're unpredictable
in timing — exactly the scenario a single-metric ALB threshold struggles
with and a risk-aware router should visibly handle better.

Every decision is logged to /shared/chaos_log.jsonl as ground truth, so the
engine's baseline-vs-optimizer comparison and the risk-model retraining
loop can both use it as a source of truth for "what actually happened."
"""
import json
import os
import random
import threading
import time

import httpx

BACKENDS = {
    "backend1": os.environ.get("BACKEND1_URL", "http://backend1:8000"),
    "backend2": os.environ.get("BACKEND2_URL", "http://backend2:8000"),
    "backend3": os.environ.get("BACKEND3_URL", "http://backend3:8000"),
}

CHAOS_TYPES = ["latency", "errors", "cpu", "combo"]
SEVERITIES = ["mild", "moderate", "severe"]
SEVERITY_WEIGHTS = [0.5, 0.3, 0.2]

SHARED_LOG = "/shared/chaos_log.jsonl"

# How many backends get hit in a given episode. Weighted so single-backend
# incidents are common (the everyday case) but multi-backend correlated
# failures — the scary real-world case ALB alone can't reason about —
# happen often enough to matter for both the demo and the training data.
N_BACKENDS_WEIGHTS = {1: 0.6, 2: 0.3, 3: 0.1}


def log_event(event: dict):
    event["ts"] = time.time()
    try:
        os.makedirs(os.path.dirname(SHARED_LOG), exist_ok=True)
        with open(SHARED_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print("chaos log write failed:", e, flush=True)
    print("[orchestrator]", event, flush=True)


def set_chaos(name: str, chaos_type: str, severity: str):
    url = BACKENDS[name]
    try:
        params = {} if chaos_type is None else {"type": chaos_type, "severity": severity}
        httpx.post(f"{url}/chaos/set", params=params, timeout=3)
    except Exception as e:
        print(f"failed to set chaos on {name}: {e}", flush=True)


def run_episode():
    n = random.choices(list(N_BACKENDS_WEIGHTS.keys()), weights=list(N_BACKENDS_WEIGHTS.values()))[0]
    targets = random.sample(list(BACKENDS.keys()), k=n)

    assignments = []
    for name in targets:
        chaos_type = random.choice(CHAOS_TYPES)
        severity = random.choices(SEVERITIES, weights=SEVERITY_WEIGHTS)[0]
        set_chaos(name, chaos_type, severity)
        assignments.append({"backend": name, "type": chaos_type, "severity": severity})

    duration = random.uniform(10, 35)
    log_event({"event": "chaos_start", "assignments": assignments, "planned_duration_s": round(duration, 1)})

    time.sleep(duration)

    for name in targets:
        set_chaos(name, None, None)
    log_event({"event": "chaos_end", "backends": targets})


def loop():
    # brief warmup so backends/router/engine are all reachable before we start
    time.sleep(10)
    while True:
        healthy_gap = random.uniform(8, 30)
        time.sleep(healthy_gap)
        try:
            run_episode()
        except Exception as e:
            print("orchestrator episode error:", e, flush=True)


if __name__ == "__main__":
    log_event({"event": "orchestrator_started"})
    loop()
