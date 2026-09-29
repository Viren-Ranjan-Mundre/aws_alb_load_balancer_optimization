import asyncio
import json
import os
import random
import threading
import time
from collections import deque

import httpx
from fastapi import FastAPI

app = FastAPI(title="Intelligent Router")

# Real AWS deployment: set BACKEND1_URL/BACKEND2_URL/BACKEND3_URL to each
# EC2 instance's address, e.g. http://10.0.1.23:8000 (private IP, if
# router and backends share a VPC -- preferred) or a public IP/DNS name.
# Defaults are docker-compose service hostnames, for local testing.
BACKENDS = {
    "backend1": os.environ.get("BACKEND1_URL", "http://backend1:8000"),
    "backend2": os.environ.get("BACKEND2_URL", "http://backend2:8000"),
    "backend3": os.environ.get("BACKEND3_URL", "http://backend3:8000"),
}

# Relative per-request cost multiplier -- stand-in for different instance
# sizes/prices behind each backend (e.g. backend3 could be a larger, pricier
# instance type). Keep this in sync with COSTS in engine/optimizer.py; the
# optimizer's cost objective term uses the same numbers so the "cost" axis
# of the pitch is a real, consistent quantity across both services rather
# than a flat, decorative placeholder.
COSTS = {"backend1": 1.0, "backend2": 1.5, "backend3": 2.0}

SLA_LATENCY_THRESHOLD_S = 0.5  # requests slower than this count as an SLA violation, alongside any error

SHARED_PATH = "/shared/weights.json"
MODE_PATH = "/shared/mode.json"
VALID_MODES = ("round_robin", "lor", "optimizer", "default")

MAX_LATENCY_SAMPLES = 5000  # bounded per-mode rolling window for percentile calculations


def _new_bucket():
    return {
        "requests": 0,
        "errors": 0,
        "sla_violations": 0,
        "cost": 0.0,
        "latencies": deque(maxlen=MAX_LATENCY_SAMPLES),
        "backend_counts": {b: 0 for b in BACKENDS},
        "window_start": time.time(),
    }


# Real, measured stats — split by which policy (mode) actually served each
# request. This is the ground truth used by the evaluation scripts to build
# comparison_report.json: real error rate, real latency percentiles, real
# throughput, real cost, real SLA violations, not a live estimate.
stats = {mode: _new_bucket() for mode in VALID_MODES}

# --- Round robin ------------------------------------------------------------
_backend_order = list(BACKENDS.keys())
_rr_lock = threading.Lock()
_rr_counter = 0


def pick_round_robin() -> str:
    global _rr_counter
    with _rr_lock:
        idx = _rr_counter % len(_backend_order)
        _rr_counter += 1
    return _backend_order[idx]


# --- Least Outstanding Requests ---------------------------------------------
# A real ALB routing algorithm: send each new request to whichever backend
# currently has the fewest in-flight (not-yet-completed) requests. Unlike
# round robin, this naturally favors faster/healthier backends over time,
# since they free up their "in-flight slot" sooner -- without needing any
# risk model or metrics at all. That's exactly why it's a meaningfully
# stronger baseline than plain round robin, and why comparing the optimizer
# against BOTH matters.
_inflight = {b: 0 for b in BACKENDS}
_inflight_lock = asyncio.Lock()


async def acquire_lor() -> str:
    async with _inflight_lock:
        choice = min(_inflight, key=lambda b: _inflight[b])
        _inflight[choice] += 1
    return choice


async def release_lor(backend: str):
    async with _inflight_lock:
        _inflight[backend] = max(0, _inflight[backend] - 1)


# --- Optimizer (ML-driven) ---------------------------------------------------
def load_weights() -> dict:
    """Reads the latest weights written by the optimization engine.
    Falls back to equal split if the file isn't there yet."""
    try:
        if os.path.exists(SHARED_PATH):
            with open(SHARED_PATH) as f:
                w = json.load(f)
                if w and abs(sum(w.values()) - 1.0) < 0.05:
                    return w
    except Exception:
        pass
    n = len(BACKENDS)
    return {b: 1 / n for b in BACKENDS}


def load_mode() -> str:
    try:
        with open(MODE_PATH) as f:
            mode = json.load(f).get("mode", "optimizer")
            return mode if mode in VALID_MODES else "optimizer"
    except Exception:
        return "optimizer"


# --- Shared persistent HTTP client -------------------------------------------
# Deliberately created once at startup and reused for every request, instead
# of opening (and tearing down) a brand-new httpx.AsyncClient — and thus a
# brand-new TCP connection — on every single call. That per-request
# connection setup/teardown was a real, measurable source of latency under
# load; a shared client with keep-alive connections avoids it.
http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup():
    global http_client
    http_client = httpx.AsyncClient(
        timeout=5.0,
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
    )


@app.on_event("shutdown")
async def shutdown():
    if http_client is not None:
        await http_client.aclose()


def _percentile(sorted_values, p):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def record_stat(mode: str, backend: str, status: int, elapsed_s: float):
    bucket = stats.get(mode, stats["optimizer"])
    bucket["requests"] += 1
    bucket["latencies"].append(elapsed_s)
    bucket["backend_counts"][backend] = bucket["backend_counts"].get(backend, 0) + 1
    bucket["cost"] += COSTS.get(backend, 1.0)
    is_error = status >= 500 or status == 599
    if is_error:
        bucket["errors"] += 1
    if is_error or elapsed_s > SLA_LATENCY_THRESHOLD_S:
        bucket["sla_violations"] += 1


@app.get("/route")
async def route():
    mode = load_mode()

    if mode == "round_robin":
        choice = pick_round_robin()
    elif mode == "lor":
        choice = await acquire_lor()
    elif mode == "default":
        # Static equal weights (1/3 per backend), no ML, no RL, no live
        # metrics involved at all -- deliberately NOT the same thing as
        # round_robin (which cycles deterministically) or the optimizer's
        # weights.json (which the risk model + solver produce). This is
        # "what if you just hardcoded even weights and never touched it
        # again" -- the third real-world baseline requested for the
        # 6-run benchmark protocol.
        n = len(BACKENDS)
        weights = {b: 1.0 / n for b in BACKENDS}
        choice = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
    else:
        weights = load_weights()
        choice = random.choices(list(weights.keys()), weights=list(weights.values()))[0]

    start = time.monotonic()
    try:
        r = await http_client.get(f"{BACKENDS[choice]}/work")
        status = r.status_code
    except Exception:
        status = 599  # backend unreachable
    elapsed = time.monotonic() - start

    if mode == "lor":
        await release_lor(choice)

    record_stat(mode, choice, status, elapsed)
    return {"routed_to": choice, "status": status, "mode": mode}


@app.get("/stats")
def get_stats():
    """Full multi-metric measured performance per routing policy: error
    rate, latency (avg/p50/p95/p99), throughput, SLA violations, real cost,
    and which backends actually served the traffic. Reset with
    POST /stats/reset before a clean measurement window."""
    out = {}
    for mode, s in stats.items():
        n = s["requests"]
        lat_sorted = sorted(s["latencies"])
        elapsed_window = max(time.time() - s["window_start"], 1e-6)
        out[mode] = {
            "requests": n,
            "errors": s["errors"],
            "error_rate": (s["errors"] / n) if n else None,
            "avg_latency_s": (sum(s["latencies"]) / len(s["latencies"])) if s["latencies"] else None,
            "p50_latency_s": _percentile(lat_sorted, 0.50),
            "p95_latency_s": _percentile(lat_sorted, 0.95),
            "p99_latency_s": _percentile(lat_sorted, 0.99),
            "throughput_rps": round(n / elapsed_window, 3) if n else 0.0,
            "sla_violations": s["sla_violations"],
            "sla_violation_rate": (s["sla_violations"] / n) if n else None,
            "total_cost": round(s["cost"], 4),
            "cost_per_request": (s["cost"] / n) if n else None,
            "backend_distribution": dict(s["backend_counts"]),
        }
    return out


@app.post("/stats/reset")
def reset_stats():
    now = time.time()
    for s in stats.values():
        s["requests"] = 0
        s["errors"] = 0
        s["sla_violations"] = 0
        s["cost"] = 0.0
        s["latencies"].clear()
        for b in s["backend_counts"]:
            s["backend_counts"][b] = 0
        s["window_start"] = now
    return {"status": "reset"}


@app.get("/health")
def health():
    return {"status": "healthy"}
