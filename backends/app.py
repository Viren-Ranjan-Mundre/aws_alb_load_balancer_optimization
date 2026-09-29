import asyncio
import os
import random
import threading
import time

from fastapi import FastAPI, Response

app = FastAPI(title="Backend Instance")

# ---------------------------------------------------------------------------
# Chaos model
# ---------------------------------------------------------------------------
# Real production failures aren't a single on/off switch on one instance —
# they're one of several distinct failure modes, of varying severity, that
# can hit any instance at any time. We model that here instead of a single
# boolean:
#
#   latency  -> instance is slow (e.g. GC pause, noisy neighbor, disk I/O)
#   errors   -> instance throws 5xx (e.g. bad deploy, dependency outage)
#   cpu      -> instance is CPU-saturated (e.g. runaway thread, hot loop)
#   combo    -> more than one of the above at once (the realistic worst case)
#
# and severity: "mild" | "moderate" | "severe" controls how bad it gets.
#
# state["chaos_type"] is None when healthy.
BACKEND_ID = os.environ.get("BACKEND_ID", "unknown")
AUTO_CHAOS = os.environ.get("AUTO_CHAOS", "false").lower() == "true"

SEVERITY_PROFILES = {
    "mild":     {"latency_add": (0.10, 0.25), "error_p": 0.05},
    "moderate": {"latency_add": (0.25, 0.55), "error_p": 0.20},
    "severe":   {"latency_add": (0.55, 1.20), "error_p": 0.45},
}
CHAOS_TYPES = ["latency", "errors", "cpu", "combo"]
SEVERITIES = ["mild", "moderate", "severe"]

state = {
    "requests": 0,
    "errors": 0,
    "latencies": [],
    "chaos_type": None,      # None | "latency" | "errors" | "cpu" | "combo"
    "chaos_severity": None,  # None | "mild" | "moderate" | "severe"
    "chaos_started_at": None,
}
_lock = threading.Lock()


def _set_chaos(chaos_type, severity):
    with _lock:
        if chaos_type not in (None, *CHAOS_TYPES):
            raise ValueError(f"unknown chaos type: {chaos_type}")
        if severity not in (None, *SEVERITIES):
            raise ValueError(f"unknown severity: {severity}")
        state["chaos_type"] = chaos_type
        state["chaos_severity"] = severity
        state["chaos_started_at"] = time.time() if chaos_type else None


@app.get("/work")
async def work():
    """Simulates handling a real request. Degrades according to whatever
    chaos type/severity is currently active on this instance.

    Deliberately async + asyncio.sleep (not a blocking time.sleep in a sync
    def): FastAPI runs sync endpoints in a limited-size thread pool, so under
    concurrent load a blocking sleep queues requests behind each other and
    inflates measured latency far beyond the chaos magnitude itself. An
    async endpoint yields the event loop instead, so latency reflects the
    actual injected delay, not thread-pool contention."""
    state["requests"] += 1
    base_latency = 0.02

    chaos_type = state["chaos_type"]
    severity = state["chaos_severity"] or "moderate"
    profile = SEVERITY_PROFILES[severity]

    is_error = False
    if chaos_type in ("latency", "combo"):
        base_latency += random.uniform(*profile["latency_add"])
    if chaos_type in ("cpu", "combo"):
        # CPU pressure shows up as latency jitter too, not just the /metrics gauge
        base_latency += random.uniform(0.05, 0.20)
    if chaos_type in ("errors", "combo") and random.random() < profile["error_p"]:
        is_error = True

    await asyncio.sleep(base_latency)

    if is_error:
        state["errors"] += 1
        return Response(status_code=500, content="simulated failure")

    state["latencies"].append(base_latency)
    state["latencies"] = state["latencies"][-200:]  # rolling window
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# NOTE: FastAPI matches routes in registration order, and "/chaos/{mode}"
# below is a catch-all path parameter — so the specific "/chaos/set" and
# "/chaos/status" routes MUST be declared before it, or they'd never be
# reached (a request to /chaos/set would match {mode}="set" first).
@app.post("/chaos/set")
def chaos_set(type: str = None, severity: str = "moderate"):
    """Fine-grained control used by the chaos orchestrator:
    POST /chaos/set?type=latency&severity=mild
    POST /chaos/set              -> clears chaos (type omitted / None)
    """
    _set_chaos(type, severity if type else None)
    return {"chaos": state["chaos_type"] is not None, "type": state["chaos_type"], "severity": state["chaos_severity"]}


@app.get("/chaos/status")
def chaos_status():
    """Ground truth for whoever's collecting training data or scoring the
    baseline-vs-optimizer comparison — this is what actually happened,
    independent of what any model predicted."""
    return {
        "backend_id": BACKEND_ID,
        "chaos_active": state["chaos_type"] is not None,
        "chaos_type": state["chaos_type"],
        "chaos_severity": state["chaos_severity"],
        "chaos_started_at": state["chaos_started_at"],
    }


@app.post("/chaos/{mode}")
def chaos_legacy(mode: str):
    """Backwards-compatible manual toggle used by the original demo:
    POST /chaos/on  -> severe combo failure
    POST /chaos/off -> healthy
    Kept working as-is so existing curl commands / muscle memory still work."""
    if mode == "on":
        _set_chaos("combo", "severe")
    elif mode == "off":
        _set_chaos(None, None)
    else:
        return Response(status_code=400, content="mode must be 'on' or 'off'")
    return {"chaos": state["chaos_type"] is not None, "type": state["chaos_type"], "severity": state["chaos_severity"]}


@app.get("/metrics")
def metrics():
    """Prometheus-format metrics, also consumed directly by the optimization engine."""
    lat = state["latencies"] or [0.05]
    sorted_lat = sorted(lat)
    p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
    p95 = sorted_lat[p95_idx]
    error_rate = state["errors"] / max(state["requests"], 1)

    chaos_type = state["chaos_type"]
    cpu_bump = 0.0
    if chaos_type in ("cpu", "combo"):
        bump_map = {"mild": 25, "moderate": 45, "severe": 70}
        cpu_bump = bump_map.get(state["chaos_severity"] or "moderate", 45)
    cpu = 20 + cpu_bump + random.uniform(0, 10)

    body = (
        "# TYPE backend_p95_latency gauge\n"
        f"backend_p95_latency {p95}\n"
        "# TYPE backend_error_rate gauge\n"
        f"backend_error_rate {error_rate}\n"
        "# TYPE backend_cpu gauge\n"
        f"backend_cpu {cpu}\n"
        "# TYPE backend_requests_total counter\n"
        f"backend_requests_total {state['requests']}\n"
        "# TYPE backend_chaos_active gauge\n"
        f"backend_chaos_active {1 if chaos_type else 0}\n"
    )
    return Response(content=body, media_type="text/plain")


# ---------------------------------------------------------------------------
# Optional autonomous chaos: each backend can independently decide, on its
# own random schedule, to enter/exit a degradation episode. Enable per
# instance with AUTO_CHAOS=true (see docker-compose.yml). This is what makes
# failures feel like the real world — unpredictable, not scripted by a human
# hitting curl on backend2 every time.
def _auto_chaos_loop():
    while True:
        # healthy interval before the next random episode
        time.sleep(random.uniform(15, 45))
        chaos_type = random.choice(CHAOS_TYPES)
        severity = random.choices(SEVERITIES, weights=[0.5, 0.3, 0.2])[0]
        _set_chaos(chaos_type, severity)
        print(f"[{BACKEND_ID}] auto-chaos: {chaos_type} ({severity})", flush=True)

        duration = random.uniform(10, 40)
        time.sleep(duration)
        _set_chaos(None, None)
        print(f"[{BACKEND_ID}] auto-chaos: recovered", flush=True)


@app.on_event("startup")
def startup():
    if AUTO_CHAOS:
        threading.Thread(target=_auto_chaos_loop, daemon=True).start()
