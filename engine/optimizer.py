import json
import os
import threading
import time
from collections import deque

import httpx
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from rl_agent import RiskCalibrationAgent
import alb_controller

BACKENDS = ["backend1", "backend2", "backend3"]

# Real AWS deployment: set BACKEND1_URL/BACKEND2_URL/BACKEND3_URL to each
# EC2 instance's address. Defaults are docker-compose service hostnames,
# for local testing. This is purely where the engine polls telemetry
# from -- it does not touch COSTS, TUNING, or the objective function below.
BACKEND_URLS = {
    "backend1": os.environ.get("BACKEND1_URL", "http://backend1:8000"),
    "backend2": os.environ.get("BACKEND2_URL", "http://backend2:8000"),
    "backend3": os.environ.get("BACKEND3_URL", "http://backend3:8000"),
}

SHARED_DIR = "/shared"
WEIGHTS_PATH = f"{SHARED_DIR}/weights.json"          # active policy — router reads this, unchanged
MODE_PATH = f"{SHARED_DIR}/mode.json"

# Tunable objective weights — exposed live for the demo (see /tune endpoint)
# alpha/beta/gamma trade off latency vs cost vs risk. delta is a smoothing
# term: without it the optimizer is a pure linear program and collapses to
# sending 100% of traffic to a single "best" backend even when all backends
# are near-identical, which looks wrong in a live demo. delta makes the
# problem strictly convex so healthy backends split load proportionally,
# while a clearly degraded backend still gets pushed toward ~0.
TUNING = {"alpha": 1.0, "beta": 0.3, "gamma": 1.5, "delta": 1.0}

# Real, differentiated per-request cost multiplier — stand-in for
# different instance sizes/prices behind each backend. Keep in sync with
# COSTS in router/router.py; previously this was a flat 1.0 for every
# backend, which made the "cost" term in the optimization decorative. With
# real relative costs, beta actually trades performance against cost.
COSTS = {"backend1": 1.0, "backend2": 1.5, "backend3": 2.0}

MODES = ("round_robin", "lor", "optimizer", "default")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "risk_model.pkl")

# --- Online retraining config -------------------------------------------
# The model ships pretrained (on real collected telemetry — see
# risk-model/generate_and_train.py), but it keeps learning: every
# RETRAIN_INTERVAL_S seconds, if enough fresh live samples with both
# classes (healthy / chaos) have accumulated, we retrain in place and hot
# swap the model — no container restart. Ground truth for each sample
# comes from the backend's own /chaos/status, not a guess.
RETRAIN_INTERVAL_S = 60
RETRAIN_MIN_SAMPLES = 40
BUFFER_MAXLEN = 1000

app = FastAPI(title="Optimization Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

risk_model = joblib.load(MODEL_PATH)
model_meta = {
    "version": 1,
    "trained_on": "pretrained (see risk-model/model_meta.json for provenance)",
    "last_retrained_at": None,
    "samples_used": None,
    "source": "startup",
}
_model_lock = threading.Lock()

training_buffer = deque(maxlen=BUFFER_MAXLEN)  # list of (features[3], label)

# RL layer: recalibrates the classifier's risk output against real observed
# outcomes (ground-truth chaos state / measured error rate) via tabular
# Q-learning — see rl_agent.py for the full formulation. This is a
# distinct adaptation mechanism from the supervised online retraining
# below: retraining improves the classifier itself from labeled examples;
# this agent corrects for whatever the classifier still gets systematically
# wrong, using real-time reward feedback instead of labels.
rl_agent = RiskCalibrationAgent()

latest_status = {
    "mode": "optimizer",
    "weights": {},            # what router is currently ACTUALLY using (only meaningful for mode=optimizer -- round_robin/lor are computed live by the router itself, not from this file)
    "optimizer_weights": {},
    "round_robin_weights_estimate": {},
    "lor_weights_estimate": {},
    "metrics": {},
    "risks": {},              # raw classifier output
    "effective_risks": {},    # after RL calibration correction
    "rl": {},
    "chaos_ground_truth": {},
    "comparison": {           # live, continuously-updated ESTIMATE (see /stats on the router for real measured numbers)
        "optimizer_expected_error_rate_avg": None,
        "round_robin_expected_error_rate_avg": None,
        "lor_expected_error_rate_avg": None,
        "ticks": 0,
    },
    "model": model_meta,
    "updated_at": None,
}

_cmp_accum = {"optimizer": 0.0, "round_robin": 0.0, "lor": 0.0, "ticks": 0}


def parse_metrics(text: str) -> dict:
    vals = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, v = line.split()
        vals[k] = float(v)
    return vals


def get_all_metrics_and_truth():
    metrics = {}
    truth = {}
    for b in BACKENDS:
        try:
            r = httpx.get(f"{BACKEND_URLS[b]}/metrics", timeout=2)
            metrics[b] = parse_metrics(r.text)
        except Exception:
            # unreachable backend = treat as maximally risky so it gets zero weight
            metrics[b] = {"backend_p95_latency": 5.0, "backend_error_rate": 1.0, "backend_cpu": 100.0}
        try:
            r = httpx.get(f"{BACKEND_URLS[b]}/chaos/status", timeout=2)
            truth[b] = r.json()
        except Exception:
            truth[b] = {"chaos_active": None, "chaos_type": None, "chaos_severity": None}
    return metrics, truth


def predict_risks(metrics: dict) -> dict:
    risks = {}
    with _model_lock:
        model = risk_model
    for b in BACKENDS:
        feat = [[
            metrics[b]["backend_p95_latency"],
            metrics[b]["backend_error_rate"],
            metrics[b]["backend_cpu"],
        ]]
        risks[b] = float(model.predict_proba(feat)[0][1])
    return risks


def objective(w, latencies, costs, risks):
    w = np.array(w)
    linear_term = (
        TUNING["alpha"] * float(np.dot(w, latencies))
        + TUNING["beta"] * float(np.dot(w, costs))
        + TUNING["gamma"] * float(np.dot(w, risks))
    )
    # quadratic smoothing term: penalizes concentrating all weight on one
    # backend, so near-equal backends split load instead of a winner-take-all
    smoothing_term = TUNING["delta"] * float(np.dot(w, w))
    return linear_term + smoothing_term


def solve_optimizer(metrics: dict, risks: dict) -> dict:
    latencies = np.array([metrics[b]["backend_p95_latency"] for b in BACKENDS])
    costs = np.array([COSTS[b] for b in BACKENDS])
    risk_arr = np.array([risks[b] for b in BACKENDS])

    n = len(BACKENDS)
    w0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons = [{"type": "eq", "fun": lambda w: sum(w) - 1}]

    res = minimize(objective, w0, args=(latencies, costs, risk_arr), bounds=bounds, constraints=cons)
    weights = res.x if res.success else w0
    weights = np.clip(weights, 0, None)
    total = weights.sum()
    weights = weights / total if total > 0 else w0
    return dict(zip(BACKENDS, weights.tolist()))


def estimate_round_robin_weights(metrics: dict) -> dict:
    """Round robin's long-run traffic share is uniform by definition,
    completely independent of backend health -- this IS the exact model,
    not an approximation. Used for the live comparison panel; the real
    measured round-robin performance under actual chaos comes from
    risk-model/collect_round_robin_data.py, not this formula."""
    n = len(BACKENDS)
    return {b: 1.0 / n for b in BACKENDS}


def estimate_lor_weights(metrics: dict) -> dict:
    """Least Outstanding Requests doesn't have a clean closed form from a
    single metrics snapshot -- it depends on real completion-time dynamics
    the router tracks live (see router/router.py's in-flight counters).
    This is a standard queueing-theory approximation for its steady-state
    traffic share: a backend that completes requests faster (lower latency)
    frees its "slot" sooner and so receives a proportionally larger share
    of new arrivals over time, roughly proportional to service rate
    (1 / latency). This is ONLY used for the live estimate panel; the real
    measured LOR performance comes from risk-model/collect_lor_data.py,
    which actually runs the router's real LOR algorithm under real load."""
    inv_latency = np.array([1.0 / max(metrics[b]["backend_p95_latency"], 0.001) for b in BACKENDS])
    weights = inv_latency / inv_latency.sum()
    return dict(zip(BACKENDS, weights.tolist()))


def expected_error_rate(weights: dict, metrics: dict) -> float:
    """What fraction of requests would fail this instant under this
    weight distribution, given the currently-measured error rates.
    This is the live 'counterfactual' comparison metric — computed for
    both policies every cycle from the same measured state, so you can
    see, continuously, what the baseline WOULD be doing right now even
    while the optimizer is the one actually serving traffic (or vice versa)."""
    return sum(weights[b] * metrics[b]["backend_error_rate"] for b in BACKENDS)


def maybe_retrain():
    """Retrain on the accumulated live buffer if we have enough samples
    spanning both classes. Hot-swaps the in-memory model and persists it,
    so the model that started life trained on real collected telemetry
    keeps adapting to real-time traffic without a restart."""
    global risk_model, model_meta
    if len(training_buffer) < RETRAIN_MIN_SAMPLES:
        return
    X = np.array([f for f, _ in training_buffer])
    y = np.array([l for _, l in training_buffer])
    if len(set(y.tolist())) < 2:
        return  # need both healthy and chaos examples to fit a classifier

    try:
        new_model = LogisticRegression()
        new_model.fit(X, y)
    except Exception as e:
        print("online retrain failed:", e, flush=True)
        return

    with _model_lock:
        risk_model = new_model
    try:
        joblib.dump(new_model, MODEL_PATH)
    except Exception as e:
        print("failed to persist retrained model:", e, flush=True)

    model_meta = {
        "version": model_meta["version"] + 1,
        "trained_on": "live telemetry (online retrain)",
        "last_retrained_at": time.time(),
        "samples_used": len(training_buffer),
        "source": "online",
    }
    print(f"[engine] online retrain #{model_meta['version']} on {len(training_buffer)} live samples", flush=True)


def loop():
    last_retrain = time.time()
    while True:
        try:
            metrics, truth = get_all_metrics_and_truth()
            risks = predict_risks(metrics)

            # RL calibration: correct the classifier's raw risk per backend,
            # then immediately give the agent the real outcome as reward so
            # it keeps learning which correction is right for each risk
            # bucket. Ground truth chaos state is preferred when available;
            # otherwise the observed error rate stands in as a softer
            # harm signal.
            effective_risks = {}
            for b in BACKENDS:
                effective_risks[b] = rl_agent.correct(b, risks[b])
                chaos_active = truth[b].get("chaos_active")
                # chaos_active is None when /chaos/status was unreachable this
                # cycle -- falls through to the error-rate signal either way.
                true_harm = 1.0 if chaos_active else metrics[b]["backend_error_rate"]
                rl_agent.update(b, true_harm)

            opt_weights = solve_optimizer(metrics, effective_risks)
            rr_weights_est = estimate_round_robin_weights(metrics)
            lor_weights_est = estimate_lor_weights(metrics)

            _cmp_accum["optimizer"] += expected_error_rate(opt_weights, metrics)
            _cmp_accum["round_robin"] += expected_error_rate(rr_weights_est, metrics)
            _cmp_accum["lor"] += expected_error_rate(lor_weights_est, metrics)
            _cmp_accum["ticks"] += 1

            mode = latest_status.get("mode", "optimizer")
            if mode not in MODES:
                mode = "optimizer"

            # Only "optimizer" mode actually consumes weights.json -- the
            # router implements round_robin and lor itself, live, with real
            # in-flight tracking (see router/router.py). We always publish
            # what the optimizer WOULD do here regardless of active mode,
            # so the dashboard can show it and switching back to optimizer
            # mode takes effect on the very next router request.
            latest_status["optimizer_weights"] = opt_weights
            latest_status["round_robin_weights_estimate"] = rr_weights_est
            latest_status["lor_weights_estimate"] = lor_weights_est
            latest_status["weights"] = opt_weights
            latest_status["metrics"] = metrics
            latest_status["risks"] = risks
            latest_status["effective_risks"] = effective_risks
            latest_status["rl"] = rl_agent.status()
            latest_status["chaos_ground_truth"] = truth
            latest_status["mode"] = mode
            latest_status["comparison"] = {
                "optimizer_expected_error_rate_avg": _cmp_accum["optimizer"] / _cmp_accum["ticks"],
                "round_robin_expected_error_rate_avg": _cmp_accum["round_robin"] / _cmp_accum["ticks"],
                "lor_expected_error_rate_avg": _cmp_accum["lor"] / _cmp_accum["ticks"],
                "ticks": _cmp_accum["ticks"],
            }
            latest_status["model"] = model_meta
            latest_status["updated_at"] = time.time()

            # feed the online-retraining buffer with real live samples;
            # label = ground truth chaos state from the backend itself
            for b in BACKENDS:
                label = truth[b].get("chaos_active")
                if label is None:
                    continue  # backend didn't respond to /chaos/status this tick
                feat = [
                    metrics[b]["backend_p95_latency"],
                    metrics[b]["backend_error_rate"],
                    metrics[b]["backend_cpu"],
                ]
                training_buffer.append((feat, int(label)))

            os.makedirs(SHARED_DIR, exist_ok=True)
            with open(WEIGHTS_PATH, "w") as f:
                json.dump(opt_weights, f)
            with open(MODE_PATH, "w") as f:
                json.dump({"mode": mode}, f)

            # Option 2 (real ALB as the data plane): push the same
            # opt_weights that would otherwise only be consumed by the
            # custom router into the ALB listener's weighted target
            # groups. No-op unless ALB_ENABLED=true -- see
            # alb_controller.py and AWS_DEPLOYMENT.md.
            alb_controller.maybe_push_weights(opt_weights)

            if time.time() - last_retrain > RETRAIN_INTERVAL_S:
                maybe_retrain()
                last_retrain = time.time()

        except Exception as e:
            print("engine loop error:", e, flush=True)
        time.sleep(2)


@app.on_event("startup")
def startup():
    threading.Thread(target=loop, daemon=True).start()


@app.get("/weights")
def get_weights():
    return latest_status["weights"]


@app.get("/status")
def get_status():
    return latest_status


@app.get("/compare")
def compare():
    """Live three-way comparison estimate (optimizer vs. round_robin vs.
    lor), computed every cycle from identical measured conditions. For the
    REAL measured comparison, see the router's /stats and the
    risk-model/*.py evaluation scripts, which actually run each policy."""
    return {
        "comparison": latest_status["comparison"],
        "optimizer_weights": latest_status["optimizer_weights"],
        "round_robin_weights_estimate": latest_status["round_robin_weights_estimate"],
        "lor_weights_estimate": latest_status["lor_weights_estimate"],
        "active_mode": latest_status["mode"],
        "updated_at": latest_status["updated_at"],
    }


@app.post("/mode")
def set_mode(mode: str):
    if mode not in MODES:
        return {"error": f"mode must be one of {MODES}"}
    latest_status["mode"] = mode
    return {"mode": mode}


@app.post("/tune")
def tune(alpha: float = None, beta: float = None, gamma: float = None, delta: float = None):
    """Live-adjust the cost/performance/risk trade-off — good for the demo.
    Lower delta = more concentrated routing (closer to always-pick-the-best).
    Higher delta = more even spreading across healthy backends."""
    if alpha is not None:
        TUNING["alpha"] = alpha
    if beta is not None:
        TUNING["beta"] = beta
    if gamma is not None:
        TUNING["gamma"] = gamma
    if delta is not None:
        TUNING["delta"] = delta
    return TUNING


@app.get("/model")
def model_status():
    return {**model_meta, "buffer_size": len(training_buffer)}


@app.get("/rl")
def rl_status():
    """Q-learning calibration agent status: current epsilon, cumulative
    average reward, and the learned correction table per risk bucket."""
    return rl_agent.status()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/alb")
def alb_status():
    """Status of the ALB weight-push controller (Option 2 deployment
    only) -- whether it's enabled, the ARNs it's configured with, the
    last weights actually pushed to the real ALB, and the last error
    if any push failed."""
    return alb_controller.status()
