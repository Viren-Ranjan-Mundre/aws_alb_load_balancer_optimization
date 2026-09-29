# Intelligent Traffic Router

*First Commit Hackathon — Multi-Objective Traffic Optimization (local/cloud-agnostic build)*

A router that jointly optimizes latency, cost, and predicted failure risk in
real time — instead of routing by a fixed policy and reacting only after a
health check fails. Built to run fully locally now; deploys onto AWS
(ALB + CloudWatch + ASG) once credits land, with no change to the decision
logic.

## What's in this version

1. **Two real AWS ALB routing algorithms as baselines** — Round Robin and
   Least Outstanding Requests (LOR), both implemented as genuine
   request-time routing logic in the router (not a synthetic strawman
   threshold policy).
2. **Real latency fixes** — two actual architectural bugs were found and
   fixed: the router was opening a brand-new HTTP connection on every
   single request instead of reusing one, and the backends used a
   blocking sleep that queued behind FastAPI's thread pool under
   concurrent load. Both are fixed; see "Why latency was inflated" below.
3. **A pluggable, duration-controlled load generator** — every
   collection/evaluation script can drive traffic with a lightweight
   built-in generator (default) or real headless Locust
   (`--load-generator locust --users N --spawn-rate R --seconds S`).
4. **Full multi-metric comparison** — error rate, latency (avg/p50/p95/p99),
   throughput, SLA violation rate, and real per-backend cost — not just
   error rate and average latency.
5. **A proper three-phase evaluation pipeline** — collect real data under
   Round Robin, then under LOR, train the risk model on the combination
   of both, then evaluate the RL-assisted optimizer under matched
   conditions and get a full comparison against both baselines.
6. **Two distinct online adaptation mechanisms**: supervised retraining of
   the classifier on live labeled telemetry, and a Q-learning
   reinforcement-learning agent that recalibrates the classifier's risk
   output against real observed outcomes.

## What's inside

| Folder | What it is |
|---|---|
| `backends/` | 3 fake backend instances (FastAPI, async) with multi-type, multi-severity chaos (`/chaos/set`), an optional self-triggered auto-chaos mode, and Prometheus-style `/metrics` |
| `chaos/` | The chaos orchestrator — randomly degrades any combination of backends, any type/severity, on random timers |
| `router/` | Implements all three routing algorithms directly (round robin, least outstanding requests, and ML-optimizer weighted routing), with a persistent shared HTTP client and full multi-metric stats per policy (`/stats`) |
| `engine/` | Polls telemetry, scores failure risk, applies the RL calibration correction, solves the constrained optimization problem, computes live round-robin/LOR estimates for the dashboard, retrains the risk model online |
| `risk-model/` | The three-phase evaluation pipeline — see below |
| `dashboard/` | Single HTML file — live view of weights, chaos status, risk, RL calibration status, three-way live comparison, and model retrain status |
| `loadtest/` | The actual Locust script used by `--load-generator locust` |
| `prometheus/` | Scrape config, for a telemetry dashboard alongside the engine |

## Requirements

- Docker + Docker Compose
- Python 3.10+ on your host (for the evaluation pipeline; also needed if you use `--load-generator locust`)
- **Windows PowerShell users:** `curl` is aliased to `Invoke-WebRequest`, which
  doesn't support `-X`. Use `curl.exe -X POST ...` or PowerShell's own syntax,
  `Invoke-WebRequest -Uri <url> -Method POST`.

## Why latency was inflated, and what was fixed

Two real bugs, not tuning knobs:

1. **Router was opening a new HTTP connection per request.** The original
   code did `async with httpx.AsyncClient() as client:` *inside* the
   request handler — creating and tearing down a fresh TCP connection on
   every single call. It now creates one `httpx.AsyncClient` at startup
   with keep-alive connections and reuses it for every request.
2. **Backends used a blocking `time.sleep()` inside a synchronous
   endpoint.** FastAPI runs `def` (non-`async`) endpoints in a
   limited-size thread pool, so under concurrent load, blocking sleeps
   queue up behind each other and inflate measured latency far beyond the
   intended chaos magnitude. `/work` is now `async def` with
   `await asyncio.sleep(...)`, which yields the event loop instead of
   blocking a thread, so latency reflects the actual injected delay, not
   thread-pool contention.

Base intrinsic latency was also trimmed slightly (50ms → 20ms) since it
was arbitrary padding on top of the real fix.

## Run it — step by step

### 1. Build and start the stack

```bash
cd risk-model
pip install -r requirements.txt
cd ..
docker compose up --build
```

Starts, on your machine:
- `backend1`, `backend2`, `backend3` → ports 8001, 8002, 8003
- `chaos-orchestrator` → background, randomly degrading backends
- `prometheus` → port 9090
- `engine` (optimizer + RL agent) → port 8010
- `router` (round robin / LOR / optimizer) → port 9000

If you see repeated `engine loop error: ...` in the engine's logs, it's
almost always a scikit-learn version mismatch between whatever trained
`risk_model.pkl` and what's pinned in `engine/requirements.txt` — retrain
(step 3) using the exact pinned version and
`docker compose up --build --force-recreate`.

### 2. PHASE 1a — collect data under Round Robin

```bash
cd risk-model
python collect_round_robin_data.py
```

Puts the router in `round_robin` mode, drives real load through it
(internal generator by default), injects randomized chaos across all
backends, and saves `data_round_robin.csv` (training data) +
`benchmark_round_robin.json` (real measured performance under Round
Robin — every metric, not just error rate).

To control how long it runs and use real Locust instead:
```bash
python collect_round_robin_data.py --seconds 300 --load-generator locust --users 100 --spawn-rate 20
```
`--seconds` controls duration either way; for Locust it's passed straight
through as `--run-time`.

### 3. PHASE 1b — collect data under Least Outstanding Requests

```bash
python collect_lor_data.py
# same flags available: --seconds, --load-generator, --users, --spawn-rate
```

Puts the router in `lor` mode (each request goes to whichever backend
currently has the fewest in-flight requests — a real ALB algorithm,
naturally favoring faster/healthier backends without any risk model at
all). Saves `data_lor.csv` + `benchmark_lor.json`.

Use the same `--seconds` (and `--users`/`--spawn-rate` if using Locust) as
step 2, so the two baselines are measured under comparable conditions.

### 4. Train the risk model on both

```bash
python generate_and_train.py
```

Combines `data_round_robin.csv` + `data_lor.csv` — the model sees backend
behavior under two different real traffic-distribution patterns, not just
one. Prints train + holdout accuracy/AUC, writes `risk_model.pkl` into
both `risk-model/` and `engine/`, and saves `model_meta.json` with
provenance (`"source": "real_live_collected"`, `"data_files_used": [...]`).

### 5. Load the retrained model

```bash
cd ..
docker compose up --build --force-recreate
```

### 6. PHASE 2 — evaluate the RL-assisted optimizer

```bash
cd risk-model
python evaluate_optimizer.py --seconds 300 --load-generator locust --users 100 --spawn-rate 20
```

Use the same `--seconds`/`--users`/`--spawn-rate` you used for the two
baselines. Puts the router in `optimizer` mode (risk-aware, RL-calibrated,
multi-objective) and saves `benchmark_optimizer.json`.

### 7. PHASE 3 — the full comparison

```bash
python compare_all.py
```

Reads all three `benchmark_*.json` files and prints a full side-by-side
table — error rate, avg/p50/p95/p99 latency, throughput, SLA violation
rate, cost per request, total cost, and which backends actually served
the traffic under each policy — plus percentage improvement of the
optimizer over each baseline. Saves `comparison_report.json`.

```
========================================================================
Metric                       Round Robin             LOR       Optimizer
========================================================================
Error rate                        0.0800          0.0357          0.0093
Avg latency (s)                   0.3100          0.1800          0.1000
p95 latency (s)                   1.1000          0.6000          0.2500
...

Optimizer improvement vs. Round Robin:
  Error rate: +88.4%
  Avg latency (s): +67.7%
  ...
```

**This is your submission evidence** — real requests, real randomized
chaos, real algorithms (not strawmen), real numbers.

### 8. Open the dashboard

```bash
cd ..
start dashboard/index.html      # Windows
open dashboard/index.html       # macOS
xdg-open dashboard/index.html   # Linux
```

Live view: per-backend weight/risk/chaos status, a three-way live
estimate (Round Robin / LOR / Optimizer expected error rate), mode
buttons to switch policy on the fly, real measured stats per mode
(all the metrics above), the RL calibration agent's learned corrections,
and the supervised model's retrain status.

### 9. Demo controls

Chaos happens on its own via the orchestrator, but you can trigger it
manually for a guaranteed demo moment:
```bash
curl.exe -X POST "http://localhost:8002/chaos/set?type=latency&severity=severe"
curl.exe -X POST "http://localhost:8002/chaos/set"     # clear it
```

Live-adjust the cost/performance/risk trade-off:
```bash
curl.exe -X POST "http://localhost:8010/tune?alpha=0.5&beta=1.5&gamma=1.0"
```

Switch policy manually at any time:
```bash
curl.exe -X POST "http://localhost:8010/mode?mode=round_robin"
curl.exe -X POST "http://localhost:8010/mode?mode=lor"
curl.exe -X POST "http://localhost:8010/mode?mode=optimizer"
```

### 10. Stop everything

```bash
docker compose down -v
```

## Useful URLs while running

- Router: `http://localhost:9000/route` · real multi-metric stats: `http://localhost:9000/stats`
- Engine status (weights, risks, effective risks, chaos ground truth, comparison, model): `http://localhost:8010/status`
- Live 3-way comparison estimate: `http://localhost:8010/compare`
- Risk model status: `http://localhost:8010/model`
- RL calibration agent status: `http://localhost:8010/rl`
- Prometheus: `http://localhost:9090`
- Backend chaos control: `http://localhost:8001/chaos/set`, `:8002`, `:8003`
- Backend chaos ground truth: `http://localhost:8001/chaos/status`, `:8002`, `:8003`
- Locust UI (if you launch it manually instead of via `--load-generator locust`): `http://localhost:8089`

## The three routing algorithms, briefly

- **Round Robin** (`router.py: pick_round_robin`): cycles through backends
  in fixed order. Completely ignores health, latency, and risk. This is a
  real ALB routing option — the honest floor to beat.
- **Least Outstanding Requests** (`router.py: acquire_lor`/`release_lor`):
  tracks real in-flight request counts per backend live, and always sends
  the next request to whichever has the fewest outstanding. Naturally
  favors faster backends over time without any risk-awareness — a
  meaningfully stronger baseline than Round Robin, and why the optimizer
  needs to beat both, not just one strawman.
- **Optimizer** (`engine/optimizer.py` + `router.py`'s weighted random
  choice): the actual system this project is about — jointly optimizes
  latency, cost, and RL-calibrated failure risk, proactively shifting
  traffic before a hard threshold is ever crossed.

The engine's `/compare` endpoint also computes a **live estimate** of all
three for the dashboard (Round Robin's is exact — its share is uniform by
definition; LOR's is a standard queueing-theory approximation based on
inverse latency, since true LOR depends on live completion-timing dynamics
the estimate can't see from a single metrics snapshot). The **real**
measured numbers always come from actually running each policy via the
phase scripts above.

## The two adaptation mechanisms, and why both

- **Supervised online retraining** (`engine/optimizer.py`, `maybe_retrain`):
  every 60s, if there are ≥40 fresh live samples spanning both classes,
  the classifier retrains from scratch on the accumulated buffer and hot
  swaps in — improves the classifier's *predictions* from labeled ground
  truth.
- **RL calibration agent** (`engine/rl_agent.py`): a tabular Q-learning
  contextual bandit that learns a correction to apply on top of whatever
  the classifier currently outputs, using real observed outcomes as
  reward (reward = negative distance between the *corrected* risk and
  what actually happened). Fixes systematic bias the classifier hasn't
  learned to correct yet, without waiting for the next retrain cycle.

## Full 6-phase benchmark protocol (round_robin / lor / default / 3 optimizer iterations)

For a rigorous, publication-ready evaluation — not just "the optimizer
looked better in one run" — `risk-model/run_aws_benchmarks.py` runs all six
phases below automatically, under identical load and randomized chaos
conditions, and `plot_benchmarks.py` turns the results into 4 charts.

1. **Round Robin** baseline — `data_round_robin.csv`, `benchmark_round_robin.json`
2. **Least Outstanding Requests** baseline — `data_lor.csv`, `benchmark_lor.json`
3. **Default** baseline — static equal weights (1/3 per backend), no ML, no
   RL, fixed for the whole run — `data_default.csv`, `benchmark_default.json`
4. **Optimizer Iteration 1** — trained on baseline data (RR+LOR+Default) →
   `data_opt1.csv`, `results_opt1.json`
5. **Optimizer Iteration 2** — trained on baseline + Iteration 1's telemetry
   (cumulative) → `data_opt2.csv`, `results_opt2.json`
6. **Optimizer Iteration 3** — trained on baseline + Iterations 1+2
   (cumulative) → `data_opt3.csv`, `results_opt3.json`

The RL calibration agent (`engine/rl_agent.py`) stays active throughout all
three optimizer iterations. **None of this touches the objective function,
trade-off parameters, or SciPy solver in `engine/optimizer.py`** — the
active-learning loop works entirely by feeding the classifier more real
data between iterations, not by changing how routing decisions are made
from a given risk estimate.

```bash
cd risk-model
python run_aws_benchmarks.py                                  # ~18-30 min, 3-minute phases by default
python run_aws_benchmarks.py --load-generator locust --users 100 --spawn-rate 20
python compare_all.py          # detailed round_robin/lor/optimizer comparison
python plot_benchmarks.py      # saves chart1-4 PNGs into plots/
```

Training on a specific combination of collected datasets directly (used
internally by the iteration protocol above, but callable on its own too):

```bash
python generate_and_train.py --datasets baseline              # round_robin + lor + default
python generate_and_train.py --datasets baseline opt1
python generate_and_train.py --datasets baseline opt1 opt2
```

See `AWS_DEPLOYMENT.md` for running this whole protocol on an EC2 instance.

## Model versioning and choice of model type

`generate_and_train.py` supports two classifiers via `--model-type
{logistic, xgboost}` (default `logistic`) — both expose the same
`.predict_proba()` interface the engine calls, so switching types needs
no engine code changes.

Every training run is saved as a new numbered version under
`risk-model/models/` (e.g. `risk_model_v3_xgboost.pkl`) and **nothing is
ever overwritten or deleted**. `models/registry.json` is the full
lineage: what data trained each version, what version (if any) it
continued from, and its holdout metrics. `risk_model.pkl` at the top
level (and the copy in `engine/`) is just whichever version is newest —
that's what the engine actually loads — but the full history stays on
disk.

Retraining is **continual by default, not from scratch**: it loads the
latest existing version of the same `--model-type` and continues from
it — `warm_start=True` for logistic regression (reuses its previous
`coef_`/`intercept_` as the optimizer's starting point) and
`xgb_model=<previous booster>` for XGBoost (adds new trees on top of the
existing ensemble rather than growing a fresh one). Pass `--fresh` to
explicitly discard history and start over.

```bash
python generate_and_train.py --datasets baseline                          # v1, from scratch
python generate_and_train.py --datasets baseline opt1                     # v2, continues from v1
python generate_and_train.py --datasets baseline opt1 --model-type xgboost # v1 of a separate xgboost lineage
python generate_and_train.py --datasets baseline opt1 --fresh             # new version, but ignores history
```

## Real AWS deployment (3 backend EC2 instances + router/engine instance)

For running on actual AWS infrastructure rather than docker-compose on a
single host: `docker-compose.backend.yml` deploys one backend per EC2
instance (3 total), and `docker-compose.router.yml` deploys router +
engine + chaos-orchestrator + Prometheus together on one more instance —
router is the real, internet-facing entry point in this deployment (it's
playing the role ALB would play). Every service already resolves backend
addresses from `BACKEND1_URL`/`BACKEND2_URL`/`BACKEND3_URL` environment
variables (defaulting to docker-compose hostnames for local testing), so
no code changes are needed to point at real EC2 IPs — only a `.env` file
per instance. See `AWS_DEPLOYMENT.md` for the full step-by-step.

Router and engine stay co-located on one instance deliberately: they
communicate via a local shared file (`weights.json`/`mode.json`), which
only works on a shared filesystem. The RL agent needs no deployment
changes at all — it already runs its full predict-then-correct-then-learn
cycle every engine tick, in-process, regardless of where the backends
physically live; that's the "live data as an RL loop for live-time
prediction and learning" already built in, not something new to wire up
for a real deployment.

`engine/optimizer.py`'s objective function, trade-off parameters, and
SciPy solver setup are unaffected by any of this — only how services find
each other over the network changed.
