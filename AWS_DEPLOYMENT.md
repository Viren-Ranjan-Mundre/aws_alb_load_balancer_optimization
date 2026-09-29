# Running on Real AWS Infrastructure

This deploys the project across **4 real EC2 instances**: one per backend
(3 total) plus one instance running the router — which is the real,
internet-facing entry point in this deployment, i.e. it plays the role
ALB would play — together with the optimization engine.

**Why router + engine share one instance, but backends don't:** router and
engine talk to each other only through a local shared file
(`/shared/weights.json`, `/shared/mode.json`), which requires the same
filesystem. Splitting them across instances would mean redesigning that
into a network call (e.g. router polling engine's `/weights` over HTTP) --
a real code change, out of scope here. The 3 backends have no such
constraint, which is exactly why they're the piece that scales out to
separate machines: it matches what "3 backends, 3 EC2 instances" is
actually testing -- real network latency and real machine-to-machine
failure between the router/optimizer and the compute it's routing to,
which a single-host docker-compose setup can't represent.

```
                    ┌─────────────────────────────┐
  Internet ────────▶│  Router EC2 instance         │
                    │  (router = the ALB, here)    │
                    │  - router  :9000              │
                    │  - engine  :8010               │─┐
                    │  - chaos-orchestrator          │ │ HTTP, real network
                    │  - prometheus :9090             │ │ (not docker network)
                    └─────────────────────────────┘  │
                                  │                    │
              ┌───────────────────┼────────────────────┤
              ▼                   ▼                    ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │ backend1 EC2    │  │ backend2 EC2    │  │ backend3 EC2    │
     │ :8000           │  │ :8000           │  │ :8000           │
     └────────────────┘  └────────────────┘  └────────────────┘
```

## 1. Launch 4 EC2 instances

- **Backend instances (x3):** `t3.small` is plenty -- each runs one tiny
  FastAPI service. Ubuntu 22.04 LTS.
- **Router instance (x1):** `t3.large` -- runs router + engine + chaos
  orchestrator + Prometheus together. Ubuntu 22.04 LTS.
- **VPC:** put all 4 in the **same VPC and subnet** so they can reach each
  other over private IPs (faster, no data-transfer cost, and you don't
  need to open backend ports to the whole internet).
- **Security groups:**
  - Backend instances: allow inbound TCP `8000` from the router instance's
    security group specifically (not `0.0.0.0/0`) -- that's the router,
    engine, and chaos-orchestrator all reaching `/work`, `/metrics`,
    `/health`, `/chaos/*`.
  - Router instance: allow inbound TCP `9000` (router, public-facing),
    `8010` (engine, for the dashboard/benchmark scripts), `9090`
    (Prometheus, optional), from your IP. Allow `22` (SSH) from your IP on
    all 4 instances.

Note each backend instance's **private IP** (e.g. `10.0.1.11`,
`10.0.1.12`, `10.0.1.13`) and the router instance's private and public IP
-- you'll need them below.

## 2. Install Docker on all 4 instances

Run this on each of the 4 instances:

```bash
ssh -i your-key.pem ubuntu@<INSTANCE_IP>

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg git
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

## 3. Get the code onto all 4 instances

```bash
# from your local machine
scp -i your-key.pem -r ./traffic-router ubuntu@<INSTANCE_IP>:~/traffic-router
# repeat for all 4 instances, or push to a git remote and `git clone` on each
```

## 4. Deploy each backend instance

**On backend1's instance:**

```bash
cd ~/traffic-router
echo "BACKEND_ID=backend1" > .env
docker compose -f docker-compose.backend.yml up -d --build
curl http://localhost:8000/health
```

**On backend2's instance:** same, but `.env` has `BACKEND_ID=backend2`.
**On backend3's instance:** same, but `.env` has `BACKEND_ID=backend3`.

## 5. Configure and deploy the router instance

**On the router instance**, point it at the 3 backends' private IPs:

```bash
cd ~/traffic-router
cat > .env << EOF
BACKEND1_URL=http://<backend1-private-ip>:8000
BACKEND2_URL=http://<backend2-private-ip>:8000
BACKEND3_URL=http://<backend3-private-ip>:8000
EOF

# Generate the real Prometheus config from the template (Prometheus doesn't
# do env-var substitution in its own YAML, so this is a one-time sed step)
sed -e "s/__BACKEND1_HOST__/<backend1-private-ip>/" \
    -e "s/__BACKEND2_HOST__/<backend2-private-ip>/" \
    -e "s/__BACKEND3_HOST__/<backend3-private-ip>/" \
    prometheus/prometheus.ec2.yml.template > prometheus/prometheus.ec2.yml

docker compose -f docker-compose.router.yml up -d --build
curl http://localhost:9000/health
curl http://localhost:8010/health
```

Confirm the engine can actually reach all 3 backends:

```bash
curl http://localhost:8010/status | python3 -m json.tool | grep -A3 metrics
# should show real, non-error metrics for backend1/2/3, not the
# unreachable-fallback values
```

## 6. Run the benchmark pipeline

The benchmark scripts (`run_aws_benchmarks.py`, `loadgen.py`, etc.) run
**on the router instance** (they need Docker socket access to recreate the
engine container between training iterations, and they talk to
`localhost:9000`/`localhost:8010` by default). SSH into the router
instance:

```bash
cd ~/traffic-router/risk-model
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install matplotlib seaborn locust xgboost

# 3-minute phases, matching the router instance's already-configured
# BACKEND1_URL/2/3_URL from step 5 -- loadgen.py picks those up
# automatically since it's running on this same instance
python run_aws_benchmarks.py --baseline-seconds 180 --opt-seconds 180
```

Run it in `tmux` so an SSH disconnect doesn't kill it -- this takes
~25-30 minutes total across all 6 phases.

To train with XGBoost instead of the default logistic regression at any
point (see `generate_and_train.py --help`):

```bash
python generate_and_train.py --datasets baseline opt1 --model-type xgboost
```

Retraining is **continual by default** -- it does not start from scratch.
It loads the latest existing version of whichever `--model-type` you
picked and continues training from it (warm-started for logistic
regression, additional boosting rounds on the existing trees for
XGBoost), and every version ever trained stays on disk under
`risk-model/models/` with full lineage in `models/registry.json` --
nothing is ever erased. Pass `--fresh` if you explicitly want to discard
history and start over.

## 7. Generate the comparison report and charts

Still on the router instance:

```bash
python compare_all.py
python plot_benchmarks.py
ls plots/
```

## 8. Pull results back to your laptop

```bash
# from your LOCAL machine
scp -i your-key.pem -r ubuntu@<ROUTER_PUBLIC_IP>:~/traffic-router/risk-model/plots ./plots
scp -i your-key.pem -r ubuntu@<ROUTER_PUBLIC_IP>:~/traffic-router/risk-model/models ./models
scp -i your-key.pem ubuntu@<ROUTER_PUBLIC_IP>:~/traffic-router/risk-model/*.json ./results
```

## 8b. Viewing the live dashboard against the real deployment

Open `dashboard/index.html` locally in your browser (it's a static file,
no need to deploy it anywhere) and change the "Engine URL" and "Router
URL" fields at the top from their `localhost` defaults to
`http://<ROUTER_PUBLIC_IP>:8010` and `http://<ROUTER_PUBLIC_IP>:9000`.

## 9. Tear down (avoid ongoing charges)

```bash
# on each instance:
docker compose -f docker-compose.backend.yml down -v   # backend instances
docker compose -f docker-compose.router.yml down -v    # router instance

# then terminate all 4 instances from the AWS Console, or:
aws ec2 terminate-instances --instance-ids <id1> <id2> <id3> <id4>
```

## Option 2: real ALB as the data plane, weighted target groups

Everything above deploys the **custom router** as the internet-facing
entry point (it plays the role ALB would play). This section replaces
that with a **real AWS Application Load Balancer**: ALB does the actual
health-checking and request forwarding, and the engine's optimizer keeps
computing risk-aware weights exactly as before, but instead of writing
`weights.json` for a router process to read, it periodically calls the
ELBv2 API to update the ALB listener's weighted forward action. See
`engine/alb_controller.py` for the implementation and
`docker-compose.engine-alb.yml` for the compose file this mode uses (no
router container).

### Why this is better than the router-as-ALB setup above

- ALB does health checks, TLS termination, connection draining and
  multi-AZ failover for you -- the custom router does none of this.
- The optimizer's decision logic (`engine/optimizer.py`'s objective
  function, RL calibration, online retraining) is completely unchanged.
  Only the last step -- how the computed weights actually get applied to
  traffic -- moves from "router reads weights.json per request" to
  "engine pushes weights.json's content to ALB every ~15s".
- Because ALB is now a real load-balancing resource, `TARGET_GROUP_ARN_*`
  can also serve as your Round Robin / Least Outstanding Requests
  **baselines** directly from ALB's own `load_balancing.algorithm.type`
  target-group attribute, instead of reimplementing those two algorithms
  in `router/router.py`.

### What changes vs. the setup above

- **3 target groups**, one per backend, each with instance-type health
  checks on `/health`.
- **1 Application Load Balancer**, internet-facing, with **1 listener**
  whose *default action* is a single weighted `forward` across all three
  target groups (this is exactly what `alb_controller.py`'s
  `modify_listener` call updates).
- **No router container.** `docker-compose.engine-alb.yml` runs only
  `engine`, `chaos-orchestrator`, and `prometheus` -- there's nothing for
  a router to do once ALB is the one actually receiving and forwarding
  client requests.
- The 3 backend EC2 instances and `docker-compose.backend.yml` are
  **unchanged** from the setup above.

### 1. Create target groups (one per backend)

For each backend, in the same VPC as the backend instances:

```bash
aws elbv2 create-target-group \
  --name backend1-tg --protocol HTTP --port 8000 \
  --vpc-id <vpc-id> --target-type instance \
  --health-check-path /health --health-check-interval-seconds 10

aws elbv2 register-targets \
  --target-group-arn <backend1-tg-arn> \
  --targets Id=<backend1-instance-id>
```

Repeat for `backend2-tg` / `backend3-tg`. Note each target group's ARN
-- you'll need all three below.

### 2. Create the ALB and listener

```bash
aws elbv2 create-load-balancer \
  --name traffic-router-alb --type application --scheme internet-facing \
  --subnets <public-subnet-1> <public-subnet-2> \
  --security-groups <alb-security-group-id>

aws elbv2 create-listener \
  --load-balancer-arn <alb-arn> --protocol HTTP --port 80 \
  --default-actions '[{
    "Type": "forward",
    "ForwardConfig": {
      "TargetGroups": [
        {"TargetGroupArn": "<backend1-tg-arn>", "Weight": 1},
        {"TargetGroupArn": "<backend2-tg-arn>", "Weight": 1},
        {"TargetGroupArn": "<backend3-tg-arn>", "Weight": 1}
      ]
    }
  }]'
```

Note the resulting listener's ARN (`aws elbv2 describe-listeners
--load-balancer-arn <alb-arn>` if you didn't capture it from the
`create-listener` output).

The ALB's security group needs inbound `80` open to the internet (or
your IP for testing); each backend's security group needs inbound
`8000` open **from the ALB's security group**, not `0.0.0.0/0`.

### 3. Give the engine instance permission to update listener weights

Attach an IAM instance profile to the engine EC2 instance with the
policy in `engine/iam-policy-alb-weights.json` (tighten `Resource` to
the specific listener ARN once you have it, instead of `*`). boto3 on
the instance picks this up automatically -- no access keys needed
anywhere in `.env` or the image.

### 4. Deploy the engine instance

```bash
cd ~/traffic-router
cat > .env << EOF
BACKEND1_URL=http://<backend1-private-ip>:8000
BACKEND2_URL=http://<backend2-private-ip>:8000
BACKEND3_URL=http://<backend3-private-ip>:8000
ALB_ENABLED=true
AWS_REGION=us-east-1
ALB_LISTENER_ARN=<listener-arn-from-step-2>
TARGET_GROUP_ARN_BACKEND1=<backend1-tg-arn>
TARGET_GROUP_ARN_BACKEND2=<backend2-tg-arn>
TARGET_GROUP_ARN_BACKEND3=<backend3-tg-arn>
EOF

docker compose -f docker-compose.engine-alb.yml up -d --build
curl http://localhost:8010/health
```

### 5. Confirm it's actually pushing weights

```bash
curl http://localhost:8010/alb | python3 -m json.tool
```

`enabled` should be `true`, `last_pushed_weights` should start filling in
within `ALB_PUSH_INTERVAL_S` seconds (default 15s) once the optimizer has
computed at least one set of weights, and `last_error` should be `null`.
If it's not, the message there is almost always a missing ARN or a
missing IAM permission.

### 6. Send real traffic and watch weights shift

```bash
curl http://<alb-dns-name>/work
```

Point your load generator (`risk-model/loadgen.py` or
`--load-generator locust`) at the ALB's DNS name instead of
`localhost:9000`, trigger chaos on a backend
(`curl -X POST http://<backend-ip>:8000/chaos/set?type=latency&severity=severe`),
and watch `curl http://localhost:8010/alb` show that backend's weight
drop over the next couple of push cycles.

### Notes on this mode

- `ALB_PUSH_INTERVAL_S` (default 15s) and `ALB_MIN_WEIGHT_DELTA` (default
  0.03) in `engine/alb_controller.py` control how often ALB's control
  plane actually gets called -- the optimizer still re-scores risk every
  2s internally, this only throttles how often that gets pushed out.
- `weights.json`/`mode.json` in `/shared` are still written every tick
  exactly as before -- nothing reads them in this mode, but nothing had
  to be removed either. That's also why round_robin/lor/default modes
  set via `POST /mode` are inert here: they only ever affected the
  router's own routing logic, which isn't in the request path anymore.
- If you also want ALB itself to serve as your Round Robin / LOR
  **baseline** measurements (rather than the router's own
  implementations in `router/router.py`), set the relevant target
  group's `load_balancing.algorithm.type` attribute and temporarily set
  all three target group weights equal via the same listener action --
  that reproduces Round Robin. LOR isn't a native ALB algorithm, so that
  baseline still needs the router's implementation or a separate
  measurement pass.

## Notes

- **The optimizer's objective function, trade-off parameters
  (alpha/beta/gamma/delta), and SciPy solver setup in
  `engine/optimizer.py` are unchanged** by anything in this deployment --
  only how services find each other over the network changed
  (`BACKEND1_URL`/`BACKEND2_URL`/`BACKEND3_URL` env vars instead of
  hardcoded docker-compose hostnames).
- **The RL calibration agent (`engine/rl_agent.py`) needs no deployment
  changes at all** -- it already runs its full predict-then-learn cycle
  (`correct()` then `update()`) every single engine tick, in-process,
  regardless of where the backends physically live. That's your "live
  data as an RL loop for live-time prediction and learning": it's been
  continuously active this whole time, on every tick, not just during
  the optimizer benchmark phases.
- For local testing without any of this (single host, docker network
  hostnames, no EC2 needed), keep using the original `docker-compose.yml`
  as before -- it's unchanged and still works exactly as it did.
- If you want the router itself fronted by a real AWS ALB (for TLS
  termination, multi-AZ failover of the router instance itself, etc.)
  that's a further layer you can add on top -- point a real ALB's target
  group at the router instance's port 9000. That's a different question
  from what this project's router is doing internally, which is the
  smarter routing decision an ALB alone can't make.
