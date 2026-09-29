"""
Pushes the optimizer's live weights into a REAL AWS Application Load
Balancer via the ELBv2 API, instead of a custom router applying them
in-process.

Architecture this supports (see AWS_DEPLOYMENT.md, "Option 2"):
  - ALB is the actual internet-facing data plane. It health-checks
    backend1/2/3 and forwards every request itself.
  - Each backend is its own ALB target group.
  - The ALB listener's default action is a single weighted "forward"
    across all three target groups.
  - This module is the ONLY thing that talks to the ALB control plane.
    It does not touch the request path at all -- it just periodically
    tells ALB "here is the new split", using the exact same
    alpha/beta/gamma/delta-optimized weights that used to be written to
    weights.json for the custom router to consume.

Why this is a periodic control-plane push, not a per-request decision:
ELBv2's ModifyListener/ModifyRule are management-plane API calls with
their own throttling limits -- calling them on every request (or every
2s engine tick) will get rate-limited and adds nothing, since ALB's own
data plane is what actually forwards each request once the weights are
set. So the engine keeps scoring risk every 2s as before; this
controller only pushes a NEW split to ALB every ALB_PUSH_INTERVAL_S
seconds, and only if the weights actually moved.

Enable by setting ALB_ENABLED=true and the ARNs below. When disabled
(the default), this module is inert -- nothing here changes local /
docker-compose behavior, and the custom router keeps working exactly
as it always has.
"""
import os
import time

BACKENDS = ["backend1", "backend2", "backend3"]

ALB_ENABLED = os.environ.get("ALB_ENABLED", "false").lower() == "true"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# The ALB listener whose DEFAULT ACTION is the weighted forward across all
# three target groups. Find it in the console under
# EC2 > Load Balancers > <your ALB> > Listeners, or via:
#   aws elbv2 describe-listeners --load-balancer-arn <alb-arn>
ALB_LISTENER_ARN = os.environ.get("ALB_LISTENER_ARN", "")

# One target group ARN per backend -- create these first (one per
# backend EC2 instance), register the instance, THEN create the ALB
# listener with a weighted forward action across all three.
TARGET_GROUP_ARNS = {
    "backend1": os.environ.get("TARGET_GROUP_ARN_BACKEND1", ""),
    "backend2": os.environ.get("TARGET_GROUP_ARN_BACKEND2", ""),
    "backend3": os.environ.get("TARGET_GROUP_ARN_BACKEND3", ""),
}

# Don't hammer the ELBv2 control plane -- push at most this often.
ALB_PUSH_INTERVAL_S = float(os.environ.get("ALB_PUSH_INTERVAL_S", "15"))

# Only push if any backend's weight moved by more than this, so a
# constant stream of near-identical optimizer outputs doesn't turn into
# a constant stream of ALB API calls.
ALB_MIN_WEIGHT_DELTA = float(os.environ.get("ALB_MIN_WEIGHT_DELTA", "0.03"))

_client = None
_last_push_time = 0.0
_last_pushed_weights = {b: None for b in BACKENDS}
_last_error = None
_push_count = 0


def _get_client():
    global _client
    if _client is None:
        import boto3  # imported lazily so boto3 is only required when ALB_ENABLED=true
        _client = boto3.client("elbv2", region_name=AWS_REGION)
    return _client


def _to_alb_weight(w: float) -> int:
    """ALB target-group weights are integers 0-999. 0 is valid and means
    'send this target group no traffic' -- exactly what we want when the
    optimizer has decided a backend is too risky right now."""
    return max(0, min(999, round(w * 1000)))


def _weights_changed(new_weights: dict) -> bool:
    for b in BACKENDS:
        prev = _last_pushed_weights[b]
        if prev is None or abs(new_weights[b] - prev) > ALB_MIN_WEIGHT_DELTA:
            return True
    return False


def status() -> dict:
    return {
        "enabled": ALB_ENABLED,
        "listener_arn": ALB_LISTENER_ARN or None,
        "target_groups": {b: (arn or None) for b, arn in TARGET_GROUP_ARNS.items()},
        "last_pushed_weights": _last_pushed_weights,
        "last_push_time": _last_push_time or None,
        "push_count": _push_count,
        "last_error": _last_error,
    }


def maybe_push_weights(opt_weights: dict) -> None:
    """Call this once per engine tick with the freshly-computed optimizer
    weights. It's a no-op unless ALB_ENABLED=true, the push interval has
    elapsed, and the weights actually moved."""
    global _last_push_time, _last_pushed_weights, _last_error, _push_count

    if not ALB_ENABLED:
        return
    now = time.time()
    if now - _last_push_time < ALB_PUSH_INTERVAL_S:
        return
    if not _weights_changed(opt_weights):
        return
    if not ALB_LISTENER_ARN or any(not TARGET_GROUP_ARNS[b] for b in BACKENDS):
        _last_error = "ALB_ENABLED=true but ALB_LISTENER_ARN / TARGET_GROUP_ARN_* are not fully set"
        print(f"[alb_controller] {_last_error}", flush=True)
        return

    target_groups = [
        {"TargetGroupArn": TARGET_GROUP_ARNS[b], "Weight": _to_alb_weight(opt_weights[b])}
        for b in BACKENDS
    ]

    try:
        client = _get_client()
        client.modify_listener(
            ListenerArn=ALB_LISTENER_ARN,
            DefaultActions=[
                {
                    "Type": "forward",
                    "ForwardConfig": {
                        "TargetGroups": target_groups,
                        "TargetGroupStickinessConfig": {"Enabled": False},
                    },
                }
            ],
        )
        _last_push_time = now
        _last_pushed_weights = dict(opt_weights)
        _last_error = None
        _push_count += 1
        print(f"[alb_controller] pushed weights to ALB: {opt_weights}", flush=True)
    except Exception as e:
        _last_error = str(e)
        print(f"[alb_controller] failed to push weights: {e}", flush=True)
