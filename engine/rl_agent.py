"""
A lightweight reinforcement-learning layer that sits on top of the
supervised risk classifier.

Why this is a *different* kind of adaptation than the online retraining
loop in optimizer.py: the classifier learns to predict "is this backend
in a chaos episode" from (latency, error_rate, cpu) using labeled
examples. It can still be systematically mis-calibrated for reasons a
label alone doesn't capture — e.g. always slightly over-confident at
medium risk, or too slow to distrust a backend once its risk crosses a
certain band. This agent learns, from real measured outcomes (not
labels), a correction to apply on top of the classifier's probability,
so the *routing decision* keeps improving even between classifier
retrains.

Formulation — a contextual bandit solved with tabular Q-learning:
  - State  : which risk bucket the classifier currently places a backend in
             (5 buckets across [0, 1]). Buckets, not raw backend identity,
             so what's learned generalizes to whichever physical backend
             is currently in that bucket.
  - Action : a small correction added to the classifier's risk estimate
             before it's fed into the routing optimization (nudge trust
             up or down).
  - Reward : negative distance between the corrected risk and what
             actually happened (ground-truth chaos state, or observed
             error rate as a softer signal) — so the action taken
             directly determines the reward, not just the state.
  - Update : single-step Q-learning (discount = 0, since the "episode" is
             one decision cycle — this is the standard bandit reduction of
             Q-learning, not a claim of modeling long-horizon dynamics).

Exploration is epsilon-greedy with decay, so the agent explores corrections
early on and converges to exploiting whatever correction has empirically
paid off best per bucket.
"""
import random

N_BUCKETS = 5
BUCKET_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ACTIONS = [-0.2, -0.1, 0.0, 0.1, 0.2]  # risk-correction deltas


def bucket_of(risk: float) -> int:
    risk = min(max(risk, 0.0), 0.999999)
    for i in range(N_BUCKETS):
        if BUCKET_EDGES[i] <= risk < BUCKET_EDGES[i + 1]:
            return i
    return N_BUCKETS - 1


class RiskCalibrationAgent:
    def __init__(self, lr=0.05, epsilon_start=0.5, epsilon_min=0.1, epsilon_decay=0.9995):
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = [[0.0 for _ in ACTIONS] for _ in range(N_BUCKETS)]
        self.total_reward = 0.0
        self.steps = 0
        self.last_actions = {}  # backend -> action index chosen this cycle, for transparency

    def choose_action(self, bucket: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))
        row = self.q[bucket]
        best = max(row)
        # break ties randomly instead of always picking the first max
        candidates = [i for i, v in enumerate(row) if v == best]
        return random.choice(candidates)

    def correct(self, backend: str, risk: float) -> float:
        """Given the classifier's raw risk for a backend, pick an action
        and return the corrected (effective) risk used for routing."""
        bucket = bucket_of(risk)
        action_idx = self.choose_action(bucket)
        corrected = min(max(risk + ACTIONS[action_idx], 0.0), 1.0)
        self.last_actions[backend] = (bucket, action_idx, corrected)
        return corrected

    def update(self, backend: str, true_harm: float):
        """Feed back what actually happened to this backend this cycle —
        true_harm in [0, 1], where 1.0 means "was genuinely in a chaos
        episode" (ground truth) and otherwise the observed error rate is
        used as a softer harm signal. Reward is the negative distance
        between the CORRECTED risk and this real outcome, so the action
        itself determines the reward: an action that overcorrects trust
        upward when a backend turns out fine is penalized, and one that
        undercorrects when a backend turns out to be in real trouble is
        penalized too. This is what lets the Q-table converge to a
        genuine per-bucket calibration bias instead of a bucket-wide
        constant that ignores which action was taken."""
        if backend not in self.last_actions:
            return
        bucket, action_idx, corrected = self.last_actions[backend]
        reward = -abs(corrected - true_harm)
        q_old = self.q[bucket][action_idx]
        self.q[bucket][action_idx] = q_old + self.lr * (reward - q_old)
        self.total_reward += reward
        self.steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def status(self) -> dict:
        return {
            "epsilon": round(self.epsilon, 4),
            "steps": self.steps,
            "avg_reward": (self.total_reward / self.steps) if self.steps else None,
            "actions": ACTIONS,
            "q_table": [
                {
                    "bucket": f"[{BUCKET_EDGES[i]:.1f}, {BUCKET_EDGES[i+1]:.1f})",
                    "values": [round(v, 4) for v in row],
                    "best_action": ACTIONS[row.index(max(row))],
                }
                for i, row in enumerate(self.q)
            ],
        }
