"""
PHASE 1c of the evaluation pipeline: Default Strategy baseline.

Puts the router into "default" mode: static equal weights (1/3 per
backend), fixed for the whole run, with no ML risk model and no RL
feedback of any kind touching the routing decision -- this is the "what
if you just hardcoded even weights and never adapted" baseline. Distinct
from round_robin (which cycles deterministically through backends one at
a time) and from lor (which reacts live to in-flight request counts):
default is the one baseline that is both weighted AND completely static.

Drives real traffic through it with your choice of load generator,
injects randomized chaos throughout (identical distribution to the other
two baseline phases, for an apples-to-apples comparison), and saves
data_default.csv (training data) + benchmark_default.json (real measured
performance).

Usage:
    python collect_default_data.py
    python collect_default_data.py --seconds 300 --load-generator locust --users 100 --spawn-rate 20
"""
import argparse

import loadgen


def main(seconds, generator, users, spawn_rate):
    loadgen.run_collection_phase(
        mode="default",
        seconds=seconds,
        generator=generator,
        users=users,
        spawn_rate=spawn_rate,
        training_csv="data_default.csv",
        benchmark_json="benchmark_default.json",
    )
    print("\nNext: python generate_and_train.py --datasets baseline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--load-generator", choices=["internal", "locust"], default="internal")
    parser.add_argument("--users", type=int, default=50, help="Locust virtual users (ignored for --load-generator internal)")
    parser.add_argument("--spawn-rate", type=int, default=10, help="Locust users spawned per second (ignored for internal)")
    args = parser.parse_args()
    main(args.seconds, args.load_generator, args.users, args.spawn_rate)
