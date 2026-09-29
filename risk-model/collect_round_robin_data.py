"""
PHASE 1a of the evaluation pipeline: Round Robin baseline.

Puts the router into pure round-robin mode (cycles through backends in
fixed order, completely ignoring health/latency/risk -- one of the actual
routing algorithms AWS ALB offers), drives real traffic through it with
your choice of load generator, injects randomized chaos throughout, and
saves data_round_robin.csv (training data) + benchmark_round_robin.json
(real measured performance).

Usage:
    python collect_round_robin_data.py
    python collect_round_robin_data.py --seconds 300 --load-generator locust --users 100 --spawn-rate 20
"""
import argparse

import loadgen


def main(seconds, generator, users, spawn_rate):
    loadgen.run_collection_phase(
        mode="round_robin",
        seconds=seconds,
        generator=generator,
        users=users,
        spawn_rate=spawn_rate,
        training_csv="data_round_robin.csv",
        benchmark_json="benchmark_round_robin.json",
    )
    print("\nNext: python collect_lor_data.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--load-generator", choices=["internal", "locust"], default="internal")
    parser.add_argument("--users", type=int, default=50, help="Locust virtual users (ignored for --load-generator internal)")
    parser.add_argument("--spawn-rate", type=int, default=10, help="Locust users spawned per second (ignored for internal)")
    args = parser.parse_args()
    main(args.seconds, args.load_generator, args.users, args.spawn_rate)
