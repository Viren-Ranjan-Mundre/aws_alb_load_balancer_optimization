"""
PHASE 2 of the evaluation pipeline: test the trained, RL-assisted
optimizer under the same conditions as both baselines.

Run this AFTER:
  1. collect_round_robin_data.py  -> benchmark_round_robin.json
  2. collect_lor_data.py          -> benchmark_lor.json
  3. generate_and_train.py        -> trains on the combined data from 1+2
  4. `docker compose up --build --force-recreate` to load the retrained model

This puts the router into "optimizer" mode (risk-aware, RL-calibrated,
multi-objective routing — the actual system this project is about), runs
the same kind of load + randomized chaos for the same duration you used
for the baselines, and saves benchmark_optimizer.json + data_optimizer.csv
(useful if you want to keep growing the training set with optimizer-mode
telemetry too).

Usage:
    python evaluate_optimizer.py                # use the same --seconds as your baseline runs
    python evaluate_optimizer.py --seconds 300 --load-generator locust --users 100 --spawn-rate 20
"""
import argparse

import loadgen


def main(seconds, generator, users, spawn_rate):
    loadgen.run_collection_phase(
        mode="optimizer",
        seconds=seconds,
        generator=generator,
        users=users,
        spawn_rate=spawn_rate,
        training_csv="data_optimizer.csv",
        benchmark_json="benchmark_optimizer.json",
    )
    print("\nNext: python compare_all.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--load-generator", choices=["internal", "locust"], default="internal")
    parser.add_argument("--users", type=int, default=50, help="Locust virtual users (ignored for --load-generator internal)")
    parser.add_argument("--spawn-rate", type=int, default=10, help="Locust users spawned per second (ignored for internal)")
    args = parser.parse_args()
    main(args.seconds, args.load_generator, args.users, args.spawn_rate)
