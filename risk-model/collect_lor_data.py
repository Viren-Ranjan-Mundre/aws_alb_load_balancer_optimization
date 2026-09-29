"""
PHASE 1b of the evaluation pipeline: Least Outstanding Requests baseline.

Puts the router into LOR mode -- each request goes to whichever backend
currently has the fewest in-flight requests (see router/router.py for the
real, live-tracked implementation; this is another real AWS ALB routing
algorithm). Naturally favors faster/healthier backends over time without
any risk model at all, which is exactly why it's a meaningfully stronger
baseline than round robin, and why the optimizer needs to beat BOTH.

Saves data_lor.csv (training data) + benchmark_lor.json (real measured
performance).

Usage:
    python collect_lor_data.py
    python collect_lor_data.py --seconds 300 --load-generator locust --users 100 --spawn-rate 20
"""
import argparse

import loadgen


def main(seconds, generator, users, spawn_rate):
    loadgen.run_collection_phase(
        mode="lor",
        seconds=seconds,
        generator=generator,
        users=users,
        spawn_rate=spawn_rate,
        training_csv="data_lor.csv",
        benchmark_json="benchmark_lor.json",
    )
    print("\nNext: python generate_and_train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--load-generator", choices=["internal", "locust"], default="internal")
    parser.add_argument("--users", type=int, default=50, help="Locust virtual users (ignored for --load-generator internal)")
    parser.add_argument("--spawn-rate", type=int, default=10, help="Locust users spawned per second (ignored for internal)")
    args = parser.parse_args()
    main(args.seconds, args.load_generator, args.users, args.spawn_rate)
