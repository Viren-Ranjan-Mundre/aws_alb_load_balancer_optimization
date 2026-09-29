"""
Trains the failure-risk classifier used by the optimization engine.

Three things this version adds on top of the dataset-token system:

1. --model-type {logistic, xgboost}: choice of classifier. Both expose
   the same .predict_proba() interface the engine already calls, so no
   engine changes are needed to switch which one is actually deployed.

2. VERSIONING: every training run is saved as a new numbered version
   under models/ (e.g. models/risk_model_v3_logistic.pkl) and NEVER
   overwrites or deletes a previous version. models/registry.json is the
   full lineage: every version, what data trained it, what version (if
   any) it continued from, and its holdout metrics. The top-level
   risk_model.pkl / ../engine/risk_model.pkl are just a copy of whichever
   version is newest -- that's the one the engine actually loads -- but
   the full history stays on disk untouched.

3. CONTINUAL TRAINING (the default, not opt-in): retraining does NOT
   start from a randomly-initialized model. It loads the latest existing
   version of the SAME model type and continues from it --
   warm_start=True for logistic regression (reuses its previous
   coef_/intercept_ as the optimizer's starting point, standard sklearn
   pattern for repeated .fit() calls on the same estimator) and
   xgb_model=<previous booster> for XGBoost (adds new trees on top of the
   existing ones instead of growing a fresh ensemble). Combined with
   --datasets baseline opt1 opt2 (cumulative data), this means neither
   the data nor the learned parameters from earlier training are thrown
   away when you retrain. Pass --fresh to force a from-scratch fit
   instead, if you genuinely want that.

Usage:
    python generate_and_train.py --datasets baseline
    python generate_and_train.py --datasets baseline opt1 --model-type xgboost
    python generate_and_train.py --datasets baseline opt1 opt2 --fresh   # ignore history, start over
"""
import argparse
import glob
import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

here = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(here, "models")
registry_path = os.path.join(models_dir, "registry.json")

DATASET_FILES = {
    "round_robin": "data_round_robin.csv",
    "lor": "data_lor.csv",
    "default": "data_default.csv",
    "opt1": "data_opt1.csv",
    "opt2": "data_opt2.csv",
    "opt3": "data_opt3.csv",
}
BASELINE_TOKENS = ["round_robin", "lor", "default"]
LEGACY_CSV = os.path.join(here, "training_data.csv")


# --------------------------------------------------------------------------
# Dataset resolution (unchanged from the previous version)
# --------------------------------------------------------------------------

def resolve_dataset_tokens(tokens):
    resolved = []
    for t in tokens:
        expansion = BASELINE_TOKENS if t == "baseline" else [t]
        for e in expansion:
            if e not in DATASET_FILES:
                raise ValueError(f"Unknown dataset token '{e}'. Valid tokens: "
                                  f"{list(DATASET_FILES.keys())} or 'baseline'.")
            if e not in resolved:
                resolved.append(e)
    return resolved


def load_real_csv(path):
    df = pd.read_csv(path)
    required = ["latency", "error_rate", "cpu", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.dropna(subset=required)


def load_datasets(tokens):
    parts, sources, missing = [], [], []
    for t in tokens:
        path = os.path.join(here, DATASET_FILES[t])
        if os.path.exists(path):
            parts.append(load_real_csv(path))
            sources.append(DATASET_FILES[t])
        else:
            missing.append(DATASET_FILES[t])

    if missing:
        raise FileNotFoundError(
            f"Requested dataset(s) not found: {missing}. Run the matching "
            f"collect_*.py script(s) first."
        )

    combined = pd.concat(parts, ignore_index=True)
    X = combined[["latency", "error_rate", "cpu"]].values
    y = combined["label"].values
    return X, y, len(combined), sources


def generate_realistic_synthetic(n_episodes=400, rng_seed=42, only_label=None):
    """Fallback when no real data is available at all, and to top up a
    real dataset that's missing one class entirely."""
    rng = np.random.default_rng(rng_seed)
    severity_profiles = {
        "mild": {"latency_add": (0.10, 0.25), "error_p": 0.05, "cpu_add": 25},
        "moderate": {"latency_add": (0.25, 0.55), "error_p": 0.20, "cpu_add": 45},
        "severe": {"latency_add": (0.55, 1.20), "error_p": 0.45, "cpu_add": 70},
    }
    chaos_types = ["latency", "errors", "cpu", "combo"]

    X, y = [], []
    for _ in range(n_episodes):
        if only_label == 0:
            is_chaos = False
        elif only_label == 1:
            is_chaos = True
        else:
            is_chaos = rng.random() < 0.35
        steps = rng.integers(4, 12)

        if not is_chaos:
            for _ in range(steps):
                lat = max(0.02, rng.normal(0.05, 0.01))
                err = max(0.0, rng.normal(0.01, 0.01))
                cpu = max(5, rng.normal(25, 6))
                X.append([lat, err, cpu])
                y.append(0)
            continue

        chaos_type = rng.choice(chaos_types)
        severity = rng.choice(list(severity_profiles.keys()), p=[0.5, 0.3, 0.2])
        profile = severity_profiles[severity]

        ramp = max(1, steps // 4)
        for i in range(steps):
            progress = min(1.0, (i + 1) / ramp) if i < ramp else 1.0
            lat = 0.05
            cpu = max(5, rng.normal(25, 6))
            if chaos_type in ("latency", "combo"):
                lat += rng.uniform(*profile["latency_add"]) * progress
            if chaos_type in ("cpu", "combo"):
                cpu += profile["cpu_add"] * progress
                lat += rng.uniform(0.05, 0.20) * progress
            err = profile["error_p"] * progress if chaos_type in ("errors", "combo") else max(0.0, rng.normal(0.01, 0.01))
            X.append([max(0.02, lat), max(0.0, min(1.0, err)), max(5, cpu)])
            y.append(1)

    return np.array(X), np.array(y)


# --------------------------------------------------------------------------
# Versioning
# --------------------------------------------------------------------------

def load_registry():
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            return json.load(f)
    return {"versions": []}


def save_registry(registry):
    os.makedirs(models_dir, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)


def latest_version_for_type(registry, model_type):
    """Highest version number trained with this model_type, or None."""
    versions = [v for v in registry["versions"] if v["model_type"] == model_type]
    if not versions:
        return None
    return max(versions, key=lambda v: v["version"])


def next_version_number(registry):
    if not registry["versions"]:
        return 1
    return max(v["version"] for v in registry["versions"]) + 1


def model_filename(version, model_type):
    return f"risk_model_v{version}_{model_type}.pkl"


# --------------------------------------------------------------------------
# Model construction / continual training
# --------------------------------------------------------------------------

def build_fresh_model(model_type):
    if model_type == "logistic":
        return LogisticRegression(warm_start=True)
    elif model_type == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
        )
    raise ValueError(f"Unknown model_type '{model_type}'")


def fit_continual(model_type, X, y, previous_model_path, fresh: bool):
    """Fits a model on (X, y). If fresh=False and a previous version of
    this model_type exists, continues training from it instead of
    starting from randomly-initialized parameters:
      - logistic: previous_model.fit(X, y) with warm_start=True already
        set -- sklearn reuses its own existing coef_/intercept_ as the
        optimizer's starting point (this is why we load and re-fit the
        SAME object, rather than building a new one and copying weights
        over by hand).
      - xgboost: fit(..., xgb_model=<previous booster>) -- new trees are
        added on top of the existing ensemble instead of growing a fresh
        one from round 0.
    Returns (fitted_model, continued_from_path_or_None).
    """
    if not fresh and previous_model_path and os.path.exists(previous_model_path):
        print(f"  Continual training: loading {previous_model_path} as the starting point "
              f"(not training from scratch).")
        model = joblib.load(previous_model_path)
        if model_type == "logistic":
            model.warm_start = True
            model.fit(X, y)
        elif model_type == "xgboost":
            import xgboost as xgb
            prev_booster = model.get_booster()
            new_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
            )
            new_model.fit(X, y, xgb_model=prev_booster)
            model = new_model
        return model, previous_model_path

    print(f"  Training {model_type} from scratch"
          + (" (--fresh set)" if fresh else " (no previous version of this model_type found)") + ".")
    model = build_fresh_model(model_type)
    model.fit(X, y)
    return model, None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Dataset tokens: round_robin lor default opt1 opt2 opt3, or 'baseline' "
             "as shorthand for round_robin+lor+default. E.g. --datasets baseline opt1 opt2.",
    )
    parser.add_argument(
        "--model-type", choices=["logistic", "xgboost"], default="logistic",
        help="Which classifier to train. Both are drop-in compatible with the engine "
             "(same .predict_proba() interface).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any previous version of this model_type and train from scratch. "
             "Default behavior is continual training (see module docstring).",
    )
    args = parser.parse_args()

    n_supplemented = 0

    if args.datasets:
        tokens = resolve_dataset_tokens(args.datasets)
        X, y, n_samples, sources = load_datasets(tokens)
        source = "real_live_collected"
        print(f"Training on datasets {args.datasets} -> files {sources} ({n_samples} samples total)")
    elif all(os.path.exists(os.path.join(here, DATASET_FILES[t])) for t in BASELINE_TOKENS):
        X, y, n_samples, sources = load_datasets(BASELINE_TOKENS)
        source = "real_live_collected"
        print(f"No --datasets given; defaulting to baseline -> files {sources} ({n_samples} samples total)")
    elif os.path.exists(LEGACY_CSV):
        df = load_real_csv(LEGACY_CSV)
        X = df[["latency", "error_rate", "cpu"]].values
        y = df["label"].values
        n_samples = len(df)
        sources = ["training_data.csv (legacy single-run format)"]
        source = "real_live_collected"
        print(f"No --datasets given, no baseline files found; using legacy {LEGACY_CSV} ({n_samples} samples)")
    else:
        print("No real training data found and no --datasets given -- falling back to "
              "realistic synthetic data.")
        X, y = generate_realistic_synthetic()
        n_samples = len(X)
        sources = ["synthetic_fallback"]
        source = "synthetic_fallback"

    classes_present = set(y.tolist())
    if len(classes_present) < 2:
        missing = 0 if 0 not in classes_present else 1
        missing_name = "healthy" if missing == 0 else "chaos"
        print(f"WARNING: the collected data only contains one class (no '{missing_name}' "
              f"examples) -- supplementing with synthetic '{missing_name}' examples so "
              f"training can proceed. Re-run the affected collect_*.py with a longer "
              f"--seconds for a fully real-data model.")
        n_supplement = max(len(y), 60)
        X_synth, y_synth = generate_realistic_synthetic(n_episodes=n_supplement // 6 + 5, only_label=missing)
        X = np.vstack([X, X_synth])
        y = np.concatenate([y, y_synth])
        n_supplemented = len(y_synth)
        source = "real_plus_synthetic_supplement"

    # Honest held-out evaluation FIRST, on a freshly-initialized model of
    # the same type/continuation-state, before refitting on everything.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    registry = load_registry()
    prev_version_entry = latest_version_for_type(registry, args.model_type)
    prev_model_path = (
        os.path.join(models_dir, model_filename(prev_version_entry["version"], args.model_type))
        if prev_version_entry else None
    )

    eval_model, _ = fit_continual(args.model_type, X_train, y_train, prev_model_path, args.fresh)
    test_acc = accuracy_score(y_test, eval_model.predict(X_test))
    train_acc = accuracy_score(y_train, eval_model.predict(X_train))
    try:
        test_auc = roc_auc_score(y_test, eval_model.predict_proba(X_test)[:, 1])
    except ValueError:
        test_auc = None

    print(f"Train accuracy: {train_acc:.3f} | Holdout accuracy: {test_acc:.3f}"
          + (f" | Holdout AUC: {test_auc:.3f}" if test_auc is not None else ""))

    # Now the real, deployed model: same continual-training logic, refit
    # on ALL available data (train+test) so nothing is held back from the
    # shipped model, exactly like the previous version of this script did.
    final_model, continued_from_path = fit_continual(args.model_type, X, y, prev_model_path, args.fresh)

    # --- Save this as a new version. Nothing before it is touched. ---
    os.makedirs(models_dir, exist_ok=True)
    version = next_version_number(registry)
    versioned_filename = model_filename(version, args.model_type)
    versioned_path = os.path.join(models_dir, versioned_filename)
    joblib.dump(final_model, versioned_path)

    # The engine loads these two paths -- always a copy of the newest version.
    engine_dir = os.path.abspath(os.path.join(here, "..", "engine"))
    os.makedirs(engine_dir, exist_ok=True)
    joblib.dump(final_model, os.path.join(here, "risk_model.pkl"))
    joblib.dump(final_model, os.path.join(engine_dir, "risk_model.pkl"))

    continued_from_version = prev_version_entry["version"] if (continued_from_path and prev_version_entry) else None

    registry_entry = {
        "version": version,
        "model_type": args.model_type,
        "file": versioned_filename,
        "trained_at": time.time(),
        "datasets_requested": args.datasets,
        "data_files_used": sources,
        "n_samples": int(n_samples),
        "n_synthetic_supplemented": int(n_supplemented),
        "continued_from_version": continued_from_version,
        "fresh": bool(args.fresh),
        "holdout_accuracy": round(float(test_acc), 4),
        "holdout_auc": round(float(test_auc), 4) if test_auc is not None else None,
    }
    registry["versions"].append(registry_entry)
    save_registry(registry)

    # model_meta.json stays as the quick "what's currently deployed" file
    # (same filename earlier tooling / the engine's own /model status
    # already expect), now just mirroring this version's registry entry.
    meta = {**registry_entry, "source": source}
    with open(os.path.join(here, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved version {version} ({args.model_type}) to {versioned_path}")
    if continued_from_version:
        print(f"Continued training from version {continued_from_version} (not from scratch).")
    else:
        print("Trained from scratch (no prior version of this model_type, or --fresh was set).")
    print(f"Deployed copy written to risk_model.pkl and ../engine/risk_model.pkl")
    print(f"Full lineage recorded in {registry_path} -- every previous version is still on disk, untouched.")


if __name__ == "__main__":
    main()
