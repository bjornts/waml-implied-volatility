"""Train the ablation models. Finished runs are skipped, so this can be re-run safely.

Example:
    python scripts/run_experiments.py --scenarios noisy --precisions single double
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from waml.calibration import MODEL_TERMS, TrainConfig, train
from waml.data import SCENARIOS, holdout_path, quotes_path
from waml.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUN_DIR = ROOT / "results" / "runs"


def run_name(scenario: str, model: str, precision: str, seed: int) -> str:
    return f"{scenario}__{model}__{precision}__seed{seed}"


def run_one(scenario: str, model: str, precision: str, seed: int, steps: int, threads: int) -> str:
    """Train and evaluate one model, and save its metrics and history as JSON."""
    torch.set_num_threads(threads)
    name = run_name(scenario, model, precision, seed)
    quotes = pd.read_csv(quotes_path(DATA_DIR, scenario, seed))
    holdout = pd.read_csv(holdout_path(DATA_DIR))
    config = TrainConfig(
        model=model,
        steps=steps,
        double=precision == "double",
        seed=seed,
        first_update=0 if model == "dcpinn_waml_step0" else 100,
        balance_floor=1e-3 if model == "waml_floor" else 0.0,
    )

    result = train(
        torch.tensor(quotes["strike"].to_numpy()),
        torch.tensor(quotes["tau"].to_numpy()),
        torch.tensor(quotes["price"].to_numpy()),
        config,
    )
    record = {
        "scenario": scenario,
        "model": model,
        "precision": precision,
        "seed": seed,
        "seconds": result.seconds,
        "nonfinite_points": result.nonfinite_points,
        "config": asdict(config),
        "metrics": evaluate(result.net, holdout),
        "history": result.history,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / f"{name}.json").write_text(json.dumps(record, indent=1))
    if seed == 0:
        torch.save(result.net.state_dict(), RUN_DIR / f"{name}.pt")
    return f"{name}: {result.seconds:.0f}s iv_rmse={record['metrics']['iv_rmse']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--models", nargs="+", default=list(MODEL_TERMS), choices=list(MODEL_TERMS))
    parser.add_argument("--precisions", nargs="+", default=["single"], choices=["single", "double"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    jobs = [
        job
        for job in itertools.product(args.scenarios, args.models, args.precisions, range(args.seeds))
        if not (RUN_DIR / f"{run_name(*job)}.json").exists()
    ]
    print(f"{len(jobs)} runs to do", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, *job, args.steps, args.threads): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            print(f"[{done}/{len(jobs)}] {future.result()}", flush=True)


if __name__ == "__main__":
    main()
