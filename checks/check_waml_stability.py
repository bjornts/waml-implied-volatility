"""Trace one WamL collapse in detail, and test the notebook-style update on the same seed."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch

from waml.calibration import TrainConfig, train
from waml.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "variants"
SEED = 4
VARIANTS = {
    "stability_trace": dict(model="dcpinn_waml", steps=2500, log_every=10),
    "stability_weighted": dict(
        model="dcpinn_waml", weighted_updates=True, point_weight_lag=50, log_every=100
    ),
}


def run(name: str) -> str:
    torch.set_num_threads(1)
    quotes = pd.read_csv(ROOT / f"data/noisy_seed{SEED}.csv")
    holdout = pd.read_csv(ROOT / "data/holdout.csv")
    inputs = [torch.tensor(quotes[c].to_numpy()) for c in ("strike", "tau", "price")]
    result = train(*inputs, TrainConfig(seed=SEED, **VARIANTS[name]))
    metrics = evaluate(result.net, holdout)
    record = dict(name=name, seconds=result.seconds, metrics=metrics, history=result.history)
    (OUT / f"{name}.json").write_text(json.dumps(record, indent=1))
    return f"{name}: {result.seconds:.0f}s iv_rmse={metrics['iv_rmse']:.4f} viol={metrics['viol_any']:.3f}"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(len(VARIANTS)) as pool:
        for line in pool.map(run, VARIANTS):
            print(line, flush=True)
