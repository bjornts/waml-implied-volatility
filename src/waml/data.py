"""Synthetic SABR option quotes.

The paper's synthetic experiment samples from traded S&P 500 grids, which are not
public. The sampling distribution and noise model here follow the lead author's
public notebook: quotes cluster near the money and at short maturities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import truncnorm

from .pricing import SabrParams, bs_call, sabr_implied_vol

STRIKE_RANGE = (1e-3, 2.5)
TAU_RANGE = (1e-3, 5.0)
HOLDOUT_SIZE = 2_000
HOLDOUT_SEED = 10_000


@dataclass(frozen=True)
class Scenario:
    """A quote-generation setting."""

    name: str
    n_quotes: int
    vol_noise: float


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        Scenario("clean", 300, 0.0),
        Scenario("noisy", 300, 0.10),
        Scenario("sparse", 50, 0.0),
    )
}


def _truncated_normal(
    mean: float, sd: float, bounds: tuple[float, float], size: int, rng: np.random.Generator
) -> np.ndarray:
    low, high = (bounds[0] - mean) / sd, (bounds[1] - mean) / sd
    return truncnorm(low, high, loc=mean, scale=sd).rvs(size, random_state=rng)


def make_quotes(
    n_quotes: int, vol_noise: float, seed: int, params: SabrParams = SabrParams()
) -> pd.DataFrame:
    """Sample quotes. ``price`` carries the noise; ``true_*`` columns do not."""
    rng = np.random.default_rng(seed)
    strike = _truncated_normal(1.0, 0.4, STRIKE_RANGE, n_quotes, rng)
    tau = _truncated_normal(TAU_RANGE[0], 2.0, TAU_RANGE, n_quotes, rng)

    strike_t = torch.tensor(strike, dtype=torch.float64)
    tau_t = torch.tensor(tau, dtype=torch.float64)
    true_vol = sabr_implied_vol(strike_t, tau_t, params)
    noise = torch.tensor(rng.normal(0.0, vol_noise, n_quotes), dtype=torch.float64)

    return pd.DataFrame(
        {
            "strike": strike,
            "tau": tau,
            "price": bs_call(strike_t, tau_t, true_vol * (1.0 + noise)).numpy(),
            "true_vol": true_vol.numpy(),
            "true_price": bs_call(strike_t, tau_t, true_vol).numpy(),
        }
    )


def quotes_path(data_dir: Path, scenario: str, seed: int) -> Path:
    return data_dir / f"{scenario}_seed{seed}.csv"


def holdout_path(data_dir: Path) -> Path:
    return data_dir / "holdout.csv"


def write_datasets(data_dir: Path, seeds: range) -> None:
    """Write one training file per scenario and seed, plus a shared noiseless holdout."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for index, scenario in enumerate(SCENARIOS.values()):
        for seed in seeds:
            quotes = make_quotes(scenario.n_quotes, scenario.vol_noise, seed + 100 * (index + 1))
            quotes.to_csv(quotes_path(data_dir, scenario.name, seed), index=False)
    make_quotes(HOLDOUT_SIZE, 0.0, HOLDOUT_SEED).to_csv(holdout_path(data_dir), index=False)
