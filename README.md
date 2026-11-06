# Whack-a-mole Learning for implied volatility calibration

A PyTorch implementation and evaluation of Whack-a-mole Learning (WamL) from Hoshisashi, Phelan and Barucca, "Whack-a-mole Learning: Physics-Informed Deep Calibration for Implied Volatility Surface" (IEEE CIFEr 2024, [copy here](paper/hoshisashi2024_waml.pdf)), for the CEE 292 implementation project at Stanford.

The report is [`report.pdf`](report.pdf). It makes three main points:

1. **The paper's PDE loss vanishes by construction.** Local volatility comes from the same network that defines the price, so the Dupire residual is zero for any network, up to rounding. The paper reports this loss between -11.4 and -15.2 on a log scale of unstated base. If the base is 10, much of that range is close to single-precision rounding error.
2. **WamL's balancing rule, as written, collapsed every run in this setup.** A near-zero but nonzero gradient produces an enormous loss weight, and the fit is lost soon after.
3. **A one-line gradient floor prevents most collapses.** 19 of 20 runs avoided collapse, but the result does not clearly beat fixed-weight no-arbitrage penalties.

I also tried the update as the lead author's notebook writes it, in my own implementation on noisy data. Starting the updates at step 0 avoided collapse, but two of five seeds got stuck at a poor fit (median IV error 0.046). Balancing on already-weighted losses got stuck at IV error 0.108 on the one seed tested.

These results come from this implementation and synthetic data. The authors' code was not run, and some of the paper's data and settings differ, so they do not show that the published results are wrong.

## The fix

The enhancement is one changed condition in `update_balance_weights` in [`src/waml/calibration.py`](src/waml/calibration.py). The paper resets a loss weight to 1 only when its gradient is exactly zero. The gradient floor also resets it when the gradient is a negligible share of the total:

```diff
- 1.0 if scales[name] == 0.0
+ 1.0 if scales[name] <= floor * total
```

The reported runs use `floor = 1e-3`, as the `waml_floor` model. The default of `0.0` keeps the paper's rule.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Reproduce the results

Datasets are already in `data/`, and the finished runs behind the report are in `results/runs/`.

```bash
pytest                                # Check formulas and training behaviour
python scripts/generate_data.py       # Regenerate the SABR datasets (optional)
python scripts/run_experiments.py --scenarios noisy --precisions single double
python scripts/summarize.py           # Write summary files and the report's table
python scripts/make_figures.py        # Write the report's figure
latexmk -pdf report.tex               # Build the report
```

`run_experiments.py` skips finished runs, so it can be stopped and restarted. A constrained run takes about 15–25 minutes in single precision and 35–55 in double precision on one CPU thread.

## Models

| Name | Loss terms | Balancing |
|---|---|---|
| `data_only` | Quotes | Fixed weights |
| `pinn` | Quotes, PDE | Fixed weights |
| `dcpinn` | Quotes, PDE, no-arbitrage | Fixed weights |
| `dcpinn_waml` | Quotes, PDE, no-arbitrage | WamL, paper's rule |
| `dcpinn_waml_no_pde` | Quotes, no-arbitrage | WamL, paper's rule |
| `dcpinn_waml_step0` | Quotes, PDE, no-arbitrage | WamL, first update at step 0 |
| `waml_floor` | Quotes, no-arbitrage | WamL with the gradient floor |

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | The paper this project implements, included for reference. Its authors and publisher retain their rights to it, and the license below does not cover it |
| `src/waml/` | Pricing, calibration, data generation and evaluation |
| `scripts/` | Generate data, train, summarize and plot |
| `checks/` | Checks behind the report: the PDE residual and one traced WamL collapse |
| `tests/` | Tests of the formulas, derivatives and weight updates |
| `data/` | Quotes for each scenario and seed, plus the held-out test set |
| `results/` | One JSON file per run, summaries and the report's table |

## Data

Quotes are synthetic, generated from Hagan's SABR formula with the paper's parameters `(alpha, beta, rho, nu) = (0.2, 1.0, -0.4, 0.6)` and `r = 0.05`. There are three scenarios across five seeds:

- **Clean:** 300 quotes
- **Noisy:** 300 quotes with 10% multiplicative noise on volatility
- **Sparse:** 50 quotes

The SABR surface itself breaks the butterfly condition at about 0.3% of the evaluation grid. Excluding those points changes violation rates by at most 0.21 percentage points.

## Attribution

This code was written independently of the lead author's [DC-PINNs notebook](https://github.com/khoshisashi/DC-PINNs), which has no license. The notebook was used only as a reference for settings and for its form of the update rule, and none of its code is included.

The code in this repository is released under the [MIT License](LICENSE).
