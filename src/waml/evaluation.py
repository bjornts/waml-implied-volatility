"""Metrics for a calibrated surface.

Everything is evaluated in double precision on a grid twice as fine as the training
mesh, so results are comparable across models regardless of training precision.
"""

from __future__ import annotations

import copy

import pandas as pd
import torch
from torch import Tensor, nn

from .calibration import first_derivative, mesh, surface_terms
from .pricing import RATE, SabrParams, bs_call, d_plus_minus, sabr_implied_vol

EVAL_MESH_POINTS = 201
VIOLATION_TOLERANCE = 1e-8
MIN_VEGA = 1e-3


def _vega(strike: Tensor, tau: Tensor, vol: Tensor) -> Tensor:
    d_plus, _ = d_plus_minus(strike, tau, vol)
    density = torch.exp(-0.5 * d_plus**2) / (2.0 * torch.pi) ** 0.5
    return density * torch.sqrt(tau)


def _rmse(error: Tensor) -> float:
    return float(error.pow(2).mean().sqrt())


def surface_violations(
    strike: Tensor, tau: Tensor, vol_fn, rate: float = RATE
) -> dict[str, Tensor]:
    """Boolean masks of points that violate each no-arbitrage condition (eq. 7)."""
    strike = strike.detach().requires_grad_(True)
    tau = tau.detach().requires_grad_(True)
    price = bs_call(strike, tau, vol_fn(strike, tau), rate)
    c_k = first_derivative(price, strike)
    c_t = first_derivative(price, tau)
    c_kk = first_derivative(c_k, strike)
    discount = torch.exp(-rate * tau)
    masks = {
        "dK": (c_k > VIOLATION_TOLERANCE) | (c_k < -discount - VIOLATION_TOLERANCE),
        "dKK": c_kk < -VIOLATION_TOLERANCE,
        "dT": c_t < -VIOLATION_TOLERANCE,
    }
    return {name: mask.detach() for name, mask in masks.items()}


def generating_surface_corner(
    strike: Tensor, tau: Tensor, params: SabrParams = SabrParams()
) -> Tensor:
    """Points where the generating SABR surface itself admits butterfly arbitrage."""
    masks = surface_violations(strike, tau, lambda k, t: sabr_implied_vol(k, t, params))
    return masks["dKK"]


def evaluate(net: nn.Module, holdout: pd.DataFrame) -> dict[str, float]:
    """Compute the report's held-out errors, violation rates and PDE residual."""
    net = copy.deepcopy(net).to(torch.float64)
    strike = torch.tensor(holdout["strike"].to_numpy(), dtype=torch.float64)
    tau = torch.tensor(holdout["tau"].to_numpy(), dtype=torch.float64)
    true_vol = torch.tensor(holdout["true_vol"].to_numpy(), dtype=torch.float64)
    true_price = torch.tensor(holdout["true_price"].to_numpy(), dtype=torch.float64)

    with torch.nofirst_derivative():
        vol = net(strike, tau)
        price = bs_call(strike, tau, vol)
    # Implied volatility is unobservable where an option has no time value, so those
    # points say nothing about calibration quality and are reported separately.
    identifiable = _vega(strike, tau, true_vol) >= MIN_VEGA

    grid_k, grid_t = mesh(EVAL_MESH_POINTS, torch.float64)
    corner = generating_surface_corner(grid_k, grid_t)
    masks = surface_violations(grid_k, grid_t, net)
    any_violation = masks["dK"] | masks["dKK"] | masks["dT"]

    with torch.nofirst_derivative():
        grid_vol = net(grid_k, grid_t)
    grid_true = sabr_implied_vol(grid_k, grid_t)
    grid_identifiable = _vega(grid_k, grid_t, grid_true) >= MIN_VEGA
    surface = surface_terms(net, grid_k, grid_t)
    pde = torch.nan_to_num(surface["pde"].detach())

    return {
        "price_rmse": _rmse(price - true_price),
        "iv_rmse_all": _rmse(vol - true_vol),
        "iv_rmse": _rmse((vol - true_vol)[identifiable]),
        "holdout_identifiable_share": float(identifiable.float().mean()),
        "mesh_iv_rmse": _rmse((grid_vol - grid_true)[grid_identifiable]),
        "mesh_iv_rmse_excl_corner": _rmse((grid_vol - grid_true)[grid_identifiable & ~corner]),
        "corner_iv_rmse": _rmse((grid_vol - grid_true)[corner]) if corner.any() else 0.0,
        "viol_dK": 100.0 * float(masks["dK"].double().mean()),
        "viol_dKK": 100.0 * float(masks["dKK"].double().mean()),
        "viol_dT": 100.0 * float(masks["dT"].double().mean()),
        "viol_any": 100.0 * float(any_violation.double().mean()),
        "viol_any_excl_corner": 100.0 * float(any_violation[~corner].double().mean()),
        "corner_share": 100.0 * float(corner.double().mean()),
        "pde_mse_double": float(pde.pow(2).mean()),
    }

