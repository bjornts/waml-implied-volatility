"""Black-Scholes pricing and the SABR implied-volatility formula.

All functions are written in PyTorch so that derivatives with respect to strike
and maturity can be taken by automatic differentiation. Spot is fixed at
``SPOT = 1`` throughout, so strikes are quoted as a multiple of spot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

SPOT = 1.0
RATE = 0.05
_FLOOR = 1e-15


def normal_cdf(x: Tensor) -> Tensor:
    """Standard normal CDF, built on ``erf`` so it is differentiable to any order."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def d_plus_minus(
    strike: Tensor, tau: Tensor, vol: Tensor, rate: float = RATE
) -> tuple[Tensor, Tensor]:
    """Return ``(d_plus, d_minus)`` from the Black-Scholes formula."""
    vol = torch.clamp(vol, min=_FLOOR)
    root_tau = torch.sqrt(tau)
    d_plus = (torch.log(SPOT / strike) + (rate + 0.5 * vol**2) * tau) / (vol * root_tau)
    return d_plus, d_plus - vol * root_tau


def bs_call(strike: Tensor, tau: Tensor, vol: Tensor, rate: float = RATE) -> Tensor:
    """European call price under Black-Scholes (paper eq. 2)."""
    d_plus, d_minus = d_plus_minus(strike, tau, vol, rate)
    discount = torch.exp(-rate * tau)
    return SPOT * normal_cdf(d_plus) - discount * strike * normal_cdf(d_minus)


@dataclass(frozen=True)
class SabrParams:
    """SABR parameters. Defaults are the values in the paper's synthetic experiment."""

    alpha: float = 0.2
    beta: float = 1.0
    rho: float = -0.4
    nu: float = 0.6


def sabr_implied_vol(
    strike: Tensor, tau: Tensor, params: SabrParams = SabrParams(), rate: float = RATE
) -> Tensor:
    """Hagan et al. (2002) implied volatility with forward ``SPOT * exp(rate * tau)``."""
    a, b, rho, nu = params.alpha, params.beta, params.rho, params.nu
    forward = SPOT * torch.exp(rate * tau)
    fk_power = (forward * strike) ** ((1.0 - b) / 2.0)
    log_fk = torch.log(forward / strike)

    z = nu / a * fk_power * log_fk
    chi = torch.log((torch.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))
    # z / chi(z) tends to 1 as z tends to 0. Substituting before dividing keeps
    # the expression free of 0/0, including in its derivatives.
    small = z.abs() < 1e-6
    safe_chi = torch.where(small, torch.ones_like(chi), chi)
    z_over_chi = torch.where(small, torch.ones_like(z), z / safe_chi)

    denominator = fk_power * (
        1.0 + (1.0 - b) ** 2 / 24.0 * log_fk**2 + (1.0 - b) ** 4 / 1920.0 * log_fk**4
    )
    time_correction = 1.0 + tau * (
        (1.0 - b) ** 2 / 24.0 * a**2 / fk_power**2
        + 0.25 * rho * b * nu * a / fk_power
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    )
    return a / denominator * z_over_chi * time_correction
