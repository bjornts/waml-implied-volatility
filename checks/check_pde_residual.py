"""Measure the PDE residual before and after a fit that never sees the PDE loss.

Local volatility comes from the same network that defines the price, so the residual
should stay at the level of rounding error however the network is trained. Rounding
depends on how derivatives are computed, so both of the available methods are measured:
nested automatic differentiation, and the closed forms used in training.
"""

import torch

from waml.calibration import ImpliedVolNet, TrainConfig, first_derivative, mesh, surface_terms, train
from waml.data import make_quotes
from waml.pricing import RATE, bs_call, d_plus_minus

SEED = 42
WIDTH = 16
NUM_QUOTES = 300
FIT_STEPS = 3_000


def closed_form_residual(net: ImpliedVolNet, strike: torch.Tensor, tau: torch.Tensor) -> float:
    terms = surface_terms(net, strike, tau)
    return terms["pde"].detach().nan_to_num().pow(2).mean().item()


def autograd_residual(net: ImpliedVolNet, strike: torch.Tensor, tau: torch.Tensor) -> float:
    """Evaluate the Dupire residual with every derivative from nested autograd (eqs. 3, 5)."""
    strike = strike.clone().requires_grad_(True)
    tau = tau.clone().requires_grad_(True)
    vol = net(strike, tau)
    price = bs_call(strike, tau, vol)
    c_k = first_derivative(price, strike)
    c_t = first_derivative(price, tau)
    c_kk = first_derivative(c_k, strike)
    vol_k = first_derivative(vol, strike)
    vol_t = first_derivative(vol, tau)
    vol_kk = first_derivative(vol_k, strike)

    d_plus, d_minus = d_plus_minus(strike, tau, vol)
    numerator = vol**2 + 2.0 * vol * tau * (vol_t + RATE * strike * vol_k)
    denominator = (
        1.0
        + 2.0 * d_plus * strike * torch.sqrt(tau) * vol_k
        + strike**2 * tau * (d_plus * d_minus * vol_k**2 + vol * vol_kk)
    )
    residual = c_t + RATE * strike * c_k - 0.5 * numerator / denominator * strike**2 * c_kk
    return residual.detach().nan_to_num().pow(2).mean().item()


def main() -> None:
    quotes = make_quotes(NUM_QUOTES, 0.0, SEED)
    inputs = [torch.tensor(quotes[column].to_numpy()) for column in ("strike", "tau", "price")]

    for dtype, is_double in ((torch.float32, False), (torch.float64, True)):
        torch.manual_seed(SEED)
        untrained = ImpliedVolNet(WIDTH).to(dtype)
        config = TrainConfig(
            model="data_only", width=WIDTH, steps=FIT_STEPS, double=is_double, seed=SEED
        )
        result = train(*inputs, config)
        strike, tau = mesh(dtype=dtype)
        scale = surface_terms(result.net, strike, tau)["c_t"].detach().nan_to_num().pow(2).mean()

        print(f"{dtype} (quote MSE {result.history[-1]['losses']['quote']:.2e}, C_tau scale {scale:.2e})")
        for label, residual in (("autograd", autograd_residual), ("closed form", closed_form_residual)):
            before = residual(untrained, strike, tau)
            after = residual(result.net, strike, tau)
            print(f"  {label}: residual untrained {before:.2e}, after fit {after:.2e}")


if __name__ == "__main__":
    main()
