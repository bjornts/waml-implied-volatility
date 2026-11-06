"""Physics-informed calibration of an implied volatility surface with WamL.

Implements the losses and the two weight updates ("whacks") from Hoshisashi,
Phelan and Barucca (2024). Equation numbers refer to that paper.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .pricing import RATE, SPOT, bs_call, d_plus_minus, normal_cdf

# Loss terms used by each model in the ablation.
MODEL_TERMS = {
    "data_only": ("quote",),
    "pinn": ("quote", "pde"),
    "dcpinn": ("quote", "pde", "dK", "dKK", "dT"),
    "dcpinn_waml": ("quote", "pde", "dK", "dKK", "dT"),
    "dcpinn_waml_no_pde": ("quote", "dK", "dKK", "dT"),
    "dcpinn_waml_step0": ("quote", "pde", "dK", "dKK", "dT"),
    "waml_floor": ("quote", "dK", "dKK", "dT"),
}
WAML_MODELS = {"dcpinn_waml", "dcpinn_waml_no_pde", "dcpinn_waml_step0", "waml_floor"}


class ImpliedVolNet(nn.Module):
    """MLP mapping ``(strike, tau)`` to a positive implied volatility (eq. 21)."""

    def __init__(self, width: int = 64, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        size = 2
        for _ in range(depth):
            layers += [nn.Linear(size, width), nn.Tanh()]
            size = width
        layers += [nn.Linear(size, 1), nn.Softplus()]
        self.layers = nn.Sequential(*layers)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, strike: Tensor, tau: Tensor) -> Tensor:
        return self.layers(torch.stack([strike, tau], dim=1)).squeeze(1)


def first_derivative(output: Tensor, wrt: Tensor) -> Tensor:
    """Pointwise derivative of ``output``, keeping the graph for higher derivatives."""
    return torch.autograd.grad(output.sum(), wrt, create_graph=True)[0]


def mesh(n: int = 101, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
    """Rectangular grid over the paper's domain, offset from the singular K=0, tau=0."""
    offset = 1e-3
    strike = torch.linspace(offset, 2.5 + offset, n, dtype=dtype)
    tau = torch.linspace(offset, 5.0 + offset, n, dtype=dtype)
    grid_k, grid_t = torch.meshgrid(strike, tau, indexing="ij")
    return grid_k.flatten(), grid_t.flatten()


def vol_with_derivatives(
    net: ImpliedVolNet, strike: Tensor, tau: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``vol, vol_K, vol_tau, vol_KK`` by forward propagation through the layers."""
    value = torch.stack([strike, tau], dim=1)
    d_k = torch.zeros_like(value)
    d_k[:, 0] = 1.0
    d_t = torch.zeros_like(value)
    d_t[:, 1] = 1.0
    d_kk = torch.zeros_like(value)

    for layer in net.layers:
        if isinstance(layer, nn.Linear):
            value = layer(value)
            d_k = d_k @ layer.weight.T
            d_t = d_t @ layer.weight.T
            d_kk = d_kk @ layer.weight.T
            continue
        if isinstance(layer, nn.Tanh):
            activated = torch.tanh(value)
            first = 1.0 - activated**2
            second = -2.0 * activated * first
        elif isinstance(layer, nn.Softplus):
            activated = F.softplus(value)
            first = torch.sigmoid(value)
            second = first * (1.0 - first)
        else:
            raise TypeError(f"Unsupported layer: {layer}")
        d_kk = first * d_kk + second * d_k**2
        d_k = first * d_k
        d_t = first * d_t
        value = activated
    return value.squeeze(1), d_k.squeeze(1), d_t.squeeze(1), d_kk.squeeze(1)


def surface_terms(
    net: nn.Module, strike: Tensor, tau: Tensor, rate: float = RATE
) -> dict[str, Tensor]:
    """Evaluate price derivatives, the PDE residual and inequality violations.

    Nested reverse-mode differentiation is too slow to repeat at every training step,
    so derivatives combine ``vol_with_derivatives`` with closed-form Black-Scholes
    sensitivities through the chain rule. ``tests/test_waml.py`` checks them against
    nested automatic differentiation.
    """
    vol, vol_k, vol_t, vol_kk = vol_with_derivatives(net, strike, tau)
    price = bs_call(strike, tau, vol, rate)
    d_plus, d_minus = d_plus_minus(strike, tau, vol, rate)
    discount = torch.exp(-rate * tau)
    root_tau = torch.sqrt(tau)
    safe_vol = torch.clamp(vol, min=1e-15)
    density = torch.exp(-0.5 * d_plus**2) / math.sqrt(2.0 * math.pi)  # phi(d_plus)
    vega = SPOT * density * root_tau

    # Partial derivatives of the Black-Scholes price at fixed volatility.
    bs_k = -discount * normal_cdf(d_minus)
    bs_t = SPOT * density * vol / (2.0 * root_tau) - rate * strike * bs_k
    bs_kk = SPOT * density / (strike**2 * safe_vol * root_tau)
    bs_k_vol = SPOT * density * d_plus / (strike * safe_vol)
    bs_vol_vol = vega * d_plus * d_minus / safe_vol

    # Total derivatives along the surface vol(K, tau).
    c_k = bs_k + vega * vol_k
    c_t = bs_t + vega * vol_t
    c_kk = bs_kk + 2.0 * bs_k_vol * vol_k + bs_vol_vol * vol_k**2 + vega * vol_kk

    # Local variance from implied volatility (eq. 5).
    numerator = vol**2 + 2.0 * vol * tau * (vol_t + rate * strike * vol_k)
    denominator = (
        1.0
        + 2.0 * d_plus * strike * root_tau * vol_k
        + strike**2 * tau * (d_plus * d_minus * vol_k**2 + vol * vol_kk)
    )
    local_var = numerator / denominator

    # Dupire residual (eq. 3). Violations are the amount outside each inequality (eq. 7, 13).
    pde = c_t + rate * strike * c_k - 0.5 * local_var * strike**2 * c_kk
    return {
        "c_k": c_k,
        "c_t": c_t,
        "c_kk": c_kk,
        "pde": pde,
        "dK": torch.relu(c_k) + torch.relu(-discount - c_k),
        "dKK": torch.relu(-c_kk),
        "dT": torch.relu(-c_t),
        "vol": vol,
        "price": price,
    }


@dataclass
class TrainConfig:
    """Training settings. Defaults follow paper section VII-A."""

    model: str = "dcpinn_waml"
    width: int = 64
    depth: int = 4
    steps: int = 10_000
    learning_rate: float = 1e-3
    decay_rate: float = 0.9
    decay_steps: int = 1_000
    eta_m: float = 1.0
    update_every: int = 100
    mesh_points: int = 101
    double: bool = False
    seed: int = 0
    log_every: int = 500
    # Whether the balance and point-weight updates use the already-weighted loss, as the
    # author's notebook does, or the loss before balancing, as the paper's equations do.
    weighted_updates: bool = False
    # Step of the first balance update: the paper's Algorithm 1 starts at k=100,
    # the notebook at 0.
    first_update: int = 100
    # Steps by which point-weight updates lag balance updates (the notebook uses 50).
    point_weight_lag: int = 0
    # A loss whose gradient scale is at most this share of the total counts as having no
    # gradient and resets to 1. Zero is the paper's rule, which only catches exact zeros.
    balance_floor: float = 0.0


@dataclass
class TrainResult:
    """A trained network, its training history and any numerical failures seen."""

    net: nn.Module
    seconds: float
    history: list[dict] = field(default_factory=list)
    nonfinite_points: int = 0


def _gradient_scale(loss: Tensor, params: list[Tensor]) -> Tensor:
    """Mean absolute gradient across all network parameters (the |.| in eq. 20)."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = [
        torch.nan_to_num(g).abs().flatten() if g is not None else torch.zeros_like(p).flatten()
        for g, p in zip(grads, params)
    ]
    return torch.cat(flat).mean()


def update_balance_weights(
    weights: dict[str, float], scales: dict[str, float], floor: float = 0.0
) -> dict[str, float]:
    """Loss-balancing update (eq. 20). A term with no usable gradient resets to 1."""
    total = sum(scales.values())
    return {
        name: 1.0
        if scales[name] <= floor * total
        else 0.5 * (weights[name] + total / scales[name])
        for name in weights
    }


def train(
    strike: Tensor, tau: Tensor, target: Tensor, config: TrainConfig
) -> TrainResult:
    """Fit the surface to quotes using the loss terms selected by ``config.model``."""
    dtype = torch.float64 if config.double else torch.float32
    strike, tau, target = strike.to(dtype), tau.to(dtype), target.to(dtype)

    torch.manual_seed(config.seed)
    net = ImpliedVolNet(config.width, config.depth).to(dtype)
    params = list(net.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: config.decay_rate ** (step / config.decay_steps),
    )

    terms = MODEL_TERMS[config.model]
    use_waml = config.model in WAML_MODELS
    uses_mesh = terms != ("quote",)
    grid_k, grid_t = mesh(config.mesh_points, dtype)

    sizes = {name: (len(target) if name == "quote" else len(grid_k)) for name in terms}
    point_weights = {name: torch.ones(sizes[name], dtype=dtype) for name in terms}
    balance = {name: 1.0 for name in terms}

    history: list[dict] = []
    nonfinite_points = 0
    started = time.perf_counter()

    for step in range(config.steps + 1):
        price = bs_call(strike, tau, net(strike, tau))
        residuals = {"quote": price - target}
        if uses_mesh:
            surface = surface_terms(net, grid_k, grid_t)
            for name in terms:
                if name != "quote":
                    raw = surface[name]
                    bad = ~torch.isfinite(raw)
                    nonfinite_points += int(bad.sum())
                    residuals[name] = torch.where(bad, torch.zeros_like(raw), raw)

        losses = {
            name: (point_weights[name] * residuals[name] ** 2).mean() for name in terms
        }

        if use_waml:
            multiplier = balance if config.weighted_updates else dict.fromkeys(terms, 1.0)
            # First whack: gradient ascent on the per-point weights (eq. 19).
            lagged = step - config.point_weight_lag
            if lagged >= config.first_update and lagged % config.update_every == 0:
                for name in terms:
                    squared = residuals[name].detach() ** 2
                    point_weights[name] = (
                        point_weights[name]
                        + config.eta_m * multiplier[name] * squared / sizes[name]
                    )
            # Second whack: rebalance losses by inverse gradient scale (eq. 20).
            if step >= config.first_update and step % config.update_every == 0:
                scales = {
                    name: float(_gradient_scale(multiplier[name] * losses[name], params))
                    for name in terms
                }
                balance = update_balance_weights(balance, scales, config.balance_floor)

        logging = step % config.log_every == 0 or step == config.steps
        if logging:
            history.append(
                {
                    "step": step,
                    "losses": {
                        name: float((residuals[name].detach() ** 2).mean()) for name in terms
                    },
                    "balance": dict(balance),
                    "point_weight_max": {
                        name: float(point_weights[name].max()) for name in terms
                    },
                }
            )
        if step == config.steps:
            break

        total = sum(balance[name] * losses[name] for name in terms)
        optimizer.zero_grad()
        total.backward()
        if logging:
            history[-1]["grad_mean_abs"] = float(
                torch.cat([torch.nan_to_num(p.grad).abs().flatten() for p in params]).mean()
            )
        for param in params:
            if param.grad is not None:
                torch.nan_to_num_(param.grad)
        optimizer.step()
        scheduler.step()

    return TrainResult(net, time.perf_counter() - started, history, nonfinite_points)
