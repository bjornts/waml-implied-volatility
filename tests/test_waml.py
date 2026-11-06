"""Checks of the pricing formulas, constraints and WamL updates against known results."""

import pytest
import torch

from waml.calibration import (
    MODEL_TERMS,
    ImpliedVolNet,
    TrainConfig,
    mesh,
    surface_terms,
    train,
    update_balance_weights,
    vol_with_derivatives,
)
from waml.data import make_quotes
from waml.evaluation import generating_surface_corner, surface_violations
from waml.pricing import RATE, SPOT, SabrParams, bs_call, d_plus_minus, sabr_implied_vol

torch.set_default_dtype(torch.float64)


def test_call_price_matches_reference_value():
    # Hull-style reference: S=K=1, r=5%, vol=20%, T=1 gives 0.104506.
    price = bs_call(torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([0.2]), 0.05)
    assert price.item() == pytest.approx(0.1045058, abs=1e-6)


def test_call_price_tends_to_payoff_and_limits():
    # These are the paper's boundary conditions (eq. 4). They hold for any positive
    # volatility, which is why no separate boundary loss is needed.
    vol = torch.tensor([0.3])
    short = torch.tensor([1e-6])
    assert bs_call(torch.tensor([0.8]), short, vol).item() == pytest.approx(0.2, abs=1e-5)
    assert bs_call(torch.tensor([1.2]), short, vol).item() == pytest.approx(0.0, abs=1e-5)
    tau = torch.tensor([1.0])
    assert bs_call(torch.tensor([1e4]), tau, vol).item() == pytest.approx(0.0, abs=1e-8)
    assert bs_call(torch.tensor([1e-6]), tau, vol).item() == pytest.approx(SPOT, abs=1e-5)


def test_sabr_at_the_money_matches_hagan_limit():
    tau = torch.tensor([2.0])
    forward = SPOT * torch.exp(RATE * tau)
    p = SabrParams()
    expected = p.alpha * (1 + (0.25 * p.rho * p.beta * p.nu * p.alpha + (2 - 3 * p.rho**2) / 24 * p.nu**2) * 2.0)
    assert sabr_implied_vol(forward, tau).item() == pytest.approx(expected, rel=1e-10)


def test_sabr_is_continuous_across_the_at_the_money_switch():
    tau = torch.tensor([1.0, 1.0])
    forward = (SPOT * torch.exp(RATE * tau[0])).item()
    vols = sabr_implied_vol(torch.tensor([forward * (1 - 1e-7), forward * (1 + 1e-7)]), tau)
    assert abs(vols[0] - vols[1]).item() < 1e-6


def test_constant_vol_surface_has_no_violations():
    k, t = mesh(41, torch.float64)
    masks = surface_violations(k, t, lambda s, u: 0.2 * torch.ones_like(s))
    assert not any(mask.any() for mask in masks.values())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_derivatives_match_nested_autograd(seed):
    torch.manual_seed(seed)
    net = ImpliedVolNet(16)
    k, t = mesh(31, torch.float64)
    fast = surface_terms(net, k, t)

    def grad(output, wrt):
        return torch.autograd.grad(output.sum(), wrt, create_graph=True)[0]

    k = k.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    price = bs_call(k, t, net(k, t))
    c_k = grad(price, k)
    c_t = grad(price, t)
    c_kk = grad(c_k, k)
    assert torch.allclose(fast["price"], price, atol=1e-12)
    assert torch.allclose(fast["c_k"], c_k, atol=1e-9)
    assert torch.allclose(fast["c_t"], c_t, atol=1e-9)
    assert torch.allclose(fast["c_kk"], c_kk, atol=1e-7)


def test_pde_residual_is_zero_for_an_untrained_network():
    torch.manual_seed(0)
    net = ImpliedVolNet(16)
    k, t = mesh(41, torch.float64)
    terms = surface_terms(net, k, t)
    assert terms["pde"].pow(2).mean().item() < 1e-26
    assert terms["price"].std().item() > 0.01


def test_price_derivatives_factor_into_the_local_variance_terms():
    # The cancellation in the report: both parts of the residual equal V / (2 sigma tau)
    # times the numerator and denominator of the local variance formula (eq. 5).
    torch.manual_seed(0)
    net = ImpliedVolNet(16).to(torch.float64)
    k, t = mesh(41, torch.float64)
    terms = surface_terms(net, k, t)
    vol, vol_k, vol_t, vol_kk = vol_with_derivatives(net, k, t)
    d_plus, d_minus = d_plus_minus(k, t, vol)

    numerator = vol**2 + 2.0 * vol * t * (vol_t + RATE * k * vol_k)
    denominator = (
        1.0
        + 2.0 * d_plus * k * t.sqrt() * vol_k
        + k**2 * t * (d_plus * d_minus * vol_k**2 + vol * vol_kk)
    )
    density = torch.exp(-0.5 * d_plus**2) / (2.0 * torch.pi) ** 0.5
    common = SPOT * density * t.sqrt() / (2.0 * vol * t)  # V / (2 sigma tau)

    left = terms["c_t"] + RATE * k * terms["c_k"]
    right = 0.5 * k**2 * terms["c_kk"]
    assert torch.allclose(left, common * numerator, atol=1e-10)
    assert torch.allclose(right, common * denominator, atol=1e-10)


def test_pde_residual_is_nonzero_with_an_independent_local_vol():
    # The residual is only trivial because local vol comes from the same network.
    # A wrong local variance has to produce a large residual, or the check means nothing.
    torch.manual_seed(0)
    net = ImpliedVolNet(16)
    k, t = mesh(41, torch.float64)
    k = k.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    price = bs_call(k, t, net(k, t))
    c_k = torch.autograd.grad(price.sum(), k, create_graph=True)[0]
    c_t = torch.autograd.grad(price.sum(), t, create_graph=True)[0]
    c_kk = torch.autograd.grad(c_k.sum(), k, create_graph=True)[0]
    wrong = c_t + RATE * k * c_k - 0.5 * 0.04 * k**2 * c_kk
    assert wrong.pow(2).mean().item() > 1e-4


def test_generating_surface_has_arbitrage_only_at_low_strikes():
    k, t = mesh(101, torch.float64)
    corner = generating_surface_corner(k, t)
    assert corner.any()
    assert k[corner].max().item() < 0.05


def test_balance_update_matches_equation_20():
    updated = update_balance_weights({"a": 1.0, "b": 3.0}, {"a": 1.0, "b": 3.0})
    assert updated["a"] == pytest.approx(0.5 * (1.0 + 4.0 / 1.0))
    assert updated["b"] == pytest.approx(0.5 * (3.0 + 4.0 / 3.0))


def test_balance_update_resets_zero_gradient_terms_to_one():
    assert update_balance_weights({"a": 7.0, "b": 1.0}, {"a": 0.0, "b": 2.0})["a"] == 1.0


def test_balance_floor_treats_a_negligible_gradient_as_zero():
    scales = {"a": 1.0, "b": 1e-20}
    assert update_balance_weights({"a": 1.0, "b": 1.0}, scales)["b"] > 1e19
    assert update_balance_weights({"a": 1.0, "b": 1.0}, scales, floor=1e-3)["b"] == 1.0


def test_quote_noise_changes_price_but_not_truth():
    clean, noisy = make_quotes(50, 0.0, 1), make_quotes(50, 0.1, 1)
    assert (clean["price"] - clean["true_price"]).abs().max() == 0.0
    assert (noisy["price"] - noisy["true_price"]).abs().max() > 0.0
    assert noisy["true_vol"].equals(clean["true_vol"])


@pytest.mark.parametrize("model", list(MODEL_TERMS))
def test_every_model_trains_and_lowers_the_quote_loss(model):
    quotes = make_quotes(40, 0.0, 3)
    config = TrainConfig(model=model, width=8, steps=40, mesh_points=11, log_every=40, update_every=10)
    result = train(
        torch.tensor(quotes["strike"].to_numpy()),
        torch.tensor(quotes["tau"].to_numpy()),
        torch.tensor(quotes["price"].to_numpy()),
        config,
    )
    assert result.history[-1]["losses"]["quote"] < result.history[0]["losses"]["quote"]


def test_first_balance_update_waits_for_the_first_interval():
    quotes = make_quotes(20, 0.0, 3)
    inputs = [torch.tensor(quotes[c].to_numpy()) for c in ("strike", "tau", "price")]
    config = TrainConfig(model="dcpinn_waml", width=8, steps=20, mesh_points=11, first_update=10, update_every=10, log_every=5)
    history = train(*inputs, config).history
    assert all(value == 1.0 for value in history[1]["balance"].values())
    assert history[2]["balance"]["quote"] != 1.0


def test_training_is_reproducible_for_a_fixed_seed():
    quotes = make_quotes(20, 0.0, 3)
    inputs = [torch.tensor(quotes[c].to_numpy()) for c in ("strike", "tau", "price")]
    config = TrainConfig(model="dcpinn_waml", width=8, steps=20, mesh_points=11, update_every=5)
    first, second = train(*inputs, config), train(*inputs, config)
    assert first.history[-1]["losses"] == second.history[-1]["losses"]
