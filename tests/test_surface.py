from datetime import date

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.fakes import DIV_YIELD, RATE, FakeProvider, smile
from volscan.config import load_settings
from volscan.pricing.black import black76_price
from volscan.surface.cleaning import estimate_forward, filter_quotes, liquidity_check
from volscan.surface.fit import fit_ticker_surface
from volscan.surface.svi import (
    SviParams,
    butterfly_density,
    calendar_violations,
    fit_svi,
    loss_and_grad,
    smile_metrics,
)
from volscan.surface.term import TermStructure

TRUE = SviParams(a=0.01, b=0.08, rho=-0.45, m=0.02, sigma=0.12)


def test_svi_recovers_known_parameters() -> None:
    k = np.linspace(-0.4, 0.3, 30)
    w = TRUE.total_variance(k)
    fit = fit_svi(k, w, T=0.25, restarts=8)
    np.testing.assert_allclose(fit.params.as_tuple(), TRUE.as_tuple(), atol=2e-3)
    assert fit.rmse_vol_pts < 0.01
    assert fit.butterfly_ok


def test_svi_fit_is_robust_to_noise() -> None:
    rng = np.random.default_rng(1)
    k = np.linspace(-0.4, 0.3, 40)
    iv = np.sqrt(TRUE.total_variance(k) / 0.25) + rng.normal(0, 0.003, k.size)
    fit = fit_svi(k, iv**2 * 0.25, T=0.25)
    assert fit.rmse_vol_pts < 0.5
    assert abs(fit.params.rho - TRUE.rho) < 0.15


def test_butterfly_checker_catches_arbitrage() -> None:
    # Huge curvature with tiny minimum variance gives a negative density.
    bad = SviParams(a=-0.02, b=1.5, rho=0.0, m=0.0, sigma=0.02)
    grid = np.linspace(-1, 1, 401)
    assert butterfly_density(bad, grid).min() < 0
    assert butterfly_density(TRUE, grid).min() >= 0


def test_calendar_checker_catches_crossing_slices() -> None:
    near = SviParams(0.04, 0.1, -0.3, 0.0, 0.1)
    far = SviParams(0.02, 0.1, -0.3, 0.0, 0.1)  # lower variance at a longer maturity
    grid = np.linspace(-0.5, 0.5, 51)
    assert calendar_violations([(0.1, near), (0.3, far)], grid)
    assert not calendar_violations([(0.1, far), (0.3, near)], grid)


@given(T=st.floats(0.01, 1.5), k=st.floats(-0.5, 0.5))
def test_term_interpolation_monotone_in_total_variance(T: float, k: float) -> None:
    slices = [
        (0.08, SviParams(0.004, 0.05, -0.4, 0.0, 0.1)),
        (0.25, SviParams(0.012, 0.07, -0.35, 0.0, 0.15)),
        (0.6, SviParams(0.03, 0.09, -0.3, 0.0, 0.2)),
    ]
    ts = TermStructure(slices)
    assert float(ts.total_variance(k, T + 0.01)) >= float(ts.total_variance(k, T)) - 1e-12


def test_constant_maturity_matches_slice_on_node() -> None:
    ts = TermStructure([(30 / 365, TRUE), (90 / 365, SviParams(0.03, 0.08, -0.4, 0.0, 0.15))])
    cm = ts.constant_maturity(30)
    assert cm["atm_vol"] == pytest.approx(float(TRUE.implied_vol(0.0, 30 / 365)))
    m = smile_metrics(TRUE, 30 / 365)
    assert m["rr25"] < 0  # negative rho means puts are richer than calls
    assert cm["rr25"] == pytest.approx(m["rr25"])


def test_forward_from_parity_handles_dividends() -> None:
    S, T = 100.0, 0.5
    F_true = S * np.exp((RATE - DIV_YIELD) * T)
    K = np.arange(90.0, 111.0, 2.5)
    rows = []
    for kind in ("C", "P"):
        px = black76_price(F_true, K, T, RATE, 0.3, kind == "C")
        rows.append(pd.DataFrame({"strike": K, "option_type": kind, "mid": px}))
    F, n = estimate_forward(pd.concat(rows), S * 1.3, T, RATE, n_pairs=5)  # stale spot
    assert pytest.approx(F_true, rel=1e-10) == F
    assert n == 5


def test_filter_quotes_drops_bad_rows() -> None:
    chain = pd.DataFrame(
        {
            "bid": [0.0, 2.0, 1.0, 1.0, 1.0],
            "ask": [0.1, 1.5, 3.0, 1.1, 1.1],
            "open_interest": [100, 100, 100, 0, 100],
        }
    )
    settings = load_settings()
    kept, stats = filter_quotes(chain, settings.cleaning)
    assert len(kept) == 1
    assert (stats.dropped_zero_bid, stats.dropped_crossed, stats.dropped_wide) == (1, 1, 1)
    assert stats.dropped_illiquid == 1


def test_end_to_end_surface_from_fake_chain() -> None:
    asof = date(2026, 3, 2)
    prov = FakeProvider(vol=0.25)
    settings = load_settings()
    expiries = [e for e in prov.list_expiries("AAPL") if 7 <= (e - asof).days <= 180]
    chain = pd.concat([prov.get_chain("AAPL", e, asof) for e in expiries])
    hist = prov.get_history("AAPL", date(2026, 2, 1), asof)
    chain["spot"] = float(hist["close"].iloc[-1])
    res = fit_ticker_surface("AAPL", asof, chain, RATE, settings)
    assert (res.slices["rmse_vol_pts"] < 0.3).all()
    m30 = res.metrics.set_index("tenor_days").loc[30.0]
    assert m30["atm_vol"] == pytest.approx(float(smile(np.array([0.0]), 30 / 365)[0]), abs=0.005)
    assert m30["rr25"] < 0
    ok, why = liquidity_check(chain, chain["spot"].iloc[0], asof, 7, settings.liquidity)
    assert ok, why


def test_svi_analytic_gradient_matches_finite_differences() -> None:
    from scipy.optimize import approx_fprime

    k = np.linspace(-0.4, 0.3, 25)
    target = TRUE.total_variance(k) * 1.1
    wt = np.full(k.size, 1 / k.size)
    x = np.array([0.02, 0.09, -0.3, 0.01, 0.15])
    _, grad = loss_and_grad(x, k, target, wt)
    fd = approx_fprime(x, lambda z: loss_and_grad(z, k, target, wt)[0], 1e-8)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-9)
