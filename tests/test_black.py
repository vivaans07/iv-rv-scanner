import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from volscan.pricing.black import (
    black76_greeks,
    black76_price,
    bsm_price,
    implied_vol,
    implied_vol_vec,
    spot_delta,
)


def test_hull_textbook_values() -> None:
    # Hull, Options Futures and Other Derivatives, Example 15.6:
    # S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81.
    call = float(bsm_price(42, 40, 0.5, 0.10, 0.20, True))
    put = float(bsm_price(42, 40, 0.5, 0.10, 0.20, False))
    assert call == pytest.approx(4.76, abs=0.005)
    assert put == pytest.approx(0.81, abs=0.005)


def test_black76_textbook_value() -> None:
    # Hull Example 18.6 (Black's model): F=20, K=20, r=9%, sigma=25%, T=4/12 -> put 1.116.
    put = float(black76_price(20, 20, 4 / 12, 0.09, 0.25, False))
    assert put == pytest.approx(1.116, abs=0.001)


@given(
    F=st.floats(50, 200),
    K=st.floats(40, 250),
    T=st.floats(0.02, 2.0),
    r=st.floats(0.0, 0.08),
    sigma=st.floats(0.05, 1.5),
)
def test_put_call_parity(F: float, K: float, T: float, r: float, sigma: float) -> None:
    c = float(black76_price(F, K, T, r, sigma, True))
    p = float(black76_price(F, K, T, r, sigma, False))
    assert c - p == pytest.approx(math.exp(-r * T) * (F - K), abs=1e-9 * max(F, K))


@settings(max_examples=300)
@given(
    k=st.floats(-0.5, 0.5),
    T=st.floats(7 / 365, 2.0),
    r=st.floats(0.0, 0.08),
    sigma=st.floats(0.05, 2.0),
    is_call=st.booleans(),
)
def test_iv_round_trip(k: float, T: float, r: float, sigma: float, is_call: bool) -> None:
    F = 100.0
    K = F * math.exp(k)
    price = float(black76_price(F, K, T, r, sigma, is_call))
    # When time value is ~1e-6 of the forward the price carries no usable vol
    # information (deep OTM, or deep ITM where price ~= intrinsic).
    intrinsic = math.exp(-r * T) * max((F - K) if is_call else (K - F), 0.0)
    if price - intrinsic < 1e-6 * F:
        return
    iv = implied_vol(price, F, K, T, r, is_call)
    assert not math.isnan(iv)
    assert float(black76_price(F, K, T, r, iv, is_call)) == pytest.approx(price, rel=1e-7)


def test_iv_rejects_arbitrage_prices() -> None:
    F, K, T, r = 100.0, 90.0, 0.5, 0.03
    df = math.exp(-r * T)
    assert math.isnan(implied_vol(df * (F - K) * 0.99, F, K, T, r, True))  # below intrinsic
    assert math.isnan(implied_vol(df * F * 1.01, F, K, T, r, True))  # above forward
    assert math.isnan(implied_vol(0.0, F, K, T, r, False))
    assert math.isnan(implied_vol(-1.0, F, K, T, r, False))


def test_iv_vectorized() -> None:
    K = np.array([80.0, 100.0, 120.0])
    sig = np.array([0.35, 0.25, 0.30])
    calls = np.array([False, True, True])
    px = black76_price(100.0, K, 0.25, 0.04, sig, calls)
    np.testing.assert_allclose(implied_vol_vec(px, 100.0, K, 0.25, 0.04, calls), sig, rtol=1e-8)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("K", [80.0, 100.0, 125.0])
def test_greeks_match_finite_differences(is_call: bool, K: float) -> None:
    F, T, r, s = 100.0, 0.4, 0.03, 0.3
    g = black76_greeks(F, K, T, r, s, is_call)

    def px(F_: float = F, T_: float = T, s_: float = s) -> float:
        return float(black76_price(F_, K, T_, r, s_, is_call))

    h, hs, ht = 1e-3, 1e-4, 1e-5
    assert float(g.delta) == pytest.approx((px(F + h) - px(F - h)) / (2 * h), rel=1e-5)
    assert float(g.gamma) == pytest.approx((px(F + h) - 2 * px() + px(F - h)) / h**2, rel=1e-3)
    assert float(g.vega) == pytest.approx((px(s_=s + hs) - px(s_=s - hs)) / (2 * hs), rel=1e-5)
    assert float(g.theta) == pytest.approx(-(px(T_=T + ht) - px(T_=T - ht)) / (2 * ht), rel=1e-4)

    def delta_at(s_: float) -> float:
        return float(black76_greeks(F, K, T, r, s_, is_call).delta)

    def vega_at(s_: float) -> float:
        return float(black76_greeks(F, K, T, r, s_, is_call).vega)

    assert float(g.vanna) == pytest.approx(
        (delta_at(s + hs) - delta_at(s - hs)) / (2 * hs), rel=1e-4, abs=1e-8
    )
    assert float(g.volga) == pytest.approx(
        (vega_at(s + hs) - vega_at(s - hs)) / (2 * hs), rel=1e-4, abs=1e-6
    )


def test_spot_delta_matches_bsm() -> None:
    S, K, T, r, q, s = 100.0, 105.0, 0.5, 0.05, 0.02, 0.25
    F = S * math.exp((r - q) * T)
    g = black76_greeks(F, K, T, r, s, True)
    h = 1e-3
    fd = (
        float(bsm_price(S + h, K, T, r, s, True, q)) - float(bsm_price(S - h, K, T, r, s, True, q))
    ) / (2 * h)
    assert float(spot_delta(g.delta, F, S)) == pytest.approx(fd, rel=1e-6)
