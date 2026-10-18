"""Black-76 pricing, Greeks, and an implied-vol solver.

Everything is written on the forward F rather than spot. For listed equity
options I estimate F per expiry from put-call parity, which absorbs dividends
and borrow without having to model them. `bsm_*` wrappers convert a spot,
rate, and dividend yield into a forward for textbook checks.

Conventions:
    T      time to expiry in years (calendar, ACT/365)
    r      continuously compounded risk-free rate
    sigma  annualized Black volatility
    df     discount factor exp(-r T)

Greeks are per one unit of the option (multiply by contract size elsewhere).
Vega is per 1.00 of vol (divide by 100 for "per vol point").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import ndtr

FloatArray = NDArray[np.float64]

IV_MIN = 1e-4
IV_MAX = 5.0
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _npdf(x: FloatArray) -> FloatArray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _arr(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _d1_d2(
    F: FloatArray, K: FloatArray, T: FloatArray, sigma: FloatArray
) -> tuple[FloatArray, FloatArray]:
    sqrt_t = np.sqrt(T)
    vol_t = sigma * sqrt_t
    d1 = (np.log(F / K) + 0.5 * vol_t * vol_t) / vol_t
    return d1, d1 - vol_t


def black76_price(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, is_call: ArrayLike
) -> FloatArray:
    """Discounted Black-76 price of a European call or put on a forward."""
    F_, K_, T_, r_, s_ = _arr(F), _arr(K), _arr(T), _arr(r), _arr(sigma)
    call = np.asarray(is_call, dtype=bool)
    df = np.exp(-r_ * T_)
    d1, d2 = _d1_d2(F_, K_, T_, s_)
    call_px = df * (F_ * ndtr(d1) - K_ * ndtr(d2))
    put_px = df * (K_ * ndtr(-d2) - F_ * ndtr(-d1))
    return np.asarray(np.where(call, call_px, put_px), dtype=np.float64)


@dataclass(frozen=True)
class Greeks:
    """Greeks of a Black-76 option.

    delta and gamma are with respect to the forward. Use `spot_delta` /
    `spot_gamma` to convert when hedging with shares.
    theta is the change in value per year of calendar time passing, holding
    F and sigma fixed (divide by 365 for a daily number).
    """

    price: FloatArray
    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray
    vanna: FloatArray
    volga: FloatArray


def black76_greeks(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, is_call: ArrayLike
) -> Greeks:
    F_, K_, T_, r_, s_ = _arr(F), _arr(K), _arr(T), _arr(r), _arr(sigma)
    call = np.asarray(is_call, dtype=bool)
    df = np.exp(-r_ * T_)
    sqrt_t = np.sqrt(T_)
    d1, d2 = _d1_d2(F_, K_, T_, s_)
    pdf = _npdf(d1)

    price = black76_price(F_, K_, T_, r_, s_, call)
    delta = np.where(call, df * ndtr(d1), -df * ndtr(-d1))
    gamma = df * pdf / (F_ * s_ * sqrt_t)
    vega = df * F_ * pdf * sqrt_t
    # dV/dT with F fixed: discounting term plus the time value decay.
    dv_dt = -r_ * price + df * F_ * pdf * s_ / (2.0 * sqrt_t)
    theta = -dv_dt
    vanna = -df * pdf * d2 / s_
    volga = vega * d1 * d2 / s_
    return Greeks(
        price=price,
        delta=np.asarray(delta, dtype=np.float64),
        gamma=gamma,
        vega=vega,
        theta=theta,
        vanna=vanna,
        volga=volga,
    )


def spot_delta(forward_delta: ArrayLike, F: ArrayLike, S: ArrayLike) -> FloatArray:
    """dV/dS given dV/dF, assuming F/S is fixed over the hedge interval."""
    return _arr(forward_delta) * _arr(F) / _arr(S)


def spot_gamma(forward_gamma: ArrayLike, F: ArrayLike, S: ArrayLike) -> FloatArray:
    ratio = _arr(F) / _arr(S)
    return _arr(forward_gamma) * ratio * ratio


def bsm_forward(S: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike = 0.0) -> FloatArray:
    return _arr(S) * np.exp((_arr(r) - _arr(q)) * _arr(T))


def bsm_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike,
    q: ArrayLike = 0.0,
) -> FloatArray:
    """Black-Scholes-Merton price via the equivalent forward."""
    return black76_price(bsm_forward(S, T, r, q), K, T, r, sigma, is_call)


def no_arb_bounds(F: float, K: float, T: float, r: float, is_call: bool) -> tuple[float, float]:
    """Model-free bounds on a European option price."""
    df = math.exp(-r * T)
    if is_call:
        return df * max(F - K, 0.0), df * F
    return df * max(K - F, 0.0), df * K


def _initial_guess(price: float, F: float, K: float, T: float, df: float) -> float:
    # Brenner-Subrahmanyam ATM approximation, widened by moneyness.
    atm = price / (df * F) * _SQRT_2PI / math.sqrt(T)
    k = abs(math.log(F / K))
    guess = math.sqrt(atm * atm + 2.0 * k / T)
    return min(max(guess, 0.05), 2.0)


def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> float:
    """Black-76 implied vol. Returns NaN if the price is outside no-arb bounds.

    Newton on vega converges in a handful of steps near the money. Far wings
    have tiny vega and Newton can overshoot, so any failure falls back to
    Brent on [IV_MIN, IV_MAX], which always converges if a root exists.
    """
    if not (price > 0 and F > 0 and K > 0 and T > 0) or not math.isfinite(price):
        return math.nan
    lo, hi = no_arb_bounds(F, K, T, r, is_call)
    if price <= lo or price >= hi:
        return math.nan

    df = math.exp(-r * T)

    def objective(s: float) -> float:
        return float(black76_price(F, K, T, r, s, is_call)) - price

    sigma = _initial_guess(price, F, K, T, df)
    for _ in range(max_iter):
        diff = objective(sigma)
        if abs(diff) < tol * max(price, 1e-8):
            return sigma
        d1, _ = _d1_d2(_arr(F), _arr(K), _arr(T), _arr(sigma))
        vega = float(df * F * _npdf(d1) * math.sqrt(T))
        if vega < 1e-12:
            break
        step = diff / vega
        new_sigma = sigma - step
        if not (IV_MIN < new_sigma < IV_MAX):
            break
        sigma = new_sigma

    f_lo, f_hi = objective(IV_MIN), objective(IV_MAX)
    if f_lo * f_hi > 0:
        return math.nan
    root: float = brentq(objective, IV_MIN, IV_MAX, xtol=1e-12, maxiter=200)
    return root


def implied_vol_vec(
    price: ArrayLike, F: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, is_call: ArrayLike
) -> FloatArray:
    """Element-wise `implied_vol` over broadcast arrays."""
    p, f, k, t, rr, c = np.broadcast_arrays(
        _arr(price), _arr(F), _arr(K), _arr(T), _arr(r), np.asarray(is_call, dtype=bool)
    )
    out = np.empty(p.shape, dtype=np.float64)
    for idx in np.ndindex(p.shape):
        out[idx] = implied_vol(
            float(p[idx]), float(f[idx]), float(k[idx]), float(t[idx]), float(rr[idx]), bool(c[idx])
        )
    return out
