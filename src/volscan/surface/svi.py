"""Raw SVI smile fits and static-arbitrage checks.

Raw SVI (Gatheral 2004) models total implied variance w = sigma_imp^2 * T as a
function of log-moneyness k = ln(K/F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

Fitting strategy:
  1. For fixed (m, sigma) the model is linear in (a, b*rho*sigma, b*sigma)
     (the quasi-explicit idea of Zeliade Systems, 2009). A small grid over
     (m, sigma) with weighted linear least squares gives good starting points
     cheaply.
  2. The best starts are polished with bounded L-BFGS-B on a reparameterized
     problem where a is replaced by the minimum total variance
     v = a + b*sigma*sqrt(1 - rho^2) >= 0. That turns the non-negativity
     constraint into a simple box bound.

The fit does not force the slice to be arbitrage-free. Butterfly and calendar
violations are measured and reported instead, so they stay visible.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize
from scipy.stats import norm

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SviParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: FloatArray | float) -> FloatArray:
        x = np.asarray(k, dtype=np.float64) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma * self.sigma))

    def dw(self, k: FloatArray | float) -> FloatArray:
        x = np.asarray(k, dtype=np.float64) - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.sigma * self.sigma))

    def d2w(self, k: FloatArray | float) -> FloatArray:
        x = np.asarray(k, dtype=np.float64) - self.m
        s2 = self.sigma * self.sigma
        return self.b * s2 / (x * x + s2) ** 1.5

    def implied_vol(self, k: FloatArray | float, T: float) -> FloatArray:
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / T)

    @property
    def min_variance(self) -> float:
        return self.a + self.b * self.sigma * float(np.sqrt(1.0 - self.rho * self.rho))

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return self.a, self.b, self.rho, self.m, self.sigma


@dataclass(frozen=True)
class SviFit:
    params: SviParams
    T: float
    rmse_vol_pts: float
    n_quotes: int
    min_g: float
    butterfly_ok: bool
    wing_ok: bool


def _linear_fit(
    k: FloatArray, w: FloatArray, wt: FloatArray, m: float, sigma: float
) -> tuple[SviParams, float] | None:
    y = (k - m) / sigma
    X = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1.0)])
    sw = np.sqrt(wt)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], w * sw, rcond=None)
    a, d, c = (float(v) for v in coef)
    # Project onto the feasible region: c = b*sigma >= 0 and |d| = |rho| c < c.
    c = max(c, 1e-8)
    d = float(np.clip(d, -0.99 * c, 0.99 * c))
    b, rho = c / sigma, d / c
    p = SviParams(a, b, rho, m, sigma)
    if p.min_variance < 0:
        p = SviParams(a - p.min_variance, b, rho, m, sigma)
    err = float(np.sum(wt * (p.total_variance(k) - w) ** 2))
    return p, err


def _to_x(p: SviParams) -> FloatArray:
    return np.array([max(p.min_variance, 0.0), p.b, p.rho, p.m, p.sigma])


def _from_x(x: FloatArray) -> SviParams:
    v, b, rho, m, sigma = (float(t) for t in x)
    a = v - b * sigma * float(np.sqrt(1.0 - rho * rho))
    return SviParams(a, b, rho, m, sigma)


def loss_and_grad(
    x: FloatArray, k: FloatArray, target: FloatArray, wt: FloatArray
) -> tuple[float, FloatArray]:
    """Weighted squared error in total variance and its gradient in (v, b, rho, m, sigma)."""
    v, b, rho, m, sig = x
    root = np.sqrt(1.0 - rho * rho)
    dx = k - m
    R = np.sqrt(dx * dx + sig * sig)
    model = v - b * sig * root + b * (rho * dx + R)
    resid = model - target
    # Analytic Jacobian of w with respect to (v, b, rho, m, sigma).
    jac = np.stack(
        [
            np.ones_like(k),
            -sig * root + rho * dx + R,
            b * sig * rho / root + b * dx,
            b * (-rho - dx / R),
            -b * root + b * sig / R,
        ]
    )
    return float(np.sum(wt * resid * resid)), 2.0 * jac @ (wt * resid)


def fit_svi(
    k: FloatArray,
    w: FloatArray,
    T: float,
    weights: FloatArray | None = None,
    restarts: int = 8,
    seed: int = 0,
    arb_grid: int = 201,
) -> SviFit:
    """Weighted least-squares raw SVI fit of one expiry slice."""
    k = np.asarray(k, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if len(k) < 5:
        raise ValueError("need at least 5 quotes to fit 5 SVI parameters")
    wt = np.ones_like(k) if weights is None else np.asarray(weights, dtype=np.float64)
    wt = wt / wt.sum()
    # Errors in w scale with T; normalizing keeps tolerances maturity-independent.
    scale = float(np.mean(w))
    wn = w / scale

    k_lo, k_hi = float(np.min(k)), float(np.max(k))
    span = max(k_hi - k_lo, 0.05)
    starts: list[tuple[float, SviParams]] = []
    for m in np.linspace(k_lo + 0.1 * span, k_hi - 0.1 * span, 5):
        for sigma in (0.02, 0.05, 0.1, 0.2, 0.4):
            res = _linear_fit(k, wn, wt, float(m), sigma * span)
            if res is not None:
                starts.append((res[1], res[0]))
    starts.sort(key=lambda t: t[0])
    rng = np.random.default_rng(seed)
    candidates = [p for _, p in starts[: max(1, restarts // 2)]]
    while len(candidates) < restarts:
        base = candidates[0]
        candidates.append(
            SviParams(
                base.a,
                base.b * float(np.exp(rng.normal(0, 0.5))),
                float(np.clip(base.rho + rng.normal(0, 0.3), -0.95, 0.95)),
                base.m + float(rng.normal(0, 0.2 * span)),
                base.sigma * float(np.exp(rng.normal(0, 0.5))),
            )
        )

    bounds = [
        (0.0, float(np.max(wn)) * 2),
        (1e-6, 10.0 / scale),
        (-0.999, 0.999),
        (k_lo - span, k_hi + span),
        (1e-4, 2.0 * span + 1.0),
    ]

    def loss(x: FloatArray) -> tuple[float, FloatArray]:
        return loss_and_grad(x, k, wn, wt)

    best_x, best_f = None, np.inf
    for p in candidates:
        x0 = np.clip(_to_x(p), [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(loss, x0, jac=True, method="L-BFGS-B", bounds=bounds)
        if res.fun < best_f:
            best_x, best_f = res.x, float(res.fun)
    assert best_x is not None
    pn = _from_x(best_x)
    params = SviParams(pn.a * scale, pn.b * scale, pn.rho, pn.m, pn.sigma)

    iv_fit = params.implied_vol(k, T)
    iv_mkt = np.sqrt(w / T)
    rmse = float(np.sqrt(np.mean((iv_fit - iv_mkt) ** 2)) * 100.0)
    # Check only where there is data (plus a small margin). Far outside the
    # quoted strikes the fit is pure extrapolation and its density says little.
    margin = max(0.1 * span, 0.02)
    grid = np.linspace(k_lo - margin, k_hi + margin, arb_grid)
    g = butterfly_density(params, grid)
    return SviFit(
        params=params,
        T=T,
        rmse_vol_pts=rmse,
        n_quotes=len(k),
        min_g=float(np.min(g)),
        butterfly_ok=bool(np.min(g) >= -1e-9),
        wing_ok=bool(params.b * (1 + abs(params.rho)) <= 2.0),
    )


def butterfly_density(p: SviParams, k: FloatArray) -> FloatArray:
    """Gatheral's g(k). The risk-neutral density is non-negative iff g >= 0.

    g(k) = (1 - k w'/(2w))^2 - (w'^2 / 4)(1/w + 1/4) + w''/2
    """
    w = p.total_variance(k)
    w1 = p.dw(k)
    w2 = p.d2w(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1 - k * w1 / (2 * w)) ** 2 - (w1 * w1 / 4) * (1 / w + 0.25) + w2 / 2
    return np.asarray(np.where(w > 0, g, -np.inf), dtype=np.float64)


def calendar_violations(
    slices: list[tuple[float, SviParams]], k: FloatArray, tol: float = 1e-6
) -> list[tuple[float, float, float]]:
    """Pairs of consecutive maturities where total variance decreases.

    Returns (T_short, T_long, worst shortfall) for each violating pair.
    """
    ordered = sorted(slices, key=lambda s: s[0])
    out = []
    for (t1, p1), (t2, p2) in itertools.pairwise(ordered):
        gap = p2.total_variance(k) - p1.total_variance(k)
        if np.min(gap) < -tol:
            out.append((t1, t2, float(-np.min(gap))))
    return out


def strike_for_delta(p: SviParams, call_delta: float) -> float:
    """Log-moneyness where the undiscounted forward call delta N(d1) equals `call_delta`.

    The vol used in d1 is the smile's own vol at that strike, which is how
    25-delta quotes are defined in practice. A 25-delta put is the strike with
    call delta 0.75.
    """
    target = float(norm.ppf(call_delta))

    def f(k: float) -> float:
        w = float(p.total_variance(k))
        return (-k + 0.5 * w) / np.sqrt(w) - target

    return float(brentq(f, -5.0, 5.0))


def smile_metrics(p: SviParams, T: float) -> dict[str, float]:
    """ATM vol, 25-delta risk reversal and butterfly for one slice."""
    atm = float(p.implied_vol(0.0, T))
    try:
        call25 = float(p.implied_vol(strike_for_delta(p, 0.25), T))
        put25 = float(p.implied_vol(strike_for_delta(p, 0.75), T))
    except ValueError:
        return {"atm_vol": atm, "rr25": float("nan"), "bf25": float("nan")}
    return {"atm_vol": atm, "rr25": call25 - put25, "bf25": 0.5 * (call25 + put25) - atm}
