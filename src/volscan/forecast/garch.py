"""GARCH(1,1) and GJR-GARCH(1,1) with Student-t innovations.

Fitted on percentage log returns (the arch package's optimizer is better
conditioned at that scale). Multi-step variance forecasts are summed over the
horizon to give total variance, then converted back to decimal units.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass(frozen=True)
class GarchForecast:
    model: str
    total_var: float  # decimal variance summed over the horizon
    params: dict[str, float]
    persistence: float


def log_returns_pct(close: pd.Series) -> pd.Series:
    return (100.0 * np.log(close.astype(float)).diff()).dropna()


def fit_garch(returns_pct: pd.Series, gjr: bool = False):  # type: ignore[no-untyped-def]
    am = arch_model(
        returns_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1 if gjr else 0,
        q=1,
        dist="t",
        rescale=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return am.fit(disp="off", show_warning=False)


def aggregate_variance_path(
    omega: float, alpha: float, beta: float, sigma2_next: float, h: int, gamma: float = 0.0
) -> float:
    """Sum of E[sigma^2_{t+1..t+h}] for a (GJR-)GARCH(1,1) with symmetric shocks.

    E[sigma^2_{t+k}] = omega + (alpha + gamma/2 + beta) E[sigma^2_{t+k-1}]  for k >= 2.
    The gamma/2 term is the expected leverage contribution when shocks are
    symmetric (half of them are negative).
    """
    phi = alpha + 0.5 * gamma + beta
    total, s2 = 0.0, sigma2_next
    for _ in range(h):
        total += s2
        s2 = omega + phi * s2
    return total


def garch_forecast(
    returns_pct: pd.Series, h: int, gjr: bool = False, window: int | None = None
) -> GarchForecast:
    data = returns_pct.iloc[-window:] if window else returns_pct
    res = fit_garch(data, gjr=gjr)
    fc = res.forecast(horizon=h, reindex=False)
    total_pct2 = float(fc.variance.to_numpy()[-1].sum())
    p = {k: float(v) for k, v in res.params.items()}
    persistence = p["alpha[1]"] + p["beta[1]"] + 0.5 * p.get("gamma[1]", 0.0)
    return GarchForecast(
        model="gjr_garch" if gjr else "garch",
        total_var=total_pct2 / 1e4,
        params=p,
        persistence=persistence,
    )
