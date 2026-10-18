"""HAR-RV (Corsi 2009) with direct multi-horizon targets.

The regressors are the daily, weekly (5-day) and monthly (22-day) averages of
the variance proxy. Rather than iterating a one-step model forward, the target
is the average variance over the *next h days*, so each horizon gets its own
coefficients. That is what an h-day option actually pays on.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


def har_features(rv: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rv_d": rv,
            "rv_w": rv.rolling(5).mean(),
            "rv_m": rv.rolling(22).mean(),
        }
    )


def har_target(rv: pd.Series, h: int) -> pd.Series:
    """Mean of rv over t+1..t+h, aligned to t."""
    return rv[::-1].rolling(h).mean()[::-1].shift(-1)


@dataclass(frozen=True)
class HarFit:
    h: int
    params: dict[str, float]
    n_obs: int
    r2: float


def fit_har(rv: pd.Series, h: int, window: int | None = None) -> HarFit:
    X = har_features(rv)
    y = har_target(rv, h)
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    if window:
        data = data.iloc[-window:]
    if len(data) < 50:
        raise ValueError("not enough history to fit HAR")
    model = sm.OLS(data["y"], sm.add_constant(data[["rv_d", "rv_w", "rv_m"]])).fit()
    return HarFit(
        h=h,
        params={k: float(v) for k, v in model.params.items()},
        n_obs=len(data),
        r2=float(model.rsquared),
    )


def har_forecast_mean_var(rv: pd.Series, h: int, window: int | None = None) -> float:
    """Forecast mean daily variance over the next h days, using data up to rv's end."""
    fit = fit_har(rv, h, window)
    x = har_features(rv).iloc[-1]
    p = fit.params
    pred = p["const"] + p["rv_d"] * x["rv_d"] + p["rv_w"] * x["rv_w"] + p["rv_m"] * x["rv_m"]
    # OLS on levels can dip below zero after a calm stretch; never forecast less
    # than a fraction of the long-run mean.
    floor = 0.1 * float(rv.iloc[-(window or len(rv)) :].mean())
    return float(max(pred, floor))
