"""Horizon-matched realized-variance forecasts, evaluation, and the ensemble.

Each model forecasts *total* variance over the trading sessions that fall in
the option tenor. Models are scored out-of-sample with QLIKE and MSE against
the realized Yang-Zhang proxy summed over the same window, using
non-overlapping evaluation points so the errors are roughly independent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd

from volscan.config import ForecastSettings
from volscan.forecast.garch import garch_forecast, log_returns_pct
from volscan.forecast.har import har_forecast_mean_var
from volscan.forecast.realized import daily_variance, total_var_to_vol, trading_days_in

log = logging.getLogger(__name__)
MODELS = ("har", "garch", "gjr_garch")


def qlike(realized: np.ndarray, forecast: np.ndarray) -> float:
    """Patton (2011) QLIKE loss, robust to noise in the realized proxy."""
    ratio = np.asarray(realized, dtype=float) / np.asarray(forecast, dtype=float)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mse(realized: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean((np.asarray(realized) - np.asarray(forecast)) ** 2))


def model_forecasts(
    rv: pd.Series, returns_pct: pd.Series, h: int, cfg: ForecastSettings
) -> dict[str, float]:
    """Total variance over the next h sessions for each model (decimal units)."""
    out: dict[str, float] = {}
    try:
        out["har"] = har_forecast_mean_var(rv, h, cfg.har_window) * h
    except Exception as exc:
        log.warning("HAR failed: %s", exc)
    for gjr in (False, True):
        name = "gjr_garch" if gjr else "garch"
        try:
            out[name] = garch_forecast(returns_pct, h, gjr=gjr, window=cfg.garch_window).total_var
        except Exception as exc:
            log.warning("%s failed: %s", name, exc)
    return out


def evaluate(
    rv: pd.Series,
    returns_pct: pd.Series,
    h: int,
    cfg: ForecastSettings,
    forecaster: Callable[
        [pd.Series, pd.Series, int, ForecastSettings], dict[str, float]
    ] = model_forecasts,
) -> pd.DataFrame:
    """Walk-forward out-of-sample scores per model at horizon h."""
    rv = rv.dropna()
    returns_pct = returns_pct.reindex(rv.index).dropna()
    rv = rv.reindex(returns_pct.index)
    n = len(rv)
    start = max(cfg.min_history, int(n * cfg.eval_start_fraction))
    points = range(start, n - h, h)
    rows = []
    for i in points:
        fc = forecaster(rv.iloc[: i + 1], returns_pct.iloc[: i + 1], h, cfg)
        realized = float(rv.iloc[i + 1 : i + 1 + h].sum())
        for model, f in fc.items():
            rows.append({"model": model, "realized": realized, "forecast": f})
    if not rows:
        return pd.DataFrame(columns=["model", "qlike", "mse", "n"])
    df = pd.DataFrame(rows)
    df = df[(df["realized"] > 0) & (df["forecast"] > 0)]
    scores = [
        {
            "model": m,
            "qlike": qlike(g["realized"].to_numpy(), g["forecast"].to_numpy()),
            "mse": mse(g["realized"].to_numpy(), g["forecast"].to_numpy()),
            "n": len(g),
        }
        for m, g in df.groupby("model")
    ]
    return pd.DataFrame(scores)


def ensemble_weights(scores: pd.DataFrame, method: str) -> dict[str, float]:
    if scores.empty or method == "mean":
        return {m: 1.0 / len(MODELS) for m in MODELS}
    inv = {r.model: 1.0 / max(float(r.qlike), 1e-6) for r in scores.itertuples()}
    total = sum(inv.values())
    return {m: v / total for m, v in inv.items()}


def combine(forecasts: dict[str, float], weights: dict[str, float]) -> float:
    usable = {m: f for m, f in forecasts.items() if np.isfinite(f) and f > 0 and m in weights}
    if not usable:
        raise ValueError("no model produced a usable forecast")
    wsum = sum(weights[m] for m in usable)
    return sum(weights[m] / wsum * f for m, f in usable.items())


def forecast_ticker(
    prices: pd.DataFrame,
    asof: date,
    tenors_days: list[int],
    cfg: ForecastSettings,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-tenor forecasts for each model plus the ensemble.

    Columns: tenor_days, h_sessions, model, total_var, vol (annualized, calendar basis).
    """
    hist = prices[pd.to_datetime(prices["date"]) <= pd.Timestamp(asof)].sort_values("date")
    if len(hist) < cfg.min_history:
        raise ValueError(f"only {len(hist)} days of history (need {cfg.min_history})")
    rv = daily_variance(hist, "yang_zhang")
    rets = log_returns_pct(hist.set_index(pd.to_datetime(hist["date"]))["close"])
    weights = weights or ensemble_weights(pd.DataFrame(), "mean")
    rows = []
    for tenor in tenors_days:
        h = trading_days_in(asof, tenor)
        fc = model_forecasts(rv, rets, h, cfg)
        fc["ensemble"] = combine(fc, weights)
        for model, tv in fc.items():
            rows.append(
                {
                    "tenor_days": tenor,
                    "h_sessions": h,
                    "model": model,
                    "total_var": tv,
                    "vol": total_var_to_vol(tv, tenor),
                }
            )
    return pd.DataFrame(rows)


def evaluate_ticker(
    prices: pd.DataFrame, asof: date, tenor_days: int, cfg: ForecastSettings
) -> pd.DataFrame:
    hist = prices[pd.to_datetime(prices["date"]) <= pd.Timestamp(asof)].sort_values("date")
    rv = daily_variance(hist, "yang_zhang")
    rets = log_returns_pct(hist.set_index(pd.to_datetime(hist["date"]))["close"])
    return evaluate(rv, rets, trading_days_in(asof, tenor_days), cfg)
