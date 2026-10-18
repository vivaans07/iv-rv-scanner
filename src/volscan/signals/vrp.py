"""Implied-minus-forecast spreads, adjusted for the normal variance risk premium.

Option sellers demand compensation for crash risk, so implied vol sits above
subsequently realized vol most of the time. A raw "IV minus forecast" spread
would therefore label almost every name rich. What matters is whether today's
spread is unusual *for that ticker*, so each spread is scored against its own
trailing history (z-score and excess over the rolling median).

Free data has no option history, so a fresh install has nothing to compare
against. Until a ticker has `min_history` stored observations, its spread is
scored cross-sectionally instead: log(IV / forecast) against the median and
MAD of the whole universe that day. The report says which mode was used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAD_TO_SD = 1.4826


@dataclass(frozen=True)
class VrpScore:
    z: float
    excess_vs_median: float  # vol points above the ticker's own rolling median spread
    mode: str  # "time-series" or "cross-sectional"
    n_history: int


def spread_vol_pts(iv: float, forecast_vol: float) -> float:
    return 100.0 * (iv - forecast_vol)


def time_series_score(today: float, history: pd.Series, lookback: int) -> VrpScore:
    h = history.dropna().iloc[-lookback:]
    sd = float(h.std(ddof=1)) if len(h) > 1 else float("nan")
    med = float(h.median())
    z = (today - float(h.mean())) / sd if sd and np.isfinite(sd) and sd > 1e-9 else 0.0
    return VrpScore(z=z, excess_vs_median=today - med, mode="time-series", n_history=len(h))


def cross_sectional_scores(iv: pd.Series, forecast_vol: pd.Series) -> pd.Series:
    """Robust z-score of log(IV/forecast) across tickers (index aligned)."""
    x = np.log(iv / forecast_vol)
    med = float(x.median())
    mad = float((x - med).abs().median()) * MAD_TO_SD
    if not np.isfinite(mad) or mad < 1e-9:
        return pd.Series(0.0, index=x.index)
    return (x - med) / mad


def score_spreads(
    today: pd.DataFrame, history: pd.DataFrame, lookback: int, min_history: int
) -> pd.DataFrame:
    """Score one tenor's spreads for every ticker.

    today:   columns ticker, iv, forecast_vol, spread (vol pts)
    history: columns ticker, asof, spread (prior days only)
    Returns today with z, excess_vs_median, z_mode, n_history added.
    """
    out = today.copy()
    xs = cross_sectional_scores(
        out.set_index("ticker")["iv"], out.set_index("ticker")["forecast_vol"]
    )
    universe_median = float(out["spread"].median())
    rows = []
    for r in out.itertuples():
        h = history.loc[history["ticker"] == r.ticker].sort_values("asof")["spread"]
        if len(h.dropna()) >= min_history:
            s = time_series_score(float(r.spread), h, lookback)
        else:
            s = VrpScore(
                z=float(xs.get(r.ticker, 0.0)),
                excess_vs_median=float(r.spread) - universe_median,
                mode="cross-sectional",
                n_history=len(h.dropna()),
            )
        rows.append(s)
    out["z"] = [s.z for s in rows]
    out["excess_vs_median"] = [s.excess_vs_median for s in rows]
    out["z_mode"] = [s.mode for s in rows]
    out["n_history"] = [s.n_history for s in rows]
    return out
