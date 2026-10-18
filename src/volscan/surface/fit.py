"""Fit a full surface for one ticker on one day."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from volscan.config import Settings
from volscan.surface.cleaning import CleanStats, filter_quotes, implied_vol_table
from volscan.surface.svi import SviParams, calendar_violations, fit_svi, smile_metrics
from volscan.surface.term import TermStructure

log = logging.getLogger(__name__)


@dataclass
class SurfaceResult:
    ticker: str
    asof: date
    spot: float
    quotes: pd.DataFrame
    slices: pd.DataFrame
    metrics: pd.DataFrame
    clean_stats: CleanStats
    calendar_issues: list[tuple[float, float, float]] = field(default_factory=list)

    def term(self) -> TermStructure:
        return term_from_slices(self.slices)


def term_from_slices(slices: pd.DataFrame) -> TermStructure:
    return TermStructure(
        [
            (
                float(r.T),
                SviParams(float(r.a), float(r.b), float(r.rho), float(r.m), float(r.sigma)),
            )
            for r in slices.itertuples()
        ]
    )


def _weights(q: pd.DataFrame, how: str) -> np.ndarray:
    if how == "vega":
        return q["vega"].to_numpy() + 1e-8
    if how == "inverse_spread":
        return 1.0 / np.maximum(q["ask"].to_numpy() - q["bid"].to_numpy(), 0.01)
    return np.ones(len(q))


def fit_ticker_surface(
    ticker: str, asof: date, chain: pd.DataFrame, rate: float, settings: Settings
) -> SurfaceResult:
    spot = float(chain["spot"].iloc[0])
    clean, stats = filter_quotes(chain, settings.cleaning)
    quotes = implied_vol_table(clean, asof, spot, rate, settings.cleaning.forward_pairs)
    if quotes.empty:
        raise ValueError("no usable implied vols after cleaning")

    rows = []
    for expiry, q in quotes.groupby("expiry"):
        if len(q) < settings.svi.min_quotes:
            continue
        T = float(q["T"].iloc[0])
        fit = fit_svi(
            q["k"].to_numpy(),
            q["w"].to_numpy(),
            T,
            weights=_weights(q, settings.svi.weight),
            restarts=settings.svi.restarts,
            arb_grid=settings.svi.arb_grid_points,
        )
        p = fit.params
        rows.append(
            {
                "asof": asof,
                "ticker": ticker,
                "expiry": expiry,
                "T": T,
                "forward": float(q["forward"].iloc[0]),
                "n_parity": int(q["n_parity"].iloc[0]),
                "a": p.a,
                "b": p.b,
                "rho": p.rho,
                "m": p.m,
                "sigma": p.sigma,
                "rmse_vol_pts": fit.rmse_vol_pts,
                "n_quotes": fit.n_quotes,
                "min_g": fit.min_g,
                "butterfly_ok": fit.butterfly_ok,
                "wing_ok": fit.wing_ok,
                **smile_metrics(p, T),
            }
        )
    if not rows:
        raise ValueError(f"no expiry had at least {settings.svi.min_quotes} clean quotes")
    slices = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    term = term_from_slices(slices)
    grid = np.linspace(-0.5, 0.5, 101)
    issues = calendar_violations(term.slices, grid)
    bad_long = {t2 for _, t2, _ in issues}
    slices["calendar_ok"] = ~slices["T"].isin(bad_long)

    metrics = pd.DataFrame([term.constant_maturity(d) for d in settings.term.tenors_days])
    metrics.insert(0, "ticker", ticker)
    metrics.insert(0, "asof", asof)
    metrics["spot"] = spot
    metrics["calendar_violations"] = len(issues)
    quotes = quotes.assign(asof=asof, ticker=ticker)
    return SurfaceResult(ticker, asof, spot, quotes, slices, metrics, stats, issues)
