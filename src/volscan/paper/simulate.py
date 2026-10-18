"""Synthetic markets for studying delta-hedged option P&L.

`FlatVolMarket` quotes every option at a single implied vol, so a simulated
spot path with a different realized vol isolates the variance-risk-premium
payoff. The same hedger used for the live paper book runs on top of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from volscan.config import PaperSettings
from volscan.paper.hedger import open_position, step_position
from volscan.paper.positions import LegQuote
from volscan.pricing.black import black76_price


@dataclass
class FlatVolMarket:
    asof: date
    spot: float
    rate: float
    vol: float
    expiry_list: list[date]
    half_spread: float = 0.0
    forecast_vol: float = float("nan")
    strike_step: float = 1.0
    missing: set[tuple[date, float, str]] = field(default_factory=set)

    def forward(self, expiry: date) -> float:
        return self.spot * math.exp(self.rate * (expiry - self.asof).days / 365.0)

    def expiries(self) -> list[date]:
        return list(self.expiry_list)

    def strikes(self, expiry: date) -> list[float]:
        c = round(self.spot / self.strike_step) * self.strike_step
        return [c + i * self.strike_step for i in range(-10, 11)]

    def option(self, expiry: date, strike: float, option_type: str) -> LegQuote | None:
        if (expiry, strike, option_type) in self.missing:
            return None
        T = (expiry - self.asof).days / 365.0
        if T <= 0:
            return None
        mid = float(
            black76_price(self.forward(expiry), strike, T, self.rate, self.vol, option_type == "C")
        )
        return LegQuote(mid=mid, half_spread=self.half_spread, iv=self.vol, source="market")


_SUMMED = (
    "total_pnl",
    "gross_pnl",
    "gamma_theta",
    "vega_pnl",
    "hedging_error",
    "residual",
    "costs",
)


def simulate_hedged_straddle(
    n_paths: int,
    n_days: int,
    implied_vol: float,
    realized_vol: float,
    label: str,
    cfg: PaperSettings,
    seed: int = 0,
    start: date = date(2026, 1, 5),
    expiry_days: int = 60,
    half_spread: float = 0.0,
    jump_day: int | None = None,
    jump_size: float = 0.0,
) -> pd.DataFrame:
    """Run the hedger over GBM paths. Returns one row per path of summed components."""
    rng = np.random.default_rng(seed)
    cfg = cfg.model_copy(
        update={"holding_days": n_days + 1, "exit_days_before_expiry": 0, "target_dte": expiry_days}
    )
    dates = list(pd.bdate_range(start, periods=n_days + 1).date)
    expiry = start + timedelta(days=expiry_days)
    rows = []
    for path in range(n_paths):
        m = FlatVolMarket(dates[0], 100.0, 0.0, implied_vol, [expiry], half_spread)
        res = open_position(
            f"p{path}", "SIM", m, label, 2.0 if label == "RICH" else -2.0, False, cfg
        )
        assert res is not None and res.position is not None
        pos = res.position
        marks = [res.mark]
        S = 100.0
        for i, d in enumerate(dates[1:], start=1):
            dt = (d - dates[i - 1]).days / 365.0
            S *= math.exp(
                -0.5 * realized_vol**2 * dt + realized_vol * math.sqrt(dt) * rng.standard_normal()
            )
            if jump_day is not None and i == jump_day:
                S *= math.exp(jump_size)
            m = FlatVolMarket(d, S, 0.0, implied_vol, [expiry], half_spread)
            marks.append(step_position(pos, m, cfg).mark)
        df = pd.DataFrame(marks)
        rows.append(
            {
                "path": path,
                "contracts": abs(pos.qty),
                **{c: float(df[c].sum()) for c in _SUMMED},
            }
        )
    return pd.DataFrame(rows)
