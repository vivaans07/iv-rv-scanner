"""Separate the earnings jump from diffusive vol in the implied term structure.

Around a scheduled event the market prices a one-off jump on top of normal
day-to-day variance. In total-variance terms, for an expiry after the event:

    sigma_imp^2 * T = sigma_base^2 * T + sigma_event^2

With one expiry before the event (no jump) and one after, sigma_base comes
from the first and sigma_event is the difference. If every listed expiry is
after the event, two of them pin down both unknowns, assuming flat base vol
between them. The implied event standard deviation is then compared with how
much the stock has actually moved on past earnings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


@dataclass(frozen=True)
class EventSplit:
    base_var_rate: float  # annualized diffusive variance
    event_var: float  # one-off total variance of the jump (>= 0)
    method: str

    @property
    def implied_move_sd(self) -> float:
        return math.sqrt(self.event_var)

    @property
    def implied_abs_move(self) -> float:
        """Expected |move| if the jump were normal: sd * sqrt(2/pi)."""
        return self.implied_move_sd * SQRT_2_OVER_PI


def split_event_variance(
    slices: pd.DataFrame, asof: date, earnings_date: date
) -> EventSplit | None:
    """slices: columns expiry, T, atm_vol (one row per fitted expiry)."""
    s = slices.sort_values("T")
    w = (s["atm_vol"] ** 2 * s["T"]).to_numpy()
    T = s["T"].to_numpy()
    expiry = pd.to_datetime(s["expiry"])
    # An expiry on the announcement date is ambiguous: it contains the move for
    # a before-the-open reporter but not for an after-the-close one. Use only
    # expiries strictly before and strictly after the date.
    before = (expiry < pd.Timestamp(earnings_date)).to_numpy()
    after = (expiry > pd.Timestamp(earnings_date)).to_numpy()
    if not after.any():
        return None
    post = np.flatnonzero(after)
    pre = np.flatnonzero(before)
    if len(pre):
        i, j = pre[-1], post[0]
        base = w[i] / T[i]
        event = w[j] - base * T[j]
        method = "pre/post expiry pair"
    else:
        if len(post) < 2:
            return None
        i, j = post[0], post[1]
        base = (w[j] - w[i]) / (T[j] - T[i])
        event = w[i] - base * T[i]
        method = "two post-event expiries"
    if not np.isfinite(base) or base <= 0:
        return None
    return EventSplit(base_var_rate=float(base), event_var=float(max(event, 0.0)), method=method)


def ex_earnings_vol(atm_vol: float, T: float, event_var: float) -> float:
    """Base (diffusive) vol at a tenor whose window contains the event."""
    base_total = atm_vol * atm_vol * T - event_var
    return math.sqrt(base_total / T) if base_total > 0 else float("nan")


def historical_move_stats(returns: pd.Series) -> dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {"hist_abs_move": float("nan"), "hist_rms_move": float("nan"), "hist_n": 0}
    return {
        "hist_abs_move": float(r.abs().mean()),
        "hist_rms_move": float(np.sqrt((r**2).mean())),
        "hist_n": len(r),
    }
