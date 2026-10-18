"""Earnings calendar helpers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def next_earnings(dates: list[date], asof: date) -> date | None:
    future = sorted(d for d in dates if d > asof)
    return future[0] if future else None


def spans_earnings(asof: date, expiry: date, earnings_date: date | None) -> bool:
    """True if the earnings release is priced into an option expiring on `expiry`.

    The move happens on the first session after the announcement. Since most
    companies report after the close, an announcement on the expiry date itself
    lands after the option has stopped trading, so it is excluded.
    """
    return earnings_date is not None and asof < earnings_date < expiry


def earnings_day_returns(
    history: pd.DataFrame, earnings: list[date], asof: date, n_events: int
) -> pd.Series:
    """Close-to-close log return across each past earnings release.

    The reaction is measured from the close before the announcement date to the
    close after it. That covers both before-open and after-close reporters
    without knowing which one a company is: it picks up one extra day of normal
    variance, which is small next to the event move.
    """
    if history.empty:
        return pd.Series(dtype=float)
    px = history.set_index(pd.to_datetime(history["date"]))["close"].sort_index()
    out = {}
    for d in sorted(e for e in earnings if e < asof)[-n_events:]:
        ts = pd.Timestamp(d)
        before = px[px.index < ts]
        after = px[px.index > ts]
        if before.empty or after.empty:
            continue
        out[d] = float(np.log(after.iloc[0] / before.iloc[-1]))
    return pd.Series(out, dtype=float)
