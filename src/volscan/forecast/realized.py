"""Realized-variance proxies from daily OHLC bars, and day-count helpers.

All per-day proxies here are *daily variances* (not annualized). Multiply by
252 to annualize. True realized variance from 5-minute returns would be far
less noisy: a single daily range or close-to-close return is a very rough
estimate of that day's variance, which is why these are averaged before use.

Day-count convention used throughout the project:
    Implied vol is quoted on calendar time (ACT/365). Realized variance only
    accrues on trading days. I compare the two as *total variance over the
    option's life*: IV^2 * (calendar_days / 365) versus the forecast sum of
    daily variances over the trading sessions in the same window. Annualized
    forecast vols are then reported on the calendar basis, sqrt(total / (D/365)),
    so they sit on the same scale as the IV they are compared with.
"""

from __future__ import annotations

import functools
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

TRADING_DAYS = 252
CALENDAR_DAYS = 365


def _logs(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, lo, c = (np.log(df[col].astype(float)) for col in ("open", "high", "low", "close"))
    return {"o": o, "h": h, "l": lo, "c": c, "c_prev": c.shift(1)}


def close_to_close(df: pd.DataFrame) -> pd.Series:
    x = _logs(df)
    return (x["c"] - x["c_prev"]) ** 2


def parkinson(df: pd.DataFrame) -> pd.Series:
    x = _logs(df)
    return (x["h"] - x["l"]) ** 2 / (4.0 * math.log(2.0))


def garman_klass(df: pd.DataFrame) -> pd.Series:
    x = _logs(df)
    return 0.5 * (x["h"] - x["l"]) ** 2 - (2 * math.log(2.0) - 1) * (x["c"] - x["o"]) ** 2


def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    x = _logs(df)
    return (x["h"] - x["c"]) * (x["h"] - x["o"]) + (x["l"] - x["c"]) * (x["l"] - x["o"])


def yang_zhang_daily(df: pd.DataFrame) -> pd.Series:
    """Daily Yang-Zhang-style proxy: overnight jump squared + Rogers-Satchell.

    Range estimators like Parkinson miss the overnight gap, which for single
    stocks (earnings, news) is a big share of total variance. Adding the
    squared close-to-open return captures it, and Rogers-Satchell is
    unbiased under drift. This is the per-day input for HAR.
    """
    x = _logs(df)
    overnight = (x["o"] - x["c_prev"]) ** 2
    return overnight + rogers_satchell(df)


def yang_zhang(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Rolling Yang-Zhang (2000) variance estimator over `window` days.

    sigma^2 = sigma_overnight^2 + k sigma_open_close^2 + (1 - k) sigma_RS^2,
    with k chosen to minimize estimator variance.
    """
    x = _logs(df)
    overnight = x["o"] - x["c_prev"]
    open_close = x["c"] - x["o"]
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_o = overnight.rolling(n).var()
    var_c = open_close.rolling(n).var()
    var_rs = rogers_satchell(df).rolling(n).mean()
    return var_o + k * var_c + (1 - k) * var_rs


ESTIMATORS = {
    "close_to_close": close_to_close,
    "parkinson": parkinson,
    "garman_klass": garman_klass,
    "yang_zhang": yang_zhang_daily,
}


def daily_variance(df: pd.DataFrame, estimator: str = "yang_zhang") -> pd.Series:
    """Per-day variance proxy indexed by date, NaNs dropped and floored at 0."""
    s = ESTIMATORS[estimator](df)
    s.index = pd.to_datetime(df["date"])
    return s.clip(lower=0.0).dropna()


def annualize(daily_var: float | np.ndarray, periods: int = TRADING_DAYS) -> float | np.ndarray:
    return daily_var * periods


def total_var_to_vol(total_var: float, calendar_days: float) -> float:
    """Annualized vol on the calendar (IV) basis from total variance over a window."""
    return math.sqrt(max(total_var, 0.0) / (calendar_days / CALENDAR_DAYS))


@functools.lru_cache(maxsize=1)
def _nyse_calendar():  # type: ignore[no-untyped-def]
    import exchange_calendars as xcals

    return xcals.get_calendar("XNYS")


def trading_days_in(start: date, calendar_days: int) -> int:
    """Number of NYSE sessions in (start, start + calendar_days].

    Falls back to calendar_days * 252/365 if the calendar cannot cover the
    range (e.g. dates beyond its horizon).
    """
    end = start + timedelta(days=calendar_days)
    try:
        cal = _nyse_calendar()
        sessions = cal.sessions_in_range(
            pd.Timestamp(start) + pd.Timedelta(days=1), pd.Timestamp(end)
        )
        return max(len(sessions), 1)
    except Exception:
        return max(round(calendar_days * TRADING_DAYS / CALENDAR_DAYS), 1)
