"""Risk-free rate from the FRED 3-month T-bill series (DTB3).

DTB3 is quoted on a bank-discount basis in percent, not as a continuously
compounded yield, so it is converted before use. With a FRED key the JSON API
is used; without one, the public CSV endpoint is tried; failing both, the
configured constant is returned.
"""

from __future__ import annotations

import io
import logging
import math
import os
from datetime import date

import pandas as pd
import requests

log = logging.getLogger(__name__)

_API = "https://api.stlouisfed.org/fred/series/observations"
_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BILL_DAYS = 91


def discount_to_continuous(discount_rate: float, days: int = BILL_DAYS) -> float:
    """Convert a T-bill bank-discount rate (decimal) to a continuous ACT/365 rate."""
    price = 1.0 - discount_rate * days / 360.0
    return -math.log(price) / (days / 365.0)


def _fetch_series(series: str, timeout: float = 10.0) -> pd.Series:
    key = os.getenv("FRED_API_KEY", "").strip()
    if key:
        params: dict[str, str | int] = {
            "series_id": series,
            "api_key": key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 30,
        }
        resp = requests.get(_API, params=params, timeout=timeout)
        resp.raise_for_status()
        obs = resp.json()["observations"]
        s = pd.Series({o["date"]: o["value"] for o in obs})
    else:
        resp = requests.get(_CSV, params={"id": series}, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        s = pd.Series(df.iloc[:, 1].to_numpy(), index=df.iloc[:, 0].to_numpy())
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def risk_free_rate(asof: date, series: str = "DTB3", fallback: float = 0.04) -> tuple[float, str]:
    """Continuous rate as of `asof`, and a label saying where it came from."""
    try:
        s = _fetch_series(series)
        s = s[s.index <= pd.Timestamp(asof)]
        if s.empty:
            raise ValueError("no observations on or before asof")
        disc = float(s.iloc[-1]) / 100.0
        return discount_to_continuous(disc), f"FRED {series} {s.index[-1].date()}"
    except Exception as exc:
        log.warning("rate fetch failed (%s); using fallback %.4f", exc, fallback)
        return fallback, "fallback constant"
