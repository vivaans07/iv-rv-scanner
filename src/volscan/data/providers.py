"""Market data providers.

`DataProvider` is the only interface the rest of the code sees. yfinance is
the default because it needs no key; it is an unofficial scraper of Yahoo
Finance, the quotes can be delayed or stale, and it is meant for personal
research only. Keyed providers are stubbed so they can be added without
touching downstream code.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

HISTORY_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume", "split_factor"]
CHAIN_COLUMNS = [
    "expiry",
    "strike",
    "option_type",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "last_trade",
]


class DataProvider(ABC):
    """Source of daily prices, option chains, and earnings dates."""

    name: str = "abstract"

    @abstractmethod
    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Daily OHLCV, unadjusted for dividends, with columns HISTORY_COLUMNS.

        `close` is the raw close (what options are struck against).
        `split_factor` is the split ratio effective on that date (1.0 if none).
        """

    @abstractmethod
    def list_expiries(self, ticker: str) -> list[date]:
        """All listed option expiries."""

    @abstractmethod
    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        """Calls and puts for one expiry with columns CHAIN_COLUMNS.

        option_type is "C" or "P".
        """

    @abstractmethod
    def get_earnings_dates(self, ticker: str) -> list[date]:
        """Past and announced future earnings dates."""


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def __init__(self) -> None:
        import yfinance as yf  # imported lazily so tests never touch the network

        self._yf = yf

    def _ticker(self, ticker: str) -> Any:
        return self._yf.Ticker(ticker)

    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        raw = self._ticker(ticker).history(
            start=start.isoformat(),
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=False,
            actions=True,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        idx = pd.DatetimeIndex(raw.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        splits = raw.get("Stock Splits", pd.Series(0.0, index=raw.index)).astype(float)
        out = pd.DataFrame(
            {
                "date": idx.normalize().date,
                "open": raw["Open"].to_numpy(dtype=float),
                "high": raw["High"].to_numpy(dtype=float),
                "low": raw["Low"].to_numpy(dtype=float),
                "close": raw["Close"].to_numpy(dtype=float),
                "adj_close": raw.get("Adj Close", raw["Close"]).to_numpy(dtype=float),
                "volume": raw["Volume"].to_numpy(dtype=float),
                "split_factor": splits.where(splits > 0, 1.0).to_numpy(dtype=float),
            }
        )
        return out.dropna(subset=["close"]).reset_index(drop=True)

    def list_expiries(self, ticker: str) -> list[date]:
        return [pd.Timestamp(e).date() for e in self._ticker(ticker).options]

    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        chain = self._ticker(ticker).option_chain(expiry.isoformat())
        frames = []
        for df, kind in ((chain.calls, "C"), (chain.puts, "P")):
            if df is None or df.empty:
                continue
            last_trade = pd.to_datetime(df["lastTradeDate"], utc=True, errors="coerce")
            frames.append(
                pd.DataFrame(
                    {
                        "expiry": expiry,
                        "strike": df["strike"].astype(float),
                        "option_type": kind,
                        "bid": df["bid"].astype(float),
                        "ask": df["ask"].astype(float),
                        "last": df["lastPrice"].astype(float),
                        "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0.0),
                        "open_interest": pd.to_numeric(df["openInterest"], errors="coerce").fillna(
                            0.0
                        ),
                        "last_trade": last_trade.dt.tz_localize(None),
                    }
                )
            )
        if not frames:
            return pd.DataFrame(columns=CHAIN_COLUMNS)
        return pd.concat(frames, ignore_index=True)[CHAIN_COLUMNS]

    def get_earnings_dates(self, ticker: str) -> list[date]:
        try:
            df = self._ticker(ticker).get_earnings_dates(limit=24)
        except Exception as exc:  # yfinance raises a variety of errors here
            log.warning("earnings dates unavailable for %s: %s", ticker, exc)
            return []
        if df is None or df.empty:
            return []
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        return sorted({ts.date() for ts in idx})


class _KeyedStub(DataProvider):
    env_var = ""

    def __init__(self) -> None:
        self.api_key = os.getenv(self.env_var, "")

    def _fail(self) -> None:
        raise NotImplementedError(
            f"The {self.name} provider is a placeholder. Implement it in "
            f"volscan/data/providers.py (key read from ${self.env_var}) or set "
            "VOLSCAN_PROVIDER=yfinance."
        )

    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self._fail()
        raise AssertionError

    def list_expiries(self, ticker: str) -> list[date]:
        self._fail()
        raise AssertionError

    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        self._fail()
        raise AssertionError

    def get_earnings_dates(self, ticker: str) -> list[date]:
        self._fail()
        raise AssertionError


class TradierProvider(_KeyedStub):
    name = "tradier"
    env_var = "TRADIER_API_KEY"


class PolygonProvider(_KeyedStub):
    name = "polygon"
    env_var = "POLYGON_API_KEY"


_PROVIDERS: dict[str, type[DataProvider]] = {
    "yfinance": YFinanceProvider,
    "tradier": TradierProvider,
    "polygon": PolygonProvider,
}


def get_provider(name: str) -> DataProvider:
    try:
        return _PROVIDERS[name.lower()]()
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(_PROVIDERS)}") from None
