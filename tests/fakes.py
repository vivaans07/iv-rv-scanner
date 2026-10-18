"""Deterministic offline data provider for tests."""

from __future__ import annotations

import zlib
from datetime import date, timedelta

import numpy as np
import pandas as pd

from volscan.data.providers import CHAIN_COLUMNS, DataProvider
from volscan.pricing.black import black76_price

RATE = 0.04
DIV_YIELD = 0.01


def smile(k: np.ndarray, T: float, atm: float = 0.25) -> np.ndarray:
    """A simple skewed smile that flattens with maturity."""
    scale = 1.0 / np.sqrt(max(T, 1 / 52) / 0.25)
    return atm - 0.12 * k * scale + 0.25 * k**2 * scale


class FakeProvider(DataProvider):
    name = "fake"

    def __init__(
        self,
        seed: int = 7,
        vol: float = 0.25,
        fail: set[str] | None = None,
        flaky: dict[str, int] | None = None,
    ) -> None:
        self.seed = seed
        self.vol = vol
        self.fail = fail or set()
        self.flaky = dict(flaky or {})

    def _check(self, ticker: str) -> None:
        if ticker in self.fail:
            raise ConnectionError(f"{ticker} unavailable")
        if self.flaky.get(ticker, 0) > 0:
            self.flaky[ticker] -= 1
            raise TimeoutError("transient")

    def _path(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        days = pd.bdate_range(start, end)
        rng = np.random.default_rng(zlib.crc32(f"{ticker}-{self.seed}".encode()))
        dt = 1 / 252
        rets = rng.normal(-0.5 * self.vol**2 * dt, self.vol * np.sqrt(dt), len(days))
        close = 100.0 * np.exp(np.cumsum(rets))
        open_ = close * np.exp(rng.normal(0, 0.003, len(days)))
        hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, len(days))))
        lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, len(days))))
        return pd.DataFrame(
            {
                "date": days.date,
                "open": open_,
                "high": hi,
                "low": lo,
                "close": close,
                "adj_close": close,
                "volume": 1e6,
                "split_factor": 1.0,
            }
        )

    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self._check(ticker)
        return self._path(ticker, start, end)

    def list_expiries(self, ticker: str) -> list[date]:
        self._check(ticker)
        base = date(2026, 1, 2)
        return [base + timedelta(days=7 * i) for i in range(1, 60)]

    def get_chain(self, ticker: str, expiry: date, asof: date | None = None) -> pd.DataFrame:
        self._check(ticker)
        asof = asof or self.asof
        hist = self._path(ticker, asof - timedelta(days=10), asof)
        spot = float(hist["close"].iloc[-1])
        T = (expiry - asof).days / 365.0
        F = spot * np.exp((RATE - DIV_YIELD) * T)
        strikes = np.round(np.arange(0.7, 1.31, 0.025) * spot, 2)
        k = np.log(strikes / F)
        vol = smile(k, T, self.vol)
        rows = []
        for kind in ("C", "P"):
            mid = black76_price(F, strikes, T, RATE, vol, kind == "C")
            half = np.maximum(0.02, 0.03 * mid)
            rows.append(
                pd.DataFrame(
                    {
                        "expiry": expiry,
                        "strike": strikes,
                        "option_type": kind,
                        "bid": np.maximum(mid - half, 0.0),
                        "ask": mid + half,
                        "last": mid,
                        "volume": 500.0,
                        "open_interest": 2000.0,
                        "last_trade": pd.Timestamp(asof),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)[CHAIN_COLUMNS]

    asof: date = date(2026, 3, 2)

    def get_earnings_dates(self, ticker: str) -> list[date]:
        self._check(ticker)
        return [date(2025, 1, 30) + timedelta(days=91 * i) for i in range(6)]
