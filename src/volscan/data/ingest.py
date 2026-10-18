"""Nightly snapshot of prices, option chains, and earnings dates."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TypeVar
from zoneinfo import ZoneInfo

import pandas as pd

from volscan.config import Settings
from volscan.data.providers import DataProvider
from volscan.data.rates import risk_free_rate
from volscan.data.store import Store

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
T = TypeVar("T")


@dataclass
class IngestResult:
    asof: date
    ok: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    rate: float = 0.0
    rate_source: str = ""


def with_retries(
    fn: Callable[[], T],
    retries: int,
    backoff: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn, retrying with exponential backoff (backoff, 2*backoff, ...)."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except NotImplementedError:
            raise
        except Exception as exc:
            if attempt == retries:
                raise
            wait = backoff * (2**attempt)
            log.info("retry %d/%d after %s (sleeping %.1fs)", attempt + 1, retries, exc, wait)
            sleep(wait)
    raise AssertionError("unreachable")


def market_date(now: datetime | None = None) -> date:
    """Today's date in New York; the snapshot is tagged with the last session date."""
    now = now or datetime.now(NY)
    return now.astimezone(NY).date()


def ingest_ticker(
    provider: DataProvider,
    store: Store,
    ticker: str,
    today: date,
    settings: Settings,
    sleep: Callable[[float], None] = time.sleep,
) -> date:
    """Snapshot one ticker. Returns the session date the snapshot belongs to."""
    cfg = settings.ingest

    def retry(fn: Callable[[], T]) -> T:
        return with_retries(fn, cfg.max_retries, cfg.backoff_seconds, sleep)

    start = today - timedelta(days=int(365.25 * cfg.history_years))
    hist = retry(lambda: provider.get_history(ticker, start, today))
    if hist.empty:
        raise ValueError("no price history returned")
    session = pd.Timestamp(hist["date"].max()).date()

    expiries = [
        e
        for e in retry(lambda: provider.list_expiries(ticker))
        if cfg.min_dte <= (e - session).days <= cfg.max_dte
    ]
    if not expiries:
        raise ValueError(f"no expiries between {cfg.min_dte} and {cfg.max_dte} days")

    chains = []
    for expiry in expiries:
        chain = retry(lambda e=expiry: provider.get_chain(ticker, e))  # type: ignore[misc]
        if not chain.empty:
            chains.append(chain)
    if not chains:
        raise ValueError("all option chains were empty")
    chain_df = pd.concat(chains, ignore_index=True)
    spot = float(hist.sort_values("date")["close"].iloc[-1])
    chain_df.insert(0, "asof", session)
    chain_df.insert(1, "ticker", ticker)
    chain_df["spot"] = spot
    chain_df["captured_at"] = pd.Timestamp.now(tz=NY).tz_localize(None)

    hist = hist.copy()
    hist.insert(0, "ticker", ticker)

    earnings: list[date] = []
    if ticker not in settings.etfs:
        try:
            earnings = retry(lambda: provider.get_earnings_dates(ticker))
        except Exception as exc:
            log.warning("%s: earnings lookup failed: %s", ticker, exc)

    store.replace("prices", hist, {"ticker": ticker})
    store.replace("chains", chain_df, {"asof": session, "ticker": ticker})
    if earnings:
        store.replace(
            "earnings",
            pd.DataFrame({"ticker": ticker, "earnings_date": earnings}),
            {"ticker": ticker},
        )
    return session


def run_ingest(
    provider: DataProvider,
    store: Store,
    settings: Settings,
    today: date | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> IngestResult:
    today = today or market_date()
    sessions: list[date] = []
    result = IngestResult(asof=today)
    for ticker in settings.tickers:
        try:
            sessions.append(ingest_ticker(provider, store, ticker, today, settings, sleep))
            result.ok.append(ticker)
            log.info("ingested %s", ticker)
        except Exception as exc:
            result.failed[ticker] = f"{type(exc).__name__}: {exc}"
            log.error("ingest failed for %s: %s", ticker, exc)
        sleep(settings.ingest.pause_between_tickers)

    if sessions:
        result.asof = max(sessions)
    result.rate, result.rate_source = risk_free_rate(
        result.asof, settings.rates.fred_series, settings.rates.fallback_rate
    )
    store.replace(
        "rates",
        pd.DataFrame(
            {"asof": [result.asof], "rate": [result.rate], "source": [result.rate_source]}
        ),
        {"asof": result.asof},
    )
    store.replace(
        "run_log",
        pd.DataFrame(
            {
                "asof": result.asof,
                "step": "ingest",
                "ticker": list(result.failed),
                "error": list(result.failed.values()),
            }
        ),
        {"asof": result.asof, "step": "ingest"},
    )
    return result
