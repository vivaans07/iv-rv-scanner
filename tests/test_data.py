from datetime import date

import pandas as pd
import pytest

from tests.fakes import FakeProvider
from volscan.config import load_settings
from volscan.data.earnings import earnings_day_returns, next_earnings, spans_earnings
from volscan.data.ingest import run_ingest, with_retries
from volscan.data.providers import get_provider
from volscan.data.rates import discount_to_continuous, risk_free_rate
from volscan.data.store import Store

ASOF = date(2026, 3, 2)


def _settings(tickers: list[str]):  # type: ignore[no-untyped-def]
    s = load_settings(tickers=tickers)
    s.ingest.pause_between_tickers = 0.0
    s.ingest.backoff_seconds = 0.0
    return s


def test_ingest_is_idempotent_and_isolates_failures() -> None:
    store = Store(":memory:")
    provider = FakeProvider(fail={"BAD"}, flaky={"AAPL": 2})
    settings = _settings(["AAPL", "BAD", "MSFT"])
    res = run_ingest(provider, store, settings, today=ASOF, sleep=lambda _s: None)
    assert res.ok == ["AAPL", "MSFT"]
    assert "BAD" in res.failed
    n_chain = len(store.read("chains"))
    n_px = len(store.read("prices"))
    assert n_chain > 0 and n_px > 700

    run_ingest(provider, store, settings, today=ASOF, sleep=lambda _s: None)
    assert len(store.read("chains")) == n_chain
    assert len(store.read("prices")) == n_px
    chains = store.read("chains")
    dte = (pd.to_datetime(chains["expiry"]) - pd.Timestamp(ASOF)).dt.days
    assert dte.between(7, 180).all()
    assert store.read("run_log")["ticker"].tolist() == ["BAD"]
    assert res.rate_source == "fallback constant"


def test_retries_back_off_exponentially() -> None:
    waits: list[float] = []
    calls = {"n": 0}

    def flaky() -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError
        return 42

    assert with_retries(flaky, retries=3, backoff=1.0, sleep=waits.append) == 42
    assert waits == [1.0, 2.0]
    with pytest.raises(TimeoutError):
        with_retries(lambda: (_ for _ in ()).throw(TimeoutError()), 1, 0.5, waits.append)


def test_keyed_provider_stubs_raise_clearly() -> None:
    p = get_provider("tradier")
    with pytest.raises(NotImplementedError, match="placeholder"):
        p.list_expiries("SPY")
    with pytest.raises(ValueError):
        get_provider("nope")


def test_discount_conversion() -> None:
    # A 5% discount yield on a 91-day bill is a bit above 5% continuous.
    r = discount_to_continuous(0.05)
    assert 0.0505 < r < 0.052
    rate, src = risk_free_rate(ASOF, fallback=0.037)
    assert (rate, src) == (0.037, "fallback constant")


def test_earnings_helpers() -> None:
    dates = [date(2026, 1, 28), date(2026, 4, 29)]
    assert next_earnings(dates, ASOF) == date(2026, 4, 29)
    assert spans_earnings(ASOF, date(2026, 5, 15), date(2026, 4, 29))
    assert not spans_earnings(ASOF, date(2026, 4, 29), date(2026, 4, 29))
    hist = pd.DataFrame({"date": pd.bdate_range("2026-01-20", "2026-02-05").date, "close": 100.0})
    hist.loc[hist["date"] > date(2026, 1, 28), "close"] = 110.0
    rets = earnings_day_returns(hist, dates, ASOF, 10)
    assert rets.iloc[0] == pytest.approx(0.0953, abs=1e-3)
