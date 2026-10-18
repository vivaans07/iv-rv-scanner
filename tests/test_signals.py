from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from volscan.config import SignalSettings
from volscan.signals.build import build_signals
from volscan.signals.earnings_adj import ex_earnings_vol, split_event_variance
from volscan.signals.rank import classify, rank_candidates
from volscan.signals.vrp import cross_sectional_scores, score_spreads, time_series_score

ASOF = date(2026, 3, 2)


def test_time_series_z_uses_own_history() -> None:
    # A name whose IV is always ~4 pts over forecast is not rich at +4.
    hist = pd.Series(np.random.default_rng(0).normal(4.0, 1.0, 300))
    assert abs(time_series_score(4.0, hist, 252).z) < 0.3
    assert time_series_score(7.0, hist, 252).z > 2.5


def test_cross_sectional_fallback_when_history_short() -> None:
    today = pd.DataFrame(
        {
            "ticker": list("ABCDE"),
            "iv": [0.30, 0.31, 0.29, 0.30, 0.45],
            "forecast_vol": [0.25] * 5,
            "spread": [5.0, 6.0, 4.0, 5.0, 20.0],
        }
    )
    out = score_spreads(today, pd.DataFrame(columns=["ticker", "asof", "spread"]), 252, 40)
    assert (out["z_mode"] == "cross-sectional").all()
    assert out.set_index("ticker").loc["E", "z"] > 3
    xs = cross_sectional_scores(
        today.set_index("ticker")["iv"], today.set_index("ticker")["forecast_vol"]
    )
    assert xs.loc["A"] == pytest.approx(0.0, abs=1e-9)


def test_event_variance_recovered_from_term_structure() -> None:
    base_vol, move = 0.30, 0.06
    e_date = ASOF + timedelta(days=20)
    expiries = [ASOF + timedelta(days=d) for d in (10, 31, 59)]
    T = np.array([(e - ASOF).days / 365 for e in expiries])
    w = base_vol**2 * T + np.where(np.array(expiries) > e_date, move**2, 0.0)
    slices = pd.DataFrame({"expiry": expiries, "T": T, "atm_vol": np.sqrt(w / T)})
    split = split_event_variance(slices, ASOF, e_date)
    assert split is not None
    assert split.implied_move_sd == pytest.approx(move, rel=1e-9)
    assert split.base_var_rate == pytest.approx(base_vol**2)
    iv31 = float(np.sqrt(w[1] / T[1]))
    assert ex_earnings_vol(iv31, T[1], split.event_var) == pytest.approx(base_vol)

    # No pre-event expiry: two post-event expiries still identify the split.
    split2 = split_event_variance(slices.iloc[1:], ASOF, e_date)
    assert split2 is not None and split2.implied_move_sd == pytest.approx(move, rel=1e-9)


def test_rank_respects_liquidity_and_earnings_rules() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E"],
            "z": [3.0, 2.5, -2.0, -3.5, 2.8],
            "liquid": [True, False, True, True, True],
            "earnings_in_window": [False, False, False, False, True],
            "base_iv": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    out = rank_candidates(df, entry_z=1.5, top_n=5, trade_through_earnings=False).set_index(
        "ticker"
    )
    assert out.loc["A", "candidate"] and out.loc["C", "candidate"] and out.loc["D", "candidate"]
    assert not out.loc["B", "candidate"]  # illiquid
    assert not out.loc["E", "candidate"]  # earnings in window, no clean split
    top1 = rank_candidates(df, 1.5, 1, True).set_index("ticker")
    assert top1["candidate"].sum() == 2  # one rich, one cheap
    assert top1.loc["D", "candidate"] and top1.loc["A", "candidate"]
    assert classify(0.2, 1.5) == "FAIR"


def test_build_signals_end_to_end() -> None:
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    ivs = [0.30, 0.32, 0.29, 0.50]
    metrics = pd.DataFrame(
        [
            {"ticker": t, "tenor_days": d, "atm_vol": v}
            for t, v in zip(tickers, ivs, strict=True)
            for d in (30, 60)
        ]
    )
    forecasts = pd.DataFrame(
        [
            {"ticker": t, "tenor_days": d, "model": "ensemble", "vol": 0.26}
            for t in tickers
            for d in (30, 60)
        ]
    )
    liquidity = pd.DataFrame({"ticker": tickers, "liquid": True, "liquidity_note": "ok"})
    out = build_signals(
        ASOF,
        metrics,
        pd.DataFrame(columns=["ticker", "expiry", "T", "atm_vol"]),
        forecasts,
        liquidity,
        pd.DataFrame(columns=["ticker", "earnings_date"]),
        pd.DataFrame(columns=["ticker", "date", "close"]),
        pd.DataFrame(),
        SignalSettings(),
    )
    p = out[out["tenor_days"] == 30].set_index("ticker")
    assert p.loc["DDD", "label"] == "RICH" and p.loc["DDD", "candidate"]
    assert "DDD 30-day IV 50.0 vs forecast 26.0: +24.0 vol pts" in p.loc["DDD", "line"]
    assert "→ RICH" in p.loc["DDD", "line"]
