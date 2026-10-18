"""Assemble the daily signal table from surface metrics, forecasts, and earnings."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from volscan.config import SignalSettings
from volscan.data.earnings import earnings_day_returns, next_earnings, spans_earnings
from volscan.signals.earnings_adj import (
    ex_earnings_vol,
    historical_move_stats,
    split_event_variance,
)
from volscan.signals.rank import describe, rank_candidates
from volscan.signals.vrp import score_spreads


def _earnings_row(
    ticker: str,
    asof: date,
    tenor: int,
    iv_raw: float,
    slices: pd.DataFrame,
    earnings: list[date],
    prices: pd.DataFrame,
    cfg: SignalSettings,
) -> dict[str, object]:
    nxt = next_earnings(earnings, asof)
    in_window = spans_earnings(asof, asof + timedelta(days=tenor), nxt)
    row: dict[str, object] = {
        "next_earnings": nxt,
        "earnings_in_window": in_window,
        "base_iv": float("nan"),
        "implied_move": float("nan"),
        "implied_abs_move": float("nan"),
        "event_method": "",
    }
    if nxt is not None and not slices.empty:
        split = split_event_variance(slices, asof, nxt)
        if split is not None:
            row["implied_move"] = split.implied_move_sd
            row["implied_abs_move"] = split.implied_abs_move
            row["event_method"] = split.method
            if in_window:
                row["base_iv"] = ex_earnings_vol(iv_raw, tenor / 365.0, split.event_var)
    hist = earnings_day_returns(prices, earnings, asof, cfg.earnings_lookback_events)
    row.update(historical_move_stats(hist))
    return row


def build_signals(
    asof: date,
    metrics: pd.DataFrame,
    slices: pd.DataFrame,
    forecasts: pd.DataFrame,
    liquidity: pd.DataFrame,
    earnings: pd.DataFrame,
    prices: pd.DataFrame,
    history: pd.DataFrame,
    cfg: SignalSettings,
) -> pd.DataFrame:
    """One row per (ticker, tenor) with raw and VRP-adjusted scores.

    metrics:   ticker, tenor_days, atm_vol            (constant-maturity, today)
    slices:    ticker, expiry, T, atm_vol             (per-expiry fits, today)
    forecasts: ticker, tenor_days, model, vol         (today)
    liquidity: ticker, liquid, liquidity_note
    earnings:  ticker, earnings_date
    prices:    ticker, date, close
    history:   ticker, tenor_days, asof, spread, spread_raw (earlier days)
    """
    ens = forecasts[forecasts["model"] == "ensemble"][["ticker", "tenor_days", "vol"]].rename(
        columns={"vol": "forecast_vol"}
    )
    base = metrics[["ticker", "tenor_days", "atm_vol"]].rename(columns={"atm_vol": "iv_raw"})
    df = base.merge(ens, on=["ticker", "tenor_days"], how="inner")
    df = df[df["tenor_days"].isin([cfg.primary_tenor, cfg.secondary_tenor])]
    if df.empty:
        return pd.DataFrame()

    extra = []
    for r in df.itertuples():
        e_dates = (
            [
                pd.Timestamp(d).date()
                for d in earnings.loc[earnings["ticker"] == r.ticker, "earnings_date"]
            ]
            if not earnings.empty
            else []
        )
        extra.append(
            _earnings_row(
                str(r.ticker),
                asof,
                int(r.tenor_days),
                float(r.iv_raw),
                slices[slices["ticker"] == r.ticker],
                e_dates,
                prices[prices["ticker"] == r.ticker],
                cfg,
            )
        )
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    use_base = df["earnings_in_window"].astype(bool) & np.isfinite(df["base_iv"].astype(float))
    df["iv"] = np.where(use_base, df["base_iv"], df["iv_raw"])
    df["spread_raw"] = 100.0 * (df["iv_raw"] - df["forecast_vol"])
    df["spread"] = 100.0 * (df["iv"] - df["forecast_vol"])

    hist = history[history["asof"] < asof] if not history.empty else history
    scored = []
    for tenor, grp in df.groupby("tenor_days"):
        h = (
            hist[hist["tenor_days"] == tenor]
            if not hist.empty
            else pd.DataFrame(columns=["ticker", "asof", "spread", "spread_raw"])
        )
        s = score_spreads(
            grp, h[["ticker", "asof", "spread"]], cfg.vrp_lookback, cfg.min_history_for_z
        )
        raw_in = grp.assign(iv=grp["iv_raw"], spread=grp["spread_raw"])
        raw_hist = h[["ticker", "asof", "spread_raw"]].rename(columns={"spread_raw": "spread"})
        s_raw = score_spreads(raw_in, raw_hist, cfg.vrp_lookback, cfg.min_history_for_z)
        s["z_raw"] = s_raw["z"].to_numpy()
        scored.append(s)
    df = pd.concat(scored, ignore_index=True)

    df = df.merge(liquidity[["ticker", "liquid", "liquidity_note"]], on="ticker", how="left")
    df["liquid"] = df["liquid"].fillna(False).astype(bool)
    primary = rank_candidates(
        df[df["tenor_days"] == cfg.primary_tenor],
        cfg.entry_z,
        cfg.top_n,
        cfg.trade_through_earnings,
    )
    secondary = df[df["tenor_days"] != cfg.primary_tenor].copy()
    secondary["label"] = np.where(
        secondary["z"] >= cfg.entry_z,
        "RICH",
        np.where(secondary["z"] <= -cfg.entry_z, "CHEAP", "FAIR"),
    )
    secondary["tradeable"] = False
    secondary["candidate"] = False
    out = pd.concat([primary, secondary], ignore_index=True)
    out.insert(0, "asof", asof)
    out["line"] = [describe(r) for _, r in out.iterrows()]
    return out
