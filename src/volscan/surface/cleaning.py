"""Quote filtering, forward estimation, and implied vols.

Listed US equity options are American. I price them as European options on
the forward implied by put-call parity and only use out-of-the-money quotes
(puts below the forward, calls above). OTM American options carry very little
early-exercise premium, so the European approximation error is small there;
deep ITM quotes are where it would bite, and those are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from volscan.config import CleaningSettings, LiquiditySettings
from volscan.pricing.black import black76_greeks, implied_vol_vec


@dataclass(frozen=True)
class CleanStats:
    raw: int
    kept: int
    dropped_zero_bid: int
    dropped_crossed: int
    dropped_wide: int
    dropped_illiquid: int


def add_mid(chain: pd.DataFrame) -> pd.DataFrame:
    out = chain.copy()
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rel_spread"] = (out["ask"] - out["bid"]) / out["mid"]
    return out


def filter_quotes(chain: pd.DataFrame, cfg: CleaningSettings) -> tuple[pd.DataFrame, CleanStats]:
    """Drop quotes that would only add noise to the smile.

    Order matters for the stats only: each quote is attributed to the first
    rule that removes it.
    """
    df = add_mid(chain)
    raw = len(df)
    zero_bid = ~(df["bid"] >= cfg.min_bid)
    crossed = ~zero_bid & (df["ask"] < df["bid"])
    wide = ~zero_bid & ~crossed & (df["rel_spread"] > cfg.max_rel_spread)
    illiquid = ~zero_bid & ~crossed & ~wide & (df["open_interest"] < cfg.min_open_interest)
    keep = ~(zero_bid | crossed | wide | illiquid)
    stats = CleanStats(
        raw=raw,
        kept=int(keep.sum()),
        dropped_zero_bid=int(zero_bid.sum()),
        dropped_crossed=int(crossed.sum()),
        dropped_wide=int(wide.sum()),
        dropped_illiquid=int(illiquid.sum()),
    )
    return df[keep].reset_index(drop=True), stats


def year_fraction(asof: date, expiry: date) -> float:
    """ACT/365 calendar time to expiry. Options decay over calendar time."""
    return (pd.Timestamp(expiry) - pd.Timestamp(asof)).days / 365.0


def estimate_forward(
    quotes: pd.DataFrame, spot: float, T: float, r: float, band: float
) -> tuple[float, int]:
    """Median forward implied by put-call parity on near-the-money strikes.

    C - P = DF (F - K)  =>  F = K + (C - P) / DF

    Returns (forward, number of strikes used). Falls back to spot * e^{rT}
    when no call/put pair survives cleaning near the money.
    """
    df = np.exp(-r * T)
    near = quotes[(quotes["strike"] - spot).abs() <= band * spot]
    calls = near[near["option_type"] == "C"].set_index("strike")["mid"]
    puts = near[near["option_type"] == "P"].set_index("strike")["mid"]
    common = calls.index.intersection(puts.index)
    if len(common) == 0:
        return float(spot * np.exp(r * T)), 0
    implied = common.to_numpy() + (calls[common].to_numpy() - puts[common].to_numpy()) / df
    return float(np.median(implied)), len(common)


def implied_vol_table(
    quotes: pd.DataFrame, asof: date, spot: float, r: float, band: float
) -> pd.DataFrame:
    """OTM implied vols per expiry with log-moneyness and total variance.

    Output columns: expiry, T, forward, n_parity, strike, option_type, mid,
    bid, ask, rel_spread, iv, k, w, vega.
    """
    frames = []
    for expiry, grp in quotes.groupby("expiry"):
        T = year_fraction(asof, pd.Timestamp(expiry).date())
        if T <= 0:
            continue
        F, n_pairs = estimate_forward(grp, spot, T, r, band)
        otm = grp[
            ((grp["option_type"] == "P") & (grp["strike"] < F))
            | ((grp["option_type"] == "C") & (grp["strike"] >= F))
        ].copy()
        if otm.empty:
            continue
        is_call = (otm["option_type"] == "C").to_numpy()
        otm["iv"] = implied_vol_vec(
            otm["mid"].to_numpy(), F, otm["strike"].to_numpy(), T, r, is_call
        )
        otm = otm[np.isfinite(otm["iv"])]
        if otm.empty:
            continue
        otm["expiry"] = pd.Timestamp(expiry).date()
        otm["T"] = T
        otm["forward"] = F
        otm["n_parity"] = n_pairs
        otm["k"] = np.log(otm["strike"] / F)
        otm["w"] = otm["iv"] ** 2 * T
        otm["vega"] = black76_greeks(
            F, otm["strike"].to_numpy(), T, r, otm["iv"].to_numpy(), otm["option_type"] == "C"
        ).vega
        frames.append(otm)
    if not frames:
        return pd.DataFrame()
    cols = [
        "expiry",
        "T",
        "forward",
        "n_parity",
        "strike",
        "option_type",
        "mid",
        "bid",
        "ask",
        "rel_spread",
        "iv",
        "k",
        "w",
        "vega",
    ]
    return pd.concat(frames, ignore_index=True)[cols].sort_values(["T", "k"]).reset_index(drop=True)


def liquidity_check(
    chain: pd.DataFrame, spot: float, asof: date, min_dte: int, cfg: LiquiditySettings
) -> tuple[bool, str]:
    """Screen a ticker on its nearest eligible expiry's at-the-money call and put."""
    df = add_mid(chain)
    dte = (pd.to_datetime(df["expiry"]) - pd.Timestamp(asof)).dt.days
    df = df[dte >= min_dte]
    if df.empty:
        return False, "no eligible expiries"
    first = df["expiry"].min()
    near = df[df["expiry"] == first]
    atm_strike = near.loc[(near["strike"] - spot).abs().idxmin(), "strike"]
    atm = near[near["strike"] == atm_strike]
    oi = float(atm["open_interest"].sum())
    vol = float(atm["volume"].sum())
    spread = float(atm["rel_spread"].max())
    reasons = []
    if oi < cfg.min_atm_open_interest:
        reasons.append(f"ATM OI {oi:.0f} < {cfg.min_atm_open_interest}")
    if vol < cfg.min_atm_volume:
        reasons.append(f"ATM volume {vol:.0f} < {cfg.min_atm_volume}")
    if not np.isfinite(spread) or spread > cfg.max_atm_rel_spread:
        reasons.append(f"ATM spread {spread:.0%} > {cfg.max_atm_rel_spread:.0%}")
    return not reasons, "; ".join(reasons) or "ok"
