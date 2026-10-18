"""Turn scored spreads into a ranked list of trade candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify(z: float, entry_z: float) -> str:
    if z >= entry_z:
        return "RICH"
    if z <= -entry_z:
        return "CHEAP"
    return "FAIR"


def describe(row: pd.Series) -> str:
    """One readable line per ticker for the report."""
    earn = "yes" if bool(row.get("earnings_in_window", False)) else "no"
    line = (
        f"{row['ticker']} {int(row['tenor_days'])}-day IV {100 * row['iv_raw']:.1f} vs forecast "
        f"{100 * row['forecast_vol']:.1f}: {row['spread_raw']:+.1f} vol pts "
        f"(z = {row['z']:+.1f}, earnings in window: {earn}) → {row['label']}"
    )
    if earn == "yes" and np.isfinite(row.get("base_iv", np.nan)):
        line += f" [ex-earnings IV {100 * row['base_iv']:.1f}]"
    return line


def rank_candidates(
    signals: pd.DataFrame, entry_z: float, top_n: int, trade_through_earnings: bool
) -> pd.DataFrame:
    """Top N rich and top N cheap names that pass liquidity and earnings rules.

    Expects columns: ticker, z (scored on the ex-earnings spread where an event
    falls in the window), liquid, earnings_in_window, base_iv.

    When an event is in the window and `trade_through_earnings` is off, the
    name is only tradeable if the event variance could be stripped out (base_iv
    is finite), so the z-score reflects diffusive vol rather than the jump.
    """
    df = signals.copy()
    df["label"] = [classify(z, entry_z) for z in df["z"]]
    df["tradeable"] = df["liquid"].astype(bool) & (df["label"] != "FAIR")
    if not trade_through_earnings:
        no_split = df["earnings_in_window"].astype(bool) & ~np.isfinite(df["base_iv"].astype(float))
        df.loc[no_split, "tradeable"] = False
    df["abs_z"] = df["z"].abs()
    rich = df[df["tradeable"] & (df["label"] == "RICH")].nlargest(top_n, "abs_z")
    cheap = df[df["tradeable"] & (df["label"] == "CHEAP")].nlargest(top_n, "abs_z")
    df["candidate"] = df.index.isin(rich.index.union(cheap.index))
    return df.sort_values("abs_z", ascending=False).drop(columns="abs_z").reset_index(drop=True)
