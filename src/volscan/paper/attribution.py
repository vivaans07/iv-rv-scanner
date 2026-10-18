"""Roll daily position marks up into book-level P&L and summary statistics.

"Signal earned" is gamma/theta carry plus vega P&L: what the volatility view
made or lost. "Hedging and costs" is hedging error, residual delta, and
transaction costs: what it cost to run the view through a discretely hedged
book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ["gamma_theta", "vega_pnl", "hedging_error", "residual", "costs"]


def add_rollups(marks: pd.DataFrame) -> pd.DataFrame:
    df = marks.copy()
    df["signal_pnl"] = df["gamma_theta"] + df["vega_pnl"]
    df["hedge_cost_pnl"] = df["hedging_error"] + df["residual"] - df["costs"]
    return df


def book_daily(marks: pd.DataFrame) -> pd.DataFrame:
    """Daily book P&L by component, plus cumulative columns."""
    if marks.empty:
        return pd.DataFrame()
    df = add_rollups(marks)
    cols = ["total_pnl", "signal_pnl", "hedge_cost_pnl", *COMPONENTS]
    daily = df.groupby("asof")[cols].sum().sort_index()
    for c in ("total_pnl", "signal_pnl", "hedge_cost_pnl"):
        daily[f"cum_{c}"] = daily[c].cumsum()
    return daily


def max_drawdown(cum: pd.Series) -> float:
    if cum.empty:
        return 0.0
    peak = cum.cummax().clip(lower=0.0)
    return float((cum - peak).min())


def per_position(marks: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    df = add_rollups(marks)
    agg = df.groupby("position_id")[
        ["total_pnl", "signal_pnl", "hedge_cost_pnl", *COMPONENTS]
    ].sum()
    meta = positions.set_index("position_id")[
        ["ticker", "label", "earnings_in_window", "status", "opened", "closed", "exit_reason"]
    ]
    return meta.join(agg, how="left").fillna({"total_pnl": 0.0}).reset_index()


def summary_stats(marks: pd.DataFrame, positions: pd.DataFrame) -> dict[str, float]:
    if marks.empty or positions.empty:
        return {}
    daily = book_daily(marks)
    pos = per_position(marks, positions)
    closed = pos[pos["status"] == "closed"]
    r = daily["total_pnl"]
    sharpe = (
        float(r.mean() / r.std(ddof=1) * np.sqrt(252))
        if len(r) > 2 and r.std() > 0
        else float("nan")
    )
    out: dict[str, float] = {
        "n_positions": float(len(pos)),
        "n_closed": float(len(closed)),
        "total_pnl": float(r.sum()),
        "signal_pnl": float(daily["signal_pnl"].sum()),
        "hedge_cost_pnl": float(daily["hedge_cost_pnl"].sum()),
        "costs": float(daily["costs"].sum()),
        "hit_rate": float((closed["total_pnl"] > 0).mean()) if len(closed) else float("nan"),
        "avg_pnl_per_trade": float(closed["total_pnl"].mean()) if len(closed) else float("nan"),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(daily["cum_total_pnl"]),
    }
    for label in ("RICH", "CHEAP"):
        out[f"pnl_{label.lower()}"] = float(pos.loc[pos["label"] == label, "total_pnl"].sum())
    earn = pos["earnings_in_window"].astype(bool)
    out["pnl_earnings"] = float(pos.loc[earn, "total_pnl"].sum())
    out["pnl_no_earnings"] = float(pos.loc[~earn, "total_pnl"].sum())
    return out
