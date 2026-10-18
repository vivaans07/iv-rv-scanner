"""Daily Markdown and HTML report."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from volscan.config import Settings
from volscan.data.store import Store
from volscan.paper.attribution import book_daily, summary_stats
from volscan.pipeline import current_positions, rate_for
from volscan.surface.svi import SviParams

N_CHARTS = 4


@dataclass
class Chart:
    name: str
    title: str
    png: bytes

    @property
    def b64(self) -> str:
        return base64.b64encode(self.png).decode()


def _fig_bytes(fig: Any) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def smile_chart(ticker: str, quotes: pd.DataFrame, slices: pd.DataFrame) -> Chart:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    s = slices.sort_values("T")
    ax1.plot(s["T"] * 365, 100 * s["atm_vol"], marker="o")
    ax1.set_xlabel("days to expiry")
    ax1.set_ylabel("ATM implied vol (%)")
    ax1.set_title(f"{ticker} term structure")
    near = s.iloc[(s["T"] * 365 - 30).abs().argsort()[:2]].sort_values("T")
    for r in near.itertuples():
        q = quotes[pd.to_datetime(quotes["expiry"]) == pd.Timestamp(r.expiry)]
        p = SviParams(r.a, r.b, r.rho, r.m, r.sigma)
        k = (
            np.linspace(q["k"].min(), q["k"].max(), 100)
            if not q.empty
            else np.linspace(-0.3, 0.3, 100)
        )
        label = f"{round(r.T * 365)}d"
        ax2.scatter(q["k"], 100 * q["iv"], s=10, alpha=0.6)
        ax2.plot(k, 100 * p.implied_vol(k, r.T), label=f"SVI {label}")
    ax2.set_xlabel("log-moneyness ln(K/F)")
    ax2.set_ylabel("implied vol (%)")
    ax2.set_title(f"{ticker} smile")
    ax2.legend(fontsize=8)
    return Chart(f"smile_{ticker}", f"{ticker} term structure and smile", _fig_bytes(fig))


def pnl_chart(daily: pd.DataFrame) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = pd.to_datetime(daily.index)
    ax.plot(x, daily["cum_total_pnl"], label="total", color="black", lw=2)
    ax.plot(x, daily["cum_signal_pnl"], label="signal (gamma/theta + vega)")
    ax.plot(x, daily["cum_hedge_cost_pnl"], label="hedging error + costs")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("cumulative P&L ($)")
    ax.legend(fontsize=8)
    ax.set_title("Paper book P&L")
    fig.autofmt_xdate()
    return Chart("pnl", "Cumulative P&L split", _fig_bytes(fig))


def _what_happened(
    asof: date, sig: pd.DataFrame, stats: dict[str, float], today_pnl: pd.Series | None
) -> str:
    parts = []
    if not sig.empty:
        rich = int((sig["label"] == "RICH").sum())
        cheap = int((sig["label"] == "CHEAP").sum())
        med = float(sig["spread_raw"].median())
        mode = sig["z_mode"].mode().iloc[0] if "z_mode" in sig else "n/a"
        parts.append(
            f"On {asof:%B %d, %Y} the median 30-day spread of implied over forecast vol was "
            f"{med:+.1f} vol points across {len(sig)} names; {rich} screened rich and {cheap} cheap "
            f"(scores mostly {mode})."
        )
        top = sig.reindex(sig["z"].abs().sort_values(ascending=False).index).head(1)
        if not top.empty:
            t = top.iloc[0]
            parts.append(f"The strongest reading was {t['ticker']} at z = {t['z']:+.1f}.")
    if today_pnl is not None and not today_pnl.empty:
        parts.append(
            f"The paper book made {today_pnl['total_pnl']:+,.0f} today: {today_pnl['signal_pnl']:+,.0f} "
            f"from the volatility view and {today_pnl['hedge_cost_pnl']:+,.0f} from hedging and costs."
        )
    if stats:
        parts.append(
            f"Since inception: {stats['total_pnl']:+,.0f} total, of which the signal earned "
            f"{stats['signal_pnl']:+,.0f} and hedging/costs contributed {stats['hedge_cost_pnl']:+,.0f} "
            f"over {int(stats['n_positions'])} positions."
        )
    return " ".join(parts) or "No signals or positions yet."


def build_report(
    store: Store, settings: Settings, asof: date, book: str = "live"
) -> dict[str, Path]:
    out_dir = settings.paths.report_dir
    asset_dir = out_dir / f"{asof.isoformat()}_files"
    asset_dir.mkdir(parents=True, exist_ok=True)

    failures = store.read("run_log", '"asof" = ?', [asof])
    sig = store.read("signals", '"asof" = ?', [asof])
    primary = sig[sig["tenor_days"] == settings.signals.primary_tenor] if not sig.empty else sig
    secondary = sig[sig["tenor_days"] == settings.signals.secondary_tenor] if not sig.empty else sig
    if not primary.empty:
        primary = primary.reindex(primary["z"].sort_values(ascending=False).index)
    slices = store.read("svi_slices", '"asof" = ?', [asof])
    quotes = store.read("iv_quotes", '"asof" = ?', [asof])
    metrics = store.read("surface_metrics", '"asof" = ?', [asof])

    charts: list[Chart] = []
    if not primary.empty and not slices.empty:
        flagged = primary[primary["label"] != "FAIR"]
        top = flagged.reindex(flagged["z"].abs().sort_values(ascending=False).index).head(N_CHARTS)
        for t in top["ticker"]:
            charts.append(
                smile_chart(t, quotes[quotes["ticker"] == t], slices[slices["ticker"] == t])
            )

    marks = store.read("marks", 'book = ? AND "asof" <= ?', [book, asof])
    positions = current_positions(store, book, asof)
    stats = summary_stats(marks, positions) if not marks.empty else {}
    daily = book_daily(marks) if not marks.empty else pd.DataFrame()
    today_pnl = None
    if not daily.empty:
        idx = pd.to_datetime(daily.index)
        if (idx == pd.Timestamp(asof)).any():
            today_pnl = daily.loc[idx == pd.Timestamp(asof)].iloc[0]
        charts.append(pnl_chart(daily))
    open_pos = pd.DataFrame()
    if not positions.empty:
        open_ids = positions.loc[positions["status"] == "open", "position_id"]
        latest = (
            marks[marks["position_id"].isin(open_ids)]
            .sort_values("asof")
            .groupby("position_id")
            .tail(1)
        )
        to_date = marks.groupby("position_id")["total_pnl"].sum().rename("pnl_to_date")
        open_pos = (
            positions[positions["position_id"].isin(open_ids)][
                ["position_id", "ticker", "label", "qty", "strike", "expiry", "opened", "flags"]
            ]
            .merge(
                latest[["position_id", "dte", "spot", "iv", "delta", "gamma", "vega_dollars"]],
                on="position_id",
            )
            .merge(to_date, left_on="position_id", right_index=True)
        )

    for c in charts:
        (asset_dir / f"{c.name}.png").write_bytes(c.png)

    env = Environment(
        loader=PackageLoader("volscan.report", "templates"),
        autoescape=select_autoescape(enabled_extensions=("html",), default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["money"] = money
    ctx = {
        "asof": asof,
        "rate": rate_for(store, asof, settings.rates.fallback_rate),
        "rate_source": _rate_source(store, asof),
        "n_scanned": len(settings.tickers),
        "n_fit": int(metrics["ticker"].nunique()) if not metrics.empty else 0,
        "n_signals": len(primary),
        "failures": failures.to_dict("records") if not failures.empty else [],
        "primary": primary.to_dict("records") if not primary.empty else [],
        "secondary": secondary.to_dict("records") if not secondary.empty else [],
        "candidates": primary[primary["candidate"].astype(bool)].to_dict("records")
        if not primary.empty
        else [],
        "earnings": primary[primary["earnings_in_window"].astype(bool)].to_dict("records")
        if not primary.empty
        else [],
        "open_positions": open_pos.to_dict("records"),
        "stats": stats,
        "charts": charts,
        "asset_dir": asset_dir.name,
        "summary": _what_happened(asof, primary, stats, today_pnl),
        "settings": settings,
    }
    md_path = out_dir / f"{asof.isoformat()}.md"
    html_path = out_dir / f"{asof.isoformat()}.html"
    md_path.write_text(env.get_template("daily.md.j2").render(**ctx), encoding="utf-8")
    html_path.write_text(env.get_template("daily.html.j2").render(**ctx), encoding="utf-8")
    return {"markdown": md_path, "html": html_path}


def money(x: float, sign: bool = True) -> str:
    """Whole dollars with thousands separators; Jinja's % formatting lacks ','."""
    if x is None or x != x:
        return "n/a"
    return format(float(x), "+,.0f" if sign else ",.0f")


def _rate_source(store: Store, asof: date) -> str:
    df = store.read("rates", '"asof" <= ?', [asof])
    return str(df.sort_values("asof")["source"].iloc[-1]) if not df.empty else "fallback constant"
