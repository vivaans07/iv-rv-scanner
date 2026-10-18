"""The nightly steps, wired to the DuckDB store.

Each step reads what the previous one wrote for the same `asof` date and
replaces its own output for that date, so any step can be re-run safely.

Paper positions are stored as one state row per position per day
(`position_states`). A step for day D starts from each position's latest
state strictly before D, so re-running day D reproduces it exactly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd

from volscan.config import Settings
from volscan.data.store import Store
from volscan.forecast.ensemble import ensemble_weights, evaluate_ticker, forecast_ticker
from volscan.paper.hedger import open_position, step_position
from volscan.paper.positions import LegQuote, Position
from volscan.pricing.black import black76_price, implied_vol
from volscan.signals.build import build_signals
from volscan.surface.cleaning import add_mid, liquidity_check
from volscan.surface.fit import fit_ticker_surface
from volscan.surface.svi import SviParams

log = logging.getLogger(__name__)
EVAL_REFRESH_DAYS = 7


def _d(x: object) -> date:
    return pd.Timestamp(x).date()


def _log_failures(store: Store, asof: date, step: str, failures: dict[str, str]) -> None:
    store.replace(
        "run_log",
        pd.DataFrame(
            {"asof": asof, "step": step, "ticker": list(failures), "error": list(failures.values())}
        ),
        {"asof": asof, "step": step},
    )


def rate_for(store: Store, asof: date, fallback: float) -> float:
    df = store.read("rates", "asof <= ?", [asof])
    if df.empty:
        return fallback
    return float(df.sort_values("asof")["rate"].iloc[-1])


def chain_dates(store: Store) -> list[date]:
    if not store.has_table("chains"):
        return []
    return [_d(x) for x in store.query("SELECT DISTINCT asof FROM chains ORDER BY asof")["asof"]]


# --------------------------------------------------------------------------- fit


def run_fit(store: Store, settings: Settings, asof: date) -> dict[str, str]:
    chains = store.read("chains", "asof = ?", [asof])
    if chains.empty:
        raise RuntimeError(f"no option chains stored for {asof}; run ingest first")
    rate = rate_for(store, asof, settings.rates.fallback_rate)
    failures: dict[str, str] = {}
    slices, metrics, quotes, liq = [], [], [], []
    for ticker in settings.tickers:
        chain = chains[chains["ticker"] == ticker]
        if chain.empty:
            continue
        chain = chain.assign(expiry=[_d(e) for e in chain["expiry"]])
        spot = float(chain["spot"].iloc[0])
        ok, note = liquidity_check(chain, spot, asof, settings.ingest.min_dte, settings.liquidity)
        liq.append({"asof": asof, "ticker": ticker, "liquid": ok, "liquidity_note": note})
        try:
            res = fit_ticker_surface(ticker, asof, chain, rate, settings)
        except Exception as exc:
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            log.error("fit failed for %s: %s", ticker, exc)
            continue
        slices.append(res.slices)
        metrics.append(res.metrics)
        quotes.append(res.quotes)
    store.replace("liquidity", pd.DataFrame(liq), {"asof": asof})
    for table, frames in (
        ("svi_slices", slices),
        ("surface_metrics", metrics),
        ("iv_quotes", quotes),
    ):
        store.replace(
            table,
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            {"asof": asof},
        )
    _log_failures(store, asof, "fit", failures)
    return failures


# ---------------------------------------------------------------------- forecast


def _weights(
    store: Store, ticker: str, prices: pd.DataFrame, asof: date, settings: Settings
) -> dict[str, float]:
    tenor = settings.signals.primary_tenor
    cached = store.read("forecast_eval", "ticker = ? AND asof <= ?", [ticker, asof])
    if not cached.empty:
        latest = _d(cached["asof"].max())
        if (asof - latest).days < EVAL_REFRESH_DAYS:
            return ensemble_weights(
                cached[pd.to_datetime(cached["asof"]) == pd.Timestamp(latest)],
                settings.forecast.ensemble,
            )
    scores = evaluate_ticker(prices, asof, tenor, settings.forecast)
    if not scores.empty:
        store.replace(
            "forecast_eval",
            scores.assign(asof=asof, ticker=ticker, tenor_days=tenor),
            {"asof": asof, "ticker": ticker},
        )
    return ensemble_weights(scores, settings.forecast.ensemble)


def _earnings_reaction_dates(store: Store, ticker: str) -> list[date]:
    df = store.read("earnings", "ticker = ?", [ticker])
    return [_d(x) for x in df["earnings_date"]] if not df.empty else []


def run_forecast(store: Store, settings: Settings, asof: date) -> dict[str, str]:
    failures: dict[str, str] = {}
    frames = []
    for ticker in settings.tickers:
        prices = store.read("prices", "ticker = ? AND date <= ?", [ticker, asof])
        if prices.empty:
            continue
        try:
            prices = _drop_earnings_days(prices, _earnings_reaction_dates(store, ticker))
            w = _weights(store, ticker, prices, asof, settings)
            fc = forecast_ticker(prices, asof, settings.term.tenors_days, settings.forecast, w)
            frames.append(fc.assign(asof=asof, ticker=ticker))
        except Exception as exc:
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            log.error("forecast failed for %s: %s", ticker, exc)
    store.replace(
        "forecasts",
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        {"asof": asof},
    )
    _log_failures(store, asof, "forecast", failures)
    return failures


def _drop_earnings_days(prices: pd.DataFrame, earnings: list[date]) -> pd.DataFrame:
    """Neutralize earnings reaction days so the forecast measures diffusive vol.

    The session after each announcement gets a flat bar at the prior close
    plus the median absolute daily return; its variance then looks like a
    typical day instead of a jump. Event variance is handled separately in
    the earnings adjustment, so counting it here as well would double count.
    """
    if not earnings:
        return prices
    df = prices.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    typical = float(np.log(df["close"]).diff().abs().median())
    for e in earnings:
        after = np.flatnonzero(dates > pd.Timestamp(e))
        if len(after) == 0 or after[0] == 0:
            continue
        i = after[0]
        prev = float(df.loc[i - 1, "close"])
        new_close = prev * math.exp(typical)
        ratio = new_close / float(df.loc[i, "close"])
        df.loc[i, ["open", "high", "low", "close"]] = [
            prev,
            max(prev, new_close) * math.exp(typical / 2),
            min(prev, new_close) * math.exp(-typical / 2),
            new_close,
        ]
        # Shift later bars so subsequent returns are unchanged.
        df.loc[i + 1 :, ["open", "high", "low", "close"]] *= ratio
    return df


# ----------------------------------------------------------------------- signals


def run_signals(store: Store, settings: Settings, asof: date) -> pd.DataFrame:
    metrics = store.read("surface_metrics", "asof = ?", [asof])
    forecasts = store.read("forecasts", "asof = ?", [asof])
    if metrics.empty or forecasts.empty:
        raise RuntimeError(f"surface metrics or forecasts missing for {asof}")
    slices = store.read("svi_slices", "asof = ?", [asof])
    liquidity = store.read("liquidity", "asof = ?", [asof])
    earnings = store.read("earnings")
    prices = store.read("prices", "date <= ?", [asof])
    history = store.read("signals", "asof < ?", [asof])
    if not history.empty:
        history = history.assign(asof=[_d(x) for x in history["asof"]])
    sig = build_signals(
        asof,
        metrics,
        slices.assign(expiry=[_d(e) for e in slices["expiry"]]) if not slices.empty else slices,
        forecasts,
        liquidity,
        earnings if not earnings.empty else pd.DataFrame(columns=["ticker", "earnings_date"]),
        prices,
        history,
        settings.signals,
    )
    store.replace("signals", sig, {"asof": asof})
    return sig


# ------------------------------------------------------------------------- trade


class StoreMarket:
    """MarketView over one ticker's stored chain and SVI fits for one day."""

    def __init__(
        self,
        ticker: str,
        asof: date,
        chain: pd.DataFrame,
        slices: pd.DataFrame,
        rate: float,
        forecast_vol: float,
    ) -> None:
        self.ticker = ticker
        self.asof = asof
        self.rate = rate
        self.forecast_vol = forecast_vol
        chain = add_mid(chain)
        self.spot = float(chain["spot"].iloc[0])
        self._quotes = {
            (_d(r.expiry), float(r.strike), str(r.option_type)): (
                float(r.mid),
                0.5 * float(r.ask - r.bid),
            )
            for r in chain.itertuples()
            if r.bid > 0 and r.ask >= r.bid
        }
        self._strikes: dict[date, list[float]] = {}
        for (e, k, _), _q in self._quotes.items():
            self._strikes.setdefault(e, []).append(k)
        self._slices = {
            _d(r.expiry): (float(r.forward), float(r.T), SviParams(r.a, r.b, r.rho, r.m, r.sigma))
            for r in slices.itertuples()
        }
        spreads = pd.Series({k: v[1] for k, v in self._quotes.items()})
        self._median_half_spread = float(spreads.median()) if not spreads.empty else 0.05

    def forward(self, expiry: date) -> float:
        if expiry in self._slices:
            return self._slices[expiry][0]
        return self.spot * math.exp(self.rate * max((expiry - self.asof).days, 0) / 365.0)

    def expiries(self) -> list[date]:
        # Only expiries with a fitted smile can be marked consistently.
        return sorted(e for e in self._strikes if e in self._slices)

    def strikes(self, expiry: date) -> list[float]:
        return sorted(set(self._strikes.get(expiry, [])))

    def option(self, expiry: date, strike: float, option_type: str) -> LegQuote | None:
        T = (expiry - self.asof).days / 365.0
        if T <= 0:
            return None
        F = self.forward(expiry)
        surf = self._slices.get(expiry)
        iv = float(surf[2].implied_vol(math.log(strike / F), T)) if surf else float("nan")
        q = self._quotes.get((expiry, float(strike), option_type))
        if q is not None:
            mid, half = q
            if not np.isfinite(iv):
                iv = implied_vol(mid, F, strike, T, self.rate, option_type == "C")
            if not np.isfinite(iv):
                return None
            return LegQuote(mid, half, iv, "market")
        if surf is None:
            return None
        mid = float(black76_price(F, strike, T, self.rate, iv, option_type == "C"))
        return LegQuote(mid, self._median_half_spread, iv, "model")


class NoDataMarket:
    """Used when a ticker has no snapshot today; every quote is missing."""

    def __init__(self, asof: date, rate: float) -> None:
        self.asof, self.rate, self.spot = asof, rate, float("nan")

    def forward(self, expiry: date) -> float:
        return float("nan")

    def option(self, expiry: date, strike: float, option_type: str) -> LegQuote | None:
        return None


def _load_positions(store: Store, book: str, asof: date) -> list[Position]:
    states = store.read("position_states", "book = ? AND asof < ?", [book, asof])
    if states.empty:
        return []
    states = states.sort_values("asof").groupby("position_id").tail(1)
    out = []
    for rec in states.drop(columns=["book", "asof"]).to_dict("records"):
        pos = Position.from_record(rec)
        if pos.is_open:
            out.append(pos)
    return out


def _split_factor(prices: pd.DataFrame, after: date, upto: date) -> float:
    if prices.empty or "split_factor" not in prices:
        return 1.0
    dates = pd.to_datetime(prices["date"])
    window = prices[(dates > pd.Timestamp(after)) & (dates <= pd.Timestamp(upto))]
    return float(window["split_factor"].astype(float).prod()) if not window.empty else 1.0


def run_trade(store: Store, settings: Settings, asof: date, book: str = "live") -> dict[str, int]:
    cfg = settings.paper
    rate = rate_for(store, asof, settings.rates.fallback_rate)
    chains = store.read("chains", "asof = ?", [asof])
    slices = store.read("svi_slices", "asof = ?", [asof])
    signals = store.read(
        "signals", "asof = ? AND tenor_days = ?", [asof, settings.signals.primary_tenor]
    )
    fc = store.read(
        "forecasts",
        "asof = ? AND model = 'ensemble' AND tenor_days = ?",
        [asof, settings.signals.primary_tenor],
    )
    fc_vol = dict(zip(fc["ticker"], fc["vol"], strict=False)) if not fc.empty else {}
    z_by_ticker = (
        dict(zip(signals["ticker"], signals["z"], strict=False)) if not signals.empty else {}
    )

    def market(ticker: str) -> StoreMarket | None:
        ch = chains[chains["ticker"] == ticker] if not chains.empty else chains
        sl = slices[slices["ticker"] == ticker] if not slices.empty else slices
        if ch.empty or sl.empty:
            return None
        return StoreMarket(ticker, asof, ch, sl, rate, float(fc_vol.get(ticker, float("nan"))))

    marks: list[dict[str, object]] = []
    fills: list[dict[str, object]] = []
    states: list[Position] = []
    positions = _load_positions(store, book, asof)
    for pos in positions:
        m = market(pos.ticker)
        prices = store.read("prices", "ticker = ?", [pos.ticker])
        split = _split_factor(prices, pos.last_asof or pos.opened, asof)
        z = z_by_ticker.get(pos.ticker)
        res = step_position(
            pos,
            m if m is not None else NoDataMarket(asof, rate),
            cfg,
            signal_z=float(z) if z is not None else None,
            split_factor=split,
            exit_z=settings.signals.exit_z,
        )
        marks.append(res.mark)
        fills.extend(asdict(f) for f in res.fills)
        states.append(pos)

    open_now = [p for p in positions if p.is_open]
    book_vega = sum(abs(p.qty * p.multiplier * sum(p.last_vega.values()) / 100) for p in open_now)
    held = {p.ticker for p in open_now}
    opened = 0
    if not signals.empty:
        cands = signals[signals["candidate"].astype(bool)].copy()
        cands["abs_z"] = cands["z"].abs()
        for r in cands.sort_values("abs_z", ascending=False).itertuples():
            if len(held) >= cfg.max_positions:
                break
            if r.ticker in held or book_vega + cfg.vega_per_vol_point > cfg.max_portfolio_vega:
                continue
            m = market(str(r.ticker))
            if m is None:
                continue
            pid = f"{book}-{asof.isoformat()}-{r.ticker}"
            opened_res = open_position(
                pid, str(r.ticker), m, str(r.label), float(r.z), bool(r.earnings_in_window), cfg
            )
            if opened_res is None or opened_res.position is None:
                continue
            marks.append(opened_res.mark)
            fills.extend(asdict(f) for f in opened_res.fills)
            states.append(opened_res.position)
            held.add(str(r.ticker))
            book_vega += abs(opened_res.mark["vega_dollars"])
            opened += 1

    keys = {"asof": asof, "book": book}
    store.replace("marks", pd.DataFrame(marks).assign(book=book) if marks else pd.DataFrame(), keys)
    store.replace("fills", pd.DataFrame(fills).assign(book=book) if fills else pd.DataFrame(), keys)
    store.replace(
        "position_states",
        pd.DataFrame([{**p.to_record(), "asof": asof, "book": book} for p in states])
        if states
        else pd.DataFrame(),
        keys,
    )
    return {"stepped": len(positions), "opened": opened, "open": sum(p.is_open for p in states)}


def current_positions(store: Store, book: str = "live", asof: date | None = None) -> pd.DataFrame:
    where, params = "book = ?", [book]
    if asof:
        where += " AND asof <= ?"
        params.append(asof)  # type: ignore[arg-type]
    states = store.read("position_states", where, params)
    if states.empty:
        return states
    return states.sort_values("asof").groupby("position_id").tail(1).reset_index(drop=True)


def reset_book(store: Store, book: str) -> None:
    for table in ("marks", "fills", "position_states"):
        if store.has_table(table):
            store.con.execute(f"DELETE FROM {table} WHERE book = ?", [book])
