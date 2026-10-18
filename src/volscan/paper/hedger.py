"""Open, mark, delta-hedge, attribute, and close straddle positions.

Each daily step does, in order:
  1. Mark both legs at mid (falling back to the model surface if a quote is
     missing, or carrying the previous mark if the ticker has no data).
  2. Attribute yesterday-to-today P&L using yesterday's Greeks:
       gamma/theta carry  0.5 * Gamma * S^2 * [(dS/S)^2 - sigma_imp^2 * dt]
       vega               sum_legs vega * d(sigma_leg)
       residual delta     (delta_at_implied + shares) * dS
                          zero when hedging at implied vol; non-zero when the
                          hedge uses a different vol
       hedging error      everything else in the mark-to-market P&L:
                          discrete rebalancing, jumps, vanna/volga, rates
  3. Exit if a rule triggers, otherwise rebalance the share hedge to zero delta.
  4. Charge costs, then check the components against the cash ledger.

The ledger check is independent of the Greeks: equity is cash plus option
marks plus shares times spot, and every fill moves cash. If the attribution
and the ledger disagree, the step raises instead of silently drifting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import numpy as np

from volscan.config import PaperSettings
from volscan.paper.costs import option_cross_cost, option_fill_price, stock_cost
from volscan.paper.positions import LEGS, LegGreeks, LegQuote, MarketView, Position, leg_greeks

LEDGER_TOL = 1e-6
STALE_LIMIT = 3
ASSIGNMENT_EXTRINSIC = 0.05


class LedgerMismatchError(RuntimeError):
    pass


class TradableMarket(MarketView, Protocol):
    """A MarketView that can also list what is quoted (needed to open trades)."""

    forecast_vol: float

    def expiries(self) -> list[date]: ...

    def strikes(self, expiry: date) -> list[float]: ...


@dataclass
class Fill:
    position_id: str
    asof: date
    instrument: str  # "C", "P", or "STOCK"
    quantity: float  # contracts or shares, signed (+ buy, - sell)
    price: float
    cost: float


@dataclass
class StepResult:
    mark: dict[str, Any]
    fills: list[Fill] = field(default_factory=list)
    position: Position | None = None


def _years(a: date, b: date) -> float:
    return (b - a).days / 365.0


def _straddle_greeks(
    m: MarketView, expiry: date, strike: float, quotes: dict[str, LegQuote], vol: float | None
) -> dict[str, LegGreeks]:
    S, F = m.spot, m.forward(expiry)
    T = max(_years(m.asof, expiry), 1e-6)
    return {
        t: leg_greeks(S, F, strike, T, m.rate, vol if vol is not None else q.iv, t)
        for t, q in quotes.items()
    }


def _hedge_vol(pos: Position, m: MarketView) -> float | None:
    if pos.hedge_vol == "forecast":
        fv = getattr(m, "forecast_vol", float("nan"))
        return float(fv) if np.isfinite(fv) and fv > 0 else None
    return None


def _rebalance(pos: Position, m: MarketView, target: float, cfg: PaperSettings) -> Fill | None:
    trade = target - pos.shares
    if abs(trade) < 1e-9:
        return None
    cost = stock_cost(trade, m.spot, cfg.stock_cost_bps, cfg.stock_cost_per_share)
    pos.cash -= trade * m.spot + cost
    pos.shares = target
    return Fill(pos.position_id, m.asof, "STOCK", trade, m.spot, cost)


def _remember(
    pos: Position, m: MarketView, quotes: dict[str, LegQuote], g: dict[str, LegGreeks]
) -> None:
    pos.last_asof = m.asof
    pos.last_spot = m.spot
    pos.last_value = sum(q.mid for q in quotes.values())
    pos.last_iv = {t: q.iv for t, q in quotes.items()}
    pos.last_vega = {t: g[t].vega for t in LEGS}
    scale = pos.qty * pos.multiplier
    pos.last_gamma = scale * sum(g[t].gamma for t in LEGS)
    pos.last_delta_imp = scale * sum(g[t].delta for t in LEGS)


def straddle_iv(iv: dict[str, float], vega: dict[str, float]) -> float:
    """Vega-weighted vol of the two legs."""
    wsum = sum(vega.values())
    if wsum <= 0:
        return float(np.mean(list(iv.values())))
    return sum(iv[t] * vega[t] for t in iv) / wsum


def open_position(
    position_id: str,
    ticker: str,
    m: TradableMarket,
    label: str,
    z: float,
    earnings_in_window: bool,
    cfg: PaperSettings,
) -> StepResult | None:
    """Open an ATM straddle near the target tenor and put on the initial hedge."""
    min_dte = cfg.exit_days_before_expiry + 2
    eligible = [e for e in m.expiries() if (e - m.asof).days >= min_dte]
    if not eligible:
        return None
    expiry = min(eligible, key=lambda e: abs((e - m.asof).days - cfg.target_dte))
    F = m.forward(expiry)
    for strike in sorted(m.strikes(expiry), key=lambda k: abs(k - F)):
        q = {t: m.option(expiry, strike, t) for t in LEGS}
        if all(v is not None for v in q.values()):
            quotes: dict[str, LegQuote] = {t: v for t, v in q.items() if v is not None}
            break
    else:
        return None
    # The ticker-level liquidity screen looks at the nearest expiry; the traded
    # expiry can still be quoted much wider, so check the actual legs.
    for v in quotes.values():
        if (
            v.source != "market"
            or v.mid <= 0
            or 2 * v.half_spread / v.mid > cfg.max_entry_rel_spread
        ):
            return None

    g = _straddle_greeks(m, expiry, strike, quotes, None)
    vega_per_contract = cfg.contract_multiplier * sum(g[t].vega for t in LEGS) / 100.0
    if vega_per_contract <= 0:
        return None
    contracts = max(1, round(cfg.vega_per_vol_point / vega_per_contract))
    qty = -contracts if label == "RICH" else contracts

    pos = Position(
        position_id=position_id,
        ticker=ticker,
        opened=m.asof,
        expiry=expiry,
        strike=float(strike),
        qty=float(qty),
        multiplier=cfg.contract_multiplier,
        label=label,
        entry_z=z,
        earnings_in_window=earnings_in_window,
        hedge_vol=cfg.hedge_vol,
    )
    fills, costs = [], 0.0
    buy = qty > 0
    for t in LEGS:
        price = option_fill_price(
            quotes[t].mid, quotes[t].half_spread, buy, cfg.spread_cross_fraction
        )
        cost = option_cross_cost(
            quotes[t].half_spread, qty, pos.multiplier, cfg.spread_cross_fraction
        )
        pos.cash -= qty * pos.multiplier * price
        fills.append(Fill(position_id, m.asof, t, qty, price, cost))
        costs += cost

    hedge_g = _straddle_greeks(m, expiry, strike, quotes, _hedge_vol(pos, m))
    target = -qty * pos.multiplier * sum(hedge_g[t].delta for t in LEGS)
    f = _rebalance(pos, m, target, cfg)
    if f:
        fills.append(f)
        costs += f.cost
    _remember(pos, m, quotes, g)
    total = pos.equity(m.spot, pos.last_value)
    mark = _mark_row(pos, m, quotes, g, total=total, costs=costs, gross=0.0)
    _check_ledger(mark)
    return StepResult(mark, fills, pos)


def apply_split(pos: Position, factor: float) -> None:
    """Restate a position after an N-for-1 split (factor N).

    Option terms are adjusted so the economics are unchanged: strike / N and
    N times the contracts (OCC usually adjusts the deliverable instead; the
    value is the same). Prior marks are restated so the next day's P&L does
    not show a fake loss.
    """
    pos.strike /= factor
    pos.qty *= factor
    pos.shares *= factor
    pos.last_spot /= factor
    pos.last_value /= factor
    # Contracts scale by N and per-unit gamma by N (price and spot both / N).
    pos.last_gamma *= factor * factor
    pos.last_delta_imp *= factor
    pos.last_vega = {t: v / factor for t, v in pos.last_vega.items()}
    pos.flags.append(f"split_{factor:g}")


def _exit_reason(
    pos: Position, m: MarketView, signal_z: float | None, exit_z: float, cfg: PaperSettings
) -> str:
    if (pos.expiry - m.asof).days <= cfg.exit_days_before_expiry:
        return "near_expiry"
    if pos.days_held >= cfg.holding_days:
        return "holding_period"
    if signal_z is not None and np.isfinite(signal_z):
        reverted = abs(signal_z) < exit_z
        flipped = np.sign(signal_z) != np.sign(pos.entry_z)
        if reverted or flipped:
            return "signal_reverted"
    return ""


def step_position(
    pos: Position,
    m: MarketView,
    cfg: PaperSettings,
    signal_z: float | None = None,
    split_factor: float = 1.0,
    force_exit: str = "",
    exit_z: float = 0.5,
) -> StepResult:
    """Advance an open position to m.asof. Mutates `pos`."""
    if not pos.is_open:
        raise ValueError(f"position {pos.position_id} is closed")
    if split_factor != 1.0:
        apply_split(pos, split_factor)

    q = {t: m.option(pos.expiry, pos.strike, t) for t in LEGS}
    if any(v is None for v in q.values()):
        return _stale_step(pos, m, cfg)
    quotes: dict[str, LegQuote] = {t: v for t, v in q.items() if v is not None}
    pos.stale_days = 0
    if any(v.source != "market" for v in quotes.values()):
        pos.flags.append("model_mark")

    assert pos.last_asof is not None
    S, Sp = m.spot, pos.last_spot
    dS = S - Sp
    dt = _years(pos.last_asof, m.asof)
    value = sum(v.mid for v in quotes.values())
    equity_prev = pos.equity(Sp, pos.last_value)

    opt_pnl = pos.qty * pos.multiplier * (value - pos.last_value)
    hedge_pnl = pos.shares * dS
    sig_prev = straddle_iv(pos.last_iv, pos.last_vega)
    gamma_theta = 0.5 * pos.last_gamma * Sp * Sp * ((dS / Sp) ** 2 - sig_prev**2 * dt)
    vega_pnl = (
        pos.qty
        * pos.multiplier
        * sum(pos.last_vega[t] * (quotes[t].iv - pos.last_iv[t]) for t in LEGS)
    )
    residual = (pos.last_delta_imp + pos.shares) * dS
    gross = opt_pnl + hedge_pnl
    hedging_error = gross - gamma_theta - vega_pnl - residual

    g = _straddle_greeks(m, pos.expiry, pos.strike, quotes, None)
    pos.days_held += 1
    _flag_assignment(pos, m, quotes)

    fills: list[Fill] = []
    costs = 0.0
    reason = force_exit or _exit_reason(pos, m, signal_z, exit_z, cfg)
    if reason:
        costs += _close(pos, m, quotes, cfg, reason, fills)
        value_after = 0.0
    else:
        hedge_g = _straddle_greeks(m, pos.expiry, pos.strike, quotes, _hedge_vol(pos, m))
        target = -pos.qty * pos.multiplier * sum(hedge_g[t].delta for t in LEGS)
        f = _rebalance(pos, m, target, cfg)
        if f:
            fills.append(f)
            costs += f.cost
        value_after = value

    equity_now = pos.cash + pos.qty * pos.multiplier * value_after + pos.shares * S
    _remember(pos, m, quotes, g)
    mark = _mark_row(
        pos,
        m,
        quotes,
        g,
        total=equity_now - equity_prev,
        costs=costs,
        gross=gross,
        gamma_theta=gamma_theta,
        vega_pnl=vega_pnl,
        hedging_error=hedging_error,
        residual=residual,
        realized_move=(dS / Sp) if Sp else 0.0,
        implied_move=sig_prev * math.sqrt(dt),
    )
    _check_ledger(mark)
    return StepResult(mark, fills)


def _close(
    pos: Position,
    m: MarketView,
    quotes: dict[str, LegQuote],
    cfg: PaperSettings,
    reason: str,
    fills: list[Fill],
) -> float:
    costs = 0.0
    buy_back = pos.qty < 0
    for t in LEGS:
        price = option_fill_price(
            quotes[t].mid, quotes[t].half_spread, buy_back, cfg.spread_cross_fraction
        )
        cost = option_cross_cost(
            quotes[t].half_spread, pos.qty, pos.multiplier, cfg.spread_cross_fraction
        )
        pos.cash += pos.qty * pos.multiplier * price
        fills.append(Fill(pos.position_id, m.asof, t, -pos.qty, price, cost))
        costs += cost
    f = _rebalance(pos, m, 0.0, cfg)
    if f:
        fills.append(f)
        costs += f.cost
    pos.status, pos.closed, pos.exit_reason = "closed", m.asof, reason
    return costs


def _stale_step(pos: Position, m: MarketView, cfg: PaperSettings) -> StepResult:
    """No quotes today: carry the last mark forward; close after STALE_LIMIT days."""
    pos.stale_days += 1
    pos.flags.append("stale_mark")
    fills: list[Fill] = []
    if pos.stale_days > STALE_LIMIT:
        # Close at the last known marks. Nothing trades, so equity is unchanged.
        pos.cash += pos.qty * pos.multiplier * pos.last_value + pos.shares * pos.last_spot
        fills.append(Fill(pos.position_id, m.asof, "C+P", -pos.qty, pos.last_value, 0.0))
        pos.shares = 0.0
        pos.status, pos.closed, pos.exit_reason = "closed", m.asof, "data_missing"
    mark = {
        "position_id": pos.position_id,
        "ticker": pos.ticker,
        "asof": m.asof,
        "status": pos.status,
        "stale": True,
        **{k: 0.0 for k in _PNL_FIELDS},
        "spot": pos.last_spot,
        "straddle_value": pos.last_value,
        "shares": pos.shares,
        "cash": pos.cash,
        "flags": ",".join(sorted(set(pos.flags))),
    }
    return StepResult(mark, fills)


_PNL_FIELDS = (
    "total_pnl",
    "gross_pnl",
    "gamma_theta",
    "vega_pnl",
    "hedging_error",
    "residual",
    "costs",
    "realized_move",
    "implied_move",
)


def _flag_assignment(pos: Position, m: MarketView, quotes: dict[str, LegQuote]) -> None:
    """Short ITM legs with almost no time value are candidates for early exercise."""
    if pos.qty >= 0:
        return
    intrinsic = {"C": max(m.spot - pos.strike, 0.0), "P": max(pos.strike - m.spot, 0.0)}
    for t in LEGS:
        if intrinsic[t] > 0 and quotes[t].mid - intrinsic[t] < ASSIGNMENT_EXTRINSIC:
            pos.flags.append("assignment_risk")


def _mark_row(
    pos: Position,
    m: MarketView,
    quotes: dict[str, LegQuote],
    g: dict[str, LegGreeks],
    total: float,
    costs: float,
    gross: float,
    gamma_theta: float = 0.0,
    vega_pnl: float = 0.0,
    hedging_error: float = 0.0,
    residual: float = 0.0,
    realized_move: float = 0.0,
    implied_move: float = 0.0,
) -> dict[str, Any]:
    scale = pos.qty * pos.multiplier
    open_now = pos.is_open
    return {
        "position_id": pos.position_id,
        "ticker": pos.ticker,
        "asof": m.asof,
        "status": pos.status,
        "stale": False,
        "total_pnl": total,
        "gross_pnl": gross,
        "gamma_theta": gamma_theta,
        "vega_pnl": vega_pnl,
        "hedging_error": hedging_error,
        "residual": residual,
        "costs": costs,
        "realized_move": realized_move,
        "implied_move": implied_move,
        "spot": m.spot,
        "straddle_value": sum(q.mid for q in quotes.values()),
        "iv": straddle_iv({t: q.iv for t, q in quotes.items()}, {t: g[t].vega for t in LEGS}),
        "shares": pos.shares,
        "cash": pos.cash,
        "delta": (scale * sum(g[t].delta for t in LEGS) + pos.shares) if open_now else 0.0,
        "gamma": scale * sum(g[t].gamma for t in LEGS) if open_now else 0.0,
        "vega_dollars": scale * sum(g[t].vega for t in LEGS) / 100.0 if open_now else 0.0,
        "dte": (pos.expiry - m.asof).days,
        "flags": ",".join(sorted(set(pos.flags))),
    }


def _check_ledger(mark: dict[str, Any]) -> None:
    parts = (
        mark["gamma_theta"]
        + mark["vega_pnl"]
        + mark["hedging_error"]
        + mark["residual"]
        - mark["costs"]
    )
    if abs(parts - mark["total_pnl"]) > LEDGER_TOL * max(1.0, abs(mark["total_pnl"])):
        raise LedgerMismatchError(
            f"{mark['position_id']} {mark['asof']}: components {parts:.6f} "
            f"!= ledger {mark['total_pnl']:.6f}"
        )
