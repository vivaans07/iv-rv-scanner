"""Position state and the market interface the hedger needs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Protocol

import numpy as np
import pandas as pd

from volscan.pricing.black import black76_greeks, spot_delta, spot_gamma

LEGS = ("C", "P")


@dataclass(frozen=True)
class LegQuote:
    mid: float
    half_spread: float
    iv: float  # implied vol used for Greeks (surface vol at this strike)
    source: str  # "market" or "model"


class MarketView(Protocol):
    """Everything the hedger needs to know about one ticker on one day."""

    asof: date
    spot: float
    rate: float

    def forward(self, expiry: date) -> float: ...

    def option(self, expiry: date, strike: float, option_type: str) -> LegQuote | None: ...


@dataclass(frozen=True)
class LegGreeks:
    value: float  # per unit of the option
    delta: float  # dV/dS (spot)
    gamma: float  # d2V/dS2 (spot)
    vega: float  # dV/dsigma per 1.00 vol


def leg_greeks(
    S: float, F: float, K: float, T: float, r: float, iv: float, option_type: str
) -> LegGreeks:
    g = black76_greeks(F, K, T, r, iv, option_type == "C")
    return LegGreeks(
        value=float(g.price),
        delta=float(spot_delta(g.delta, F, S)),
        gamma=float(spot_gamma(g.gamma, F, S)),
        vega=float(g.vega),
    )


@dataclass
class Position:
    """A delta-hedged ATM straddle.

    qty is in contracts: positive is long the straddle (one call and one put
    per contract), negative is short. All dollar amounts are for the whole
    position. `cash` is the running cash balance from every fill, so
    cash + option value + shares * spot is the position's equity.
    """

    position_id: str
    ticker: str
    opened: date
    expiry: date
    strike: float
    qty: float
    multiplier: int
    label: str
    entry_z: float
    earnings_in_window: bool
    hedge_vol: str
    status: str = "open"
    closed: date | None = None
    exit_reason: str = ""
    cash: float = 0.0
    shares: float = 0.0
    last_asof: date | None = None
    last_spot: float = float("nan")
    last_value: float = float("nan")  # straddle value per unit (call + put mids)
    last_iv: dict[str, float] = field(default_factory=dict)
    last_vega: dict[str, float] = field(default_factory=dict)
    last_gamma: float = 0.0  # position dollar gamma per $1^2 (qty * mult * gamma)
    last_delta_imp: float = 0.0  # position delta at implied vol
    days_held: int = 0
    stale_days: int = 0
    flags: list[str] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def equity(self, spot: float, value: float) -> float:
        return self.cash + self.qty * self.multiplier * value + self.shares * spot

    def to_record(self) -> dict[str, object]:
        rec = asdict(self)
        rec["last_iv_c"] = self.last_iv.get("C", np.nan)
        rec["last_iv_p"] = self.last_iv.get("P", np.nan)
        rec["last_vega_c"] = self.last_vega.get("C", np.nan)
        rec["last_vega_p"] = self.last_vega.get("P", np.nan)
        rec["flags"] = ",".join(sorted(set(self.flags)))
        for key in ("last_iv", "last_vega"):
            rec.pop(key)
        return rec

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> Position:
        data = dict(rec)
        iv = {"C": float(data.pop("last_iv_c")), "P": float(data.pop("last_iv_p"))}
        vega = {"C": float(data.pop("last_vega_c")), "P": float(data.pop("last_vega_p"))}
        flags = str(data.pop("flags") or "")
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for k in ("opened", "expiry", "closed", "last_asof"):
            kwargs[k] = _as_date(kwargs.get(k))
        pos = cls(**kwargs)
        pos.last_iv, pos.last_vega = iv, vega
        pos.flags = [f for f in flags.split(",") if f]
        return pos


def _as_date(v: Any) -> date | None:
    if v is None or (isinstance(v, float) and v != v) or v is pd.NaT:
        return None
    if isinstance(v, date) and not isinstance(v, pd.Timestamp):
        return v
    return pd.Timestamp(v).date()
