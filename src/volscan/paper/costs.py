"""Transaction-cost model for paper fills."""

from __future__ import annotations


def option_fill_price(mid: float, half_spread: float, buy: bool, cross_fraction: float) -> float:
    """Mid shifted toward the far side by a fraction of the half-spread."""
    slip = cross_fraction * max(half_spread, 0.0)
    return mid + slip if buy else max(mid - slip, 0.0)


def option_cross_cost(
    half_spread: float, contracts: float, multiplier: int, cross_fraction: float
) -> float:
    """Dollar cost of crossing, relative to marking at mid. Always >= 0."""
    return abs(contracts) * multiplier * cross_fraction * max(half_spread, 0.0)


def stock_cost(shares: float, price: float, bps: float, per_share: float) -> float:
    return abs(shares) * price * bps * 1e-4 + abs(shares) * per_share
