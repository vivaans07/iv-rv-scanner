from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from volscan.config import PaperSettings
from volscan.paper.attribution import book_daily, max_drawdown, summary_stats
from volscan.paper.hedger import open_position, step_position
from volscan.paper.positions import Position
from volscan.paper.simulate import FlatVolMarket, simulate_hedged_straddle

CFG = PaperSettings()
START = date(2026, 1, 5)


def test_fair_vol_hedged_pnl_is_zero_on_average() -> None:
    # Realized == implied: gamma gains and theta losses cancel in expectation.
    res = simulate_hedged_straddle(300, 20, 0.25, 0.25, "CHEAP", CFG, seed=1)
    gross = res["gross_pnl"]
    stderr = gross.std(ddof=1) / np.sqrt(len(gross))
    assert abs(gross.mean()) < 3 * stderr
    # And small relative to what a 5-vol-point miss would earn on this size.
    assert abs(gross.mean()) < 0.1 * CFG.vega_per_vol_point * 5


def test_short_straddle_earns_carry_when_realized_below_implied() -> None:
    res = simulate_hedged_straddle(200, 20, 0.30, 0.20, "RICH", CFG, seed=2)
    gt = res["gamma_theta"]
    assert gt.mean() > 0
    assert gt.mean() / (gt.std(ddof=1) / np.sqrt(len(gt))) > 5
    assert res["gross_pnl"].mean() > 0
    # Long the same straddle loses.
    long_res = simulate_hedged_straddle(200, 20, 0.30, 0.20, "CHEAP", CFG, seed=2)
    assert long_res["gross_pnl"].mean() < 0


def test_components_sum_to_ledger_with_costs_and_jumps() -> None:
    cfg = CFG.model_copy(update={"stock_cost_bps": 2.0, "stock_cost_per_share": 0.005})
    res = simulate_hedged_straddle(
        20, 15, 0.3, 0.3, "RICH", cfg, seed=3, half_spread=0.05, jump_day=7, jump_size=-0.08
    )
    parts = res[["gamma_theta", "vega_pnl", "hedging_error", "residual"]].sum(axis=1) - res["costs"]
    np.testing.assert_allclose(parts, res["total_pnl"], rtol=1e-9, atol=1e-6)
    assert (res["costs"] > 0).all()
    # A -8% gap hurts a short-gamma book beyond what daily gamma/theta predicts.
    assert res["hedging_error"].mean() < 0


def test_forecast_hedge_vol_shows_up_in_residual() -> None:
    cfg = CFG.model_copy(update={"hedge_vol": "forecast"})
    m = FlatVolMarket(START, 100.0, 0.0, 0.3, [START + timedelta(days=45)], forecast_vol=0.2)
    res = open_position("x", "SIM", m, "RICH", 2.0, False, cfg)
    assert res is not None and res.position is not None
    m2 = FlatVolMarket(START + timedelta(days=1), 103.0, 0.0, 0.3, m.expiry_list, forecast_vol=0.2)
    mark = step_position(res.position, m2, cfg).mark
    assert abs(mark["residual"]) > 0


def _open(label: str = "RICH") -> tuple[Position, FlatVolMarket]:
    m = FlatVolMarket(START, 100.0, 0.0, 0.25, [START + timedelta(days=35)], half_spread=0.02)
    res = open_position("t1", "SIM", m, label, 2.0, False, CFG)
    assert res is not None and res.position is not None
    return res.position, m


def test_sizing_and_entry_costs() -> None:
    pos, m = _open()
    assert pos.qty < 0
    vega_dollars = abs(pos.qty) * pos.multiplier * sum(pos.last_vega.values()) / 100
    assert vega_dollars == pytest.approx(CFG.vega_per_vol_point, rel=0.1)
    assert pos.strike == 100.0
    # Entry day equity is just the cost of crossing half the half-spread on two legs.
    expected_cost = 2 * abs(pos.qty) * 100 * 0.5 * 0.02
    assert pos.equity(m.spot, pos.last_value) == pytest.approx(
        -expected_cost - abs(pos.shares) * 100 * 1e-4, rel=1e-6
    )


def test_exit_rules() -> None:
    pos, m = _open()
    m2 = FlatVolMarket(START + timedelta(days=1), 100.0, 0.0, 0.25, m.expiry_list)
    mark = step_position(pos, m2, CFG, signal_z=0.2).mark
    assert pos.status == "closed" and pos.exit_reason == "signal_reverted"
    assert pos.shares == 0.0 and mark["delta"] == 0.0

    pos, m = _open()
    for i in range(1, 40):
        d = START + timedelta(days=i)
        if d >= pos.expiry - timedelta(days=CFG.exit_days_before_expiry):
            step_position(pos, FlatVolMarket(d, 100.0, 0.0, 0.25, m.expiry_list), CFG, signal_z=2.0)
            break
        step_position(pos, FlatVolMarket(d, 100.0, 0.0, 0.25, m.expiry_list), CFG, signal_z=2.0)
        if not pos.is_open:
            break
    assert pos.exit_reason in {"holding_period", "near_expiry"}


def test_missing_data_carries_forward_then_closes() -> None:
    pos, m = _open()
    for i in range(1, 6):
        d = START + timedelta(days=i)
        miss = {(m.expiry_list[0], pos.strike, "C")}
        mark = step_position(
            pos, FlatVolMarket(d, 100.0, 0.0, 0.25, m.expiry_list, missing=miss), CFG
        ).mark
        assert mark["stale"]
        if not pos.is_open:
            break
    assert pos.exit_reason == "data_missing"
    assert "stale_mark" in pos.flags


def test_split_restates_position_without_fake_pnl() -> None:
    pos, m = _open("CHEAP")
    m2 = FlatVolMarket(START + timedelta(days=1), 50.0, 0.0, 0.25, m.expiry_list, strike_step=0.5)
    mark = step_position(pos, m2, CFG, split_factor=2.0).mark
    assert pos.strike == 50.0
    # One day of theta on a long straddle: small loss, not a 50% crash.
    assert -0.02 * CFG.vega_per_vol_point * 25 < mark["total_pnl"] < 0


def test_record_round_trip() -> None:
    pos, _ = _open()
    again = Position.from_record(pos.to_record())
    assert again == pos


def test_summary_stats() -> None:
    marks = pd.DataFrame(
        {
            "position_id": ["a", "a", "b", "b"],
            "asof": [date(2026, 1, 5), date(2026, 1, 6)] * 2,
            "total_pnl": [-10.0, 50.0, -5.0, -20.0],
            "gamma_theta": [0.0, 40.0, 0.0, -10.0],
            "vega_pnl": [0.0, 15.0, 0.0, -8.0],
            "hedging_error": [0.0, -3.0, 0.0, -1.0],
            "residual": [0.0, 0.0, 0.0, 0.0],
            "costs": [10.0, 2.0, 5.0, 1.0],
        }
    )
    positions = pd.DataFrame(
        {
            "position_id": ["a", "b"],
            "ticker": ["X", "Y"],
            "label": ["RICH", "CHEAP"],
            "earnings_in_window": [False, True],
            "status": ["closed", "closed"],
            "opened": [date(2026, 1, 5)] * 2,
            "closed": [date(2026, 1, 6)] * 2,
            "exit_reason": ["x", "y"],
        }
    )
    s = summary_stats(marks, positions)
    assert s["total_pnl"] == pytest.approx(15.0)
    assert s["hit_rate"] == pytest.approx(0.5)
    assert s["pnl_rich"] == pytest.approx(40.0) and s["pnl_cheap"] == pytest.approx(-25.0)
    assert s["signal_pnl"] + s["hedge_cost_pnl"] == pytest.approx(s["total_pnl"])
    daily = book_daily(marks)
    assert daily["cum_total_pnl"].iloc[-1] == pytest.approx(15.0)
    assert max_drawdown(pd.Series([0.0, 10.0, 4.0, 12.0, 1.0])) == pytest.approx(-11.0)
