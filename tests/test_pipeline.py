from datetime import date

import pandas as pd

from tests.fakes import FakeProvider
from volscan.config import load_settings
from volscan.data.ingest import run_ingest
from volscan.data.store import Store
from volscan.pipeline import current_positions, run_fit, run_forecast, run_signals, run_trade
from volscan.report.daily import build_report

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def test_multi_day_pipeline_end_to_end(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = load_settings(tickers=TICKERS)
    settings.paths.db_path = tmp_path / "t.duckdb"
    settings.paths.report_dir = tmp_path / "reports"
    settings.ingest.pause_between_tickers = 0.0
    settings.forecast.eval_start_fraction = 0.9  # keep the walk-forward short in tests
    provider = FakeProvider(iv_bump={"AAA": 0.10, "EEE": -0.08})
    days = list(pd.bdate_range("2026-03-02", periods=4).date)

    with Store(settings.paths.db_path) as store:
        for d in days:
            provider.asof = d
            run_ingest(provider, store, settings, today=d, sleep=lambda _s: None)
            assert not run_fit(store, settings, d)
            assert not run_forecast(store, settings, d)
            sig = run_signals(store, settings, d)
            run_trade(store, settings, d)

        primary = sig[sig["tenor_days"] == 30].set_index("ticker")
        assert primary.loc["AAA", "label"] == "RICH"
        assert primary.loc["EEE", "label"] == "CHEAP"

        pos = current_positions(store)
        assert set(pos["ticker"]) >= {"AAA", "EEE"}
        assert set(pos.loc[pos["ticker"] == "AAA", "label"]) == {"RICH"}
        marks = store.read("marks")
        assert len(marks) >= 2 * len(days) - 1

        # Re-running the last day must reproduce it exactly (idempotent).
        before = store.read("marks", "asof = ?", [days[-1]]).sort_values("position_id")
        run_trade(store, settings, days[-1])
        after = store.read("marks", "asof = ?", [days[-1]]).sort_values("position_id")
        pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))

        paths = build_report(store, settings, days[-1])
    md = paths["markdown"].read_text()
    assert "AAA 30-day IV" in md and "→ RICH" in md
    assert "Open positions" in md
    assert paths["html"].read_text().startswith("<!doctype html>")
    assert any(p.suffix == ".png" for p in (tmp_path / "reports" / f"{days[-1]}_files").iterdir())
    assert date(2026, 3, 5) == days[-1]
