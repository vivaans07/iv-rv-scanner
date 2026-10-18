from volscan.config import load_settings


def test_settings_load_and_universe_size() -> None:
    s = load_settings()
    assert 40 <= len(s.tickers) <= 60
    assert s.conventions.trading_days_per_year == 252
    assert s.paths.db_path.name.endswith(".duckdb")


def test_ticker_override() -> None:
    s = load_settings(tickers=["spy", "aapl"])
    assert s.tickers == ["SPY", "AAPL"]
