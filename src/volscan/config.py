"""Typed configuration loaded from config/*.yaml and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class IngestSettings(BaseModel):
    min_dte: int = 7
    max_dte: int = 180
    history_years: int = 4
    max_retries: int = 4
    backoff_seconds: float = 2.0
    pause_between_tickers: float = 0.5


class CleaningSettings(BaseModel):
    max_rel_spread: float = 0.25
    min_open_interest: int = 10
    min_bid: float = 0.01
    forward_strike_band: float = 0.05


class LiquiditySettings(BaseModel):
    min_atm_open_interest: int = 100
    min_atm_volume: int = 10
    max_atm_rel_spread: float = 0.10


class SviSettings(BaseModel):
    min_quotes: int = 6
    restarts: int = 8
    weight: Literal["vega", "inverse_spread", "equal"] = "vega"
    arb_grid_points: int = 201


class TermSettings(BaseModel):
    tenors_days: list[int] = Field(default_factory=lambda: [30, 60, 90])


class ForecastSettings(BaseModel):
    har_window: int = 1000
    garch_window: int = 1000
    min_history: int = 300
    ensemble: Literal["inverse_qlike", "mean"] = "inverse_qlike"
    eval_start_fraction: float = 0.6


class SignalSettings(BaseModel):
    primary_tenor: int = 30
    secondary_tenor: int = 60
    vrp_lookback: int = 252
    min_history_for_z: int = 40
    entry_z: float = 1.5
    exit_z: float = 0.5
    top_n: int = 5
    earnings_lookback_events: int = 10
    trade_through_earnings: bool = False


class PaperSettings(BaseModel):
    target_dte: int = 30
    vega_per_vol_point: float = 1000.0
    max_portfolio_vega: float = 10000.0
    max_positions: int = 10
    holding_days: int = 10
    exit_days_before_expiry: int = 5
    spread_cross_fraction: float = 0.5
    hedge_vol: Literal["implied", "forecast"] = "implied"
    stock_cost_bps: float = 1.0
    stock_cost_per_share: float = 0.0
    contract_multiplier: int = 100


class RateSettings(BaseModel):
    fred_series: str = "DTB3"
    fallback_rate: float = 0.04


class Conventions(BaseModel):
    trading_days_per_year: int = 252
    calendar_days_per_year: int = 365


class Paths(BaseModel):
    db_path: Path
    report_dir: Path
    log_dir: Path


class Settings(BaseModel):
    tickers: list[str]
    etfs: list[str] = Field(default_factory=list)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    cleaning: CleaningSettings = Field(default_factory=CleaningSettings)
    liquidity: LiquiditySettings = Field(default_factory=LiquiditySettings)
    svi: SviSettings = Field(default_factory=SviSettings)
    term: TermSettings = Field(default_factory=TermSettings)
    forecast: ForecastSettings = Field(default_factory=ForecastSettings)
    signals: SignalSettings = Field(default_factory=SignalSettings)
    paper: PaperSettings = Field(default_factory=PaperSettings)
    rates: RateSettings = Field(default_factory=RateSettings)
    conventions: Conventions = Field(default_factory=Conventions)
    provider: str = "yfinance"
    paths: Paths


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(config_dir: Path | None = None, tickers: list[str] | None = None) -> Settings:
    """Load YAML config plus env overrides. `tickers` narrows the universe."""
    load_dotenv(PROJECT_ROOT / ".env")
    config_dir = config_dir or PROJECT_ROOT / "config"
    with open(config_dir / "universe.yaml") as fh:
        universe = yaml.safe_load(fh) or {}
    with open(config_dir / "settings.yaml") as fh:
        raw = yaml.safe_load(fh) or {}

    paths = Paths(
        db_path=_resolve(os.getenv("VOLSCAN_DB_PATH", "data/volscan.duckdb")),
        report_dir=_resolve(os.getenv("VOLSCAN_REPORT_DIR", "reports/output")),
        log_dir=_resolve(os.getenv("VOLSCAN_LOG_DIR", "logs")),
    )
    selected = [t.upper() for t in tickers] if tickers else universe.get("tickers", [])
    return Settings(
        tickers=selected,
        etfs=universe.get("etfs", []),
        provider=os.getenv("VOLSCAN_PROVIDER", "yfinance"),
        paths=paths,
        **raw,
    )
