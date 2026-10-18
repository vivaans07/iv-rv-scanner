"""Command-line entry points: `volscan <command>`."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Annotated

import pandas as pd
import typer

from volscan.config import Settings, load_settings
from volscan.data.ingest import NY, market_date, run_ingest
from volscan.data.providers import get_provider
from volscan.data.store import Store
from volscan.pipeline import (
    chain_dates,
    run_fit,
    run_forecast,
    run_signals,
    run_trade,
)

app = typer.Typer(
    add_completion=False, help="Implied-vs-realized volatility scanner (paper trading only)."
)
log = logging.getLogger("volscan")

TickersOpt = Annotated[
    str | None, typer.Option("--tickers", "-t", help="Comma-separated subset of the universe.")
]
AsofOpt = Annotated[
    str | None, typer.Option("--asof", help="Session date YYYY-MM-DD (default: latest snapshot).")
]


def _setup(tickers: str | None) -> Settings:
    settings = load_settings(tickers=[t.strip() for t in tickers.split(",")] if tickers else None)
    settings.paths.log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(settings.paths.log_dir / "volscan.log"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy in ("yfinance", "urllib3", "peewee", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return settings


def _asof(store: Store, asof: str | None) -> date:
    if asof:
        return date.fromisoformat(asof)
    dates = chain_dates(store)
    if not dates:
        raise typer.BadParameter("no snapshots stored yet; run `volscan ingest` first")
    return dates[-1]


def is_trading_day(d: date) -> bool:
    import exchange_calendars as xcals

    return bool(xcals.get_calendar("XNYS").is_session(pd.Timestamp(d)))


@app.command()
def ingest(tickers: TickersOpt = None) -> None:
    """Snapshot option chains, price history, and earnings dates."""
    settings = _setup(tickers)
    with Store(settings.paths.db_path) as store:
        res = run_ingest(get_provider(settings.provider), store, settings)
    typer.echo(f"ingested {len(res.ok)} tickers for {res.asof}; {len(res.failed)} failed")
    for t, err in res.failed.items():
        typer.echo(f"  {t}: {err}")


@app.command()
def fit(tickers: TickersOpt = None, asof: AsofOpt = None) -> None:
    """Fit SVI surfaces and constant-maturity metrics."""
    settings = _setup(tickers)
    with Store(settings.paths.db_path) as store:
        d = _asof(store, asof)
        failures = run_fit(store, settings, d)
    typer.echo(f"fit surfaces for {d}; {len(failures)} failed")


@app.command()
def forecast(tickers: TickersOpt = None, asof: AsofOpt = None) -> None:
    """Forecast realized vol with HAR, GARCH, GJR-GARCH and the ensemble."""
    settings = _setup(tickers)
    with Store(settings.paths.db_path) as store:
        d = _asof(store, asof)
        failures = run_forecast(store, settings, d)
    typer.echo(f"forecast {d}; {len(failures)} failed")


@app.command()
def signals(tickers: TickersOpt = None, asof: AsofOpt = None) -> None:
    """Score and rank IV-vs-forecast spreads."""
    settings = _setup(tickers)
    with Store(settings.paths.db_path) as store:
        d = _asof(store, asof)
        sig = run_signals(store, settings, d)
    primary = sig[sig["tenor_days"] == settings.signals.primary_tenor]
    for line in primary[primary["label"] != "FAIR"]["line"]:
        typer.echo(line)
    typer.echo(f"{int(primary['candidate'].sum())} trade candidates on {d}")


@app.command()
def trade(tickers: TickersOpt = None, asof: AsofOpt = None) -> None:
    """Mark, hedge, and open paper positions."""
    settings = _setup(tickers)
    with Store(settings.paths.db_path) as store:
        d = _asof(store, asof)
        out = run_trade(store, settings, d)
    typer.echo(f"{d}: stepped {out['stepped']}, opened {out['opened']}, open now {out['open']}")


@app.command()
def report(tickers: TickersOpt = None, asof: AsofOpt = None) -> None:
    """Write the Markdown and HTML report."""
    from volscan.report.daily import build_report

    settings = _setup(tickers)
    with Store(settings.paths.db_path) as store:
        d = _asof(store, asof)
        paths = build_report(store, settings, d)
    typer.echo(f"wrote {paths['markdown']} and {paths['html']}")


@app.command("run-daily")
def run_daily(
    tickers: TickersOpt = None,
    force: Annotated[bool, typer.Option(help="Run even if today is not an NYSE session.")] = False,
) -> None:
    """Ingest, fit, forecast, score, trade, and report, in order."""
    from volscan.report.daily import build_report

    settings = _setup(tickers)
    today = market_date(datetime.now(NY))
    if not force and not is_trading_day(today):
        typer.echo(f"{today} is not an NYSE session; skipping (use --force to override)")
        raise typer.Exit(0)
    with Store(settings.paths.db_path) as store:
        res = run_ingest(get_provider(settings.provider), store, settings, today=today)
        if not res.ok:
            typer.echo("ingest returned no data; stopping")
            raise typer.Exit(1)
        d = res.asof
        run_fit(store, settings, d)
        run_forecast(store, settings, d)
        run_signals(store, settings, d)
        out = run_trade(store, settings, d)
        paths = build_report(store, settings, d)
    typer.echo(
        f"{d}: {len(res.ok)} ingested, {len(res.failed)} failed; opened {out['opened']}, "
        f"open {out['open']}. Report: {paths['html']}"
    )


@app.command()
def backtest(
    start: Annotated[str, typer.Option(help="First session YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="Last session YYYY-MM-DD")],
    tickers: TickersOpt = None,
    refit: Annotated[
        bool, typer.Option(help="Re-fit surfaces and forecasts for each day.")
    ] = False,
) -> None:
    """Replay stored snapshots through signals and a separate 'backtest' paper book."""
    from volscan.paper.attribution import summary_stats
    from volscan.pipeline import current_positions, reset_book

    settings = _setup(tickers)
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    with Store(settings.paths.db_path) as store:
        days = [d for d in chain_dates(store) if s <= d <= e]
        if not days:
            raise typer.BadParameter(f"no stored snapshots between {s} and {e}")
        reset_book(store, "backtest")
        for d in days:
            have_fit = not store.read("surface_metrics", '"asof" = ?', [d]).empty
            if refit or not have_fit:
                run_fit(store, settings, d)
            if refit or store.read("forecasts", '"asof" = ?', [d]).empty:
                run_forecast(store, settings, d)
            run_signals(store, settings, d)
            out = run_trade(store, settings, d, book="backtest")
            typer.echo(f"{d}: opened {out['opened']}, open {out['open']}")
        marks = store.read("marks", "book = 'backtest'")
        stats = summary_stats(marks, current_positions(store, "backtest"))
    for k, v in stats.items():
        typer.echo(f"{k:>20}: {v:,.2f}")


if __name__ == "__main__":
    app()
