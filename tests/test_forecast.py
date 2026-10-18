from datetime import date

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from tests.fakes import FakeProvider
from volscan.config import ForecastSettings
from volscan.forecast.ensemble import combine, ensemble_weights, evaluate, forecast_ticker, qlike
from volscan.forecast.garch import aggregate_variance_path, fit_garch, garch_forecast
from volscan.forecast.har import fit_har, har_target
from volscan.forecast.realized import (
    annualize,
    close_to_close,
    daily_variance,
    garman_klass,
    parkinson,
    total_var_to_vol,
    trading_days_in,
    yang_zhang,
    yang_zhang_daily,
)


def _gbm_bars(vol: float, n: int = 1500, seed: int = 0, steps: int = 1000) -> pd.DataFrame:
    """OHLC bars from a finely simulated GBM so range estimators are testable."""
    rng = np.random.default_rng(seed)
    dt = 1 / (252 * steps)
    paths = rng.normal(-0.5 * vol**2 * dt, vol * np.sqrt(dt), (n, steps)).cumsum(axis=1)
    level = np.concatenate([[0.0], paths[:-1, -1]]).cumsum()
    intraday = level[:, None] + paths
    open_ = np.exp(level)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2010-01-01", periods=n).date,
            "open": open_,
            "high": np.maximum(np.exp(intraday.max(axis=1)), open_),
            "low": np.minimum(np.exp(intraday.min(axis=1)), open_),
            "close": np.exp(intraday[:, -1]),
        }
    )


def test_estimators_recover_known_vol() -> None:
    bars = _gbm_bars(0.3)
    cc = float(np.sqrt(annualize(close_to_close(bars).mean())))
    assert cc == pytest.approx(0.3, rel=0.03)
    for fn in (parkinson, garman_klass, yang_zhang_daily):
        ann = float(np.sqrt(annualize(fn(bars).mean())))
        # Sampling 1000 points per day still misses a little of the true range
        # (bias ~ 1/sqrt(steps)), so allow a few percent.
        assert ann == pytest.approx(0.3, rel=0.05), fn.__name__
    # With coarse sampling the true high/low falls between observed points and
    # range estimators run low. Real exchange data has a milder version of this.
    coarse = _gbm_bars(0.3, steps=20)
    assert float(np.sqrt(annualize(parkinson(coarse).mean()))) < 0.29
    yz = float(np.sqrt(annualize(yang_zhang(bars, 21).mean())))
    assert yz == pytest.approx(0.3, rel=0.06)


def test_annualization_and_day_counts() -> None:
    assert annualize(0.0001) == pytest.approx(0.0252)
    # 20% vol over 30 calendar days is 0.04 * 30/365 total variance.
    assert total_var_to_vol(0.04 * 30 / 365, 30) == pytest.approx(0.20)
    # Mon Oct 19 2026 + 30 calendar days: sessions Oct 20 .. Nov 18, no holidays.
    assert trading_days_in(date(2026, 10, 19), 30) == 22
    # Thanksgiving week: Nov 20 2026 + 14 days -> 9 sessions (Nov 26 closed).
    assert trading_days_in(date(2026, 11, 20), 14) == 9


def test_har_target_alignment() -> None:
    rv = pd.Series(np.arange(10, dtype=float))
    y = har_target(rv, 3)
    assert y.iloc[0] == pytest.approx((1 + 2 + 3) / 3)
    assert np.isnan(y.iloc[-1])


def test_har_recovers_simulated_coefficients() -> None:
    rng = np.random.default_rng(3)
    n, c, bd, bw, bm = 6000, 0.00002, 0.35, 0.3, 0.25
    rv = np.full(n, 0.0001)
    for t in range(22, n - 1):
        mean = c + bd * rv[t] + bw * rv[t - 4 : t + 1].mean() + bm * rv[t - 21 : t + 1].mean()
        rv[t + 1] = mean * rng.gamma(20, 1 / 20)
    fit = fit_har(pd.Series(rv), h=1)
    assert fit.params["rv_d"] == pytest.approx(bd, abs=0.06)
    assert fit.params["rv_w"] == pytest.approx(bw, abs=0.1)
    assert fit.params["rv_m"] == pytest.approx(bm, abs=0.1)


def test_garch_aggregation_matches_arch_and_recovers_params() -> None:
    sim = arch_model(None, vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    data = sim.simulate([0.0, 0.05, 0.08, 0.9], 4000, burn=500)["data"]
    res = fit_garch(pd.Series(data))
    p = res.params
    assert p["alpha[1]"] == pytest.approx(0.08, abs=0.04)
    assert p["beta[1]"] == pytest.approx(0.9, abs=0.05)
    fc = res.forecast(horizon=21, reindex=False).variance.to_numpy()[-1]
    ours = aggregate_variance_path(p["omega"], p["alpha[1]"], p["beta[1]"], fc[0], 21)
    assert ours == pytest.approx(fc.sum(), rel=1e-8)
    out = garch_forecast(pd.Series(data), 21)
    assert out.total_var == pytest.approx(fc.sum() / 1e4, rel=1e-6)
    assert out.persistence < 1


def test_qlike_minimized_at_truth() -> None:
    r = np.array([1.0, 2.0, 3.0])
    assert qlike(r, r) == pytest.approx(0.0)
    assert qlike(r, r * 1.2) > 0 and qlike(r, r * 0.8) > 0


def test_ensemble_weights_and_combination() -> None:
    scores = pd.DataFrame({"model": ["har", "garch"], "qlike": [0.1, 0.3]})
    w = ensemble_weights(scores, "inverse_qlike")
    assert w["har"] == pytest.approx(0.75)
    assert combine({"har": 1.0, "garch": 2.0}, w) == pytest.approx(1.25)
    assert combine({"har": 1.0, "garch": float("nan")}, w) == pytest.approx(1.0)


def test_forecast_ticker_on_fake_history() -> None:
    asof = date(2026, 3, 2)
    prices = FakeProvider(vol=0.25).get_history("AAPL", date(2022, 3, 1), asof)
    cfg = ForecastSettings()
    fc = forecast_ticker(prices, asof, [30, 60], cfg)
    ens = fc[fc["model"] == "ensemble"].set_index("tenor_days")["vol"]
    # The fake path is 25% vol GBM. Forecasts on the calendar basis should be
    # near 25% * sqrt(sessions*365 / (252*days)) ~ 25%.
    assert ens.loc[30] == pytest.approx(0.25, abs=0.05)
    assert set(fc["model"]) == {"har", "garch", "gjr_garch", "ensemble"}


def test_walk_forward_evaluation_runs() -> None:
    prices = FakeProvider().get_history("MSFT", date(2022, 3, 1), date(2026, 3, 2))
    rv = daily_variance(prices)
    rets = pd.Series(
        100 * np.log(prices["close"]).diff().to_numpy(), index=pd.to_datetime(prices["date"])
    )
    calls = {"n": 0}

    def fake_forecaster(rv_, r_, h, cfg):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return {"har": float(rv_.iloc[-22:].mean() * h), "garch": float(rv_.mean() * h)}

    scores = evaluate(rv, rets, 21, ForecastSettings(), forecaster=fake_forecaster)
    assert set(scores["model"]) == {"har", "garch"}
    assert (scores["n"] == calls["n"]).all()
