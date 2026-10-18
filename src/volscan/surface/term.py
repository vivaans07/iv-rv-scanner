"""Term-structure interpolation across SVI slices.

Interpolation is linear in total variance w(k, T) along calendar time at
fixed log-moneyness. If the slices are calendar-arbitrage-free (w increasing
in T at every k), the interpolated surface is too. Outside the quoted range
the vol is held flat, i.e. total variance scales proportionally with T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from volscan.surface.svi import SviParams, smile_metrics

FloatArray = NDArray[np.float64]


@dataclass
class TermStructure:
    slices: list[tuple[float, SviParams]]

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("term structure needs at least one slice")
        self.slices = sorted(self.slices, key=lambda s: s[0])
        self.times = np.array([t for t, _ in self.slices])

    def total_variance(self, k: FloatArray | float, T: float) -> FloatArray:
        k = np.asarray(k, dtype=np.float64)
        ts = self.times
        if ts[0] >= T:
            return self.slices[0][1].total_variance(k) * (T / ts[0])
        if ts[-1] <= T:
            return self.slices[-1][1].total_variance(k) * (T / ts[-1])
        i = int(np.searchsorted(ts, T)) - 1
        (t1, p1), (t2, p2) = self.slices[i], self.slices[i + 1]
        alpha = (T - t1) / (t2 - t1)
        return (1 - alpha) * p1.total_variance(k) + alpha * p2.total_variance(k)

    def implied_vol(self, k: FloatArray | float, T: float) -> FloatArray:
        return np.sqrt(np.maximum(self.total_variance(k, T), 0.0) / T)

    def constant_maturity(self, tenor_days: int) -> dict[str, float]:
        """ATM vol, 25d risk reversal and butterfly at a fixed calendar tenor.

        ATM comes from interpolated total variance. RR and BF are interpolated
        linearly in T between the bracketing slices (flat outside).
        """
        T = tenor_days / 365.0
        atm = float(self.implied_vol(0.0, T))
        metrics = [smile_metrics(p, t) for t, p in self.slices]
        rr = np.array([m["rr25"] for m in metrics])
        bf = np.array([m["bf25"] for m in metrics])
        return {
            "tenor_days": float(tenor_days),
            "atm_vol": atm,
            "rr25": float(np.interp(T, self.times, rr)),
            "bf25": float(np.interp(T, self.times, bf)),
            "atm_total_var": atm * atm * T,
        }
