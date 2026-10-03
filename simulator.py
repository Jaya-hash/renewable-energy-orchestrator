import numpy as np
from config import *

class Simulator:
    def __init__(self, seed=42, steps=HORIZON_STEPS):
        self.rng = np.random.default_rng(seed)
        self.n = steps
        self.hour = np.arange(steps) * STEP_MIN / 60
        self.events = []
        self._build_truth()

    def _build_truth(self):
        h, n, r = self.hour, self.n, self.rng
        sun = np.clip(np.sin(np.pi * (h - 6) / 12), 0, None)
        self.solar = np.outer(sun, SOLAR_FARMS) * r.uniform(0.85, 1.0)
        cf = np.zeros((n, len(WIND_FARMS))); cf[0] = 0.35
        for i in range(1, n):
            cf[i] = np.clip(0.35 + 0.9 * (cf[i-1] - 0.35)
                            + r.normal(0, 0.05, len(WIND_FARMS)), 0.02, 0.95)
        self.wind = cf * np.array(WIND_FARMS)
        shape = 1 + 0.15 * np.sin(2*np.pi*(h-9)/24) + 0.10 * np.exp(-((h-19)**2)/4)
        self.demand = INDUSTRIAL_LOAD_MW * shape + r.normal(0, 3, n)
        self.price = (50 + 25*np.exp(-((h-19)**2)/6)
                      + 10*np.exp(-((h-8)**2)/4) + r.normal(0, 3, n))
        self.carbon = 0.4 + 0.2*np.exp(-((h-19)**2)/6)        # tCO2/MWh
        self.line_limit = np.full(n, float(GRID_IMPORT_LIMIT_MW))
        self.batt_ok = np.ones((n, len(BATTERIES)), dtype=bool)
        # copies for forecasting, taken BEFORE events are applied
        self.base = dict(solar=self.solar.sum(1), wind=self.wind.sum(1),
                         demand=self.demand.copy(), price=self.price.copy())

    def add_event(self, kind, start, end, **kw):
        s, e = start, end
        if kind == "cloud":            self.solar[s:e] *= kw.get("factor", 0.3)
        elif kind == "wind_surge":     self.wind[s:e] = np.minimum(self.wind[s:e] * kw.get("factor", 1.8), WIND_FARMS)
        elif kind == "price_spike":    self.price[s:e] *= kw.get("factor", 3)
        elif kind == "battery_outage": self.batt_ok[s:e, kw.get("battery", 0)] = False
        elif kind == "line_limit":     self.line_limit[s:e] = kw.get("limit", 60)
        elif kind == "demand_jump":    self.demand[s:e] += kw.get("mw", 40)
        self.events.append(dict(kind=kind, start=s, end=e, **kw))

    def observe(self, now):
        """The real situation at step `now`."""
        m = int(self.hour[now] % 1 * 60)
        return dict(step=now, time=f"{int(self.hour[now]):02d}:{m:02d}",
                    solar_mw=self.solar[now].sum(), wind_mw=self.wind[now].sum(),
                    demand_mw=self.demand[now], price=self.price[now],
                    carbon=self.carbon[now], line_limit=self.line_limit[now],
                    batt_ok=self.batt_ok[now].tolist())

    def forecast(self, now, horizon=16):
        """What the agent expects. Noisier the further ahead it looks."""
        idx = np.arange(now, min(now + horizon, self.n))
        sig = 0.02 + 0.01 * (idx - now)
        return {k: v[idx] * (1 + self.rng.normal(0, sig)) for k, v in self.base.items()}