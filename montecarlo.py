import numpy as np
from simulator import Simulator


def random_day(seed):
    """A random 24-hour test day: different weather and prices, plus 3 to 5 random surprises."""
    rng = np.random.default_rng(10_000 + seed)
    sim = Simulator(seed=seed)
    for _ in range(int(rng.integers(3, 6))):
        kind = str(rng.choice(["cloud", "wind_drop", "wind_surge", "price_spike",
                               "battery_outage", "line_limit", "demand_jump"]))
        start = int(rng.integers(8, 80))
        end = min(start + int(rng.integers(6, 17)), 96)
        if kind == "cloud":            sim.add_event("cloud", start, end, factor=float(rng.uniform(0.2, 0.5)))
        elif kind == "wind_drop":      sim.add_event("wind_surge", start, end, factor=0.25)
        elif kind == "wind_surge":     sim.add_event("wind_surge", start, end, factor=1.8)
        elif kind == "price_spike":    sim.add_event("price_spike", start, end, factor=float(rng.uniform(2, 3.5)))
        elif kind == "battery_outage": sim.add_event("battery_outage", start, end, battery=int(rng.integers(2)))
        elif kind == "line_limit":     sim.add_event("line_limit", start, end, limit=float(rng.uniform(70, 110)))
        else:                          sim.add_event("demand_jump", start, end, mw=float(rng.uniform(20, 50)))
    return sim