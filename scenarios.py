import copy
import numpy as np
from config import *
from dispatch import Plant
from optimizer import optimizer_decide, _persist


def belief_world(sim, t, obs, seed, steps, windows=None):
    """One imagined future, built only from what the agent can know: its forecast, what it sees now,
    and random 'what could go wrong' events. It never looks at the real future."""
    rng = np.random.default_rng(seed)
    end = min(t + steps, sim.n)
    L = end - t
    lead = np.arange(L)
    sigma = 0.03 + 0.012 * lead                       # forecasts get less reliable further ahead
    sigma[0] = 0.0                                    # right now we know the truth

    def series(name, actual):
        s = _persist(sim.base[name][t:end], actual)   # same "gap fades out" logic as the optimizer
        for a in windows or []:                       # alerts the LLM has already interpreted
            if a["series"] == name:
                lo, hi = max(a["abs_start"] - t, 1), min(a["abs_end"] - t, L)
                if hi > lo:
                    s[lo:hi] *= a["factor"]
        return s * np.exp(rng.normal(0, sigma))

    solar = series("solar", obs["solar_mw"])
    wind = series("wind", obs["wind_mw"])
    demand = series("demand", obs["demand_mw"])
    price = series("price", obs["price"])
    line = np.full(L, float(GRID_IMPORT_LIMIT_MW)); line[:8] = obs["line_limit"]
    batt = np.ones((L, len(BATTERIES)), dtype=bool); batt[:8] = obs["batt_ok"]

    def window():                                     # a random stretch of 2-4 hours
        d = int(rng.integers(8, 17)); s = int(rng.integers(1, max(2, L - d)))
        return s, min(s + d, L)

    if rng.random() < 0.30: s, e = window(); solar[s:e] *= rng.uniform(0.3, 0.6)       # clouds
    if rng.random() < 0.15: s, e = window(); wind[s:e] *= 0.3                           # wind drops
    if rng.random() < 0.30: s, e = window(); price[s:e] *= rng.uniform(1.8, 3.0)       # price spike
    if rng.random() < 0.20: s, e = window(); batt[s:e, int(rng.integers(len(BATTERIES)))] = False
    if rng.random() < 0.15: s, e = window(); line[s:e] = rng.uniform(70, 110)          # line derated
    if rng.random() < 0.20: s, e = window(); demand[s:e] += rng.uniform(15, 40)        # demand jump

    w = copy.copy(sim)                                # cheap copy: we only replace the arrays below
    w.solar = sim.solar.copy(); w.wind = sim.wind.copy()
    w.demand = sim.demand.copy(); w.price = sim.price.copy()
    w.line_limit = sim.line_limit.copy(); w.batt_ok = sim.batt_ok.copy()
    w.solar[t:end] = np.outer(solar, np.array(SOLAR_FARMS) / sum(SOLAR_FARMS))
    w.wind[t:end] = np.outer(wind, np.array(WIND_FARMS) / sum(WIND_FARMS))
    w.demand[t:end], w.price[t:end] = demand, price
    w.line_limit[t:end], w.batt_ok[t:end] = line, batt
    w.n_end = end
    return w


def rollout(world, start_plant, t0, steps, weights, windows, seed):
    """Play the imagined future forward with the optimizer using the given weights. Returns the cost."""
    world.rng = np.random.default_rng(seed)           # same forecast noise for every candidate
    plant = Plant()
    plant.soc, plant.deferred_mwh = list(start_plant.soc), start_plant.deferred_mwh
    end = min(t0 + steps, world.n)
    total = 0.0
    for k in range(t0, end):
        obs = world.observe(k)
        c, b, s = optimizer_decide(obs, plant, world, k, weights=weights, adjustments=windows)
        total += plant.step(obs, c, b, s)["total_cost"]
    ref = float(np.mean(world.base["price"][t0:end]))
    total -= 0.9 * ref * sum(plant.soc)               # leftover battery energy has value
    total += ref * plant.deferred_mwh                 # postponed load still has to be paid for
    return total


def pick_best(obs, plant, sim, t, proposed, windows=None, n_scenarios=10, steps=24, risk=0.5):
    """Stress-test several strategies on many imagined futures and pick the most robust one.
    Score = average cost + risk * (average of the worst 20% minus the average)."""
    candidates = [("proposed", dict(proposed))]
    for sv in (0.6, 1.0, 1.4, 1.9):
        if abs(sv - proposed["storage_value"]) > 0.15:
            candidates.append((f"reserve {sv}", dict(proposed, storage_value=sv)))
    worlds = [belief_world(sim, t, obs, 1000 + i, steps, windows) for i in range(n_scenarios)]

    table = []
    for label, w in candidates:
        costs = np.array([rollout(world, plant, t, steps, w, windows, 2000 + i)
                          for i, world in enumerate(worlds)])
        worst = np.sort(costs)[-max(2, n_scenarios // 5):].mean()
        mean = costs.mean()
        table.append(dict(label=label, weights=w, mean=mean, worst=worst, score=mean + risk * (worst - mean)))
    best = min(table, key=lambda r: r["score"])
    if best["score"] > table[0]["score"] * 0.995:     # not clearly better: keep the proposed strategy
        best = table[0]
    return best["weights"], table