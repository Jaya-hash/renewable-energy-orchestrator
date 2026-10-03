import numpy as np
from scipy.optimize import linprog
from config import *
from dispatch import DT
from baseline import baseline_decide

NB = len(BATTERIES)
PER = 3 * NB + 6          # number of unknowns per time step


def _i(k, name, b=0):
    """Position of an unknown inside the big list of unknowns."""
    base = k * PER
    if name == "c": return base + b                 # battery charge (MW)
    if name == "d": return base + NB + b            # battery discharge (MW)
    if name == "s": return base + 2 * NB + b        # battery energy at end of step (MWh)
    return base + 3 * NB + {"u": 0, "gin": 1, "gout": 2, "us": 3, "sh": 4, "rb": 5}[name]
    # u = curtailed clean power, gin = grid import, gout = grid export, us = unserved demand
    # sh = industrial load postponed, rb = postponed load caught up (rebound)


def _persist(series, actual, tau=6):
    """If reality differs from the forecast right now, assume the gap fades out slowly."""
    series = series.copy()
    if series[0] < 1.0:
        series[0] = actual
        return series
    ratio = np.clip(actual / series[0], 0.2, 5.0)
    return series * (1 + (ratio - 1) * np.exp(-np.arange(len(series)) / tau))


def plan(obs, plant, sim, t, horizon=32, weights=None, adjustments=None):
    """Look ahead `horizon` steps and find the cheapest plan. Returns the full plan."""
    w = dict(carbon_price=CARBON_PRICE, curtail_penalty=CURTAIL_PENALTY, storage_value=0.9)
    w.update(weights or {})

    fc = sim.forecast(t, horizon)
    K = len(fc["price"])
    solar = np.maximum(_persist(fc["solar"], obs["solar_mw"]), 0)
    wind = np.maximum(_persist(fc["wind"], obs["wind_mw"]), 0)
    demand = np.maximum(_persist(fc["demand"], obs["demand_mw"]), 0)
    price = np.maximum(_persist(fc["price"], obs["price"]), 0)
    # Forecast changes suggested by the LLM after reading text alerts (e.g. "storm at 18:00")
    series = {"solar": solar, "wind": wind, "demand": demand, "price": price}
    for a in adjustments or []:
        start = max(a["abs_start"] - t, 1)             # step 0 is what we actually see, never altered
        end = min(a["abs_end"] - t, K)
        if end > start:
            series[a["series"]][start:end] *= a["factor"]
    gen = solar + wind
    carbon = sim.carbon[t:t + K]

    n = K * PER + 1                                    # +1: load still owed after the plan ends
    LEFT = n - 1
    cost = np.zeros(n)
    lo, hi = np.zeros(n), np.full(n, np.inf)
    A_eq, b_eq = [], []
    A_ub, b_ub = [], []
    owed = np.zeros(n)                                 # row used for the "postponed load" rule

    for k in range(K):
        cost[_i(k, "gin")] = DT * (price[k] + carbon[k] * w["carbon_price"])
        cost[_i(k, "gout")] = -DT * price[k] * EXPORT_PRICE_FACTOR
        cost[_i(k, "u")] = DT * w["curtail_penalty"]
        cost[_i(k, "us")] = DT * UNSERVED_PENALTY * (1.005 if k == 0 else 1.0)   # tie-break: serve the present first
        cost[_i(k, "sh")] = DT * DR_COST
        hi[_i(k, "u")] = gen[k]
        hi[_i(k, "gin")] = obs["line_limit"]
        hi[_i(k, "gout")] = GRID_EXPORT_LIMIT_MW
        hi[_i(k, "sh")] = hi[_i(k, "rb")] = SHIFTABLE_SHARE * demand[k]
        owed[_i(k, "rb")], owed[_i(k, "sh")] = 1, -1
        A_ub.append(owed.copy()); b_ub.append(plant.deferred_mwh / DT)   # never catch up more than was postponed

        row = np.zeros(n)                              # power balance
        row[_i(k, "u")], row[_i(k, "gin")] = -1, 1
        row[_i(k, "gout")], row[_i(k, "us")] = -1, 1
        row[_i(k, "sh")], row[_i(k, "rb")] = 1, -1
        for b, bat in enumerate(BATTERIES):
            ok = obs["batt_ok"][b]
            hi[_i(k, "c", b)] = bat["max_mw"] if ok else 0
            hi[_i(k, "d", b)] = bat["max_mw"] if ok else 0
            hi[_i(k, "s", b)] = bat["cap_mwh"]
            cost[_i(k, "c", b)] = cost[_i(k, "d", b)] = DT * bat["deg_cost"]
            row[_i(k, "d", b)], row[_i(k, "c", b)] = 1, -1

            r = np.zeros(n)                            # battery energy bookkeeping
            r[_i(k, "s", b)] = 1
            r[_i(k, "c", b)] = -DT * bat["eff"]
            r[_i(k, "d", b)] = DT / bat["eff"]
            if k == 0:
                A_eq.append(r); b_eq.append(min(max(plant.soc[b], 0.0), bat["cap_mwh"]))
            else:
                r[_i(k - 1, "s", b)] = -1
                A_eq.append(r); b_eq.append(0.0)
        A_eq.append(row); b_eq.append(demand[k] - gen[k])

    # Postponed industrial load should be caught up within the plan. Whatever cannot be
    # (for example near the end of the day) is counted as a cost instead of making the plan impossible.
    eq = owed.copy(); eq[LEFT] = 1
    A_eq.append(eq); b_eq.append(plant.deferred_mwh / DT)
    cost[LEFT] = DT * UNSERVED_PENALTY

    # Energy left in the batteries at the end is worth something (otherwise it would drain them)
    for b in range(NB):
        cost[_i(K - 1, "s", b)] -= w["storage_value"] * price.mean()

    res = linprog(cost, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=list(zip(lo, hi)), method="highs")
    if res.status != 0:
        return None
    x = res.x
    return dict(
        curtail=x[_i(0, "u")],
        batt=[x[_i(0, "d", b)] - x[_i(0, "c", b)] for b in range(NB)],
        shift=x[_i(0, "sh")] - x[_i(0, "rb")],
        soc_path=[[x[_i(k, "s", b)] for b in range(NB)] for k in range(K)],
        expected_cost=res.fun,
    )


def optimizer_decide(obs, plant, sim, t, weights=None, adjustments=None):
    p = plan(obs, plant, sim, t, weights=weights, adjustments=adjustments)
    if p is None:                                      # solver failed: fall back to the simple rules
        c, b = baseline_decide(obs, plant, sim, t)
        return c, b, 0.0
    return p["curtail"], p["batt"], p["shift"]