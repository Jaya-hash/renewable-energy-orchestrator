from config import *

DT = STEP_MIN / 60   # one step in hours (0.25)


class Plant:
    """Holds the battery charge and applies one decision for one 15-minute step."""

    def __init__(self):
        self.soc = [b["cap_mwh"] * INITIAL_SOC for b in BATTERIES]
        self.deferred_mwh = 0.0        # industrial energy postponed and still owed

    def step(self, obs, curtail_mw, batt_mw, shift_mw=0.0):
        # curtail_mw: clean power we throw away (>= 0)
        # batt_mw: one number per battery. Positive = discharge, negative = charge
        # shift_mw: positive = postpone this much industrial load, negative = catch up on postponed load
        gen = obs["solar_mw"] + obs["wind_mw"]
        curtail = min(max(curtail_mw, 0.0), gen)

        limit = SHIFTABLE_SHARE * obs["demand_mw"]
        if shift_mw >= 0:
            shift = min(shift_mw, limit)
        else:
            shift = -min(-shift_mw, limit, self.deferred_mwh / DT)
        self.deferred_mwh += shift * DT

        actual = []
        for i, b in enumerate(BATTERIES):
            p = batt_mw[i] if obs["batt_ok"][i] else 0.0
            p = max(-b["max_mw"], min(b["max_mw"], p))
            if p > 0:                                   # discharging
                p = min(p, self.soc[i] * b["eff"] / DT)
                self.soc[i] -= p * DT / b["eff"]
            elif p < 0:                                 # charging
                room = b["cap_mwh"] - self.soc[i]
                p = -min(-p, room / (b["eff"] * DT))
                self.soc[i] += -p * DT * b["eff"]
            actual.append(p)

        # Power balance: positive grid = import, negative = export
        grid = (obs["demand_mw"] - shift) - (gen - curtail) - sum(actual)
        unserved = max(0.0, grid - obs["line_limit"])
        grid = min(grid, obs["line_limit"])
        forced = max(0.0, -GRID_EXPORT_LIMIT_MW - grid)  # can't export it, so it is wasted
        grid = max(grid, -GRID_EXPORT_LIMIT_MW)
        curtail += forced

        imp, exp = max(grid, 0.0), max(-grid, 0.0)
        energy_cost = (imp * obs["price"] - exp * obs["price"] * EXPORT_PRICE_FACTOR) * DT
        carbon_t = imp * obs["carbon"] * DT
        carbon_cost = carbon_t * CARBON_PRICE
        curtail_cost = curtail * DT * CURTAIL_PENALTY
        deg_cost = sum(abs(p) * DT * b["deg_cost"] for p, b in zip(actual, BATTERIES))
        unserved_cost = unserved * DT * UNSERVED_PENALTY
        dr_cost = max(shift, 0.0) * DT * DR_COST
        return dict(
            time=obs["time"], grid_mw=grid, curtail_mw=curtail, batt_mw=actual,
            shift_mw=shift, deferred_mwh=self.deferred_mwh,
            soc=list(self.soc), unserved_mw=unserved,
            energy_cost=energy_cost, carbon_t=carbon_t, carbon_cost=carbon_cost,
            curtail_mwh=curtail * DT, curtail_cost=curtail_cost, deg_cost=deg_cost,
            dr_cost=dr_cost,
            unserved_mwh=unserved * DT, unserved_cost=unserved_cost,
            total_cost=energy_cost + carbon_cost + curtail_cost + deg_cost + dr_cost + unserved_cost,
        )


def run_episode(sim, decide, safety=True):
    """Run a whole day. `decide(obs, plant, sim, t)` returns (curtail_mw, [battery_mw, ...])
    or (curtail_mw, [battery_mw, ...], shift_mw)."""
    from validator import validate_and_fix
    from events import detect_events

    plant = Plant()
    log = []
    for t in range(sim.n):
        obs = sim.observe(t)
        out = decide(obs, plant, sim, t)
        curtail, batt = out[0], out[1]
        shift = out[2] if len(out) > 2 else 0.0
        notes = []
        if safety:
            curtail, batt, shift, notes = validate_and_fix(obs, plant, curtail, batt, shift)
        rec = plant.step(obs, curtail, batt, shift)
        rec["notes"] = notes
        rec["events"] = detect_events(obs, sim, t)
        log.append(rec)

    keys = ["energy_cost", "carbon_t", "carbon_cost", "curtail_mwh", "curtail_cost",
            "deg_cost", "dr_cost", "unserved_mwh", "unserved_cost", "total_cost"]
    totals = {k: sum(r[k] for r in log) for k in keys}
    # Postponed load that was never caught up counts as unserved demand
    if plant.deferred_mwh > 1e-6:
        totals["unserved_mwh"] += plant.deferred_mwh
        totals["unserved_cost"] += plant.deferred_mwh * UNSERVED_PENALTY
        totals["total_cost"] += plant.deferred_mwh * UNSERVED_PENALTY
    return totals, log