import math
from config import *
from dispatch import DT


def validate_and_fix(obs, plant, curtail, batt, shift):
    """Safety layer. Checks every command against hard limits and repairs anything unsafe.
    Returns the safe command plus a list of notes describing what was changed."""
    notes = []

    def clean(x, name):
        if x is None or not math.isfinite(float(x)):
            notes.append(f"{name}: invalid value replaced with 0")
            return 0.0
        return float(x)

    curtail = clean(curtail, "curtailment")
    shift = clean(shift, "load shift")
    batt = [clean(p, f"battery {i + 1}") for i, p in enumerate(batt)]

    gen = obs["solar_mw"] + obs["wind_mw"]
    if curtail < 0 or curtail > gen + 1e-6:
        curtail = min(max(curtail, 0.0), gen)
        notes.append("curtailment limited to available generation")

    for i, b in enumerate(BATTERIES):
        p = batt[i]
        if not obs["batt_ok"][i]:
            if abs(p) > 1e-6:
                notes.append(f"battery {i + 1} is offline: command cancelled")
            batt[i] = 0.0
            continue
        if abs(p) > b["max_mw"] + 1e-6:
            p = max(-b["max_mw"], min(b["max_mw"], p))
            notes.append(f"battery {i + 1}: power limited to its maximum rate")
        if p > 0 and p > plant.soc[i] * b["eff"] / DT + 1e-6:
            p = plant.soc[i] * b["eff"] / DT
            notes.append(f"battery {i + 1}: discharge limited to energy available")
        if p < 0 and -p > (b["cap_mwh"] - plant.soc[i]) / (b["eff"] * DT) + 1e-6:
            p = -(b["cap_mwh"] - plant.soc[i]) / (b["eff"] * DT)
            notes.append(f"battery {i + 1}: charge limited to free capacity")
        batt[i] = p

    limit = SHIFTABLE_SHARE * obs["demand_mw"]
    if shift > limit + 1e-6:
        shift = limit
        notes.append("load shift limited to the shiftable share of demand")
    if shift < 0:
        max_back = min(limit, plant.deferred_mwh / DT)
        if -shift > max_back + 1e-6:
            shift = -max_back
            notes.append("load rebound limited to energy that was deferred")

    grid = (obs["demand_mw"] - shift) - (gen - curtail) - sum(batt)
    if curtail > 1e-6 and grid > 1e-6:          # never waste clean energy while importing
        cut = min(curtail, grid)
        curtail -= cut
        notes.append("curtailment reduced: no clean energy is wasted while importing")

    return curtail, batt, shift, notes