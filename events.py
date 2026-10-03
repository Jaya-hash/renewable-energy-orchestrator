from config import *


def detect_events(obs, sim, t):
    """Compare what is happening now with what was expected, and describe the surprises.
    Each event is a small dictionary: a short type name and a readable message."""
    exp = {k: v[t] for k, v in sim.base.items()}     # what the forecast expected before any surprise
    found = []

    if exp["solar"] > 5 and obs["solar_mw"] < 0.6 * exp["solar"]:
        drop = 100 * (1 - obs["solar_mw"] / exp["solar"])
        found.append(dict(type="solar_drop",
                          msg=f"Solar output is {drop:.0f}% below forecast ({obs['solar_mw']:.0f} MW vs {exp['solar']:.0f} MW)."))
    if obs["wind_mw"] > 1.4 * exp["wind"]:
        rise = 100 * (obs["wind_mw"] / exp["wind"] - 1)
        found.append(dict(type="wind_surge",
                          msg=f"Wind output is {rise:.0f}% above forecast ({obs['wind_mw']:.0f} MW vs {exp['wind']:.0f} MW)."))
    if exp["wind"] > 5 and obs["wind_mw"] < 0.5 * exp["wind"]:
        drop = 100 * (1 - obs["wind_mw"] / exp["wind"])
        found.append(dict(type="wind_drop",
                          msg=f"Wind output is {drop:.0f}% below forecast ({obs['wind_mw']:.0f} MW vs {exp['wind']:.0f} MW)."))
    if obs["price"] > 1.6 * exp["price"]:
        found.append(dict(type="price_spike",
                          msg=f"Electricity price spiked to {obs['price']:.0f} $/MWh (expected {exp['price']:.0f})."))
    if obs["demand_mw"] > exp["demand"] + 25:
        found.append(dict(type="demand_jump",
                          msg=f"Demand is {obs['demand_mw'] - exp['demand']:.0f} MW above forecast."))
    if obs["line_limit"] < GRID_IMPORT_LIMIT_MW:
        found.append(dict(type="line_limit",
                          msg=f"Transmission line limited to {obs['line_limit']:.0f} MW (normal {GRID_IMPORT_LIMIT_MW} MW)."))
    for i, ok in enumerate(obs["batt_ok"]):
        if not ok:
            found.append(dict(type="battery_outage", msg=f"Battery {i + 1} is unavailable."))
    return found