from config import *


def baseline_decide(obs, plant, sim, t):
    """A simple rule-of-thumb controller. This is the one we will beat."""
    gen = obs["solar_mw"] + obs["wind_mw"]
    net = obs["demand_mw"] - gen              # positive = shortfall, negative = surplus
    batt = [0.0] * len(BATTERIES)
    if net < 0:                               # surplus: charge batteries in order
        left = -net
        for i, b in enumerate(BATTERIES):
            p = min(left, b["max_mw"]); batt[i] = -p; left -= p
    else:                                     # shortfall
        price_high = obs["price"] > 65
        left = net if price_high else net - obs["line_limit"]
        if left > 0:
            for i, b in enumerate(BATTERIES):
                p = min(left, b["max_mw"]); batt[i] = p; left -= p
    return 0.0, batt