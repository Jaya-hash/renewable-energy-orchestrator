import pandas as pd
from simulator import Simulator
from dispatch import run_episode, DT
from baseline import baseline_decide
from agent import Agent
from montecarlo import random_day
from test_agent import make_storm_sim, ALERTS as STORM_ALERTS, stand_in_llm

# Surprises the user can switch on in the "Custom day" scenario
EVENT_LIBRARY = {
    "Clouds over the solar farms": dict(kind="cloud", params=dict(factor=0.3), default=(11.0, 14.0), on=True),
    "Wind drops (storm)": dict(kind="wind_surge", params=dict(factor=0.15), default=(18.0, 21.0), on=False),
    "Wind surge": dict(kind="wind_surge", params=dict(factor=1.8), default=(3.0, 6.0), on=False),
    "Electricity price spike": dict(kind="price_spike", params=dict(factor=3), default=(17.5, 19.5), on=True),
    "Battery 2 outage": dict(kind="battery_outage", params=dict(battery=1), default=(15.0, 20.0), on=True),
    "Transmission line limit": dict(kind="line_limit", params=dict(limit=100), default=(19.0, 22.0), on=False),
    "Industrial demand jump": dict(kind="demand_jump", params=dict(mw=40), default=(9.0, 12.0), on=False),
}
STORM_TEXT = STORM_ALERTS[64][0]


def to_step(hour):
    return min(max(int(round(hour * 4)), 0), 96)


def make_factory(scenario, seed=0, chosen=None):
    """Returns a function that builds a fresh simulated day each time it is called."""
    if scenario.startswith("Storm"):
        return make_storm_sim
    if scenario.startswith("Random"):
        return lambda: random_day(seed)

    def custom():
        sim = Simulator()
        for name, (h1, h2) in (chosen or {}).items():
            spec = EVENT_LIBRARY[name]
            s, e = to_step(h1), to_step(h2)
            sim.add_event(spec["kind"], s, max(e, s + 1), **spec["params"])
        return sim
    return custom


def no_llm(system, user):
    raise RuntimeError("rules only")


def run_scenario(factory, controller, simulate, alert_schedule, n_scenarios=10):
    """Run the baseline and the agent on identical copies of the same day."""
    base_totals, base_log = run_episode(factory(), baseline_decide)
    ask = {"stand-in": stand_in_llm, "rules": no_llm}.get(controller)   # None = the real LLM
    agent = Agent(ask=ask, alert_schedule=alert_schedule, simulate=simulate, n_scenarios=n_scenarios)
    sim = factory()
    totals, log = run_episode(sim, agent.decide)
    return dict(sim=sim, base_totals=base_totals, base_log=base_log,
                totals=totals, log=log, decisions=agent.decisions)


def describe_step(rec):
    """One plain-English sentence about what the agent did in a 15-minute step."""
    parts = []
    for i, p in enumerate(rec["batt_mw"]):
        if p > 1:
            parts.append(f"discharging battery {i + 1} at {p:.0f} MW")
        elif p < -1:
            parts.append(f"charging battery {i + 1} at {-p:.0f} MW")
    if rec["shift_mw"] > 1:
        parts.append(f"postponing {rec['shift_mw']:.0f} MW of industrial load")
    elif rec["shift_mw"] < -1:
        parts.append(f"catching up {-rec['shift_mw']:.0f} MW of postponed load")
    if rec["curtail_mw"] > 1:
        parts.append(f"curtailing {rec['curtail_mw']:.0f} MW of clean power")
    if rec["grid_mw"] > 1:
        parts.append(f"importing {rec['grid_mw']:.0f} MW from the grid")
    elif rec["grid_mw"] < -1:
        parts.append(f"exporting {-rec['grid_mw']:.0f} MW to the grid")
    if rec["unserved_mw"] > 0.5:
        parts.append(f"WARNING: {rec['unserved_mw']:.0f} MW of demand cannot be served")
    if not parts:
        return "Idle."
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "."


def build_frames(res):
    sim, base, agent = res["sim"], res["base_log"], res["log"]
    times = [r["time"] for r in agent]
    obs = [sim.observe(t) for t in range(sim.n)]
    idx = pd.Index(times, name="time")

    power = pd.DataFrame({"Solar (MW)": [o["solar_mw"] for o in obs],
                          "Wind (MW)": [o["wind_mw"] for o in obs],
                          "Demand (MW)": [o["demand_mw"] for o in obs]}, index=idx)
    price = pd.DataFrame({"Price ($/MWh)": [o["price"] for o in obs]}, index=idx)
    battery = pd.DataFrame({
        "Agent battery 1": [r["soc"][0] for r in agent], "Agent battery 2": [r["soc"][1] for r in agent],
        "Baseline battery 1": [r["soc"][0] for r in base], "Baseline battery 2": [r["soc"][1] for r in base]},
        index=idx)
    grid = pd.DataFrame({"Agent": [r["grid_mw"] for r in agent],
                         "Baseline": [r["grid_mw"] for r in base]}, index=idx)
    unserved = pd.DataFrame({"Agent": [r["unserved_mw"] for r in agent],
                             "Baseline": [r["unserved_mw"] for r in base]}, index=idx)
    cost = pd.DataFrame({"Agent": pd.Series([r["total_cost"] for r in agent]).cumsum().values,
                         "Baseline": pd.Series([r["total_cost"] for r in base]).cumsum().values}, index=idx)
    return dict(times=times, power=power, price=price, battery=battery, grid=grid,
                unserved=unserved, cost=cost)


def event_runs(log):
    """Group the detected surprises into runs, for example 'solar drop 11:00 to 14:00'."""
    runs, open_runs = [], {}
    for i, rec in enumerate(log):
        now = {e["type"]: e["msg"] for e in rec["events"]}
        for typ in list(open_runs):
            if typ not in now:
                s = open_runs.pop(typ)
                runs.append(dict(event=typ.replace("_", " "), start=log[s[0]]["time"],
                                 end=log[i - 1]["time"], first_message=s[1]))
        for typ, msg in now.items():
            open_runs.setdefault(typ, (i, msg))
    for typ, s in open_runs.items():
        runs.append(dict(event=typ.replace("_", " "), start=log[s[0]]["time"],
                         end=log[-1]["time"], first_message=s[1]))
    return sorted(runs, key=lambda r: r["start"])


def summary(res):
    b, a = res["base_totals"], res["totals"]
    saving = 100 * (b["total_cost"] - a["total_cost"]) / b["total_cost"]
    shifted = sum(max(r["shift_mw"], 0) for r in res["log"]) * DT
    return dict(base=b, agent=a, saving_pct=saving, shifted_mwh=shifted)