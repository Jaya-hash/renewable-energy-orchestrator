import json
import sys
from dispatch import run_episode
from baseline import baseline_decide
from optimizer import optimizer_decide
from agent import Agent
from test_baseline import make_sim


def make_storm_sim():
    sim = make_sim()
    sim.add_event("wind_surge", 72, 84, factor=0.15)       # the storm really shuts down most wind power
    return sim


# A text alert arrives at step 64 (16:00) and stays active for 20 steps
ALERTS = {64: ("STORM WARNING: severe storm expected between 18:00 and 21:00. "
               "Wind farms are likely to shut down for safety during this time.", 20)}


def stand_in_llm(system, user):
    """Pretend LLM for testing the plumbing without a key. It answers the way a good LLM should."""
    s = json.loads(user)
    if any("STORM" in a for a in s["active_alerts"]):
        start = max(72 - s["step_now"], 0)
        return json.dumps({"mode": "storm_prep",
                           "weights": {"carbon_price": 50, "curtail_penalty": 20, "storage_value": 1.0},
                           "forecast_adjustments": [{"series": "wind", "factor": 0.15,
                                                     "start_in_steps": start, "duration_steps": 12 - max(s["step_now"] - 72, 0)}],
                           "explanation": "Storm warning: wind will drop sharply from 18:00, so batteries are charged beforehand."})
    return json.dumps({"mode": "normal", "weights": {"carbon_price": 50, "curtail_penalty": 20, "storage_value": 0.9},
                       "forecast_adjustments": [], "explanation": "No special conditions."})


if __name__ == "__main__":
    live = len(sys.argv) > 1 and sys.argv[1] == "live"
    base, _ = run_episode(make_storm_sim(), baseline_decide)
    plain, _ = run_episode(make_storm_sim(), optimizer_decide)
    agent = Agent(ask=None if live else stand_in_llm, alert_schedule=ALERTS)
    smart, log = run_episode(make_storm_sim(), agent.decide)

    print("Using:", "REAL LLM" if live else "stand-in LLM (no key needed)")
    print(f"{'':16s}{'baseline':>12s}{'optimizer':>12s}{'LLM agent':>12s}")
    for k in base:
        print(f"{k:16s}{base[k]:12.1f}{plain[k]:12.1f}{smart[k]:12.1f}")
    print("\nAgent consulted the LLM", len(agent.decisions), "times. Examples:")
    for d in agent.decisions[:6]:
        print(f"  {d['time']} [{d['source']}] {d['mode']}: {d['explanation']}")