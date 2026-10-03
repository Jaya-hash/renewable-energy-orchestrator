import numpy as np
from dispatch import run_episode
from baseline import baseline_decide
from optimizer import optimizer_decide
from agent import Agent
from montecarlo import random_day
from test_agent import make_storm_sim, ALERTS, stand_in_llm


def no_llm(system, user):
    raise RuntimeError("offline")        # forces the agent to use its fixed rules


if __name__ == "__main__":
    # ---- Part 1: 15 random days (different weather, prices and surprises) ----
    N = 15
    rows = []
    for d in range(N):
        b, _ = run_episode(random_day(d), baseline_decide)
        o, _ = run_episode(random_day(d), optimizer_decide)
        a, _ = run_episode(random_day(d), Agent(ask=no_llm).decide)
        rows.append([b["total_cost"], o["total_cost"], a["total_cost"]])
    r = np.array(rows)
    print(f"PART 1: {N} random days")
    print(f"{'':16s}{'baseline':>12s}{'optimizer':>12s}{'agent':>12s}")
    print(f"{'average cost':16s}{r.mean(0)[0]:12.0f}{r.mean(0)[1]:12.0f}{r.mean(0)[2]:12.0f}")
    print(f"{'worst day':16s}{r.max(0)[0]:12.0f}{r.max(0)[1]:12.0f}{r.max(0)[2]:12.0f}")
    print(f"Optimizer cheaper than baseline on {int((r[:, 1] < r[:, 0]).sum())} of {N} days "
          f"({100 * (1 - r[:, 1].sum() / r[:, 0].sum()):.1f}% lower total cost)")

    # ---- Part 2: the storm day, with and without scenario testing ----
    print("\nPART 2: storm day (takes about a minute)")
    plain = Agent(ask=stand_in_llm, alert_schedule=ALERTS)
    p, _ = run_episode(make_storm_sim(), plain.decide)
    tested = Agent(ask=stand_in_llm, alert_schedule=ALERTS, simulate=True)
    t, _ = run_episode(make_storm_sim(), tested.decide)
    print(f"Agent without scenario testing: {p['total_cost']:.0f}")
    print(f"Agent with scenario testing:    {t['total_cost']:.0f}")
    d = [x for x in tested.decisions if x["alerts"]][0]
    print(f"\nStrategies tested at {d['time']} (10 imagined futures each):")
    print(f"  {'strategy':14s}{'average':>10s}{'worst 20%':>11s}{'score':>10s}")
    for s in d["scenarios"]:
        print(f"  {s['label']:14s}{s['mean']:10.0f}{s['worst']:11.0f}{s['score']:10.0f}")
    print("Chosen battery reserve value:", d["weights"]["storage_value"])
