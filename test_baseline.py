from simulator import Simulator
from dispatch import run_episode
from baseline import baseline_decide


def make_sim():
    sim = Simulator()
    sim.add_event("cloud", 44, 56)
    sim.add_event("price_spike", 70, 78)
    sim.add_event("battery_outage", 60, 80, battery=1)
    sim.add_event("line_limit", 76, 88, limit=100)
    return sim


if __name__ == "__main__":
    totals, log = run_episode(make_sim(), baseline_decide)
    print("BASELINE RESULTS")
    for k, v in totals.items():
        print(f"  {k:15s} {v:12.1f}")