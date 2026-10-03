from dispatch import run_episode
from baseline import baseline_decide
from optimizer import optimizer_decide
from test_baseline import make_sim

base, _ = run_episode(make_sim(), baseline_decide)
smart, _ = run_episode(make_sim(), optimizer_decide)

print(f"{'':16s}{'baseline':>12s}{'optimizer':>12s}")
for k in base:
    print(f"{k:16s}{base[k]:12.1f}{smart[k]:12.1f}")
saving = 100 * (base["total_cost"] - smart["total_cost"]) / base["total_cost"]
print(f"\nOptimizer is {saving:.1f}% cheaper")