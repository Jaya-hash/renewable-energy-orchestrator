from simulator import Simulator
from dispatch import Plant
from validator import validate_and_fix

sim = Simulator()
sim.add_event("battery_outage", 0, 96, battery=1)   # battery 2 is offline all day
plant = Plant()
obs = sim.observe(50)

# A deliberately bad command: curtail far too much, battery 1 way over its limit,
# a command to the offline battery 2, and too much load shifting
curtail, batt, shift, notes = validate_and_fix(obs, plant, 999, [500, 20], 80)
print("Safe command:", round(curtail, 1), [round(b, 1) for b in batt], round(shift, 1))
for n in notes:
    print(" -", n)