from simulator import Simulator
import matplotlib.pyplot as plt

sim = Simulator()
sim.add_event("cloud", 44, 56)
sim.add_event("price_spike", 70, 78)
sim.add_event("battery_outage", 60, 80, battery=1)

plt.plot(sim.hour, sim.solar.sum(1), label="solar")
plt.plot(sim.hour, sim.wind.sum(1), label="wind")
plt.plot(sim.hour, sim.demand, label="demand")
plt.legend()
plt.show()

print(sim.observe(48))
print(sim.forecast(48)["solar"][:4])