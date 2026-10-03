import json
import numpy as np
from config import *
from events import detect_events
from optimizer import optimizer_decide
from scenarios import pick_best
import llm

SYSTEM_PROMPT = """You are the control-room assistant of a renewable energy orchestrator.
Every 15 minutes a mathematical optimizer decides battery, grid, curtailment and load-shifting actions.
Your job is to read the situation and any text alerts, then tune that optimizer.

Reply with ONE JSON object and nothing else, in exactly this shape:
{
  "mode": "normal" | "storm_prep" | "price_response" | "conservative" | "low_carbon",
  "weights": {"carbon_price": number, "curtail_penalty": number, "storage_value": number},
  "forecast_adjustments": [
    {"series": "solar" | "wind" | "demand" | "price", "factor": number,
     "start_in_steps": integer, "duration_steps": integer}
  ],
  "explanation": "one or two plain sentences for a human operator"
}

Meaning of the weights (defaults in brackets):
- carbon_price [50], range 20 to 200: dollars per tonne of CO2. Higher means cleaner choices matter more.
- curtail_penalty [20], range 5 to 100: dollars per MWh of clean energy wasted.
- storage_value [0.9], range 0.5 to 2.5: how much energy kept in batteries is worth.
  Higher keeps a reserve for emergencies. Lower uses batteries more aggressively.
  Before a predicted supply shortfall (for example a storm), use about 1.5 to 2.0 so batteries are full when it starts.

Forecast adjustments express what text alerts imply. factor multiplies the forecast
(0.2 means 80 percent lower, 1.5 means 50 percent higher). start_in_steps counts 15-minute steps
from now (4 steps = 1 hour, so an event at 18:00 when it is 16:00 starts in 8 steps).
Every answer REPLACES your previous one, so always include adjustments for every alert that is still active.
Use an empty list when nothing needs adjusting. Never invent facts that are not in the input."""

ALLOWED_MODES = {"normal", "storm_prep", "price_response", "conservative", "low_carbon"}
DEFAULT_WEIGHTS = dict(carbon_price=CARBON_PRICE, curtail_penalty=CURTAIL_PENALTY, storage_value=0.9)
LIMITS = dict(carbon_price=(20, 200), curtail_penalty=(5, 100), storage_value=(0.5, 2.5))


def build_situation(obs, plant, sim, t, events, alerts):
    nxt = slice(t, min(t + 16, sim.n))               # the next 4 hours, as expected before any surprise
    return {
        "time_now": obs["time"], "step_now": t, "steps_per_hour": 4,
        "now": {"solar_mw": round(obs["solar_mw"]), "wind_mw": round(obs["wind_mw"]),
                "demand_mw": round(obs["demand_mw"]), "price_per_mwh": round(obs["price"]),
                "grid_import_limit_mw": round(obs["line_limit"]),
                "batteries_online": obs["batt_ok"]},
        "battery_energy_mwh": [round(x) for x in plant.soc],
        "battery_capacity_mwh": [b["cap_mwh"] for b in BATTERIES],
        "postponed_load_mwh": round(plant.deferred_mwh),
        "detected_events": [e["msg"] for e in events],
        "active_alerts": alerts,
        "expected_next_4h": {
            "solar_mw_avg": round(float(np.mean(sim.base["solar"][nxt]))),
            "wind_mw_avg": round(float(np.mean(sim.base["wind"][nxt]))),
            "demand_mw_avg": round(float(np.mean(sim.base["demand"][nxt]))),
            "price_max": round(float(np.max(sim.base["price"][nxt]))),
        },
    }


def parse_answer(text):
    """Turn the LLM's text into a safe, clamped decision. Raises ValueError if it is unusable."""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("no JSON found")
    raw = json.loads(text[a:b + 1])
    if not isinstance(raw, dict):
        raise ValueError("answer is not an object")

    weights = dict(DEFAULT_WEIGHTS)
    for k, (lo, hi) in LIMITS.items():
        try:
            weights[k] = min(max(float(raw.get("weights", {}).get(k, weights[k])), lo), hi)
        except (TypeError, ValueError):
            pass

    adjustments = []
    for adj in (raw.get("forecast_adjustments") or [])[:4]:
        try:
            if adj["series"] not in ("solar", "wind", "demand", "price"):
                continue
            adjustments.append(dict(
                series=adj["series"],
                factor=min(max(float(adj["factor"]), 0.1), 3.0),
                start=min(max(int(adj["start_in_steps"]), 0), 31),
                duration=min(max(int(adj["duration_steps"]), 1), 32)))
        except (KeyError, TypeError, ValueError):
            continue

    mode = raw.get("mode") if raw.get("mode") in ALLOWED_MODES else "normal"
    return dict(mode=mode, weights=weights, adjustments=adjustments,
                explanation=str(raw.get("explanation", ""))[:400])


def fallback_rules(events, alerts):
    """Used when the LLM is unavailable or answers badly. Simple fixed rules, so the agent never stops."""
    types = {e["type"] for e in events}
    w = dict(DEFAULT_WEIGHTS)
    if any("storm" in a.lower() for a in alerts):
        w["storage_value"] = 1.6
        return dict(mode="storm_prep", weights=w, adjustments=[],
                    explanation="Fallback rule: storm alert active, keeping extra battery reserve.")
    if types & {"battery_outage", "line_limit", "wind_drop", "solar_drop"}:
        w["storage_value"] = 1.3
        return dict(mode="conservative", weights=w, adjustments=[],
                    explanation="Fallback rule: supply or grid constraint detected, keeping a larger reserve.")
    if "price_spike" in types:
        w["storage_value"] = 0.7
        return dict(mode="price_response", weights=w, adjustments=[],
                    explanation="Fallback rule: price spike, using batteries more aggressively.")
    return dict(mode="normal", weights=w, adjustments=[], explanation="Fallback rule: normal operation.")


class Agent:
    """Wraps the optimizer. Asks the LLM for guidance whenever the situation changes.
    With simulate=True it also stress-tests the guidance on imagined futures before using it."""

    def __init__(self, ask=None, alert_schedule=None, simulate=False, n_scenarios=10):
        self.ask = ask or llm.call_llm            # a function (system, user) -> text
        self.alert_schedule = alert_schedule or {}   # {step: (alert text, how many steps it stays active)}
        self.simulate = simulate
        self.n_scenarios = n_scenarios
        self.active = []                           # [(text, last_step)]
        self.last_key = None
        self.guidance = dict(mode="normal", weights=dict(DEFAULT_WEIGHTS), adjustments=[], explanation="")
        self.windows = []
        self.decisions = []                        # one entry per time the LLM (or fallback) was consulted

    def decide(self, obs, plant, sim, t):
        events = detect_events(obs, sim, t)
        if t in self.alert_schedule:
            text, steps = self.alert_schedule[t]
            self.active.append((text, t + steps))
        self.active = [(x, end) for x, end in self.active if end > t]
        alerts = [x for x, _ in self.active]

        key = (tuple(sorted(e["type"] for e in events)), tuple(alerts))
        if key != self.last_key:                   # something changed: ask for fresh guidance
            self.last_key = key
            self._consult(obs, plant, sim, t, events, alerts)

        return optimizer_decide(obs, plant, sim, t,
                                weights=self.guidance["weights"], adjustments=self.windows)

    def _consult(self, obs, plant, sim, t, events, alerts):
        situation = build_situation(obs, plant, sim, t, events, alerts)
        try:
            guidance = parse_answer(self.ask(SYSTEM_PROMPT, json.dumps(situation, indent=1)))
            source = "llm"
        except Exception as err:                   # no key, no internet, bad answer...
            guidance = fallback_rules(events, alerts)
            guidance["explanation"] += f" (LLM unavailable: {type(err).__name__})"
            source = "rules"
        windows = [dict(series=a["series"], factor=a["factor"],
                        abs_start=t + a["start"], abs_end=t + a["start"] + a["duration"])
                   for a in guidance["adjustments"]]

        table = []
        if self.simulate:                          # stress-test the proposal on imagined futures
            chosen, table = pick_best(obs, plant, sim, t, guidance["weights"],
                                      windows=windows, n_scenarios=self.n_scenarios)
            if chosen != guidance["weights"]:
                guidance["explanation"] += (f" Scenario test changed the battery reserve value "
                                            f"from {guidance['weights']['storage_value']} to {chosen['storage_value']}.")
            guidance["weights"] = chosen

        self.guidance, self.windows = guidance, windows
        self.decisions.append(dict(step=t, time=obs["time"], source=source, mode=guidance["mode"],
                                   weights=guidance["weights"], adjustments=guidance["adjustments"],
                                   explanation=guidance["explanation"],
                                   events=[e["msg"] for e in events], alerts=alerts,
                                   scenarios=[dict(label=r["label"], mean=r["mean"], worst=r["worst"],
                                                   score=r["score"]) for r in table]))
    