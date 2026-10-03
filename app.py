import pandas as pd
import streamlit as st
import dashboard_data as D

st.set_page_config(page_title="Renewable Energy Orchestrator", layout="wide")
st.title("Renewable Energy Orchestrator")
st.caption("An AI agent that coordinates solar, wind, batteries, the grid and industrial load "
           "every 15 minutes, and explains what it does.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1. Choose a day")
    scenario = st.selectbox("Scenario", ["Storm day (text alert)", "Random day", "Custom day"])
    seed, chosen = 0, {}
    if scenario == "Random day":
        seed = st.slider("Random day number", 0, 99, 3)
    if scenario == "Custom day":
        st.caption("Tick the surprises to inject and choose when they happen.")
        for name, spec in D.EVENT_LIBRARY.items():
            if st.checkbox(name, value=spec["on"], key="chk_" + name):
                chosen[name] = st.slider(name + " (hour of day)", 0.0, 24.0, spec["default"],
                                         step=0.25, key="sl_" + name)

    st.header("2. Text alert (optional)")
    is_storm = scenario.startswith("Storm")
    alert_text = st.text_area("Alert message", value=D.STORM_TEXT if is_storm else "",
                              key="alert_" + scenario,
                              help="The LLM reads this and decides how to prepare.")
    alert_hour = st.slider("Alert arrives at (hour)", 0.0, 24.0, 16.0, step=0.25)
    alert_len = st.slider("Alert stays active (hours)", 0.5, 8.0, 5.0, step=0.5)

    st.header("3. Choose the controller")
    controller_label = st.radio("Who guides the optimizer?", [
        "Stand-in LLM (no key needed)", "Real LLM (uses your API key)", "Rules only (no LLM)"])
    simulate = st.checkbox("Stress-test strategies on imagined futures (slower, about a minute)")

    run = st.button("Run simulation", type="primary")

controller = {"Stand-in": "stand-in", "Real LLM": "live", "Rules only": "rules"}[
    next(k for k in ("Stand-in", "Real LLM", "Rules only") if controller_label.startswith(k))]

if run:
    schedule = {}
    if alert_text.strip():
        schedule = {D.to_step(alert_hour): (alert_text.strip(), max(int(alert_len * 4), 1))}
    with st.spinner("Simulating the day..."):
        st.session_state["result"] = D.run_scenario(
            D.make_factory(scenario, seed, chosen), controller, simulate, schedule)
        st.session_state["controller"] = controller

if "result" not in st.session_state:
    st.info("Choose a scenario in the sidebar and press **Run simulation**.")
    st.stop()

res = st.session_state["result"]
S = D.summary(res)
F = D.build_frames(res)
times = F["times"]

if st.session_state.get("controller") == "live" and any(d["source"] == "rules" for d in res["decisions"]):
    st.warning("The real LLM could not be reached for some decisions, so fixed rules were used instead. "
               "Check your API key and the model name in llm.py. See the Agent decisions tab for details.")

# ------------------------------------------------------------------ headline numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total cost, agent", f"${S['agent']['total_cost']:,.0f}",
          delta=f"{-S['saving_pct']:+.1f}% vs baseline", delta_color="inverse")
c2.metric("Unserved demand, agent", f"{S['agent']['unserved_mwh']:,.0f} MWh",
          delta=f"{S['agent']['unserved_mwh'] - S['base']['unserved_mwh']:+,.0f} vs baseline",
          delta_color="inverse")
c3.metric("CO2 from imports, agent", f"{S['agent']['carbon_t']:,.0f} t",
          delta=f"{S['agent']['carbon_t'] - S['base']['carbon_t']:+,.0f} vs baseline", delta_color="inverse")
c4.metric("Industrial load shifted", f"{S['shifted_mwh']:,.0f} MWh")

tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Agent decisions", "Baseline vs agent", "How it works"])

# ------------------------------------------------------------------ tab 1: timeline
with tab1:
    default = times[64] if len(times) > 64 else times[0]
    chosen_time = st.select_slider("Scrub through the day", options=times, value=default)
    step = times.index(chosen_time)
    rec = res["log"][step]
    row = F["power"].iloc[step]

    a, b, c = st.columns(3)
    with a:
        st.subheader("Situation")
        st.write(f"Solar **{row['Solar (MW)']:.0f} MW**, wind **{row['Wind (MW)']:.0f} MW**, "
                 f"demand **{row['Demand (MW)']:.0f} MW**")
        st.write(f"Price **${F['price'].iloc[step, 0]:.0f}/MWh**")
        if rec["events"]:
            for e in rec["events"]:
                st.warning(e["msg"])
        else:
            st.success("Nothing unusual detected.")
    with b:
        st.subheader("Agent action")
        st.write(D.describe_step(rec))
        st.write(f"Batteries hold **{rec['soc'][0]:.0f}** and **{rec['soc'][1]:.0f} MWh**")
        if rec["notes"]:
            st.info("Safety layer: " + "; ".join(rec["notes"]))
    with c:
        st.subheader("What the agent knows")
        past = [d for d in res["decisions"] if d["step"] <= step]
        if past:
            d = past[-1]
            st.write(f"Mode: **{d['mode']}** ({d['source']})")
            st.write(d["explanation"])
            for al in d["alerts"]:
                st.caption("Active alert: " + al)

    left, right = st.columns(2)
    with left:
        st.caption("Generation and demand (MW)")
        st.line_chart(F["power"])
        st.caption("Battery energy (MWh)")
        st.line_chart(F["battery"])
    with right:
        st.caption("Electricity price ($/MWh)")
        st.line_chart(F["price"])
        st.caption("Grid import (+) and export (-), MW")
        st.line_chart(F["grid"])

    runs = D.event_runs(res["log"])
    if runs:
        st.caption("Surprises the agent detected during the day")
        st.dataframe(pd.DataFrame(runs).rename(columns={"first_message": "example"}))

# ------------------------------------------------------------------ tab 2: decisions
with tab2:
    st.write("Every time the situation changed, the agent asked the LLM for guidance. "
             "Here is what it was told and what it answered.")
    for d in res["decisions"]:
        with st.expander(f"{d['time']}  |  {d['mode']}  |  source: {d['source']}"):
            st.write(d["explanation"] or "(no explanation)")
            if d["events"]:
                st.caption("Detected: " + " ".join(d["events"]))
            for al in d["alerts"]:
                st.caption("Alert: " + al)
            st.write("Priorities chosen:", {k: round(v, 2) for k, v in d["weights"].items()})
            if d["adjustments"]:
                st.write("Forecast changes from the alert:", d["adjustments"])
            if d["scenarios"]:
                st.caption("Strategies stress-tested on imagined futures")
                st.dataframe(pd.DataFrame(d["scenarios"]).round(0))

# ------------------------------------------------------------------ tab 3: comparison
with tab3:
    keys = ["energy_cost", "carbon_cost", "curtail_cost", "deg_cost", "dr_cost",
            "unserved_cost", "total_cost", "unserved_mwh", "carbon_t"]
    table = pd.DataFrame({"Baseline": [S["base"][k] for k in keys],
                          "Agent": [S["agent"][k] for k in keys]}, index=keys).round(0)
    st.dataframe(table)
    st.caption("Cumulative cost during the day ($)")
    st.line_chart(F["cost"])
    st.caption("Demand that could not be served (MW)")
    st.bar_chart(F["unserved"])

# ------------------------------------------------------------------ tab 4: how it works
with tab4:
    st.markdown("""
**Every 15 minutes the agent runs this loop**

1. **Observe**: solar, wind, demand, price, line limit and battery availability.
2. **Detect surprises**: compares what is happening with what was forecast.
3. **Ask the LLM** (only when something changes): it reads text alerts, adjusts the forecast
   (for example "wind drops from 18:00") and chooses priorities, then explains why.
4. **Stress-test (optional)**: tries several battery-reserve strategies on imagined futures and
   keeps the proposal unless another is clearly more robust.
5. **Optimize**: a mathematical optimizer plans 8 hours ahead for batteries, grid, curtailment and
   industrial load shifting, and re-plans every step.
6. **Safety check**: a validator repairs any command that breaks a hard limit.
7. **Act and log**: the plant is updated and every decision is recorded.

**Self-assessed position on the 3x3 grid**

| Axis | Claim | Evidence |
|---|---|---|
| Features | **F3** | Optimal action bundles, an objective-driven choice under uncertainty, and scenario simulation over time (Monte Carlo days and imagined futures) |
| Depth | **D2** | Structured numeric data plus text alerts, with reliability shown on 15 random days |

All data in this demo is simulated.
""")