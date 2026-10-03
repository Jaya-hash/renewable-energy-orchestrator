STEP_MIN = 15
HORIZON_STEPS = 96            # 24 hours

SOLAR_FARMS = [60, 50, 40, 30, 20]       # MW capacity each (5 farms)
WIND_FARMS  = [80, 60, 40]               # MW capacity each (3 farms)

BATTERIES = [
  {"cap_mwh": 100, "max_mw": 50, "eff": 0.92, "deg_cost": 4.0},
  {"cap_mwh": 60,  "max_mw": 30, "eff": 0.92, "deg_cost": 4.0},
]

GRID_IMPORT_LIMIT_MW = 120
GRID_EXPORT_LIMIT_MW = 150
INDUSTRIAL_LOAD_MW   = 180
SHIFTABLE_SHARE      = 0.2

# ---- costs used to score every decision ----
INITIAL_SOC          = 0.5     # batteries start half full
EXPORT_PRICE_FACTOR  = 0.8     # we sell power at 80% of the market price
CARBON_PRICE         = 50.0    # $ per tonne CO2
CURTAIL_PENALTY      = 20.0    # $ per MWh of clean energy thrown away
UNSERVED_PENALTY     = 1000.0  # $ per MWh of demand we fail to supply
DR_COST              = 15.0    # $ per MWh of industrial load shifted (lost productivity)