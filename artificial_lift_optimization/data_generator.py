import numpy as np
import pandas as pd

LIFT_TYPES = ["ESP", "rod_pump", "gas_lift"]

FAILURE_MODES = ["normal", "pump_wear", "gas_lock", "rod_fatigue", "motor_overheat"]

LIFT_PARAMS = {
    "ESP": {
        "pump_speed_rpm": (1500, 3600),
        "rod_load_klbf": (0, 5),
        "gas_injection_mcf": (0, 0.5),
        "motor_current_amp": (15, 80),
    },
    "rod_pump": {
        "pump_speed_rpm": (2, 12),
        "rod_load_klbf": (5, 45),
        "gas_injection_mcf": (0, 0.3),
        "motor_current_amp": (10, 60),
    },
    "gas_lift": {
        "pump_speed_rpm": (0, 0),
        "rod_load_klbf": (0, 0),
        "gas_injection_mcf": (200, 2000),
        "motor_current_amp": (5, 30),
    },
}

WELL_DEPTH_RANGE = (2000, 12000)
WATER_CUT_RANGE = (10, 95)
PRESSURE_RANGE = (200, 5000)


def _base_production(lift_type, well_depth, water_cut):
    depth_factor = 1.0 - (well_depth - 2000) / 20000
    wc_factor = 1.0 - (water_cut / 100.0) * 0.6
    base = {"ESP": 800, "rod_pump": 400, "gas_lift": 500}[lift_type]
    return base * depth_factor * wc_factor


def _production_signal(row):
    base = _base_production(row["lift_type"], row["well_depth_ft"], row["water_cut_pct"])
    if row["lift_type"] == "ESP":
        speed_eff = 1.0 - abs(row["pump_speed_rpm"] - 2900) / 3000
        motor_eff = 1.0 - abs(row["motor_current_amp"] - 45) / 80
        prod = base * max(speed_eff, 0.1) * max(motor_eff, 0.1)
        prod *= 1.0 - row["downhole_pressure_psi"] / 10000
    elif row["lift_type"] == "rod_pump":
        rod_eff = 1.0 - abs(row["rod_load_klbf"] - 25) / 50
        speed_eff = 1.0 - abs(row["pump_speed_rpm"] - 7) / 15
        prod = base * max(rod_eff, 0.1) * max(speed_eff, 0.1)
        prod *= 1.0 - row["downhole_pressure_psi"] / 12000
    else:
        gas_eff = 1.0 - abs(row["gas_injection_mcf"] - 800) / 2000
        prod = base * max(gas_eff, 0.1)
        prod *= 1.0 - row["downhole_pressure_psi"] / 8000
    noise = np.random.normal(0, base * 0.05)
    return max(prod + noise, 10)


def _failure_label(row):
    scores = {
        "normal": 0.3,
        "pump_wear": 0.0,
        "gas_lock": 0.0,
        "rod_fatigue": 0.0,
        "motor_overheat": 0.0,
    }
    if row["lift_type"] == "ESP":
        if row["motor_current_amp"] > 55:
            scores["motor_overheat"] += 0.4
        if row["motor_current_amp"] > 65:
            scores["motor_overheat"] += 0.4
        if row["motor_current_amp"] > 72:
            scores["motor_overheat"] += 0.3
        if row["pump_speed_rpm"] > 3000:
            scores["pump_wear"] += 0.3
        if row["pump_speed_rpm"] > 3400:
            scores["pump_wear"] += 0.4
        if row["downhole_pressure_psi"] > 2500:
            scores["gas_lock"] += 0.2
        if row["downhole_pressure_psi"] > 3500:
            scores["gas_lock"] += 0.3
        if row["water_cut_pct"] > 75:
            scores["pump_wear"] += 0.2
        if row["water_cut_pct"] > 85:
            scores["pump_wear"] += 0.2
    elif row["lift_type"] == "rod_pump":
        if row["rod_load_klbf"] > 28:
            scores["rod_fatigue"] += 0.3
        if row["rod_load_klbf"] > 35:
            scores["rod_fatigue"] += 0.5
        if row["rod_load_klbf"] > 42:
            scores["rod_fatigue"] += 0.3
        if row["pump_speed_rpm"] > 8:
            scores["pump_wear"] += 0.3
        if row["pump_speed_rpm"] > 10:
            scores["pump_wear"] += 0.3
        if row["downhole_pressure_psi"] > 2500:
            scores["gas_lock"] += 0.3
        if row["downhole_pressure_psi"] > 3000:
            scores["gas_lock"] += 0.35
        if row["motor_current_amp"] > 42:
            scores["motor_overheat"] += 0.3
        if row["motor_current_amp"] > 50:
            scores["motor_overheat"] += 0.3
    else:
        if row["gas_injection_mcf"] < 400:
            scores["gas_lock"] += 0.3
        if row["gas_injection_mcf"] < 300:
            scores["gas_lock"] += 0.4
        if row["gas_injection_mcf"] > 1200:
            scores["motor_overheat"] += 0.2
        if row["gas_injection_mcf"] > 1500:
            scores["motor_overheat"] += 0.3
        if row["downhole_pressure_psi"] > 3000:
            scores["gas_lock"] += 0.2
        if row["downhole_pressure_psi"] > 3500:
            scores["gas_lock"] += 0.3

    scores["normal"] += np.random.normal(0, 0.08)
    for k in scores:
        if k != "normal":
            scores[k] += np.random.normal(0, 0.08)

    return max(scores, key=scores.get)


def generate_dataset(n_samples=5000, random_state=42):
    rng = np.random.RandomState(random_state)
    records = []
    for _ in range(n_samples):
        lift_type = rng.choice(LIFT_TYPES)
        well_depth = rng.randint(*WELL_DEPTH_RANGE)
        water_cut = rng.uniform(*WATER_CUT_RANGE)
        downhole_pressure = rng.uniform(*PRESSURE_RANGE)

        params = LIFT_PARAMS[lift_type]
        pump_speed = rng.uniform(*params["pump_speed_rpm"])
        rod_load = rng.uniform(*params["rod_load_klbf"])
        gas_injection = rng.uniform(*params["gas_injection_mcf"])
        motor_current = rng.uniform(*params["motor_current_amp"])

        row = {
            "lift_type": lift_type,
            "pump_speed_rpm": round(pump_speed, 1),
            "rod_load_klbf": round(rod_load, 2),
            "gas_injection_mcf": round(gas_injection, 2),
            "downhole_pressure_psi": round(downhole_pressure, 1),
            "motor_current_amp": round(motor_current, 1),
            "well_depth_ft": well_depth,
            "water_cut_pct": round(water_cut, 1),
        }
        row["production_bbl_d"] = round(_production_signal(row), 1)
        row["failure_mode"] = _failure_label(row)
        records.append(row)

    return pd.DataFrame(records)
