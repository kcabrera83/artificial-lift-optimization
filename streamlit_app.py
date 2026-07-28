import streamlit as st
import pickle
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Artificial Lift Optimization", layout="wide")
st.title("Artificial Lift Optimization")
st.markdown("Optimize artificial lift systems (ESP, Rod Pump, Gas Lift) to maximize production.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: pickle.load(open(d / v, "rb")) for k, v in [("optimizer", "lift_optimizer.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
lift_type = st.sidebar.selectbox("Lift Type", ['ESP', 'gas_lift', 'rod_pump'])
pump_speed_rpm = st.sidebar.slider("Pump Speed Rpm", 500, 4000, 2250)
well_depth_ft = st.sidebar.slider("Well Depth Ft", 1000, 15000, 8000)
water_cut_pct = st.sidebar.slider("Water Cut Pct", 0, 100, 50)
downhole_pressure_psi = st.sidebar.slider("Downhole Pressure Psi", 500, 5000, 2750)
motor_current_amp = st.sidebar.slider("Motor Current Amp", 10, 200, 105)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[lift_type, pump_speed_rpm, well_depth_ft, water_cut_pct, downhole_pressure_psi, motor_current_amp]])
        m = models["optimizer"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Optimizer", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")