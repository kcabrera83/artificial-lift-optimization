import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Artificial Lift Optimization", layout="wide")
st.title("Artificial Lift Optimization")
st.markdown("Optimize artificial lift systems (ESP, Rod Pump, Gas Lift).")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'optimizer': joblib.load(d / 'lift_optimizer.pkl')}

st.sidebar.header("Input Parameters")
lift_type = st.sidebar.selectbox('Lift Type', ['ESP','gas_lift','rod_pump'])
pump_speed = st.sidebar.slider('Pump Speed', 500, 4000, 2250)
well_depth = st.sidebar.slider('Well Depth', 1000, 15000, 8000)
water_cut = st.sidebar.slider('Water Cut', 0, 100, 50)
pressure = st.sidebar.slider('Pressure', 500, 5000, 2750)
motor_current = st.sidebar.slider('Motor Current', 10, 200, 105)

if st.sidebar.button("Run"):
    try:
        x = np.array([[lift_type, pump_speed, well_depth, water_cut, pressure, motor_current]])
        cols = st.columns(1)
        for i, (k, m) in enumerate(models.items()):
            if isinstance(m, dict):
                X = m['scaler'].transform(x)
                p = m['model'].predict(X)
                if 'label_encoder' in m:
                    val = m['label_encoder'].inverse_transform(p)[0]
                else:
                    val = f'{p[0]:.2f}'
            else:
                val = f'{m.predict(x)[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))