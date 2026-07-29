
import streamlit as st
import numpy as np
import joblib, os, asyncio

st.set_page_config(page_title="Artificial Lift Optimization", layout="wide")
st.title(":rocket: Artificial Lift Optimization")
st.divider()

SERVICE_REGISTRY = {}
for f in os.listdir("outputs/models"):
    if f.endswith(".pkl"):
        SERVICE_REGISTRY[f.replace(".pkl", "")] = joblib.load(os.path.join("outputs/models", f))

async def async_infer(service: str, inputs: list) -> float:
    await asyncio.sleep(0.01)
    data = SERVICE_REGISTRY[service]
    X = np.array(inputs).reshape(1, -1)
    if data.get("scaler"):
        X = data["scaler"].transform(X)
    return float(data["model"].predict(X)[0])

svc = st.sidebar.selectbox("Service Endpoint", list(SERVICE_REGISTRY.keys()))
data = SERVICE_REGISTRY.get(svc, {})
feats = data.get("feature_names", [f"param_{i}" for i in range(4)])
inputs = [st.number_input(f, key=f"async_{svc}_{i}") for i, f in enumerate(feats)]

if st.button("Call async service"):
    with st.spinner("Awaiting inference..."):
        result = asyncio.run(async_infer(svc, inputs))
    st.success(f"Service returned: {result:.4f}")
