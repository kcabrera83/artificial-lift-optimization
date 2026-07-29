
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import numpy as np
import joblib, os

app = FastAPI(title="Artificial Lift Optimization")
security = HTTPBearer()

SERVICES = {}
for f in os.listdir("outputs/models"):
    if f.endswith(".pkl"):
        SERVICES[f.replace(".pkl", "")] = joblib.load(os.path.join("outputs/models", f))

@app.get("/")
async def root():
    return {"service": "Artificial Lift Optimization", "endpoints": list(SERVICES.keys())}

@app.post("/invoke/{service_name}")
async def invoke(service_name: str, body: dict, cred=Depends(security)):
    svc = SERVICES.get(service_name)
    if not svc:
        raise HTTPException(404)
    await asyncio.sleep(0.01)
    feats = svc.get("feature_names", list(body.keys()))
    X = np.array([body.get(f, 0) for f in feats]).reshape(1, -1)
    if svc.get("scaler"):
        X = svc["scaler"].transform(X)
    return {"prediction": float(svc["model"].predict(X)[0])}
