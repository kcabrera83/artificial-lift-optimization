"""FastAPI web server for artificial lift optimization."""

import os
import traceback
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from artificial_lift_optimization.models.lift_optimizer import LiftOptimizer
from artificial_lift_optimization.models.failure_predictor import FailurePredictor

app = FastAPI(
    title="Artificial Lift Optimization",
    description="Artificial lift parameter optimization and failure mode prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")
models: dict[str, Any] = {}


@app.on_event("startup")
async def load_models():
    try:
        opt_path = os.path.join(MODEL_DIR, "lift_optimizer.pkl")
        pred_path = os.path.join(MODEL_DIR, "failure_predictor.pkl")
        if os.path.exists(opt_path):
            models["optimizer"] = LiftOptimizer.load(opt_path)
        if os.path.exists(pred_path):
            models["predictor"] = FailurePredictor.load(pred_path)
    except Exception as e:
        print(f"  Error loading models: {e}")


class OptimizeRequest(BaseModel):
    lift_type: str = "ESP"
    pump_speed_rpm: float = 0.0
    rod_load_klbf: float = 0.0
    gas_injection_mcf: float = 0.0
    downhole_pressure_psi: float = 2000.0
    motor_current_amp: float = 30.0
    well_depth_ft: float = 6000.0
    water_cut_pct: float = 50.0
    n_iterations: int = 500


class FailureRequest(BaseModel):
    lift_type: str = "ESP"
    pump_speed_rpm: float = 0.0
    rod_load_klbf: float = 0.0
    gas_injection_mcf: float = 0.0
    downhole_pressure_psi: float = 2000.0
    motor_current_amp: float = 30.0
    well_depth_ft: float = 6000.0
    water_cut_pct: float = 50.0


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": {
            "optimizer": "optimizer" in models,
            "failure_predictor": "predictor" in models,
        },
    }


@app.get("/api/models")
async def models_info():
    info = {}
    if "optimizer" in models:
        o = models["optimizer"]
        info["optimizer"] = {
            "type": "GradientBoostingRegressor",
            "r2": o.metadata.get("r2"),
            "mae": o.metadata.get("mae"),
            "cv_r2_mean": o.metadata.get("cv_r2_mean"),
            "n_samples": o.metadata.get("n_samples"),
            "n_features": o.metadata.get("n_features"),
            "feature_importance": o.metadata.get("feature_importance"),
        }
    if "predictor" in models:
        p = models["predictor"]
        info["failure_predictor"] = {
            "type": "RandomForestClassifier",
            "accuracy": p.metadata.get("accuracy"),
            "cv_accuracy_mean": p.metadata.get("cv_accuracy_mean"),
            "n_classes": p.metadata.get("n_classes"),
            "classes": p.metadata.get("classes"),
            "per_class_report": p.metadata.get("per_class_report"),
        }
    return info


@app.post("/api/optimize")
async def optimize(request: OptimizeRequest):
    if "optimizer" not in models:
        raise HTTPException(status_code=503, detail="Optimizer model not loaded")
    try:
        base_record = {
            "lift_type": request.lift_type,
            "pump_speed_rpm": request.pump_speed_rpm,
            "rod_load_klbf": request.rod_load_klbf,
            "gas_injection_mcf": request.gas_injection_mcf,
            "downhole_pressure_psi": request.downhole_pressure_psi,
            "motor_current_amp": request.motor_current_amp,
            "well_depth_ft": request.well_depth_ft,
            "water_cut_pct": request.water_cut_pct,
        }
        result = models["optimizer"].optimize(base_record, request.lift_type, n_iter=request.n_iterations)
        result["current_production_bbl_d"] = models["optimizer"].predict_production(base_record)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/failure")
async def failure(request: FailureRequest):
    if "predictor" not in models:
        raise HTTPException(status_code=503, detail="Failure predictor model not loaded")
    try:
        record = {
            "lift_type": request.lift_type,
            "pump_speed_rpm": request.pump_speed_rpm,
            "rod_load_klbf": request.rod_load_klbf,
            "gas_injection_mcf": request.gas_injection_mcf,
            "downhole_pressure_psi": request.downhole_pressure_psi,
            "motor_current_amp": request.motor_current_amp,
            "well_depth_ft": request.well_depth_ft,
            "water_cut_pct": request.water_cut_pct,
        }
        result = models["predictor"].predict(record)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5007)

