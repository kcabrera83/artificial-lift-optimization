# API Documentation - Artificial Lift Optimization

## Base URL
```
http://localhost:5007
```

## Endpoints

### GET /
Main dashboard with interactive web interface.

**Response:** HTML page with optimization and failure prediction panels.

---

### GET /api/health
Service health check with model status.

**Response (200):**
```json
{
  "status": "ok",
  "models_loaded": {
    "optimizer": true,
    "failure_predictor": true
  }
}
```

---

### GET /api/models
Model metadata and performance metrics.

**Response (200):**
```json
{
  "optimizer": {
    "type": "GradientBoostingRegressor",
    "r2": 0.96,
    "mae": 25.3,
    "cv_r2_mean": 0.95,
    "n_samples": 8000,
    "n_features": 8,
    "feature_importance": {
      "pump_speed_rpm": 0.25,
      "water_cut_pct": 0.20,
      "well_depth_ft": 0.15,
      ...
    }
  },
  "failure_predictor": {
    "type": "RandomForestClassifier",
    "accuracy": 0.97,
    "cv_accuracy_mean": 0.96,
    "n_classes": 5,
    "classes": ["normal", "pump_wear", "gas_lock", "rod_fatigue", "motor_overheat"],
    "per_class_report": {
      "normal": {"precision": 0.98, "recall": 0.99, "f1-score": 0.98},
      ...
    }
  }
}
```

---

### POST /api/optimize
Optimize artificial lift parameters for maximum production.

**Request:**
```json
{
  "lift_type": "ESP",
  "pump_speed_rpm": 2500,
  "rod_load_klbf": 0,
  "gas_injection_mcf": 0,
  "downhole_pressure_psi": 2000,
  "motor_current_amp": 40,
  "well_depth_ft": 6000,
  "water_cut_pct": 50,
  "n_iterations": 500
}
```

**Parameters:**
| Parameter | Unit | Description |
|-----------|------|-------------|
| lift_type | - | ESP, rod_pump, or gas_lift |
| pump_speed_rpm | RPM | Current pump speed |
| rod_load_klbf | klbf | Rod load (rod_pump only) |
| gas_injection_mcf | mcf | Gas injection rate (gas_lift only) |
| downhole_pressure_psi | psi | Downhole pressure |
| motor_current_amp | A | Motor current |
| well_depth_ft | ft | Well depth |
| water_cut_pct | % | Water cut percentage |
| n_iterations | - | Optimization iterations (default: 500) |

**Response (200):**
```json
{
  "current_production_bbl_d": 350.0,
  "predicted_production_bbl_d": 520.0,
  "optimal_params": {
    "pump_speed_rpm": 3200,
    "motor_current_amp": 55
  },
  "improvement_pct": 48.6
}
```

**Error Response (503):**
```json
{"error": "Optimizer model not loaded"}
```

**Error Response (400):**
```json
{"error": "No JSON body provided"}
```

---

### POST /api/failure
Predict equipment failure mode.

**Request:**
```json
{
  "lift_type": "ESP",
  "pump_speed_rpm": 3200,
  "rod_load_klbf": 0,
  "gas_injection_mcf": 0,
  "downhole_pressure_psi": 3000,
  "motor_current_amp": 75,
  "well_depth_ft": 8000,
  "water_cut_pct": 80
}
```

**Response (200):**
```json
{
  "predicted_failure_mode": "motor_overheat",
  "probabilities": {
    "normal": 0.05,
    "pump_wear": 0.10,
    "gas_lock": 0.15,
    "rod_fatigue": 0.05,
    "motor_overheat": 0.65
  },
  "confidence": 0.65,
  "recommendation": "Reduce motor current and pump speed immediately"
}
```

**Error Response (503):**
```json
{"error": "Failure predictor model not loaded"}
```

---

### GET /api/docs
OpenAPI 3.0 self-documentation.

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request - missing or invalid data |
| 500 | Internal server error |
| 503 | Service unavailable - model not loaded |
