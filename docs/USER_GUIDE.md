# User Guide - Artificial Lift Optimization

## Overview
Machine Learning system for optimizing artificial lift systems in oil wells. Supports three lift types (ESP, Rod Pump, Gas Lift), optimizes operating parameters for maximum production, and predicts equipment failure modes.

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/kcabrera83/artificial-lift-optimization.git
cd artificial-lift-optimization
pip install -r requirements.txt
```

### Training Models
```bash
python train.py
```
Generates 8000 synthetic samples, trains Lift Optimizer and Failure Predictor.

### Starting the Server
```bash
python app.py
```
Open http://localhost:5007 in your browser.

## Dashboard Features
- **Optimization Panel**: Input lift parameters and get optimal settings for maximum production
- **Failure Prediction**: Classify equipment failure modes with probability breakdown
- **Model Information**: View trained model metrics and feature importance
- **Interactive Forms**: Real-time prediction with validation

## Supported Lift Types

### ESP (Electric Submersible Pump)
- **Optimizes**: Pump speed (RPM), motor current (A)
- **Parameters**: downhole_pressure, well_depth, water_cut

### Rod Pump
- **Optimizes**: Rod load (klbf), pump speed (RPM)
- **Parameters**: downhole_pressure, well_depth, water_cut

### Gas Lift
- **Optimizes**: Gas injection rate (mcf)
- **Parameters**: downhole_pressure, well_depth, water_cut

## Failure Modes
| Mode | Description | Symptoms |
|------|-------------|----------|
| normal | Equipment operating normally | All parameters within range |
| pump_wear | Pump degradation | Increased vibration, reduced efficiency |
| gas_lock | Gas interference in pump | Reduced production, erratic pressure |
| rod_fatigue | Sucker rod fatigue/stress | Cyclic loading, rod stress |
| motor_overheat | Motor temperature exceeding limits | High current, elevated temperature |

## API Usage

### Using curl
```bash
# Optimize ESP parameters
curl -X POST http://localhost:5007/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "lift_type": "ESP",
    "pump_speed_rpm": 2500,
    "well_depth_ft": 6000,
    "water_cut_pct": 50,
    "downhole_pressure_psi": 2000,
    "motor_current_amp": 40
  }'

# Predict failure mode
curl -X POST http://localhost:5007/api/failure \
  -H "Content-Type: application/json" \
  -d '{
    "lift_type": "ESP",
    "pump_speed_rpm": 3200,
    "well_depth_ft": 8000,
    "water_cut_pct": 80,
    "downhole_pressure_psi": 3000,
    "motor_current_amp": 75
  }'

# Get model info
curl http://localhost:5007/api/models
```

### Using Python
```python
import requests

# Optimize lift parameters
response = requests.post("http://localhost:5007/api/optimize", json={
    "lift_type": "ESP",
    "pump_speed_rpm": 2500,
    "well_depth_ft": 6000,
    "water_cut_pct": 50,
    "downhole_pressure_psi": 2000,
    "motor_current_amp": 40
})
result = response.json()
print(f"Current: {result['current_production_bbl_d']} bbl/d")
print(f"Optimized: {result['predicted_production_bbl_d']} bbl/d")
print(f"Improvement: {result['improvement_pct']}%")

# Predict failure
response = requests.post("http://localhost:5007/api/failure", json={
    "lift_type": "ESP",
    "pump_speed_rpm": 3200,
    "well_depth_ft": 8000,
    "water_cut_pct": 80,
    "downhole_pressure_psi": 3000,
    "motor_current_amp": 75
})
failure = response.json()
print(f"Failure mode: {failure['predicted_failure_mode']}")
print(f"Confidence: {failure['confidence']:.1%}")
```

## Models
- **Lift Optimizer**: GradientBoostingRegressor - predicts production rate and optimizes parameters
- **Failure Predictor**: RandomForestClassifier - classifies 5 failure modes with probability distribution
