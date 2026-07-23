# Artificial Lift Optimization

Machine learning project for optimizing artificial lift systems in oil wells.

## Overview

This project provides ML-based optimization for three types of artificial lift systems:

- **ESP (Electric Submersible Pump):** Optimizes pump speed, motor current for maximum production.
- **Rod Pump:** Optimizes rod load, pump speed for maximum production.
- **Gas Lift:** Optimizes gas injection rate for maximum production.

## Models

- **Lift Optimizer** (`GradientBoostingRegressor`): Predicts production rate and finds optimal parameters to maximize output.
- **Failure Predictor** (`RandomForestClassifier`): Classifies equipment failure modes (normal, pump wear, gas lock, rod fatigue, motor overheat).

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web dashboard |
| `/api/health` | GET | System health check |
| `/api/models` | GET | Model metadata and metrics |
| `/api/optimize` | POST | Optimize lift parameters |
| `/api/failure` | POST | Predict equipment failure mode |

## Quick Start

```bash
pip install -r requirements.txt
python train.py
python app.py
```

The API runs on port 5007.

## Example API Calls

### Optimize

```bash
curl -X POST http://localhost:5007/api/optimize -H "Content-Type: application/json" -d '{"lift_type":"ESP","pump_speed_rpm":2500,"well_depth_ft":6000,"water_cut_pct":50,"downhole_pressure_psi":2000,"motor_current_amp":40}'
```

### Predict Failure

```bash
curl -X POST http://localhost:5007/api/failure -H "Content-Type: application/json" -d '{"lift_type":"ESP","pump_speed_rpm":3200,"well_depth_ft":8000,"water_cut_pct":80,"downhole_pressure_psi":3000,"motor_current_amp":75}'
```

## Running Tests

```bash
python test_api.py
```

## Project Structure

```
artificial-lift-optimization/
├── artificial_lift_optimization/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lift_optimizer.py
│   │   └── failure_predictor.py
│   └── utils/
│       ├── __init__.py
│       └── preprocessor.py
├── outputs/
│   └── models/
├── templates/
│   └── index.html
├── .github/
│   └── workflows/
│       └── ci.yml
├── train.py
├── app.py
├── test_api.py
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Author

Elaborado por Ing. Kelvin Cabrera
