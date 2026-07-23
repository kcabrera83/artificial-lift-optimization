# Architecture - Artificial Lift Optimization

## System Overview
```
                    +-------------------+
                    |   Flask Server    |
                    |   (app.py)        |
                    |   Port 5007       |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
+-------------v-----------+  +-------------v-----------+
| Lift Optimizer           |  | Failure Predictor       |
| (GradientBoosting)       |  | (RandomForest)          |
| Maximize Production      |  | Classify Failure Modes  |
+-------------+-----------+  +-------------+-----------+
              |                             |
+-------------v-----------------------------v-----------+
|        Optimization Engine + Preprocessor              |
|   (Grid Search / Random Search + Feature Encoding)     |
+------------------------+------------------------------+
                         |
               +---------v-----------+
               |  Synthetic Dataset  |
               |  (8000 samples)     |
               +--------------------+
```

## Components

### Data Layer
- **Data Source**: Synthetic dataset generator producing 8000 samples across 3 lift types
- **Lift Types**: ESP, rod_pump, gas_lift (balanced distribution)
- **Features**: 8 input features (lift_type, pump_speed, rod_load, gas_injection, pressure, motor_current, depth, water_cut)
- **Targets**: Production rate (bbl/d) for optimizer, failure mode (5 classes) for predictor
- **Preprocessing**: One-hot encoding for lift_type, standard scaling for numeric features

### Model Layer

#### Lift Optimizer
- **Algorithm**: GradientBoostingRegressor
- **Input**: 8 features describing well and lift system configuration
- **Output**: Predicted production rate (bbl/d)
- **Optimization**: Random search over parameter space (500 iterations default)
- **Objective**: Maximize predicted production
- **Metrics**: R2, MAE, 5-fold CV R2
- **Feature Importance**: Top features for production prediction

#### Failure Predictor
- **Algorithm**: RandomForestClassifier
- **Input**: Same 8 features
- **Output**: Failure mode classification (5 classes) with probabilities
- **Classes**: normal, pump_wear, gas_lock, rod_fatigue, motor_overheat
- **Metrics**: Accuracy, 5-fold CV Accuracy, per-class Precision/Recall/F1

### Optimization Engine
- **Method**: Random search over parameter ranges
- **Parameters by Lift Type**:
  - ESP: pump_speed_rpm (1500-4000), motor_current_amp (20-80)
  - Rod Pump: rod_load_klbf (5-40), pump_speed_rpm (2-12)
  - Gas Lift: gas_injection_mcf (100-1000)
- **Constraint**: Parameters must stay within physically valid ranges
- **Output**: Optimal parameters + predicted production + improvement percentage

### API Layer
- **Framework**: Flask
- **Endpoints**: 5 REST endpoints (health, models, optimize, failure, docs)
- **Error Handling**: Detailed error responses with tracebacks
- **Model Loading**: Eager loading at startup

### Dashboard Layer
- **Frontend**: Flask + HTML/CSS/JS
- **Features**: Optimization form, failure prediction, model metrics visualization

## Data Flow

### Optimization Flow
1. **Input**: Current lift parameters + lift type + n_iterations
2. **Validation**: Check model availability, validate input
3. **Current Prediction**: Predict current production rate
4. **Optimization Loop**: Generate random parameter combinations, predict production for each
5. **Selection**: Choose parameters with maximum predicted production
6. **Response**: Optimal parameters, predicted production, improvement percentage

### Failure Prediction Flow
1. **Input**: Lift system operating parameters
2. **Validation**: Check model availability, validate input
3. **Prediction**: RandomForest classifies failure mode
4. **Probability**: Return probability distribution across all 5 modes
5. **Response**: Predicted failure mode, confidence, probabilities

## Training Pipeline
1. Generate synthetic dataset (8000 samples, 3 lift types)
2. Print dataset statistics
3. Train Lift Optimizer (GradientBoosting)
4. Print R2, MAE, CV R2, top features
5. Train Failure Predictor (RandomForest)
6. Print Accuracy, CV Accuracy, per-class metrics
7. Run optimization demo for each lift type
8. Save models to `outputs/models/`

## File Structure
```
artificial-lift-optimization/
├── artificial_lift_optimization/
│   ├── data_generator.py       # Synthetic dataset generation
│   ├── models/
│   │   ├── lift_optimizer.py   # Production optimization
│   │   └── failure_predictor.py # Failure classification
│   └── utils/
│       └── preprocessor.py     # Feature encoding
├── outputs/models/             # Trained models
├── templates/index.html        # Dashboard
├── app.py                      # Flask server
├── train.py                    # Training pipeline
├── test_api.py                 # API tests
├── setup.py                    # Package setup
└── .github/workflows/ci.yml   # CI/CD
```
