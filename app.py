import os
import json
import traceback
from flask import Flask, request, jsonify, render_template

from artificial_lift_optimization.models.lift_optimizer import LiftOptimizer
from artificial_lift_optimization.models.failure_predictor import FailurePredictor

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")

optimizer = None
predictor = None


def load_models():
    global optimizer, predictor
    opt_path = os.path.join(MODEL_DIR, "lift_optimizer.pkl")
    pred_path = os.path.join(MODEL_DIR, "failure_predictor.pkl")
    if os.path.exists(opt_path):
        optimizer = LiftOptimizer.load(opt_path)
    if os.path.exists(pred_path):
        predictor = FailurePredictor.load(pred_path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": {
            "optimizer": optimizer is not None,
            "failure_predictor": predictor is not None,
        },
    })


@app.route("/api/models", methods=["GET"])
def models_info():
    info = {}
    if optimizer:
        info["optimizer"] = {
            "type": "GradientBoostingRegressor",
            "r2": optimizer.metadata.get("r2"),
            "mae": optimizer.metadata.get("mae"),
            "cv_r2_mean": optimizer.metadata.get("cv_r2_mean"),
            "n_samples": optimizer.metadata.get("n_samples"),
            "n_features": optimizer.metadata.get("n_features"),
            "feature_importance": optimizer.metadata.get("feature_importance"),
        }
    if predictor:
        info["failure_predictor"] = {
            "type": "RandomForestClassifier",
            "accuracy": predictor.metadata.get("accuracy"),
            "cv_accuracy_mean": predictor.metadata.get("cv_accuracy_mean"),
            "n_classes": predictor.metadata.get("n_classes"),
            "classes": predictor.metadata.get("classes"),
            "per_class_report": predictor.metadata.get("per_class_report"),
        }
    return jsonify(info)


@app.route("/api/optimize", methods=["POST"])
def optimize():
    if optimizer is None:
        return jsonify({"error": "Optimizer model not loaded"}), 503
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        lift_type = data.get("lift_type", "ESP")
        base_record = {
            "lift_type": lift_type,
            "pump_speed_rpm": data.get("pump_speed_rpm", 0),
            "rod_load_klbf": data.get("rod_load_klbf", 0),
            "gas_injection_mcf": data.get("gas_injection_mcf", 0),
            "downhole_pressure_psi": data.get("downhole_pressure_psi", 2000),
            "motor_current_amp": data.get("motor_current_amp", 30),
            "well_depth_ft": data.get("well_depth_ft", 6000),
            "water_cut_pct": data.get("water_cut_pct", 50),
        }
        n_iter = data.get("n_iterations", 500)

        result = optimizer.optimize(base_record, lift_type, n_iter=n_iter)
        result["current_production_bbl_d"] = optimizer.predict_production(base_record)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/failure", methods=["POST"])
def failure():
    if predictor is None:
        return jsonify({"error": "Failure predictor model not loaded"}), 503
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        record = {
            "lift_type": data.get("lift_type", "ESP"),
            "pump_speed_rpm": data.get("pump_speed_rpm", 0),
            "rod_load_klbf": data.get("rod_load_klbf", 0),
            "gas_injection_mcf": data.get("gas_injection_mcf", 0),
            "downhole_pressure_psi": data.get("downhole_pressure_psi", 2000),
            "motor_current_amp": data.get("motor_current_amp", 30),
            "well_depth_ft": data.get("well_depth_ft", 6000),
            "water_cut_pct": data.get("water_cut_pct", 50),
        }

        result = predictor.predict(record)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


load_models()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007, debug=True)
