import os
import sys
import time

from artificial_lift_optimization.data_generator import generate_dataset
from artificial_lift_optimization.models.lift_optimizer import LiftOptimizer
from artificial_lift_optimization.models.failure_predictor import FailurePredictor

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  ARTIFICIAL LIFT OPTIMIZATION - CatBoost + pymoo")
    print("=" * 60)

    print("\n[1/4] Generating synthetic dataset (8000 samples)...")
    t0 = time.time()
    df = generate_dataset(n_samples=8000, random_state=42)
    print(f"  Generated {len(df)} samples in {time.time()-t0:.2f}s")
    print(f"  Lift types: {df['lift_type'].value_counts().to_dict()}")
    print(f"  Failure modes: {df['failure_mode'].value_counts().to_dict()}")
    print(f"  Production range: {df['production_bbl_d'].min():.0f} - {df['production_bbl_d'].max():.0f} bbl/d")

    print("\n[2/4] Training Lift Optimizer (CatBoost + pymoo NSGA2)...")
    t0 = time.time()
    optimizer = LiftOptimizer()
    opt_meta = optimizer.train(df)
    optimizer.save(os.path.join(OUTPUT_DIR, "lift_optimizer.pkl"))
    print(f"  Trained in {time.time()-t0:.2f}s")
    print(f"  R2 Score: {opt_meta['r2']:.4f}")
    print(f"  MAE: {opt_meta['mae']:.2f} bbl/d")
    print(f"  CV R2 (5-fold): {opt_meta['cv_r2_mean']:.4f} +/- {opt_meta['cv_r2_std']:.4f}")
    print(f"  Top features:")
    sorted_fi = sorted(opt_meta["feature_importance"].items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_fi[:5]:
        print(f"    {feat}: {imp:.4f}")

    print("\n[3/4] Training Failure Predictor (CatBoost)...")
    t0 = time.time()
    predictor = FailurePredictor()
    pred_meta = predictor.train(df)
    predictor.save(os.path.join(OUTPUT_DIR, "failure_predictor.pkl"))
    print(f"  Trained in {time.time()-t0:.2f}s")
    print(f"  Accuracy: {pred_meta['accuracy']:.4f}")
    print(f"  CV Accuracy (5-fold): {pred_meta['cv_accuracy_mean']:.4f} +/- {pred_meta['cv_accuracy_std']:.4f}")
    print(f"  Per-class metrics:")
    for cls, metrics in pred_meta["per_class_report"].items():
        print(f"    {cls}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f}")

    print("\n[4/4] Running pymoo NSGA2 optimization demo...")
    base_record = {
        "lift_type": "ESP",
        "pump_speed_rpm": 2500,
        "rod_load_klbf": 0,
        "gas_injection_mcf": 0,
        "downhole_pressure_psi": 2000,
        "motor_current_amp": 40,
        "well_depth_ft": 6000,
        "water_cut_pct": 50,
    }
    for lt in ["ESP", "rod_pump", "gas_lift"]:
        result = optimizer.optimize(base_record, lt, n_iter=200)
        print(f"  {lt}: {result['predicted_production_bbl_d']:.0f} bbl/d -> {result['optimal_params']}")

    print(f"\nModels saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
