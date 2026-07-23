import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

from artificial_lift_optimization.data_generator import generate_dataset, LIFT_TYPES
from artificial_lift_optimization.utils.preprocessor import Preprocessor


class LiftOptimizer:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
        )
        self.preprocessor = Preprocessor()
        self.metadata = {}

    def train(self, df):
        X = self.preprocessor.fit_transform(df)
        y = df["production_bbl_d"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="r2")

        self.metadata = {
            "mae": float(mae),
            "r2": float(r2),
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "n_samples": len(df),
            "n_features": X.shape[1],
            "feature_importance": dict(zip(
                self.preprocessor.feature_cols,
                [float(v) for v in self.model.feature_importances_],
            )),
        }
        return self.metadata

    def predict_production(self, record_dict):
        X = self.preprocessor.transform_single(record_dict)
        return float(self.model.predict(X)[0])

    def optimize(self, base_record, lift_type, param_ranges=None, n_iter=500):
        if param_ranges is None:
            from artificial_lift_optimization.data_generator import LIFT_PARAMS
            param_ranges = LIFT_PARAMS[lift_type]

        best_prod = -1
        best_params = {}
        rng = np.random.RandomState(42)

        for _ in range(n_iter):
            candidate = base_record.copy()
            candidate["lift_type"] = lift_type
            for param, (lo, hi) in param_ranges.items():
                if hi > lo:
                    candidate[param] = round(rng.uniform(lo, hi), 2)
                else:
                    candidate[param] = 0

            prod = self.predict_production(candidate)
            if prod > best_prod:
                best_prod = prod
                best_params = {
                    k: candidate[k] for k in param_ranges
                }

        return {
            "lift_type": lift_type,
            "optimal_params": best_params,
            "predicted_production_bbl_d": round(best_prod, 1),
        }

    def save(self, path):
        joblib.dump({"model": self.model, "preprocessor": self.preprocessor, "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.preprocessor = data["preprocessor"]
        obj.metadata = data["metadata"]
        return obj
