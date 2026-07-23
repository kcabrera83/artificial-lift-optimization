import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS

from artificial_lift_optimization.data_generator import generate_dataset, LIFT_TYPES
from artificial_lift_optimization.utils.preprocessor import Preprocessor


class LiftProductionProblem(Problem):
    def __init__(self, model, preprocessor, base_record, lift_type, param_ranges):
        param_names = list(param_ranges.keys())
        n_var = len(param_names)
        xl = np.array([param_ranges[k][0] for k in param_names])
        xu = np.array([param_ranges[k][1] for k in param_names])
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)
        self.model = model
        self.preprocessor = preprocessor
        self.base_record = base_record
        self.lift_type = lift_type
        self.param_names = param_names

    def _evaluate(self, X, out, *args, **kwargs):
        results = []
        for row in X:
            candidate = self.base_record.copy()
            candidate["lift_type"] = self.lift_type
            for i, name in enumerate(self.param_names):
                candidate[name] = round(float(row[i]), 2)
            X_t = self.preprocessor.transform_single(candidate)
            prod = float(self.model.predict(X_t)[0])
            results.append(-prod)
        out["F"] = np.array(results).reshape(-1, 1)


class LiftOptimizer:
    def __init__(self):
        self.model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            loss_function='RMSE',
            verbose=0,
            random_seed=42,
        )
        self.preprocessor = Preprocessor()
        self.metadata = {}

    def train(self, df):
        X = self.preprocessor.fit_transform(df)
        y = df["production_bbl_d"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)

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

        problem = LiftProductionProblem(
            self.model, self.preprocessor, base_record, lift_type, param_ranges
        )

        algorithm = NSGA2(
            pop_size=min(n_iter, 200),
            sampling=LHS(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
        )

        res = minimize(
            problem,
            algorithm,
            ('n_gen', max(n_iter // 20, 10)),
            seed=42,
            verbose=False,
        )

        best_idx = np.argmin(res.F[:, 0])
        best_x = res.X[best_idx]
        best_prod = -float(res.F[best_idx, 0])

        best_params = {k: round(float(best_x[i]), 2) for i, k in enumerate(param_ranges.keys())}

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
