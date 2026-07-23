import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from artificial_lift_optimization.data_generator import generate_dataset, FAILURE_MODES
from artificial_lift_optimization.utils.preprocessor import Preprocessor


class FailurePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.preprocessor = Preprocessor()
        self.failure_le = None
        self.metadata = {}

    def train(self, df):
        X = self.preprocessor.fit_transform(df)
        y, self.failure_le = self.preprocessor.encode_labels(df["failure_mode"])

        min_class_count = min(np.bincount(y))
        use_stratify = min_class_count >= 2

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if use_stratify else None,
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        all_labels = list(range(len(self.failure_le.classes_)))
        report = classification_report(
            y_test, y_pred,
            labels=all_labels,
            target_names=self.failure_le.classes_,
            output_dict=True,
            zero_division=0,
        )

        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="accuracy")

        self.metadata = {
            "accuracy": float(acc),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "n_samples": len(df),
            "n_classes": len(self.failure_le.classes_),
            "classes": list(self.failure_le.classes_),
            "per_class_report": {
                cls: {
                    "precision": float(report[cls]["precision"]),
                    "recall": float(report[cls]["recall"]),
                    "f1-score": float(report[cls]["f1-score"]),
                }
                for cls in self.failure_le.classes_
            },
        }
        return self.metadata

    def predict(self, record_dict):
        X = self.preprocessor.transform_single(record_dict)
        pred_idx = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        label = self.failure_le.inverse_transform([pred_idx])[0]
        confidence = {
            self.failure_le.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)
        }
        return {"failure_mode": label, "confidence": confidence}

    def save(self, path):
        joblib.dump({
            "model": self.model,
            "preprocessor": self.preprocessor,
            "failure_le": self.failure_le,
            "metadata": self.metadata,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.preprocessor = data["preprocessor"]
        obj.failure_le = data["failure_le"]
        obj.metadata = data["metadata"]
        return obj
