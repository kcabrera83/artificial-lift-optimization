import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Preprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_cols = ["lift_type"]
        self.feature_cols = None
        self._fitted = False

    def fit_transform(self, df):
        df = df.copy()
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        self.feature_cols = [c for c in df.columns if c not in ("production_bbl_d", "failure_mode")]
        X = df[self.feature_cols].values.astype(np.float64)
        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True
        return X_scaled

    def transform(self, df):
        assert self._fitted, "Must call fit_transform first"
        df = df.copy()
        for col in self.categorical_cols:
            df[col] = self.label_encoders[col].transform(df[col])
        X = df[self.feature_cols].values.astype(np.float64)
        return self.scaler.transform(X)

    def transform_single(self, record_dict):
        df = pd.DataFrame([record_dict])
        return self.transform(df)

    def encode_labels(self, labels):
        le = LabelEncoder()
        return le.fit_transform(labels), le

    @property
    def n_features(self):
        return len(self.feature_cols) if self.feature_cols else 0
