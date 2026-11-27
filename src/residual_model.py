"""Wrapper sklearn cho mô hình lai Climatology base_margin + XGBoost học phần dư."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier


def _logit(p: np.ndarray) -> np.ndarray:
    p_c = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p_c / (1.0 - p_c))


class ClimatologyResidualXGB(BaseEstimator, ClassifierMixin):
    classes_ = np.array([0, 1])

    def __init__(self, f107_col: str, clim_tbl: pd.DataFrame, fallback_prob: float):
        self.f107_col = f107_col
        self.clim_tbl = clim_tbl
        self.fallback_prob = fallback_prob
        self.clf = XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            random_state=42, eval_metric="logloss",
        )

    def _margins(self, X: pd.DataFrame) -> np.ndarray:
        key = pd.DataFrame({
            "utc_hour": X["utc_hour"].to_numpy(),
            "hi_f107": (X[self.f107_col].fillna(150) >= 150).astype(int).to_numpy(),
        })
        p = pd.merge(key, self.clim_tbl, on=["utc_hour", "hi_f107"], how="left")["prob"].fillna(self.fallback_prob).to_numpy()
        return _logit(p)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.clf.fit(X, y, base_margin=self._margins(X))
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X, base_margin=self._margins(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
