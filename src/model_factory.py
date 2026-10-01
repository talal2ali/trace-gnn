"""
Model factory keyed by profile.

  tree   -> XGBoost (GPU on your machine); falls back to sklearn HistGradientBoosting
            (also handles NaN natively) when xgboost is unavailable.
  linear -> LogisticRegression
  mlp    -> MLPClassifier

The tree profile consumes NaN directly; linear/mlp receive the fully imputed+scaled
matrix from the FoldPreprocessor.
"""

from __future__ import annotations

from experiment_config import ExperimentConfig


def make_model(profile: str, cfg: ExperimentConfig):
    if profile == "tree":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                eval_metric="aucpr", n_jobs=-1, random_state=cfg.random_state,
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_depth=6, learning_rate=0.05, max_iter=400,
                random_state=cfg.random_state,
            )
    if profile == "linear":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=cfg.random_state)
    if profile == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60,
                             random_state=cfg.random_state)
    raise ValueError(profile)


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
