"""Week 7 gradient boosting wrappers (LightGBM and XGBoost) and validation helpers.

Reuses the exact row-validity, train price bounds, coordinate columns, engineered features,
and log-target structure from `model_pipeline.py` and `features.py`, while adapting the
ColumnTransformer for boosted trees:
- Numeric columns: train-only QuantileClipper(0.005, 0.995) + median SimpleImputer(add_indicator=True).
  No StandardScaler is used (trees are invariant to monotonic linear rescaling).
- Categorical columns (`PostalCode`, plus `SchoolDistrict` for F2/F3):
  (a) `categorical_handling='target'`: 5-fold cross-fitted TargetEncoder(random_state=42) on log price.
  (b) `categorical_handling='native'`: train-only rare-level grouping (`PostalCode` min_freq=5,
      `SchoolDistrict` min_freq=20) output as pandas `category` dtype for LightGBM `categorical_feature`
      and XGBoost `enable_categorical=True` (`tree_method='hist'`).
"""
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore", message=".*eval_set.*")
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

import features as ft
import model_pipeline as mp

RANDOM_STATE = 42
DEFAULT_MIN_FREQ = {"PostalCode": 5, "SchoolDistrict": 20}


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group rare/unseen categorical levels using train-only frequencies and output pandas Categorical."""

    def __init__(self, min_frequency=5, rare_label="infrequent"):
        self.min_frequency = min_frequency
        self.rare_label = rare_label

    def _freq_for(self, col):
        if isinstance(self.min_frequency, dict):
            return self.min_frequency.get(col, 5)
        return int(self.min_frequency)

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        self.columns_ = list(df.columns)
        self.frequent_ = {}
        self.categories_ = {}
        for col in self.columns_:
            s = df[col].astype("string").fillna("Unknown")
            vc = s.value_counts(dropna=False)
            mf = self._freq_for(col)
            kept = sorted(vc[vc >= mf].index.astype(str).tolist())
            self.frequent_[col] = set(kept)
            cats = kept + ([self.rare_label] if self.rare_label not in self.frequent_[col] else [])
            self.categories_[col] = cats
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=df.index)
        for col in self.columns_:
            s = df[col].astype("string").fillna("Unknown").astype(str)
            mapped = s.where(s.isin(self.frequent_[col]), self.rare_label)
            out[col] = pd.Categorical(mapped, categories=self.categories_[col], ordered=False)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_ if input_features is None else input_features, dtype=object)


class TreeQuantileClipper(mp.QuantileClipper):
    """Subclass of mp.QuantileClipper that stores feature_names_in_ for pandas set_output."""

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.asarray([f"x{i}" for i in range(np.asarray(X).shape[1])], dtype=object)
        return super().fit(X, y)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features, dtype=object)


def _tree_numeric_pipe():
    """Train-only quantile clipping + median imputation with missingness flags (no scaling for trees)."""
    return Pipeline([
        ("clip", TreeQuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ])


def make_boosting_preprocessor(
    feature_set,
    categorical_handling="target",
    zip_min_frequency=5,
    district_min_frequency=20,
):
    """Build ColumnTransformer for gradient boosting on `feature_set` ('F0', 'F1', 'F2', 'F3')."""
    num, extra, cats = ft.feature_columns(feature_set, "tree")
    num_cols = num + extra
    transformers = [("num", _tree_numeric_pipe(), num_cols)]

    if categorical_handling == "target":
        enc = TargetEncoder(
            target_type="continuous",
            cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        )
        transformers.append(("cat", enc, cats))
        ct = ColumnTransformer(transformers, verbose_feature_names_out=False)
        ct.set_output(transform="pandas")
        return ct
    elif categorical_handling == "native":
        min_freq_map = {"PostalCode": zip_min_frequency, "SchoolDistrict": district_min_frequency}
        grouper = RareCategoryGrouper(min_frequency=min_freq_map)
        transformers.append(("cat", grouper, cats))
        ct = ColumnTransformer(transformers, verbose_feature_names_out=False)
        ct.set_output(transform="pandas")
        return ct
    elif categorical_handling == "onehot":
        transformers.append((
            "zip",
            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=zip_min_frequency, sparse_output=False),
            list(mp.LOCATION_CATEGORICAL),
        ))
        if "SchoolDistrict" in cats:
            transformers.append((
                "district",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=district_min_frequency, sparse_output=False),
                ["SchoolDistrict"],
            ))
        return ColumnTransformer(transformers, verbose_feature_names_out=False)
    else:
        raise ValueError(f"Unknown categorical_handling: {categorical_handling}")


def _resolve_objective(name, objective, huber_alpha=0.15):
    """Map canonical objective name ('squared_error', 'huber', 'mae') to library-specific params and eval_metric."""
    if name in ("LightGBM", "LGBM", "lgbm"):
        if objective == "squared_error":
            return {"objective": "regression"}, "rmse"
        elif objective == "huber":
            return {"objective": "huber", "alpha": huber_alpha}, "huber"
        elif objective == "mae":
            return {"objective": "regression_l1"}, "l1"
        else:
            raise ValueError(f"Unknown objective {objective} for LightGBM")
    elif name in ("XGBoost", "XGB", "xgb"):
        if objective == "squared_error":
            return {"objective": "reg:squarederror"}, "rmse"
        elif objective == "huber":
            return {"objective": "reg:pseudohubererror", "huber_slope": huber_alpha, "base_score": 13.5}, "mphe"
        elif objective == "mae":
            return {"objective": "reg:absoluteerror", "base_score": 13.5}, "mae"
        else:
            raise ValueError(f"Unknown objective {objective} for XGBoost")
    else:
        raise ValueError(f"Unknown model name: {name}")


def _make_estimator(name, categorical_handling="target", objective="squared_error", n_jobs=-1, **params):
    """Instantiate LGBMRegressor or XGBRegressor with fixed seed 42 and requested hyperparameters."""
    obj_params, _ = _resolve_objective(name, objective)
    merged = {**obj_params, **params}
    if name in ("LightGBM", "LGBM", "lgbm"):
        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 1.0,
            "subsample_freq": 1,
            "colsample_bytree": 1.0,
            "reg_lambda": 0.0,
            "random_state": RANDOM_STATE,
            "n_jobs": n_jobs,
            "verbose": -1,
            "importance_type": "gain",
        }
        defaults.update(merged)
        if defaults.get("subsample", 1.0) == 1.0:
            defaults["subsample_freq"] = 0
        return lgb.LGBMRegressor(**defaults)
    elif name in ("XGBoost", "XGB", "xgb"):
        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 10,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "enable_categorical": (categorical_handling == "native"),
            "random_state": RANDOM_STATE,
            "n_jobs": n_jobs,
            "verbosity": 0,
            "importance_type": "total_gain",
        }
        defaults.update(merged)
        return xgb.XGBRegressor(**defaults)
    else:
        raise ValueError(f"Unknown boosting model: {name}")


def make_boosting_model(
    name,
    feature_set,
    categorical_handling="target",
    objective="squared_error",
    zip_min_frequency=5,
    district_min_frequency=20,
    n_jobs=-1,
    **params,
):
    """Create a full leakage-safe TransformedTargetRegressor(Pipeline([('prep', ...), ('model', ...)])) for GBM."""
    prep = make_boosting_preprocessor(
        feature_set,
        categorical_handling=categorical_handling,
        zip_min_frequency=zip_min_frequency,
        district_min_frequency=district_min_frequency,
    )
    est = _make_estimator(
        name,
        categorical_handling=categorical_handling,
        objective=objective,
        n_jobs=n_jobs,
        **params,
    )
    pipe = Pipeline([("prep", prep), ("model", est)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)


def fit_validate_with_early_stopping(
    name,
    feature_set,
    val_train_df,
    val_df,
    categorical_handling="target",
    objective="squared_error",
    max_estimators=2000,
    patience=50,
    n_jobs=-1,
    record_curve=False,
    prepped=None,
    **params,
):
    """Fit preprocessor on val_train ONLY, then fit GBM with early stopping on val_df (April 2026).

    Returns dict with validation metrics (dollar scale), best_iteration, fit_s, and optional learning curve.
    """
    y_tr_raw = val_train_df[mp.TARGET].to_numpy(dtype=float)
    y_tr_log = np.log(y_tr_raw)
    y_val_raw = val_df[mp.TARGET].to_numpy(dtype=float)
    y_val_log = np.log(y_val_raw)

    if prepped is not None:
        X_tr_p, X_val_p, prep_s = prepped
    else:
        cols = ft.input_columns(feature_set, "tree")
        t_prep = time.perf_counter()
        prep = make_boosting_preprocessor(feature_set, categorical_handling=categorical_handling)
        X_tr_p = prep.fit_transform(val_train_df[cols], y_tr_log)
        X_val_p = prep.transform(val_df[cols])
        prep_s = time.perf_counter() - t_prep

    t0 = time.perf_counter()

    _, eval_metric = _resolve_objective(name, objective)
    run_params = {**params, "n_estimators": max_estimators}

    if name in ("LightGBM", "LGBM", "lgbm"):
        est = _make_estimator(
            name,
            categorical_handling=categorical_handling,
            objective=objective,
            n_jobs=n_jobs,
            **run_params,
        )
        eval_sets = [(X_tr_p, y_tr_log), (X_val_p, y_val_log)] if record_curve else [(X_val_p, y_val_log)]
        eval_names = ["train", "val"] if record_curve else ["val"]
        est.fit(
            X_tr_p,
            y_tr_log,
            eval_set=eval_sets,
            eval_names=eval_names,
            eval_metric=eval_metric,
            callbacks=[lgb.early_stopping(stopping_rounds=patience, verbose=False)],
        )
        best_iter = int(est.best_iteration_ or max_estimators)
        pred_log = est.predict(X_val_p, num_iteration=best_iter)
        curve = est.evals_result_ if record_curve else None
    elif name in ("XGBoost", "XGB", "xgb"):
        est = _make_estimator(
            name,
            categorical_handling=categorical_handling,
            objective=objective,
            n_jobs=n_jobs,
            early_stopping_rounds=patience,
            eval_metric=eval_metric,
            **run_params,
        )
        eval_sets = [(X_tr_p, y_tr_log), (X_val_p, y_val_log)] if record_curve else [(X_val_p, y_val_log)]
        est.fit(
            X_tr_p,
            y_tr_log,
            eval_set=eval_sets,
            verbose=False,
        )
        # XGBoost best_iteration is 0-indexed, so tree count is best_iteration + 1
        best_iter = int(est.best_iteration + 1) if getattr(est, "best_iteration", None) is not None else max_estimators
        pred_log = est.predict(X_val_p)
        curve = est.evals_result() if record_curve else None
    else:
        raise ValueError(f"Unknown model name: {name}")

    fit_s = prep_s + (time.perf_counter() - t0)
    pred_dollars = np.exp(np.clip(pred_log, 10.0, 20.0))
    metrics = ft.full_metrics(y_val_raw, pred_dollars)
    return {
        "library": name,
        "feature_set": feature_set,
        "categorical_handling": categorical_handling,
        "objective": objective,
        "params": params,
        "best_iteration": best_iter,
        "val_MdAPE": metrics["MdAPE"],
        "val_MAE": metrics["MAE"],
        "val_RMSE": metrics["RMSE"],
        "val_MAPE": metrics["MAPE"],
        "val_R2": metrics["R2"],
        "fit_s": fit_s,
        "n_features": X_tr_p.shape[1],
        "curve": curve,
    }


def scale_iterations(best_iteration, n_train, n_val_train):
    """Scale validation early-stopped iteration count proportionally to full training window size."""
    return max(1, int(round(best_iteration * (float(n_train) / float(n_val_train)))))
