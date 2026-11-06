"""Phase 0: Verify lightgbm/xgboost imports and reproduce Week 6 F0/F3 baselines."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

import features as ft
import model_pipeline as mp

print(f"lightgbm={lgb.__version__}, xgboost={xgb.__version__}, joblib={joblib.__version__}")

data = pd.read_csv("outputs/03_features_v3.csv", dtype={"PostalCode": "object", "SchoolDistrict": "object"})
train, bounds = mp.filter_train_prices(data.loc[data["split"].eq("train")])
test = data.loc[data["split"].eq("test")]
print(f"train={len(train):,} (bounds ${bounds[0]:,.0f}-${bounds[1]:,.0f}) | test={len(test):,}")

configs = [
    ("LR", "F0", "none", {}, (0.4860, 235941, 0.0980)),
    ("LR", "F3", "onehot", {}, (0.5024, 228644, 0.0959)),
    ("DT", "F0", "none", {"max_depth": 18, "min_samples_leaf": 5}, (0.4303, 256543, 0.1032)),
    ("RF", "F0", "none", {"min_samples_leaf": 3}, (0.4445, 220621, 0.0819)),
]

for m, fs, enc, params, (exp_r2, exp_mae, exp_mdape) in configs:
    cols = ft.input_columns(fs, "lr" if m == "LR" else "tree")
    model = ft.make_model(m, fs, district_encoding="onehot" if enc == "none" else enc, **params)
    model.fit(train[cols], train[mp.TARGET])
    tm = ft.full_metrics(test[mp.TARGET], model.predict(test[cols]))
    ok = (
        abs(tm["R2"] - exp_r2) < 5e-4
        and abs(tm["MAE"] - exp_mae) < 5
        and abs(tm["MdAPE"] - exp_mdape) < 5e-4
    )
    print(
        f"{m} {fs}: R2={tm['R2']:.4f} (exp {exp_r2:.4f}), "
        f"MAE=${tm['MAE']:,.0f} (exp ${exp_mae:,}), "
        f"MdAPE={tm['MdAPE']:.2%} (exp {exp_mdape:.2%}) -> MATCH={ok}"
    )
    assert ok, f"Regression mismatch on {m} {fs}!"

print("PHASE 0 REGRESSION PASSED ALL CHECKS.")

