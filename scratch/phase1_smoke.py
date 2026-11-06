"""Phase 1 smoke test: verify boosting.py pipelines and compare target vs native categoricals on val_split."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import boosting as bt
import features as ft
import model_pipeline as mp

data = pd.read_csv("outputs/03_features_v3.csv", dtype={"PostalCode": "object", "SchoolDistrict": "object"})
# Strictly validation split only — never read split == 'test'
assert "test" not in set(data.loc[data["val_split"].isin(["train", "val"]), "split"])
val_train, val_bounds = mp.filter_train_prices(data.loc[data["val_split"].eq("train")])
val_rows = data.loc[data["val_split"].eq("val")]
print(f"val_train={len(val_train):,} (bounds ${val_bounds[0]:,.0f}-${val_bounds[1]:,.0f}) | val={len(val_rows):,}")

rows = []
for lib in ["LightGBM", "XGBoost"]:
    for fs in ["F0", "F3"]:
        for cat in ["target", "native"]:
            res = bt.fit_validate_with_early_stopping(
                lib,
                fs,
                val_train,
                val_rows,
                categorical_handling=cat,
                objective="squared_error",
                max_estimators=2000,
                patience=50,
            )
            rows.append({k: v for k, v in res.items() if k not in ("params", "curve")})
            print(
                f"{lib:8s} {fs} {cat:6s}: best_iter={res['best_iteration']:4d}, "
                f"val_MdAPE={res['val_MdAPE']:.2%}, val_MAE=${res['val_MAE']:,.0f}, "
                f"n_feat={res['n_features']}, fit_s={res['fit_s']:.2f}s"
            )

# Also verify make_boosting_model Pipeline wrapper on a small 20-tree fit
for lib in ["LightGBM", "XGBoost"]:
    for cat in ["target", "native"]:
        cols = ft.input_columns("F3", "tree")
        m = bt.make_boosting_model(lib, "F3", categorical_handling=cat, n_estimators=20)
        m.fit(val_train[cols], val_train[mp.TARGET])
        p = m.predict(val_rows[cols])
        assert len(p) == len(val_rows) and np.all(np.isfinite(p))

print("PHASE 1 SMOKE TEST PASSED.")

