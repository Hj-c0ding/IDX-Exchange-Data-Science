"""Phase 2: Week 7 validation experiments for LightGBM and XGBoost.

STRICT PROTOCOL:
- Uses ONLY `val_split == 'train'` (2025-04..2026-03, price-filtered) and `val_split == 'val'` (2026-04).
- NEVER reads `split == 'test'`.
- Preprocessors (QuantileClipper, SimpleImputer, TargetEncoder / RareCategoryGrouper) are fit on
  `val_train` ONLY; `val_rows` is transformed and passed to `eval_set` solely for early stopping
  (cap=2000, patience=50).

SELECTION RULE (locked BEFORE running/inspecting validation results):
1. Per library (LightGBM, XGBoost), find the best F0 config and best alternative feature-set config
   (F3 / F2) by lowest validation MdAPE (tie-break: lower validation MAE).
2. Carry-over rule from Week 6: move off F0 to F3 (or F2) ONLY IF validation MdAPE improves by at least
   0.10 percentage points (>= 0.0010) AND validation MAE is also lower than the best F0 config.
   Otherwise keep F0.
3. Within the selected feature set per library, choose the configuration with the lowest validation MdAPE
   (tie-break: lower validation MAE).
4. Sanity check (STOP condition): if any validation MdAPE < 5.0% or > 30% relative improvement over
   RF F0 validation MdAPE (8.163% * 0.70 = 5.714%), raise an assertion error and stop for leakage audit.
"""
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import boosting as bt
import features as ft
import model_pipeline as mp

SEED = 42
MIN_FS_GAIN_MDAPE = 0.0010  # 0.10 pp threshold carried over from Week 6
RF_VAL_MDAPE = 0.08162994159950387

data = pd.read_csv("outputs/03_features_v3.csv", dtype={"PostalCode": "object", "SchoolDistrict": "object"})
# Enforce zero access to test split
val_sub = data.loc[data["val_split"].isin(["train", "val"])].copy()
assert "test" not in set(val_sub["split"]), "LEAKAGE ERROR: test split present in validation subset!"
del data

val_train, val_bounds = mp.filter_train_prices(val_sub.loc[val_sub["val_split"].eq("train")])
val_rows = val_sub.loc[val_sub["val_split"].eq("val")]
print(f"Validation setup: val_train={len(val_train):,} (bounds ${val_bounds[0]:,.0f}-${val_bounds[1]:,.0f}), val={len(val_rows):,}")

import time
records = []
existing_df = pd.read_csv("outputs/week7_validation.csv") if Path("outputs/week7_validation.csv").exists() else pd.DataFrame()
prep_cache = {}
y_tr_log = np.log(val_train[mp.TARGET].to_numpy(dtype=float))


def get_prepped(fs, cat):
    key = (fs, cat)
    if key not in prep_cache:
        cols = ft.input_columns(fs, "tree")
        t0 = time.perf_counter()
        prep = bt.make_boosting_preprocessor(fs, categorical_handling=cat)
        X_tr_p = prep.fit_transform(val_train[cols], y_tr_log)
        X_val_p = prep.transform(val_rows[cols])
        prep_cache[key] = (X_tr_p, X_val_p, time.perf_counter() - t0)
    return prep_cache[key]


def run_one(tag, lib, fs, cat, obj, params):
    p_str = json.dumps(params, sort_keys=True)
    if not existing_df.empty:
        hit = existing_df.loc[
            existing_df["stage"].eq(tag)
            & existing_df["library"].eq(lib)
            & existing_df["feature_set"].eq(fs)
            & existing_df["categorical_handling"].eq(cat)
            & existing_df["objective"].eq(obj)
            & existing_df["params"].eq(p_str)
        ]
        if not hit.empty:
            row = hit.iloc[0].to_dict()
            records.append(row)
            return row
    res = bt.fit_validate_with_early_stopping(
        lib,
        fs,
        val_train,
        val_rows,
        categorical_handling=cat,
        objective=obj,
        max_estimators=2000,
        patience=50,
        prepped=get_prepped(fs, cat),
        **params,
    )
    row = {
        "stage": tag,
        "library": lib,
        "feature_set": fs,
        "categorical_handling": cat,
        "objective": obj,
        "params": json.dumps(params, sort_keys=True),
        "learning_rate": params.get("learning_rate", 0.05),
        "max_depth": params.get("max_depth", -1 if lib == "LightGBM" else 6),
        "num_leaves": params.get("num_leaves", 31 if lib == "LightGBM" else np.nan),
        "min_child": params.get("min_child_samples" if lib == "LightGBM" else "min_child_weight", 20 if lib == "LightGBM" else 10),
        "subsample": params.get("subsample", 1.0),
        "colsample_bytree": params.get("colsample_bytree", 1.0),
        "reg_lambda": params.get("reg_lambda", 0.0 if lib == "LightGBM" else 1.0),
        "best_iteration": res["best_iteration"],
        "val_MdAPE": res["val_MdAPE"],
        "val_MAE": res["val_MAE"],
        "val_RMSE": res["val_RMSE"],
        "val_MAPE": res["val_MAPE"],
        "val_R2": res["val_R2"],
        "fit_s": res["fit_s"],
        "n_features": res["n_features"],
    }
    records.append(row)
    print(
        f"[{len(records):02d}] {tag:12s} | {lib:8s} | {fs} | {cat:6s} | {obj:13s} | "
        f"iter={res['best_iteration']:4d} | MdAPE={res['val_MdAPE']:.2%} | "
        f"MAE=${res['val_MAE']:,.0f} | {res['fit_s']:.1f}s"
    )
    # Leakage sanity check
    assert res["val_MdAPE"] >= 0.05, f"STOP: suspiciously low val_MdAPE={res['val_MdAPE']:.2%} (<5%)"
    assert res["val_MdAPE"] >= RF_VAL_MDAPE * 0.70, (
        f"STOP: val_MdAPE={res['val_MdAPE']:.2%} improves >30% relative over RF ({RF_VAL_MDAPE:.2%})"
    )
    return row


# Stage 1: Baseline factorial screen (F0, F2, F3 x target/native + objective comparison)
print("\n--- Stage 1: Baseline Categorical, Feature-Set & Objective Screen ---")
for lib in ["LightGBM", "XGBoost"]:
    base_p = (
        {"learning_rate": 0.05, "max_depth": 8, "num_leaves": 63, "min_child_samples": 30, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0}
        if lib == "LightGBM"
        else {"learning_rate": 0.05, "max_depth": 6, "min_child_weight": 30, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0}
    )
    # Compare target vs native on F0, F2, F3 with squared_error
    for fs in ["F0", "F2", "F3"]:
        for cat in ["target", "native"]:
            run_one("baseline_cat", lib, fs, cat, "squared_error", base_p)
    # Compare objectives (huber and mae) on F0 and F3 across both target and native
    for fs in ["F0", "F3"]:
        for cat in ["target", "native"]:
            for obj in ["huber", "mae"]:
                run_one("baseline_obj", lib, fs, cat, obj, base_p)

# Stage 2: Seeded Random Search (20 configs per library across F0 and F3)
print("\n--- Stage 2: Seeded Random Search (20 configs per library, seed=42) ---")
rng = np.random.default_rng(SEED)

lgb_space = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8, 10],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [10, 30, 100],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_lambda": [0.0, 1.0, 10.0],
}

xgb_space = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8, 10],
    "min_child_weight": [10, 30, 100],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_lambda": [0.0, 1.0, 10.0],
}


def sample_configs(space, n, rng):
    seen = set()
    configs = []
    while len(configs) < n:
        c = {k: rng.choice(v).item() for k, v in space.items()}
        if "num_leaves" in c:
            # Cap num_leaves at 2**max_depth - 1 for consistency when max_depth is small
            max_possible = (2 ** int(c["max_depth"])) - 1
            if c["num_leaves"] > max_possible:
                c["num_leaves"] = min(127, max_possible)
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(c)
    return configs


# Pick the best (categorical_handling, objective) from Stage 1 per (library, feature_set) to focus the 20 random search draws
df_stage1 = pd.DataFrame(records)
for lib, space in [("LightGBM", lgb_space), ("XGBoost", xgb_space)]:
    sampled = sample_configs(space, 10, rng)  # 10 random parameter draws evaluated on BOTH F0 and F3 = 20 configs per library
    for fs in ["F0", "F3"]:
        sub = df_stage1.loc[df_stage1["library"].eq(lib) & df_stage1["feature_set"].eq(fs)].sort_values(["val_MdAPE", "val_MAE"])
        best_cat = sub.iloc[0]["categorical_handling"]
        best_obj = sub.iloc[0]["objective"]
        print(f"Random search for {lib} {fs} using Stage-1 winner: cat={best_cat}, obj={best_obj}")
        for idx, p in enumerate(sampled, 1):
            run_one(f"rand_{idx:02d}", lib, fs, best_cat, best_obj, p)
    # Also evaluate F2 and F3 with the top tuned config from F0/F3 so F0, F2, F3 are compared apples-to-apples at the best tuned config
    df_cur = pd.DataFrame(records)
    top_row = df_cur.loc[df_cur["library"].eq(lib)].sort_values(["val_MdAPE", "val_MAE"]).iloc[0]
    run_one("tuned_F2", lib, "F2", top_row["categorical_handling"], top_row["objective"], json.loads(top_row["params"]))
    if top_row["feature_set"] != "F3":
        # Check if (F3, top_row.cat, top_row.obj, top_row.params) was already run
        already = df_cur.loc[
            df_cur["library"].eq(lib)
            & df_cur["feature_set"].eq("F3")
            & df_cur["categorical_handling"].eq(top_row["categorical_handling"])
            & df_cur["objective"].eq(top_row["objective"])
            & df_cur["params"].eq(top_row["params"])
        ]
        if len(already) == 0:
            run_one("tuned_F3", lib, "F3", top_row["categorical_handling"], top_row["objective"], json.loads(top_row["params"]))

df_val = pd.DataFrame(records)
df_val.to_csv("outputs/week7_validation.csv", index=False)
print(f"\nSaved {len(df_val)} validation runs to outputs/week7_validation.csv")

# Apply locked selection rule
print("\n--- Locked Selection Rule Evaluation ---")
for lib in ["LightGBM", "XGBoost"]:
    sub = df_val.loc[df_val["library"].eq(lib)]
    best_by_fs = sub.sort_values(["val_MdAPE", "val_MAE"]).groupby("feature_set").head(1).set_index("feature_set")
    b_f0 = best_by_fs.loc["F0"]
    alt_sets = [s for s in best_by_fs.index if s != "F0"]
    best_alt_fs = best_by_fs.loc[alt_sets].sort_values(["val_MdAPE", "val_MAE"]).index[0]
    b_alt = best_by_fs.loc[best_alt_fs]
    mdape_gain = b_f0["val_MdAPE"] - b_alt["val_MdAPE"]
    mae_better = b_alt["val_MAE"] < b_f0["val_MAE"]
    adopt_alt = (mdape_gain >= MIN_FS_GAIN_MDAPE) and mae_better
    chosen_fs = best_alt_fs if adopt_alt else "F0"
    winner = best_by_fs.loc[chosen_fs]
    print(
        f"{lib}: best F0 MdAPE={b_f0['val_MdAPE']:.2%} (MAE=${b_f0['val_MAE']:,.0f}) vs "
        f"best {best_alt_fs} MdAPE={b_alt['val_MdAPE']:.2%} (MAE=${b_alt['val_MAE']:,.0f}) | "
        f"gain={mdape_gain*100:+.3f} pp, MAE better={mae_better} -> CHOSEN={chosen_fs} "
        f"(cat={winner['categorical_handling']}, obj={winner['objective']}, iter={winner['best_iteration']}, params={winner['params']})"
    )
