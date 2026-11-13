"""E3 (Week 8): price-band and geography error breakdowns on the test month (2026-05).

Uses the Week 7 saved test predictions (scratch/week7_test_predictions.csv) - the test month is NOT
re-scored. Every group table carries n, the signed median % error, a 95% bootstrap CI on MdAPE, and a
flag for groups with fewer than 200 rows.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import features as ft, geo, model_pipeline as mp

SEED, N_BOOT, MIN_N = 42, 2000, 200
MODELS = ["LR", "DT", "RF", "LightGBM", "XGBoost"]
HEADLINE = "XGBoost"

v3 = pd.read_csv("outputs/03_features_v3.csv", dtype={"PostalCode": "object", "SchoolDistrict": "object"})
test = v3.loc[v3["split"].eq("test")].reset_index(drop=True)
preds = pd.read_csv("scratch/week7_test_predictions.csv")
y = test["ClosePrice"].to_numpy(float)
assert np.array_equal(preds["ClosePrice"].to_numpy(float), y), "prediction/test alignment broken"
P = {m: preds[m].to_numpy(float) for m in MODELS}

# County from the school-district boundary file (identity attribute only, no demographics).
districts = geo.load_districts()
cmap = districts.drop_duplicates("CDSCode").set_index("CDSCode")["CountyName"]
test["County"] = test["SchoolDistrict"].map(cmap).fillna(geo.UNKNOWN).astype(object)
print(f"test rows {len(test):,} | county named for {100*test['County'].ne(geo.UNKNOWN).mean():.2f}% "
      f"| {test['County'].nunique()} counties")

# ---- Price bands -------------------------------------------------------------------------------
train, bounds = mp.filter_train_prices(v3.loc[v3["split"].eq("train")])
y_train = train[mp.TARGET].to_numpy(float)
qe = np.quantile(y_train, [0.2, 0.4, 0.6, 0.8])
print(f"train-only quintile edges (n_train={len(train):,}, bounds ${bounds[0]:,.0f}-${bounds[1]:,.0f}): "
      + ", ".join(f"${e:,.0f}" for e in qe))
qi = np.digitize(y, qe)
# Labels deliberately carry NO dollar sign: a label with two "$" is parsed as LaTeX math by both
# matplotlib mathtext and the notebook's MathJax. Units are stated in the surrounding text (USD).
qlab = [f"Q1 <{qe[0]/1e3:,.0f}k", f"Q2 {qe[0]/1e3:,.0f}-{qe[1]/1e3:,.0f}k",
        f"Q3 {qe[1]/1e3:,.0f}-{qe[2]/1e3:,.0f}k", f"Q4 {qe[2]/1e3:,.0f}k-{qe[3]/1e6:,.2f}M",
        f"Q5 >{qe[3]/1e6:,.2f}M"]
quintile = pd.Series([qlab[i] for i in qi], index=test.index)

fe = [0, 5e5, 7.5e5, 1e6, 1.5e6, 3e6, np.inf]
flab = ["<500k", "500k-750k", "750k-1M", "1M-1.5M", "1.5M-3M", ">3M"]
fixed = pd.Series([flab[i] for i in np.digitize(y, fe[1:-1], right=False)], index=test.index)

# ---- Metric + bootstrap helpers ----------------------------------------------------------------
rng = np.random.default_rng(SEED)

def group_rows(scope, name, mask, order):
    n = int(mask.sum())
    idx = rng.integers(0, n, size=(N_BOOT, n)) if n >= 20 else None
    out = []
    for m in MODELS:
        yt, yp = y[mask], P[m][mask]
        met = ft.full_metrics(yt, yp)
        ape = np.abs(yp - yt) / yt
        spe = (yp - yt) / yt
        lo, hi = (np.percentile(np.median(ape[idx], axis=1), [2.5, 97.5]) * 100) if idx is not None else (np.nan, np.nan)
        out.append({"scope": scope, "group": name, "order": order, "n": n,
                    "pct_of_test": 100 * n / len(y), "median_actual_price": float(np.median(yt)),
                    "model": m, "R2": met["R2"], "MAE": met["MAE"], "RMSE": met["RMSE"],
                    "MAPE": met["MAPE"], "MdAPE": met["MdAPE"],
                    "median_signed_pct_error": 100 * float(np.median(spe)),
                    "MdAPE_ci_lo_pct": float(lo), "MdAPE_ci_hi_pct": float(hi),
                    "below_min_n": n < MIN_N})
    return out

rows = []
for o, lab in enumerate(qlab):
    rows += group_rows("train_quintile", lab, (quintile == lab).to_numpy(), o)
for o, lab in enumerate(flab):
    rows += group_rows("fixed_band", lab, (fixed == lab).to_numpy(), o)
band = pd.DataFrame(rows)
band.to_csv("outputs/week8_by_price_band.csv", index=False)

# Regression cross-check against Week 7's quintile table (same edges, same predictions)
w7q = pd.read_csv("outputs/week7_mdape_by_quintile.csv")
chk = band[band.scope.eq("train_quintile")].pivot(index="order", columns="model", values="MdAPE")
# XGBoost's Week 7 test predictions were produced as float32, so its CSV round-trip carries ~1e-7
# relative precision; the other four models must be bit-exact.
TOLS = {m: (1e-6 if m == "XGBoost" else 1e-12) for m in MODELS}
worst = {m: max(abs(chk.loc[i, m] - w7q[m].iloc[i]) for i in range(5)) for m in MODELS}
bad = {m: v for m, v in worst.items() if v >= TOLS[m]}
assert not bad, f"quintile MdAPE drifted vs week7_mdape_by_quintile.csv: {bad}"
print("Cross-check vs outputs/week7_mdape_by_quintile.csv (max abs MdAPE diff per model): "
      + ", ".join(f"{m} {v:.1e}" for m, v in worst.items()) + " (PASS)")

# ---- Geography: county (min-count rule), ZIP (reported, but no ZIP reaches the minimum) ---------
cc = test["County"].value_counts()
big = sorted(cc[cc >= MIN_N].index)
rows = []
for o, c in enumerate(big):
    rows += group_rows("county", c, test["County"].eq(c).to_numpy(), o)
small_mask = (~test["County"].isin(big)).to_numpy()
if small_mask.sum():
    rows += group_rows("county", f"Other ({int((cc < MIN_N).sum())} counties, each n<{MIN_N})",
                       small_mask, len(big))
cty = pd.DataFrame(rows)
cty.to_csv("outputs/week8_by_county.csv", index=False)
print(f"\nCounties with n>={MIN_N}: {len(big)} of {test['County'].nunique()} "
      f"({100*cc[cc >= MIN_N].sum()/len(test):.1f}% of test rows); "
      f"{int((cc < MIN_N).sum())} pooled into 'Other'")

zc = test["PostalCode"].value_counts()
rows = []
for o, z in enumerate(zc.head(15).index):
    rows += group_rows("zip", str(z), test["PostalCode"].eq(z).to_numpy(), o)
zp = pd.DataFrame(rows)
zp.to_csv("outputs/week8_by_zip.csv", index=False)
print(f"ZIPs: {len(zc)} distinct, largest n={int(zc.iloc[0])}, {int((zc >= MIN_N).sum())} reach n>={MIN_N} "
      f"-> ZIP-level conclusions are NOT supportable; top 15 saved for transparency, all flagged.")

# ---- Band-vs-band contrasts for the headline model (independent two-sample bootstrap) -----------
def boot_md(mask):
    n = int(mask.sum())
    ape = np.abs(P[HEADLINE][mask] - y[mask]) / y[mask]
    return np.median(ape[rng.integers(0, n, size=(N_BOOT, n))], axis=1) * 100

con = []
for scope, labs, ser in [("train_quintile", qlab, quintile), ("fixed_band", flab, fixed)]:
    md = {l: float(np.median(np.abs(P[HEADLINE][(ser == l).to_numpy()] - y[(ser == l).to_numpy()])
                             / y[(ser == l).to_numpy()]) * 100) for l in labs}
    base = min(md, key=md.get)
    bb = boot_md((ser == base).to_numpy())
    for l in labs:
        if l == base:
            continue
        diff = boot_md((ser == l).to_numpy()) - bb
        lo, hi = np.percentile(diff, [2.5, 97.5])
        con.append({"scope": scope, "model": HEADLINE, "band": l, "best_band": base,
                    "band_MdAPE_pct": md[l], "best_band_MdAPE_pct": md[base],
                    "diff_pp": md[l] - md[base], "diff_ci_lo_pp": float(lo), "diff_ci_hi_pp": float(hi),
                    "worse_than_best_beyond_noise": bool(lo > 0),
                    "n_band": int((ser == l).sum()), "n_best_band": int((ser == base).sum())})
pd.DataFrame(con).to_csv("outputs/week8_band_contrasts.csv", index=False)

# ---- Console view -------------------------------------------------------------------------------
pd.set_option("display.width", 220)
for scope in ["train_quintile", "fixed_band"]:
    s = band[band.scope.eq(scope) & band.model.eq(HEADLINE)].sort_values("order")
    print(f"\n{scope} ({HEADLINE}):")
    print(s[["group", "n", "MdAPE", "MdAPE_ci_lo_pct", "MdAPE_ci_hi_pct",
             "median_signed_pct_error", "MAE", "below_min_n"]].to_string(index=False,
             formatters={"MdAPE": "{:.2%}".format, "MAE": "${:,.0f}".format}))
print(f"\nBand contrasts vs the best band ({HEADLINE}):")
print(pd.DataFrame(con)[["scope", "band", "best_band", "diff_pp", "diff_ci_lo_pp", "diff_ci_hi_pp",
                         "worse_than_best_beyond_noise", "n_band"]].to_string(index=False))
print("\ncounty (XGBoost):")
cs = cty[cty.model.eq(HEADLINE)].sort_values("n", ascending=False)
print(cs[["group", "n", "MdAPE", "MdAPE_ci_lo_pct", "MdAPE_ci_hi_pct", "median_signed_pct_error",
          "median_actual_price", "below_min_n"]].to_string(index=False,
          formatters={"MdAPE": "{:.2%}".format, "median_actual_price": "${:,.0f}".format}))
print("\nWrote outputs/week8_by_price_band.csv, week8_by_county.csv, week8_by_zip.csv, week8_band_contrasts.csv")
