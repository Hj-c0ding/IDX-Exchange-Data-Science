"""Week 6 validation ablation: F0..F3 x {LR, DT, RF} x district encoding. VALIDATION ONLY (no test rows used)."""
import time

import pandas as pd

import features as ft
import model_pipeline as mp

COLS = {'PostalCode': 'object', 'SchoolDistrict': 'object', 'HighSchoolDistrict': 'object'}
data = ft.add_engineered_features(pd.read_csv('outputs/03_features_v3.csv', dtype=COLS)) \
    if 'BedBathRatio' not in pd.read_csv('outputs/03_features_v3.csv', nrows=1).columns \
    else pd.read_csv('outputs/03_features_v3.csv', dtype=COLS)
val_train, _ = mp.filter_train_prices(data.loc[data['val_split'].eq('train')])
val = data.loc[data['val_split'].eq('val')]
print(f'val-train {len(val_train):,} | val {len(val):,}', flush=True)

PARAMS = {'LR': {}, 'DT': {'max_depth': 18, 'min_samples_leaf': 5}, 'RF': {'min_samples_leaf': 3}}
runs = []
for name in ['LR', 'DT', 'RF']:
    for fs in ['F0', 'F1', 'F2', 'F3']:
        encs = ['onehot', 'target'] if fs in ('F2', 'F3') else ['none']
        for enc in encs:
            model = ft.make_model(name, fs, district_encoding='onehot' if enc == 'none' else enc, **PARAMS[name])
            cols = ft.input_columns(fs, 'lr' if name == 'LR' else 'tree')
            t = time.perf_counter()
            model.fit(val_train[cols], val_train[mp.TARGET])
            fit_s = time.perf_counter() - t
            m = ft.full_metrics(val[mp.TARGET], model.predict(val[cols]))
            runs.append({'model': name, 'feature_set': fs, 'district_encoding': enc, **m, 'fit_s': fit_s})
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in runs[-1].items()}, flush=True)

# LR district grouping threshold sweep (validation only)
for mf in [5, 20, 50]:
    model = ft.make_model('LR', 'F2', 'onehot', district_min_frequency=mf)
    cols = ft.input_columns('F2', 'lr')
    model.fit(val_train[cols], val_train[mp.TARGET])
    m = ft.full_metrics(val[mp.TARGET], model.predict(val[cols]))
    runs.append({'model': 'LR', 'feature_set': 'F2', 'district_encoding': f'onehot_mf{mf}', **m, 'fit_s': float('nan')})
    print('LR F2 mf', mf, {k: round(v, 4) for k, v in m.items()}, flush=True)

pd.DataFrame(runs).to_csv('outputs/week6_validation.csv', index=False)
print('saved outputs/week6_validation.csv')
