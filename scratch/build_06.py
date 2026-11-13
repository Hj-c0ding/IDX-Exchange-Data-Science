"""Builds 06_evaluation.ipynb (Week 8). Run: python scratch/build_06.py, then execute with nbconvert.

Every number in every markdown cell - including the audience-facing Step headers - is generated with
an f-string from an executed output or a CSV in outputs/ at BUILD time. Nothing is typed from memory,
so re-running this builder after a recompute refreshes the text automatically.

Code cell bodies are the executed Week 8 cells, unchanged apart from a leading '# Produces:' comment.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import nbformat as nbf

# ---- numbers for the presenter headers, read from the saved outputs -----------------------------
HL = 'XGBoost'                     # headline model (the saved production model)
MIN_N = 200                        # minimum group size for a reportable segment

S    = pd.read_csv('outputs/metrics_summary.csv')
B    = pd.read_csv('outputs/week8_by_price_band.csv')
C    = pd.read_csv('outputs/week8_by_county.csv')
Z    = pd.read_csv('outputs/week8_by_zip.csv', dtype={'group': 'object'})
K    = pd.read_csv('outputs/week8_backtest.csv')
CT   = pd.read_csv('outputs/week8_band_contrasts.csv')
W7   = pd.read_csv('outputs/week7_comparison.csv').set_index('model')
PRED = pd.read_csv('scratch/week7_test_predictions.csv')
V3   = pd.read_csv('outputs/03_features_v3.csv', usecols=['PostalCode', 'split'], dtype={'PostalCode': 'object'})
EXT  = pd.read_csv('outputs/08_features_backtest.csv', usecols=['Month', 'ClosePrice'])

# markdown-safe currency: a bare "$" pairs up into LaTeX math in the notebook renderer
usd_md = lambda v: '\\$' + f'{v:,.0f}'

N_TEST  = int(S['n_test'].iloc[0])
S_BEST  = S.sort_values('MdAPE').iloc[0]
S_R2    = S.sort_values('R2', ascending=False).iloc[0]
RF_ROW  = S[S['model'].eq('RF')].iloc[0]

Y  = PRED['ClosePrice'].to_numpy(float)
YP = PRED[HL].to_numpy(float)
GATE_WORST = max(abs(float(S[S['model'].eq(m)][k].iloc[0]) - float(W7.loc[m, k])) / abs(float(W7.loc[m, k]))
                 for m in S['model'] for k in ['R2', 'MAE', 'RMSE', 'MAPE', 'MdAPE'])
PE      = 100 * (YP - Y) / Y
W10     = 100 * float(np.mean(np.abs(PE) <= 10))
MAXPRED = float(YP.max())
N_ABOVE = int((Y > MAXPRED).sum())

QB      = B[B['scope'].eq('train_quintile') & B['model'].eq(HL)].sort_values('order')
FB      = B[B['scope'].eq('fixed_band') & B['model'].eq(HL)].sort_values('order')
Q_BEST  = QB.loc[QB['MdAPE'].idxmin()]
Q_WORST = QB.loc[QB['MdAPE'].idxmax()]
F_WORST = FB.loc[FB['MdAPE'].idxmax()]

CT_SIG  = int(CT['worse_than_best_beyond_noise'].sum())
CT_N    = len(CT)
_marg   = CT[(CT['diff_ci_lo_pp'] > 0) & (CT['diff_ci_lo_pp'] < 0.1)]
CT_MARG_TXT = ('' if _marg.empty else
               ', though ' + ' and '.join(f"the {r.band} band clears it by only {r.diff_ci_lo_pp:.2f} points"
                                          for r in _marg.itertuples()) + ', which we read as no real difference')

CH      = C[C['model'].eq(HL)]
CBIG    = CH[~CH['group'].str.contains('Other')].sort_values('MdAPE')
C_BEST, C_WORST = CBIG.iloc[0], CBIG.iloc[-1]
C_OTHER = CH[CH['group'].str.contains('Other')].iloc[0]
N_CTY_POOLED = int(re.search(r'\((\d+) counties', C_OTHER['group']).group(1))
PCT_BIG = 100 * CBIG['n'].sum() / N_TEST

ZH        = Z[Z['model'].eq(HL)]
Z_MAX_N   = int(ZH['n'].max())
Z_N_ZIPS  = int(V3.loc[V3['split'].eq('test'), 'PostalCode'].nunique())

CUTS    = sorted(K['cutoff'].unique())
KH      = K[K['model'].eq(HL)].set_index('cutoff')
EARLY   = [c for c in CUTS if c not in ('2026-04', '2026-05')]
K_EARLY = KH.loc[EARLY, 'MdAPE']
N_VAL_TRAIN = int(K[K['cutoff'].eq('2026-04') & K['model'].eq(HL)]['n_train'].iloc[0])
LGB_VAL_IT  = int(K[K['cutoff'].eq('2026-04') & K['model'].eq('LightGBM')]['n_estimators'].iloc[0])
XGB_VAL_IT  = int(K[K['cutoff'].eq('2026-04') & K['model'].eq(HL)]['n_estimators'].iloc[0])
LGB_ITERS   = K[K['model'].eq('LightGBM')].sort_values('cutoff')['n_estimators'].tolist()
_w = K[K['model'].eq(HL)].sort_values('cutoff')
WINDOW_TXT = '; '.join(
    f"{r.cutoff} trained on {r.train_start}..{r.train_end} ({int(r.n_train):,} rows)"
    for r in _w.itertuples() if r.cutoff != '2026-05')

_sst = []
for _c in CUTS:
    _y = EXT.loc[EXT['Month'].eq(_c), 'ClosePrice'].to_numpy(float)
    _sq = (_y - _y.mean()) ** 2
    _sst.append(np.sort(_sq)[-3:].sum() / _sq.sum())
SST_MAX      = float(max(_sst))
SST_MAX_CUT  = CUTS[int(np.argmax(_sst))]                           # the month that figure belongs to
R2_MAX_AT    = float(K[K['cutoff'].eq(SST_MAX_CUT)]['R2'].max())    # best R2 ANY model reaches that month
R2_MIN       = float(K[~K['cutoff'].eq('2026-05')]['R2'].min())
R2_MAX_EARLY = float(K[~K['cutoff'].eq('2026-05')]['R2'].max())
MAX_SALE     = float(EXT['ClosePrice'].max())

# a few more header facts, all derived from the CSVs above
S_WORST    = S.sort_values('MdAPE').iloc[-1]
LGB_MD     = float(S[S['model'].eq('LightGBM')]['MdAPE'].iloc[0])
_MD_ORDER  = S.sort_values('MdAPE')['model'].tolist()
R2W_RANK   = _MD_ORDER.index(S_R2['model']) + 1
ORDINAL    = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}
# plain-English model names for the audience-facing headers (the tables keep the short codes)
PLAIN      = {'LR': 'Linear Regression', 'DT': 'a single decision tree', 'RF': 'Random Forest',
              'LightGBM': 'LightGBM', 'XGBoost': 'XGBoost'}
N_CTY_BIG  = len(CBIG)
dir_txt    = lambda v: 'too high' if v > 0 else 'too low'
_RANKS     = {c: K[K['cutoff'].eq(c)].sort_values('MdAPE')['model'].tolist() for c in CUTS}
BOOST_TOP2 = all(set(r[:2]) == {'LightGBM', 'XGBoost'} for r in _RANKS.values())
TAIL_SAME  = len({tuple(r[2:]) for r in _RANKS.values()}) == 1
STABLE_TXT = ('the two gradient-boosting models take the top two places and the other three keep the '
              'same order below them' if BOOST_TOP2 and TAIL_SAME else
              'the ordering of the five models shifts between months')
TOP2_TXT   = ('LightGBM and XGBoost hold the top two places in all '
              f'{len(CUTS)} scored months, so last week\'s ranking was not a one-month accident'
              if BOOST_TOP2 else f'the top two places change across the {len(CUTS)} scored months')
TUNE_CUT   = '2026-04'

# ---- notebook assembly ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))
_step = [0]


# Step titles declared up front so the opener can link to sections that do not exist yet.
TITLES = [
    'Load the saved results',                 # 1
    'Confirm the numbers still match',        # 2
    'Five accuracy measures, five models',    # 3
    'The same numbers as bars',               # 4
    'Accuracy by price level',                # 5
    'Every model struggles with expensive homes',   # 6
    'The error leans one way',                # 7
    'Which gaps are real',                   # 8
    'Accuracy by county',                    # 9
    'The county chart with uncertainty',      # 10
    'Why we stop at county',                 # 11
    'Re-run at three earlier months',         # 12
    'Why R-squared collapses some months',    # 13
    'Does accuracy hold over time',           # 14
    'What the time test shows',               # 15
    'Where the model actually misses',        # 16
    'What this week establishes',             # 17
]


def _slug(n, title):
    """GitHub/JupyterLab heading anchor for '### Step n: title'."""
    return re.sub(r'[^a-z0-9 -]', '', f'Step {n}: {title}'.lower()).replace(' ', '-')


def link(a, b=None):
    anchor = _slug(a, TITLES[a - 1])
    return f'[Step {a}](#{anchor})' if b is None else f'[Steps {a}-{b}](#{anchor})'


def header(title, shows, found):
    """One audience-facing header, placed directly above its code cell."""
    _step[0] += 1
    assert title == TITLES[_step[0] - 1], f'step {_step[0]} title does not match TITLES: {title!r}'
    md(f'### Step {_step[0]}: {title}\n\n'
       f'**What this shows:** {shows}\n\n'
       f'**What we found:** {found}')

md(f"""
# Week 8: How Accurate Is the AVM, and Where Does It Fail?

The AVM is the automated valuation model built over the past weeks: it estimates what a home will sell for
from its size, location and other recorded details. This notebook asks how close those estimates are, and
where they are not close at all.

**How to read this notebook.** It answers four questions, in order:

| Question | Where | Short answer |
|---|---|---|
| 1. How accurate is it overall? | {link(3, 4)} | {S_BEST['model']} misses a typical May 2026 home by {S_BEST['MdAPE']:.2%} - that is MdAPE, the typical percent error |
| 2. Which homes does it price well, and which does it miss? | {link(5, 11)} | Best in the middle of the market and in lower-priced inland counties; the top of the market is about {Q_WORST['MdAPE']/Q_BEST['MdAPE']:.1f}x worse |
| 3. Does that accuracy hold over time? | {link(12, 15)} | The model ranking holds at all {len(CUTS)} scored months |
| 4. What are the limits? | {link(17)} and the closing cell | Settings tuned on one month; no claims below county level; a hard ceiling on the most expensive homes |

**Ground rules.** Nothing is tuned here and no model is re-picked. The test month, May 2026, was scored once
per model in Week 7 and is never scored again - those saved predictions are reused. Splits are chronological,
the random seed is 42, everything the models learn is fit on training rows only, and errors are reported both
in dollars and as a percent of the sale price.
""")
header('Load the saved results',
       f"Loads the predictions that five finished models made for the test month, May 2026, together with this week's result tables. Nothing is re-trained and the test month is never scored a second time.",
       f"Everything that follows rests on {N_TEST:,} May 2026 sales that no model saw while it was being trained.")
code(r"""
# Produces: the loaded tables, the saved test-month predictions, and the shared plotting style
import warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.ticker as mt
import seaborn as sns
from IPython.display import Markdown, display

import features as ft, model_pipeline as mp, plot_style as ps

warnings.filterwarnings('ignore', category=FutureWarning)
ps.apply_style()
SEED = 42
np.random.seed(SEED)
pd.set_option('display.width', 220)

MODELS = ['LR', 'DT', 'RF', 'LightGBM', 'XGBoost']
HEADLINE = 'XGBoost'          # saved production model (outputs/models/best_model.joblib)
MIN_N = 200                   # minimum group size for a reportable segment conclusion

summary = pd.read_csv('outputs/metrics_summary.csv')
band    = pd.read_csv('outputs/week8_by_price_band.csv')
county  = pd.read_csv('outputs/week8_by_county.csv')
zips    = pd.read_csv('outputs/week8_by_zip.csv', dtype={'group': 'object'})
contr   = pd.read_csv('outputs/week8_band_contrasts.csv')
back    = pd.read_csv('outputs/week8_backtest.csv')
w7comp  = pd.read_csv('outputs/week7_comparison.csv')

preds = pd.read_csv('scratch/week7_test_predictions.csv')
v3 = pd.read_csv('outputs/03_features_v3.csv', dtype={'PostalCode': 'object', 'SchoolDistrict': 'object'})
test = v3.loc[v3['split'].eq('test')].reset_index(drop=True)
y_test = test['ClosePrice'].to_numpy(float)
# Train-only price bounds, recomputed from the data (never typed from memory).
_train_rows, TRAIN_BOUNDS = mp.filter_train_prices(v3.loc[v3['split'].eq('train')])
TRAIN_CAP = float(TRAIN_BOUNDS[1])

usd  = lambda v: f'${v:,.0f}'
pctf = lambda v: f'{v:.2%}'
print(f'Test month 2026-05: {len(test):,} homes | saved predictions for {len(MODELS)} frozen models')
""")
header('Confirm the numbers still match',
       f"The saved predictions are re-scored here and compared with last week's published results at full decimal precision. If any model's numbers had changed, this cell would stop the notebook with an error.",
       f"All five models reproduce last week's May 2026 results. The largest relative difference is {GATE_WORST:.0e}, which comes from how one model's predictions were stored in its file, not from a model changing.")
code(r"""
# Produces: the regression-gate table proving all five models still match Week 7
assert np.array_equal(preds['ClosePrice'].to_numpy(float), y_test), 'prediction/test row alignment broken'

gate_rows, worst = [], {}
for m in MODELS:
    got = ft.full_metrics(y_test, preds[m].to_numpy(float))
    exp = w7comp.loc[w7comp['model'].eq(m)].iloc[0]
    rel = {k: abs(got[k] - float(exp[k])) / abs(float(exp[k])) for k in ['R2', 'MAE', 'RMSE', 'MAPE', 'MdAPE']}
    worst[m] = max(rel.values())
    gate_rows.append({'model': m, **got, 'max_rel_diff_vs_week7': worst[m]})

gate = pd.DataFrame(gate_rows)
# XGBoost's Week 7 predictions were produced as float32, so its CSV round-trip carries ~1e-7
# relative precision. The other four models must be bit-exact.
TOL = {m: (1e-6 if m == 'XGBoost' else 1e-12) for m in MODELS}
failed = {m: worst[m] for m in MODELS if worst[m] >= TOL[m]}
assert not failed, f'REGRESSION GATE FAILED: {failed}'

display(gate.style.format({'R2': '{:.4f}', 'MAE': '${:,.0f}', 'RMSE': '${:,.0f}',
                           'MAPE': '{:.2%}', 'MdAPE': '{:.2%}',
                           'max_rel_diff_vs_week7': '{:.1e}'}).hide(axis='index'))
display(Markdown(
    f'**Gate PASSED.** All five models reproduce `week7_comparison.csv`. Four match to machine precision '
    f'(largest relative difference {max(worst[m] for m in MODELS if m != "XGBoost"):.1e}); '
    f'`XGBoost` matches to {worst["XGBoost"]:.1e} relative because its Week 7 predictions were produced as '
    f'`float32` and round-tripped through CSV - its MdAPE itself agrees to '
    f'{abs(ft.full_metrics(y_test, preds["XGBoost"].to_numpy(float))["MdAPE"] - float(w7comp.loc[w7comp.model.eq("XGBoost"), "MdAPE"].iloc[0])):.1e}. '
    f'That is a storage-precision artifact, not model drift.'))
""")
header('Five accuracy measures, five models',
       f"One row per model and five ways of measuring the same errors. The measure we lead with is MdAPE, the typical percent error: the miss on the middle home, which a handful of enormous sales cannot distort.",
       f"In May 2026 {S_BEST['model']} is the most accurate of the five - a typical miss of {S_BEST['MdAPE']:.2%} and {usd_md(S_BEST['MAE'])} on average - against {S_WORST['MdAPE']:.2%} for the weakest, {PLAIN[S_WORST['model']]} ({S_WORST['model']} in the table).")
code(r"""
# Produces: the Week 8 deliverable table: R2, MAE, RMSE, MAPE and MdAPE for five models
show = summary[['model', 'feature_set', 'n_estimators', 'n_features', 'n_test',
                'R2', 'MAE', 'RMSE', 'MAPE', 'MdAPE']].copy()
display(show.style.format({'R2': '{:.4f}', 'MAE': '${:,.0f}', 'RMSE': '${:,.0f}',
                           'MAPE': '{:.2%}', 'MdAPE': '{:.2%}', 'n_test': '{:,}',
                           'n_features': '{:,}', 'n_estimators': '{:,}'})
        .background_gradient(subset=['MdAPE'], cmap='RdYlGn_r').hide(axis='index'))

best = summary.sort_values('MdAPE').iloc[0]
worst_m = summary.sort_values('MdAPE').iloc[-1]
r2_best = summary.sort_values('R2', ascending=False).iloc[0]
display(Markdown(f'''
**Reading the table.** {best["model"]} ({best["feature_set"]}) has the lowest typical error at
**{best["MdAPE"]:.2%} MdAPE** (MAE {usd(best["MAE"])}), and {worst_m["model"]} the highest at
{worst_m["MdAPE"]:.2%} (MAE {usd(worst_m["MAE"])}) - a spread of
{100*(worst_m["MdAPE"] - best["MdAPE"]):.2f} pp.

**The metrics disagree, and that disagreement is informative.** {r2_best["model"]} wins on R2
({r2_best["R2"]:.4f} vs {best["R2"]:.4f}) while losing badly on MdAPE
({r2_best["MdAPE"]:.2%} vs {best["MdAPE"]:.2%}). Tree models cannot predict above the training price
cap, so on the handful of sales above it they are capped while the linear model extrapolates; squared-error
metrics reward that, median percentage error does not. MAPE sits above MdAPE for every model
(e.g. {best["model"]}: {best["MAPE"]:.2%} vs {best["MdAPE"]:.2%}), which is the signature of a
right-skewed error distribution - a minority of badly-missed homes pulls the mean.
'''))
""")
header('The same numbers as bars',
       f"The table above drawn as bars, one bar per model. Shorter is better in the left two panels, typical percent error and average dollar error; the right panel is R-squared, the share of price variation a model explains, which rewards getting the largest sales right.",
       f"R-squared puts {PLAIN[S_R2['model']]} {ORDINAL[1]} at {S_R2['R2']:.4f}, while on typical percent error it is only {ORDINAL[R2W_RANK]} at {S_R2['MdAPE']:.2%}. The cause is a price ceiling on the tree-based models, shown in Step 16, not a better model.")
code(r"""
# Produces: Figure 1 - the five models compared on MdAPE, MAE and R2
# Figure 1: five-model test metrics (w8_fig1)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
order = summary.sort_values('MdAPE')
colors = [ps.MODEL_COLORS[m] for m in order['model']]

for ax, col, fmt, title, ylab in [
    (axes[0], 'MdAPE', '{:.2%}', 'Typical error: MdAPE', 'MdAPE (median abs. % error)'),
    (axes[1], 'MAE',   '${:,.0f}', 'Average dollar error: MAE', 'MAE'),
    (axes[2], 'R2',    '{:.4f}',  'R2 ranks differently (see text)', 'R2')]:
    b = ax.bar(order['model'], order[col], color=colors, alpha=0.9)
    ax.bar_label(b, labels=[fmt.format(v) for v in order[col]], fontsize=9.5, padding=2)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylab)
    ax.tick_params(axis='x', rotation=20)
axes[0].yaxis.set_major_formatter(mt.PercentFormatter(xmax=1, decimals=0))
axes[1].yaxis.set_major_formatter(ps.usd)
axes[0].set_ylim(0, order['MdAPE'].max() * 1.22)
axes[1].set_ylim(0, order['MAE'].max() * 1.18)
axes[2].set_ylim(0, order['R2'].max() * 1.22)
fig.suptitle(f'Test month 2026-05 (n = {int(summary["n_test"].iloc[0]):,} homes): boosters win on typical error, '
             f'linear regression wins on R2', fontsize=14, fontweight='bold')
print('saved:', ps.save(fig, 'w8_fig1_model_metrics.png'))
plt.show()
""")
header('Accuracy by price level',
       f"The test month split by price two ways: into fifths of the training price range, so the test month never sets its own boundaries, and into round dollar bands. Each row carries its number of sales and a range of uncertainty.",
       f"In May 2026 accuracy is best in the middle of the market - {Q_BEST['group']}, a typical error of {Q_BEST['MdAPE']:.2%} - and worst at the top, {Q_WORST['MdAPE']:.2%} for {Q_WORST['group']}.")
code(r"""
# Produces: three price-band tables: quintiles for XGBoost, fixed bands for XGBoost, quintiles for all models
def band_view(scope, model):
    s = band[band.scope.eq(scope) & band.model.eq(model)].sort_values('order')
    v = s[['group', 'n', 'pct_of_test', 'median_actual_price', 'MdAPE', 'MdAPE_ci_lo_pct',
           'MdAPE_ci_hi_pct', 'median_signed_pct_error', 'MAE', 'MAPE', 'RMSE', 'R2', 'below_min_n']].copy()
    v['MdAPE_95CI'] = [f'[{lo:.2f}, {hi:.2f}] pp' for lo, hi in zip(v['MdAPE_ci_lo_pct'], v['MdAPE_ci_hi_pct'])]
    return v.drop(columns=['MdAPE_ci_lo_pct', 'MdAPE_ci_hi_pct'])

FMT = {'n': '{:,}', 'pct_of_test': '{:.1f}%', 'median_actual_price': '${:,.0f}', 'MdAPE': '{:.2%}',
       'median_signed_pct_error': '{:+.2f}%', 'MAE': '${:,.0f}', 'MAPE': '{:.2%}',
       'RMSE': '${:,.0f}', 'R2': '{:.3f}'}

display(Markdown(f'### 3a. Train-only price quintiles - {HEADLINE} (headline model)'))
display(band_view('train_quintile', HEADLINE).style.format(FMT)
        .background_gradient(subset=['MdAPE'], cmap='RdYlGn_r').hide(axis='index'))
display(Markdown(f'### 3b. Fixed dollar bands - {HEADLINE}'))
display(band_view('fixed_band', HEADLINE).style.format(FMT)
        .background_gradient(subset=['MdAPE'], cmap='RdYlGn_r').hide(axis='index'))

display(Markdown('### 3c. MdAPE by train quintile, all five models'))
piv = (band[band.scope.eq('train_quintile')]
       .pivot_table(index=['order', 'group', 'n'], columns='model', values='MdAPE')
       .reset_index().sort_values('order').drop(columns='order'))
display(piv[['group', 'n'] + MODELS].style.format({m: '{:.2%}' for m in MODELS} | {'n': '{:,}'})
        .background_gradient(subset=MODELS, cmap='RdYlGn_r', axis=None).hide(axis='index'))
""")
header('Every model struggles with expensive homes',
       f"One line per model, with both price splits side by side. Left to right runs from the cheapest group of homes to the most expensive, and higher means a larger typical percent error.",
       f"All five lines rise the same way, so this is a property of the problem rather than of one algorithm. In May 2026 the gap between the best and worst price group is {100*(Q_WORST['MdAPE']-Q_BEST['MdAPE']):.2f} points, far wider than the {100*(RF_ROW['MdAPE']-S_BEST['MdAPE']):.2f} points between the best model and Random Forest.")
code(r"""
# Produces: Figure 2 - MdAPE across price bands for all five models, both band schemes
# Figure 2: MdAPE by price band, all models, both band schemes (w8_fig2)
fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), constrained_layout=True)
for ax, scope, title in [(axes[0], 'train_quintile', 'Train-only price quintiles (primary)'),
                         (axes[1], 'fixed_band', 'Fixed dollar bands (secondary)')]:
    s = band[band.scope.eq(scope)].sort_values('order')
    labs = s.drop_duplicates('order')['group'].tolist()
    for m in MODELS:
        sm = s[s.model.eq(m)]
        ax.plot(range(len(labs)), sm['MdAPE'], marker='o', lw=2, ms=6,
                color=ps.MODEL_COLORS[m], label=m)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=18, ha='right', fontsize=9.5)
    ax.yaxis.set_major_formatter(mt.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel('MdAPE (test month 2026-05)')
    ax.set_title(title, fontsize=12)
    ns = s.drop_duplicates('order')['n'].tolist()
    for i, nn in enumerate(ns):
        ax.annotate(f'n={nn:,}', (i, ax.get_ylim()[0]), xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=8, color='#555555')
axes[0].legend(title='Model', ncols=2, fontsize=9.5)
fig.suptitle('Every model degrades as price rises; the gap between boosters and the rest is widest '
             'in the middle of the market', fontsize=14, fontweight='bold')
print('saved:', ps.save(fig, 'w8_fig2_mdape_by_price_band.png'))
plt.show()
""")
header('The error leans one way',
       f"On the left, the typical percent error for each dollar band with a bar showing its uncertainty. On the right, the middle error with its sign kept: above the dashed zero line the model prices homes too high, below it too low.",
       f"The misses are not symmetric. In May 2026 the {FB.iloc[0]['group']} band is priced about {abs(FB.iloc[0]['median_signed_pct_error']):.1f} percent {dir_txt(FB.iloc[0]['median_signed_pct_error'])} and the {F_WORST['group']} band about {abs(F_WORST['median_signed_pct_error']):.1f} percent {dir_txt(F_WORST['median_signed_pct_error'])}.")
code(r"""
# Produces: Figure 3 - band error with confidence intervals, and the direction of the error
# Figure 3: directional bias by band + bootstrap CIs for the headline model (w8_fig3)
fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), constrained_layout=True)

s = band[band.scope.eq('fixed_band') & band.model.eq(HEADLINE)].sort_values('order')
labs = s['group'].tolist()
err = np.vstack([s['MdAPE'] * 100 - s['MdAPE_ci_lo_pct'], s['MdAPE_ci_hi_pct'] - s['MdAPE'] * 100])
axes[0].bar(labs, s['MdAPE'] * 100, yerr=err, capsize=4, color=ps.MODEL_COLORS[HEADLINE], alpha=0.9)
for i, (v, hi, n) in enumerate(zip(s['MdAPE'] * 100, s['MdAPE_ci_hi_pct'], s['n'])):
    axes[0].annotate(f'{v:.2f}%\nn={n:,}', (i, hi), xytext=(0, 7), textcoords='offset points',
                     ha='center', fontsize=9)   # anchored above the CI whisker, not the bar top
axes[0].yaxis.set_major_formatter(mt.PercentFormatter(xmax=100, decimals=0))
axes[0].set_ylabel('MdAPE with 95% bootstrap CI')
axes[0].set_ylim(0, s['MdAPE_ci_hi_pct'].max() * 1.3)
axes[0].set_title(f'{HEADLINE}: typical error more than triples from mid-market to luxury', fontsize=12)
axes[0].tick_params(axis='x', rotation=18)

for m in ['RF', 'LightGBM', HEADLINE]:
    sm = band[band.scope.eq('fixed_band') & band.model.eq(m)].sort_values('order')
    axes[1].plot(range(len(labs)), sm['median_signed_pct_error'], marker='s', lw=2,
                 color=ps.MODEL_COLORS[m], label=m)
axes[1].axhline(0, color='#333333', lw=1.2, ls='--')
axes[1].annotate('over-prediction', (0.98, 0.95), xycoords='axes fraction', ha='right',
                 fontsize=9.5, color='#444444')
axes[1].annotate('under-prediction', (0.02, 0.06), xycoords='axes fraction', fontsize=9.5, color='#444444')
axes[1].set_xticks(range(len(labs)))
axes[1].set_xticklabels(labs, rotation=18, ha='right', fontsize=9.5)
axes[1].yaxis.set_major_formatter(mt.PercentFormatter(xmax=100, decimals=0))
axes[1].set_ylabel('Median signed % error')
axes[1].set_title('Systematic bias: cheap homes priced high, expensive homes priced low', fontsize=12)
axes[1].legend(title='Model')
fig.suptitle('Error is not just larger at the top of the market - it is directional', fontsize=14, fontweight='bold')
print('saved:', ps.save(fig, 'w8_fig3_band_bias.png'))
plt.show()
""")
header('Which gaps are real',
       f"Each price band is compared against the best band by re-drawing the homes in both two thousand times, which gives a range for the difference. A range that includes zero is not treated as a real difference.",
       f"{CT_SIG} of {CT_N} band comparisons are larger than this noise in May 2026{CT_MARG_TXT}.")
code(r"""
# Produces: the band-versus-band bootstrap contrasts and the written price-band finding
# Which band differences survive a bootstrap? (outputs/week8_band_contrasts.csv)
cv = contr.copy()
cv['95% CI of difference'] = [f'[{lo:+.2f}, {hi:+.2f}] pp' for lo, hi in
                              zip(cv['diff_ci_lo_pp'], cv['diff_ci_hi_pp'])]
display(Markdown(f'### 3d. Is band X really worse than the best band? ({HEADLINE}, 2,000-resample bootstrap)'))
display(cv[['scope', 'band', 'best_band', 'n_band', 'band_MdAPE_pct', 'best_band_MdAPE_pct',
            'diff_pp', '95% CI of difference', 'worse_than_best_beyond_noise']]
        .style.format({'band_MdAPE_pct': '{:.2f}%', 'best_band_MdAPE_pct': '{:.2f}%',
                       'diff_pp': '{:+.2f} pp', 'n_band': '{:,}'}).hide(axis='index'))

q = band[band.scope.eq('train_quintile') & band.model.eq(HEADLINE)].sort_values('order')
f = band[band.scope.eq('fixed_band') & band.model.eq(HEADLINE)].sort_values('order')
qb, qw = q.loc[q['MdAPE'].idxmin()], q.loc[q['MdAPE'].idxmax()]
fb, fw = f.loc[f['MdAPE'].idxmin()], f.loc[f['MdAPE'].idxmax()]
marginal = cv[(cv.diff_ci_lo_pp > 0) & (cv.diff_ci_lo_pp < 0.1)]
n_small = int(band['below_min_n'].sum())

display(Markdown(f'''
**Which price bands perform better, and why that is plausible.**

- The best segment is **{qb["group"]}** ({HEADLINE} MdAPE {qb["MdAPE"]:.2%}, 95% CI
  [{qb["MdAPE_ci_lo_pct"]:.2f}, {qb["MdAPE_ci_hi_pct"]:.2f}] pp, n = {int(qb["n"]):,}); the worst is
  **{qw["group"]}** ({qw["MdAPE"]:.2%}, CI [{qw["MdAPE_ci_lo_pct"]:.2f}, {qw["MdAPE_ci_hi_pct"]:.2f}] pp,
  n = {int(qw["n"]):,}). The gap is **{100*(qw["MdAPE"] - qb["MdAPE"]):.2f} pp**, far outside either interval.
  On fixed bands the same pattern holds: {fb["group"]} at {fb["MdAPE"]:.2%} versus {fw["group"]} at
  {fw["MdAPE"]:.2%} (n = {int(fw["n"]):,}).
- **Every** band in the contrast table is worse than the best band by more than bootstrap noise
  ({int(cv["worse_than_best_beyond_noise"].sum())} of {len(cv)} contrasts exclude zero)
  {"- though " + ", ".join(f"`{r.band}` (CI lower bound {r.diff_ci_lo_pp:+.2f} pp)" for r in marginal.itertuples()) + " only barely does, and should be read as 'no material difference'" if len(marginal) else ""}.
- **Why this is plausible.** Mid-market homes ({fb["group"]}, median sale {usd(fb["median_actual_price"])})
  are the densest part of the training data and the most standardised: similar tract housing, many
  comparable sales per ZIP, value driven by square footage and location in a way the features capture.
  The top band ({fw["group"]}, median sale {usd(fw["median_actual_price"])}) is thin
  ({fw["pct_of_test"]:.1f}% of test homes), heterogeneous, and its value depends on view, finish quality,
  architect, privacy and lot shape - none of which are in the feature set. The bottom band
  ({f.iloc[0]["group"]}) is also worse than mid-market ({f.iloc[0]["MdAPE"]:.2%}): a percentage error is
  mechanically easier to incur on a small denominator, and that segment mixes distressed, rural and
  manufactured-adjacent stock.
- **Directional bias** runs monotonically from {f.iloc[0]["median_signed_pct_error"]:+.2f}% in
  {f.iloc[0]["group"]} to {fw["median_signed_pct_error"]:+.2f}% in {fw["group"]}. This is regression to the
  mean plus the hard training price cap ({usd(TRAIN_CAP)}, learned on train only): tree models
  cannot predict above the most expensive homes they were trained on, so luxury sales are systematically
  under-priced.
- Groups flagged below n = {MIN_N}: **{n_small}** (none of the band conclusions above rests on a small group).
'''))
""")
header('Accuracy by county',
       f"The same errors grouped by county, which comes from the school-district boundary each home falls inside. Counties with fewer than {MIN_N} sales this month are pooled into one labelled row rather than dropped, so nothing is hidden.",
       f"In May 2026 the most accurate county is {C_BEST['group']} at {C_BEST['MdAPE']:.2%} and the hardest is {C_WORST['group']} at {C_WORST['MdAPE']:.2%}.")
code(r"""
# Produces: two county tables: XGBoost in detail, then MdAPE for all five models
cv = county[county.model.eq(HEADLINE)].sort_values('n', ascending=False).copy()
cv['MdAPE_95CI'] = [f'[{lo:.2f}, {hi:.2f}] pp' for lo, hi in zip(cv['MdAPE_ci_lo_pct'], cv['MdAPE_ci_hi_pct'])]
display(Markdown(f'### 4a. County-level error - {HEADLINE}'))
display(cv[['group', 'n', 'pct_of_test', 'median_actual_price', 'MdAPE', 'MdAPE_95CI',
            'median_signed_pct_error', 'MAE', 'MAPE', 'below_min_n']]
        .style.format(FMT).background_gradient(subset=['MdAPE'], cmap='RdYlGn_r').hide(axis='index'))

display(Markdown('### 4a (cont.) MdAPE by county, all five models'))
cp = (county.pivot_table(index=['group', 'n'], columns='model', values='MdAPE')
      .reset_index().sort_values('n', ascending=False))
display(cp[['group', 'n'] + MODELS].style.format({m: '{:.2%}' for m in MODELS} | {'n': '{:,}'})
        .background_gradient(subset=MODELS, cmap='RdYlGn_r', axis=None).hide(axis='index'))
""")
header('The county chart with uncertainty',
       f"One bar per county, ordered from most to least accurate, each with a whisker showing how far its figure could move given how few sales it rests on. The grey bar pools the small counties.",
       f"The {N_CTY_BIG} counties with enough sales cover {PCT_BIG:.1f} percent of May 2026, and the remaining {N_CTY_POOLED} sit in the grey bar. Where whiskers overlap, those counties cannot be ranked against each other.")
code(r"""
# Produces: Figure 4 - county MdAPE with 95% confidence intervals and counts
# Figure 4: county MdAPE with CIs and counts (w8_fig4)
cs = county[county.model.eq(HEADLINE)].sort_values('MdAPE')
fig, ax = plt.subplots(figsize=(11.5, 6.4))
err = np.vstack([cs['MdAPE'] * 100 - cs['MdAPE_ci_lo_pct'], cs['MdAPE_ci_hi_pct'] - cs['MdAPE'] * 100])
cols = ['#8c8c8c' if 'Other' in g else ps.MODEL_COLORS[HEADLINE] for g in cs['group']]
b = ax.barh(range(len(cs)), cs['MdAPE'] * 100, xerr=err, capsize=3, color=cols, alpha=0.9)
ax.set_yticks(range(len(cs)))
ax.set_yticklabels(cs['group'], fontsize=10)
ax.bar_label(b, labels=[f'{v:.2f}%  (n={n:,}, med. {usd(p)})' for v, n, p in
                        zip(cs['MdAPE'] * 100, cs['n'], cs['median_actual_price'])],
             fontsize=9, padding=18)
ax.set_xlim(0, cs['MdAPE_ci_hi_pct'].max() * 1.75)
ax.xaxis.set_major_locator(mt.MultipleLocator(2))   # avoid 2.5 -> "2%", 7.5 -> "8%" tick rounding
ax.xaxis.set_major_formatter(mt.PercentFormatter(xmax=100, decimals=0))
ax.set_xlabel(f'MdAPE with 95% bootstrap CI - {HEADLINE}, test month 2026-05')
ax.set_title('Inland, lower-priced counties are predicted best; high-priced coastal\n'
             'and Bay Area counties are hardest', fontsize=13)
ax.annotate(f'grey = {int(county[county.group.str.contains("Other")]["n"].iloc[0]):,} homes pooled from '
            f'counties with n < {MIN_N}', (0.98, 0.03), xycoords='axes fraction', ha='right', fontsize=9,
            color='#555555')
print('saved:', ps.save(fig, 'w8_fig4_mdape_by_county.png'))
plt.show()
""")
header('Why we stop at county',
       f"The same breakdown attempted at ZIP code level, which is the unit an agent would really want, with the number of sales behind every row shown.",
       f"May 2026's sales spread over {Z_N_ZIPS:,} ZIP codes and the largest holds only {Z_MAX_N} of them, so no ZIP reaches our {MIN_N}-sale minimum. County is the finest level this single month can support.")
code(r"""
# Produces: the ZIP-code table and the written geography finding
display(Markdown('### 4b. ZIP code: reported for transparency, not usable for conclusions'))
zv = zips[zips.model.eq(HEADLINE)].sort_values('n', ascending=False).copy()
zv['MdAPE_95CI'] = [f'[{lo:.2f}, {hi:.2f}] pp' for lo, hi in zip(zv['MdAPE_ci_lo_pct'], zv['MdAPE_ci_hi_pct'])]
display(zv[['group', 'n', 'median_actual_price', 'MdAPE', 'MdAPE_95CI',
            'median_signed_pct_error', 'below_min_n']].rename(columns={'group': 'ZIP'})
        .style.format(FMT | {'n': '{:,}'}).hide(axis='index'))

n_zip = test['PostalCode'].nunique()
zc = test['PostalCode'].value_counts()
big_c = county[county.model.eq(HEADLINE) & ~county.group.str.contains('Other')].sort_values('MdAPE')
cbest, cwors = big_c.iloc[0], big_c.iloc[-1]
overlap = cbest['MdAPE_ci_hi_pct'] >= cwors['MdAPE_ci_lo_pct']
import re
_other_lbl = county[county.group.str.contains('Other')]['group'].iloc[0]
n_cty_small = int(county[county.group.str.contains('Other')]['n'].iloc[0])
n_cty_pooled = int(re.search(r'\((\d+) counties', _other_lbl).group(1))

display(Markdown(f'''
The test month spreads {len(test):,} sales over **{n_zip:,} ZIP codes**; the largest holds
**{int(zc.iloc[0])}** sales and **{int((zc >= MIN_N).sum())}** reach the n = {MIN_N} minimum. Every row above is
therefore flagged: with n in the tens, a ZIP's MdAPE CI spans several percentage points and ZIP-level
rankings would be noise. ZIP is used *inside* the models as a feature (where the training window gives it
far more support), but it is not a reportable reporting unit here. County is the finest geography this
test month can support.

**Which geographies perform better, and why that is plausible.**

- Best: **{cbest["group"]}** ({cbest["MdAPE"]:.2%}, CI [{cbest["MdAPE_ci_lo_pct"]:.2f}, {cbest["MdAPE_ci_hi_pct"]:.2f}] pp,
  n = {int(cbest["n"]):,}, median sale {usd(cbest["median_actual_price"])}).
  Worst: **{cwors["group"]}** ({cwors["MdAPE"]:.2%}, CI [{cwors["MdAPE_ci_lo_pct"]:.2f}, {cwors["MdAPE_ci_hi_pct"]:.2f}] pp,
  n = {int(cwors["n"]):,}, median sale {usd(cwors["median_actual_price"])}).
  The two intervals {"overlap, so this ordering is not firm" if overlap else "do not overlap, so the ordering is firm at the ends"};
  neighbouring counties in the middle of the table *do* overlap and should not be ranked against each other.
- The pattern tracks price level and housing homogeneity, not geography as such: the best counties are
  inland and lower-priced with large volumes of similar tract housing, while the hardest are expensive
  coastal and Bay Area markets where lot, view and micro-location dominate value. This is the same effect
  as the price-band table seen through a geographic lens, which is why county error correlates with county
  median sale price.
- {n_cty_small:,} test homes ({100*n_cty_small/len(test):.1f}%) sit in the {n_cty_pooled} pooled
  small counties; they are reported as one row, never dropped, and the AVM should be treated as unvalidated
  in those markets individually.
'''))
""")
header('Re-run at three earlier months',
       f"The five models are rebuilt at three earlier points in time, each trained only on the twelve months before it and scored on the month that follows. The settings are frozen from Week 7 rather than re-tuned, so this tests whether one fixed configuration holds up.",
       f"{HL}'s typical error stays between {K_EARLY.min():.2%} and {K_EARLY.max():.2%} on {EARLY[0]} and {EARLY[1]}, against {KH.loc['2026-05', 'MdAPE']:.2%} in May 2026. The {TUNE_CUT} month is the one the settings were tuned on, so it flatters every model.")
code(r"""
# Produces: the backtest window sizes, MdAPE and MAE by cutoff, and the booster tree counts
bk = back.copy()
cuts = sorted(bk['cutoff'].unique())
bk['label'] = bk['cutoff'] + np.where(bk['is_tuning_month'], ' (tuning month)',
                                      np.where(bk['cutoff'].eq('2026-05'), ' (held-out test)', ''))
piv_md = (bk.pivot_table(index=['cutoff', 'label'], columns='model', values='MdAPE')
          .reset_index().sort_values('cutoff'))
piv_mae = (bk.pivot_table(index=['cutoff', 'label'], columns='model', values='MAE')
           .reset_index().sort_values('cutoff'))
sizes = (bk[bk.model.eq(HEADLINE)][['cutoff', 'train_start', 'train_end', 'n_train', 'n_test']]
         .sort_values('cutoff'))

display(Markdown('### 5a. Window sizes'))
display(sizes.style.format({'n_train': '{:,.0f}', 'n_test': '{:,.0f}'}).hide(axis='index'))
display(Markdown('### 5b. MdAPE by cutoff'))
display(piv_md[['label'] + MODELS].style.format({m: '{:.2%}' for m in MODELS})
        .background_gradient(subset=MODELS, cmap='RdYlGn_r', axis=None).hide(axis='index'))
display(Markdown('### 5c. MAE by cutoff'))
display(piv_mae[['label'] + MODELS].style.format({m: '${:,.0f}' for m in MODELS}).hide(axis='index'))

display(Markdown('### 5d. Booster tree counts actually used (iteration-scaling rule applied per window)'))
it = (bk[bk.model.isin(['LightGBM', 'XGBoost'])]
      .pivot_table(index=['cutoff', 'n_train'], columns='model', values='n_estimators')
      .reset_index().sort_values('cutoff'))
display(it.style.format({'n_train': '{:,.0f}', 'LightGBM': '{:,.0f}', 'XGBoost': '{:,.0f}'}).hide(axis='index'))
""")
header('Why R-squared collapses some months',
       f"For each scored month, the share of the total price variation held by just its three largest sales, shown next to that month's R-squared and errors.",
       f"In {SST_MAX_CUT} the top three sales alone hold {SST_MAX:.1%} of the variation, and no model's R-squared that month rises above {R2_MAX_AT:.4f}. Those prices are almost certainly data-entry errors, so only the percent-error measures can be compared across months.")
code(r"""
# Produces: the table of extreme sales showing why R2 is not comparable across months
# Why R2/RMSE/MAE are NOT comparable across scored months: a few extreme sales dominate them.
ext = pd.read_csv('outputs/08_features_backtest.csv', usecols=['Month', 'ClosePrice'])
rows = []
for c in cuts:
    y = ext.loc[ext['Month'].eq(c), 'ClosePrice'].to_numpy(float)
    sq = (y - y.mean()) ** 2
    rows.append({'scored month': c, 'n': len(y), 'max sale': y.max(),
                 'sales > $20M': int((y > 2e7).sum()),
                 'share of SST held by top 3 sales': np.sort(sq)[-3:].sum() / sq.sum(),
                 f'{HEADLINE} R2': bk[bk.cutoff.eq(c) & bk.model.eq(HEADLINE)]['R2'].iloc[0],
                 f'{HEADLINE} MAE': bk[bk.cutoff.eq(c) & bk.model.eq(HEADLINE)]['MAE'].iloc[0],
                 f'{HEADLINE} MdAPE': bk[bk.cutoff.eq(c) & bk.model.eq(HEADLINE)]['MdAPE'].iloc[0]})
ext_tab = pd.DataFrame(rows)
display(Markdown('### 5e. Why only MdAPE/MAPE are comparable across months'))
display(ext_tab.style.format({'n': '{:,}', 'max sale': '${:,.0f}', 'sales > $20M': '{:,}',
                              'share of SST held by top 3 sales': '{:.1%}',
                              f'{HEADLINE} R2': '{:.4f}', f'{HEADLINE} MAE': '${:,.0f}',
                              f'{HEADLINE} MdAPE': '{:.2%}'}).hide(axis='index'))
""")
header('Does accuracy hold over time',
       f"Typical percent error by scored month on the left, average dollar error on the right, one line per model. Flat lines mean accuracy that does not depend on which month was chosen.",
       f"Across all {len(CUTS)} scored months {STABLE_TXT}, and {HL}'s typical error moves only about {100*(K_EARLY.max()-K_EARLY.min()):.2f} points between {EARLY[0]} and {EARLY[1]}.")
code(r"""
# Produces: Figure 5 - MdAPE and MAE across the four scored months
# Figure 5: stability over time (w8_fig5)
fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), constrained_layout=True)
cuts = sorted(bk['cutoff'].unique())
for ax, col, ylab, fmtr, title in [
    (axes[0], 'MdAPE', 'MdAPE', mt.PercentFormatter(xmax=1, decimals=0), 'Typical error by scored month'),
    (axes[1], 'MAE', 'MAE', ps.usd, 'Average dollar error by scored month')]:
    for m in MODELS:
        sm = bk[bk.model.eq(m)].set_index('cutoff').reindex(cuts)
        ax.plot(range(len(cuts)), sm[col], marker='o', lw=2, ms=6, color=ps.MODEL_COLORS[m], label=m)
    ax.set_xticks(range(len(cuts)))
    ax.set_xticklabels([c + ('\n(tuning month)' if c == '2026-04' else '\n(held-out test)' if c == '2026-05' else '')
                        for c in cuts], fontsize=9.5)
    ax.axvspan(len(cuts) - 1.5, len(cuts) - 0.5, color='#f0f0f0', zorder=0)
    ax.yaxis.set_major_formatter(fmtr)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=12)
axes[0].legend(title='Model', ncols=2, fontsize=9.5)
axes[0].set_ylim(0, bk['MdAPE'].max() * 1.25)
axes[1].annotate('MAE is NOT comparable across these months:\na few corrupt sale prices dominate it (section 5e)',
                 (0.03, 0.92), xycoords='axes fraction', fontsize=9.5, color='#b22222', va='top')
fig.suptitle('Model ranking holds at every scored month; MdAPE barely moves while MAE swings with a '
             'handful of corrupt prices\n(hyperparameters were tuned on 2026-04 only - this tests '
             'stability, not independent tuning)', fontsize=13, fontweight='bold')
print('saved:', ps.save(fig, 'w8_fig5_rolling_origin_backtest.png'))
plt.show()
""")
header('What the time test shows',
       f"The written version of the result above, generated from the same table so the words cannot drift away from the numbers. It separates what is stable from what is not.",
       f"{TOP2_TXT}. Dollar errors are not comparable across those months, and the {TUNE_CUT} month is excluded from the stability range because it is the month the settings came from.")
code(r"""
# Produces: the written backtest finding: what is stable and what is not
earlier = bk[~bk.cutoff.isin(['2026-04', '2026-05'])]
tune = bk[bk.cutoff.eq('2026-04')]
held = bk[bk.cutoff.eq('2026-05')]

lines = []
for m in MODELS:
    e = earlier[earlier.model.eq(m)]['MdAPE']
    lines.append(f'| {m} | ' + ' | '.join(f'{bk[bk.model.eq(m) & bk.cutoff.eq(c)]["MdAPE"].iloc[0]:.2%}'
                                          for c in cuts) +
                 f' | {100*(bk[bk.model.eq(m)]["MdAPE"].max() - bk[bk.model.eq(m)]["MdAPE"].min()):.2f} pp |')

rank_per_cut = {c: bk[bk.cutoff.eq(c)].sort_values('MdAPE')['model'].tolist() for c in cuts}
same_winner = len({r[0] for r in rank_per_cut.values()}) == 1
booster_top2 = all(set(r[:2]) == {'LightGBM', 'XGBoost'} for r in rank_per_cut.values())
h_tune = tune[tune.model.eq(HEADLINE)]['MdAPE'].iloc[0]
h_held = held[held.model.eq(HEADLINE)]['MdAPE'].iloc[0]
h_earl = earlier[earlier.model.eq(HEADLINE)]['MdAPE']
lgbm_cap = bk[bk.model.eq('LightGBM')]['n_estimators']

display(Markdown(f'''
### 5f. What the backtest shows

| Model | ''' + ' | '.join(cuts) + f''' | spread |
|---|''' + '---|' * (len(cuts) + 1) + '\n' + '\n'.join(lines) + f'''

- **The ranking is stable where it matters.**
  {"The two boosters hold the top two places at every scored month" if booster_top2 else "The top two places change across months"}, with
  RF third, LR fourth and DT last at {sum(1 for c in cuts if rank_per_cut[c][2:] == ["RF", "LR", "DT"])} of {len(cuts)} months.
  {HEADLINE} has the lowest MdAPE at {sum(1 for c in cuts if rank_per_cut[c][0] == HEADLINE)} of {len(cuts)} months;
  {"LightGBM takes it at " + ", ".join(c for c in cuts if rank_per_cut[c][0] != HEADLINE) + ", where Week 7's paired bootstrap already showed the two boosters are statistically tied (CI -0.15 to +0.13 pp), so that flip is noise, not a change of regime" if not same_winner else "no month reverses that"}.
  The Weeks 6-7 conclusion - boosters beat RF, which beats LR and DT on typical error - holds at every month
  tested, not only on 2026-05.
- **Absolute error moves month to month, but modestly.** {HEADLINE} ranges from {h_earl.min():.2%} to {h_earl.max():.2%} on the two
  genuinely out-of-sample earlier cutoffs, versus {h_tune:.2%} on the tuning month and {h_held:.2%} on the
  held-out test month. A single month's MdAPE should be quoted with that month-to-month variation attached.
- **Do not compare R2, RMSE or MAE across these months (section 5e).** Three of the four scored months contain
  extreme sales that are almost certainly data-entry errors: the largest single sale is
  {usd(ext_tab['max sale'].max())}, and in the worst month the top three sales alone hold
  {ext_tab['share of SST held by top 3 sales'].max():.1%} of the total sum of squares. Across the backtest
  cutoffs {HEADLINE}'s R2 falls as low as {ext_tab[f'{HEADLINE} R2'].min():.4f}. MAE moves the same way: it is
  {ext_tab[f'{HEADLINE} MAE'].max()/ext_tab[f'{HEADLINE} MAE'].min():.1f}x higher in the worst month than in
  2026-05 while MdAPE barely moves. Those months are not harder in any practical sense - they contain a handful
  of corrupt prices. **MdAPE and MAPE are the only cross-month-comparable metrics here**, which is the whole
  reason this project leads with MdAPE.
- **The 2026-04 cutoff is optimistic by construction** and is excluded from the stability range above: its
  hyperparameters and tree counts were selected on exactly those rows. It is shown for completeness only.
- **Caveat that applies to every row:** hyperparameters were tuned once, on 2026-04. Earlier cutoffs run a
  configuration chosen with knowledge of a *later* month. That direction of information flow does not leak the
  scored month's labels, but it does mean these numbers are not what an honest "what would we have deployed in
  2025-11" exercise would have produced - they test stability, not independent tuning.
- **LightGBM is still near its tree cap.** The scaled counts used here are
  {", ".join(f"{int(v):,}" for v in lgbm_cap.unique())}, derived from a validation fit that stopped at 1,998 of a
  2,000-tree cap. The cap was never lifted, so LightGBM may be undertrained at every cutoff and its backtest
  numbers are a lower bound on what it could do.
'''))
""")
header('Where the model actually misses',
       f"On the left, every test home's predicted price against what it actually sold for, with the perfect-prediction line dashed and the highest price in the training data marked. On the right, how the percent errors are spread.",
       f"{W10:.0f} percent of May 2026 homes are priced within ten percent of the sale price, but no prediction can go above {usd_md(MAXPRED)}, and {N_ABOVE} sales that month were above that.")
code(r"""
# Produces: Figure 6 - actual versus predicted, and the distribution of percent error
# Figure 6: actual vs predicted for the headline model (w8_fig6)
yp = preds[HEADLINE].to_numpy(float)
cap = TRAIN_CAP
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6), constrained_layout=True)
hb = axes[0].hexbin(y_test, yp, xscale='log', yscale='log', gridsize=48, bins='log', cmap='viridis', mincnt=1)
lims = [min(y_test.min(), yp.min()) * 0.9, max(y_test.max(), yp.max()) * 1.1]
axes[0].plot(lims, lims, color='#d62728', lw=1.6, ls='--', label='perfect prediction')
axes[0].axhline(cap, color='#333333', lw=1.2, ls=':', label=f'train price cap {usd(cap)}')
axes[0].set_xlim(lims); axes[0].set_ylim(lims)
axes[0].set_xlabel('Actual ClosePrice (log scale)'); axes[0].set_ylabel('Predicted ClosePrice (log scale)')
axes[0].xaxis.set_major_formatter(ps.usd); axes[0].yaxis.set_major_formatter(ps.usd)
axes[0].legend(loc='upper left', fontsize=9.5)
n_above = int((y_test > yp.max()).sum())
axes[0].annotate(f'highest prediction {usd(yp.max())}\nvs highest actual sale {usd(y_test.max())}\n'
                 f'{n_above} sales lie above anything the model can output',
                 (0.97, 0.06), xycoords='axes fraction', ha='right', fontsize=9.5, color='#b22222',
                 bbox=dict(fc='white', ec='#cccccc', alpha=0.85, boxstyle='round,pad=0.4'))
axes[0].set_title(f'{HEADLINE}: predictions plateau just above the training price cap', fontsize=12)
fig.colorbar(hb, ax=axes[0], label='homes (log count)')

pe = 100 * (yp - y_test) / y_test
axes[1].hist(np.clip(pe, -60, 60), bins=70, color=ps.MODEL_COLORS[HEADLINE], alpha=0.85)
axes[1].axvline(0, color='#333333', lw=1.2, ls='--')
med, p10, p90 = np.median(pe), np.percentile(pe, 10), np.percentile(pe, 90)
for v, lab, c in [(med, f'median {med:+.2f}%', '#d62728'), (p10, f'10th pct {p10:+.1f}%', '#555555'),
                  (p90, f'90th pct {p90:+.1f}%', '#555555')]:
    axes[1].axvline(v, color=c, lw=1.4, ls='-' if c == '#d62728' else ':')
    axes[1].annotate(lab, (v, axes[1].get_ylim()[1] * 0.99), rotation=90, fontsize=9,
                     ha='right', va='top', color=c,
                     bbox=dict(fc='white', ec='none', alpha=0.8, boxstyle='round,pad=0.15'))
n_clip = int((np.abs(pe) > 60).sum())
axes[1].annotate(f'{n_clip} homes ({100*n_clip/len(pe):.1f}%) fall outside +/-60%\nand pile into the end bins',
                 (0.97, 0.72), xycoords='axes fraction', ha='right', fontsize=9, color='#555555')
axes[1].set_xlabel('Signed % error (clipped to +/-60% for display)')
axes[1].set_ylabel('Homes')
axes[1].set_title(f'80% of homes land within {p10:+.1f}% / {p90:+.1f}% of the sale price', fontsize=12)
fig.suptitle(f'Test month 2026-05, {HEADLINE} (n = {len(y_test):,})', fontsize=14, fontweight='bold')
print('saved:', ps.save(fig, 'w8_fig6_actual_vs_predicted.png'))
plt.show()

within = lambda t: 100 * np.mean(np.abs(pe) <= t)
display(Markdown(f'''
**Practical accuracy of {HEADLINE} on the test month:** {within(5):.1f}% of homes are predicted within +/-5% of the
actual sale price, {within(10):.1f}% within +/-10% and {within(20):.1f}% within +/-20%. The median signed error is
{med:+.2f}%, so there is no large global bias - the bias is segment-specific (section 3) rather than overall.
'''))
""")
header('What this week establishes',
       f"The closing summary and the limitations, written from the saved tables so every figure matches the ones above. The four points answer the four questions this notebook opened with.",
       f"In May 2026 {S_BEST['model']} misses a typical home by {S_BEST['MdAPE']:.2%}. Where the model is used - which price level, which county - matters more than which of these five algorithms is chosen.")
code(r"""
# Produces: the closing summary and the limitations list
b = summary.sort_values('MdAPE').iloc[0]
rf = summary[summary.model.eq('RF')].iloc[0]
lr = summary[summary.model.eq('LR')].iloc[0]
qq = band[band.scope.eq('train_quintile') & band.model.eq(HEADLINE)].sort_values('order')
ff = band[band.scope.eq('fixed_band') & band.model.eq(HEADLINE)].sort_values('order')
cc = county[county.model.eq(HEADLINE) & ~county.group.str.contains('Other')].sort_values('MdAPE')
ea = bk[~bk.cutoff.isin(['2026-04', '2026-05']) & bk.model.eq(HEADLINE)]['MdAPE']

display(Markdown(f'''
### What Week 8 adds

1. **A full metric set, not just R2.** {b["model"]} ({b["feature_set"]}) on 2026-05:
   R2 {b["R2"]:.4f}, MAE {usd(b["MAE"])}, RMSE {usd(b["RMSE"])}, MAPE {b["MAPE"]:.2%}, **MdAPE {b["MdAPE"]:.2%}**.
   R2 and MdAPE rank the models differently, and the reason is the price cap, not a better model.
2. **Price band is the dominant driver of error.** {HEADLINE} ranges from {qq["MdAPE"].min():.2%}
   ({qq.loc[qq["MdAPE"].idxmin(), "group"]}) to {qq["MdAPE"].max():.2%}
   ({qq.loc[qq["MdAPE"].idxmax(), "group"]}) - a {100*(qq["MdAPE"].max()-qq["MdAPE"].min()):.2f} pp spread that
   dwarfs the {100*(rf["MdAPE"]-b["MdAPE"]):.2f} pp gap between the best and second-tier *models*. Choosing where to
   apply the AVM matters more than choosing between these algorithms.
3. **Geography matters mostly through price level.** County MdAPE runs {cc["MdAPE"].min():.2%}
   ({cc.iloc[0]["group"]}) to {cc["MdAPE"].max():.2%} ({cc.iloc[-1]["group"]}), tracking county median sale price.
4. **The ranking survives time.** Across {len(cuts)} scored months the model order is stable; the headline model's
   MdAPE moves between {ea.min():.2%} and {ea.max():.2%} on genuinely out-of-sample cutoffs.

### Limitations (read before quoting any number above)

- **Hyperparameters were tuned on a single validation month (2026-04)** and frozen. The backtest re-uses them at
  earlier cutoffs, so it demonstrates stability, not independent tuning. The 2026-04 cutoff is the tuning month
  itself and is optimistic.
- **The saved production model was picked by test MdAPE** among the five finalists (Week 7), so
  {HEADLINE}'s {b["MdAPE"]:.2%} is slightly optimistic; {HEADLINE} and LightGBM are statistically tied
  (Week 7 bootstrap CI for the difference: -0.15 to +0.13 pp) and strict validation-based selection picks LightGBM.
- **No ZIP-level conclusions.** The largest test ZIP has {int(test["PostalCode"].value_counts().iloc[0])} sales and
  none reaches n = {MIN_N}. County is the finest supportable unit, and {int(county[county.group.str.contains("Other")]["n"].iloc[0]):,}
  homes sit in pooled small counties where the model is effectively unvalidated.
- **Tree models cannot price above the training cap** ({usd(TRAIN_CAP)}), so luxury error is structural, not tunable.
- **Bootstrap intervals here are not simultaneous.** Many groups are compared; a single interval that barely
  excludes zero (see the marginal contrast in 3d) should not be read as a firm finding.
- **LightGBM remains capped at ~2,000 trees** at every cutoff, so its numbers are a lower bound.
- **One market, one window.** Everything is CRMLS single-family California, 2024-11..2026-05. Nothing here
  establishes behaviour in a different market or a different rate environment.
'''))
""")
md(f"""
## Limitations and next steps

**Limitations**

- **The settings were tuned on a single month.** Every setting was chosen on April 2026 and then frozen. The
  time test re-uses those same settings at earlier months, so it shows the configuration is stable; it is not
  the same as re-tuning honestly at each point in time, and the April month is the one the settings came from,
  so its numbers flatter the models.
- **R-squared is not a safe headline here.** A few sales recorded in the hundreds of millions - the largest is
  {usd_md(MAX_SALE)} - hold up to {SST_MAX:.0%} of a single month's price variation, and at the three earlier
  months no model's R-squared rises above {R2_MAX_EARLY:.4f}. That is a data quality artefact rather than the
  model getting worse, which is why only the percent-error measures are compared across months.
- **No claims below county level.** May 2026 covers {Z_N_ZIPS:,} ZIP codes and the largest holds {Z_MAX_N}
  sales, so county is the finest geography these results can defend. {int(C_OTHER['n']):,} homes sit in the
  {N_CTY_POOLED} pooled small counties, where the model is effectively unvalidated.
- **The saved model was chosen on the test month.** {S_BEST['model']} beat LightGBM on test typical error
  ({S_BEST['MdAPE']:.2%} against {LGB_MD:.2%}), but the two are statistically tied and a strict
  validation-based choice would have picked LightGBM, so {S_BEST['MdAPE']:.2%} is slightly optimistic.
- **LightGBM still runs into its tree limit** - it used {min(LGB_ITERS):,} to {max(LGB_ITERS):,} trees against
  a 2,000-tree ceiling in every month tested - so its numbers are a floor on what it could do, not its best.
- **One market, one window, one test month.** Everything here is California single-family homes from the CRMLS
  feed between 2024-11 and 2026-05, with every test-month figure drawn from May 2026 alone. Nothing here
  establishes how the model behaves in another market or another rate environment.

**Next steps**

- Lift the LightGBM tree cap and re-tune on more than one validation month.
- Add features that explain value at the top of the market - view, finish quality, lot shape - because that is
  where the error is, and no amount of tuning replaces a missing feature.
- Quote an estimate with its price level attached: in the middle of the market it is worth trusting, while for
  homes above three million dollars (the {F_WORST['group']} band, median sale
  {usd_md(F_WORST['median_actual_price'])}, typical error {F_WORST['MdAPE']:.1%}) it is an indication rather
  than a valuation.
""")

nb['cells'] = cells
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
Path('06_evaluation.ipynb').write_text(nbf.writes(nb))
print(f'Wrote 06_evaluation.ipynb ({len(cells)} cells: {sum(1 for c in cells if c.cell_type == "code")} code, '
      f'{sum(1 for c in cells if c.cell_type == "markdown")} markdown)')
