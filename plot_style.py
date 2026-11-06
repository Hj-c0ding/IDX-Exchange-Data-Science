"""Shared plotting style for all Week 6+ figures (see SKILL.md section 6)."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import seaborn as sns

FIG_DIR = Path('outputs/figures')
MODEL_COLORS = {
    'LR': '#1f77b4',
    'DT': '#ff7f0e',
    'RF': '#2ca02c',
    'LightGBM': '#9467bd',
    'XGBoost': '#d62728',
}   # same model = same color everywhere
SET_COLORS = {'F0': '#8c8c8c', 'F1': '#4c9be8', 'F2': '#e8a33d', 'F3': '#2a9d5c'}
usd = mt.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}k')
pct = mt.PercentFormatter(xmax=1, decimals=0)


def apply_style():
    plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 120, 'savefig.dpi': 200,
                         'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .25, 'axes.titlesize': 14,
                         'axes.titleweight': 'bold', 'axes.labelsize': 12, 'legend.frameon': False,
                         'font.size': 11})
    sns.set_palette('colorblind')


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    return path
