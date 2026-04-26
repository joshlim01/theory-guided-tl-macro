"""
oos_evaluation.py
=================
Out-of-sample evaluation of the phase 2 fine-tuned network on 2023Q1-2025Q4.

Skips 2020-2022 (COVID structural break outside training distribution).

Steps:
  1. Reconstruct US state variables (k_hat, c_hat, z_hat, xi_hat) from FRED
     data using the same pipeline as training (HP filter, policy function residual)
  2. Build phase 2 financial features for the OOS period
  3. Roll the network forward quarter by quarter
  4. Compare predictions vs actuals for k_hat, z_hat, xi_hat
  5. Compare against AR(1) baseline
  6. Plot results

Files needed (upload to /content/ in Colab):
  Financial features (FRED):
    DBAA_2019-2026.csv, DAAA_2019-2026.csv, FEDFUNDS_2019-2026.csv,
    VIXCLS_2019-2026.csv, GS10_2019-2026.csv, sp500_2019_2026.csv
  Macro (FRED):
    GDPC1_2019-2026.csv, PCECC96_2019-2026.csv,
    HOANBS_2019-2026.csv, GPDI_2019-2026.csv
  Models:
    rbc_pretrained_15.pt, rbc_finetuned_final.pt,
    rbc_finetuned_phase2_final.pt
  Training data (for normalisation stats and AR1 fit):
    finetune_dataset.csv, phase2_features.csv

Usage:
  python oos_evaluation.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import root
from statsmodels.tsa.filters.hp_filter import hpfilter
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR = '/content/'
OUT_DIR  = '/content/oos_outputs/'
os.makedirs(OUT_DIR, exist_ok=True)

PRETRAINED_PATH = DATA_DIR + 'rbc_pretrained_15.pt'
PHASE1_PATH     = DATA_DIR + 'rbc_finetuned_final.pt'
PHASE2_PATH     = DATA_DIR + 'rbc_finetuned_phase2_final (1).pt'
PHASE1_CSV      = DATA_DIR + 'finetune_dataset.csv'
PHASE2_CSV      = DATA_DIR + 'phase2_features (1).csv'

# OOS window — skip COVID
OOS_START = '2023Q1'
OOS_END   = '2025Q4'

# RBC calibration (US from PWT, same as training)
ALPHA  = 0.383
BETA   = 0.99
DELTA  = 0.0355
PHI    = 1.0
RHO_Z  = 0.90
RHO_XI = 0.70
CTOY   = 0.661

# ── NETWORK ───────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.block(x))

class RBCNet(nn.Module):
    def __init__(self, input_dim=19, output_dim=3,
                 hidden_dim=512, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.output_proj(self.res_blocks(self.input_proj(x)))

def load_model(path):
    m = RBCNet(input_dim=19, hidden_dim=512)
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model_state' in sd:
        sd = sd['model_state']
    m.load_state_dict(sd)
    m.eval()
    return m

# ── RBC SOLVER ────────────────────────────────────────────────────────────────

def solve_rbc_two_shock_model(alpha, beta, delta, phi, rho_z, rho_xi, ctoy_rat):
    i_over_y = 1.0 - ctoy_rat
    c_over_i = ctoy_rat / i_over_y
    y_over_i = 1.0 / i_over_y
    r        = 1.0 / beta - 1.0 + delta
    g_c  = -delta*y_over_i*(1-alpha)/(phi+alpha) - delta*c_over_i
    g_k  = (1-delta) + delta*y_over_i*alpha*(phi+1)/(phi+alpha)
    g_z  = delta*y_over_i*(phi+1)/(phi+alpha)
    g_xi = delta
    M = 1 + beta*r*(1-alpha)/(phi+alpha)
    N = beta*r*phi*(1-alpha)/(phi+alpha)
    H = beta*r*(phi+1)/(phi+alpha)

    def residuals(v):
        a_k, a_z, a_xi = v
        Gk  = g_c*a_k + g_k
        Gz  = g_c*a_z + g_z
        Gxi = g_c*a_xi + g_xi
        return np.array([
            a_k  - (M*a_k + N)*Gk,
            a_z  - (M*(a_k*Gz  + a_z *rho_z)  + N*Gz  - H*rho_z),
            a_xi - (M*(a_k*Gxi + a_xi*rho_xi) + N*Gxi),
        ])

    guesses = [[0.1,0.1,0.1],[0.1,0.1,-0.1],[0.5,0.1,0.1],[1.0,0.1,0.1]]
    candidates = []
    for g in guesses:
        sol = root(residuals, np.array(g, dtype=float))
        if not sol.success: continue
        a_k, a_z, a_xi = sol.x
        Gk  = g_c*a_k+g_k; Gz = g_c*a_z+g_z; Gxi = g_c*a_xi+g_xi
        P = np.array([
            [a_k*g_c, a_k*g_k, a_k*g_z+a_z*rho_z,  a_k*g_xi+a_xi*rho_xi],
            [g_c, g_k, g_z, g_xi],
            [0., 0., rho_z, 0.],
            [0., 0., 0., rho_xi],
        ])
        eigvals = np.linalg.eigvals(P)
        candidates.append(dict(
            a_k=a_k, a_z=a_z, a_xi=a_xi, P=P,
            max_abs_eig=np.max(np.abs(eigvals)),
            res_norm=np.linalg.norm(residuals([a_k,a_z,a_xi]))
        ))
    stable = [c for c in candidates if c['max_abs_eig'] < 1-1e-8]
    if not stable: raise RuntimeError("No stable RBC solution found")
    return min(stable, key=lambda x: (x['res_norm'], x['max_abs_eig']))

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_fred(filepath, col, start=None, end=None):
    df = pd.read_csv(filepath, parse_dates=['observation_date'])
    df = df.rename(columns={col: 'value', 'observation_date': 'date'})
    df['value']  = pd.to_numeric(df['value'], errors='coerce')
    df['period'] = df['date'].dt.to_period('Q')
    s = df.set_index('period')['value']
    if start: s = s[s.index >= pd.Period(start, 'Q')]
    if end:   s = s[s.index <= pd.Period(end,   'Q')]
    return s

def hp_cycle(series, lamb=1600):
    clean = series.dropna()
    if len(clean) < 8:
        return pd.Series(np.nan, index=series.index)
    cycle, _ = hpfilter(clean, lamb=lamb)
    return pd.Series(cycle, index=clean.index).reindex(series.index)

# ── STEP 1: LOAD TRAINING DATA FOR NORMALISATION STATS ────────────────────────

print("=" * 60)
print("Loading training data for normalisation stats...")
print("=" * 60)

df_train    = pd.read_csv(PHASE1_CSV)
df_train_us = df_train[df_train['countrycode'] == 'USA'].copy()
df_p2_train = pd.read_csv(PHASE2_CSV)

PHASE1_COLS = ['k_hat','c_hat','z_hat','xi_hat',
               'gdp_growth_level','gdp_growth_vol',
               'ulc_growth_level','ulc_growth_vol',
               'inflation_level','inflation_vol',
               'unemp_level','unemp_vol',
               'spread_level','spread_vol']
PHASE2_COLS = ['baa_aaa','fedfunds','vix_log','term_spread','sp_logret']
TARGET_COLS = ['k_hat','z_hat','xi_hat']

# Compute normalisation stats from training data
norm_stats = {}
df_merged_train = (df_train_us[['period'] + PHASE1_COLS]
                   .merge(df_p2_train[['period'] + PHASE2_COLS],
                          on='period', how='inner'))

for col in PHASE1_COLS + PHASE2_COLS:
    mu  = df_merged_train[col].mean()
    sig = df_merged_train[col].std()
    if sig < 1e-10: sig = 1.0
    norm_stats[col] = dict(mean=float(mu), std=float(sig))

print(f"  Normalisation stats computed from {len(df_merged_train)} training quarters")

# ── STEP 2: SOLVE RBC ─────────────────────────────────────────────────────────

print("\nSolving RBC model...")
sol = solve_rbc_two_shock_model(ALPHA, BETA, DELTA, PHI, RHO_Z, RHO_XI, CTOY)
a_k, a_z, a_xi = sol['a_k'], sol['a_z'], sol['a_xi']
print(f"  a_k={a_k:.3f}  a_z={a_z:.3f}  a_xi={a_xi:.3f}")

# ── STEP 3: RECONSTRUCT OOS STATE VARIABLES ───────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 — Reconstructing OOS state variables from FRED")
print("=" * 60)

# Load macro series — use full history for HP filter, then trim to OOS
# HP filter needs long history to work well at the end of the sample
gdp  = load_fred(DATA_DIR + 'GDPC1_2019-2026.csv',    'GDPC1')
cons = load_fred(DATA_DIR + 'PCECC96_2019-2026.csv',  'PCECC96')
hrs  = load_fred(DATA_DIR + 'HOANBS_2019-2026.csv',   'HOANBS')
inv  = load_fred(DATA_DIR + 'GPDI_2019-2026.csv',     'GPDI')

# Append to training US data for better HP filter at sample end
# Use the training GDP/cons series to extend backwards
# For a cleaner HP filter we use 2010Q1 onwards
# Pull the in-sample US series from training CSV for GDP proxy
# (we use OECD GDP growth from training to reconstruct level)

# Build capital via perpetual inventory: K_{t+1} = (1-delta)*K_t + I_t
# Initialise K at 2019Q1 using training capital series endpoint
df_us_sorted = df_train_us.sort_values('period')
# k_hat is HP-filtered log capital - we need the raw trend
# Use investment to accumulate capital from 2019Q1
inv_vals = inv.values
K = np.zeros(len(inv_vals))
# Initialise: set K[0] such that it's consistent with training
# Use GDP/capital ratio from calibration
K[0] = gdp.iloc[0] / (ALPHA / (DELTA + 0.005))  # rough steady state K/Y * Y
for t in range(1, len(inv_vals)):
    if not np.isnan(inv_vals[t]):
        K[t] = (1 - DELTA) * K[t-1] + inv_vals[t]
    else:
        K[t] = (1 - DELTA) * K[t-1]

K_series = pd.Series(K, index=inv.index)

# Log series
log_k    = np.log(K_series)
log_c    = np.log(cons)
log_gdp  = np.log(gdp)

# TFP: Solow residual from production function
# z = log(Y) - alpha*log(K) - (1-alpha)*log(L)
log_L    = np.log(hrs)
log_tfp  = log_gdp - ALPHA * log_k - (1 - ALPHA) * log_L

# HP filter — use all available data, extract cycle
k_hat_raw  = hp_cycle(log_k)
c_hat_raw  = hp_cycle(log_c)
z_hat_raw  = hp_cycle(log_tfp)

# xi: policy function residual
xi_hat_raw = pd.Series(np.nan, index=k_hat_raw.index)
valid = k_hat_raw.notna() & c_hat_raw.notna() & z_hat_raw.notna()
xi_hat_raw[valid] = (
    (c_hat_raw[valid] - a_k * k_hat_raw[valid] - a_z * z_hat_raw[valid]) / a_xi
)

# GDP growth for feature
gdp_growth = log_gdp.diff()

print(f"  State variables reconstructed: {valid.sum()} valid quarters")
print(f"  Sample: {k_hat_raw.dropna().index[0]} to {k_hat_raw.dropna().index[-1]}")
print(f"  k_hat range: [{k_hat_raw.dropna().min():.4f}, {k_hat_raw.dropna().max():.4f}]")
print(f"  z_hat range: [{z_hat_raw.dropna().min():.4f}, {z_hat_raw.dropna().max():.4f}]")
print(f"  xi_hat range: [{xi_hat_raw.dropna().min():.4f}, {xi_hat_raw.dropna().max():.4f}]")

# ── STEP 4: BUILD OOS PHASE 2 FEATURES ────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 — Building OOS financial features")
print("=" * 60)

ALL_Q = pd.period_range(start='2019Q1', end='2025Q4', freq='Q')

def load_fred_q(filepath, col):
    df = pd.read_csv(filepath, parse_dates=['observation_date'])
    df['value']  = pd.to_numeric(df[col], errors='coerce')
    df['period'] = df['observation_date'].dt.to_period('Q')
    return df.set_index('period')['value'].reindex(ALL_Q)

gs10     = load_fred_q(DATA_DIR + 'GS10_2019-2026.csv',     'GS10')
vix      = load_fred_q(DATA_DIR + 'VIXCLS_2019-2026.csv',   'VIXCLS')
fedfunds = load_fred_q(DATA_DIR + 'FEDFUNDS_2019-2026.csv', 'FEDFUNDS')
daaa     = load_fred_q(DATA_DIR + 'DAAA_2019-2026.csv',     'DAAA')
dbaa     = load_fred_q(DATA_DIR + 'DBAA_2019-2026.csv',     'DBAA')

# SP500 log returns
sp = pd.read_csv(DATA_DIR + 'sp500_2019_2026.csv', parse_dates=['Date'])
sp['period'] = sp['Date'].dt.to_period('Q')
sp_q = sp.set_index('period')['^GSPC'].reindex(ALL_Q)
sp_logret = np.log(sp_q).diff()

# Construct features
baa_aaa     = dbaa - daaa
fedfunds_d  = fedfunds / 100
vix_log     = np.log(vix)
term_spread = gs10 - fedfunds

# GDP growth proxy for phase 1 features
gdp_growth_q = gdp_growth.reindex(ALL_Q)

# Build full feature dataframe
df_oos = pd.DataFrame({
    'k_hat'      : k_hat_raw.reindex(ALL_Q),
    'c_hat'      : c_hat_raw.reindex(ALL_Q),
    'z_hat'      : z_hat_raw.reindex(ALL_Q),
    'xi_hat'     : xi_hat_raw.reindex(ALL_Q),
    'baa_aaa'    : baa_aaa,
    'fedfunds'   : fedfunds_d,
    'vix_log'    : vix_log,
    'term_spread': term_spread,
    'sp_logret'  : sp_logret,
    'gdp_growth' : gdp_growth_q,
}, index=ALL_Q)

df_oos.index.name = 'period'

# Trim to OOS window
oos_quarters = pd.period_range(start=OOS_START, end=OOS_END, freq='Q')
df_oos_window = df_oos.reindex(oos_quarters)

print(f"  OOS window: {OOS_START} to {OOS_END} ({len(oos_quarters)} quarters)")
print(f"  Missing values:\n{df_oos_window.isnull().sum()}")

# ── STEP 5: NORMALISE USING TRAINING STATS ────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 — Normalising using training statistics")
print("=" * 60)

# For phase 1 cols we use training norm stats
# For phase 2 cols we use training norm stats
# State variables also normalised

# Build a combined normalised dataframe for the OOS window
# We need to map our reconstructed series to the training feature names

# Phase 1 features we can reconstruct: k_hat, c_hat, z_hat, xi_hat, gdp_growth_level
# Others (ulc, inflation, unemp, spread) — use training means as neutral values
# This is conservative: unknown features set to their training mean (= 0 after normalisation)

def normalise(val, col):
    mu  = norm_stats[col]['mean']
    sig = norm_stats[col]['std']
    return (val - mu) / sig

def build_input_vector(t_idx, df_window):
    """Build normalised 19-dim input vector for quarter t_idx in df_window."""
    row = df_window.iloc[t_idx]

    # State vars (positions 0-3)
    k  = normalise(row['k_hat'],  'k_hat')
    c  = normalise(row['c_hat'],  'c_hat')
    z  = normalise(row['z_hat'],  'z_hat')
    xi = normalise(row['xi_hat'], 'xi_hat')

    # Phase 1 features (positions 4-13)
    # GDP growth level and vol: use reconstructed, others set to training mean (0 after norm)
    gdp_l = normalise(row['gdp_growth'], 'gdp_growth_level')
    gdp_v = 0.0   # rolling vol not available, use neutral
    ulc_l = 0.0   # not available OOS, use neutral
    ulc_v = 0.0
    inf_l = 0.0   # not available OOS, use neutral
    inf_v = 0.0
    une_l = 0.0   # not available OOS, use neutral
    une_v = 0.0
    spr_l = normalise(row['term_spread'], 'spread_level')  # use term spread as proxy
    spr_v = 0.0

    # Phase 2 features (positions 14-18)
    baa  = normalise(row['baa_aaa'],     'baa_aaa')
    ffr  = normalise(row['fedfunds'],    'fedfunds')
    vix  = normalise(row['vix_log'],     'vix_log')
    tspr = normalise(row['term_spread'], 'term_spread')
    sp   = normalise(row['sp_logret'],   'sp_logret') if not np.isnan(row['sp_logret']) else 0.0

    vec = np.array([k, c, z, xi,
                    gdp_l, gdp_v, ulc_l, ulc_v, inf_l, inf_v,
                    une_l, une_v, spr_l, spr_v,
                    baa, ffr, vix, tspr, sp], dtype=np.float32)
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

# ── STEP 6: LOAD MODELS ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 — Loading models")
print("=" * 60)

models = {
    'Pretrained': load_model(PRETRAINED_PATH),
    'Phase 1'   : load_model(PHASE1_PATH),
    'Phase 2'   : load_model(PHASE2_PATH),
}
print("  All models loaded.")

# ── STEP 7: OOS FORECASTING ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7 — One-step OOS forecasting")
print("=" * 60)

n_oos = len(df_oos_window) - 1  # pairs t -> t+1

results = {name: {'k_hat': [], 'z_hat': [], 'xi_hat': []} for name in models}
actuals = {'k_hat': [], 'z_hat': [], 'xi_hat': []}
periods_oos = []

for t in range(n_oos):
    x = build_input_vector(t, df_oos_window)
    actual_next = df_oos_window.iloc[t+1]
    periods_oos.append(str(df_oos_window.index[t+1]))

    for tname in TARGET_COLS:
        actuals[tname].append(actual_next[tname])

    for mname, model in models.items():
        with torch.no_grad():
            pred = model(x)[0].numpy()
        # Denormalise predictions
        for i, tname in enumerate(['k_hat', 'z_hat', 'xi_hat']):
            pred_val = pred[i] * norm_stats[tname]['std'] + norm_stats[tname]['mean']
            results[mname][tname].append(pred_val)

# AR(1) baseline: fit on training data, predict OOS
ar1_preds = {}
for tname in TARGET_COLS:
    train_series = df_train_us[tname].values
    rho = np.corrcoef(train_series[:-1], train_series[1:])[0,1]
    mu  = train_series.mean()
    # Roll forward from last training value
    ar1_preds[tname] = []
    last_val = df_train_us[tname].iloc[-1]
    for t in range(n_oos):
        pred_ar1 = mu + rho * (last_val - mu)
        ar1_preds[tname].append(pred_ar1)
        last_val = actuals[tname][t]  # use actual for next step

# ── STEP 8: METRICS ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 8 — OOS Metrics")
print("=" * 60)

TARGET_LABELS = ['k̂ (capital)', 'ẑ (TFP)', 'ξ̂ (inv. shock)']

def compute_metrics(preds, acts):
    p = np.array(preds)
    a = np.array(acts)
    valid = ~(np.isnan(p) | np.isnan(a))
    p, a  = p[valid], a[valid]
    if len(p) == 0:
        return dict(MAE=np.nan, RMSE=np.nan, R2=np.nan, DirAcc=np.nan)
    mae  = np.mean(np.abs(p - a))
    rmse = np.sqrt(np.mean((p - a)**2))
    ss_res = np.sum((p - a)**2)
    ss_tot = np.sum((a - a.mean())**2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    dir_acc = np.mean(np.sign(np.diff(np.concatenate([[a[0]], p]))) ==
                      np.sign(np.diff(np.concatenate([[a[0]], a])))) if len(a) > 1 else np.nan
    return dict(MAE=mae, RMSE=rmse, R2=r2, DirAcc=dir_acc)

for tname, tlabel in zip(TARGET_COLS, TARGET_LABELS):
    print(f"\n  {tlabel}:")
    print(f"  {'Model':12s}  {'MAE':>8s}  {'RMSE':>8s}  {'R²':>8s}  {'DirAcc':>8s}")
    print("  " + "-" * 50)
    ar1_m = compute_metrics(ar1_preds[tname], actuals[tname])
    print(f"  {'AR(1)':12s}  {ar1_m['MAE']:8.5f}  {ar1_m['RMSE']:8.5f}  {ar1_m['R2']:8.3f}  {ar1_m['DirAcc']:8.2f}")
    for mname in ['Pretrained', 'Phase 1', 'Phase 2']:
        m = compute_metrics(results[mname][tname], actuals[tname])
        beat = "✓" if m['MAE'] < ar1_m['MAE'] else " "
        print(f"  {mname:12s}  {m['MAE']:8.5f}  {m['RMSE']:8.5f}  {m['R2']:8.3f}  {m['DirAcc']:8.2f}  {beat}")

# ── STEP 9: PLOTS ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 9 — Plotting")
print("=" * 60)

colors = {'Pretrained': '#4C78C8', 'Phase 1': '#F28E2B', 'Phase 2': '#2CA02C', 'AR(1)': '#9467BD'}
n_periods = len(periods_oos)
x_ticks = range(n_periods)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'Out-of-Sample Forecast vs Actual — {OOS_START} to {OOS_END}',
             fontsize=13, fontweight='bold')

for ax, tname, tlabel in zip(axes, TARGET_COLS, TARGET_LABELS):
    acts = actuals[tname]
    ax.plot(x_ticks, acts, 'k-', linewidth=2.5, label='Actual', zorder=5)
    for mname in ['AR(1)', 'Pretrained', 'Phase 1', 'Phase 2']:
        preds = ar1_preds[tname] if mname == 'AR(1)' else results[mname][tname]
        ax.plot(x_ticks, preds, color=colors[mname],
                linewidth=1.5, alpha=0.85, label=mname,
                linestyle='--' if mname == 'AR(1)' else '-')
    ax.set_title(tlabel)
    ax.set_xlabel('Quarter')
    ax.set_xticks(x_ticks[::2])
    ax.set_xticklabels(periods_oos[::2], rotation=45, fontsize=7)
    ax.axhline(0, color='gray', linewidth=0.7, linestyle=':')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR + 'E_oos_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Plot saved: E_oos_forecast.png")

# MAE bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(f'OOS MAE by Model — {OOS_START} to {OOS_END}',
             fontsize=13, fontweight='bold')

model_names = ['AR(1)', 'Pretrained', 'Phase 1', 'Phase 2']
bar_colors  = [colors[m] for m in model_names]

for ax, tname, tlabel in zip(axes, TARGET_COLS, TARGET_LABELS):
    maes = []
    for mname in model_names:
        preds = ar1_preds[tname] if mname == 'AR(1)' else results[mname][tname]
        m = compute_metrics(preds, actuals[tname])
        maes.append(m['MAE'])
    bars = ax.bar(model_names, maes, color=bar_colors, alpha=0.8, edgecolor='white')
    ax.set_title(tlabel)
    ax.set_ylabel('MAE')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(maes)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR + 'E_oos_mae_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Plot saved: E_oos_mae_bar.png")

print(f"\n✅ OOS evaluation complete.")
print(f"   Window: {OOS_START} to {OOS_END} ({n_oos} quarters)")
print(f"   Outputs saved to: {OUT_DIR}")
print(f"   Files: E_oos_forecast.png  E_oos_mae_bar.png")
