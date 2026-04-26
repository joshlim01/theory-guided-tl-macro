"""
var_benchmark_v3.py
===================
VAR(2) and BVAR benchmarks vs Phase 2 network.
Standardises variables before VAR fitting to handle scale differences.

Files needed in /content/:
  - finetune_dataset.csv
  - phase2_features (1).csv
  - rbc_finetuned_phase2_final (1).pt
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.api import VAR

DATA_DIR = '/content/'
OUT_DIR  = '/content/var_outputs/'
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COLS   = ['k_hat', 'z_hat', 'xi_hat']
TARGET_LABELS = ['k\u0302 (capital)', '\u1e91 (TFP)', '\u03be\u0302 (inv. shock)']
VAR_COLS      = ['k_hat', 'c_hat', 'z_hat', 'xi_hat']
LAGS          = 2

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

# ── BVAR Minnesota ────────────────────────────────────────────────────────────

class BVARMinnesota:
    def __init__(self, lags=2, lambda1=0.2, lambda2=0.5):
        self.lags    = lags
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def fit(self, Y):
        T, k = Y.shape
        p    = self.lags
        sigmas = np.zeros(k)
        for i in range(k):
            y   = Y[:, i]
            rho = np.corrcoef(y[:-1], y[1:])[0, 1]
            sigmas[i] = np.std(y[1:] - rho * y[:-1]) + 1e-8
        self.sigmas_ = sigmas

        X = np.ones((T - p, k * p + 1))
        for lag in range(1, p + 1):
            X[:, (lag-1)*k : lag*k] = Y[p-lag : T-lag, :]
        Z = Y[p:, :]

        n_coef = k * p + 1
        coefs  = np.zeros((n_coef, k))

        for j in range(k):
            prior_var  = np.zeros(n_coef)
            prior_mean = np.zeros(n_coef)
            idx = 0
            for lag in range(1, p + 1):
                for i in range(k):
                    if i == j:
                        prior_var[idx] = (self.lambda1 / lag) ** 2
                        if lag == 1:
                            prior_mean[idx] = 1.0
                    else:
                        prior_var[idx] = (self.lambda1 * self.lambda2 *
                                          sigmas[j] / (lag * sigmas[i])) ** 2
                    idx += 1
            prior_var[idx] = 1e12
            D    = np.diag(1.0 / (prior_var + 1e-12))
            beta = np.linalg.solve(X.T @ X + D,
                                   X.T @ Z[:, j] + D @ prior_mean)
            coefs[:, j] = beta
        self.coefs_ = coefs
        return self

    def predict_one_step(self, Y_history):
        p   = self.lags
        k   = Y_history.shape[1]
        row = np.ones(k * p + 1)
        for lag in range(1, p + 1):
            row[(lag-1)*k : lag*k] = Y_history[-lag, :]
        return row @ self.coefs_

# ── HELPERS ───────────────────────────────────────────────────────────────────

def compute_metrics(preds, actuals):
    p = np.array(preds, dtype=float)
    a = np.array(actuals, dtype=float)
    valid = ~(np.isnan(p) | np.isnan(a))
    p, a  = p[valid], a[valid]
    if len(p) == 0:
        return dict(MAE=np.nan, RMSE=np.nan, R2=np.nan)
    mae    = np.mean(np.abs(p - a))
    rmse   = np.sqrt(np.mean((p - a)**2))
    ss_res = np.sum((p - a)**2)
    ss_tot = np.sum((a - np.mean(a))**2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    return dict(MAE=mae, RMSE=rmse, R2=r2)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading data...")
print("=" * 60)

PHASE1_COLS = ['k_hat','c_hat','z_hat','xi_hat',
               'gdp_growth_level','gdp_growth_vol',
               'ulc_growth_level','ulc_growth_vol',
               'inflation_level','inflation_vol',
               'unemp_level','unemp_vol',
               'spread_level','spread_vol']
PHASE2_COLS = ['baa_aaa','fedfunds','vix_log','term_spread','sp_logret']

df_all  = pd.read_csv(DATA_DIR + 'finetune_dataset.csv')
df_usa  = df_all[df_all['countrycode'] == 'USA'].sort_values('period').reset_index(drop=True)
df_p2   = pd.read_csv(DATA_DIR + 'phase2_features (1).csv').sort_values('period').reset_index(drop=True)

df_merged = (df_usa[['period'] + PHASE1_COLS]
             .merge(df_p2[['period'] + PHASE2_COLS], on='period', how='inner')
             .sort_values('period').reset_index(drop=True))

n_train = int(len(df_merged) * 0.8)
val_idx = list(range(n_train, len(df_merged) - 1))

print(f"  Total: {len(df_merged)}  Train: {n_train}  Val: {len(val_idx)}")
print(f"  Val: {df_merged['period'].iloc[n_train]} to {df_merged['period'].iloc[-2]}")

# ── STANDARDISE FOR VAR ───────────────────────────────────────────────────────
# Fit scaler on training data, apply to full series
# Predictions are transformed back to original scale

Y_raw = df_merged[VAR_COLS].values.astype(float)

var_mu  = Y_raw[:n_train].mean(axis=0)
var_sig = Y_raw[:n_train].std(axis=0) + 1e-10
Y_scaled = (Y_raw - var_mu) / var_sig

Y_train_sc = Y_scaled[:n_train]
Y_all_sc   = Y_scaled

def unscale(Y_sc, col_indices):
    """Transform scaled predictions back to original units."""
    result = Y_sc.copy()
    for j, ci in enumerate(col_indices):
        result[:, j] = Y_sc[:, j] * var_sig[ci] + var_mu[ci]
    return result

TARGET_CI = [VAR_COLS.index(t) for t in TARGET_COLS]

# ── FIT MODELS ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"Fitting VAR({LAGS}) and BVAR on standardised data...")
print("=" * 60)

var_fit = VAR(Y_train_sc).fit(LAGS, trend='c')
print(f"  VAR({LAGS}) fitted.")

bvar = BVARMinnesota(lags=LAGS, lambda1=0.2, lambda2=0.5).fit(Y_train_sc)
print(f"  BVAR fitted.")

# AR(1) on original scale
ar1 = {}
for col in TARGET_COLS:
    s   = df_merged[col].iloc[:n_train].values
    rho = np.corrcoef(s[:-1], s[1:])[0, 1]
    mu  = s.mean()
    ar1[col] = dict(rho=rho, mu=mu)
    print(f"  AR(1) {col}: rho={rho:.3f}  mu={mu:.6f}")

# Neural net
print("\nLoading neural network...")
net = load_model(DATA_DIR + 'rbc_finetuned_phase2_final (1).pt')

norm_stats = {}
for col in PHASE1_COLS + PHASE2_COLS:
    mu  = df_merged[col].iloc[:n_train].mean()
    sig = df_merged[col].iloc[:n_train].std()
    if sig < 1e-10: sig = 1.0
    norm_stats[col] = dict(mean=float(mu), std=float(sig))

def net_predict(t):
    row    = df_merged[PHASE1_COLS + PHASE2_COLS].iloc[t].values.astype(np.float32)
    normed = np.array([(row[i] - norm_stats[c]['mean']) / norm_stats[c]['std']
                       for i, c in enumerate(PHASE1_COLS + PHASE2_COLS)], dtype=np.float32)
    x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = net(x)[0].numpy()
    return {TARGET_COLS[i]: pred[i] * norm_stats[TARGET_COLS[i]]['std'] +
                              norm_stats[TARGET_COLS[i]]['mean']
            for i in range(3)}

# ── VALIDATION FORECASTS ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("One-step validation forecasts...")
print("=" * 60)

MODEL_NAMES = ['AR(1)', f'VAR({LAGS})', 'BVAR', 'Phase 2 Net']
preds_val   = {m: {t: [] for t in TARGET_COLS} for m in MODEL_NAMES}
actuals_val = {t: [] for t in TARGET_COLS}

for t in val_idx:
    for tname in TARGET_COLS:
        actuals_val[tname].append(df_merged[tname].iloc[t+1])

    # AR(1)
    for tname in TARGET_COLS:
        p = ar1[tname]
        preds_val['AR(1)'][tname].append(
            p['mu'] + p['rho'] * (df_merged[tname].iloc[t] - p['mu']))

    # VAR and BVAR (scaled → predict → unscale)
    if t >= LAGS:
        hist_sc  = Y_all_sc[t-LAGS+1:t+1, :]
        fc_var   = var_fit.forecast(hist_sc, steps=1)[0]
        fc_bvar  = bvar.predict_one_step(hist_sc)
        # Unscale
        for i, tname in enumerate(TARGET_COLS):
            ci = VAR_COLS.index(tname)
            preds_val[f'VAR({LAGS})'][tname].append(
                fc_var[ci]  * var_sig[ci] + var_mu[ci])
            preds_val['BVAR'][tname].append(
                fc_bvar[ci] * var_sig[ci] + var_mu[ci])
    else:
        for tname in TARGET_COLS:
            preds_val[f'VAR({LAGS})'][tname].append(np.nan)
            preds_val['BVAR'][tname].append(np.nan)

    # Net
    np_pred = net_predict(t)
    for tname in TARGET_COLS:
        preds_val['Phase 2 Net'][tname].append(np_pred[tname])

# Metrics
print(f"\n  {'Model':16s}  {'k MAE':>8s}  {'k R²':>7s}  "
      f"{'z MAE':>8s}  {'z R²':>7s}  {'xi MAE':>9s}  {'xi R²':>8s}")
print("  " + "-" * 75)

val_metrics = {}
for mname in MODEL_NAMES:
    val_metrics[mname] = {}
    row = f"  {mname:16s}"
    for tname in TARGET_COLS:
        m = compute_metrics(preds_val[mname][tname], actuals_val[tname])
        val_metrics[mname][tname] = m
        row += f"  {m['MAE']:8.5f}  {m['R2']:7.3f}"
    print(row)

# ── MULTI-STEP ROLLOUT ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Multi-step rollout (1, 4, 8 quarters)...")
print("=" * 60)

HORIZONS    = [1, 4, 8]
n_eval      = min(20, n_train - max(HORIZONS) - LAGS)
eval_starts = list(range(n_train - n_eval, n_train))

ms = {m: {h: {t: [] for t in TARGET_COLS} for h in HORIZONS}
      for m in MODEL_NAMES}

for t_start in eval_starts:
    for h in HORIZONS:
        if t_start + h >= len(df_merged): continue
        actual_h = df_merged[TARGET_COLS].iloc[t_start + h].values.astype(float)

        # AR(1)
        state = {tname: df_merged[tname].iloc[t_start] for tname in TARGET_COLS}
        for _ in range(h):
            state = {tname: ar1[tname]['mu'] + ar1[tname]['rho'] *
                     (state[tname] - ar1[tname]['mu']) for tname in TARGET_COLS}
        for i, tname in enumerate(TARGET_COLS):
            ms['AR(1)'][h][tname].append(abs(state[tname] - actual_h[i]))

        # VAR and BVAR rollout in scaled space
        if t_start >= LAGS:
            hist_var  = Y_all_sc[t_start-LAGS+1:t_start+1, :].copy()
            hist_bvar = hist_var.copy()
            for _ in range(h):
                fc_v = var_fit.forecast(hist_var[-LAGS:, :], steps=1)[0]
                fc_b = bvar.predict_one_step(hist_bvar[-LAGS:, :])
                hist_var  = np.vstack([hist_var,  fc_v])
                hist_bvar = np.vstack([hist_bvar, fc_b])
            for i, tname in enumerate(TARGET_COLS):
                ci = VAR_COLS.index(tname)
                var_pred  = hist_var[-1, ci]  * var_sig[ci] + var_mu[ci]
                bvar_pred = hist_bvar[-1, ci] * var_sig[ci] + var_mu[ci]
                ms[f'VAR({LAGS})'][h][tname].append(abs(var_pred  - actual_h[i]))
                ms['BVAR'][h][tname].append(abs(bvar_pred - actual_h[i]))

        # Net at h=1 only
        if h == 1:
            np_pred = net_predict(t_start)
            for i, tname in enumerate(TARGET_COLS):
                ms['Phase 2 Net'][h][tname].append(abs(np_pred[tname] - actual_h[i]))

# Print table
print(f"\n  {'Model':16s}  H", end='')
for tlabel in TARGET_LABELS:
    print(f"  {tlabel+' MAE':>18s}", end='')
print()
print("  " + "-" * 85)
for mname in MODEL_NAMES:
    for h in HORIZONS:
        print(f"  {mname:16s}  {h}Q", end='')
        for tname in TARGET_COLS:
            vals = ms[mname][h][tname]
            print(f"  {np.mean(vals) if vals else np.nan:>18.5f}", end='')
        print()
    print()

# ── PLOTS ─────────────────────────────────────────────────────────────────────

print("Plotting...")
colors = {'AR(1)': '#9467BD', f'VAR({LAGS})': '#D62728',
          'BVAR': '#FF7F0E', 'Phase 2 Net': '#2CA02C'}

# Plot 1: Forecast trajectories
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('One-Step Forecast vs Actual — Validation Set',
             fontsize=13, fontweight='bold')
for ax, tname, tlabel in zip(axes, TARGET_COLS, TARGET_LABELS):
    ax.plot(actuals_val[tname], 'k-', linewidth=2.5, label='Actual', zorder=5)
    for mname in MODEL_NAMES:
        ls = '--' if mname == 'AR(1)' else '-'
        ax.plot(preds_val[mname][tname], color=colors[mname],
                linewidth=1.5, alpha=0.85, label=mname, linestyle=ls)
    ax.set_title(tlabel); ax.set_xlabel('Val quarter')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.7, linestyle=':')
plt.tight_layout()
plt.savefig(OUT_DIR + 'VAR_val_forecast.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 2: MAE bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('One-Step MAE — All Benchmarks vs Phase 2 Network',
             fontsize=13, fontweight='bold')
for ax, tname, tlabel in zip(axes, TARGET_COLS, TARGET_LABELS):
    maes = [val_metrics[m][tname]['MAE'] for m in MODEL_NAMES]
    bars = ax.bar(MODEL_NAMES, maes,
                  color=[colors[m] for m in MODEL_NAMES],
                  alpha=0.85, edgecolor='white')
    ax.set_title(tlabel); ax.set_ylabel('MAE'); ax.grid(axis='y', alpha=0.3)
    top = max(m for m in maes if not np.isnan(m))
    for bar, val in zip(bars, maes):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + top * 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR + 'VAR_mae_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 3: Multi-step MAE
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Multi-Step Forecast MAE by Horizon',
             fontsize=13, fontweight='bold')
for ax, tname, tlabel in zip(axes, TARGET_COLS, TARGET_LABELS):
    for mname in [f'VAR({LAGS})', 'BVAR', 'AR(1)']:
        maes = [np.mean(ms[mname][h][tname])
                if ms[mname][h][tname] else np.nan for h in HORIZONS]
        ax.plot(HORIZONS, maes, 'o-', color=colors[mname],
                linewidth=2, markersize=7, label=mname)
    net1 = ms['Phase 2 Net'][1][tname]
    if net1:
        ax.scatter([1], [np.mean(net1)], color=colors['Phase 2 Net'],
                   s=120, zorder=5, label='Phase 2 Net (1Q)', marker='*')
    ax.set_title(tlabel); ax.set_xlabel('Horizon (quarters)')
    ax.set_ylabel('MAE'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_xticks(HORIZONS)
plt.tight_layout()
plt.savefig(OUT_DIR + 'VAR_multistep_mae.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Done. Outputs in: {OUT_DIR}")
print("   VAR_val_forecast.png  VAR_mae_bar.png  VAR_multistep_mae.png")
