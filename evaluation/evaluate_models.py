"""
evaluate_models.py
==================
Full evaluation of three model stages:
  1. Pretrained (RBC synthetic only)
  2. Phase 1 fine-tuned (cross-country macro panel)
  3. Phase 2 fine-tuned (US + financial features)

Evaluations:
  A. Structural check  — IRFs and autocorrelations
  B. One-step forecast — MAE / RMSE / R² vs AR(1) baseline
  C. Multi-step rollout — 1, 4, 8 quarters ahead
  D. Event study       — 2001, 2008 crisis rollouts

Files needed (upload to /content/ in Colab):
  - rbc_pretrained_15.pt
  - rbc_finetuned_final.pt
  - rbc_finetuned_phase2_final.pt
  - finetune_dataset.csv
  - phase2_features.csv
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

# Set these to wherever your files are
DATA_DIR    = '/content/'
OUT_DIR     = '/content/eval_outputs/'
os.makedirs(OUT_DIR, exist_ok=True)

PRETRAINED_PATH = DATA_DIR + 'rbc_pretrained_15.pt'
PHASE1_PATH     = DATA_DIR + 'rbc_finetuned_final.pt'
PHASE2_PATH     = DATA_DIR + 'rbc_finetuned_phase2_final (1).pt'
PHASE1_CSV      = DATA_DIR + 'finetune_dataset.csv'
PHASE2_CSV      = DATA_DIR + 'phase2_features (1).csv'

TARGET_NAMES = ['k_hat', 'z_hat', 'xi_hat']
TARGET_LABELS = ['k̂ (capital)', 'ẑ (TFP)', 'ξ̂ (inv. shock)']

# RBC benchmark calibration for structural checks
ALPHA  = 0.33
BETA   = 0.99
DELTA  = 0.025
PHI    = 1.0
RHO_Z  = 0.90
RHO_XI = 0.70
CTOY   = 0.75

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
    def __init__(self, input_dim=19, output_dim=3, hidden_dim=512,
                 n_layers=4, dropout=0.1):
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
    r = 1.0 / beta - 1.0 + delta
    g_c  = -delta*y_over_i*(1-alpha)/(phi+alpha) - delta*c_over_i
    g_k  = (1-delta) + delta*y_over_i*alpha*(phi+1)/(phi+alpha)
    g_z  = delta*y_over_i*(phi+1)/(phi+alpha)
    g_xi = delta
    M = 1 + beta*r*(1-alpha)/(phi+alpha)
    N = beta*r*phi*(1-alpha)/(phi+alpha)
    H = beta*r*(phi+1)/(phi+alpha)

    def residuals(v):
        a_k, a_z, a_xi = v
        Gk  = g_c*a_k + g_k; Gz = g_c*a_z + g_z; Gxi = g_c*a_xi + g_xi
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
        Gk = g_c*a_k+g_k; Gz = g_c*a_z+g_z; Gxi = g_c*a_xi+g_xi
        P = np.array([
            [a_k*g_c, a_k*g_k, a_k*g_z+a_z*rho_z, a_k*g_xi+a_xi*rho_xi],
            [g_c, g_k, g_z, g_xi],
            [0., 0., rho_z, 0.],
            [0., 0., 0., rho_xi],
        ])
        eigvals = np.linalg.eigvals(P)
        candidates.append(dict(a_k=a_k, a_z=a_z, a_xi=a_xi, P=P,
                               max_abs_eig=np.max(np.abs(eigvals)),
                               res_norm=np.linalg.norm(residuals([a_k,a_z,a_xi]))))
    stable = [c for c in candidates if c['max_abs_eig'] < 1-1e-8]
    if not stable: raise RuntimeError("No stable solution")
    return min(stable, key=lambda x: (x['res_norm'], x['max_abs_eig']))

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

print("Loading data...")
df_all  = pd.read_csv(PHASE1_CSV)
df_usa  = df_all[df_all['countrycode'] == 'USA'].copy().sort_values('period').reset_index(drop=True)
df_p2   = pd.read_csv(PHASE2_CSV).sort_values('period').reset_index(drop=True)

# Merge for phase 2 evaluation
PHASE1_COLS = ['k_hat','c_hat','z_hat','xi_hat',
               'gdp_growth_level','gdp_growth_vol',
               'ulc_growth_level','ulc_growth_vol',
               'inflation_level','inflation_vol',
               'unemp_level','unemp_vol',
               'spread_level','spread_vol']
PHASE2_COLS = ['baa_aaa','fedfunds','vix_log','term_spread','sp_logret']
TARGET_COLS = ['k_hat','z_hat','xi_hat']

df_merged = (df_usa[['period'] + PHASE1_COLS]
             .merge(df_p2[['period'] + PHASE2_COLS], on='period', how='inner')
             .sort_values('period').reset_index(drop=True))

print(f"  USA phase1 rows: {len(df_usa)}")
print(f"  Merged rows (phase2): {len(df_merged)}")

# ── HELPER: BUILD INPUT TENSORS ───────────────────────────────────────────────

def build_inputs_phase1(df, idx):
    """Build 19-dim input for phase1 model (zeros in slots 14-18)."""
    row = df[PHASE1_COLS].iloc[idx].values.astype(np.float32)
    x = np.concatenate([row, np.zeros(5, dtype=np.float32)])
    return torch.tensor(x).unsqueeze(0)

def build_inputs_phase2(df_merged, idx):
    """Build 19-dim input for phase2 model (all features)."""
    row = np.concatenate([
        df_merged[PHASE1_COLS].iloc[idx].values.astype(np.float32),
        df_merged[PHASE2_COLS].iloc[idx].values.astype(np.float32),
    ])
    return torch.tensor(row).unsqueeze(0)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────

print("Loading models...")
models = {
    'Pretrained' : load_model(PRETRAINED_PATH),
    'Phase 1'    : load_model(PHASE1_PATH),
    'Phase 2'    : load_model(PHASE2_PATH),
}
print("  All models loaded.")

# Solve RBC for structural checks
sol    = solve_rbc_two_shock_model(ALPHA, BETA, DELTA, PHI, RHO_Z, RHO_XI, CTOY)
a_k, a_z, a_xi = sol['a_k'], sol['a_z'], sol['a_xi']
P_anal = sol['P']
print(f"  RBC solution: a_k={a_k:.3f}  a_z={a_z:.3f}  a_xi={a_xi:.3f}")

# ── SECTION A: STRUCTURAL CHECK ───────────────────────────────────────────────

print("\n" + "="*60)
print("A. STRUCTURAL CHECK — IRFs and Autocorrelations")
print("="*60)

SIGMA_Z = 0.01
T_IRF   = 20
T_ROLL  = 300

def compute_irf(model, sigma_z=SIGMA_Z, T=T_IRF):
    """Compute IRF to a 1 std TFP shock from steady state."""
    s_shock = torch.zeros(1, 19); s_shock[0, 2] = sigma_z
    s_base  = torch.zeros(1, 19)
    irf_k, irf_z, irf_xi, irf_c = [], [], [], []
    with torch.no_grad():
        for _ in range(T):
            ps = model(s_shock); pb = model(s_base)
            dk  = (ps[0,0] - pb[0,0]).item()
            dz  = (ps[0,1] - pb[0,1]).item()
            dxi = (ps[0,2] - pb[0,2]).item()
            dc  = a_k*dk + a_z*dz + a_xi*dxi
            irf_k.append(dk); irf_z.append(dz)
            irf_xi.append(dxi); irf_c.append(dc)
            ks = ps[0,0].item(); zs = ps[0,1].item(); xis = ps[0,2].item()
            cs = a_k*ks + a_z*zs + a_xi*xis
            kb = pb[0,0].item(); zb = pb[0,1].item(); xib = pb[0,2].item()
            cb = a_k*kb + a_z*zb + a_xi*xib
            s_shock = torch.tensor([[ks,cs,zs,xis]+[0]*15], dtype=torch.float32)
            s_base  = torch.tensor([[kb,cb,zb,xib]+[0]*15], dtype=torch.float32)
    return np.array(irf_k), np.array(irf_z), np.array(irf_xi), np.array(irf_c)

def compute_anal_irf(sigma_z=SIGMA_Z, T=T_IRF):
    s0 = np.array([a_z*sigma_z, 0.0, sigma_z, 0.0])
    traj = [s0]
    for _ in range(T-1):
        traj.append(P_anal @ traj[-1])
    traj = np.array(traj)
    return traj[:,1], traj[:,2], traj[:,3], traj[:,0]

def compute_autocorr(model, T=T_ROLL, sigma_z=0.01, sigma_xi=0.01, seed=42):
    rng = np.random.default_rng(seed)
    s = torch.zeros(1, 19)
    z_hist, xi_hist = [], []
    with torch.no_grad():
        for _ in range(T):
            pred = model(s)
            kn = pred[0,0].item(); zn = pred[0,1].item(); xin = pred[0,2].item()
            cn = a_k*kn + a_z*zn + a_xi*xin
            z_hist.append(zn); xi_hist.append(xin)
            eps_z  = rng.normal(0, sigma_z)
            eps_xi = rng.normal(0, sigma_xi)
            s = torch.tensor([[kn,cn,zn+eps_z,xin+eps_xi]+[0]*15], dtype=torch.float32)
    def ar1(x):
        x = np.array(x)
        return np.corrcoef(x[:-1], x[1:])[0,1]
    return ar1(z_hist), ar1(xi_hist)

# Compute IRFs
anal_k, anal_z, anal_xi, anal_c = compute_anal_irf()
irf_results = {}
ac_results  = {}
quarters = np.arange(T_IRF)

print("\n  Autocorrelations (calibrated: rho_z=0.90, rho_xi=0.70):")
print(f"  {'Model':12s}  rho_z    rho_xi")
print("  " + "-"*35)
for name, model in models.items():
    irf_results[name] = compute_irf(model)
    rz, rxi = compute_autocorr(model)
    ac_results[name] = (rz, rxi)
    flag_z  = "OK" if abs(rz  - RHO_Z)  < 0.10 else "WARN"
    flag_xi = "OK" if abs(rxi - RHO_XI) < 0.15 else "WARN"
    print(f"  {name:12s}  {rz:.3f} [{flag_z}]  {rxi:.3f} [{flag_xi}]")

# Plot IRFs
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Impulse Response Functions — 1σ TFP Shock", fontsize=14, fontweight='bold')
colors = {'Pretrained':'steelblue', 'Phase 1':'darkorange', 'Phase 2':'green'}
irf_vars = [
    ('k̂ (capital)',     0, anal_k),
    ('ẑ (TFP shock)',   1, anal_z),
    ('ξ̂ (inv. shock)', 2, anal_xi),
    ('ĉ (consumption)', 3, anal_c),
]
for ax, (title, idx, anal) in zip(axes.flat, irf_vars):
    ax.plot(quarters, anal, 'k--', linewidth=2, label='Analytical', zorder=5)
    for name, model in models.items():
        irf = irf_results[name]
        ax.plot(quarters, irf[idx] if idx < 3 else irf[3],
                color=colors[name], linewidth=1.5, label=name)
    ax.axhline(0, color='gray', linewidth=0.7, linestyle=':')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Quarters after shock')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR + 'A_irf.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  IRF plot saved.")

# ── SECTION B: ONE-STEP FORECAST ──────────────────────────────────────────────

print("\n" + "="*60)
print("B. ONE-STEP FORECAST — MAE / RMSE / R²")
print("="*60)

# Val set: last 24 quarters of merged data (chronological)
N       = len(df_merged)
n_train = int(N * 0.80)
val_idx = list(range(n_train, N - 1))  # -1 because we need t+1

print(f"\n  Val set: {df_merged['period'].iloc[n_train]} to "
      f"{df_merged['period'].iloc[-2]}  ({len(val_idx)} quarters)")

def ar1_forecast(series, val_start):
    """Simple AR(1) baseline trained on first val_start observations."""
    train = series[:val_start]
    rho   = np.corrcoef(train[:-1], train[1:])[0,1]
    mu    = train.mean()
    preds = mu + rho * (series[val_start-1:-1] - mu)
    actuals = series[val_start:]
    return preds, actuals

def compute_metrics(preds, actuals):
    mae  = np.mean(np.abs(preds - actuals))
    rmse = np.sqrt(np.mean((preds - actuals)**2))
    ss_res = np.sum((actuals - preds)**2)
    ss_tot = np.sum((actuals - actuals.mean())**2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    dir_acc = np.mean(np.sign(preds) == np.sign(actuals))
    return dict(MAE=mae, RMSE=rmse, R2=r2, DirAcc=dir_acc)

results_1step = {}

# AR(1) baselines
print("\n  AR(1) baselines:")
ar1_results = {}
for i, (tname, tlabel) in enumerate(zip(TARGET_COLS, TARGET_LABELS)):
    series = df_merged[tname].values
    preds, actuals = ar1_forecast(series, n_train)
    ar1_results[tname] = compute_metrics(preds, actuals)
    m = ar1_results[tname]
    print(f"  {tlabel:20s}: MAE={m['MAE']:.5f}  RMSE={m['RMSE']:.5f}  R²={m['R2']:+.3f}  DirAcc={m['DirAcc']:.2f}")

# Model forecasts
for mname, model in models.items():
    results_1step[mname] = {}
    for i, (tname, tlabel) in enumerate(zip(TARGET_COLS, TARGET_LABELS)):
        preds, actuals = [], []
        for t in val_idx:
            x = build_inputs_phase2(df_merged, t) if mname == 'Phase 2' else build_inputs_phase1(df_usa, t) if mname in ['Pretrained','Phase 1'] else build_inputs_phase2(df_merged, t)
            with torch.no_grad():
                pred = model(x)[0].numpy()
            # Map output index: 0=k_hat, 1=z_hat, 2=xi_hat
            out_map = {'k_hat':0, 'z_hat':1, 'xi_hat':2}
            preds.append(pred[out_map[tname]])
            actuals.append(df_merged[tname].iloc[t+1])
        preds   = np.array(preds)
        actuals = np.array(actuals)
        results_1step[mname][tname] = compute_metrics(preds, actuals)

# Print table
print("\n  One-step forecast metrics on val set (2014Q1-2019Q4):")
for tname, tlabel in zip(TARGET_COLS, TARGET_LABELS):
    print(f"\n  {tlabel}:")
    print(f"  {'Model':12s}  {'MAE':>8s}  {'RMSE':>8s}  {'R²':>8s}  {'DirAcc':>8s}")
    print("  " + "-"*50)
    m = ar1_results[tname]
    print(f"  {'AR(1)':12s}  {m['MAE']:8.5f}  {m['RMSE']:8.5f}  {m['R2']:8.3f}  {m['DirAcc']:8.2f}")
    for mname in ['Pretrained','Phase 1','Phase 2']:
        m = results_1step[mname][tname]
        beat = "✓" if m['MAE'] < ar1_results[tname]['MAE'] else " "
        print(f"  {mname:12s}  {m['MAE']:8.5f}  {m['RMSE']:8.5f}  {m['R2']:8.3f}  {m['DirAcc']:8.2f}  {beat}")

# Plot forecast comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("One-Step Ahead Forecast vs Actual — Val Set", fontsize=13, fontweight='bold')
val_periods = [df_merged['period'].iloc[t+1] for t in val_idx]
for ax, (tname, tlabel) in zip(axes, zip(TARGET_COLS, TARGET_LABELS)):
    actuals = [df_merged[tname].iloc[t+1] for t in val_idx]
    ax.plot(range(len(val_idx)), actuals, 'k-', linewidth=2, label='Actual')
    for mname, model in models.items():
        preds = []
        for t in val_idx:
            x = build_inputs_phase2(df_merged, t) if mname == 'Phase 2' else build_inputs_phase1(df_usa, t)
            with torch.no_grad():
                pred = model(x)[0].numpy()
            out_map = {'k_hat':0,'z_hat':1,'xi_hat':2}
            preds.append(pred[out_map[tname]])
        ax.plot(range(len(val_idx)), preds, color=colors[mname],
                linewidth=1.2, alpha=0.8, label=mname)
    ax.set_title(tlabel); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel('Val quarter')
plt.tight_layout()
plt.savefig(OUT_DIR + 'B_onestep_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  One-step forecast plot saved.")

# ── SECTION C: MULTI-STEP ROLLOUT ─────────────────────────────────────────────

print("\n" + "="*60)
print("C. MULTI-STEP ROLLOUT — 1, 4, 8 quarters ahead")
print("="*60)

HORIZONS = [1, 4, 8]

def multistep_rollout(model, df, start_idx, horizon, use_phase2=False):
    """Roll model forward h steps from start_idx, return predictions."""
    row = df[PHASE1_COLS].iloc[start_idx].values.astype(np.float32)
    if use_phase2:
        p2_row = df[PHASE2_COLS].iloc[start_idx].values.astype(np.float32)
        s = torch.tensor(np.concatenate([row, p2_row]), dtype=torch.float32).unsqueeze(0)
    else:
        s = torch.tensor(np.concatenate([row, np.zeros(5, dtype=np.float32)]), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        for _ in range(horizon):
            pred = model(s)
            kn, zn, xin = pred[0,0].item(), pred[0,1].item(), pred[0,2].item()
            cn = a_k*kn + a_z*zn + a_xi*xin
            if use_phase2 and start_idx + 1 < len(df):
                p2_next = df[PHASE2_COLS].iloc[min(start_idx+1, len(df)-1)].values.astype(np.float32)
                s = torch.tensor(np.concatenate([[kn,cn,zn,xin], df[PHASE1_COLS[4:]].iloc[min(start_idx+1,len(df)-1)].values.astype(np.float32), p2_next]), dtype=torch.float32).unsqueeze(0)
            else:
                s = torch.tensor(np.concatenate([[kn,cn,zn,xin], row[4:], np.zeros(5, dtype=np.float32)]), dtype=torch.float32).unsqueeze(0)
    return np.array([kn, zn, xin])

# Compute multi-step MAE for each horizon
print(f"\n  {'Model':12s}  {'Horizon':>8s}  {'k_hat MAE':>10s}  {'z_hat MAE':>10s}  {'xi_hat MAE':>10s}")
print("  " + "-"*60)

multistep_results = {name: {h: [] for h in HORIZONS} for name in models}
n_eval = min(30, len(df_merged) - max(HORIZONS) - 1)
eval_starts = list(range(n_train - n_eval, n_train))

for h in HORIZONS:
    for mname, model in models.items():
        maes = []
        for t in eval_starts:
            if t + h >= len(df_merged): continue
            use_p2 = (mname == 'Phase 2')
            pred_h = multistep_rollout(model, df_merged, t, h, use_phase2=use_p2)
            actual = df_merged[TARGET_COLS].iloc[t+h].values.astype(np.float32)
            maes.append(np.abs(pred_h - actual))
        multistep_results[mname][h] = np.mean(maes, axis=0) if maes else np.zeros(3)

    for mname in ['Pretrained','Phase 1','Phase 2']:
        m = multistep_results[mname][h]
        print(f"  {mname:12s}  {h:>8d}Q  {m[0]:10.5f}  {m[1]:10.5f}  {m[2]:10.5f}")
    print()

# Plot multi-step MAE by horizon
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Multi-Step Forecast MAE by Horizon", fontsize=13, fontweight='bold')
for ax, (ti, (tname, tlabel)) in zip(axes, enumerate(zip(TARGET_COLS, TARGET_LABELS))):
    for mname in ['Pretrained','Phase 1','Phase 2']:
        maes = [multistep_results[mname][h][ti] for h in HORIZONS]
        ax.plot(HORIZONS, maes, 'o-', color=colors[mname], linewidth=2,
                markersize=7, label=mname)
    ax.set_title(tlabel); ax.set_xlabel('Horizon (quarters)')
    ax.set_ylabel('MAE'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xticks(HORIZONS)
plt.tight_layout()
plt.savefig(OUT_DIR + 'C_multistep_mae.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Multi-step plot saved.")

# ── SECTION D: EVENT STUDY ────────────────────────────────────────────────────

print("\n" + "="*60)
print("D. EVENT STUDY — 2001 dotcom and 2008 financial crisis")
print("="*60)

EVENTS = {
    '2001 Dotcom Crash' : ('2000Q1', '2003Q4'),
    '2008 Financial Crisis': ('2007Q1', '2011Q4'),
}

fig, axes = plt.subplots(len(EVENTS), 3, figsize=(15, 8))
fig.suptitle("Event Studies — Network Rollout vs Actual", fontsize=13, fontweight='bold')

for row_idx, (event_name, (start_period, end_period)) in enumerate(EVENTS.items()):
    # Find start index in merged data
    periods = df_merged['period'].values
    start_mask = periods == start_period
    if not start_mask.any():
        # Find closest
        start_idx = df_merged[df_merged['period'] >= start_period].index[0]
    else:
        start_idx = df_merged[start_mask].index[0]

    end_mask = periods == end_period
    if not end_mask.any():
        end_idx = min(start_idx + 16, len(df_merged) - 1)
    else:
        end_idx = df_merged[end_mask].index[0]

    n_steps  = end_idx - start_idx
    actuals  = df_merged[TARGET_COLS].iloc[start_idx:end_idx+1].values

    for col_idx, (tname, tlabel) in enumerate(zip(TARGET_COLS, TARGET_LABELS)):
        ax = axes[row_idx, col_idx]
        ax.plot(range(n_steps+1), actuals[:, col_idx], 'k-',
                linewidth=2.5, label='Actual', zorder=5)

        for mname, model in models.items():
            use_p2 = (mname == 'Phase 2')
            row_data = df_merged[PHASE1_COLS].iloc[start_idx].values.astype(np.float32)
            if use_p2:
                p2_row = df_merged[PHASE2_COLS].iloc[start_idx].values.astype(np.float32)
                s = torch.tensor(np.concatenate([row_data, p2_row]), dtype=torch.float32).unsqueeze(0)
            else:
                s = torch.tensor(np.concatenate([row_data, np.zeros(5, dtype=np.float32)]), dtype=torch.float32).unsqueeze(0)

            rollout = [df_merged[tname].iloc[start_idx]]
            out_map = {'k_hat':0,'z_hat':1,'xi_hat':2}
            with torch.no_grad():
                for step in range(n_steps):
                    pred = model(s)
                    kn = pred[0,0].item(); zn = pred[0,1].item(); xin = pred[0,2].item()
                    cn = a_k*kn + a_z*zn + a_xi*xin
                    rollout.append(pred[0, out_map[tname]].item())
                    next_idx = min(start_idx + step + 1, len(df_merged)-1)
                    if use_p2:
                        p2_next = df_merged[PHASE2_COLS].iloc[next_idx].values.astype(np.float32)
                        feat_next = df_merged[PHASE1_COLS[4:]].iloc[next_idx].values.astype(np.float32)
                        s = torch.tensor(np.concatenate([[kn,cn,zn,xin], feat_next, p2_next]), dtype=torch.float32).unsqueeze(0)
                    else:
                        s = torch.tensor(np.concatenate([[kn,cn,zn,xin], row_data[4:], np.zeros(5, dtype=np.float32)]), dtype=torch.float32).unsqueeze(0)

            ax.plot(range(n_steps+1), rollout, color=colors[mname],
                    linewidth=1.5, alpha=0.85, label=mname)

        if col_idx == 0:
            ax.set_ylabel(event_name, fontsize=10, fontweight='bold')
        ax.set_title(tlabel if row_idx == 0 else '')
        ax.axhline(0, color='gray', linewidth=0.7, linestyle=':')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_xlabel('Quarters from event start')

plt.tight_layout()
plt.savefig(OUT_DIR + 'D_event_study.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Event study plot saved.")

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
print(f"\n  One-step val MAE (averaged across k̂, ẑ, ξ̂):")
for mname in ['Pretrained','Phase 1','Phase 2']:
    avg_mae = np.mean([results_1step[mname][t]['MAE'] for t in TARGET_COLS])
    avg_r2  = np.mean([results_1step[mname][t]['R2']  for t in TARGET_COLS])
    print(f"  {mname:12s}: avg MAE={avg_mae:.5f}  avg R²={avg_r2:+.3f}")
ar1_avg = np.mean([ar1_results[t]['MAE'] for t in TARGET_COLS])
print(f"  {'AR(1)':12s}: avg MAE={ar1_avg:.5f}")

print(f"\n  Autocorrelations vs calibrated (rho_z=0.90, rho_xi=0.70):")
for mname in ['Pretrained','Phase 1','Phase 2']:
    rz, rxi = ac_results[mname]
    print(f"  {mname:12s}: rho_z={rz:.3f}  rho_xi={rxi:.3f}")

print(f"\nAll plots saved to: {OUT_DIR}")
print("Files: A_irf.png  B_onestep_forecast.png  C_multistep_mae.png  D_event_study.png")
