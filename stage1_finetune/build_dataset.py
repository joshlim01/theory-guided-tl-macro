"""
Fine-tuning dataset construction
"""

import pandas as pd
import numpy as np
from scipy.optimize import root
from scipy.interpolate import CubicSpline
from statsmodels.tsa.filters.hp_filter import hpfilter
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/mnt/user-data/uploads/'
OUT_DIR  = '/mnt/user-data/outputs/'

CPI_FILE       = DATA_DIR + 'OECD_SDD_TPS_DSD_PRICES_DF_PRICES_ALL_1_0_CAN_FIN_DNK_JPN_NLD_NOR_SWE_CHE_GBR_USA_Q_N_CPI_IX__T__.csv'
UNEMP_FILE     = DATA_DIR + 'OECD_SDD_TPS_DSD_LFS_DF_IALFS_INDIC_1_0_CAN_DNK_FIN_JPN_NLD_NOR_SWE_GBR_USA_UNE_LF_PT_LF_SUB__Y__T_Y_GE15__Q.csv'
CHE_UNEMP_FILE = DATA_DIR + 'LRHUTTTTCHQ156S.csv'
IR_FILE        = DATA_DIR + 'OECD_SDD_STES_DSD_STES_DF_FINMARK_4_0_NOR_FIN_JPN_NLD_SWE_CHE_GBR_USA_CAN_DNK_Q_IRLT_IR3TIB_PA_____.csv'
ULC_FILE       = DATA_DIR + 'OECD_SDD_TPS_DSD_PDB_DF_PDB_ULC_Q_1_0_CAN_DNK_FIN_JPN_NLD_NOR_SWE_CHE_GBR_USA_Q_ULCE__IX_V__Z_S_.csv'
GDP_FILE       = DATA_DIR + 'OECD_SDD_NAD_DSD_NAMAIN1_DF_QNA_1_1_Q_Y_CAN_DNK_FIN_JPN_NLD_NOR_SWE_CHE_GBR_USA_S1__B1GQ__Z___IX_LR_N_T0102.csv'
PWT_FILE       = DATA_DIR + 'pwt110.dta'

COUNTRIES    = ['CAN', 'DNK', 'FIN', 'JPN', 'NLD', 'NOR', 'SWE', 'CHE', 'GBR', 'USA']
ALL_QUARTERS = pd.period_range(start='1980Q1', end='2019Q4', freq='Q')
FIXED_PARAMS = dict(beta=0.99, phi=1.0, rho_z=0.90, rho_xi=0.70)

# ── SOLVER ────────────────────────────────────────────────────────────────────

def solve_rbc_two_shock_model(alpha, beta, delta, phi, rho_z, rho_xi, ctoy_rat):
    i_over_y = 1.0 - ctoy_rat
    if i_over_y <= 0:
        raise ValueError("Need 1 - ctoy_rat > 0.")
    c_over_i = ctoy_rat / i_over_y
    y_over_i = 1.0 / i_over_y
    r = 1.0 / beta - 1.0 + delta
    g_c  = -delta * y_over_i * (1.0 - alpha) / (phi + alpha) - delta * c_over_i
    g_k  = (1.0 - delta) + delta * y_over_i * alpha * (phi + 1.0) / (phi + alpha)
    g_z  = delta * y_over_i * (phi + 1.0) / (phi + alpha)
    g_xi = delta
    M = 1.0 + beta * r * (1.0 - alpha) / (phi + alpha)
    N = beta * r * phi * (1.0 - alpha) / (phi + alpha)
    H = beta * r * (phi + 1.0) / (phi + alpha)

    def residuals(v):
        a_k, a_z, a_xi = v
        Gk  = g_c * a_k  + g_k
        Gz  = g_c * a_z  + g_z
        Gxi = g_c * a_xi + g_xi
        eq_k  = a_k  - (M * a_k + N) * Gk
        eq_z  = a_z  - (M * (a_k * Gz  + a_z  * rho_z)  + N * Gz  - H * rho_z)
        eq_xi = a_xi - (M * (a_k * Gxi + a_xi * rho_xi) + N * Gxi)
        return np.array([eq_k, eq_z, eq_xi])

    guesses = [[0.1,0.1,0.1],[0.1,0.1,-0.1],[0.1,-0.1,0.1],[-0.1,0.1,0.1],
               [0.5,0.1,0.1],[0.5,-0.1,0.1],[1.0,0.1,0.1],[-0.5,0.1,0.1]]
    candidates = []
    for guess in guesses:
        sol = root(residuals, np.array(guess, dtype=float))
        if not sol.success:
            continue
        a_k, a_z, a_xi = sol.x
        Gk  = g_c*a_k + g_k; Gz = g_c*a_z + g_z; Gxi = g_c*a_xi + g_xi
        P = np.array([
            [a_k*g_c, a_k*g_k, a_k*g_z+a_z*rho_z,  a_k*g_xi+a_xi*rho_xi],
            [g_c, g_k, g_z, g_xi],
            [0., 0., rho_z, 0.],
            [0., 0., 0., rho_xi],
        ], dtype=float)
        Q = np.array([[a_z,a_xi],[0.,0.],[1.,0.],[0.,1.]], dtype=float)
        eigvals = np.linalg.eigvals(P)
        candidates.append(dict(a_k=a_k, a_z=a_z, a_xi=a_xi, P=P, Q=Q,
                               eigvals=eigvals, max_abs_eig=np.max(np.abs(eigvals)),
                               res_norm=np.linalg.norm(residuals([a_k,a_z,a_xi])),
                               g_c=g_c, g_k=g_k, g_z=g_z, g_xi=g_xi))
    stable = [c for c in candidates if c['max_abs_eig'] < 1.0 - 1e-8]
    if not stable:
        raise RuntimeError(f"No stable solution: alpha={alpha:.3f} delta={delta:.4f} ctoy={ctoy_rat:.3f}")
    return min(stable, key=lambda x: (x['res_norm'], x['max_abs_eig']))

# ── HELPERS ───────────────────────────────────────────────────────────────────

def hp_cycle(series, lamb=1600):
    clean = series.dropna()
    if len(clean) < 12:
        return pd.Series(np.nan, index=series.index)
    cycle, _ = hpfilter(clean, lamb=lamb)
    return pd.Series(cycle, index=clean.index).reindex(series.index)

def rolling_vol(series, window=8):
    return series.rolling(window=window, min_periods=4).std()

def load_oecd(filepath, value_col='OBS_VALUE', country_col='REF_AREA',
              time_col='TIME_PERIOD', measure_col=None, measure_val=None):
    df = pd.read_csv(filepath)
    if measure_col and measure_val:
        df = df[df[measure_col] == measure_val]
    df = df[[country_col, time_col, value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df['period']  = pd.PeriodIndex(df[time_col], freq='Q')
    wide = (df.drop(columns=[time_col])
              .pivot_table(index='period', columns=country_col,
                           values=value_col, aggfunc='mean'))
    return wide.reindex(ALL_QUARTERS)

def interpolate_annual_to_quarterly(annual_series):
    s = annual_series.dropna()
    if len(s) < 4:
        return pd.Series(np.nan, index=ALL_QUARTERS)
    cs = CubicSpline(s.index.astype(float) + 0.5, s.values.astype(float))
    q_vals = [float(cs(p.year + (p.quarter - 0.5) / 4.0)) for p in ALL_QUARTERS]
    return pd.Series(q_vals, index=ALL_QUARTERS)

# ── STEP 1: LOAD OECD DATA ────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Loading OECD data")
print("=" * 60)

cpi_raw   = load_oecd(CPI_FILE)
unemp_raw = load_oecd(UNEMP_FILE)
ulc_raw   = load_oecd(ULC_FILE)
gdp_raw   = load_oecd(GDP_FILE)

# CHE unemployment patch from FRED
che_u = pd.read_csv(CHE_UNEMP_FILE, parse_dates=['observation_date'])
che_u = che_u.dropna(subset=['LRHUTTTTCHQ156S'])
che_u['period'] = che_u['observation_date'].dt.to_period('Q')
che_series = (che_u.set_index('period')['LRHUTTTTCHQ156S']
                   .reindex(ALL_QUARTERS)
                   .ffill().bfill())
unemp_raw['CHE'] = che_series

# Interest rates
ir_df = pd.read_csv(IR_FILE)
ir_df['OBS_VALUE'] = pd.to_numeric(ir_df['OBS_VALUE'], errors='coerce')
ir_df['period']    = pd.PeriodIndex(ir_df['TIME_PERIOD'], freq='Q')
lt_raw = (ir_df[ir_df['MEASURE']=='IRLT']
          .pivot_table(index='period', columns='REF_AREA', values='OBS_VALUE', aggfunc='mean')
          .reindex(ALL_QUARTERS))
st_raw = (ir_df[ir_df['MEASURE']=='IR3TIB']
          .pivot_table(index='period', columns='REF_AREA', values='OBS_VALUE', aggfunc='mean')
          .reindex(ALL_QUARTERS))

print(f"  CPI countries:   {sorted(cpi_raw.columns.tolist())}")
print(f"  Unemp countries: {sorted(unemp_raw.columns.tolist())}")
print(f"  Long rate:       {sorted(lt_raw.columns.tolist())}")
print(f"  Short rate:      {sorted(st_raw.columns.tolist())}")
print(f"  ULC countries:   {sorted(ulc_raw.columns.tolist())}")
print(f"  GDP countries:   {sorted(gdp_raw.columns.tolist())}")

# ── STEP 2: PWT STATE VARIABLES ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 — Building state variables from PWT")
print("=" * 60)

pwt = pd.read_stata(PWT_FILE)
pwt = pwt[pwt['countrycode'].isin(COUNTRIES)].sort_values(['countrycode','year'])

def calibrate_and_build_states(cc, pwt):
    g     = pwt[pwt['countrycode']==cc].set_index('year')
    alpha = float(np.clip(1 - g['labsh'].mean(), 0.20, 0.50))
    if g['delta'].notna().any():
        delta = float(np.clip(g['delta'].mean(), 0.01, 0.12))
    else:
        delta = float(np.clip(g['csh_i'].mean() / (g['rkna']/g['rgdpna']).mean(), 0.01, 0.12))
    ctoy   = float(np.clip(g['csh_c'].mean(), 0.40, 0.85))
    params = dict(alpha=alpha, delta=delta, ctoy_rat=ctoy, **FIXED_PARAMS)

    k_hat = hp_cycle(interpolate_annual_to_quarterly(np.log(g['rkna'])))
    c_hat = hp_cycle(interpolate_annual_to_quarterly(np.log(g['csh_c'] * g['rgdpna'])))
    z_hat = hp_cycle(interpolate_annual_to_quarterly(np.log(g['rtfpna'])))

    sol   = solve_rbc_two_shock_model(**params)
    a_k, a_z, a_xi = sol['a_k'], sol['a_z'], sol['a_xi']
    xi_hat = (c_hat - a_k*k_hat - a_z*z_hat) / a_xi if abs(a_xi) > 1e-8 else pd.Series(0.0, index=ALL_QUARTERS)

    states = pd.DataFrame({'k_hat':k_hat,'c_hat':c_hat,'z_hat':z_hat,'xi_hat':xi_hat}, index=ALL_QUARTERS)
    return states, params, sol

state_dict, params_dict, sol_dict = {}, {}, {}
for cc in COUNTRIES:
    try:
        states, params, sol = calibrate_and_build_states(cc, pwt)
        state_dict[cc]  = states
        params_dict[cc] = params
        sol_dict[cc]    = sol
        print(f"  {cc}: alpha={params['alpha']:.3f} delta={params['delta']:.4f} ctoy={params['ctoy']:.3f} | "
              f"a_k={sol['a_k']:.3f} a_z={sol['a_z']:.3f} a_xi={sol['a_xi']:.3f}")
    except Exception as e:
        print(f"  {cc}: FAILED — {e}")

# ── STEP 3: BUILD FEATURES ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 — Building 10 real features per country")
print("=" * 60)

def build_features(cc):
    feats = {}
    feats['gdp_growth'] = gdp_raw[cc].pct_change().reindex(ALL_QUARTERS) if cc in gdp_raw.columns else pd.Series(np.nan, index=ALL_QUARTERS)
    feats['ulc_growth'] = ulc_raw[cc].pct_change().reindex(ALL_QUARTERS) if cc in ulc_raw.columns else pd.Series(np.nan, index=ALL_QUARTERS)
    feats['inflation']  = cpi_raw[cc].pct_change().reindex(ALL_QUARTERS) if cc in cpi_raw.columns else pd.Series(np.nan, index=ALL_QUARTERS)
    feats['unemp']      = unemp_raw[cc].reindex(ALL_QUARTERS)            if cc in unemp_raw.columns else pd.Series(np.nan, index=ALL_QUARTERS)
    lt = lt_raw[cc] if cc in lt_raw.columns else pd.Series(np.nan, index=ALL_QUARTERS)
    st = st_raw[cc] if cc in st_raw.columns else pd.Series(np.nan, index=ALL_QUARTERS)
    feats['spread'] = (lt - st).reindex(ALL_QUARTERS)
    result = {}
    for name, s in feats.items():
        result[f'{name}_level'] = s.reindex(ALL_QUARTERS)
        result[f'{name}_vol']   = rolling_vol(s.reindex(ALL_QUARTERS))
    return pd.DataFrame(result, index=ALL_QUARTERS)

feat_dict = {cc: build_features(cc) for cc in COUNTRIES}

# ── STEP 4: ASSEMBLE AND IMPUTE ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 — Assembling panel and imputing gaps")
print("=" * 60)

STATE_COLS   = ['k_hat', 'c_hat', 'z_hat', 'xi_hat']
FEATURE_COLS = ['gdp_growth_level','gdp_growth_vol','ulc_growth_level','ulc_growth_vol',
                'inflation_level','inflation_vol','unemp_level','unemp_vol',
                'spread_level','spread_vol']
PAD_COLS     = ['pad_1','pad_2','pad_3','pad_4','pad_5']
ALL_COLS     = STATE_COLS + FEATURE_COLS + PAD_COLS

rows = []
for cc in COUNTRIES:
    if cc not in state_dict:
        continue
    combined = pd.concat([state_dict[cc], feat_dict[cc]], axis=1)
    combined['countrycode'] = cc
    for col in PAD_COLS:
        combined[col] = 0.0
    rows.append(combined)

panel = pd.concat(rows).reset_index().rename(columns={'index':'period'})

# Missingness report
print("\n  Missing observations per feature per country:")
print(f"  {'country':8s} {'gdpG':>6s} {'ulcG':>6s} {'infl':>6s} {'unemp':>6s} {'sprd':>6s}")
for cc in COUNTRIES:
    sub = panel[panel['countrycode']==cc]
    vals = [sub[f'{f}_level'].isna().sum() for f in ['gdp_growth','ulc_growth','inflation','unemp','spread']]
    print(f"  {cc:8s} {vals[0]:6d} {vals[1]:6d} {vals[2]:6d} {vals[3]:6d} {vals[4]:6d}")

# Impute: cross-country quarterly mean, then ffill/bfill within country
for col in FEATURE_COLS:
    if panel[col].isna().any():
        qmean = panel.groupby('period')[col].transform('mean')
        panel[col] = panel[col].fillna(qmean)
        panel[col] = panel.groupby('countrycode')[col].transform(lambda x: x.ffill().bfill())

for col in STATE_COLS:
    if panel[col].isna().any():
        panel[col] = panel.groupby('countrycode')[col].transform(lambda x: x.ffill().bfill())

panel_clean = panel.dropna(subset=ALL_COLS)
print(f"\n  Panel after imputation: {panel_clean.shape}")
print(f"  Quarters per country:   {panel_clean.groupby('countrycode').size().to_dict()}")

# ── STEP 5: NORMALISE ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 — Normalising (z-score across full panel)")
print("=" * 60)

norm_stats  = {}
panel_norm  = panel_clean.copy()

for col in STATE_COLS + FEATURE_COLS:
    mu  = panel_clean[col].mean()
    sig = panel_clean[col].std()
    if sig < 1e-10: sig = 1.0
    panel_norm[col] = (panel_clean[col] - mu) / sig
    norm_stats[col] = dict(mean=float(mu), std=float(sig))
    print(f"  {col:28s}  mu={mu:+.4f}  sigma={sig:.4f}")

for col in PAD_COLS:
    panel_norm[col] = 0.0

# ── STEP 6: BUILD DATASET ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 — Building PyTorch dataset")
print("=" * 60)

TARGET_COLS = ['k_hat', 'z_hat', 'xi_hat']

class FineTuneDataset(Dataset):
    def __init__(self, panel, all_cols, target_cols):
        Xs, Ys, meta = [], [], []
        for cc in sorted(panel['countrycode'].unique()):
            sub = panel[panel['countrycode']==cc].sort_values('period').reset_index(drop=True)
            X_arr = sub[all_cols].values.astype(np.float32)
            Y_arr = sub[target_cols].values.astype(np.float32)
            for t in range(len(sub)-1):
                Xs.append(X_arr[t])
                Ys.append(Y_arr[t+1])
                meta.append((cc, str(sub['period'].iloc[t])))
        self.X    = torch.tensor(np.stack(Xs), dtype=torch.float32)
        self.Y    = torch.tensor(np.stack(Ys), dtype=torch.float32)
        self.meta = meta
    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

dataset = FineTuneDataset(panel_norm, ALL_COLS, TARGET_COLS)
print(f"  Total samples : {len(dataset)}")
print(f"  X shape       : {dataset.X.shape}  (19 = 4 states + 10 features + 5 pads)")
print(f"  Y shape       : {dataset.Y.shape}  (k_hat, z_hat, xi_hat at t+1)")

x0, y0   = dataset[0]
cc0, q0  = dataset.meta[0]
print(f"\n  Sample 0 ({cc0}, {q0}):")
print(f"    X states   {x0[:4].numpy().round(3)}")
print(f"    X features {x0[4:14].numpy().round(3)}")
print(f"    X pads     {x0[14:].numpy().round(3)}")
print(f"    Y          {y0.numpy().round(3)}")

print(f"\n  Column index reference:")
for i, col in enumerate(ALL_COLS):
    print(f"    [{i:2d}] {col}")

# ── STEP 7: SAVE ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7 — Saving")
print("=" * 60)

panel_clean.to_csv(OUT_DIR + 'finetune_dataset.csv', index=False)

torch.save({
    'X'           : dataset.X,
    'Y'           : dataset.Y,
    'meta'        : dataset.meta,
    'all_cols'    : ALL_COLS,
    'state_cols'  : STATE_COLS,
    'feature_cols': FEATURE_COLS,
    'target_cols' : TARGET_COLS,
    'pad_cols'    : PAD_COLS,
    'norm_stats'  : norm_stats,
    'params_dict' : params_dict,
    'sol_dict'    : {cc: {k: v for k,v in s.items() if k in ('a_k','a_z','a_xi')}
                     for cc, s in sol_dict.items()},
}, OUT_DIR + 'finetune_dataset.pt')

print(f"  finetune_dataset.csv  ({panel_clean.shape[0]} rows)")
print(f"  finetune_dataset.pt   ({len(dataset)} training pairs)")
print("\nDone.")
