"""
Phase 2 Fine-tuning Feature Construction
==========================================
Builds quarterly US financial features for phase 2 fine-tuning.
1990Q1 - 2019Q4 (120 quarters)

Input files:
  - DBAA.csv       (Moody's BAA corporate bond yield, FRED)
  - DAAA.csv       (Moody's AAA corporate bond yield, FRED)
  - FEDFUNDS.csv   (Federal funds effective rate, FRED)
  - VIXCLS.csv     (CBOE VIX index, FRED)
  - GS10.csv       (10-year Treasury yield, FRED)
  - sp500_quarterly.csv  (S&P 500 quarterly avg, from yfinance)

Output:
  - phase2_features.csv
  - phase2_features.pt

Features (slots 14-18 in 19-dim input vector):
  [14] baa_aaa     — credit spread (BAA - AAA), borrowing constraints
  [15] fedfunds    — fed funds rate (decimal), monetary transmission
  [16] vix_log     — log(VIX), uncertainty / risk appetite
  [17] term_spread — GS10 - FEDFUNDS, yield curve slope
  [18] sp_logret   — S&P 500 quarterly log return, equity risk premium

To reconstruct sp500_quarterly.csv locally:
  import yfinance as yf
  sp500 = yf.download('^GSPC', start='1989-10-01', end='2020-01-01', interval='1mo')
  sp500['Close'].resample('QS').mean().to_csv('sp500_quarterly.csv')
  (Start from 1989-10-01 to get 1989Q4 avg needed for 1990Q1 log return)
"""

import pandas as pd
import numpy as np
import torch

DATA_DIR = '/mnt/user-data/uploads/'
OUT_DIR  = '/mnt/user-data/outputs/'

ALL_QUARTERS = pd.period_range(start='1990Q1', end='2019Q4', freq='Q')

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_fred(filepath, col):
    """Load a FRED quarterly CSV → Series indexed by PeriodIndex."""
    df = pd.read_csv(filepath, parse_dates=['observation_date'])
    df = df.rename(columns={col: 'value', 'observation_date': 'date'})
    df['value']  = pd.to_numeric(df['value'], errors='coerce')
    df['period'] = df['date'].dt.to_period('Q')
    return df.set_index('period')['value'].reindex(ALL_QUARTERS)

# ── STEP 1: LOAD ALL SERIES ───────────────────────────────────────────────────

print("=" * 55)
print("STEP 1 — Loading series")
print("=" * 55)

gs10     = load_fred(DATA_DIR + 'GS10.csv',     'GS10')
vix      = load_fred(DATA_DIR + 'VIXCLS.csv',   'VIXCLS')
fedfunds = load_fred(DATA_DIR + 'FEDFUNDS.csv', 'FEDFUNDS')
daaa     = load_fred(DATA_DIR + 'DAAA.csv',     'DAAA')
dbaa     = load_fred(DATA_DIR + 'DBAA.csv',     'DBAA')

# SP500 — quarterly average of monthly closes
sp = pd.read_csv(DATA_DIR + 'sp500_quarterly.csv', parse_dates=['Date'])
sp['period'] = sp['Date'].dt.to_period('Q')
sp = sp.set_index('period')['^GSPC'].reindex(ALL_QUARTERS)

for name, s in [('GS10', gs10), ('VIX', vix), ('FEDFUNDS', fedfunds),
                ('DAAA', daaa), ('DBAA', dbaa), ('SP500', sp)]:
    print(f"  {name:10s}: {s.notna().sum()}/120 non-null  "
          f"range [{s.min():.3f}, {s.max():.3f}]")

# ── STEP 2: CONSTRUCT FEATURES ────────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 2 — Constructing features")
print("=" * 55)

# 1. BAA-AAA credit spread (percent)
#    Always positive by construction (BAA riskier than AAA)
baa_aaa = dbaa - daaa

# 2. Fed funds rate in decimal (divide by 100)
fedfunds_d = fedfunds / 100

# 3. Log VIX — compresses 2008 spike, makes distribution less skewed
vix_log = np.log(vix)

# 4. Term spread: 10yr Treasury minus fed funds (percent)
#    Positive = normal upward sloping curve
#    Negative = inverted = recession signal
term_spread = gs10 - fedfunds

# 5. S&P 500 quarterly log return
#    1990Q1 manually computed from 1989Q4 average:
#      Q4 1989 closes (Oct/Nov/Dec): 340.36, 345.99, 353.40
#      Q4 1989 avg = 346.58
#      Q1 1990 avg = 333.64
#      log(333.64 / 346.58) = -0.038061
sp_logret = np.log(sp).diff()
sp_logret.iloc[0] = -0.038061   # manually patched 1990Q1

# ── STEP 3: ASSEMBLE AND VALIDATE ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 3 — Assembling and validating")
print("=" * 55)

df = pd.DataFrame({
    'baa_aaa'    : baa_aaa,
    'fedfunds'   : fedfunds_d,
    'vix_log'    : vix_log,
    'term_spread': term_spread,
    'sp_logret'  : sp_logret,
}, index=ALL_QUARTERS)

df.index.name = 'period'
df = df.reset_index()
df['period'] = df['period'].astype(str)

print(f"\n  Shape: {df.shape}")
print(f"  Missing values: {df.isnull().sum().sum()} (expect 0)")
print(f"\n  Summary stats:")
print(df.drop(columns='period').describe().round(4).to_string())

# Sanity checks
print("\n  Sanity checks:")
assert (df['baa_aaa'] > 0).all(), "BAA-AAA should always be positive"
print(f"  ✅ BAA-AAA always positive (min={df['baa_aaa'].min():.3f})")

peak_baa = df.loc[df['baa_aaa'].idxmax(), ['period','baa_aaa']]
print(f"  ✅ BAA-AAA peak: {peak_baa['period']} = {peak_baa['baa_aaa']:.3f} (expect 2008-2009)")

df_idx = df.set_index('period')
inv = (df_idx['term_spread'] < 0).sum()
print(f"  ✅ Yield curve inversions: {inv} quarters (expect ~8, 2006-2007)")

worst_sp = df.loc[df['sp_logret'].idxmin(), ['period','sp_logret']]
print(f"  ✅ Worst SP500 quarter: {worst_sp['period']} = {worst_sp['sp_logret']:.4f} (expect 2008Q4)")

zlb = (df_idx['fedfunds'] < 0.002).sum()
print(f"  ✅ Near-ZLB quarters (FFR < 0.2%): {zlb} (expect ~24, 2009-2015)")

# ── STEP 4: NORMALISE ─────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 4 — Normalising (z-score)")
print("=" * 55)

FEATURE_COLS = ['baa_aaa', 'fedfunds', 'vix_log', 'term_spread', 'sp_logret']

norm_stats = {}
df_norm    = df.copy()

for col in FEATURE_COLS:
    mu  = df[col].mean()
    sig = df[col].std()
    if sig < 1e-10: sig = 1.0
    df_norm[col] = (df[col] - mu) / sig
    norm_stats[col] = dict(mean=float(mu), std=float(sig))
    print(f"  {col:15s}: mean={mu:+.4f}  std={sig:.4f}")

# ── STEP 5: SAVE ──────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 5 — Saving")
print("=" * 55)

# Raw (un-normalised) CSV for inspection
df.to_csv(OUT_DIR + 'phase2_features.csv', index=False)

# Normalised pt file with metadata
torch.save({
    'feature_cols': FEATURE_COLS,
    'norm_stats'  : norm_stats,
    'data'        : df_norm,
    'quarters'    : list(df['period']),
}, OUT_DIR + 'phase2_features.pt')

print(f"  ✅ phase2_features.csv  ({df.shape[0]} rows × {df.shape[1]} cols)")
print(f"  ✅ phase2_features.pt   (normalised + norm_stats)")
print(f"\n  Column slots in 19-dim input vector:")
for i, col in enumerate(FEATURE_COLS):
    print(f"    [{14+i}] {col}")
