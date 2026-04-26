"""
Diagnostic checks on the pretrained RBC ResNet model.
Runs MAE, autocorrelation, IRF, and unconditional variance checks.
Standalone script — no notebook dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import root
import pickle

# ── Rebuild model ──
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.block(x))

class RBCNet(nn.Module):
    def __init__(self, input_dim=19, output_dim=3, hidden_dim=256, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.output_proj(self.res_blocks(self.input_proj(x)))

print("Loading checkpoint...")
ckpt = torch.load("code/Pretraining/weights/zeroes/pretrained_rbc_resnet.pt", weights_only=False)
model = RBCNet(19, 3, 256, 4, 0.1)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ── RBC solver ──
def solve_rbc(alpha, beta, delta, phi, rho_z, rho_xi, ctoy_rat):
    i_over_y = 1.0 - ctoy_rat
    c_over_i = ctoy_rat / i_over_y
    y_over_i = 1.0 / i_over_y
    r = 1.0 / beta - 1.0 + delta
    g_c = -delta * y_over_i * (1.0 - alpha) / (phi + alpha) - delta * c_over_i
    g_k = (1.0 - delta) + delta * y_over_i * alpha * (phi + 1.0) / (phi + alpha)
    g_z = delta * y_over_i * (phi + 1.0) / (phi + alpha)
    g_xi = delta
    M = 1.0 + beta * r * (1.0 - alpha) / (phi + alpha)
    N = beta * r * phi * (1.0 - alpha) / (phi + alpha)
    H = beta * r * (phi + 1.0) / (phi + alpha)

    def residuals(v):
        a_k, a_z, a_xi = v
        Gk = g_c * a_k + g_k
        Gz = g_c * a_z + g_z
        Gxi = g_c * a_xi + g_xi
        return np.array([
            a_k - (M * a_k + N) * Gk,
            a_z - (M * (a_k * Gz + a_z * rho_z) + N * Gz - H * rho_z),
            a_xi - (M * (a_k * Gxi + a_xi * rho_xi) + N * Gxi)
        ])

    guesses = [[0.1,0.1,0.1],[0.1,0.1,-0.1],[0.5,0.1,0.1],[1.0,0.1,0.1],[-0.5,0.1,0.1]]
    candidates = []
    for g in guesses:
        sol = root(residuals, np.array(g, dtype=float))
        if not sol.success:
            continue
        a_k, a_z, a_xi = sol.x
        P = np.array([
            [a_k * g_c, a_k * g_k, a_k * g_z + a_z * rho_z, a_k * g_xi + a_xi * rho_xi],
            [g_c, g_k, g_z, g_xi],
            [0, 0, rho_z, 0],
            [0, 0, 0, rho_xi]
        ], dtype=float)
        eigvals = np.linalg.eigvals(P)
        candidates.append({
            "a_k": a_k, "a_z": a_z, "a_xi": a_xi, "P": P,
            "max_abs_eig": np.max(np.abs(eigvals)),
            "res_norm": np.linalg.norm(residuals([a_k, a_z, a_xi])),
        })

    stable = [c for c in candidates if c["max_abs_eig"] < 1.0 - 1e-8]
    return min(stable, key=lambda x: (x["res_norm"], x["max_abs_eig"]))


# ── Calibration parameters (matching JW's latest) ──
ALPHA = 0.33
BETA = 0.99
DELTA = 0.025
PHI = 1.20
RHO_Z = 0.73
RHO_XI = 0.86
CTOY = 0.64
SIGMA_Z = 0.01
SIGMA_XI = 0.01

sol = solve_rbc(ALPHA, BETA, DELTA, PHI, RHO_Z, RHO_XI, CTOY)
a_k, a_z, a_xi = sol["a_k"], sol["a_z"], sol["a_xi"]
P_true = sol["P"]

print("=" * 60)
print("DIAGNOSTIC REPORT -- PRETRAINED RBC RESNET (ZEROS)")
print("=" * 60)

print("\nPolicy function (analytical ground truth)")
print(f"  a_k  = {a_k:.4f}")
print(f"  a_z  = {a_z:.4f}")
print(f"  a_xi = {a_xi:.4f}")

# ── 1. Full val set MAE ──
print("\nLoading synthetic data for MAE check...")
with open("synthetic data/rbc_synthetic_data.pkl", "rb") as f:
    raw = pickle.load(f)

rng = np.random.RandomState(42)
mae_k, mae_z, mae_xi = [], [], []
for _ in range(5000):
    sim = raw[rng.randint(len(raw))]
    t = rng.randint(len(sim["k_hat"]) - 1)
    x = torch.tensor([[sim["k_hat"][t], sim["c_hat"][t], sim["z_hat"][t], sim["xi_hat"][t]] + [0]*15], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).numpy()[0]
    mae_k.append(abs(pred[0] - sim["k_hat"][t + 1]))
    mae_z.append(abs(pred[1] - sim["z_hat"][t + 1]))
    mae_xi.append(abs(pred[2] - sim["xi_hat"][t + 1]))

mk, mz, mxi = np.mean(mae_k), np.mean(mae_z), np.mean(mae_xi)
print("\n--- 1. Full val set MAE ---")
print(f"  k_{{t+1}}: MAE = {mk:.6f}  {'[OK]' if mk < 0.001 else '[WARN]'}")
print(f"  z_{{t+1}}: MAE = {mz:.6f}  {'[OK]' if mz < 0.005 else '[WARN]'}")
print(f"  xi_{{t+1}}: MAE = {mxi:.6f}  {'[OK]' if mxi < 0.005 else '[WARN]'}")

# ── 2. Autocorrelation check (network rollout) ──
T_rollout = 2000
k_sim = np.zeros(T_rollout)
z_sim = np.zeros(T_rollout)
xi_sim = np.zeros(T_rollout)
c_sim = np.zeros(T_rollout)

rng2 = np.random.default_rng(123)
eps_z = rng2.normal(0, SIGMA_Z, T_rollout)
eps_xi = rng2.normal(0, SIGMA_XI, T_rollout)

for t in range(T_rollout - 1):
    c_sim[t] = a_k * k_sim[t] + a_z * z_sim[t] + a_xi * xi_sim[t]
    x = torch.tensor([[k_sim[t], c_sim[t], z_sim[t], xi_sim[t]] + [0]*15], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).numpy()[0]
    k_sim[t + 1] = pred[0]
    z_sim[t + 1] = pred[1] + eps_z[t + 1]
    xi_sim[t + 1] = pred[2] + eps_xi[t + 1]


def acf1(x):
    x = x - x.mean()
    return np.corrcoef(x[:-1], x[1:])[0, 1]


rho_z_net = acf1(z_sim[200:])
rho_xi_net = acf1(xi_sim[200:])

print("\n--- 2. Autocorrelation check ---")
print(f"  rho_z  -- calibrated: {RHO_Z:.3f} | network rollout: {rho_z_net:.3f}  {'[OK]' if abs(rho_z_net - RHO_Z) < 0.05 else '[WARN]'}")
print(f"  rho_xi -- calibrated: {RHO_XI:.3f} | network rollout: {rho_xi_net:.3f}  {'[OK]' if abs(rho_xi_net - RHO_XI) < 0.05 else '[WARN]'}")

# ── 3. IRF (TFP shock) ──
def irf_network(model, a_k, a_z, a_xi, shock_var, shock_size, T_irf=20):
    k_irf = np.zeros(T_irf)
    z_irf = np.zeros(T_irf)
    xi_irf = np.zeros(T_irf)
    c_irf = np.zeros(T_irf)
    if shock_var == "z":
        z_irf[0] = shock_size
    elif shock_var == "xi":
        xi_irf[0] = shock_size
    for t in range(T_irf - 1):
        c_irf[t] = a_k * k_irf[t] + a_z * z_irf[t] + a_xi * xi_irf[t]
        x = torch.tensor([[k_irf[t], c_irf[t], z_irf[t], xi_irf[t]] + [0]*15], dtype=torch.float32)
        with torch.no_grad():
            pred = model(x).numpy()[0]
        k_irf[t + 1] = pred[0]
        z_irf[t + 1] = pred[1]
        xi_irf[t + 1] = pred[2]
    c_irf[-1] = a_k * k_irf[-1] + a_z * z_irf[-1] + a_xi * xi_irf[-1]
    return k_irf, z_irf, xi_irf, c_irf


def irf_analytical(P, a_k, a_z, a_xi, shock_var, shock_size, T_irf=20):
    s = np.zeros(4)
    if shock_var == "z":
        s[2] = shock_size
    elif shock_var == "xi":
        s[3] = shock_size
    s[0] = a_k * s[1] + a_z * s[2] + a_xi * s[3]
    paths = [s.copy()]
    for t in range(T_irf - 1):
        s = P @ s
        paths.append(s.copy())
    paths = np.array(paths)
    return paths[:, 1], paths[:, 2], paths[:, 3], paths[:, 0]  # k, z, xi, c


T_irf = 20
k_net, z_net, xi_net, c_net = irf_network(model, a_k, a_z, a_xi, "z", SIGMA_Z, T_irf)
k_ana, z_ana, xi_ana, c_ana = irf_analytical(P_true, a_k, a_z, a_xi, "z", SIGMA_Z, T_irf)

print("\n--- 3. Impulse Response Functions (TFP shock) ---")
print(f"  {'t':<4} {'k_net':<12} {'k_ana':<12} {'z_net':<12} {'z_ana':<12} {'c_net':<12} {'c_ana':<12}")
print(f"  {'-' * 72}")
for t in [0, 1, 2, 3, 5, 10, 15, 19]:
    print(f"  {t:<4} {k_net[t]:<12.6f} {k_ana[t]:<12.6f} {z_net[t]:<12.6f} {z_ana[t]:<12.6f} {c_net[t]:<12.6f} {c_ana[t]:<12.6f}")

# ── 4. Unconditional variance ──
var_z_theory = SIGMA_Z**2 / (1 - RHO_Z**2)
var_xi_theory = SIGMA_XI**2 / (1 - RHO_XI**2)
var_z_net = np.var(z_sim[200:])
var_xi_net = np.var(xi_sim[200:])
var_k_net = np.var(k_sim[200:])
mean_z = np.mean(z_sim[200:])
mean_xi = np.mean(xi_sim[200:])
mean_k = np.mean(k_sim[200:])

print("\n--- 4. Unconditional variance ---")
print(f"  Var(z)  -- theory: {var_z_theory:.6f} | network: {var_z_net:.6f}  {'[OK]' if abs(var_z_net / var_z_theory - 1) < 0.3 else '[WARN]'}")
print(f"  Var(xi) -- theory: {var_xi_theory:.6f} | network: {var_xi_net:.6f}  {'[OK]' if abs(var_xi_net / var_xi_theory - 1) < 0.3 else '[WARN]'}")
print(f"  Var(k)  -- network: {var_k_net:.6f} (no closed-form, just eyeball)")
print(f"  Mean(z)  = {mean_z:.6f} (should be ~ 0)")
print(f"  Mean(xi) = {mean_xi:.6f} (should be ~ 0)")
print(f"  Mean(k)  = {mean_k:.6f} (should be ~ 0)")

# ── Summary ──
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("Check all [WARN] flags above before fine-tuning.")
print("IRF shape is the most important -- network and")
print("analytical lines should be close, especially for z.")
