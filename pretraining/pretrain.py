"""
Pretrain a ResNet MLP on synthetic RBC data.
Learns the state transition dynamics: given current state variables,
predict next-period states {k_{t+1}, z_{t+1}, xi_{t+1}}.
Uses economics-informed loss (policy function + capital accumulation constraints).
Saves pretrained checkpoint to code/Pretraining/weights/zeroes/pretrained_rbc_resnet.pt.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.optimize import root

# ──────────────────────────────────────────────
# 1. RBC model solver (from CalibrationActual_V2)
# ──────────────────────────────────────────────
def solve_rbc_two_shock_model(alpha, beta, delta, phi, rho_z, rho_xi, ctoy_rat):
    i_over_y = 1.0 - ctoy_rat
    if i_over_y <= 0:
        raise ValueError("Need 1 - ctoy_rat > 0.")

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
        eq_k = a_k - (M * a_k + N) * Gk
        eq_z = a_z - (M * (a_k * Gz + a_z * rho_z) + N * Gz - H * rho_z)
        eq_xi = a_xi - (M * (a_k * Gxi + a_xi * rho_xi) + N * Gxi)
        return np.array([eq_k, eq_z, eq_xi])

    guesses = [
        [0.1, 0.1, 0.1], [0.1, 0.1, -0.1], [0.1, -0.1, 0.1], [-0.1, 0.1, 0.1],
        [0.5, 0.1, 0.1], [0.5, -0.1, 0.1], [1.0, 0.1, 0.1], [-0.5, 0.1, 0.1],
    ]
    candidates = []
    for guess in guesses:
        sol = root(residuals, np.array(guess, dtype=float))
        if not sol.success:
            continue
        a_k, a_z, a_xi = sol.x
        P = np.array([
            [a_k * g_c, a_k * g_k, a_k * g_z + a_z * rho_z, a_k * g_xi + a_xi * rho_xi],
            [g_c, g_k, g_z, g_xi],
            [0.0, 0.0, rho_z, 0.0],
            [0.0, 0.0, 0.0, rho_xi]
        ], dtype=float)
        eigvals = np.linalg.eigvals(P)
        max_abs_eig = np.max(np.abs(eigvals))
        res_norm = np.linalg.norm(residuals([a_k, a_z, a_xi]))
        candidates.append({
            "a_k": a_k, "a_z": a_z, "a_xi": a_xi,
            "max_abs_eig": max_abs_eig, "res_norm": res_norm,
        })

    stable = [c for c in candidates if c["max_abs_eig"] < 1.0 - 1e-8]
    if not stable:
        raise RuntimeError("No stable solution found.")
    return min(stable, key=lambda x: (x["res_norm"], x["max_abs_eig"]))


# ──────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────
class RBCDataset(Dataset):
    def __init__(self, data_list):
        Xs, Ys, auxs = [], [], []

        for idx, sim in enumerate(data_list):
            if idx % 5000 == 0:
                print(f"  Processing simulation {idx}/{len(data_list)}...")
            p = sim["params"]
            sol = solve_rbc_two_shock_model(
                p["alpha"], p["beta"], p["delta"], p["phi"],
                p["rho_z"], p["rho_xi"], p["ctoy"]
            )
            a_k, a_z, a_xi = sol["a_k"], sol["a_z"], sol["a_xi"]

            k = sim["k_hat"]
            c = sim["c_hat"]
            z = sim["z_hat"]
            xi = sim["xi_hat"]
            i_ = sim["i_hat"]
            T = len(k)

            for t in range(T - 1):
                # Input: [k, c, z, xi] + 15 zeros
                x = np.array([k[t], c[t], z[t], xi[t]] + [0]*15, dtype=np.float32)
                # Output: next-period state
                y = np.array([k[t + 1], z[t + 1], xi[t + 1]], dtype=np.float32)
                # Aux: everything needed for economics loss
                aux = np.array([
                    k[t], i_[t], xi[t], p["delta"],
                    c[t + 1], i_[t + 1],
                    a_k, a_z, a_xi,
                    p["alpha"], p["phi"], p["ctoy"]
                ], dtype=np.float32)
                Xs.append(x)
                Ys.append(y)
                auxs.append(aux)

        self.X = torch.tensor(np.stack(Xs), dtype=torch.float32)
        self.Y = torch.tensor(np.stack(Ys), dtype=torch.float32)
        self.aux = torch.tensor(np.stack(auxs), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.aux[idx]


# ──────────────────────────────────────────────
# 3. Economics-informed loss
# ──────────────────────────────────────────────
def economics_loss(pred, target, aux, lambda0=1.0, lambda1=1.0):
    k_t = aux[:, 0]
    i_t = aux[:, 1]
    xi_t = aux[:, 2]
    delta = aux[:, 3]
    c_next = aux[:, 4]
    i_next = aux[:, 5]
    a_k = aux[:, 6]
    a_z = aux[:, 7]
    a_xi = aux[:, 8]
    alpha = aux[:, 9]
    phi = aux[:, 10]
    ctoy = aux[:, 11]

    k_pred = pred[:, 0]
    z_pred = pred[:, 1]
    xi_pred = pred[:, 2]

    # Predicted c_{t+1} from policy function
    c_pred_next = a_k * k_pred + a_z * z_pred + a_xi * xi_pred

    # Predicted y_{t+1} and i_{t+1} from model equations
    i_over_y = 1.0 - ctoy
    c_over_i = ctoy / i_over_y
    y_over_i = 1.0 / i_over_y

    y_pred_next = ((phi + 1) * z_pred + alpha * (phi + 1) * k_pred - (1 - alpha) * c_pred_next) / (phi + alpha)
    i_pred_next = y_over_i * y_pred_next - c_over_i * c_pred_next

    # L_controls: MAE on c and i at t+1
    L_controls = torch.mean(torch.abs(c_pred_next - c_next) + torch.abs(i_pred_next - i_next))

    # L_dynamics: capital accumulation residual
    k_next_implied = (1 - delta) * k_t + delta * i_t + delta * xi_t
    L_dynamics = torch.mean(torch.abs(k_pred - k_next_implied))

    L_total = lambda0 * L_controls + lambda1 * L_dynamics
    return L_total, L_controls.item(), L_dynamics.item()


# ──────────────────────────────────────────────
# 4. ResNet MLP architecture
# ──────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class RBCNet(nn.Module):
    def __init__(self, input_dim=19, output_dim=3, hidden_dim=512, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_proj(x)


# ──────────────────────────────────────────────
# 5. Load data and build dataset
# ──────────────────────────────────────────────
print("Loading synthetic data...")
with open("synthetic data/rbc_synthetic_data.pkl", "rb") as f:
    raw_data = pickle.load(f)
print(f"Loaded {len(raw_data)} simulations")

print("Building dataset (solving RBC model for each simulation)...")
dataset = RBCDataset(raw_data)
print(f"Total samples: {len(dataset):,}")

# 90/10 split
n_val = int(len(dataset) * 0.1)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
print(f"Train: {n_train:,}, Val: {n_val:,}")

# ──────────────────────────────────────────────
# 6. Training
# ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = RBCNet(input_dim=19, output_dim=3, hidden_dim=512, n_layers=4, dropout=0.1).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

EPOCHS = 50
LR = 1e-3
LAMBDA0 = 1.0
LAMBDA1 = 1.0

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_loss = float("inf")
train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    for x_batch, y_batch, aux_batch in train_loader:
        x_batch, y_batch, aux_batch = x_batch.to(device), y_batch.to(device), aux_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch)
        loss, _, _ = economics_loss(pred, y_batch, aux_batch, LAMBDA0, LAMBDA1)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch, aux_batch in val_loader:
            x_batch, y_batch, aux_batch = x_batch.to(device), y_batch.to(device), aux_batch.to(device)
            pred = model(x_batch)
            loss, _, _ = economics_loss(pred, y_batch, aux_batch, LAMBDA0, LAMBDA1)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

print(f"\nBest validation loss: {best_val_loss:.6f}")

# ──────────────────────────────────────────────
# 7. Save checkpoint
# ──────────────────────────────────────────────
checkpoint = {
    "model_state_dict": best_state,
    "architecture": {
        "input_dim": 19,
        "output_dim": 3,
        "hidden_dim": 512,
        "n_layers": 4,
        "dropout": 0.1,
        "activation": "GELU",
    },
    "feature_names": ["k_hat_t", "c_hat_t", "z_hat_t", "xi_hat_t"] +
                     [f"zero_{i}" for i in range(1, 16)],
    "target_names": ["k_hat_t1", "z_hat_t1", "xi_hat_t1"],
    "history": {
        "train_loss": train_losses,
        "val_loss": val_losses,
    },
}

CHECKPOINT_PATH = "code/Pretraining/weights/zeroes/pretrained_rbc_resnet.pt"
torch.save(checkpoint, CHECKPOINT_PATH)
print(f"Saved {CHECKPOINT_PATH}")

# ──────────────────────────────────────────────
# 8. Sanity check: MAE on validation batch
# ──────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    x_batch, y_batch, aux_batch = next(iter(val_loader))
    x_batch = x_batch.to(device)
    pred = model(x_batch).cpu()

for i, name in enumerate(["k_{t+1}", "z_{t+1}", "xi_{t+1}"]):
    actual = y_batch[:, i]
    predicted = pred[:, i]
    mae = torch.mean(torch.abs(predicted - actual)).item()
    print(f"{name}: MAE = {mae:.6f} | Actual mean = {actual.mean().item():.6f} | Pred mean = {predicted.mean().item():.6f}")
