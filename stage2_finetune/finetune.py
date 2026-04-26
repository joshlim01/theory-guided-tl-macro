"""
finetune_phase2.py
==================
Phase 2 fine-tuning: USA only, 1990Q1–2019Q4.

Starts from the phase 1 weights (rbc_finetuned_final.pt) and trains on
US data with 5 new real features replacing pad_1–pad_5 (positions 14–18):

  Input vector (19-dim):
  [k_hat, c_hat, z_hat, xi_hat,           ← 0–3   state vars (unchanged)
   gdp_growth_level, gdp_growth_vol,       ← 4–5   from phase 1 CSV
   ulc_growth_level, ulc_growth_vol,       ← 6–7
   inflation_level,  inflation_vol,        ← 8–9
   unemp_level,      unemp_vol,            ← 10–11
   spread_level,     spread_vol,           ← 12–13
   baa_aaa, fedfunds, vix_log,             ← 14–16  NEW — US financial features
   term_spread, sp_logret]                 ← 17–18  NEW

Train/val split: 80/20 chronological (96 train / 24 val).
Only 1990Q1–2019Q4 quarters used — no zero-padded pre-1990 rows.

Usage:
  python finetune_phase2.py \
    --phase1_weights /path/to/rbc_finetuned_final.pt \
    --phase1_data    /path/to/finetune_dataset.csv \
    --phase2_data    /path/to/phase2_features.csv \
    --out_dir        ./checkpoints_phase2 \
    --epochs         50 \
    --patience       10 \
    --lr             5e-5 \
    --batch          16
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 fine-tune: USA + new financial features")
    p.add_argument("--phase1_weights", required=True,
                   help="Path to rbc_finetuned_final.pt from phase 1")
    p.add_argument("--phase1_data",    required=True,
                   help="Path to finetune_dataset.csv (phase 1 CSV)")
    p.add_argument("--phase2_data",    required=True,
                   help="Path to phase2_features.csv (new US financial features)")
    p.add_argument("--out_dir",        default="./checkpoints_phase2")
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--patience",       type=int,   default=20)
    p.add_argument("--lr",             type=float, default=1e-5)
    p.add_argument("--batch",          type=int,   default=8)
    p.add_argument("--train_frac",     type=float, default=0.8)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Network (identical to phase 1)
# ---------------------------------------------------------------------------
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
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.output_proj(self.res_blocks(self.input_proj(x)))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Phase 1 feature columns (positions 0–13, pad cols dropped — replaced below)
PHASE1_COLS = [
    "k_hat", "c_hat", "z_hat", "xi_hat",
    "gdp_growth_level", "gdp_growth_vol",
    "ulc_growth_level",  "ulc_growth_vol",
    "inflation_level",   "inflation_vol",
    "unemp_level",       "unemp_vol",
    "spread_level",      "spread_vol",
]

# Phase 2 new features (positions 14–18, replace pad_1–pad_5)
PHASE2_COLS = ["baa_aaa", "fedfunds", "vix_log", "term_spread", "sp_logret"]

TARGET_COLS = ["k_hat", "z_hat", "xi_hat"]


class USAPhase2Dataset(Dataset):
    """
    Merges phase 1 USA rows (1990Q1–2019Q4) with phase 2 financial features,
    builds (X_t, Y_{t+1}) pairs in chronological order.

    X shape: (T-1, 19)  — 14 phase1 features + 5 new US features
    Y shape: (T-1,  3)  — [k_hat_{t+1}, z_hat_{t+1}, xi_hat_{t+1}]
    """
    def __init__(self, df_usa: pd.DataFrame, df_p2: pd.DataFrame):
        # Merge on period — inner join keeps only 1990Q1–2019Q4
        # NOTE: TARGET_COLS (k_hat, z_hat, xi_hat) are already in PHASE1_COLS
        # so we select them separately to avoid duplicating columns in the merge
        left_cols = list(dict.fromkeys(["period"] + PHASE1_COLS + TARGET_COLS))
        df = (df_usa[left_cols]
              .merge(df_p2[["period"] + PHASE2_COLS], on="period", how="inner")
              .sort_values("period")
              .reset_index(drop=True))

        print(f"  Merged dataset: {len(df)} quarters "
              f"({df['period'].iloc[0]} → {df['period'].iloc[-1]})")

        # Build feature matrix: phase1 cols (0–13) + phase2 cols (14–18)
        X_all = np.hstack([
            df[PHASE1_COLS].values.astype(np.float32),   # (T, 14)
            df[PHASE2_COLS].values.astype(np.float32),   # (T,  5)
        ])                                                # (T, 19)
        Y_all = df[TARGET_COLS].values.astype(np.float32)  # (T, 3)

        # Chronological (t → t+1) pairs
        self.X = torch.tensor(X_all[:-1])   # (T-1, 19)
        self.Y = torch.tensor(Y_all[1:])    # (T-1,  3)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    n = 0
    with torch.set_grad_enabled(train):
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = criterion(pred, Y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(X)
            n += len(X)
    return total_loss / n


def save_checkpoint(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():        device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                                device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    df_all = pd.read_csv(args.phase1_data)
    df_usa = df_all[df_all["countrycode"] == "USA"].copy()
    df_p2  = pd.read_csv(args.phase2_data)
    print(f"USA rows (phase 1): {len(df_usa)}")
    print(f"Phase 2 feature rows: {len(df_p2)}")

    # ── Build dataset ──
    dataset = USAPhase2Dataset(df_usa, df_p2)
    T = len(dataset)

    n_train = int(T * args.train_frac)
    n_val   = T - n_train
    train_ds = Subset(dataset, list(range(n_train)))
    val_ds   = Subset(dataset, list(range(n_train, T)))

    print(f"\nSplit — train: {n_train} quarters  |  val: {n_val} quarters")
    print(f"  (approx train: {dataset.X[:n_train, :].shape[0]} samples, "
          f"val: {n_val} samples)\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, drop_last=False)

    # ── Load phase 1 weights ──
    model = RBCNet(input_dim=19, hidden_dim=512).to(device)
    model.load_state_dict(torch.load(args.phase1_weights, map_location=device))
    print(f"Phase 1 weights loaded from: {args.phase1_weights}")

    # ── Optimizer & scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=args.lr * 0.01)
    criterion = nn.MSELoss()

    best_val_loss     = float("inf")
    best_epoch        = 0
    epochs_no_improve = 0
    start             = time.time()

    print("=" * 60)
    print("Phase 2 fine-tuning: USA")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion,
                               optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion,
                               optimizer, device, train=False)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train: {train_loss:.6f}  val: {val_loss:.6f} | "
              f"lr: {lr_now:.2e} | {elapsed:.1f}s")

        # Per-epoch checkpoint
        epoch_ckpt = out_dir / f"ckpt_USA_phase2_ep{epoch:03d}.pt"
        save_checkpoint({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "train_loss":  train_loss,
            "val_loss":    val_loss,
        }, str(epoch_ckpt))

        # Best model
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_epoch        = epoch
            epochs_no_improve = 0
            best_ckpt = out_dir / "best_USA_phase2.pt"
            save_checkpoint({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
            }, str(best_ckpt))
            print(f"  ✓ New best val loss → saved to {best_ckpt.name}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stop: no improvement for {args.patience} epochs.")
                break

    total_time = time.time() - start

    # Reload best weights and save as final
    best_state = torch.load(out_dir / "best_USA_phase2.pt", map_location=device)
    model.load_state_dict(best_state["model_state"])
    final_path = out_dir / "rbc_finetuned_phase2_final.pt"
    torch.save(model.state_dict(), str(final_path))

    print(f"\n{'='*60}")
    print(f"Done. Best epoch: {best_epoch}  |  best val loss: {best_val_loss:.6f}")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Final weights → {final_path}")


if __name__ == "__main__":
    main()
