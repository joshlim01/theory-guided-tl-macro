"""
finetune_rbc.py
===============
Fine-tunes the pretrained RBCNet (19-input, 3-output) on real-world country data.

Strategy
--------
- Iterates over countries in chronological order of their data spans.
- For each country:
    - Splits data 80% train / 20% test  (test = most recent quarters — no leakage).
    - Runs up to MAX_EPOCHS with early stopping (patience = EARLY_STOP_PATIENCE).
    - Saves a checkpoint every epoch and keeps the best val-loss weights per country.
- The fine-tuned weights from country N are the starting point for country N+1
  (sequential/curriculum transfer — later countries build on earlier adaptations).

Input columns expected in finetune_dataset.csv
-----------------------------------------------
  period, k_hat, c_hat, z_hat, xi_hat,
  gdp_growth_level, gdp_growth_vol,
  ulc_growth_level, ulc_growth_vol,
  inflation_level,  inflation_vol,
  unemp_level,      unemp_vol,
  spread_level,     spread_vol,
  countrycode,
  pad_1, pad_2, pad_3, pad_4, pad_5

Network input vector (19-dim)
------------------------------
  [k_hat, c_hat, z_hat, xi_hat,
   gdp_growth_level, gdp_growth_vol,
   ulc_growth_level, ulc_growth_vol,
   inflation_level,  inflation_vol,
   unemp_level,      unemp_vol,
   spread_level,     spread_vol,
   pad_1, pad_2, pad_3, pad_4, pad_5]

Target vector (3-dim)
----------------------
  [k_hat_{t+1}, z_hat_{t+1}, xi_hat_{t+1}]
  (next-period values shifted from the same country timeline)

Usage
-----
  python finetune_rbc.py \
    --weights  /path/to/rbc_pretrained_15.pt \
    --data     /path/to/finetune_dataset.csv  \
    --out_dir  ./checkpoints
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
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune RBCNet country by country")
    p.add_argument("--weights",  required=True,
                   help="Path to pretrained .pt file (state dict)")
    p.add_argument("--data",     required=True,
                   help="Path to finetune_dataset.csv")
    p.add_argument("--out_dir",  default="./checkpoints",
                   help="Directory for checkpoint files (created if absent)")
    p.add_argument("--epochs",   type=int, default=50)
    p.add_argument("--patience", type=int, default=10,
                   help="Early-stopping patience (epochs without val improvement)")
    p.add_argument("--lr",       type=float, default=1e-4,
                   help="Initial learning rate (keep low for fine-tuning)")
    p.add_argument("--batch",    type=int, default=32,
                   help="Batch size (small dataset — 32 is fine)")
    p.add_argument("--train_frac", type=float, default=0.8,
                   help="Fraction of each country's timeline used for training")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   default="auto",
                   help="'auto', 'cpu', 'cuda', or 'mps'")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Network  (identical to pretraining definition)
# ---------------------------------------------------------------------------
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
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "k_hat", "c_hat", "z_hat", "xi_hat",
    "gdp_growth_level", "gdp_growth_vol",
    "ulc_growth_level",  "ulc_growth_vol",
    "inflation_level",   "inflation_vol",
    "unemp_level",       "unemp_vol",
    "spread_level",      "spread_vol",
    "pad_1", "pad_2", "pad_3", "pad_4", "pad_5",
]
TARGET_COLS = ["k_hat", "z_hat", "xi_hat"]  # next-period versions


class CountryDataset(Dataset):
    """
    Builds (X_t, Y_{t+1}) pairs from a single country's chronological data.
    X  : 19-dim input at time t
    Y  : [k_hat, z_hat, xi_hat] at time t+1
    """
    def __init__(self, df_country: pd.DataFrame):
        df = df_country.sort_values("period").reset_index(drop=True)

        X_all = df[FEATURE_COLS].values.astype(np.float32)    # (T, 19)
        Y_all = df[TARGET_COLS].values.astype(np.float32)     # (T,  3)

        # Pair t -> t+1  (drop last row which has no next period)
        self.X = torch.tensor(X_all[:-1])   # (T-1, 19)
        self.Y = torch.tensor(Y_all[1:])    # (T-1,  3)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ---------------------------------------------------------------------------
# Country ordering  (chronological by earliest period in the dataset)
# ---------------------------------------------------------------------------
def get_country_order(df: pd.DataFrame) -> list[str]:
    """
    Sort countries by their earliest 'period' entry so we fine-tune in
    the order that the data history begins — more data-rich / earlier
    countries first to anchor the model before less-covered ones.
    """
    earliest = df.groupby("countrycode")["period"].min()
    # period strings like '1980Q1' sort lexicographically correctly
    return earliest.sort_values().index.tolist()


# ---------------------------------------------------------------------------
# Training helpers
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


def save_checkpoint(state: dict, path: str):
    """Atomic-ish save: write to tmp then rename."""
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

    # ---- device ----
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- output directory ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load data ----
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} rows | {df['countrycode'].nunique()} countries")

    countries = get_country_order(df)
    print(f"Country order: {countries}\n")

    # ---- load pretrained weights ----
    model = RBCNet(input_dim=19, hidden_dim=512).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    print(f"Pretrained weights loaded from: {args.weights}\n")

    criterion = nn.MSELoss()

    # ---- summary log ----
    summary_rows = []

    # ===================================================================
    # Country loop
    # ===================================================================
    for country_idx, country in enumerate(countries):
        print("=" * 60)
        print(f"[{country_idx+1}/{len(countries)}]  Country: {country}")
        print("=" * 60)

        df_c = df[df["countrycode"] == country].copy()
        full_ds = CountryDataset(df_c)
        T = len(full_ds)

        # chronological 80/20 split  (no shuffling across time)
        n_train = int(T * args.train_frac)
        n_val   = T - n_train
        # Use Subset to preserve order rather than random_split
        from torch.utils.data import Subset
        train_ds = Subset(full_ds, list(range(n_train)))
        val_ds   = Subset(full_ds, list(range(n_train, T)))

        print(f"  Samples — train: {n_train}  |  val: {n_val}")

        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                  shuffle=True,  drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                                  shuffle=False, drop_last=False)

        # Optimizer & scheduler — reset fresh for each country
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                      eta_min=args.lr * 0.01)

        best_val_loss   = float("inf")
        best_epoch      = 0
        epochs_no_improve = 0
        country_start   = time.time()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_loss = run_epoch(model, train_loader, criterion,
                                   optimizer, device, train=True)
            val_loss   = run_epoch(model, val_loader, criterion,
                                   optimizer, device, train=False)
            scheduler.step()

            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]

            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"train: {train_loss:.6f}  val: {val_loss:.6f} | "
                  f"lr: {lr_now:.2e} | {elapsed:.1f}s")

            # ---- per-epoch checkpoint ----
            epoch_ckpt = out_dir / f"ckpt_{country}_ep{epoch:03d}.pt"
            save_checkpoint({
                "epoch":       epoch,
                "country":     country,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "train_loss":  train_loss,
                "val_loss":    val_loss,
            }, str(epoch_ckpt))

            # ---- best-model checkpoint (overwrites) ----
            if val_loss < best_val_loss:
                best_val_loss   = val_loss
                best_epoch      = epoch
                epochs_no_improve = 0
                best_ckpt = out_dir / f"best_{country}.pt"
                save_checkpoint({
                    "epoch":       epoch,
                    "country":     country,
                    "model_state": model.state_dict(),
                    "val_loss":    val_loss,
                }, str(best_ckpt))
                print(f"  ✓ New best val loss → saved to {best_ckpt.name}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"  Early stop: no improvement for {args.patience} epochs.")
                    break

        # Reload best weights before moving to next country
        best_state = torch.load(out_dir / f"best_{country}.pt",
                                map_location=device)
        model.load_state_dict(best_state["model_state"])

        country_time = time.time() - country_start
        print(f"  Best epoch: {best_epoch}  |  best val loss: {best_val_loss:.6f}")
        print(f"  Country {country} done in {country_time/60:.1f} min\n")

        summary_rows.append({
            "country":        country,
            "best_epoch":     best_epoch,
            "best_val_loss":  best_val_loss,
            "n_train":        n_train,
            "n_val":          n_val,
            "minutes":        round(country_time / 60, 2),
        })

    # ===================================================================
    # Final save — weights after all countries
    # ===================================================================
    final_path = out_dir / "rbc_finetuned_final.pt"
    torch.save(model.state_dict(), str(final_path))
    print(f"\nFinal model saved → {final_path}")

    # ---- summary table ----
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "finetune_summary.csv"
    summary_df.to_csv(str(summary_csv), index=False)
    print(f"Summary saved   → {summary_csv}")
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
