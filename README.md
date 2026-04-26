# Theory-Guided Transfer Learning for Macroeconomic Forecasting

Code and weights for the CPSC 440/550 course project: embedding a Real
Business Cycle (RBC) prior into a residual neural network via synthetic
pretraining and two-stage empirical fine-tuning.

The project tests whether the theory-guided transfer-learning recipe of
Chen et al. (2024) — originally developed for option pricing — extends to
macroeconomic forecasting, where the structural model is a DSGE rather than
Black–Scholes and the real-data sample is orders of magnitude smaller.

Report: `docs/report.pdf` (drop the compiled CPSC 440 PDF here).

## Pipeline

Three training stages on a fixed `19 → 512 → 3` residual MLP (~2.1M params):

1. **Pretrain** on ~12.5M state-transition pairs simulated from a
   calibrated two-shock RBC model, with an economics-informed auxiliary
   loss that penalises violations of the model's policy function and
   capital-accumulation identity.
2. **Stage 1 fine-tune** on a panel of 10 OECD economies (CAN, DNK, FIN,
   JPN, NLD, NOR, SWE, CHE, GBR, USA), 1980Q1–2019Q4, with a sequential
   country-by-country curriculum.
3. **Stage 2 fine-tune** on USA only, 1990Q1–2019Q4, with five financial
   features (BAA–AAA spread, Fed funds rate, log VIX, term spread, S&P 500
   log return) replacing the five zero-pad slots from Stage 1.

Evaluation: AR(1), VAR(2), and Bayesian VAR baselines on a held-out
2014–2019 in-sample window; 2001 and 2008 event-study rollouts; an
out-of-sample window covering 2023Q1–2025Q4 (skipping the COVID structural
break).

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.x, scipy 1.11+, pandas 2.x,
statsmodels 0.14+.

## Reproducing the results

The repo ships with the canonical pretrained, Stage 1 final, and Stage 2
final weights in `weights/`, so you can skip directly to evaluation. To
retrain end-to-end:

### Stage 0: Pretrain

```bash
python pretraining/pretrain.py
# Output: weights/pretrained.pt
# Runtime: ~30 min on a single GPU (50 epochs, batch 2048, 12.5M pairs)
```

`pretraining/pretrain.py` expects a synthetic-data pickle at
`synthetic data/rbc_synthetic_data.pkl` containing 50,000 calibrated RBC
simulations of 250 quarters each. Generate this from the calibrated solver
(see the report appendix for parameter ranges).

Run structural diagnostics (autocorrelations, IRFs, variance checks)
against the pretrained checkpoint:

```bash
python pretraining/diagnostics.py
```

### Stage 1: Cross-country fine-tune

```bash
# Build the cross-country panel from OECD + PWT data
python stage1_finetune/build_dataset.py

# Sequential fine-tune across 10 countries (~5 min on GPU)
python stage1_finetune/finetune.py \
    --weights  weights/pretrained.pt \
    --data     stage1_finetune/finetune_dataset.csv \
    --out_dir  weights/stage1/
```

Output: per-country best checkpoints under `weights/stage1/best_*.pt` and
the final after-USA weights at `weights/stage1/rbc_finetuned_final.pt`.
Per-country validation losses are written to `finetune_summary.csv`.

### Stage 2: USA + financial features

```bash
# Build the 5 US financial-conditions features from FRED + Yahoo Finance
python stage2_finetune/build_features.py

# US-only fine-tune (~30 sec on GPU)
python stage2_finetune/finetune.py \
    --phase1_weights weights/stage1/rbc_finetuned_final.pt \
    --phase1_data    stage1_finetune/finetune_dataset.csv \
    --phase2_data    stage2_finetune/phase2_features.csv \
    --out_dir        weights/stage2/
```

Output: `weights/stage2/rbc_finetuned_phase2_final.pt`.

### Evaluation

```bash
# In-sample (2014Q1–2019Q4) + IRFs + 1/4/8-quarter rollouts + event studies
python evaluation/evaluate_models.py

# Out-of-sample (2023Q1–2025Q4, skipping COVID) vs AR(1)
python evaluation/oos_evaluation.py

# VAR(2) and BVAR (Minnesota prior) baselines
python evaluation/var_benchmark.py
```

> **Note**: the eval scripts are written to read from `/content/` (Colab
> default). If you're running locally, change `DATA_DIR` at the top of each
> file to `data/` (or wherever you put the FRED CSVs).

## Repository structure

```
theory-guided-tl-macro/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── pretraining/
│   ├── pretrain.py             # AdamW, lr=1e-3, batch 2048, 50 epochs, econ-informed loss
│   └── diagnostics.py          # autocorrelation / variance / IRF checks
│
├── stage1_finetune/
│   ├── build_dataset.py        # OECD + PWT → cross-country panel CSV
│   ├── finetune.py             # sequential per-country fine-tune
│   ├── finetune_dataset.csv    # cached output from build_dataset.py
│   ├── finetune_summary.csv    # per-country best validation losses
│   └── data/                   # OECD CSVs + PWT 11.0 + Switzerland unemployment patch
│
├── stage2_finetune/
│   ├── build_features.py       # FRED + Yahoo → 5 US financial features
│   ├── finetune.py             # USA-only fine-tune, lr=1e-5, batch 8
│   ├── phase2_features.csv     # cached output from build_features.py
│   └── data/                   # BAA, AAA, FEDFUNDS, GS10, VIX, sp500
│
├── evaluation/
│   ├── evaluate_models.py      # in-sample + IRFs + multi-step + event studies
│   ├── oos_evaluation.py       # 2023–2025 OOS vs AR(1)
│   ├── var_benchmark.py        # VAR(2) and Minnesota BVAR
│   └── data/                   # OOS FRED CSVs (2019–2026)
│
├── weights/
│   ├── pretrained.pt           # Stage 0 output (~8 MB)
│   ├── stage1_final.pt         # Stage 1 output, after all 10 countries
│   └── stage2_final.pt         # Stage 2 output, used in main results
│
└── docs/
    └── report.pdf              # CPSC 440 report (drop here after compiling)
```

## Architecture

```
Input (19 dims) → LayerNorm → GELU                       # input projection
→ 4 × { Linear → LN → GELU → Dropout(0.1) → Linear → LN  # residual block
        + skip → GELU }
→ Linear (3 dims)                                        # output: (k', z', ξ')
```

Hidden dim 512, ~2.1M parameters. Consumption `c'` is recovered from the
RBC policy function and is not a network output.

## Input layout (19 dimensions, fixed across all stages)

| Slots | Stage 0 (pretrain) | Stage 1 fine-tune | Stage 2 fine-tune |
|---|---|---|---|
| 0–3 | `(k̂, ĉ, ẑ, ξ̂)` state vars | same | same |
| 4–13 | uniform noise | 10 macro features | 10 macro features |
| 14–18 | uniform noise | zero-padded | 5 US financial features |

Same input dimensionality across stages is what lets fine-tuning swap pad
slots for real features without architectural change.

## Hyperparameters

| | Pretrain | Stage 1 | Stage 2 |
|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | `1e-3` | `1e-4` | `1e-5` |
| Weight decay | `1e-4` | `1e-4` | `1e-4` |
| Batch size | 2048 | 32 | 8 |
| Epochs (max) | 50 | 50 per country | 100 |
| Patience | – | 10 | 20 |
| Schedule | Cosine | Cosine | Cosine |
| Grad clip | – | 1.0 | 1.0 |

## Citation

If you use this code, please cite the CPSC 440 report:

```bibtex
@misc{lim2026tgtl,
  author = {Lim, Joshua},
  title  = {Theory-Guided Transfer Learning for Macroeconomic Forecasting:
            Embedding a Real Business Cycle Prior into a Residual Network},
  year   = {2026},
  note   = {CPSC 440/550 course project, University of British Columbia},
  url    = {https://github.com/joshlim01/theory-guided-tl-macro}
}
```

The methodological inspiration is Chen, Cheng, Liu, and Tang (2024),
*Teaching Economics to the Machines*.

## License

MIT — see `LICENSE`.
