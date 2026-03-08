# Spectral Bridge — Recovering Lost Audio with an Attentive Neural Process

## The Challenge

> *In the winter of 1974, The New Yardbirds retreated to Headley Grange to record their final
> experimental album, codenamed **Spectral Graffiti**. The tapes were never released. Seized by
> creditors and buried in a non-climate-controlled London basement, they developed catastrophic
> Sticky Shed Syndrome — the magnetic oxide physically separated from the polyester reels, leaving
> microscopic gaps of absolute signal loss.*

**Spectral Bridge** is a few-shot signal in-painting challenge hosted at Cognizance, IIT Roorkee.
Given 80,000 partially observed audio clips (1 kHz sampling over 100 ms), reconstructing the
missing waveform fragments using only the surviving context points.

This is **not** a denoising task — the signal isn't noisy, it's **completely missing** in the gaps.
And each clip is a different acoustic event (bass pluck, cymbal crash, synth drone), so the model
must infer the instrument's spectral signature on the fly from whatever fragments survived.

**Constraint**: amortized inference only — a single trained model that predicts all targets in one
forward pass. No per-sample fitting loops allowed.

**Metric**: Mean Squared Error (MSE) on target points (where `Is_Context == 0`).

---

## Approach: Attentive Neural Process (ANP)

A neural process is a meta-learning architecture purpose-built for this kind of problem: take a
set of observed points (context), and predict values at arbitrary query locations (targets) in a
single forward pass.

I chose the **Attentive** variant because vanilla neural processes compress all context into a
single vector, which loses positional information. Cross-attention lets each target point
selectively attend to the most relevant context fragments — critical when different parts of a
waveform have different characteristics.

### Architecture

```
  Context (time, value)  ──>  Fourier Encoding  ──>  MLP Encoder  ──>  Keys, Values ─┐
                                                                                      │
  Target timestamps  ────>  Fourier Encoding  ─────────────────────>  Queries ────────┤
                                                                                      │
                                                                             Cross-Attention
                                                                                      │
                                                                   [target_enc, attn_output]
                                                                                      │
                                                                               MLP Decoder
                                                                                      │
                                                                           Predicted values
```

### Why These Design Choices

| Component | Choice | Reasoning |
|---|---|---|
| Time encoding | Fourier (32 frequency bands) | Audio signals are inherently periodic; sin/cos features at multiple frequencies let the model represent harmonics naturally |
| Context encoding | Per-point MLP (no pooling) | Preserves individual point identity — each context observation keeps its own embedding |
| Aggregation | Multi-head cross-attention (4 heads) | Each target point gets a custom, weighted summary of context; handles variable context sizes |
| Decoder | MLP (4 layers, 128 hidden) | Maps (target time encoding + attention output) to a scalar voltage prediction |
| Optimiser | AdamW (lr=1e-3, wd=1e-4) | Decoupled weight decay works well with LayerNorm-based architectures |
| LR schedule | Cosine annealing over 80 epochs | Smooth decay; combined with early stopping (patience=10) |
| Augmentation | Stochastic context/target reshuffle (70%) | Prevents memorising the fixed CSV split; teaches the model to work at any observation density |

---

## Project Structure

```
Spectral Bridge/
├── spectral_bridge_anp_notebook.ipynb   # Full solution — code + explanation (illustrated blog)
├── best_anp.pt                          # Saved model checkpoint (best validation MSE)
├── train.csv                            # Training data (80,000 clips, ~8M rows)
├── submission.csv                       # Generated predictions (after inference)
└── README.md                            # This file
```

---

## Requirements

Python 3.8+ with:

```bash
pip install torch pandas numpy matplotlib tqdm scikit-learn
```

A CUDA GPU speeds up training significantly but isn't required. The code auto-detects the best
available device (CUDA > Apple MPS > CPU).

---

## How to Run

Open `spectral_bridge_anp_notebook.ipynb` in Jupyter and **run all cells top-to-bottom**.

The final cell calls `main(CFG)` which executes the full pipeline:

1. **Load** `train.csv` and print exploratory statistics
2. **Split** samples into train/val (90/10 by `Sample_ID` — no data leakage)
3. **Train** the ANP with cosine LR + early stopping
4. **Evaluate** on the validation set with MSE and prediction plots
5. **Infer** on test data and save `submission.csv`

### Test data

Place the test CSV in one of these locations (the notebook auto-detects):
- Same directory: `test.csv`, `Test.csv`, `round1_test.csv`
- Subdirectory: `test_data/test.csv`, `test_data/Test.csv`, `test_data/round1_test.csv`

If no test file is found, inference runs on the training data as a sanity check.

---

## Data Description

The training CSV has ~8 million rows across 80,000 unique audio clips:

| Column | Type | Description |
|---|---|---|
| `Sample_ID` | int | Unique clip identifier (each clip is an independent acoustic event) |
| `Time_ms` | float | Timestamp in milliseconds (0–100 ms range, 1 kHz sampling) |
| `Is_Context` | 0/1 | 1 = surviving fragment (visible), 0 = missing gap (to predict) |
| `Value` | float | Voltage reading from the magnetic scan |

**Each sample is independent** — Sample #10001 might be a drum hit, Sample #10002 a flute note.
No information can be shared across samples. The model must treat every clip as a fresh problem.

---

## Hyperparameters

All settings live in a single `Config` dataclass at the top of the notebook:

| Parameter | Value | Purpose |
|---|---|---|
| `epochs` | 80 | Upper bound; early stopping usually halts earlier |
| `patience` | 10 | Stop after 10 epochs without val MSE improvement |
| `batch_size` | 64 | Number of clips per gradient step |
| `learning_rate` | 1e-3 | AdamW initial learning rate |
| `weight_decay` | 1e-4 | L2 regularisation |
| `d_hidden` | 128 | Hidden dimension for encoder + decoder MLPs |
| `num_frequencies` | 32 | Fourier frequency bands (captures harmonics up to ~500 Hz) |
| `encoder_layers` | 4 | Context encoder depth |
| `decoder_layers` | 4 | Target decoder depth |
| `attention_heads` | 4 | Multi-head cross-attention heads |
| `dropout` | 0.1 | Regularisation during training |
| `augment_prob` | 0.70 | Probability of full random context/target repartition |

---

## Training Details

### Data augmentation (the most important trick)

During training, 70% of the time the context/target split is **completely re-randomised** for each
sample — the context fraction is drawn uniformly from [0.35, 0.85]. The remaining 30%, individual
points are flipped with 10% probability as a mild perturbation.

Without this, the model memorises the fixed CSV split and generalises poorly. With it, the model
learns to reconstruct waveforms from any observation density.

### Training loop

- **Loss**: masked MSE on target points only (padding from variable-length batching is excluded)
- **Gradient clipping**: max norm 1.0 (stabilises attention layers)
- **Checkpoint**: saved whenever validation MSE improves; best model is reloaded before inference
- **Baseline comparison**: a naive "predict context mean" baseline is computed before training as
  a sanity check — the trained model must significantly beat it

---

## Notebook Structure (Illustrated Blog)

| Section | Contents |
|---|---|
| 1 — Configuration | All hyperparameters and paths in one dataclass |
| 2 — Utilities | Seeding, device selection, masked MSE loss function |
| 3 — Data Loading + EDA | CSV reader, summary stats, random sample visualisations |
| 4 — Dataset Pipeline | Record pre-processing, stochastic augmentation, variable-length padding |
| 5 — Model | Fourier time encoding, MLP encoder, cross-attention, MLP decoder |
| 6 — Training | Training loop, naive baseline, AdamW + cosine LR + early stopping |
| 7 — Validation Plots | Side-by-side predicted vs true waveforms on held-out samples |
| 8 — Inference | Batch prediction and submission CSV export |
| 9 — Main | Single entry point to run the full pipeline |

Each section includes markdown explanations of *why* the design decisions were made, not just
*what* the code does — making it a self-contained illustrated blog suitable for Round 2 judging.

---

## References

- Kim et al., *Attentive Neural Processes* (ICLR 2019) — the base architecture
- Garnelo et al., *Conditional Neural Processes* (ICML 2018) — the neural process framework
- Mildenhall et al., *NeRF* (ECCV 2020) — Fourier positional encoding
