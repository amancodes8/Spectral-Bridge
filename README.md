# Spectral Bridge - Residual Spline Reconstruction Pipeline

This repository contains a notebook-first solution for waveform gap reconstruction.

The current notebook does not train an Attentive Neural Process. It uses a two-stage approach:

1. Build an edge-aware cubic spline baseline from context points.
2. Train a 1D residual neural network to predict corrections on top of that baseline.

The final output is a competition-style submission with columns:
Sample_ID, Time_ms, Predicted_Value.

## What The Notebook Implements

Notebook file:
- spectral_bridge_anp_notebook.ipynb 17-12-27-516.ipynb

Main stages in code:

1. Configuration
- Defines paths, split ratio, plotting counts, and artifact directory.

2. Data loading and validation
- Reads CSV with typed columns.
- Requires Sample_ID, Time_ms, Is_Context.
- Requires Value for training data.

3. Record construction
- Groups rows by Sample_ID.
- Builds per-sample arrays for time, value, and context mask.
- Splits samples into train/validation sets using train_test_split.

4. Classical reconstruction baselines
- context_mean
- linear interpolation
- edge-aware cubic spline with linear extrapolation at boundaries

5. Residual model training (PyTorch)
- Input features per time point:
  observed_value, context_mask, spline_baseline, normalized_time
- Model: ResidualSplineNet (Conv1D stem + residual dilated blocks + Conv1D head)
- Prediction: baseline + learned residual
- Context points are hard-clamped to observed values
- Loss/metrics are computed only on target points (Is_Context == 0)
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Gradient clipping: max norm 1.0

6. Checkpointing and diagnostics
- Saves last checkpoint each epoch:
  last_residual_spline_net.pt
- Saves best checkpoint by validation MSE:
  best_residual_spline_net.pt
- Writes plots into artifacts directory.

7. Inference
- Uses last checkpoint when available.
- If no trained model is available, falls back to spline-only inference.
- Auto-detects test CSV from several candidate paths.
- Writes submission.csv.

## Requirements

Use Python 3.9+ (3.10+ recommended) and install:

```bash
pip install numpy pandas matplotlib scipy scikit-learn tqdm torch
```

Device selection in notebook:
- CUDA if available
- else Apple MPS if available
- else CPU

## Expected Data Format

Training CSV columns:
- Sample_ID
- Time_ms
- Is_Context
- Value

Test CSV columns:
- Sample_ID
- Time_ms
- Is_Context
- Value is optional (filled with NaN for inference if missing)

## Variables And Default Values

Primary runtime config (BridgeRunConfig):

| Variable | Default value |
|---|---|
| train_csv | train.csv |
| test_csv_candidates | (test.csv, Test.csv, round1_test.csv, test_data/test.csv, test_data/Test.csv, test_data/round1_test.csv) |
| submission_path | submission.csv |
| summary_path | run_summary.csv |
| artifact_dir | artifacts |
| seed | 42 |
| val_size | 0.10 |
| n_eda_samples | 5 |
| n_val_plots | 10 |
| n_test_plots | 10 |

Residual model config (RESIDUAL_MODEL_CFG):

| Variable | Default value |
|---|---|
| epochs | 30 |
| batch_size | 256 if CUDA is available, else 128 |
| num_workers | 0 |
| learning_rate | 2e-3 |
| weight_decay | 1e-4 |
| hidden_channels | 64 |
| num_blocks | 6 |
| dropout | 0.05 |
| accuracy_tolerance | 0.05 |
| last_checkpoint_path | last_residual_spline_net.pt |
| best_checkpoint_path | best_residual_spline_net.pt |

Per-sample record variables used in the pipeline:

| Variable | Meaning |
|---|---|
| sample_id | Unique waveform/sample identifier |
| time_raw | Full time axis for one sample |
| y / y_full | Ground-truth waveform values |
| is_context / context_mask | Boolean/float mask where 1 means observed context point |
| observed | Values known at context points (0 elsewhere) |
| baseline | Edge-aware cubic spline estimate over full time axis |
| features | 4 x T tensor: [observed, context_mask, baseline, normalized_time] |

## How To Run

1. Open the notebook:
   spectral_bridge_anp_notebook.ipynb 17-12-27-516.ipynb
2. Run cells in order from top to bottom.
3. The training cell trains ResidualSplineNet and saves checkpoints.
4. The next cell performs inference and writes submission.csv.

Test file auto-discovery candidates:
- test.csv
- Test.csv
- round1_test.csv
- test_data/test.csv
- test_data/Test.csv
- test_data/round1_test.csv

If no test CSV is found, the notebook runs a sanity inference on validation records.

## Generated Outputs

Common outputs produced by notebook cells:

- submission.csv
- last_residual_spline_net.pt
- best_residual_spline_net.pt
- artifacts/eda_random_samples.png
- artifacts/residual_training_curve.png
- artifacts/val_predictions_recent_model.png
- artifacts/test_predictions.png

## Notes

- Existing file best_anp.pt can still be present in the repository, but it is not used by the current notebook pipeline.
- The final inference cell prefers the last checkpoint for prediction (not necessarily the best checkpoint).
