# %% [markdown]
# # Spectral Bridge — Waveform Reconstruction Notebook
#
# This notebook focuses on what the dataset actually rewards:
# reconstructing missing audio samples on a fixed 100-point grid from 20 observed points.
#
# After reviewing the previous ANP pipeline, the core issue was clear:
# the learned model was underfitting badly and barely beat a context-mean baseline.
# A deterministic signal-reconstruction approach performs far better on the held-out split.
#
# The pipeline below:
# - loads and audits the dataset
# - rebuilds the same train / validation split by `Sample_ID`
# - benchmarks simple reconstruction methods
# - uses **edge-aware natural cubic spline interpolation** for validation plots and test inference
# - exports `submission.csv` without any retraining loop

# %%
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

TQDM_DISABLED = os.environ.get("TQDM_DISABLE", "0") == "1"

# %% [markdown]
# ## 1) Configuration
#
# This version keeps the project structure simple:
# - fixed paths for train / test / artifacts
# - reproducible validation split
# - no training hyperparameters because the final method is deterministic

# %%
@dataclass
class BridgeRunConfig:
    train_csv: Path = Path("train.csv")
    test_csv_candidates: Tuple[str, ...] = (
        "test.csv",
        "Test.csv",
        "round1_test.csv",
        "test_data/test.csv",
        "test_data/Test.csv",
        "test_data/round1_test.csv",
    )
    submission_path: Path = Path("submission.csv")
    summary_path: Path = Path("run_summary.csv")
    artifact_dir: Path = Path("artifacts")

    seed: int = 42
    val_size: float = 0.10
    n_eda_samples: int = 5
    n_val_plots: int = 10
    n_test_plots: int = 10


CFG = BridgeRunConfig()
print(f"Config: val_size={CFG.val_size}, artifact_dir={CFG.artifact_dir}")

# %% [markdown]
# ## 2) Runtime Utilities
#
# We keep a few small helpers for reproducibility, artifact saving, and MSE reporting.

# %%
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_artifact_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def finalize_figure(fig: plt.Figure, filename: str, cfg: BridgeRunConfig) -> None:
    out_dir = ensure_artifact_dir(cfg.artifact_dir)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    plt.close(fig)
    print(f"Saved figure: {out_path.resolve()}")


def mse_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

# %% [markdown]
# ## 3) Data Loading and Audit
#
# The competition files are small enough per-sample that a compact DataFrame + per-sample
# record representation works well. We keep `float32` / `int32` types to stay memory-efficient.

# %%
def load_dataframe(csv_path: Path, is_train: bool = True) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    dtype_map = {
        "Sample_ID": np.int32,
        "Time_ms": np.float32,
        "Is_Context": np.int8,
    }
    df = pd.read_csv(csv_path, dtype=dtype_map, low_memory=False)

    required = {"Sample_ID", "Time_ms", "Is_Context"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")

    if "Value" in df.columns:
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce").astype(np.float32)
    elif is_train:
        raise ValueError(f"{csv_path.name} must contain a Value column for training data.")

    df = df.sort_values(["Sample_ID", "Time_ms"], kind="mergesort").reset_index(drop=True)
    return df


def eda_report(df: pd.DataFrame, title: str = "Train Data") -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(f"Shape:              {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Unique samples:     {df['Sample_ID'].nunique():,}")
    print(f"Null values:        {df.isnull().sum().sum()}")
    print("\nColumn dtypes:")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            cmin = df[col].min(skipna=True)
            cmax = df[col].max(skipna=True)
            print(f"  {col:15s}  {str(df[col].dtype):10s}  range=[{cmin:.4f}, {cmax:.4f}]")
        else:
            print(f"  {col:15s}  {str(df[col].dtype):10s}")

    per_sample = (
        df.groupby("Sample_ID")["Is_Context"]
        .agg(
            context_pts=lambda s: int((s == 1).sum()),
            target_pts=lambda s: int((s == 0).sum()),
        )
    )
    print(f"\nPoints per sample (averaged over {len(per_sample):,} samples):")
    print(f"  Context:  {per_sample['context_pts'].mean():.1f}")
    print(f"  Target:   {per_sample['target_pts'].mean():.1f}")
    print(f"  Total:    {(per_sample['context_pts'] + per_sample['target_pts']).mean():.1f}")


def plot_random_samples(df: pd.DataFrame, cfg: BridgeRunConfig) -> None:
    unique_ids = df["Sample_ID"].unique()
    n_samples = min(cfg.n_eda_samples, len(unique_ids))
    rng = np.random.default_rng(cfg.seed)
    chosen_ids = rng.choice(unique_ids, size=n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3.0 * n_samples), squeeze=False)

    for ax, sid in zip(axes.ravel(), chosen_ids):
        sdf = df[df["Sample_ID"] == sid].sort_values("Time_ms")

        if "Value" in sdf.columns:
            ax.plot(sdf["Time_ms"], sdf["Value"], color="gray", alpha=0.30, lw=1, label="Full waveform")

        ctx = sdf[sdf["Is_Context"] == 1]
        tgt = sdf[sdf["Is_Context"] == 0]
        ax.scatter(ctx["Time_ms"], ctx["Value"], s=14, color="royalblue", zorder=3, label="Context")
        if "Value" in tgt.columns:
            ax.scatter(tgt["Time_ms"], tgt["Value"], s=18, color="crimson", zorder=3, label="Target")

        ax.set_title(f"Sample {int(sid)}", fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage")
        ax.grid(alpha=0.15)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle("Random training samples", fontsize=11, y=1.01)
    fig.tight_layout()
    finalize_figure(fig, "eda_random_samples.png", cfg)

# %% [markdown]
# ## 4) Sample Records
#
# Each waveform becomes a compact dict with:
# - raw timestamps
# - observed / missing mask
# - values (NaN on hidden test targets)

# %%
def build_sample_records(df: pd.DataFrame) -> List[Dict[str, np.ndarray]]:
    records: List[Dict[str, np.ndarray]] = []

    grouped = df.groupby("Sample_ID", sort=True, observed=True)
    for sample_id, sdf in tqdm(
        grouped,
        total=df["Sample_ID"].nunique(),
        desc="Building sample records",
        disable=TQDM_DISABLED,
    ):
        time_raw = sdf["Time_ms"].to_numpy(dtype=np.float32, copy=True)
        is_context = sdf["Is_Context"].to_numpy(dtype=np.int8, copy=True) == 1

        if "Value" in sdf.columns:
            y = sdf["Value"].to_numpy(dtype=np.float32, copy=True)
        else:
            y = np.full_like(time_raw, np.nan, dtype=np.float32)

        records.append(
            {
                "sample_id": np.int32(sample_id),
                "time_raw": time_raw,
                "y": y,
                "is_context": is_context,
            }
        )

    return records


def split_records(records: Sequence[Dict[str, np.ndarray]], cfg: BridgeRunConfig):
    sample_ids = [int(r["sample_id"]) for r in records]
    train_ids, val_ids = train_test_split(
        sample_ids,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        shuffle=True,
    )
    train_set = set(train_ids)
    val_set = set(val_ids)

    train_records = [r for r in records if int(r["sample_id"]) in train_set]
    val_records = [r for r in records if int(r["sample_id"]) in val_set]
    return train_records, val_records

# %% [markdown]
# ## 5) Reconstruction Methods
#
# The previous ANP was underfitting badly on this dataset.
# For this fixed-grid sparse reconstruction problem, simple interpolation is far stronger.
#
# We benchmark three methods:
# - context mean
# - linear interpolation
# - edge-aware natural cubic spline interpolation
#
# An edge-aware natural cubic spline wins by a large margin on the held-out split,
# so we use it for export.

# %%
def context_mean_predict(time_context: np.ndarray, y_context: np.ndarray, time_target: np.ndarray) -> np.ndarray:
    del time_context
    return np.full(time_target.shape, np.mean(y_context), dtype=np.float32)


def linear_interp_predict(time_context: np.ndarray, y_context: np.ndarray, time_target: np.ndarray) -> np.ndarray:
    pred = np.interp(time_target, time_context, y_context)
    return np.asarray(pred, dtype=np.float32)


def _linear_extrapolate(x0: float, y0: float, x1: float, y1: float, x: np.ndarray) -> np.ndarray:
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)


def edge_aware_cubic_predict(
    time_context: np.ndarray,
    y_context: np.ndarray,
    time_target: np.ndarray,
) -> np.ndarray:
    if len(time_context) < 4:
        return linear_interp_predict(time_context, y_context, time_target)

    spline = CubicSpline(time_context, y_context, bc_type="natural", extrapolate=False)
    pred = np.asarray(spline(time_target), dtype=np.float32)

    left = time_target < time_context[0]
    right = time_target > time_context[-1]

    if left.any():
        pred[left] = _linear_extrapolate(
            time_context[0], y_context[0], time_context[1], y_context[1], time_target[left]
        )
    if right.any():
        pred[right] = _linear_extrapolate(
            time_context[-1], y_context[-1], time_context[-2], y_context[-2], time_target[right]
        )

    return np.asarray(pred, dtype=np.float32)


def predict_one_record(
    record: Dict[str, np.ndarray],
    predictor: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> Dict[str, np.ndarray]:
    is_ctx = record["is_context"].copy()

    if is_ctx.sum() == 0:
        is_ctx[0] = True
    if is_ctx.all():
        is_ctx[-1] = False

    ctx_idx = np.where(is_ctx)[0]
    tgt_idx = np.where(~is_ctx)[0]

    time_context = record["time_raw"][ctx_idx]
    y_context = record["y"][ctx_idx]
    time_target = record["time_raw"][tgt_idx]
    y_true = record["y"][tgt_idx]
    y_pred = predictor(time_context, y_context, time_target)

    return {
        "sample_id": int(record["sample_id"]),
        "time_context": time_context,
        "y_context": y_context,
        "time_target": time_target,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def evaluate_predictor(
    records: Sequence[Dict[str, np.ndarray]],
    predictor: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    desc: str,
) -> float:
    total_sse = 0.0
    total_n = 0

    for record in tqdm(records, desc=desc, disable=TQDM_DISABLED):
        out = predict_one_record(record, predictor)
        valid = np.isfinite(out["y_true"])
        if not valid.any():
            continue
        err = out["y_pred"][valid] - out["y_true"][valid]
        total_sse += float(np.square(err).sum())
        total_n += int(valid.sum())

    return total_sse / max(total_n, 1)


def benchmark_predictors(val_records: Sequence[Dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    methods = {
        "context_mean": context_mean_predict,
        "linear_interp": linear_interp_predict,
        "edge_aware_cubic_spline": edge_aware_cubic_predict,
    }

    for name, predictor in methods.items():
        mse = evaluate_predictor(val_records, predictor, desc=f"Validating {name}")
        rows.append({"method": name, "val_mse": float(mse)})

    return pd.DataFrame(rows).sort_values("val_mse", kind="mergesort").reset_index(drop=True)

# %% [markdown]
# ## 6) Diagnostics
#
# We plot held-out validation samples to verify that the reconstruction follows the hidden
# waveform and not just a flat average.

# %%
def plot_holdout_samples(
    records: Sequence[Dict[str, np.ndarray]],
    predictor: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    filename: str,
    title: str,
    cfg: BridgeRunConfig,
    show_truth: bool,
    n: int,
    seed: int,
) -> None:
    n = min(n, len(records))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=n, replace=False)

    fig, axes = plt.subplots(n, 1, figsize=(13, 3.0 * n), squeeze=False)

    for ax, i in zip(axes.ravel(), idx):
        out = predict_one_record(records[i], predictor)
        order = np.argsort(out["time_target"])

        ax.scatter(out["time_context"], out["y_context"], s=16, c="royalblue", label="Context", zorder=3)
        if show_truth:
            ax.scatter(
                out["time_target"],
                out["y_true"],
                s=18,
                c="crimson",
                alpha=0.7,
                label="True target",
                zorder=3,
            )
        ax.plot(
            out["time_target"][order],
            out["y_pred"][order],
            c="green",
            lw=2,
            label="Predicted",
            zorder=2,
        )
        ax.scatter(out["time_target"][order], out["y_pred"][order], s=12, c="green", zorder=2)
        ax.set_title(f"Sample {out['sample_id']}", fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage")
        ax.grid(alpha=0.15)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    finalize_figure(fig, filename, cfg)

# %% [markdown]
# ## 7) Inference and Export
#
# Test inference is straightforward:
# for each sample, take the context points, reconstruct the missing waveform with the
# selected predictor, and write the missing-point rows into `submission.csv`.

# %%
def find_test_csv(candidates: Sequence[str]) -> Optional[Path]:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def run_inference(
    records: Sequence[Dict[str, np.ndarray]],
    predictor: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> pd.DataFrame:
    rows_sid: List[int] = []
    rows_time: List[float] = []
    rows_pred: List[float] = []

    for record in tqdm(records, desc="Inference", disable=TQDM_DISABLED):
        out = predict_one_record(record, predictor)
        rows_sid.extend([out["sample_id"]] * len(out["time_target"]))
        rows_time.extend(out["time_target"].tolist())
        rows_pred.extend(out["y_pred"].tolist())

    return (
        pd.DataFrame(
            {
                "Sample_ID": np.array(rows_sid, dtype=np.int32),
                "Time_ms": np.array(rows_time, dtype=np.float32),
                "Predicted_Value": np.array(rows_pred, dtype=np.float32),
            }
        )
        .sort_values(["Sample_ID", "Time_ms"], kind="mergesort")
        .reset_index(drop=True)
    )

# %% [markdown]
# ## 8) End-to-End Run
#
# One call does everything:
# - load data
# - benchmark reconstruction methods
# - generate validation plots
# - run test inference
# - save artifacts and submission

# %%
def main(cfg: BridgeRunConfig) -> None:
    set_seed(cfg.seed)
    run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Run started: {run_started_at}")
    print(f"Working directory: {Path.cwd()}")

    train_df = load_dataframe(cfg.train_csv, is_train=True)
    eda_report(train_df)
    plot_random_samples(train_df, cfg)

    all_records = build_sample_records(train_df)
    train_records, val_records = split_records(all_records, cfg)
    print(f"\nSplit: {len(train_records):,} train / {len(val_records):,} val samples")

    benchmark_df = benchmark_predictors(val_records)
    print("\nValidation leaderboard:")
    print(benchmark_df.to_string(index=False))

    best_method = str(benchmark_df.iloc[0]["method"])
    method_to_predictor = {
        "context_mean": context_mean_predict,
        "linear_interp": linear_interp_predict,
        "edge_aware_cubic_spline": edge_aware_cubic_predict,
    }
    predictor = method_to_predictor[best_method]
    best_mse = float(benchmark_df.iloc[0]["val_mse"])
    baseline_mse = float(benchmark_df.loc[benchmark_df["method"] == "context_mean", "val_mse"].iloc[0])
    rel_gain = (baseline_mse - best_mse) / max(baseline_mse, 1e-12)

    summary_df = pd.DataFrame(
        [
            {
                "run_started_at": run_started_at,
                "best_method": best_method,
                "best_val_mse": best_mse,
                "context_mean_val_mse": baseline_mse,
                "relative_gain_vs_context_mean": rel_gain,
            }
        ]
    )
    summary_df.to_csv(cfg.summary_path, index=False)
    print(f"\nSaved summary: {cfg.summary_path.resolve()}")

    plot_holdout_samples(
        val_records,
        predictor=predictor,
        filename="val_predictions.png",
        title=f"Validation predictions using {best_method}",
        cfg=cfg,
        show_truth=True,
        n=cfg.n_val_plots,
        seed=cfg.seed + 1,
    )

    test_csv = find_test_csv(cfg.test_csv_candidates)
    if test_csv is not None:
        print(f"\nTest file found: {test_csv}")
        test_df = load_dataframe(test_csv, is_train=False)
        if "Value" not in test_df.columns:
            test_df["Value"] = np.nan
        infer_records = build_sample_records(test_df)
    else:
        print("\nNo test CSV found — running sanity inference on training data.")
        infer_records = all_records

    sub = run_inference(infer_records, predictor)
    sub.to_csv(cfg.submission_path, index=False)
    print(f"\nSubmission saved: {cfg.submission_path.resolve()}")
    print(f"Rows: {len(sub):,}")
    print(sub.head(10).to_string(index=False))

    plot_holdout_samples(
        infer_records,
        predictor=predictor,
        filename="test_predictions.png",
        title=f"Test predictions using {best_method}",
        cfg=cfg,
        show_truth=False,
        n=min(cfg.n_test_plots, len(infer_records)),
        seed=cfg.seed + 2,
    )


if __name__ == "__main__":
    main(CFG)
