#!/usr/bin/env python3
"""Rerun the synthetic v3 sweeps with job-level checkpoints.

This mirrors notebooks/experimentation_synthetic_data_v3.ipynb, but saves
completed jobs immediately so a long run can be resumed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generate_synthetic_panel import generate_synthetic_panel
from src.rff import RandomFourierFeatures
from src._grass_worker import (
    build_jobs,
    build_rff_inputs,
    run_sweep_checkpointed,
    aggregate,
    make_results_df,
)


TRUE_K0 = 12
NUM_FACTORS_LIST = [4, 8, 12, 16, 20, 24, 28, 32]
GAMMA = 0.25
WINDOW_LEN = 24
NUM_ITER_RFF = 1

SWEEPS = {
    "A": {
        "result_set": "sweep_A_k_z_P1024",
        "P": [1024],
        "K": NUM_FACTORS_LIST,
        "Z": [0, 10, 50],
    },
    "B": {
        "result_set": "sweep_B_P_grid",
        "P": [256, 1024, 4096],
        "K": [TRUE_K0, 24],
        "Z": [10, 50],
    },
    "C": {
        "result_set": "sweep_C_fine_z",
        "P": [1024],
        "K": [12],
        "Z": [0, 1, 5, 10, 20, 50, 100],
    },
    "D": {
        "result_set": "sweep_D_fine_P",
        "P": [128, 256, 512, 1024, 2048, 4096],
        "K": [12, 24],
        "Z": [10],
    },
}


def build_synthetic_inputs():
    df_synth, truth, char_cols = generate_synthetic_panel(
        T=360,
        N=50,
        m=30,
        k0=TRUE_K0,
        seed=42,
        snr_base=5.0,
        eigenvalue_decay=0.82,
        sigma_eps=1.0,
        factor_ar1=0.5,
        factor_vol=0.3,
        w_drift_scale=None,
        target_dproj=0.6,
        z_corr=0.3,
        hetero_strength=0.3,
    )

    x_chars = (
        df_synth[char_cols]
        .apply(lambda s: s.fillna(s.mean()), axis=0)
        .astype(np.float64)
    )
    return df_synth, truth, x_chars


def principal_angle_anisotropy(agg_entry: dict) -> float:
    ratios = []
    for pa_series in agg_entry["principal_angles_series"]:
        for angles in pa_series:
            if len(angles) and angles[0] > 1e-8:
                ratios.append(float(np.mean(angles) / angles[0]))
    return float(np.mean(ratios)) if ratios else np.nan


def flatten_summary(df: pd.DataFrame, agg: dict, result_set: str) -> pd.DataFrame:
    out = df.reset_index()
    out.insert(0, "result_set", result_set)

    aniso = []
    for _, row in out.iterrows():
        key = (int(row["k"]), int(row["P"]), row["z"])
        aniso.append(principal_angle_anisotropy(agg[key]))
    out["anisotropy"] = aniso
    out["angle_anisotropy_ratio"] = (
        out["avg_mean_principal_angle"] / out["avg_max_principal_angle"]
    )
    return out


def run_one_sweep(
    name: str,
    spec: dict,
    df_synth: pd.DataFrame,
    x_chars: pd.DataFrame,
    checkpoint_root: Path,
    output_dir: Path,
    n_workers: int | None,
    force: bool,
) -> pd.DataFrame:
    print(f"\n=== Sweep {name}: {spec['result_set']} ===", flush=True)
    rff_inputs = build_rff_inputs(
        x_chars,
        df_synth,
        spec["P"],
        NUM_ITER_RFF,
        GAMMA,
        RandomFourierFeatures,
    )
    jobs = build_jobs(
        rff_inputs,
        spec["K"],
        spec["P"],
        spec["Z"],
        NUM_ITER_RFF,
        WINDOW_LEN,
    )
    print(f"Sweep {name}: {len(jobs)} jobs", flush=True)

    raw = run_sweep_checkpointed(
        jobs,
        checkpoint_root / f"sweep_{name}",
        n_workers=n_workers,
        verbose=True,
        force=force,
    )
    agg = aggregate(raw)
    df = make_results_df(agg)
    flat = flatten_summary(df, agg, spec["result_set"])

    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / f"synthetic_v3_sweep_{name}_summary.csv"
    flat.to_csv(sweep_path, index=False)
    print(f"Saved sweep {name} summary -> {sweep_path}", flush=True)
    return flat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweeps",
        default="A,B,C,D",
        help="Comma-separated sweeps to run/resume. Default: A,B,C,D",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Override worker count. Default uses src._grass_worker heuristic.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun jobs even if checkpoints exist.",
    )
    parser.add_argument(
        "--overwrite-main",
        action="store_true",
        help="Also overwrite data/all_synthetic_datarun_results.csv.",
    )
    return parser.parse_args()


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    args = parse_args()
    requested = [s.strip().upper() for s in args.sweeps.split(",") if s.strip()]
    unknown = sorted(set(requested) - set(SWEEPS))
    if unknown:
        raise ValueError(f"Unknown sweeps: {unknown}. Valid sweeps: {sorted(SWEEPS)}")

    checkpoint_root = REPO_ROOT / "cache" / "synthetic_v3_checkpoints"
    output_dir = REPO_ROOT / "data"

    df_synth, truth, x_chars = build_synthetic_inputs()
    print(f"Panel shape: {df_synth.shape}", flush=True)
    print(f"True k0: {truth['true_k0']}", flush=True)
    print(f"w_drift_scale: {truth['w_drift_scale']:.6f}", flush=True)

    pieces = []
    for name in requested:
        pieces.append(
            run_one_sweep(
                name,
                SWEEPS[name],
                df_synth,
                x_chars,
                checkpoint_root,
                output_dir,
                args.n_workers,
                args.force,
            )
        )

    combined = pd.concat(pieces, ignore_index=True)
    full_path = output_dir / "all_synthetic_datarun_results_v3_full.csv"
    combined.to_csv(full_path, index=False)
    print(f"\nSaved combined full v3 results -> {full_path}", flush=True)

    if args.overwrite_main:
        main_path = output_dir / "all_synthetic_datarun_results.csv"
        combined.to_csv(main_path, index=False)
        print(f"Overwrote main synthetic results -> {main_path}", flush=True)


if __name__ == "__main__":
    main()
