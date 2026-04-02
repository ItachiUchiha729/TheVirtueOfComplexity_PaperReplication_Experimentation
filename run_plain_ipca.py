"""
run_plain_ipca.py
=================
Standalone script for Run 16a: plain IPCA on the top-50 universe.
Replicates the notebook data pipeline (cells 9, 11, 22, 27) without
needing WRDS — uses average 'Size' characteristic to select top-50.

Run:  python3 run_plain_ipca.py
"""

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from backtest_GRASS_IPCA import run_ipca_grass_v2
from src._grass_worker import _z_label

# ---------------------------------------------------------------------------
# 1. Load & clean (replicate cells 8-9)
# ---------------------------------------------------------------------------
df_raw = pd.read_parquet(ROOT_DIR / "data" / "ipca_char_data_filtered.parquet")

# Drop the 4 permnos with the most missing values (cell 9)
missing_by_permno = df_raw.isna().sum(axis=1).groupby(df_raw["permno"]).sum().sort_values(ascending=False)
permnos_to_drop   = missing_by_permno.head(4).index
df_raw = df_raw[~df_raw["permno"].isin(permnos_to_drop)].copy()

# Forward-fill remaining gaps within each permno
char_like = [c for c in df_raw.columns if c not in ("permno", "yyyymm")]
df_raw[char_like] = df_raw.groupby("permno")[char_like].ffill()

# ---------------------------------------------------------------------------
# 2. Select top-50 by average Size characteristic (proxy for market cap)
# ---------------------------------------------------------------------------
top50_permnos = (
    df_raw.groupby("permno")["Size"]
    .mean()
    .nlargest(50)
    .index
)
df_top50 = df_raw[df_raw["permno"].isin(top50_permnos)].copy()
print(f"Top-50 stocks selected: {df_top50['permno'].nunique()}")

# ---------------------------------------------------------------------------
# 3. Panel setup (replicate cell 27)
# ---------------------------------------------------------------------------
df = df_top50.copy()
df["date"] = pd.to_datetime(df["yyyymm"])
df = df.drop(columns=["yyyymm"]).set_index(["date", "permno"]).sort_index()

char_cols = [c for c in df.columns if c not in ("y_ipca", "excess_ret")]
df = df.rename(columns={"y_ipca": "ret"})
df = df.dropna(subset=["ret"])
df["Price"] = 1.0   # dummy required by API

print(f"char_cols ({len(char_cols)}): {char_cols[:5]} ...")
print(f"Panel: {df.shape}  |  permnos: {df.index.get_level_values('permno').nunique()}")
print(f"Date range: {df.index.get_level_values('date').min().date()} → "
      f"{df.index.get_level_values('date').max().date()}")

# ---------------------------------------------------------------------------
# 4. Impute characteristics
# ---------------------------------------------------------------------------
X_chars_plain = df[char_cols].copy()
X_chars_plain = (
    X_chars_plain
    .groupby(level="date").transform(lambda s: s.fillna(s.median()))
    .fillna(X_chars_plain.median())
    .fillna(0.0)
    .astype(np.float64)
)
df_plain = (
    pd.concat([df[["Price", "ret"]], X_chars_plain], axis=1)
    .dropna(subset=["ret"])
)
print(f"\ndf_plain: {df_plain.shape}  ({len(char_cols)} char cols as instruments)")

# ---------------------------------------------------------------------------
# 5. Sweep
# ---------------------------------------------------------------------------
NUM_FACTORS_LIST_PLAIN = [4, 8, 12, 16, 20, 24, 28]
Z_VALUES_PLAIN         = [10.0, 50.0, 100.0]
WINDOW_LEN_PLAIN       = 24

# "Price" is in char_cols (defined before df["Price"]=1.0 overwrites it) AND
# in df[["Price","ret"]], so df_plain would have a duplicate "Price" column.
# That causes unstack() to return a DataFrame instead of a Series inside the
# backtest, silently poisoning all predictions with NaN.
# Exclude it: as a constant (std=0) it contributes nothing after standardisation.
char_feats_plain = [c for c in char_cols if c != "Price"]
print(f"char_feats_plain ({len(char_feats_plain)}): {char_feats_plain[:5]} ...")

plain_raw = {}
t0 = time.time()
for num_fact in NUM_FACTORS_LIST_PLAIN:
    for z in Z_VALUES_PLAIN:
        t1 = time.time()
        try:
            res = run_ipca_grass_v2(
                df_trunc          = df_plain,
                char_feats        = char_feats_plain,
                num_fact          = num_fact,
                verbose           = False,
                window_len        = WINDOW_LEN_PLAIN,
                max_nan           = 0.3,
                max_zero          = 1.0,
                min_non_nan_frac  = 0.7,
                shrinkage         = z,
                min_gradient_norm = 1e-6,
            )
            elapsed = time.time() - t1
            print(f"  k={num_fact:2d}  z={z:6.0f}  r2_oos={res['r2_oos']:+.4f}  "
                  f"sharpe={res['sharpe']:+.3f}  ({elapsed:.1f}s)")
            plain_raw[(num_fact, _z_label(z))] = {
                "avg_r2_oos"              : res["r2_oos"],
                "avg_sharpe"              : res["sharpe"],
                "avg_subspace_stability"  : res["mean_subspace_stability"],
                "avg_max_principal_angle" : res["mean_max_principal_angle"],
                "avg_mean_principal_angle": res["mean_mean_principal_angle"],
                "avg_geodesic_accel"      : res["mean_geodesic_accel"],
                "avg_erank"               : res["mean_erank"],
                "avg_erank_collapse"      : res["erank_collapse_frac"],
                "avg_spectral_gap"        : res["mean_spectral_gap"],
            }
        except Exception as e:
            elapsed = time.time() - t1
            print(f"  k={num_fact:2d}  z={z:6.0f}  FAILED: {e}  ({elapsed:.1f}s)")

total_time = time.time() - t0
print(f"\nRun 16a total: {total_time/60:.1f} min  ({total_time:.0f}s)")
if plain_raw:
    print(f"  → per (k,z) avg: {total_time/len(plain_raw):.1f}s")

# ---------------------------------------------------------------------------
# 6. Display results
# ---------------------------------------------------------------------------
ipca_plain_df = pd.DataFrame(plain_raw).T
ipca_plain_df.index = pd.MultiIndex.from_tuples(ipca_plain_df.index, names=["k", "z"])
ipca_plain_df = ipca_plain_df.sort_index()

with pd.option_context("display.max_rows", 50, "display.max_columns", 15,
                       "display.float_format", "{:.4f}".format,
                       "display.width", 140):
    print("\n=== Run 16a — Plain IPCA (top-50) ===")
    print(ipca_plain_df)
