"""
load_data.py

Loads and cleans the two Marine Geophysical Research (2026) supplementary
datasets used for the ML stage:
    - Training/validation set (.pkl): 7,485 pre-joined CPTu-Vs pairs,
      already feature-engineered.
    - Testing set (.xlsx): 45 individual SCPTu profile sheets, each
      containing two SEPARATE depth grids (dense CPTu readings, sparse
      Vs measurements) that must be matched together before use.

Column naming convention (matching this project's established
convention across other portfolio projects): full field name + unit
suffix, snake_case -- e.g. depth_m, vs_mps, qt_mpa. This also resolves
a minor inconsistency in the source paper's own materials, where the
training pkl uses "qt/fs Ratio (-)" but their inference script computes
and expects "qt/fs (-)" -- we standardize both to a single name
(qt_fs_ratio) regardless of source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data file locations, resolved relative to this file's location so the
# script works regardless of what directory you run it from.
# ---------------------------------------------------------------------------
DATA_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "marine_geophys_2026"
TRAIN_PKL_PATH = DATA_RAW_DIR / "11001_2025_9602_MOESM4_ESM.pkl"
TEST_XLSX_PATH = DATA_RAW_DIR / "11001_2025_9602_MOESM6_ESM.xlsx"

# Small epsilon added to fs in the qt/fs ratio denominator, matching the
# source paper's own inference script exactly (their choice, not ours --
# we replicate it rather than "improve" it, so train and test features
# are computed identically).
QT_FS_EPSILON_MPA = 0.001

# Default tolerance (m) for nearest-neighbor matching a Vs measurement
# depth to its closest CPTu reading depth in the test set. CPTu is
# sampled roughly every 2 cm, so a well-matched pair should be within a
# few cm; 0.1 m is generous enough to tolerate minor depth-registration
# offsets between the two instruments without accidentally pairing a Vs
# reading to a CPTu point from a genuinely different depth (e.g. across
# a data gap).
DEFAULT_MATCH_TOLERANCE_M = 0.10

# Standardized column name mapping: source column name -> project
# convention. Applied to BOTH train and test after their respective
# loading/matching steps, so downstream code only ever sees one
# consistent schema regardless of which file a row came from.
TRAIN_COLUMN_RENAME = {
    "Vs (m/s)": "vs_mps",
    "Depth (m)": "depth_m",
    "fs (MPa)": "fs_mpa",
    "qt (MPa)": "qt_mpa",
    "u2 (MPa)": "u2_mpa",
    "Qt (-)": "qt_normalized",
    "Fr (%)": "fr_percent",
    "Bq (-)": "bq_dimensionless",
    "Ic (-)": "ic_dimensionless",
    "qt/fs Ratio (-)": "qt_fs_ratio",
    "Bq*Fr (-)": "bq_fr_product",
    "Soil Type": "soil_type",
}

TEST_CPTU_COLUMN_RENAME = {
    "Depth (m)": "depth_m",
    "qt (MPa)": "qt_mpa",
    "fs (MPa)": "fs_mpa",
    "u2 (MPa)": "u2_mpa",
    "Qt (-)": "qt_normalized",
    "Fr (%)": "fr_percent",
    "Bq (-)": "bq_dimensionless",
    "Ic (-)": "ic_dimensionless",
}

TEST_VS_COLUMN_RENAME = {
    "Depth_1 (m)": "vs_depth_m",
    "Shear Wave Velocity (m/s)": "vs_mps",
}

# Final feature set used for model training -- matches the source
# paper's RFECV-retained feature set exactly (all 10 features were
# retained; see paper's "Model optimisation" section), so our model
# is trained on the same inputs as theirs for a fair comparison.
FEATURE_COLUMNS = [
    "depth_m", "fs_mpa", "qt_mpa", "u2_mpa",
    "qt_normalized", "fr_percent", "bq_dimensionless", "ic_dimensionless",
    "qt_fs_ratio", "bq_fr_product",
]
TARGET_COLUMN = "vs_mps"


def load_training_data(path: Path = TRAIN_PKL_PATH) -> pd.DataFrame:
    """
    Load the 7,485-pair training/validation dataset, already fully
    feature-engineered by the source paper. Renames columns to this
    project's naming convention; no depth-matching needed since this
    file already has one row per CPTu-Vs pair.

    Parameters
    ----------
    path : Path, optional
        Location of the .pkl file. Defaults to the standard project
        location under data/raw/marine_geophys_2026/.

    Returns
    -------
    pd.DataFrame
        Shape (7485, 12), with columns renamed per TRAIN_COLUMN_RENAME.

    Raises
    ------
    FileNotFoundError
        If path doesn't exist -- with a clear message pointing back to
        the download instructions, rather than a generic pandas error.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. Download it from the "
            "Marine Geophysical Research (2026) paper's supplementary "
            "materials (MOESM4_ESM.pkl) -- see project README / prior "
            "conversation for the direct link."
        )

    df = pd.read_pickle(path)
    df = df.rename(columns=TRAIN_COLUMN_RENAME)

    expected_rows = 7485
    if len(df) != expected_rows:
        print(
            f"WARNING: loaded {len(df)} rows, expected {expected_rows} "
            "per the source paper. Data may differ from what this "
            "script was written against -- double check the source file."
        )

    return df


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply documented data-quality cleaning to the loaded training
    dataset. This is a SEPARATE step from load_training_data() so the
    raw, as-supplied data and our own cleaning judgment stay clearly
    distinguishable -- load_training_data() returns exactly what's in
    the source file; this function is where we make and document any
    decisions to alter it.

    CLEANING DECISIONS:

    1. DROP rows with vs_mps == 0. Inspection showed these 10 rows are
       scattered across 5 different soil types at shallow depths
       (1.5-4.5 m) with otherwise-normal feature values -- a pattern
       much more consistent with a failed arrival-time pick recorded
       as 0 (a data artifact) than a genuine physical measurement,
       since a true Vs=0 would imply zero shear stiffness (fluid-like
       behaviour), and if it were real it would be expected to
       cluster within one especially soft soil type rather than
       appear arbitrarily across types that otherwise show normal
       Vs values.

    2. KEEP high-Vs rows (>550 m/s), INCLUDING the "Other" soil type.
       116 of 142 such rows are the "Other" category, which the source
       paper explicitly identifies as chalk / non-siliciclastic
       material -- a real, physically distinct lithology, not an
       artifact. Dropping these would silently narrow the range of
       soil conditions the model is trained to handle. The remaining
       ~26 high-Vs rows in normal soil-type categories are plausible
       for well-cemented or heavily overconsolidated sediment and are
       also kept without further evidence they are erroneous.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_training_data().

    Returns
    -------
    pd.DataFrame
        Cleaned copy (a NEW dataframe -- the input is not modified in
        place), with a reset index.
    """
    n_before = len(df)

    zero_vs_mask = df["vs_mps"] == 0
    n_zero_vs = zero_vs_mask.sum()
    if n_zero_vs > 0:
        print(f"Dropping {n_zero_vs} row(s) with vs_mps == 0 (see function docstring for rationale):")
        print(
            df.loc[zero_vs_mask, ["depth_m", "vs_mps", "soil_type"]]
            .to_string(index=False)
        )

    df_clean = df.loc[~zero_vs_mask].reset_index(drop=True)

    n_high_vs = (df_clean["vs_mps"] > 550).sum()
    n_high_vs_other = ((df_clean["vs_mps"] > 550) & (df_clean["soil_type"] == "Other")).sum()
    print(
        f"\nRetaining {n_high_vs} row(s) with vs_mps > 550 m/s "
        f"({n_high_vs_other} of which are soil_type='Other', i.e. "
        f"chalk/non-siliciclastic material per the source paper) -- "
        f"these are treated as legitimate data, not artifacts. "
        f"See function docstring for rationale."
    )

    n_after = len(df_clean)
    print(f"\nTraining data: {n_before} rows -> {n_after} rows after cleaning "
          f"({n_before - n_after} dropped).")

    return df_clean


def _match_one_sheet(sheet_name: str, raw_sheet: pd.DataFrame, tolerance_m: float) -> pd.DataFrame:
    """
    For a single SCPTu profile sheet, nearest-neighbor-match the sparse
    Vs measurements to the dense CPTu readings by depth, and compute
    the two derived features (qt_fs_ratio, bq_fr_product) that are
    present in the training data but not pre-computed in this sheet.

    WHY merge_asof: both Vs depths and CPTu depths are 1-D, naturally
    sortable values -- merge_asof(..., direction="nearest") is
    pandas' built-in tool for exactly this "match each row in A to its
    nearest row in B on a sorted numeric key" problem, and is simpler
    and more auditable than a hand-rolled nearest-neighbor search.

    Parameters
    ----------
    sheet_name : str
        Name of the source sheet (e.g. "HKW_SCPT02") -- retained in
        the output as a 'source_sheet' column for traceability.
    raw_sheet : pd.DataFrame
        The raw, unprocessed sheet as read by pd.read_excel, containing
        BOTH the dense CPTu columns and the sparse Vs columns
        interleaved in one table.
    tolerance_m : float
        Maximum allowed depth difference (m) for a Vs-to-CPTu match.
        Vs measurements with no CPTu reading within tolerance are
        dropped (and the drop count is reported by the caller).

    Returns
    -------
    pd.DataFrame
        One row per successfully matched Vs measurement, with columns
        renamed to project convention plus 'source_sheet' and the two
        derived features.
    """
    # Split the interleaved sheet into its two logical parts, each
    # keeping only rows where that part's key measurement is present.
    vs_part = (
        raw_sheet[list(TEST_VS_COLUMN_RENAME.keys())]
        .rename(columns=TEST_VS_COLUMN_RENAME)
        .dropna(subset=["vs_mps"])
        .sort_values("vs_depth_m")
        .reset_index(drop=True)
    )
    cptu_part = (
        raw_sheet[list(TEST_CPTU_COLUMN_RENAME.keys())]
        .rename(columns=TEST_CPTU_COLUMN_RENAME)
        .dropna(subset=["qt_mpa"])
        .sort_values("depth_m")
        .reset_index(drop=True)
    )

    if vs_part.empty or cptu_part.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET_COLUMN, "source_sheet"])

    matched = pd.merge_asof(
        vs_part, cptu_part,
        left_on="vs_depth_m", right_on="depth_m",
        direction="nearest", tolerance=tolerance_m,
    )

    n_before = len(matched)
    matched = matched.dropna(subset=["qt_mpa"])  # drops unmatched (out-of-tolerance) rows
    n_dropped = n_before - len(matched)
    if n_dropped > 0:
        print(
            f"  {sheet_name}: dropped {n_dropped}/{n_before} Vs "
            f"measurement(s) with no CPTu reading within "
            f"{tolerance_m} m."
        )

    if matched.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET_COLUMN, "source_sheet"])

    # Derived features, computed identically to the source paper's own
    # inference script (including their epsilon choice -- see module
    # docstring).
    matched["qt_fs_ratio"] = matched["qt_mpa"] / (matched["fs_mpa"] + QT_FS_EPSILON_MPA)
    matched["bq_fr_product"] = matched["bq_dimensionless"] * matched["fr_percent"]
    matched["source_sheet"] = sheet_name

    return matched[FEATURE_COLUMNS + [TARGET_COLUMN, "source_sheet"]]


def load_testing_data(
    path: Path = TEST_XLSX_PATH,
    tolerance_m: float = DEFAULT_MATCH_TOLERANCE_M,
) -> pd.DataFrame:
    """
    Load the 45-sheet testing dataset, matching sparse Vs measurements
    to dense CPTu readings by nearest depth within each sheet, and
    computing the two derived features not pre-supplied in this file.

    Parameters
    ----------
    path : Path, optional
        Location of the .xlsx file. Defaults to the standard project
        location.
    tolerance_m : float, optional
        Passed through to _match_one_sheet -- see DEFAULT_MATCH_TOLERANCE_M
        for rationale.

    Returns
    -------
    pd.DataFrame
        One row per successfully matched Vs measurement across all 45
        sheets, columns = FEATURE_COLUMNS + [TARGET_COLUMN,
        'source_sheet']. Expected ~1526 rows total per the source
        paper (a small number may be dropped if any Vs measurements
        fall outside tolerance -- see per-sheet warnings printed
        during loading).

    Raises
    ------
    FileNotFoundError
        See load_training_data.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Testing data not found at {path}. Download it from the "
            "Marine Geophysical Research (2026) paper's supplementary "
            "materials (MOESM6_ESM.xlsx) -- see project README / prior "
            "conversation for the direct link."
        )

    xl = pd.ExcelFile(path)
    print(f"Matching {len(xl.sheet_names)} SCPTu profile sheets (tolerance={tolerance_m} m)...")

    matched_sheets = []
    for sheet_name in xl.sheet_names:
        raw_sheet = pd.read_excel(xl, sheet_name=sheet_name)
        matched_sheets.append(_match_one_sheet(sheet_name, raw_sheet, tolerance_m))

    df = pd.concat(matched_sheets, ignore_index=True)

    expected_rows = 1526
    if len(df) != expected_rows:
        print(
            f"\nNOTE: matched {len(df)} rows total, source paper reports "
            f"{expected_rows}. A small discrepancy can arise from the "
            f"tolerance setting or from measurements right at a profile's "
            f"depth-grid edge -- review the per-sheet drop warnings above "
            f"if the difference is large."
        )

    return df


if __name__ == "__main__":
    print("=" * 70)
    print("LOADING TRAINING/VALIDATION DATA")
    print("=" * 70)
    train_df_raw = load_training_data()
    print(f"Loaded (raw): {train_df_raw.shape[0]} rows x {train_df_raw.shape[1]} columns")
    print(f"Columns: {list(train_df_raw.columns)}")
    print()
    train_df = clean_training_data(train_df_raw)
    print()
    print(train_df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe())
    print()
    print("Soil type distribution:")
    print(train_df["soil_type"].value_counts())

    print()
    print("=" * 70)
    print("LOADING TESTING DATA")
    print("=" * 70)
    test_df = load_testing_data()
    print()
    print(f"Loaded: {test_df.shape[0]} rows x {test_df.shape[1]} columns")
    print(f"Columns: {list(test_df.columns)}")
    print(f"Number of distinct source profiles: {test_df['source_sheet'].nunique()}")
    print()
    print(test_df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe())