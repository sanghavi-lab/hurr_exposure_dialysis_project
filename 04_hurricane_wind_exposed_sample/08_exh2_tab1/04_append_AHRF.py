#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 29, 2026
# Description: This script appends year-specific county-level AHRF control variables to the analytical panel with
# comorbidity scores by merging on county FIPS. It preserves all original bene rows, checks merge and missingness quality,
# and writes one AHRF-appended analytical file for each year.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import numpy as np
import pandas as pd

# -------------------------
# Paths and spec
# -------------------------
YEAR_MIN, YEAR_MAX = 2011, 2022

STEP5D_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05d_analytical_panel_hurr_exposure_v05_wkm2_facclust_cumpost_cumdeath_plus_comorb_v01"
)

AHRF_FILE = (
    "/gpfs/data/public/AHRF/derived/"
    "ahrf_controls_pct_only_final_phys_optionA_stgenbeds.csv"
)

OUT_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05e_analytical_panel_hurr_exposure_v05_wkm2_facclust_cumpost_cumdeath_plus_comorb_v01_plus_ahrf_v01"
)

os.makedirs(OUT_BASE, exist_ok=True)

# -------------------------
# Functions
# -------------------------
def _exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False

def _as_clean_str(s: pd.Series) -> pd.Series:
    out = s.astype(str)
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return out

def _clean_fips(s: pd.Series) -> pd.Series:
    out = _as_clean_str(s).str.strip()
    out = out.where(out.isna(), out.str.zfill(5))
    return out

def step5d_file(year: int) -> str:
    return os.path.join(STEP5D_BASE, f"year_{year}", "analytical_panel.csv")

def out_year_dir(year: int) -> str:
    return os.path.join(OUT_BASE, f"year_{year}")

def out_file(year: int) -> str:
    return os.path.join(out_year_dir(year), "analytical_panel.csv")

def get_ahrf_cols_for_year(df_ahrf: pd.DataFrame, year: int):
    """
    Select year-specific AHRF columns for a given analytical year.
    Keep stable CBSA columns plus year-matched annual fields when available.
    """
    candidates = [
        f"median_hh_income_{year}",
        f"pct_below_poverty_{year}",
        f"pct_population_65plus_{year}",
        "cbsa_indicator_2020",
        "cbsa_indicator_2020_label",
        f"pct_gen_pract_md_patientcare_{year}",
        f"pct_st_gen_hospital_beds_of_all_hospital_beds_{year}",
        f"st_gen_hospital_beds_per_1000_pop_{year}",
    ]
    keep = ["fips"] + [c for c in candidates if c in df_ahrf.columns]
    return keep

def event_level_missing_share(df: pd.DataFrame, event_id_col: str, varlist):
    """
    Event-level missingness:
    collapse to one row per event_id and compute missing share for each variable.
    """
    tmp = df[[event_id_col] + varlist].drop_duplicates(subset=[event_id_col]).copy()
    out = []
    for c in varlist:
        n_missing = tmp[c].isna().sum()
        pct_missing = n_missing / len(tmp) if len(tmp) > 0 else np.nan
        out.append((c, n_missing, pct_missing))
    return pd.DataFrame(out, columns=["variable", "n_missing_events", "pct_missing_events"])

# ... Read AHRF once ...
if not _exists(AHRF_FILE):
    raise FileNotFoundError(f"AHRF file not found: {AHRF_FILE}")

ahrf = pd.read_csv(AHRF_FILE, low_memory=False)
if "fips" not in ahrf.columns:
    raise ValueError("AHRF file must contain column 'fips'.")

ahrf["fips"] = _clean_fips(ahrf["fips"])

dup_ahrf = ahrf.duplicated(subset=["fips"]).sum()
if dup_ahrf > 0:
    raise ValueError(f"AHRF file has {dup_ahrf:,} duplicated fips values.")

print(f"[INFO] AHRF rows = {len(ahrf):,}")
print(f"[INFO] AHRF unique fips = {ahrf['fips'].nunique(dropna=True):,}")

# ... Merge ...
def process_year(year: int):
    print(f"\n=== Processing year {year} ===")

    f_panel = step5d_file(year)
    if not _exists(f_panel):
        print(f"[SKIP] missing analytical input: {f_panel}")
        return

    # ... Read analytical panel ...
    analytical = pd.read_csv(f_panel, low_memory=False)
    print(f"[INFO] analytical rows = {len(analytical):,}")

    if "event_id" not in analytical.columns:
        raise ValueError(f"{year}: analytical file must contain event_id.")
    if "fips" not in analytical.columns:
        raise ValueError(f"{year}: analytical file must contain fips.")

    analytical["event_id"] = pd.to_numeric(analytical["event_id"], errors="coerce").astype("Int64")
    analytical["fips"] = _clean_fips(analytical["fips"])

    missing_event_id = analytical["event_id"].isna().sum()
    if missing_event_id > 0:
        raise ValueError(f"{year}: analytical file has {missing_event_id:,} rows with missing event_id.")

    # Expected panel structure: 2 rows per event
    rows_per_event = analytical.groupby("event_id", dropna=False).size()
    bad_events = (rows_per_event != 2).sum()
    print(f"[QC] event_id with !=2 panel rows = {bad_events:,}")

    n_rows_before = len(analytical)
    n_events_before = analytical["event_id"].nunique(dropna=True)

    # -------------------------
    # Select year-specific AHRF vars
    # -------------------------
    ahrf_cols = get_ahrf_cols_for_year(ahrf, year)
    ahrf_y = ahrf[ahrf_cols].copy()

    # Identify appended variables
    ahrf_vars = [c for c in ahrf_y.columns if c != "fips"]
    if not ahrf_vars:
        raise ValueError(f"{year}: no AHRF variables found for year {year}.")

    print(f"[INFO] AHRF variables for {year}:")
    for c in ahrf_vars:
        print(f"       - {c}")

    # rename appended vars
    rename_map = {c: f"{c}_ahrf" for c in ahrf_vars}
    ahrf_y = ahrf_y.rename(columns=rename_map)
    ahrf_vars_renamed = list(rename_map.values())

    # -------------------------
    # Merge AHRF to analytical panel by county FIPS
    # -------------------------
    analytical2 = analytical.merge(
        ahrf_y,
        on="fips",
        how="left",
        indicator=True
    )

    print("[QC] analytical <- AHRF merge:")
    print(analytical2["_merge"].value_counts(dropna=False).to_string())

    n_rows_after = len(analytical2)
    if n_rows_after != n_rows_before:
        raise ValueError(
            f"{year}: row count changed after AHRF merge. "
            f"Before={n_rows_before:,}, After={n_rows_after:,}"
        )

    n_events_after = analytical2["event_id"].nunique(dropna=True)
    if n_events_after != n_events_before:
        raise ValueError(
            f"{year}: unique event count changed after AHRF merge. "
            f"Before={n_events_before:,}, After={n_events_after:,}"
        )

    # -------------------------
    # Missingness QC
    # -------------------------

    # Missing AHRF only if all appended AHRF vars are missing
    analytical2["ahrf_missing"] = analytical2[ahrf_vars_renamed].isna().all(axis=1).astype("int8")

    # Row-level missingness
    n_missing_rows = analytical2["ahrf_missing"].sum()
    pct_missing_rows = n_missing_rows / len(analytical2) if len(analytical2) > 0 else np.nan

    # Event-level missingness (one record per bene–storm event)
    event_missing = (
        analytical2[["event_id", "ahrf_missing"]]
        .drop_duplicates(subset=["event_id"])
        .copy()
    )
    n_missing_events = event_missing["ahrf_missing"].sum()
    pct_missing_events = n_missing_events / len(event_missing) if len(event_missing) > 0 else np.nan

    print(f"[QC] rows missing all AHRF vars = {n_missing_rows:,} / {len(analytical2):,} ({pct_missing_rows:.3%})")
    print(f"[QC] events missing all AHRF vars = {n_missing_events:,} / {len(event_missing):,} ({pct_missing_events:.3%})")

    # Variable-specific row-level missingness
    miss_rows = pd.DataFrame({
        "variable": ahrf_vars_renamed,
        "n_missing_rows": [analytical2[c].isna().sum() for c in ahrf_vars_renamed],
        "pct_missing_rows": [analytical2[c].isna().mean() for c in ahrf_vars_renamed],
    })
    print("[QC] row-level missingness by AHRF variable:")
    print(miss_rows.to_string(index=False))

    # Variable-specific event-level missingness
    miss_events = event_level_missing_share(analytical2, "event_id", ahrf_vars_renamed)
    print("[QC] event-level missingness by AHRF variable:")
    print(miss_events.to_string(index=False))

    # -------------------------
    # Additional QCs: AHRF should not vary within event_id
    # -------------------------
    bad_repeat = 0
    for c in ahrf_vars_renamed:
        chk = analytical2.groupby("event_id", dropna=False)[c].nunique(dropna=False)
        bad_repeat += (chk > 1).sum()

    if bad_repeat > 0:
        raise ValueError(
            f"{year}: found {bad_repeat:,} event-by-variable inconsistencies in appended AHRF values."
        )

    analytical2 = analytical2.drop(columns=["_merge"])

    # -------------------------
    # Reorder columns
    # -------------------------
    preferred_front = [
        "year", "storm_year", "storm_id", "event_id", "BENE_ID", "fips",
        "week_rel", "hazard_week",
        "combinedscore", "combinedscore_missing",
        "ahrf_missing",
    ] + ahrf_vars_renamed

    front = [c for c in preferred_front if c in analytical2.columns]
    rest = [c for c in analytical2.columns if c not in front]
    analytical2 = analytical2[front + rest]

    # -------------------------
    # Export
    # -------------------------
    odir = out_year_dir(year)
    os.makedirs(odir, exist_ok=True)

    fout = out_file(year)
    analytical2.to_csv(fout, index=False)

    print(f"[WROTE] {year}: {fout}")
    print(f"[QC] final panel rows = {len(analytical2):,}")
    print(f"[QC] final unique events = {analytical2['event_id'].nunique(dropna=True):,}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        process_year(year)

    print(f"\n[DONE] AHRF-appended analytical files written under: {OUT_BASE}")