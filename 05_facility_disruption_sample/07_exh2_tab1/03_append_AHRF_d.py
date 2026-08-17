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
    "05d_analytical_sample_anchor_exposure_plus_comorb_v01"
)

AHRF_FILE = (
    "/gpfs/data/public/AHRF/derived/"
    "ahrf_controls_pct_only_final_phys_optionA_stgenbeds.csv"
)

OUT_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05e_analytical_sample_anchor_exposure_plus_comorb_plus_ahrf_v01"
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
    out = out.replace({"": pd.NA})
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

def pair_level_missing_share(df: pd.DataFrame, id_cols, varlist):
    """
    Event-bene-level missingness:
    collapse to one row per (event_id, BENE_ID) pair and compute missing share.
    """
    tmp = df[id_cols + varlist].drop_duplicates(subset=id_cols).copy()
    out = []
    for c in varlist:
        n_missing = tmp[c].isna().sum()
        pct_missing = n_missing / len(tmp) if len(tmp) > 0 else np.nan
        out.append((c, n_missing, pct_missing))
    return pd.DataFrame(out, columns=["variable", "n_missing_pairs", "pct_missing_pairs"])

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

    required_cols = {"event_id", "BENE_ID", "facility_county_fips"}
    missing_required = required_cols - set(analytical.columns)
    if missing_required:
        raise ValueError(
            f"{year}: analytical file missing required columns: {sorted(missing_required)}"
        )

    analytical["event_id"] = pd.to_numeric(analytical["event_id"], errors="coerce").astype("Int64")
    analytical["BENE_ID"] = _as_clean_str(analytical["BENE_ID"])
    analytical["facility_county_fips"] = _clean_fips(analytical["facility_county_fips"])

    missing_event_id = analytical["event_id"].isna().sum()
    if missing_event_id > 0:
        raise ValueError(f"{year}: analytical file has {missing_event_id:,} rows with missing event_id.")

    missing_bene = analytical["BENE_ID"].isna().sum()
    if missing_bene > 0:
        raise ValueError(f"{year}: analytical file has {missing_bene:,} rows with missing BENE_ID.")

    # Expected panel structure: 2 rows per event_id x BENE_ID pair
    rows_per_pair = analytical.groupby(["event_id", "BENE_ID"], dropna=False).size()
    bad_pairs = (rows_per_pair != 2).sum()
    print(f"[QC] event_id x BENE_ID with !=2 panel rows = {bad_pairs:,}")

    n_rows_before = len(analytical)
    n_pairs_before = analytical[["event_id", "BENE_ID"]].drop_duplicates().shape[0]

    # -------------------------
    # Select year-specific AHRF vars
    # -------------------------
    ahrf_cols = get_ahrf_cols_for_year(ahrf, year)
    ahrf_y = ahrf[ahrf_cols].copy()

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
        left_on="facility_county_fips",
        right_on="fips",
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

    n_pairs_after = analytical2[["event_id", "BENE_ID"]].drop_duplicates().shape[0]
    if n_pairs_after != n_pairs_before:
        raise ValueError(
            f"{year}: unique event-bene pair count changed after AHRF merge. "
            f"Before={n_pairs_before:,}, After={n_pairs_after:,}"
        )

    # -------------------------
    # Missingness QC
    # -------------------------
    analytical2["ahrf_missing"] = analytical2[ahrf_vars_renamed].isna().all(axis=1).astype("int8")

    # Row-level missingness
    n_missing_rows = analytical2["ahrf_missing"].sum()
    pct_missing_rows = n_missing_rows / len(analytical2) if len(analytical2) > 0 else np.nan

    # Event-bene-level missingness
    pair_missing = (
        analytical2[["event_id", "BENE_ID", "ahrf_missing"]]
        .drop_duplicates(subset=["event_id", "BENE_ID"])
        .copy()
    )
    n_missing_pairs = pair_missing["ahrf_missing"].sum()
    pct_missing_pairs = n_missing_pairs / len(pair_missing) if len(pair_missing) > 0 else np.nan

    print(f"[QC] rows missing all AHRF vars = {n_missing_rows:,} / {len(analytical2):,} ({pct_missing_rows:.3%})")
    print(f"[QC] event-bene pairs missing all AHRF vars = {n_missing_pairs:,} / {len(pair_missing):,} ({pct_missing_pairs:.3%})")

    # Variable-specific row-level missingness
    miss_rows = pd.DataFrame({
        "variable": ahrf_vars_renamed,
        "n_missing_rows": [analytical2[c].isna().sum() for c in ahrf_vars_renamed],
        "pct_missing_rows": [analytical2[c].isna().mean() for c in ahrf_vars_renamed],
    })
    print("[QC] row-level missingness by AHRF variable:")
    print(miss_rows.to_string(index=False))

    # Variable-specific pair-level missingness
    miss_pairs = pair_level_missing_share(
        analytical2,
        ["event_id", "BENE_ID"],
        ahrf_vars_renamed
    )
    print("[QC] event-bene-level missingness by AHRF variable:")
    print(miss_pairs.to_string(index=False))

    # -------------------------
    # Additional QCs: AHRF should not vary within event_id
    # -------------------------
    bad_repeat = 0
    for c in ahrf_vars_renamed:
        chk = analytical2.groupby(["event_id", "BENE_ID"], dropna=False)[c].nunique(dropna=False)
        bad_repeat += (chk > 1).sum()

    if bad_repeat > 0:
        raise ValueError(
            f"{year}: found {bad_repeat:,} event-bene-by-variable inconsistencies in appended AHRF values."
        )

    analytical2 = analytical2.drop(columns=["_merge"])

    # Keep only one county key in final file
    if "fips" in analytical2.columns:
        analytical2 = analytical2.drop(columns=["fips"])

    # -------------------------
    # Reorder columns
    # -------------------------
    preferred_front = [
        "year", "storm_id", "event_id", "BENE_ID", "facility_county_fips",
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
    print(f"[QC] final unique event-bene pairs = {analytical2[['event_id', 'BENE_ID']].drop_duplicates().shape[0]:,}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        process_year(year)

    print(f"\n[DONE] AHRF-appended analytical files written under: {OUT_BASE}")