#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 30, 2026
# Description: This script appends county-level hurricane wind exposure data from the hurricaneexposuredata R package 
# to the analytical panel.
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

STEP5F_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05f_analytical_sample_anchor_exposure_plus_comorb_plus_ahrf_plus_mbsf_demo_cc_otcc_v01"
)

WIND_FILE = (
    "/gpfs/data/cms-share/duas/52484/Jessy/data/public_data/data/"
    "brooke_hurricane/update_ryanzomorrodi/storm_winds.csv"
)

OUT_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05g_analytical_sample_anchor_exposure_plus_comorb_plus_ahrf_plus_mbsf_demo_cc_otcc_plus_wind_v01"
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

def step5f_file(year: int) -> str:
    return os.path.join(STEP5F_BASE, f"year_{year}", "analytical_panel.csv")

def out_year_dir(year: int) -> str:
    return os.path.join(OUT_BASE, f"year_{year}")

def out_file(year: int) -> str:
    return os.path.join(out_year_dir(year), "analytical_panel.csv")

# -------------------------
# Import once and quality check wind data
# -------------------------
if not _exists(WIND_FILE):
    raise FileNotFoundError(f"Wind exposure file not found: {WIND_FILE}")

wind = pd.read_csv(WIND_FILE, dtype={"fips": str}, low_memory=False)

required_wind_cols = ["storm_id", "fips", "vmax_sust"]
missing_wind_cols = [c for c in required_wind_cols if c not in wind.columns]
if missing_wind_cols:
    raise ValueError(f"Wind file is missing required columns: {missing_wind_cols}")

wind["storm_id"] = _as_clean_str(wind["storm_id"])
wind["fips"] = _clean_fips(wind["fips"])
wind["vmax_sust"] = pd.to_numeric(wind["vmax_sust"], errors="coerce")

wind = wind[["storm_id", "fips", "vmax_sust"]].copy()

dup_wind = wind.duplicated(subset=["storm_id", "fips"]).sum()
print(f"[INFO] wind rows = {len(wind):,}")
print(f"[INFO] unique storm_id-fips pairs = {wind[['storm_id','fips']].drop_duplicates().shape[0]:,}")
print(f"[QC] duplicated storm_id-fips pairs in wind file = {dup_wind:,}")

if dup_wind > 0:
    dup_show = (
        wind.loc[wind.duplicated(subset=["storm_id", "fips"], keep=False), ["storm_id", "fips"]]
        .drop_duplicates()
        .sort_values(["storm_id", "fips"])
    )
    print("[ERROR] Example duplicated storm_id-fips pairs:")
    print(dup_show.head(20).to_string(index=False))
    raise ValueError("Wind file has duplicated storm_id-fips pairs. Resolve before merging.")


def process_year(year: int):
    print(f"\n=== Processing year {year} ===")

    f_panel = step5f_file(year) # analytical path from prior script
    if not _exists(f_panel):
        print(f"[SKIP] missing analytical input: {f_panel}")
        return

    analytical = pd.read_csv(f_panel, low_memory=False)
    print(f"[INFO] analytical rows = {len(analytical):,}")

    required_cols = {"event_id", "BENE_ID", "storm_id", "facility_county_fips"}
    missing_required = required_cols - set(analytical.columns)
    if missing_required:
        raise ValueError(
            f"{year}: analytical file missing required columns: {sorted(missing_required)}"
        )

    analytical["event_id"] = pd.to_numeric(analytical["event_id"], errors="coerce").astype("Int64")
    analytical["BENE_ID"] = _as_clean_str(analytical["BENE_ID"])
    analytical["storm_id"] = _as_clean_str(analytical["storm_id"])
    analytical["facility_county_fips"] = _clean_fips(analytical["facility_county_fips"])

    n_rows_before = len(analytical)
    n_pairs_before = analytical[["event_id", "BENE_ID"]].drop_duplicates().shape[0]

    rows_per_pair = analytical.groupby(["event_id", "BENE_ID"], dropna=False).size()
    bad_pairs = (rows_per_pair != 2).sum() # should be two rows per bene-storm (event)
    print(f"[QC] event_id x BENE_ID with !=2 panel rows = {bad_pairs:,}") # print if any rows do not have two

    analytical2 = analytical.merge( # merge so each county-storm gets a winddata
        wind,
        left_on=["storm_id", "facility_county_fips"],
        right_on=["storm_id", "fips"],
        how="left",
        indicator=True
    )

    print("[QC] analytical <- wind merge:")
    print(analytical2["_merge"].value_counts(dropna=False).to_string())

    if len(analytical2) != n_rows_before:
        raise ValueError(
            f"{year}: row count changed after wind merge. "
            f"Before={n_rows_before:,}, After={len(analytical2):,}"
        )

    n_pairs_after = analytical2[["event_id", "BENE_ID"]].drop_duplicates().shape[0]
    if n_pairs_after != n_pairs_before:
        raise ValueError(
            f"{year}: unique event-bene pair count changed after wind merge. "
            f"Before={n_pairs_before:,}, After={n_pairs_after:,}"
        )

    # ... Missingness ...
    analytical2["wind_missing"] = analytical2["vmax_sust"].isna().astype("int8")

    n_missing_rows = analytical2["wind_missing"].sum()
    pct_missing_rows = n_missing_rows / len(analytical2) if len(analytical2) > 0 else np.nan

    pair_missing = (
        analytical2[["event_id", "BENE_ID", "wind_missing"]]
        .drop_duplicates(subset=["event_id", "BENE_ID"])
        .copy()
    )
    n_missing_pairs = pair_missing["wind_missing"].sum()
    pct_missing_pairs = n_missing_pairs / len(pair_missing) if len(pair_missing) > 0 else np.nan

    print(f"[QC] rows missing vmax_sust = {n_missing_rows:,} / {len(analytical2):,} ({pct_missing_rows:.3%})")
    print(f"[QC] event-bene pairs missing vmax_sust = {n_missing_pairs:,} / {len(pair_missing):,} ({pct_missing_pairs:.3%})")

    # ... QC: wind should not vary within bene_storm (event) ...
    chk = analytical2.groupby(["event_id", "BENE_ID"], dropna=False)["vmax_sust"].nunique(dropna=False)
    bad_repeat = (chk > 1).sum()
    if bad_repeat > 0:
        raise ValueError(
            f"{year}: found {bad_repeat:,} event_id-BENE_ID pairs with inconsistent vmax_sust across panel rows."
        )

    analytical2 = analytical2.drop(columns=["_merge"])

    # Drop wind-side FIPS after merge; keep analytical facility_county_fips
    if "fips" in analytical2.columns:
        analytical2 = analytical2.drop(columns=["fips"])

    # Reorder
    preferred_front = [
        "year", "storm_id", "event_id", "BENE_ID", "facility_county_fips",
        "week_rel", "hazard_week",
        "vmax_sust", "wind_missing",
        "combinedscore", "combinedscore_missing",
        "ahrf_missing",
    ]
    front = [c for c in preferred_front if c in analytical2.columns]
    rest = [c for c in analytical2.columns if c not in front]
    analytical2 = analytical2[front + rest]

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

    print(f"\n[DONE] Wind-appended analytical files written under: {OUT_BASE}")