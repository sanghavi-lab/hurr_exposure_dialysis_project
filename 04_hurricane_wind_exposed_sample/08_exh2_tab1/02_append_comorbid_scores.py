#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 29, 2026
# Description: This script grabs the outputs from SAS and crosswalk patid -> event_id and merges with the analytical
# file to append the comorbidity scores
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import numpy as np
import pandas as pd

# -------------------------
# Paths and other specs
# -------------------------
YEAR_MIN, YEAR_MAX = 2011, 2022

STEP4_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "04_analytical_panel_hurr_exposure_v05_wkm2_facclust_cumpost_cumdeath"
)

XWALK_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05b_comorbidity_prep_for_sas_from_step4_v01_crosswalk"
)

SAS_OUT_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05c_comorbidity_scores_from_sas_v01"
)

OUT_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05d_analytical_panel_hurr_exposure_v05_wkm2_facclust_cumpost_cumdeath_plus_comorb_v01"
)

os.makedirs(OUT_BASE, exist_ok=True)

# If True, keep 2011 file but print warning about incomplete lookback.
# If False, skip 2011 entirely.
KEEP_2011 = True

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

def step4_file(year: int) -> str:
    return os.path.join(STEP4_BASE, f"year_{year}", "analytical_panel.csv")

def xwalk_file(year: int) -> str:
    return os.path.join(XWALK_BASE, f"year_{year}.csv")

def sas_file(year: int) -> str:
    return os.path.join(SAS_OUT_BASE, f"year_{year}_sas_output.csv")

def out_year_dir(year: int) -> str:
    return os.path.join(OUT_BASE, f"year_{year}")

def out_file(year: int) -> str:
    return os.path.join(out_year_dir(year), "analytical_panel.csv")

# ... Merge ...
def process_year(year: int):
    print(f"\n=== Processing year {year} ===")

    if year == 2011 and not KEEP_2011:
        print("[SKIP] 2011 skipped because prior-year 2010 lookback data are unavailable.")
        return

    f_step4 = step4_file(year)
    f_xwalk = xwalk_file(year)
    f_sas   = sas_file(year)

    missing = [p for p in [f_step4, f_xwalk, f_sas] if not _exists(p)]
    if missing:
        print("[SKIP] missing required input(s):")
        for p in missing:
            print(f"       - {p}")
        return

    # ... Read original analytical panel ...
    analytical = pd.read_csv(f_step4, low_memory=False)
    print(f"[INFO] step4 rows = {len(analytical):,}")

    # Normalize key fields
    if "BENE_ID" in analytical.columns:
        analytical["BENE_ID"] = _as_clean_str(analytical["BENE_ID"])
    if "storm_id" in analytical.columns:
        analytical["storm_id"] = _as_clean_str(analytical["storm_id"])
    if "event_id" in analytical.columns:
        analytical["event_id"] = pd.to_numeric(analytical["event_id"], errors="coerce").astype("Int64")
    if "exposure_start_dt" in analytical.columns:
        analytical["exposure_start_dt"] = pd.to_datetime(
            analytical["exposure_start_dt"], errors="coerce"
        ).dt.normalize()

    # ... Read crosswalk from patid -> event_id ...
    xwalk = pd.read_csv(f_xwalk, low_memory=False)
    print(f"[INFO] crosswalk rows = {len(xwalk):,}")

    if "BENE_ID" in xwalk.columns:
        xwalk["BENE_ID"] = _as_clean_str(xwalk["BENE_ID"])
    if "storm_id" in xwalk.columns:
        xwalk["storm_id"] = _as_clean_str(xwalk["storm_id"])
    if "event_id" in xwalk.columns:
        xwalk["event_id"] = pd.to_numeric(xwalk["event_id"], errors="coerce").astype("Int64")
    if "exposure_start_dt" in xwalk.columns:
        xwalk["exposure_start_dt"] = pd.to_datetime(
            xwalk["exposure_start_dt"], errors="coerce"
        ).dt.normalize()
    if "patid" in xwalk.columns:
        xwalk["patid"] = _as_clean_str(xwalk["patid"])

    # One row per event expected in xwalk
    xwalk = (
        xwalk[["patid", "year", "event_id", "BENE_ID", "storm_id", "exposure_start_dt"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    dup_xwalk = xwalk.duplicated(subset=["patid"]).sum()
    if dup_xwalk > 0:
        raise ValueError(f"{year}: crosswalk has {dup_xwalk:,} duplicated patid values.")

    dup_event = xwalk.duplicated(subset=["event_id"]).sum()
    if dup_event > 0:
        raise ValueError(f"{year}: crosswalk has {dup_event:,} duplicated event_id values.")

    # ... Read SAS comorbidity output ...
    sas = pd.read_csv(f_sas, low_memory=False)
    print(f"[INFO] sas output rows = {len(sas):,}")

    if "patid" not in sas.columns or "combinedscore" not in sas.columns:
        raise ValueError(f"{year}: SAS output must contain patid and combinedscore.")

    sas["patid"] = _as_clean_str(sas["patid"])
    sas["combinedscore"] = pd.to_numeric(sas["combinedscore"], errors="coerce")

    sas = sas[["patid", "combinedscore"]].drop_duplicates().reset_index(drop=True)

    dup_sas = sas.duplicated(subset=["patid"]).sum()
    if dup_sas > 0:
        raise ValueError(f"{year}: SAS output has {dup_sas:,} duplicated patid values.")

    # ... Merge SAS score to crosswalk ...
    score_event = xwalk.merge(sas, on="patid", how="left", indicator=True)

    print("[QC] xwalk <- sas merge:")
    print(score_event["_merge"].value_counts(dropna=False).to_string())

    missing_scores = score_event["combinedscore"].isna().sum()
    print(f"[QC] events missing combinedscore = {missing_scores:,} / {len(score_event):,}")

    score_event = score_event.drop(columns=["_merge"])

    # Keep one row per event_id for merge back to panel
    score_event = (
        score_event[["event_id", "patid", "combinedscore"]]
        .drop_duplicates(subset=["event_id"], keep="first")
        .reset_index(drop=True)
    )

    # ... Merge back to analytical panel by event_id ...
    before_rows = len(analytical)

    analytical2 = analytical.merge(
        score_event[["event_id", "combinedscore"]],
        on="event_id",
        how="left",
        indicator=True
    )

    print("[QC] analytical <- score_event merge:")
    print(analytical2["_merge"].value_counts(dropna=False).to_string())

    if len(analytical2) != before_rows:
        raise ValueError(
            f"{year}: row count changed after merge. Before={before_rows:,}, After={len(analytical2):,}"
        )

    # Confirm score repeats consistently within event_id
    tmp = analytical2.groupby("event_id", dropna=False)["combinedscore"].nunique(dropna=False)
    bad_repeat = (tmp > 1).sum()
    if bad_repeat > 0:
        raise ValueError(
            f"{year}: found {bad_repeat:,} event_id values with inconsistent combinedscore across panel rows."
        )

    analytical2 = analytical2.drop(columns=["_merge"])

    # Flag for missing score
    analytical2["combinedscore_missing"] = analytical2["combinedscore"].isna().astype("int8")

    # Put combinedscore near other baseline fields if possible
    preferred_order = [
        "year", "storm_year", "storm_id", "event_id", "BENE_ID", "fips",
        "week_rel", "hazard_week",
        "combinedscore", "combinedscore_missing"
    ]
    front = [c for c in preferred_order if c in analytical2.columns]
    rest = [c for c in analytical2.columns if c not in front]
    analytical2 = analytical2[front + rest]

    # ... Export ...
    odir = out_year_dir(year)
    os.makedirs(odir, exist_ok=True)

    fout = out_file(year)
    analytical2.to_csv(fout, index=False)

    # ... QCs ...
    n_panel_rows = len(analytical2)
    n_events = analytical2["event_id"].nunique(dropna=True) if "event_id" in analytical2.columns else np.nan
    n_missing = analytical2["combinedscore"].isna().sum()
    pct_missing = n_missing / n_panel_rows if n_panel_rows > 0 else np.nan

    print(f"[WROTE] {year}: {fout}")
    print(f"[QC] panel rows = {n_panel_rows:,}")
    print(f"[QC] unique events = {n_events:,}")
    print(f"[QC] rows missing combinedscore = {n_missing:,} ({pct_missing:.3%})")

    if year == 2011:
        print("[NOTE] 2011 scores may have incomplete 365-day lookback because 2010 dx data were unavailable.")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        process_year(year)

    print(f"\n[DONE] Appended analytical files written under: {OUT_BASE}")
