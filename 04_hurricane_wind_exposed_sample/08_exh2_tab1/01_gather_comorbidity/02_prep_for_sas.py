#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 28, 2026
# Description: This script takes the raw inpatient and outpatient diagnosis claims created in the prior step and prepares 
# them for SAS-based comorbidity construction at the bene-storm (event) level. For each storm year, it builds one unique 
# bene-storm record from the analytical panel, creates a unique patid and 365-day lookback window, links all diagnosis 
# claims for that beneficiary from the current and prior year, keeps only claims within the lookback period, reshapes 
# the diagnosis fields into a long format with ICD-9 versus ICD-10 code type, and writes both parquet and single-file 
# CSV outputs for SAS.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client

# -------------------------
# Dask setup
# -------------------------
cust_temp_dir = "/gpfs/data/cms-share/duas/52484/Jessy/temp_space/tmp/"
dask.config.set({"temporary-directory": cust_temp_dir})
dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s"
})
client = Client("10.50.87.31:42109")
print(client)

# -------------------------
# Paths and other specs
# -------------------------
YEAR_MIN, YEAR_MAX = 2011, 2022

STEP4_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "04_analytical_panel_hurr_exposure_v05_wkm2_facclust_cumpost_cumdeath"
)

RAW_DX_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05a_comorbidity_raw_ip_op_from_step4_v01"
)

OUT_PARQUET_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05b_comorbidity_prep_for_sas_from_step4_v01"
)

OUT_CSV_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05b_comorbidity_prep_for_sas_from_step4_v01_csv"
)

OUT_XWALK_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05b_comorbidity_prep_for_sas_from_step4_v01_crosswalk"
)

os.makedirs(OUT_PARQUET_BASE, exist_ok=True)
os.makedirs(OUT_CSV_BASE, exist_ok=True)
os.makedirs(OUT_XWALK_BASE, exist_ok=True)

LOOKBACK_DAYS = 365 # gather all dx codes up to one year prior to exposure to storm.

# -------------------------
# Functions
# -------------------------
def _exists(p: str) -> bool: # check if path exists
    try:
        return os.path.exists(p)
    except Exception:
        return False

def _as_clean_str(s): # clean
    s = s.astype(str)
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return s

def step4_file(year: int) -> str:
    return os.path.join(STEP4_BASE, f"year_{year}", "analytical_panel.csv")

def raw_dx_dir(year: int) -> str:
    return os.path.join(RAW_DX_BASE, f"year={year}")

def out_parquet_dir(year: int) -> str:
    return os.path.join(OUT_PARQUET_BASE, f"year={year}")

def out_csv_file(year: int) -> str:
    return os.path.join(OUT_CSV_BASE, f"year_{year}.csv")

def out_crosswalk_file(year: int) -> str:
    return os.path.join(OUT_XWALK_BASE, f"year_{year}.csv")


# ... Read unique bene-storm (event) file from analytical ...
def load_unique_events(year: int) -> pd.DataFrame:
    # Build one unique row per bene-storm (event) for comorbidity construction.
    # We use week_rel == 0 rows only so we get one row per bene-storm (event). Recall that analytical has two rows per bene-storm (event)
    
    f = step4_file(year)
    if not _exists(f):
        print(f"[SKIP] {year}: missing STEP 4 analytical file -> {f}")
        return pd.DataFrame()

    usecols = ["year", "event_id", "BENE_ID", "storm_id", "exposure_start_dt", "week_rel"]
    df = dd.read_csv(
        f,
        usecols=usecols,
        dtype={
            "year": "float64",
            "event_id": "float64",
            "BENE_ID": "object",
            "storm_id": "object",
            "exposure_start_dt": "object",
            "week_rel": "float64",
        },
        assume_missing=True
    ).compute()

    df["BENE_ID"] = _as_clean_str(df["BENE_ID"])
    df["storm_id"] = _as_clean_str(df["storm_id"])
    df["exposure_start_dt"] = pd.to_datetime(df["exposure_start_dt"], errors="coerce").dt.normalize()

    df = df[df["week_rel"] == 0].copy() # use one of the two weeks
    df = df[df["exposure_start_dt"].notna()].copy()

    # Make sure these are integer-like where appropriate
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")

    df = (
        df[["year", "event_id", "BENE_ID", "storm_id", "exposure_start_dt"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Create patid for SAS (unique identifier for each bene-storm (event)
    df["patid"] = (
        "y" + df["year"].astype(str) +
        "_e" + df["event_id"].astype(str) +
        "_b" + df["BENE_ID"].astype(str) +
        "_s" + df["storm_id"].astype(str)
    )

    # 365-day lookback start (will be used to gather all dx within 1 year of exposure)
    df["lookback_start_dt"] = df["exposure_start_dt"] - pd.Timedelta(days=LOOKBACK_DAYS)

    return df


# ... Read raw dx claims created in previous script ...
def load_raw_dx_for_year_window(year: int) -> dd.DataFrame | None:
    # For storm-year Y, read raw dx claims from year Y and year Y-1.
    # This covers the 365-day lookback
    
    parts = [] # to store df list in order to concat later

    this_year_dir = raw_dx_dir(year)
    prev_year_dir = raw_dx_dir(year - 1) # read in previous year since we are looking back one year

    if _exists(this_year_dir):
        parts.append(dd.read_parquet(this_year_dir, engine="pyarrow"))
    else:
        print(f"[WARN] {year}: missing raw dx current-year dir -> {this_year_dir}")

    if year - 1 >= YEAR_MIN and _exists(prev_year_dir):
        parts.append(dd.read_parquet(prev_year_dir, engine="pyarrow"))
    else:
        if year - 1 >= YEAR_MIN:
            print(f"[WARN] {year}: missing raw dx prior-year dir -> {prev_year_dir}")

    if len(parts) == 0:
        return None

    raw_dx = dd.concat(parts, axis=0, interleave_partitions=True)
    raw_dx["BENE_ID"] = raw_dx["BENE_ID"].astype(str)
    raw_dx["SRVC_BGN_DT"] = dd.to_datetime(raw_dx["SRVC_BGN_DT"], errors="coerce")

    return raw_dx


# ... ICD code type assignment ...
def assign_icd_type(df: pd.DataFrame) -> pd.DataFrame:
    # ICD-9 before 2015-10-01, ICD-10 on/after 2015-10-01.
    
    cutoff = pd.Timestamp("2015-10-01")

    df["Dx_CodeType"] = np.where(
        df["SRVC_BGN_DT"] < cutoff,
        "09", # icd9
        "10" # icd10
    )
    return df

# ... Prepare parquet ...
def prepare_for_sas(year: int):
    print(f"\n=== preparing year {year} ===")

    events = load_unique_events(year)
    if events.empty:
        print(f"[SKIP] {year}: no events found")
        return

    print(f"[INFO] {year}: unique events = {len(events):,}")
    print(f"[INFO] {year}: unique benes  = {events['BENE_ID'].nunique():,}")

    # Save for later merge-back
    xwalk = events[["year", "event_id", "BENE_ID", "storm_id", "exposure_start_dt", "patid"]].copy()
    xwalk.to_csv(out_crosswalk_file(year), index=False)
    print(f"[WROTE] crosswalk -> {out_crosswalk_file(year)}")

    raw_dx = load_raw_dx_for_year_window(year)
    if raw_dx is None:
        print(f"[SKIP] {year}: no raw dx found")
        return

    # Keep only fields we need
    dx_columns = [f"dx{i}" for i in range(1, 27)]
    keep_cols = ["BENE_ID", "SRVC_BGN_DT"] + [c for c in dx_columns if c in raw_dx.columns]
    raw_dx = raw_dx[keep_cols]

    # Merge events to claims on BENE_ID only, then apply lookback window.
    events_dd = dd.from_pandas(
        events[["year", "event_id", "BENE_ID", "storm_id", "exposure_start_dt", "lookback_start_dt", "patid"]],
        npartitions=1
    )
    events_dd["BENE_ID"] = events_dd["BENE_ID"].astype(str)

    merged = events_dd.merge(raw_dx, on="BENE_ID", how="inner")

    # Keep dx claims within 365 days prior to and including exposure_start_dt
    merged = merged[
        (merged["SRVC_BGN_DT"] >= merged["lookback_start_dt"]) &
        (merged["SRVC_BGN_DT"] <= merged["exposure_start_dt"])
    ]

    # Build long format for SAS
    dx_columns_present = [c for c in dx_columns if c in merged.columns]
    df_to_melt = merged[["patid", "SRVC_BGN_DT"] + dx_columns_present]

    df_long = dd.melt(
        df_to_melt,
        id_vars=["patid", "SRVC_BGN_DT"],
        value_vars=dx_columns_present,
        value_name="DX",
        var_name="column_names"
    )

    df_long = df_long.drop(columns=["column_names"])

    # Assign ICD code type 
    df_long = df_long.map_partitions(assign_icd_type)

    # Clean diagnosis values
    df_long["DX"] = df_long["DX"].astype(str).str.strip()

    na_patterns = [r"^na$", r"^nan$", r"^Na$", r"^NaN$", r"^NA$", r"^$", r"^\s*$"]
    df_long = df_long[~df_long["DX"].str.contains("|".join(na_patterns), case=False, na=True)]

    # Uppercase DX for consistency
    df_long["DX"] = df_long["DX"].str.upper()

    # Drop claim date after Dx_CodeType assignment
    df_long = df_long.drop(columns=["SRVC_BGN_DT"])

    # Keep only SAS-required columns
    df_long = df_long[["patid", "DX", "Dx_CodeType"]]

    # Convert all columns to string for SAS compatibility
    df_long = df_long.astype(str)

    # Write parquet
    outdir = out_parquet_dir(year)
    os.makedirs(outdir, exist_ok=True)

    df_long.to_parquet(
        outdir,
        engine="pyarrow",
        compression="gzip",
        write_index=False,
        overwrite=True
    )

    print(f"[WROTE] parquet prep -> {outdir}")


# ... Convert parquet to CSV for SAS ...
def convert_to_csv(year: int):
    print(f"[INFO] converting year {year} parquet prep to CSV")

    in_dir = out_parquet_dir(year)
    if not _exists(in_dir):
        print(f"[SKIP] {year}: missing parquet prep dir -> {in_dir}")
        return

    # Read in the parquet from before
    df = dd.read_parquet(in_dir, engine="pyarrow")

    # Group same patid together to make deduplication easier
    df = df.set_index("patid", drop=False)
    df = df.persist()

    # Deduplicate
    df = df.map_partitions(
        lambda part: part.drop_duplicates(subset=["patid", "DX", "Dx_CodeType"]) # important! do not want multiple dx of the same code.
    )

    df.to_csv(
        out_csv_file(year),
        index=False,
        single_file=True
    )

    print(f"[WROTE] csv for SAS -> {out_csv_file(year)}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        prepare_for_sas(year)
        convert_to_csv(year)

    print("\n[DONE] SAS prep files written.")