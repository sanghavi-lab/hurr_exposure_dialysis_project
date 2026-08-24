#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 29, 2026
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
cust_temp_dir = "/gpfs/data/cms-share/duas/54200/Jessy/temp_space/tmp/"
dask.config.set({"temporary-directory": cust_temp_dir})
dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s"
})

client = Client("[redacted]")
print(client)

# -------------------------
# Paths and other specs
# -------------------------
YEAR_MIN, YEAR_MAX = 2011, 2022
LOOKBACK_DAYS = 365

# Analytical disrupted
def analytical_file(year: int) -> str:
    return (
        "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/"
        f"dialysis/01_analytical_sample/esrd_crossover_{year}/"
        "analytical_simple_case_crossover_anchor_exposure_refwk_m2_early_wkm1_class_wkm3_cumpost_cumdeath_v03.csv"
    )

# Raw dx
RAW_DX_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05a_comorbidity_raw_ip_op_from_exposure_anchor_v02"
)

OUT_PARQUET_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05b_comorbidity_prep_for_sas_from_exposure_anchor_v02"
)

OUT_CSV_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05b_comorbidity_prep_for_sas_from_exposure_anchor_v02_csv"
)

OUT_XWALK_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05b_comorbidity_prep_for_sas_from_exposure_anchor_v02_crosswalk"
)

os.makedirs(OUT_PARQUET_BASE, exist_ok=True)
os.makedirs(OUT_CSV_BASE, exist_ok=True)
os.makedirs(OUT_XWALK_BASE, exist_ok=True)

DX_COLS = [f"dx{i}" for i in range(1, 27)]

# -------------------------
# Functions
# -------------------------
def _exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


def _as_clean_str(s):
    s = s.astype(str)
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return s


def raw_dx_dir(year: int) -> str:
    return os.path.join(RAW_DX_BASE, f"year={year}")


def out_parquet_dir(year: int) -> str:
    return os.path.join(OUT_PARQUET_BASE, f"year={year}")


def out_csv_file(year: int) -> str:
    return os.path.join(OUT_CSV_BASE, f"year_{year}.csv")


def out_crosswalk_file(year: int) -> str:
    return os.path.join(OUT_XWALK_BASE, f"year_{year}.csv")


def load_unique_events(year: int) -> pd.DataFrame:
    """
    Build one unique row per bene-storm/event for comorbidity construction.

    The analytical file has a two-row panel, usually week -2 and week 0.
    I collapse to one row per bene-storm (event) using one unique combination of:
      year, event_id, BENE_ID, anchor_dt

    This event file is then merged to raw dx by BENE_ID, and the exact
    365-day lookback window is applied at the event level.
    """

    f = analytical_file(year)

    if not _exists(f):
        print(f"[SKIP] {year}: missing analytical file -> {f}")
        return pd.DataFrame()

    usecols = ["event_id", "BENE_ID", "anchor_dt"]

    df = dd.read_csv(
        f,
        usecols=usecols,
        dtype={
            "event_id": "float64",
            "BENE_ID": "object",
            "anchor_dt": "object",
        },
        assume_missing=True
    ).compute()

    df["year"] = year
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")
    df["BENE_ID"] = _as_clean_str(df["BENE_ID"])
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"], errors="coerce").dt.normalize()

    df = df.dropna(subset=["event_id", "BENE_ID", "anchor_dt"]).copy()

    df = (
        df[["year", "event_id", "BENE_ID", "anchor_dt"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if df.empty:
        print(f"[SKIP] {year}: no event-bene rows found")
        return pd.DataFrame()

    # Create patid for SAS: one unique ID per bene-storm (event)
    df["patid"] = (
        "y" + df["year"].astype(str) +
        "_e" + df["event_id"].astype(str) +
        "_b" + df["BENE_ID"].astype(str)
    )

    # 365-day lookback window
    df["lookback_start_dt"] = df["anchor_dt"] - pd.Timedelta(days=LOOKBACK_DAYS)

    return df


def load_raw_dx_for_year_window(year: int) -> dd.DataFrame | None:
    """
    For analytical year Y, read raw dx claims from claim year Y and Y-1.

    Example:
      If anchor_dt is July 2020, the 365-day lookback can include claims
      from July 2019 through July 2020. Therefore, we need both 2019 and 2020
      raw dx folders.
    """

    parts = []

    this_year_dir = raw_dx_dir(year)
    prev_year_dir = raw_dx_dir(year - 1)

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


def assign_icd_type(df: pd.DataFrame) -> pd.DataFrame:
    #ICD-9 before 2015-10-01.
    # ICD-10 on or after 2015-10-01.

    cutoff = pd.Timestamp("2015-10-01")
    df["Dx_CodeType"] = np.where(df["SRVC_BGN_DT"] < cutoff, "09", "10")
    return df


# ... Prepare long dx file for SAS ...
def prepare_for_sas(year: int):
    print(f"\n=== Preparing year {year} ===")

    events = load_unique_events(year)

    if events.empty:
        print(f"[SKIP] {year}: no events found")
        return

    print(f"[INFO] {year}: unique event-bene rows = {len(events):,}")
    print(f"[INFO] {year}: unique benes = {events['BENE_ID'].nunique():,}")
    print(f"[INFO] {year}: unique events = {events['event_id'].nunique():,}")

    # Save crosswalk for merge-back after SAS creates comorbidity scores
    xwalk = events[["year", "event_id", "BENE_ID", "anchor_dt", "patid"]].copy()
    xwalk.to_csv(out_crosswalk_file(year), index=False)
    print(f"[WROTE] crosswalk -> {out_crosswalk_file(year)}")

    raw_dx = load_raw_dx_for_year_window(year)

    if raw_dx is None:
        print(f"[SKIP] {year}: no raw dx found")
        return

    # Keep only fields needed for SAS prep
    keep_cols = ["BENE_ID", "SRVC_BGN_DT"] + [c for c in DX_COLS if c in raw_dx.columns]
    raw_dx = raw_dx[keep_cols]

    # Merge event rows to raw claims by BENE_ID.
    # This creates one copy of a claim for each bene-storm (event) where the bene appears.
    events_dd = dd.from_pandas(
        events[["year", "event_id", "BENE_ID", "anchor_dt", "lookback_start_dt", "patid"]],
        npartitions=1
    )

    events_dd["BENE_ID"] = events_dd["BENE_ID"].astype(str)
    events_dd["anchor_dt"] = dd.to_datetime(events_dd["anchor_dt"], errors="coerce")
    events_dd["lookback_start_dt"] = dd.to_datetime(events_dd["lookback_start_dt"], errors="coerce")

    merged = events_dd.merge(raw_dx, on="BENE_ID", how="inner")

    # Apply 365-day lookback window.
    merged = merged[
        (merged["SRVC_BGN_DT"] >= merged["lookback_start_dt"]) &
        (merged["SRVC_BGN_DT"] <= merged["anchor_dt"])
    ]

    dx_cols_present = [c for c in DX_COLS if c in merged.columns]

    if len(dx_cols_present) == 0:
        print(f"[SKIP] {year}: no dx columns present after merge")
        return

    # Build long format for SAS
    df_to_melt = merged[["patid", "SRVC_BGN_DT"] + dx_cols_present]

    df_long = dd.melt(
        df_to_melt,
        id_vars=["patid", "SRVC_BGN_DT"],
        value_vars=dx_cols_present,
        value_name="DX",
        var_name="dx_position"
    )

    df_long = df_long.drop(columns=["dx_position"])

    # Assign ICD type before dropping service date
    df_long = df_long.map_partitions(assign_icd_type)

    # Clean diagnosis code values
    df_long["DX"] = df_long["DX"].astype(str).str.strip().str.upper()

    na_patterns = [r"^na$", r"^nan$", r"^<na>$", r"^none$", r"^$", r"^\s*$"]
    df_long = df_long[~df_long["DX"].str.contains("|".join(na_patterns), case=False, na=True)]

    # Drop service date after ICD type assignment
    df_long = df_long.drop(columns=["SRVC_BGN_DT"])

    # Keep only SAS-required columns
    df_long = df_long[["patid", "DX", "Dx_CodeType"]]

    # Deduplicate so the same diagnosis code does not appear repeatedly for the same patid
    df_long = df_long.shuffle(on="patid")
    df_long = df_long.map_partitions(
        lambda part: part.drop_duplicates(subset=["patid", "DX", "Dx_CodeType"])
    )

    # Convert all fields to string for SAS compatibility
    df_long = df_long.astype(str)

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


# -------------------------
# Convert parquet to CSV for SAS
# -------------------------
def convert_to_csv(year: int):
    print(f"[INFO] converting year {year} parquet prep to CSV")

    in_dir = out_parquet_dir(year)

    if not _exists(in_dir):
        print(f"[SKIP] {year}: missing parquet prep dir -> {in_dir}")
        return

    df = dd.read_parquet(in_dir, engine="pyarrow")

    # Deduplicate again just in case
    df = df.shuffle(on="patid")
    df = df.map_partitions(
        lambda part: part.drop_duplicates(subset=["patid", "DX", "Dx_CodeType"])
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
