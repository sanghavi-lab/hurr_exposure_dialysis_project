#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 20, 2026
# Description: This code takes the county-level hurricane exposure file from the previous step and links it to Medicare 
# beneficiaries using the MBSF monthly county-of-residence fields. In other words, it identifies any beneficiaries who 
# were living in counties exposed to each storm during the relevant exposure month, then writes out a beneficiary–storm 
# exposure dataset. It does this year by year and month by month, reading only the needed MBSF monthly county columns.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
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
client = Client("[redacted]")
print(client)

# -------------------------
# Paths and other specs
# -------------------------
EXPOSURE_PATH = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/hurricane_county_exposure_start_v02_track64kt_2011_2022/county_storm_exposure_with_startdate_2011_2022_ts17ms.csv"

YEAR_MIN, YEAR_MAX = 2011, 2022
MBSF_BASE = "/gpfs/data/cms-share/data/medicare/{year}/mbsf/mbsf_abcd/parquet/"

OUT_BASE = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis"
OUT_DIR  = os.path.join(OUT_BASE, "bene_storm_exposure_clean_schema_dask_month_match_2011_2022")
os.makedirs(OUT_DIR, exist_ok=True)

DROP_DUPES_BENE_STORM = False  # True => one row per (BENE_ID, storm_id) within each year-month. Ended up being the same True or False so just kept it at False

# -------------------------
# Functions
# -------------------------
def month_col(mm: int) -> str:
    return f"STATE_CNTY_FIPS_CD_{mm:02d}"

def parse_track_datetime_series(s: pd.Series) -> pd.Series:
    # date like "198808051800" -> YYYYMMDDHHMM
    return pd.to_datetime(s.astype(str), format="%Y%m%d%H%M", errors="coerce")

def read_exposure_table_csv(path: str) -> pd.DataFrame:

    # Import
    df = pd.read_csv(path, dtype={"fips": str, "storm_id": str})

    if "storm_year" not in df.columns:
        df["storm_year"] = df["storm_id"].str[-4:].astype(int)

    # Build exposure_start_dt
    if "exposure_start_dt" in df.columns:
        df["exposure_start_dt"] = pd.to_datetime(df["exposure_start_dt"], errors="coerce")
    elif "exposure_start_trackdate" in df.columns:
        # Avoid scientific notation problems
        df["exposure_start_trackdate"] = df["exposure_start_trackdate"].apply(
            lambda x: (str(int(x)) if pd.notna(x) else pd.NA)
        )
        df["exposure_start_dt"] = parse_track_datetime_series(df["exposure_start_trackdate"])
    else:
        raise ValueError("Exposure table needs exposure_start_dt or exposure_start_trackdate")

    df = df[df["exposure_start_dt"].notna()].copy()
    df["month"] = df["exposure_start_dt"].dt.month

    # Standardize fips as 5-char string (strip .0 then zfill)
    df["fips"] = df["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)

    # Filter to target years and valid month
    df = df[(df["storm_year"] >= YEAR_MIN) & (df["storm_year"] <= YEAR_MAX)].copy()
    df = df[df["month"].between(1, 12, inclusive="both")].copy()

    # Keep minimal join columns
    df = df[["storm_id", "storm_year", "month", "fips", "exposure_start_dt"]].drop_duplicates()

    # Force pandas month to int (prevents float carry-over)
    df["month"] = df["month"].astype(int)

    return df

def read_mbsf_cols_dask(year: int, cols: list[str]) -> dd.DataFrame:

    pq_path = MBSF_BASE.format(year=year)

    # Need to do this because some files has bene id in index and some do not...
    try:
        m = dd.read_parquet(pq_path, columns=cols, engine="pyarrow")
        if "BENE_ID" not in m.columns:
            if m._meta.index.name == "BENE_ID":
                m = m.reset_index()
            else:
                raise ValueError(f"{year}: BENE_ID not found as column or index.")
    except Exception:
        m = dd.read_parquet(pq_path, columns=[c for c in cols if c != "BENE_ID"], engine="pyarrow")
        if m._meta.index.name == "BENE_ID":
            m = m.reset_index()
        else:
            raise

    return m

def enforce_clean_schema(out: dd.DataFrame) -> dd.DataFrame:
    # Force a stable schema across all files BEFORE writing.
    # This is to eliminate ArrowTypeError downstream.

    out["BENE_ID"] = out["BENE_ID"].astype(str)
    out["storm_id"] = out["storm_id"].astype(str)
    out["fips"] = out["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)

    # Fixed integer widths
    out["storm_year"] = out["storm_year"].astype("int16")
    out["month"] = out["month"].astype("int8")

    # Datetime
    out["exposure_start_dt"] = dd.to_datetime(out["exposure_start_dt"], errors="coerce")
    out = out[out["exposure_start_dt"].notnull()]

    return out[["BENE_ID", "storm_id", "storm_year", "month", "fips", "exposure_start_dt"]]

# -------------------------
# Main
# -------------------------
exp = read_exposure_table_csv(EXPOSURE_PATH) # focus on 17 m/s
print(f"Exposure rows: {len(exp):,}")
print(f"Unique storms: {exp['storm_id'].nunique():,} | Unique counties: {exp['fips'].nunique():,}")

for year in range(YEAR_MIN, YEAR_MAX + 1):
    exp_y = exp[exp["storm_year"] == year].copy() # grabs one year all counties exposed to at least 17 m/s
    if exp_y.empty:
        print(f"[SKIP] {year}: no exposures")
        continue

    months = sorted(exp_y["month"].unique().tolist())
    months = [int(m) for m in months]  # ensures int

    # Get cols to read from mbsf
    month_cols = [month_col(m) for m in months]
    cols_to_read = ["BENE_ID"] + month_cols

    print(f"[INFO] {year}: reading MBSF columns: {cols_to_read}")
    m = read_mbsf_cols_dask(year, cols_to_read) # import mbsf

    # Month by month, take the mbsf county-of-residence field for that month, convert it into a clean 5-digit county FIPS, and match beneficiaries to the exposure table on fips
    for mm in months:
        col = month_col(mm) # month col

        exp_ym = exp_y[exp_y["month"] == mm][["storm_id", "storm_year", "month", "fips", "exposure_start_dt"]].copy()
        if exp_ym.empty:
            continue

        exp_ym_dd = dd.from_pandas(exp_ym, npartitions=1) # convert pd to dd since i am using dd

        m_mm = m[["BENE_ID", col]].copy() # keep only relevant month exposed
        m_mm = m_mm[m_mm[col].notnull()]

        # clean month county fips
        m_mm["fips"] = (
            m_mm[col]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(5)
        )
        exp_ym_dd["fips"] = exp_ym_dd["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)

        # Important step: keep only beneficiaries from the mbsf if they were living in a county exposed to hurricane in that same month. Note that these are NOT dialysis bene's. That will come in the next step.
        out = m_mm[["BENE_ID", "fips"]].merge(exp_ym_dd, on="fips", how="inner")
        out = out[["BENE_ID", "storm_id", "storm_year", "month", "fips", "exposure_start_dt"]]

        if DROP_DUPES_BENE_STORM:
            out = out.drop_duplicates(subset=["BENE_ID", "storm_id"])

        # Enforce stable schema BEFORE write. Ensures no error from ArrowTypeError
        out = enforce_clean_schema(out)

        out_path = os.path.join(OUT_DIR, f"year_{year}", f"month_{mm:02d}")
        os.makedirs(out_path, exist_ok=True)

        out.to_parquet(
            out_path,
            write_index=False,
            engine="pyarrow",
            overwrite=True
        )

print(f"\nOutputs: {OUT_DIR}")




