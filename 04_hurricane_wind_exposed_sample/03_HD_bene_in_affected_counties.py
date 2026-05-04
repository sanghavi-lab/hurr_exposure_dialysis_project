#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 20, 2026
# Description: This code links the beneficiary–storm exposure file to hemodialysis claim lines occurring in the 2 months 
# before through 2 months after each storm exposure date. Basically, it keeps only HD line items if they were from
# beneficiaries living in counties during the month the counties were exposed to the hurricane (based on track time stamp)
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

client = Client("10.50.87.228:38999")
print(client)

# -------------------------
# Paths and spec
# -------------------------
YEAR_MIN, YEAR_MAX = 2011, 2022

BENE_EXPOSURE_DIR = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "bene_storm_exposure_clean_schema_dask_month_match_2011_2022"
)

HD_YEAR_DIR_TMPL = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/{year}/"
) # HD line items

OUT_BASE = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis"
OUT_DIR  = os.path.join(OUT_BASE, "bene_storm_hd_services_pm1mo_byyear_v01")
os.makedirs(OUT_DIR, exist_ok=True)

# Define date window to collect HD line items (relative to exposure_start_dt):
WIN_BEFORE_MONTHS = 2  # two months prior
WIN_AFTER_MONTHS  = 2  # two months after

# "inner" => keeps only exposures where HD exists in the window (dialysis-only cohort)
# "left"  => keeps all exposures; HD fields will be null if no HD lines
JOIN_HOW = "inner"

# Cols
HD_COLS = ["BENE_ID", "CLM_ID", "REV_CNTR", "REV_CNTR_DT", "PRVDR_NUM"]

# -------------------------
# Functions
# -------------------------
def _month_diff(a: pd.Period, b: pd.Period) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)

def _expand_to_candidate_months(pdf: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "BENE_ID", "storm_id", "storm_year", "fips",
        "exposure_start_dt", "win_start", "win_end",
        "svc_year", "svc_month", "svc_bucket"
    ]

    # --- Return empty but correctly-typed DataFrame ---
    # This is just a safeguard. If a partition ends up empty, the function can still return an empty table with the correct structure
    empty = pd.DataFrame({
        "BENE_ID": pd.Series(dtype="object"),
        "storm_id": pd.Series(dtype="object"),
        "storm_year": pd.Series(dtype="int16"),
        "fips": pd.Series(dtype="object"),
        "exposure_start_dt": pd.Series(dtype="datetime64[ns]"),
        "win_start": pd.Series(dtype="datetime64[ns]"),
        "win_end": pd.Series(dtype="datetime64[ns]"),
        "svc_year": pd.Series(dtype="int64"),
        "svc_month": pd.Series(dtype="int64"),
        "svc_bucket": pd.Series(dtype="object"),
    })

    # Makes a copy of the input pandas partition and converts exposure_start_dt to datetime
    pdf = pdf.copy()
    pdf["exposure_start_dt"] = pd.to_datetime(pdf["exposure_start_dt"], errors="coerce")
    pdf = pdf[pdf["exposure_start_dt"].notna()]

    # Exit if partition is empty after filtering
    if pdf.empty:
        return empty

    # Creates the actual date window around each exposure
    pdf["win_start"] = pdf["exposure_start_dt"] - pd.DateOffset(months=WIN_BEFORE_MONTHS)
    pdf["win_end"]   = pdf["exposure_start_dt"] + pd.DateOffset(months=WIN_AFTER_MONTHS)

    out_rows = []
    for r in pdf.itertuples(index=False):
        start_m = pd.Timestamp(r.win_start).to_period("M")
        end_m   = pd.Timestamp(r.win_end).to_period("M")
        exp_m   = pd.Timestamp(r.exposure_start_dt).to_period("M")

        # ^ the function processes one bene-storm exposure row at a time. For each row, it creates three month-level objects. These convert the window start, window end, and exposure date into monthly periods, like 2012-10, instead of exact day-level timestamps.

        # Builds all calendar months touched by that window (2 months prior and 2 months after)
        months = pd.period_range(start_m, end_m, freq="M")

        for m in months:
            rel = _month_diff(m, exp_m) # for each of those months, calculate how far that month is from the exposure month
            out_rows.append({ # For each month, the function appends one row to out_rows containing
                "BENE_ID": r.BENE_ID,
                "storm_id": r.storm_id,
                "storm_year": r.storm_year,
                "fips": r.fips,
                "exposure_start_dt": r.exposure_start_dt,
                "win_start": r.win_start,
                "win_end": r.win_end,
                "svc_year": int(m.year),
                "svc_month": int(m.month),
                "svc_bucket": f"m{rel:+d}",
            })

    # Guard: out_rows could still be empty if all period_ranges were empty
    if not out_rows:
        return empty

    out = pd.DataFrame(out_rows)

    out = out.drop_duplicates(
        subset=["BENE_ID", "storm_id", "fips", "exposure_start_dt", "svc_year", "svc_month"]
    )

    return out[keep_cols]

def _safe_exists(path: str) -> bool: # checks whether a file path or folder path exists
    try:
        return os.path.exists(path)
    except Exception:
        return False

def _read_hd_year(y: int) -> dd.DataFrame | None:
    p = HD_YEAR_DIR_TMPL.format(year=y) # build path
    if not _safe_exists(p): # Basically, check before trying to read the HD parquet folder. If the path is missing, inaccessible, malformed, or causes some unexpected filesystem issue, the script skips that year instead of failing.
        return None

    # Imports
    hd = dd.read_parquet(p, engine="pyarrow", columns=HD_COLS)

    # Clean: normalize types
    hd["BENE_ID"] = hd["BENE_ID"].astype(str)
    hd["CLM_ID"]  = hd["CLM_ID"].astype(str)
    hd["REV_CNTR"] = hd["REV_CNTR"].astype(str).str.zfill(4)
    hd["PRVDR_NUM"] = (
        hd["PRVDR_NUM"]
          .astype(str)
          .str.replace(r"\.0$", "", regex=True)
          .str.strip()
          .str.zfill(6)
    )

    hd["REV_CNTR_DT"] = dd.to_datetime(hd["REV_CNTR_DT"], errors="coerce")
    hd = hd[hd["REV_CNTR_DT"].notnull()]

    # VERY IMPORTANT: takes actual service date and extract the calendar year and month. This is needed to merge later to grab relevant line items.
    hd["svc_year"]  = hd["REV_CNTR_DT"].dt.year.astype("int16")
    hd["svc_month"] = hd["REV_CNTR_DT"].dt.month.astype("int8")
    return hd

# -------------------------
# 1) Load exposures once (then subset per storm_year)
# -------------------------
bene_exp_all = dd.read_parquet(BENE_EXPOSURE_DIR, engine="pyarrow")

# Clean: normalize types
bene_exp_all["BENE_ID"] = bene_exp_all["BENE_ID"].astype(str)
bene_exp_all["storm_id"] = bene_exp_all["storm_id"].astype(str)
bene_exp_all["storm_year"] = bene_exp_all["storm_year"].astype("int16")
bene_exp_all["fips"] = (bene_exp_all["fips"].astype(str)
                        .str.replace(r"\.0$", "", regex=True)
                        .str.zfill(5))

bene_exp_all["exposure_start_dt"] = dd.to_datetime(bene_exp_all["exposure_start_dt"], errors="coerce")
bene_exp_all = bene_exp_all[bene_exp_all["exposure_start_dt"].notnull()]
bene_exp_all = bene_exp_all[(bene_exp_all["storm_year"] >= YEAR_MIN) & (bene_exp_all["storm_year"] <= YEAR_MAX)] # specified as 2011-2022 but 2022 does not have wind data so ultimately will be skipped in script

# Meta for map_partitions expansion. Will tell Dask what the output schema will look like.
meta = pd.DataFrame({
    "BENE_ID": pd.Series(dtype="object"),
    "storm_id": pd.Series(dtype="object"),
    "storm_year": pd.Series(dtype="int16"),
    "fips": pd.Series(dtype="object"),
    "exposure_start_dt": pd.Series(dtype="datetime64[ns]"),
    "win_start": pd.Series(dtype="datetime64[ns]"),
    "win_end": pd.Series(dtype="datetime64[ns]"),
    "svc_year": pd.Series(dtype="int16"),
    "svc_month": pd.Series(dtype="int8"),
    "svc_bucket": pd.Series(dtype="object"), # which calendar month a dialysis claim belongs to. E.g., bucket m-2 = two months before the exposure month
})

# -------------------------
# 2) Process year-by-year
# -------------------------
for year in range(YEAR_MIN, YEAR_MAX + 1):
    print(f"\nProcessing storm_year={year}")

    bene_exp_y = bene_exp_all[bene_exp_all["storm_year"] == year]

    # Emptiness check (e.g., like for 2022 where no wind data)
    n_exp = bene_exp_y.shape[0].compute()
    if n_exp == 0:
        print(f"[SKIP] {year}: no exposures")
        continue
    print(f"[INFO] {year}: exposures rows = {n_exp:,}")

    # Expand to candidate months spanning full [-2mo, +2mo] window
    expM = bene_exp_y.map_partitions(_expand_to_candidate_months, meta=meta)
        # Basically, take the bene-storm exposures for this year (from the previous script), and for each one, create all the month-level rows needed to later join to dialysis claims in the 2 months before and after exposure. E.g., if exposure date is 10-29-2012 then will create rows with August 2012, September 2012, October 2012, November 2012, December 2012. These will be used to obtain HD line items.

    # Read HD lines for (year-1, year, year+1)
    # NOTE: for +/- 2 months, year-1 and year+1 is sufficient for Jan/Dec cases.
    hd_parts = []
    for y in [year - 1, year, year + 1]:
        hd_y = _read_hd_year(y)
        if hd_y is not None:
            hd_parts.append(hd_y)

    if not hd_parts:
        print(f"[SKIP] {year}: no HD inputs found for {year-1},{year},{year+1}")
        continue

    hd_all = dd.concat(hd_parts, axis=0, interleave_partitions=True) # concat all years

    # Match each beneficiary-storm row to any dialysis claim lines for that same beneficiary in that same calendar month
    joined = expM.merge(
        hd_all,
        on=["BENE_ID", "svc_year", "svc_month"],
        how=JOIN_HOW
    )

    # Keeps only dialysis lines whose service date is actually between win_start and win_end (ie 2 months prior and after)
    joined = joined[
        (joined["REV_CNTR_DT"].notnull()) &
        (joined["REV_CNTR_DT"] >= joined["win_start"]) &
        (joined["REV_CNTR_DT"] <= joined["win_end"])
    ]

    # Keep tidy columns
    joined = joined[[
        "BENE_ID", "storm_id", "storm_year", "fips",
        "exposure_start_dt", "win_start", "win_end",
        "svc_bucket",
        "CLM_ID", "PRVDR_NUM", "REV_CNTR", "REV_CNTR_DT"
    ]]

    # Output paths for this storm_year
    year_dir = os.path.join(OUT_DIR, f"year={year}")
    os.makedirs(year_dir, exist_ok=True)

    detail_out = os.path.join(year_dir, "detail_hd_lines")
    summary_out = os.path.join(year_dir, "summary_bene_storm_hd")

    # Write detail
    joined.to_parquet(detail_out, engine="pyarrow", write_index=False, overwrite=True)

    # Build summary (unique-day count).
    joined2 = joined.assign(hd_day=joined["REV_CNTR_DT"].dt.floor("D")) # This creates a new variable hd_day, which is just the service datetime rounded down to the calendar day. That matters because there can be multiple claim lines on the same day and the summary wants both the number of lines and the number of unique dialysis days.
    gcols = ["BENE_ID", "storm_id", "storm_year", "fips", "exposure_start_dt"]

    # This counts the number of claim lines for each bene-storm row.
    lines = joined2.groupby(gcols).CLM_ID.count().rename("n_hd_lines")

    # This computes the number of unique service days rather than the number of lines.
    uniq_days = joined2[gcols + ["hd_day"]].drop_duplicates()
    days = uniq_days.groupby(gcols).hd_day.count().rename("n_hd_days")

    # Combines the line count and unique-day count
    summary = dd.concat([lines, days], axis=1).reset_index()
    summary["has_hd"] = (summary["n_hd_lines"] > 0).astype("int8")

    # Export summary
    summary.to_parquet(summary_out, engine="pyarrow", write_index=False, overwrite=True)

print(f"\nOutputs under: {OUT_DIR}")
