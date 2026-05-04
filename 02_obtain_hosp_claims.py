#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 20, 2026
# Description: This code processes medpar inpatient claims files from 2011–2022 and keeps only stays from 
# short-stay and long-stay hospitals, excluding SNF claims.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import gc
import dask
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd

# -------------------------
# Configuration
# -------------------------
YEARS = list(range(2011, 2023))  # 2011-2022 inclusive

MEDICARE_BASE = "/gpfs/data/cms-share/data/medicare"
OUTPUT_BASE = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/00b_hospital_SL"

# -------------------------
# Dask setup
# -------------------------
cust_temp_dir = "/gpfs/data/cms-share/duas/52484/Jessy/temp_space/tmp/"
dask.config.set({"temporary-directory": cust_temp_dir})
dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s"
})

client = Client("10.50.87.115:33209")
print(client)

# -------------------------
# Paths
# -------------------------
def get_input_path(year):
    return f"{MEDICARE_BASE}/{year}/medpar/parquet"


def get_output_dir(year):
    return f"{OUTPUT_BASE}/{year}/"


# -------------------------
# Columns
# -------------------------
def get_columns():
    return [
        "MEDPAR_ID",
        "PRVDR_NUM",
        "ADMSN_DT",
        "DSCHRG_DT",
        "SS_LS_SNF_IND_CD"
    ]


# -------------------------
# Read data
# -------------------------
def read_medpar(src, use_cols):
    """
    Read MedPAR and make sure BENE_ID is available as a column.
    Some years may store BENE_ID as the index instead of a regular column.
    """
    med = dd.read_parquet(src, columns=use_cols, engine="pyarrow")
    med = med.reset_index()

    if "BENE_ID" not in med.columns:
        bene_like = [c for c in med.columns if c.lower() in ("bene_id", "beneid")]
        if bene_like:
            med = med.rename(columns={bene_like[0]: "BENE_ID"})
        elif "index" in med.columns:
            med = med.rename(columns={"index": "BENE_ID"})
        else:
            raise ValueError(f"Could not find BENE_ID. Columns are: {list(med.columns)}")

    return med


# -------------------------
# Type cleanup
# -------------------------
def normalize_types(med):
    med["ADMSN_DT"] = dd.to_datetime(med["ADMSN_DT"], errors="coerce")
    med["DSCHRG_DT"] = dd.to_datetime(med["DSCHRG_DT"], errors="coerce")
    med["PRVDR_NUM"] = med["PRVDR_NUM"].astype("string")
    med["SS_LS_SNF_IND_CD"] = med["SS_LS_SNF_IND_CD"].astype("string")
    return med


# -------------------------
# Filtering logic
# -------------------------

# Technically this function is not needed anymore but I left it because there was a time when we wanted to filter to a specific time frame.
def build_year_window(year): 
    win_start = pd.Timestamp(f"{year}-01-01")
    win_end = pd.Timestamp(f"{year}-12-31")
    return win_start, win_end


def filter_short_long_hospital_stays(med, win_start, win_end):
    # Keep MedPAR stays that overlap the calendar year and are short-stay ('S') or long-stay ('L') hospital claims. Excludes SNF ('N')
    overlaps = (med["ADMSN_DT"] <= win_end) & (med["DSCHRG_DT"] >= win_start)
    is_hosp_sl = med["SS_LS_SNF_IND_CD"].fillna("").str.upper().isin(["S", "L"])

    keep_cols = [
        "BENE_ID",
        "MEDPAR_ID",
        "PRVDR_NUM",
        "ADMSN_DT",
        "DSCHRG_DT",
        "SS_LS_SNF_IND_CD"
    ]

    hosp_sl = med.loc[overlaps & is_hosp_sl, keep_cols]
    return hosp_sl


# -------------------------
# Output
# -------------------------
def write_output(hosp_sl, year):
    out_dir = get_output_dir(year)
    os.makedirs(out_dir, exist_ok=True)

    hosp_sl.to_parquet(
        out_dir,
        engine="pyarrow",
        compression="gzip",
        write_index=False,
        overwrite=True,
    )


# -------------------------
# Process one year
# -------------------------
def process_year(year):
    print(f"{year}")

    # Grab correct year's path from Medpar
    src = get_input_path(year)

    # Grab set of columns needed
    use_cols = get_columns()

    # Reads the medpar file
    med = read_medpar(src, use_cols)

    # Converts variables into the right data types
    med = normalize_types(med)

    # Defines the date window for the year. We are using all year so technically not needed but left it since there was a time we wanted to explore a specific date window
    win_start, win_end = build_year_window(year)

    # Keeps only hospital claims (short or long stay)
    hosp_sl = filter_short_long_hospital_stays(med, win_start, win_end)

    # Counts
    n_rows = hosp_sl.index.size.compute()
    print(f"Short/long hospital rows to write: {n_rows:,}")

    # Export
    write_output(hosp_sl, year)
    print(f"Wrote short/long hospital stays to: {get_output_dir(year)}")

    # Recover memory
    del med, hosp_sl
    gc.collect()

    return {
        "year": year,
        "n_rows_hospital_sl": int(n_rows)
    }


# -------------------------
# Main
# -------------------------
def main():
    yearly_results = []

    for year in YEARS:
        yearly_results.append(process_year(year)) # See comments above for further explanation of this function.

    print("\nFinished.")
    for row in yearly_results:
        print(row)

    return yearly_results


if __name__ == "__main__":
    final_results = main()