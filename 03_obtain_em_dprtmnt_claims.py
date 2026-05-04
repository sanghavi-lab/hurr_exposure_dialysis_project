#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 20, 2026
# Description: This script processes outpatient revenue files from 2011 through 2022 and keeps rows that are emergency 
# department services based on revenue center codes 0450–0459 and 0981.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import gc
import dask
import dask.dataframe as dd
from dask.distributed import Client

# -------------------------
# Configuration
# -------------------------
YEARS = list(range(2011, 2023))  # 2011-2022 inclusive

MEDICARE_BASE = "/gpfs/data/cms-share/data/medicare"
OUTPUT_BASE = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/00c"

# -------------------------
# Dask setup
# -------------------------
cust_temp_dir = "/gpfs/data/cms-share/duas/52484/Jessy/temp_space/tmp/"
dask.config.set({"temporary-directory": cust_temp_dir})
dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s"
})

client = Client("10.50.87.62:44091")
print(client)

# -------------------------
# Paths
# -------------------------
def get_input_path(year):
    return f"{MEDICARE_BASE}/{year}/otpt/opr/parquet/"


def get_output_dir(year):
    return f"{OUTPUT_BASE}/{year}/"


# -------------------------
# Columns
# -------------------------
def get_columns():
    return [
        "BENE_ID",
        "CLM_ID",
        "REV_CNTR",
        "REV_CNTR_DT"
    ]


# -------------------------
# Read data
# -------------------------
def read_opr(src, use_cols):
    opr = dd.read_parquet(
        src,
        engine="pyarrow",
        columns=use_cols
    )
    return opr


# -------------------------
# Type and field cleanup
# -------------------------
def normalize_fields(opr):
    # Keep only digits from REV_CNTR, then zero-pad to 4 characters
    opr["REV_CNTR"] = (
        opr["REV_CNTR"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna("")
        .str.zfill(4)
    )

    opr["REV_CNTR_DT"] = dd.to_datetime(opr["REV_CNTR_DT"], errors="coerce")
    return opr


# -------------------------
# Filtering logic
# -------------------------
def filter_ed_rows(opr):
    # Keep outpatient revenue center rows that indicate ED services:
    #  - 0450-0459: ED facility 
    #  - 0981: ER professional fees

    ed_mask = (
        opr["REV_CNTR"].str.match(r"^045[0-9]$") |
        opr["REV_CNTR"].eq("0981")
    )

    keep_cols = [
        "BENE_ID",
        "CLM_ID",
        "REV_CNTR",
        "REV_CNTR_DT"
    ]

    ed_rows = opr.loc[ed_mask, keep_cols]
    return ed_rows


# -------------------------
# Optional deduplication
# -------------------------
def dedupe_claim_day(ed_rows):
    # Optional: If I only need one row per claim-day rather than all ED revenue rows, drop duplicate CLM_ID + REV_CNTR_DT combinations after removing missing dates.
    ed_rows = (
        ed_rows
        .dropna(subset=["REV_CNTR_DT"])
        .drop_duplicates(subset=["CLM_ID", "REV_CNTR_DT"])
    )
    return ed_rows


# -------------------------
# Output
# -------------------------
def write_output(ed_rows, year):
    out_dir = get_output_dir(year)
    os.makedirs(out_dir, exist_ok=True)

    ed_rows.to_parquet(
        out_dir,
        engine="pyarrow",
        compression="gzip",
        overwrite=True
    )


# -------------------------
# Process each year
# -------------------------
def process_year(year, dedupe=False): # dedupe would drop would call below function to keep unique CLM_ID + REV_CNTR_DT. But, False for now since I may need all of the claims.
    print(f"{year}")

    # Get paths and columns
    src = get_input_path(year)
    use_cols = get_columns()

    # Import, clean data, and keep ED visits. (Remember that I am using revenue cent line items)
    opr = read_opr(src, use_cols)
    opr = normalize_fields(opr)
    ed_rows = filter_ed_rows(opr)

    if dedupe:
        ed_rows = dedupe_claim_day(ed_rows)

    n_rows = ed_rows.index.size.compute()
    print(f"ED rows to write: {n_rows:,}")

    write_output(ed_rows, year)
    print(f"Wrote ED-filtered outpatient revenue rows to: {get_output_dir(year)}")

    # Recover memory
    del opr, ed_rows
    gc.collect()

    return {
        "year": year,
        "n_rows_ed": int(n_rows)
    }


# -------------------------
# Main
# -------------------------
def main():
    yearly_results = []

    for year in YEARS:
        yearly_results.append(process_year(year, dedupe=False))

    print("\nFinished.")
    for row in yearly_results:
        print(row)

    return yearly_results


if __name__ == "__main__":
    final_results = main()