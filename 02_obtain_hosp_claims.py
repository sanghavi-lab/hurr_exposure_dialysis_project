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

OUTPUT_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/"
    "climate_change/dialysis/00b_hospital_SL"
)

# -------------------------
# Dask setup
# -------------------------

cust_temp_dir = "/gpfs/data/cms-share/duas/52484/Jessy/temp_space/tmp/"

dask.config.set({
    "temporary-directory": cust_temp_dir
})

dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s"
})

client = Client("[redacted]")
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
    """
    Columns needed from the MedPAR data other than BENE_ID.

    BENE_ID is handled separately in read_medpar() because its storage
    differs across years:
        2011-2016: Parquet index
        2017-2022: regular column
    """
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
    Read MedPAR while accounting for differences in how BENE_ID is stored.

    In the source MedPAR files:
        - 2011-2016: BENE_ID is stored as the Parquet index.
        - 2017-2022: BENE_ID is stored as a regular column.

    This function checks the source structure and handles either case.
    """

    # Read only metadata/lazy structure first so we can determine
    # whether BENE_ID is a regular column or the index.
    med_structure = dd.read_parquet(
        src,
        engine="pyarrow"
    )

    available_cols = list(med_structure.columns)
    index_name = med_structure.index.name

    # --------------------------------------------------
    # Case 1: BENE_ID is stored as a regular column
    # e.g., 2017-2022
    # --------------------------------------------------
    if "BENE_ID" in available_cols:

        cols_to_read = ["BENE_ID"] + [
            col for col in use_cols
            if col != "BENE_ID"
        ]

        med = dd.read_parquet(
            src,
            columns=cols_to_read,
            engine="pyarrow"
        )

    # --------------------------------------------------
    # Case 2: BENE_ID is stored as the Parquet index
    # e.g., 2011-2016
    # --------------------------------------------------
    elif index_name == "BENE_ID":

        med = dd.read_parquet(
            src,
            columns=use_cols,
            engine="pyarrow"
        )

        # Convert BENE_ID from index to regular column
        med = med.reset_index()

    # --------------------------------------------------
    # If neither is found, stop rather than incorrectly
    # treating a generic row index as beneficiary ID.
    # --------------------------------------------------
    else:

        raise ValueError(
            "Could not locate BENE_ID in the MedPAR source data.\n"
            f"Source: {src}\n"
            f"Index name: {index_name}\n"
            f"Available columns: {available_cols}"
        )

    # Final safety check
    if "BENE_ID" not in med.columns:
        raise ValueError(
            "BENE_ID was not successfully recovered after reading MedPAR.\n"
            f"Source: {src}\n"
            f"Columns after read: {list(med.columns)}"
        )

    return med


# -------------------------
# Type cleanup
# -------------------------

def normalize_types(med):

    med["ADMSN_DT"] = dd.to_datetime(
        med["ADMSN_DT"],
        errors="coerce"
    )

    med["DSCHRG_DT"] = dd.to_datetime(
        med["DSCHRG_DT"],
        errors="coerce"
    )

    med["PRVDR_NUM"] = med["PRVDR_NUM"].astype("string")

    med["SS_LS_SNF_IND_CD"] = (
        med["SS_LS_SNF_IND_CD"].astype("string")
    )

    return med


# -------------------------
# Filtering logic
# -------------------------

# Technically, this function is not needed anymore, but it is retained because there was a time when we wanted to filter to a specific time frame (e.g., summer only). 
def build_year_window(year):

    win_start = pd.Timestamp(f"{year}-01-01")
    win_end = pd.Timestamp(f"{year}-12-31")

    return win_start, win_end


def filter_short_long_hospital_stays(
    med,
    win_start,
    win_end
):
    """
    Keep MedPAR stays that overlap the calendar year and are
    short-stay ('S') or long-stay ('L') hospital claims.

    SNF claims ('N') are excluded.
    """

    # Hospital stay overlaps the calendar year
    overlaps = (
        (med["ADMSN_DT"] <= win_end)
        &
        (med["DSCHRG_DT"] >= win_start)
    )

    # Short-stay or long-stay hospitals only
    is_hosp_sl = (
        med["SS_LS_SNF_IND_CD"]
        .fillna("")
        .str.upper()
        .isin(["S", "L"])
    )

    keep_cols = [
        "BENE_ID",
        "MEDPAR_ID",
        "PRVDR_NUM",
        "ADMSN_DT",
        "DSCHRG_DT",
        "SS_LS_SNF_IND_CD"
    ]

    hosp_sl = med.loc[
        overlaps & is_hosp_sl,
        keep_cols
    ]

    return hosp_sl


# -------------------------
# Output
# -------------------------

def write_output(hosp_sl, year):

    out_dir = get_output_dir(year)

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    hosp_sl.to_parquet(
        out_dir,
        engine="pyarrow",
        compression="gzip",
        write_index=False,
        overwrite=True
    )


# -------------------------
# Process one year
# -------------------------

def process_year(year):

    print(f"\nProcessing {year}")

    # Grab correct year's path from MedPAR
    src = get_input_path(year)

    # Grab set of columns needed
    use_cols = get_columns()

    # Read MedPAR.
    # read_medpar() handles the difference in BENE_ID storage
    # between 2011-2016 and 2017-2022.
    med = read_medpar(
        src,
        use_cols
    )

    # Confirm BENE_ID successfully exists
    print(
        f"{year}: BENE_ID successfully loaded "
        f"(index name after read: {med.index.name})"
    )

    # Convert variables into the correct data types
    med = normalize_types(med)

    # Define the date window for the year.
    # We are currently using the full calendar year.
    win_start, win_end = build_year_window(year)

    # Keep only short-stay and long-stay hospital claims
    hosp_sl = filter_short_long_hospital_stays(
        med,
        win_start,
        win_end
    )

    # Count rows
    n_rows = hosp_sl.index.size.compute()

    print(
        f"Short/long hospital rows to write: {n_rows:,}"
    )

    # Export
    write_output(
        hosp_sl,
        year
    )

    print(
        f"Wrote short/long hospital stays to: "
        f"{get_output_dir(year)}"
    )

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

        yearly_results.append(
            process_year(year)
        )

    print("\nFinished.")

    for row in yearly_results:
        print(row)

    return yearly_results


if __name__ == "__main__":
    final_results = main()
