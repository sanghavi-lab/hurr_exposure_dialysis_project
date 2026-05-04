#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 20, 2026
# Description: This script identifies Medicare beneficiaries with outpatient hemodialysis claims from 2011-2022 using 
# the outpatient revenue-center line files. It flags hemodialysis lines based on revenue center codes beginning with 082
# and excludes claims where dialysis was done in the emergency department setting.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import gc
import csv
import dask
import dask.dataframe as dd
from dask.distributed import Client

# -------------------------
# Configuration
# -------------------------
YEARS = list(range(2011, 2023))  # 2011-2022 inclusive

DIALYSIS_BASE = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis"

# temp/supporting path for pooled unique bene counting
TEMP_UNIQUE_BENE_BASE = os.path.join(DIALYSIS_BASE, "_tmp_hd_unique_benes_2011_2022")

# final counts file
COUNTS_OUTFILE = os.path.join(DIALYSIS_BASE, "hd_not_ed_bene_counts_2011_2022.csv")

# -------------------------
# Dask setup
# -------------------------
cust_temp_dir = "/gpfs/data/cms-share/duas/52484/Jessy/temp_space/tmp/"
dask.config.set({"temporary-directory": cust_temp_dir})
dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s"
})

client = Client("10.50.87.6:45779")
print(client)

# -------------------------
# Paths
# -------------------------
def get_input_paths(year):
    opr_dir = f"/gpfs/data/cms-share/data/medicare/{year}/otpt/opr/parquet/"
    opb_dir = f"/gpfs/data/cms-share/data/medicare/{year}/otpt/opb/parquet/"
    return opr_dir, opb_dir


def get_output_dir(year):
    return f"/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/{year}/"


# -------------------------
# Columns
# -------------------------
def get_columns():
    col_rev = ["BENE_ID", "CLM_ID", "REV_CNTR", "REV_CNTR_DT"]
    col_claims = [
        "CLM_ID",
        "CLM_FROM_DT",
        "NCH_CLM_TYPE_CD",
        "CLM_FAC_TYPE_CD",
        "CLM_SRVC_CLSFCTN_TYPE_CD",
        "PRVDR_NUM"
    ]
    return col_rev, col_claims


# -------------------------
# Read data
# -------------------------
def read_opr(opr_dir, col_rev):
    opr = dd.read_parquet(opr_dir, engine="pyarrow", columns=col_rev)
    opr["REV_CNTR"] = opr["REV_CNTR"].astype(str).str.zfill(4)
    opr["REV_CNTR_DT"] = dd.to_datetime(opr["REV_CNTR_DT"], errors="coerce")
    return opr


def read_opb(opb_dir, col_claims):
    opb = dd.read_parquet(opb_dir, engine="pyarrow", columns=col_claims)
    opb["CLM_FAC_TYPE_CD"] = opb["CLM_FAC_TYPE_CD"].astype(str)
    opb["CLM_SRVC_CLSFCTN_TYPE_CD"] = opb["CLM_SRVC_CLSFCTN_TYPE_CD"].astype(str)
    opb["PRVDR_NUM"] = opb["PRVDR_NUM"].astype("string")
    return opb


# -------------------------
# OPR processing
# -------------------------
def add_hd_ed_flags(opr):
    opr["is_hd"] = opr["REV_CNTR"].str.startswith("082")
    opr["is_ed"] = opr["REV_CNTR"].str.match(r"^(045[0-9]|0981)$")
    return opr


def build_claim_has_ed(opr):
    claim_has_ed = (
        opr.groupby("CLM_ID")["is_ed"]
        .max()
        .rename("claim_has_ed")
        .reset_index()
    )
    return claim_has_ed


def extract_hd_not_ed(opr, claim_has_ed):
    hd_only = opr[opr["is_hd"]].merge(claim_has_ed, on="CLM_ID", how="left")
    hd_not_ed = hd_only[~hd_only["claim_has_ed"].fillna(False)][
        ["BENE_ID", "CLM_ID", "REV_CNTR", "REV_CNTR_DT"]
    ]
    return hd_not_ed


# -------------------------
# OPB merge
# -------------------------
def attach_provider_number(hd_not_ed, opb):
    hd_not_ed = (
        hd_not_ed.merge(opb[["CLM_ID", "PRVDR_NUM"]], on="CLM_ID", how="left")
        .astype({"PRVDR_NUM": "string"})
    )
    return hd_not_ed


# -------------------------
# Appendix (just to count)
# -------------------------
def compute_yearly_unique_benes(hd_not_ed):
    return hd_not_ed["BENE_ID"].dropna().nunique().compute()


def write_yearly_unique_benes_temp(hd_not_ed, year):
    temp_out = os.path.join(TEMP_UNIQUE_BENE_BASE, f"year={year}")
    unique_benes = hd_not_ed[["BENE_ID"]].dropna().drop_duplicates()
    unique_benes.to_parquet(temp_out, engine="pyarrow", compression="gzip", overwrite=True)


def compute_pooled_unique_benes_from_temp(start_year=None, end_year=None):
    filters = []
    if start_year is not None:
        filters.append(("year", ">=", start_year))
    if end_year is not None:
        filters.append(("year", "<=", end_year))

    if filters:
        pooled = dd.read_parquet(
            TEMP_UNIQUE_BENE_BASE,
            engine="pyarrow",
            columns=["BENE_ID"],
            filters=filters
        )
    else:
        pooled = dd.read_parquet(
            TEMP_UNIQUE_BENE_BASE,
            engine="pyarrow",
            columns=["BENE_ID"]
        )

    pooled_n = (
        pooled["BENE_ID"]
        .dropna()
        .drop_duplicates()
        .count()
        .compute()
    )
    return pooled_n


# -------------------------
# Write output
# -------------------------
def write_main_output(hd_not_ed, year):
    out_dir = get_output_dir(year)
    hd_not_ed.to_parquet(out_dir, engine="pyarrow", compression="gzip", overwrite=True)


# -------------------------
# Process year by year
# -------------------------
def process_year(year):
    print(f"{year}")

    # Grab paths
    opr_dir, opb_dir = get_input_paths(year)

    # List of columns to read from OPR and OPB
    col_rev, col_claims = get_columns()

    # Read files
    opr = read_opr(opr_dir, col_rev)
    opb = read_opb(opb_dir, col_claims)

    # Create flags for HD and ED.
    opr = add_hd_ed_flags(opr)

    # Checks whether any row in the claim has an ED revenue center
    claim_has_ed = build_claim_has_ed(opr)

    # Keeps only HD claims not from ED
    hd_not_ed = extract_hd_not_ed(opr, claim_has_ed)

    # Attach provider number
    hd_not_ed = attach_provider_number(hd_not_ed, opb)

    # Annual count for flowchart
    n_unique_benes = compute_yearly_unique_benes(hd_not_ed)

    # Export counts to reference later (e.g., flowchart, etc..._
    write_yearly_unique_benes_temp(hd_not_ed, year)

    # Export HD data line items
    write_main_output(hd_not_ed, year)

    # Preserve memory
    del opr, opb, claim_has_ed, hd_not_ed
    gc.collect()

    return {
        "year": year,
        "n_unique_benes_hd_not_ed": int(n_unique_benes)
    }


# -------------------------
# Main
# -------------------------
def main():
    # Creates the temp folder
    os.makedirs(TEMP_UNIQUE_BENE_BASE, exist_ok=True)

    yearly_results = []
    for year in YEARS:
        yearly_results.append(process_year(year)) # Returns dictionary for counting

    # Sort the list of dictionaries
    yearly_results = sorted(yearly_results, key=lambda x: x["year"])

    # Counts. Ie how many unique beneficiaries are there in total from 2011 to 2021?
    pooled_unique_benes = compute_pooled_unique_benes_from_temp(
        start_year=2011,
        end_year=2021
    )

    # Adds one more dictionary to the end of yearly_results - the unique num of bene across 2011-2021
    yearly_results.append({
        "year": "2011-2021 pooled unique",
        "n_unique_benes_hd_not_ed": int(pooled_unique_benes)
    })

    # Open the CSV file so it can write the results of the counts into it
    with open(COUNTS_OUTFILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["year", "n_unique_benes_hd_not_ed"]
        )
        writer.writeheader()
        writer.writerows(yearly_results)

    print("\nFinished.")
    for row in yearly_results:
        print(row)
    print(f"\nCounts saved to path: {COUNTS_OUTFILE}")

    return yearly_results


if __name__ == "__main__":
    final_counts = main()



















# ------------------------------
# Counts for appendix flowchart
# ------------------------------
# Note: run this script to get counts for flowchart without running the above again

import os
import dask
import dask.dataframe as dd
from dask.distributed import Client

DIALYSIS_BASE = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis"
TEMP_UNIQUE_BENE_BASE = os.path.join(DIALYSIS_BASE, "_tmp_hd_unique_benes_2011_2022")

cust_temp_dir = "/gpfs/data/cms-share/duas/52484/Jessy/temp_space/tmp/"
dask.config.set({"temporary-directory": cust_temp_dir})
dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s"
})

client = Client("10.50.87.6:45779")
print(client)

# Read root temp folder
tmp_benes = dd.read_parquet(TEMP_UNIQUE_BENE_BASE, engine="pyarrow")

print(tmp_benes.year.value_counts().compute())

# Keep only 2011-2021
tmp_benes = tmp_benes[~tmp_benes["year"].isin([2022,"2022"])]

n_unique_benes_2011_2021 = (
    tmp_benes["BENE_ID"]
    .dropna()
    .drop_duplicates()
    .count()
    .compute()
)

print(tmp_benes.year.value_counts().compute())

print("Unique beneficiaries with HD across 2011-2021:", n_unique_benes_2011_2021)














    
