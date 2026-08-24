#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 29, 2026
# Description: This script gathers the raw diagnosis codes for each beneficiary in our analytical sample. For each year, 
# it pulls the unique BENE_IDs from the analytical file, reads the corresponding MedPAR inpatient and outpatient 
# institutional claims for those beneficiaries, and cleans the diagnosis fields into a common layout (dx1–dx26).
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

# Analytical file for disrupted
def analytical_file(year: int) -> str:
    return (
        "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/"
        f"dialysis/01_analytical_sample/esrd_crossover_{year}/"
        "analytical_simple_case_crossover_anchor_exposure_refwk_m2_early_wkm1_class_wkm3_cumpost_cumdeath_v03.csv"
    )

OUT_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05a_comorbidity_raw_ip_op_from_exposure_anchor_v02"
)

os.makedirs(OUT_BASE, exist_ok=True)

def medpar_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/medpar/parquet/"

def opb_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/otpt/opb/parquet/"

def out_year_dir(year: int) -> str:
    return os.path.join(OUT_BASE, f"year={year}")

columns_ip = (
    ["BENE_ID", "ADMSN_DT", "SS_LS_SNF_IND_CD", "ADMTG_DGNS_CD", "PRVDR_NUM"] +
    [f"DGNS_{i}_CD" for i in range(1, 26)]
)

columns_op = (
    ["BENE_ID", "CLM_FROM_DT", "PRNCPAL_DGNS_CD", "PRVDR_NUM"] +
    [f"ICD_DGNS_CD{i}" for i in range(1, 26)]
)

# Raw dx output should only contain claim-level dx information.
FINAL_COLS = ["BENE_ID", "SRVC_BGN_DT", "claim_source"] + [f"dx{i}" for i in range(1, 27)]

# -------------------------
# Functions
# -------------------------
def _exists(p: str) -> bool:
    # Check whether a file or directory path exists.
    try:
        return os.path.exists(p)
    except Exception:
        return False


def _as_clean_str(s):
    # Clean ID-like columns read from csv.
    s = s.astype(str)
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return s


def _standardize_bene_id_ddf(df: dd.DataFrame) -> dd.DataFrame:
    # Standardize BENE_ID as string before filtering or merging.
    df["BENE_ID"] = df["BENE_ID"].astype(str)
    return df


def load_relevant_benes(year: int):
    """
    Read analytical BENE_IDs for the current year and next year.

    Why include next year?
    If we are processing raw claims year Y, we also need claims for benes
    whose storm/event occurs in Y+1 but whose 365-day lookback reaches back
    into year Y.

    Example:
      Exposure date = July 2020
      Lookback includes July 2019-July 2020
      Therefore, when processing 2019 raw dx, we need to include benes from
      the 2020 analytical file as well.
    """

    years_to_read = [year]

    if year + 1 <= YEAR_MAX:
        years_to_read.append(year + 1)

    bene_parts = []

    for yy in years_to_read:
        f = analytical_file(yy)

        if not _exists(f):
            print(f"[WARN] {year}: missing analytical file for bene filter year {yy} -> {f}")
            continue

        analytical = dd.read_csv(
            f,
            usecols=["BENE_ID"],
            dtype={"BENE_ID": "object"},
            assume_missing=True
        )

        analytical["BENE_ID"] = analytical["BENE_ID"].astype(str)
        bene_parts.append(analytical[["BENE_ID"]])

    if len(bene_parts) == 0:
        print(f"[SKIP] {year}: no current or next-year analytical BENE_ID files found")
        return None

    analytical_all = dd.concat(bene_parts, axis=0, interleave_partitions=True)

    relevant_benes = (
        analytical_all["BENE_ID"]
        .dropna()
        .drop_duplicates()
        .compute()
        .tolist()
    )

    if len(relevant_benes) == 0:
        print(f"[SKIP] {year}: no BENE_ID found in current/next analytical files")
        return None

    return relevant_benes


# ... IP ...
def get_ip_data(year: int, relevant_benes):
    # Load MedPAR claims for one claim year and keep only relevant beneficiaries.

    p = medpar_path(year)

    if not _exists(p):
        print(f"[WARN] {year}: MedPAR path not found -> {p}")
        return None

    medpar = dd.read_parquet(p, engine="pyarrow")

    # Older years may have BENE_ID stored in the index.
    if year <= 2016 and "BENE_ID" not in medpar.columns:
        medpar = medpar.reset_index()

    # Keep only columns needed for dx extraction.
    keep_cols = [c for c in columns_ip if c in medpar.columns]
    medpar = medpar[keep_cols]

    if "BENE_ID" not in medpar.columns:
        raise ValueError(f"{year}: BENE_ID not found in MedPAR after reset/subset.")

    medpar = _standardize_bene_id_ddf(medpar)

    # Drop SNF claims.
    # Keep short-stay / long-stay hospital claims.
    if "SS_LS_SNF_IND_CD" in medpar.columns:
        medpar = medpar[medpar["SS_LS_SNF_IND_CD"] != "N"]

    # Convert admission date to service begin date.
    medpar["ADMSN_DT"] = dd.to_datetime(medpar["ADMSN_DT"], errors="coerce")

    # Filter to current-year and next-year analytical benes.
    medpar = medpar[medpar["BENE_ID"].isin(relevant_benes)]

    # Rename dx columns into SAS-compatible dx1-dx26 layout.
    medpar = medpar.rename(columns={
        "ADMSN_DT": "SRVC_BGN_DT",
        "ADMTG_DGNS_CD": "dx1"
    })

    for n in range(1, 26):
        old = f"DGNS_{n}_CD"
        new = f"dx{n+1}"
        if old in medpar.columns:
            medpar = medpar.rename(columns={old: new})

    medpar["claim_source"] = "IP"

    # Add missing columns to make layout consistent across years.
    for c in FINAL_COLS:
        if c not in medpar.columns:
            medpar[c] = pd.NA

    # Use provider only for claim-level deduplication.
    medpar = medpar[
        ["BENE_ID", "PRVDR_NUM", "SRVC_BGN_DT", "claim_source"] +
        [f"dx{i}" for i in range(1, 27)]
    ]

    # Deduplicate within IP claims.
    medpar = medpar.shuffle(on=["BENE_ID", "PRVDR_NUM", "SRVC_BGN_DT"])
    medpar = medpar.map_partitions(
        lambda x: x.drop_duplicates(
            subset=["BENE_ID", "PRVDR_NUM", "SRVC_BGN_DT"],
            keep="first"
        )
    )

    # Drop provider after deduplication.
    medpar = medpar.drop(columns=["PRVDR_NUM"])

    medpar = medpar[FINAL_COLS]

    return medpar


# ... OP ...
def get_op_data(year: int, relevant_benes):
    # Load OPB claims for one claim year and keep only relevant beneficiaries.

    p = opb_path(year)

    if not _exists(p):
        print(f"[WARN] {year}: OPB path not found -> {p}")
        return None

    op_df = dd.read_parquet(
        p,
        engine="pyarrow",
        columns=columns_op
    )

    op_df = _standardize_bene_id_ddf(op_df)

    # Convert claim from date to service begin date.
    op_df["CLM_FROM_DT"] = dd.to_datetime(op_df["CLM_FROM_DT"], errors="coerce")

    # Filter to current-year and next-year analytical benes.
    op_df = op_df[op_df["BENE_ID"].isin(relevant_benes)]

    # Rename dx columns into SAS-compatible dx1-dx26 layout.
    op_df = op_df.rename(columns={
        "CLM_FROM_DT": "SRVC_BGN_DT",
        "PRNCPAL_DGNS_CD": "dx1"
    })

    for n in range(1, 26):
        old = f"ICD_DGNS_CD{n}"
        new = f"dx{n+1}"
        if old in op_df.columns:
            op_df = op_df.rename(columns={old: new})

    op_df["claim_source"] = "OP"

    # Add missing columns to make layout consistent across years.
    for c in FINAL_COLS:
        if c not in op_df.columns:
            op_df[c] = pd.NA

    # Use provider only for claim-level deduplication.
    op_df = op_df[
        ["BENE_ID", "PRVDR_NUM", "SRVC_BGN_DT", "claim_source"] +
        [f"dx{i}" for i in range(1, 27)]
    ]

    # Deduplicate within OP claims.
    op_df = op_df.shuffle(on=["BENE_ID", "PRVDR_NUM", "SRVC_BGN_DT"])
    op_df = op_df.map_partitions(
        lambda x: x.drop_duplicates(
            subset=["BENE_ID", "PRVDR_NUM", "SRVC_BGN_DT"],
            keep="first"
        )
    )

    # Drop provider after deduplication.
    op_df = op_df.drop(columns=["PRVDR_NUM"])

    op_df = op_df[FINAL_COLS]

    return op_df


def process_year(year: int):
    print(f"\n=== Processing raw dx claim year {year} ===")

    relevant_benes = load_relevant_benes(year)

    if relevant_benes is None:
        return

    if year + 1 <= YEAR_MAX:
        print(
            f"[INFO] {year}: relevant benes from analytical years "
            f"{year} and {year + 1} = {len(relevant_benes):,}"
        )
    else:
        print(
            f"[INFO] {year}: relevant benes from analytical year "
            f"{year} only = {len(relevant_benes):,}"
        )

    ip_data = get_ip_data(year, relevant_benes)
    op_data = get_op_data(year, relevant_benes)

    parts = []

    if ip_data is not None:
        parts.append(ip_data)

    if op_data is not None:
        parts.append(op_data)

    if len(parts) == 0:
        print(f"[SKIP] {year}: no IP/OP data found")
        return

    raw_inst = dd.concat(parts, axis=0, interleave_partitions=True)

    # Keep final column order only.
    for c in FINAL_COLS:
        if c not in raw_inst.columns:
            raw_inst[c] = pd.NA

    raw_inst = raw_inst[FINAL_COLS]

    # Drop records missing service begin date.
    raw_inst = raw_inst[raw_inst["SRVC_BGN_DT"].notnull()]

    # Repartition by BENE_ID for downstream event-level lookback merge.
    raw_inst = raw_inst.shuffle(on="BENE_ID")

    outdir = out_year_dir(year)
    os.makedirs(outdir, exist_ok=True)

    raw_inst.to_parquet(
        outdir,
        engine="pyarrow",
        compression="gzip",
        write_index=False,
        overwrite=True
    )

    print(f"[WROTE] {year}: {outdir}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    for year in range(YEAR_MIN, YEAR_MAX + 1):
        process_year(year)

    print(f"\n[DONE] Raw diagnosis claims written under: {OUT_BASE}")
