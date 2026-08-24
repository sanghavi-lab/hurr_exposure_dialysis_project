#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: July 22, 2026
# Description: This script reads the previously created placebo analytical file and appends an bene-storm level (i.e., event 
# level) chronic kidney disease (CKD) indicator using the MBSF CC CHRONICKIDNEY_EVER field. It flags ckd = 1 when
# the CKD-ever date is on or before the event’s placebo_exposure_start_dt, merges that information back onto the two-row 
# placebo panel, and runs QC checks to ensure the merge does not change the panel structure. Note that most bene's from
# the placebo analysis are the same as the analytical. However, because we pulled back the exposure date, we are capturing
# bene's who have died prior to exposure date but have NOT died prior to placebo date. Thus, this script is necessary to 
# gather CKD info for them (the ckd indicator for the analytical file was created in exh 2 scripts).
#----------------------------------------------------------------------------------------------------------------------#

import os
import re
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client

# =========================
# Dask client
# =========================
cust_temp_dir = "/gpfs/data/cms-share/duas/52484/Jessy/temp_space/tmp/"
dask.config.set({"temporary-directory": cust_temp_dir})
dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s"
})
client = Client("[redacted]")
print(client)

# =========================
# Config
# =========================
YEAR_MIN, YEAR_MAX = 2011, 2022

INPUT_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/"
    "dialysis/01_analytical_sample/"
)

INPUT_FILENAME = (
    "analytical_simple_case_crossover_anchor_exposure_"
    "placebo5wk_refwk_m7_hazwk_m5_early_wkm6_class_wkm8_"
    "cumpost_cumdeath_cumdisrupt_v02.csv"
)

OUTPUT_FILENAME = (
    "analytical_simple_case_crossover_anchor_exposure_"
    "placebo5wk_refwk_m7_hazwk_m5_early_wkm6_class_wkm8_"
    "cumpost_cumdeath_cumdisrupt_plus_ckd_v01.csv"
)

# =========================
# Helpers
# =========================
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


def _normalize_colname(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def _parse_date_series(raw: pd.Series) -> pd.Series:
    """
    Stable date parsing for CC ever dates.
    Tries YYYY-MM-DD, then YYYYMMDD, then generic fallback.
    """
    raw = _as_clean_str(raw).str.strip()

    dt = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")

    miss1 = dt.isna() & raw.notna()
    if miss1.any():
        dt.loc[miss1] = pd.to_datetime(raw.loc[miss1], format="%Y%m%d", errors="coerce")

    miss2 = dt.isna() & raw.notna()
    if miss2.any():
        dt.loc[miss2] = pd.to_datetime(raw.loc[miss2], errors="coerce")

    return dt.dt.normalize()


def input_path(year: int) -> str:
    return os.path.join(INPUT_BASE, f"esrd_crossover_{year}", INPUT_FILENAME)


def out_path(year: int) -> str:
    return os.path.join(INPUT_BASE, f"esrd_crossover_{year}", OUTPUT_FILENAME)


def mbsf_cc_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/mbsf/mbsf_cc/parquet/"

def mbsf_chronic_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/mbsf/mbsf_chronic/parquet/"


def ckd_source_path_and_label(year: int):
    """
    Use 27 CCW CC file through 2021.
    Use 30 CCW CHRONIC file from 2022 onward.
    CKD keeps the same CHRONICKIDNEY_EVER name in both files.
    """
    if year >= 2022:
        return mbsf_chronic_path(year), "CHRONIC_30CCW"
    return mbsf_cc_path(year), "CC_27CCW"

def _find_ckd_col(cols) -> str | None:
    target = "chronickidney_ever"
    norm_map = {_normalize_colname(c): c for c in cols}
    return norm_map.get(target)


def read_cc_ckd_subset(year: int, bene_ids: pd.Series) -> pd.DataFrame:
    """
    Read only BENE_ID + CHRONICKIDNEY_EVER from either:
    - MBSF CC 27 CCW file through 2021
    - MBSF CHRONIC 30 CCW file from 2022 onward

    Handles pre-2018 vs post-2017 storage of BENE_ID.
    """
    pq, source_label = ckd_source_path_and_label(year)

    if not _exists(pq):
        msg = f"{year}: missing {source_label} path: {pq}"
        if year >= 2022:
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg}")
        return pd.DataFrame(columns=["BENE_ID", "CHRONICKIDNEY_EVER"])

    meta = dd.read_parquet(pq, rows=0)
    cols_all = list(meta.columns)
    ckd_col = _find_ckd_col(cols_all)

    if ckd_col is None:
        msg = f"{year}: CHRONICKIDNEY_EVER not found in {source_label} parquet."
        if year >= 2022:
            raise ValueError(msg)
        print(f"[WARN] {msg}")
        return pd.DataFrame(columns=["BENE_ID", "CHRONICKIDNEY_EVER"])

    bene_ser = pd.Series(bene_ids)
    bene_ser = _as_clean_str(bene_ser).dropna().drop_duplicates().reset_index(drop=True)
    bene_df = pd.DataFrame({"BENE_ID": bene_ser})

    bene_dd = dd.from_pandas(bene_df, npartitions=1)
    bene_dd["BENE_ID"] = bene_dd["BENE_ID"].astype(str)

    if year > 2017:
        x = dd.read_parquet(pq, columns=["BENE_ID", ckd_col])
        x["BENE_ID"] = x["BENE_ID"].astype(str)
    else:
        x = dd.read_parquet(pq, columns=[ckd_col])
        if x._meta.index.name == "BENE_ID":
            x = x.reset_index()
        else:
            x = x.reset_index()
            if "BENE_ID" not in x.columns:
                raise ValueError(f"{year}: could not recover BENE_ID from older CC parquet.")
        x["BENE_ID"] = x["BENE_ID"].astype(str)

    x = x.merge(bene_dd, on="BENE_ID", how="inner")
    x = x.compute()

    x["BENE_ID"] = _as_clean_str(x["BENE_ID"])
    x[ckd_col] = _as_clean_str(x[ckd_col]).str.strip()

    if ckd_col != "CHRONICKIDNEY_EVER":
        x = x.rename(columns={ckd_col: "CHRONICKIDNEY_EVER"})

    print(f"[INFO] {year}: {source_label} matched bene-year rows = {len(x):,}")
    return x[["BENE_ID", "CHRONICKIDNEY_EVER"]]


def make_ckd_flag(cc_df: pd.DataFrame, analytical: pd.DataFrame) -> pd.DataFrame:
    """
    Create event-level CKD variables from CHRONICKIDNEY_EVER.
    ckd = 1 if CHRONICKIDNEY_EVER <= placebo_anchor_dt else 0.
    """
    ev = (
        analytical[["event_id", "BENE_ID", "placebo_anchor_dt"]]
        .drop_duplicates(subset=["event_id", "BENE_ID"])
        .copy()
        .reset_index(drop=True)
    )
    ev["event_id"] = _as_clean_str(ev["event_id"])
    ev["BENE_ID"] = _as_clean_str(ev["BENE_ID"])
    ev["placebo_anchor_dt"] = pd.to_datetime(
        ev["placebo_anchor_dt"], errors="coerce"
    ).dt.normalize()

    if cc_df.empty or "CHRONICKIDNEY_EVER" not in cc_df.columns:
        out = ev[["event_id", "BENE_ID"]].copy()
        out["CHRONICKIDNEY_EVER"] = pd.NA
        out["chronickidney_ever_dt"] = pd.NaT
        out["ckd"] = pd.Series(0, index=out.index, dtype="int8")
        return out

    tmp = ev.merge(cc_df, on="BENE_ID", how="left", indicator=True)
    print("[QC] event crosswalk <- CC merge:")
    print(tmp["_merge"].value_counts(dropna=False).to_string())
    tmp = tmp.drop(columns=["_merge"])

    out = tmp[["event_id", "BENE_ID"]].copy()
    out["CHRONICKIDNEY_EVER"] = tmp["CHRONICKIDNEY_EVER"]
    out["chronickidney_ever_dt"] = _parse_date_series(tmp["CHRONICKIDNEY_EVER"])
    out["ckd"] = (
        out["chronickidney_ever_dt"].notna()
        & (out["chronickidney_ever_dt"] <= tmp["placebo_anchor_dt"])
    ).astype("int8")

    dup_event_bene = out.duplicated(subset=["event_id", "BENE_ID"]).sum()
    if dup_event_bene > 0:
        raise ValueError(
            f"event-level CKD output has {dup_event_bene:,} duplicated event_id/BENE_ID rows."
        )

    return out


def qc_event_consistency(df: pd.DataFrame, group_cols: list[str], varlist: list[str]) -> int:
    bad_total = 0
    for c in varlist:
        chk = df.groupby(group_cols, dropna=False)[c].nunique(dropna=False)
        bad_total += (chk > 1).sum()
    return int(bad_total)


# =========================
# Main processing
# =========================
def process_year(year: int):
    print(f"\n=== STEP 4.5 PLACEBO CKD (DISRUPTED SAMPLE): processing year {year} ===")

    f_panel = input_path(year)
    if not _exists(f_panel):
        print(f"[SKIP] missing placebo analytical input: {f_panel}")
        return

    analytical = pd.read_csv(f_panel, low_memory=False)
    print(f"[INFO] placebo analytical rows = {len(analytical):,}")

    required = ["event_id", "BENE_ID", "placebo_anchor_dt"]
    for c in required:
        if c not in analytical.columns:
            raise ValueError(f"{year}: placebo analytical file must contain {c}.")

    analytical["event_id"] = _as_clean_str(analytical["event_id"])
    analytical["BENE_ID"] = _as_clean_str(analytical["BENE_ID"])
    analytical["placebo_anchor_dt"] = pd.to_datetime(
        analytical["placebo_anchor_dt"], errors="coerce"
    ).dt.normalize()

    n_rows_before = len(analytical)
    n_event_bene_before = analytical[["event_id", "BENE_ID"]].drop_duplicates().shape[0]

    rows_per_event_bene = analytical.groupby(["event_id", "BENE_ID"], dropna=False).size()
    bad_events = int((rows_per_event_bene != 2).sum())
    print(f"[QC] event_id/BENE_ID with !=2 panel rows = {bad_events:,}")

    # -------------------------
    # Read CKD subset from 27 CCW through 2021 and 30 CCW from 2022 onward
    # -------------------------
    cc = read_cc_ckd_subset(year, analytical["BENE_ID"])
    
    if year >= 2022 and (
        cc.empty
        or "CHRONICKIDNEY_EVER" not in cc.columns
        or cc["CHRONICKIDNEY_EVER"].isna().all()
    ):
        raise ValueError(
            f"{year}: CKD source was read but CHRONICKIDNEY_EVER is missing or entirely missing. "
            "Check the MBSF CHRONIC path and column names before continuing."
        )

    if "BENE_ID" in cc.columns:
        dup_bene_cc = int(cc.duplicated(subset=["BENE_ID"]).sum())
        if dup_bene_cc > 0:
            print(f"[WARN] {year}: CC subset has {dup_bene_cc:,} duplicated BENE_ID rows; keeping first.")
            cc = cc.drop_duplicates(subset=["BENE_ID"], keep="first").reset_index(drop=True)

    # -------------------------
    # Build event-level CKD flag
    # -------------------------
    ckd_event = make_ckd_flag(cc, analytical)
    print(f"[INFO] event-level CKD rows = {len(ckd_event):,}")

    # -------------------------
    # Merge back to panel
    # -------------------------
    analytical2 = analytical.merge(
        ckd_event,
        on=["event_id", "BENE_ID"],
        how="left",
        indicator=True,
    )

    print("[QC] analytical <- ckd_event merge:")
    print(analytical2["_merge"].value_counts(dropna=False).to_string())

    if len(analytical2) != n_rows_before:
        raise ValueError(
            f"{year}: row count changed after CKD merge. Before={n_rows_before:,}, After={len(analytical2):,}"
        )

    n_event_bene_after = analytical2[["event_id", "BENE_ID"]].drop_duplicates().shape[0]
    if n_event_bene_after != n_event_bene_before:
        raise ValueError(
            f"{year}: unique event_id/BENE_ID count changed after CKD merge. "
            f"Before={n_event_bene_before:,}, After={n_event_bene_after:,}"
        )

    new_vars = ["CHRONICKIDNEY_EVER", "chronickidney_ever_dt", "ckd"]

    print("[QC] row-level missingness for CKD-derived variables:")
    miss_rows = pd.DataFrame({
        "variable": new_vars,
        "n_missing_rows": [analytical2[c].isna().sum() for c in new_vars],
        "pct_missing_rows": [analytical2[c].isna().mean() for c in new_vars],
    })
    print(miss_rows.to_string(index=False))

    print(f"[QC] {year} CKD flag frequency:")
    print(analytical2["ckd"].value_counts(dropna=False).sort_index().to_string())

    bad_repeat = qc_event_consistency(analytical2, ["event_id", "BENE_ID"], new_vars)
    if bad_repeat > 0:
        raise ValueError(
            f"{year}: found {bad_repeat:,} event-by-variable inconsistencies in appended CKD values."
        )

    analytical2 = analytical2.drop(columns=["_merge"])

    # -------------------------
    # Reorder
    # -------------------------
    preferred_front = [
        "year", "event_id", "BENE_ID", "storm_id", "facility_id",
        "week_rel", "hazard_week",
        "ckd", "CHRONICKIDNEY_EVER", "chronickidney_ever_dt",
        "placebo_anchor_dt", "anchor_dt", "county_exposure_start_dt",
    ]
    front = [c for c in preferred_front if c in analytical2.columns]
    rest = [c for c in analytical2.columns if c not in front]
    analytical2 = analytical2[front + rest]

    # -------------------------
    # Write
    # -------------------------
    fout = out_path(year)
    os.makedirs(os.path.dirname(fout), exist_ok=True)
    analytical2.to_csv(fout, index=False)

    print(f"[WROTE] {year}: {fout}")
    print(f"[QC] final panel rows = {len(analytical2):,}")
    print(f"[QC] final unique event_id/BENE_ID = {analytical2[['event_id','BENE_ID']].drop_duplicates().shape[0]:,}")


# =========================
# Run
# =========================
if __name__ == "__main__":
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        process_year(year)

    print("\n[DONE] CKD-appended disrupted-sample placebo analytical files written next to the original yearly files.")


    









# -------------------------
# Import modules
# -------------------------

import os
import re
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
client = Client("10.50.87.29:45637")
print(client)

# -------------------------
# Paths and spec
# -------------------------
YEAR_MIN, YEAR_MAX = 2011, 2022

STEP4_PLACEBO_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "04_placebo_analytical_panel_hurr_exposure_v02_wkm7_vs_m5_facclust_cumpost_cumdeath_cumdisrupt"
)

OUT_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "04_placebo_analytical_panel_hurr_exposure_v02_wkm7_vs_m5_facclust_"
    "cumpost_cumdeath_cumdisrupt_plus_ckd_v01"
)

os.makedirs(OUT_BASE, exist_ok=True)

# -------------------------
# Functions
# -------------------------
def _exists(p: str) -> bool: # just used to check if paths exists
    try:
        return os.path.exists(p)
    except Exception:
        return False


def _as_clean_str(s: pd.Series) -> pd.Series: # standardizes a pandas Series into a clean string version
    out = s.astype(str)
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return out


def _normalize_colname(c: str) -> str: # clean column name (e.g., "Chronic Kidney Ever" -> "chronic_kidney_ever"
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def _parse_date_series(raw: pd.Series) -> pd.Series:  # parses date fields into normalized pandas datetimes
    raw = _as_clean_str(raw).str.strip()

    dt = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")

    miss1 = dt.isna() & raw.notna()
    if miss1.any():
        dt.loc[miss1] = pd.to_datetime(raw.loc[miss1], format="%Y%m%d", errors="coerce")

    miss2 = dt.isna() & raw.notna()
    if miss2.any():
        dt.loc[miss2] = pd.to_datetime(raw.loc[miss2], errors="coerce")

    return dt.dt.normalize()


def step4_file(year: int) -> str:
    return os.path.join(STEP4_PLACEBO_BASE, f"year_{year}", "analytical_panel.csv")


def out_year_dir(year: int) -> str:
    return os.path.join(OUT_BASE, f"year_{year}")


def out_file(year: int) -> str:
    return os.path.join(out_year_dir(year), "analytical_panel.csv")


def mbsf_cc_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/mbsf/mbsf_cc/parquet/"

def mbsf_chronic_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/mbsf/mbsf_chronic/parquet/"

def ckd_source_path_and_label(year: int):
    """
    Use 27 CCW CC file through 2021.
    Use 30 CCW CHRONIC file from 2022 onward.
    CKD keeps the same CHRONICKIDNEY_EVER name in both files.
    """
    if year >= 2022:
        return mbsf_chronic_path(year), "CHRONIC_30CCW"
    return mbsf_cc_path(year), "CC_27CCW"

def _find_ckd_col(cols) -> str | None: # locate ckd col
    target = "chronickidney_ever"
    norm_map = {_normalize_colname(c): c for c in cols}
    return norm_map.get(target)


def read_cc_ckd_subset(year: int, bene_ids: pd.Series) -> pd.DataFrame:
    # Read only BENE_ID + CHRONICKIDNEY_EVER from either:
    # - MBSF CC 27 CCW file through 2021
    # - MBSF CHRONIC 30 CCW file from 2022 onward

    pq, source_label = ckd_source_path_and_label(year)

    if not _exists(pq):
        msg = f"{year}: missing {source_label} path: {pq}"
        if year >= 2022:
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg}")
        return pd.DataFrame(columns=["BENE_ID", "CHRONICKIDNEY_EVER"])

    meta = dd.read_parquet(pq, rows=0) # read only the parquet metadata, not the full file. This is a lightweight way to inspect the available cols.
    cols_all = list(meta.columns) # extract the col names into a list.
    ckd_col = _find_ckd_col(cols_all) # find ckd col

    if ckd_col is None:
        msg = f"{year}: CHRONICKIDNEY_EVER not found in {source_label} parquet."
        if year >= 2022:
            raise ValueError(msg)
        print(f"[WARN] {msg}")
        return pd.DataFrame(columns=["BENE_ID", "CHRONICKIDNEY_EVER"])

    # Clean list of unique beneficiaries
    bene_ser = pd.Series(bene_ids)
    bene_ser = _as_clean_str(bene_ser).dropna().drop_duplicates().reset_index(drop=True)
    bene_df = pd.DataFrame({"BENE_ID": bene_ser})
    bene_dd = dd.from_pandas(bene_df, npartitions=1)
    bene_dd["BENE_ID"] = bene_dd["BENE_ID"].astype(str)

    if year > 2017: # conditional import due to discrepancy in bene_id index
        x = dd.read_parquet(pq, columns=["BENE_ID", ckd_col])
        x["BENE_ID"] = x["BENE_ID"].astype(str)
    else:
        x = dd.read_parquet(pq, columns=[ckd_col])
        if x._meta.index.name == "BENE_ID":
            x = x.reset_index()
        else:
            x = x.reset_index()
            if "BENE_ID" not in x.columns:
                raise ValueError(f"{year}: could not recover BENE_ID from older CC parquet.")
        x["BENE_ID"] = x["BENE_ID"].astype(str)

    x = x.merge(bene_dd, on="BENE_ID", how="inner") # keep only the rows in the CC file that match the bene from analytical file.
    x = x.compute()

    x["BENE_ID"] = _as_clean_str(x["BENE_ID"])
    x[ckd_col] = _as_clean_str(x[ckd_col]).str.strip()

    if ckd_col != "CHRONICKIDNEY_EVER":
        x = x.rename(columns={ckd_col: "CHRONICKIDNEY_EVER"})

    print(f"[INFO] {year}: {source_label} matched bene-year rows = {len(x):,}")
    return x[["BENE_ID", "CHRONICKIDNEY_EVER"]]


def make_ckd_flag(cc_df: pd.DataFrame, analytical: pd.DataFrame) -> pd.DataFrame:
    # ckd = 1 if CHRONICKIDNEY_EVER <= placebo_exposure_start_dt else 0.
    
    ev = (
        analytical[["event_id", "BENE_ID", "placebo_exposure_start_dt"]]
        .drop_duplicates(subset=["event_id"])
        .copy()
        .reset_index(drop=True)
    )
    ev["event_id"] = pd.to_numeric(ev["event_id"], errors="coerce").astype("Int64")
    ev["BENE_ID"] = _as_clean_str(ev["BENE_ID"])
    ev["placebo_exposure_start_dt"] = pd.to_datetime(
        ev["placebo_exposure_start_dt"], errors="coerce"
    ).dt.normalize()

    if cc_df.empty or "CHRONICKIDNEY_EVER" not in cc_df.columns:
        out = ev[["event_id", "BENE_ID"]].copy()
        out["CHRONICKIDNEY_EVER"] = pd.NA
        out["chronickidney_ever_dt"] = pd.NaT
        out["ckd"] = pd.Series(0, index=out.index, dtype="int8")
        return out

    tmp = ev.merge(cc_df, on="BENE_ID", how="left", indicator=True) # merge the bene-event (i.e., bene-storm) table with the CC data by bene_id
    print("[QC] event crosswalk <- CC merge:")
    print(tmp["_merge"].value_counts(dropna=False).to_string())
    tmp = tmp.drop(columns=["_merge"])

    out = tmp[["event_id", "BENE_ID"]].copy()
    out["CHRONICKIDNEY_EVER"] = tmp["CHRONICKIDNEY_EVER"]
    out["chronickidney_ever_dt"] = _parse_date_series(tmp["CHRONICKIDNEY_EVER"])
    out["ckd"] = (
        out["chronickidney_ever_dt"].notna()
        & (out["chronickidney_ever_dt"] <= tmp["placebo_exposure_start_dt"])
    ).astype("int8")
    # ^ create the bene-event-level CKD flag. This becomes 1 only if: the CKD-ever date is not missing and the CKD-ever date is on or before the placebo exposure start date, Otherwise it becomes 0.

    dup_event = out.duplicated(subset=["event_id"]).sum()
    if dup_event > 0:
        raise ValueError(f"event-level CKD output has {dup_event:,} duplicated event_id values.")

    return out


def qc_event_consistency(df: pd.DataFrame, event_id_col: str, varlist: list[str]) -> int:
    # This is a quality check function that checks whether variables like ckd are identical across the two panel rows (place ref week vs exposure week) belonging to the same bene-event (i.e., bene-storm). If not, it counts those problems.
    bad_total = 0 # start counter at 0
    for c in varlist: # Loops through each variable in the variable list.
        chk = df.groupby(event_id_col, dropna=False)[c].nunique(dropna=False) # For that variable, groups rows by event ID, counts how many unique values the variable has within each event. If a variable is truly event-level, each event should have exactly one unique value.
        bad_total += (chk > 1).sum() # Counts how many bene-events have more than one unique value for that variable and adds that number to the running total.
    return bad_total


def process_year(year: int):
    print(f"\n=== STEP 4.5 PLACEBO CKD: processing year {year} ===")

    f_panel = step4_file(year)
    if not _exists(f_panel):
        print(f"[SKIP] missing placebo analytical input: {f_panel}")
        return

    analytical = pd.read_csv(f_panel, low_memory=False)
    print(f"[INFO] placebo analytical rows = {len(analytical):,}")

    required = ["event_id", "BENE_ID", "placebo_exposure_start_dt"] # required cols
    for c in required:
        if c not in analytical.columns:
            raise ValueError(f"{year}: placebo analytical file must contain {c}.")

    analytical["event_id"] = pd.to_numeric(analytical["event_id"], errors="coerce").astype("Int64")
    analytical["BENE_ID"] = _as_clean_str(analytical["BENE_ID"])
    analytical["placebo_exposure_start_dt"] = pd.to_datetime(
        analytical["placebo_exposure_start_dt"], errors="coerce"
    ).dt.normalize()

    n_rows_before = len(analytical)
    n_events_before = analytical["event_id"].nunique(dropna=True)

    rows_per_event = analytical.groupby("event_id", dropna=False).size()
    bad_events = (rows_per_event != 2).sum()
    print(f"[QC] event_id with !=2 panel rows = {bad_events:,}")

    # ... Read CKD subset from 27 CCW through 2021 and 30 CCW from 2022 onward ...
    cc = read_cc_ckd_subset(year, analytical["BENE_ID"])
    
    if year >= 2022 and (
        cc.empty
        or "CHRONICKIDNEY_EVER" not in cc.columns
        or cc["CHRONICKIDNEY_EVER"].isna().all()
    ):
        raise ValueError(
            f"{year}: CKD source was read but CHRONICKIDNEY_EVER is missing or entirely missing. "
            "Check the MBSF CHRONIC path and column names before continuing."
        )

    if "BENE_ID" in cc.columns:
        dup_bene_cc = cc.duplicated(subset=["BENE_ID"]).sum()
        if dup_bene_cc > 0:
            print(f"[WARN] {year}: CC subset has {dup_bene_cc:,} duplicated BENE_ID rows; keeping first.") # print a warning so I know the CC subset was not one-row-per-beneficiary as expected.
            cc = cc.drop_duplicates(subset=["BENE_ID"], keep="first").reset_index(drop=True) # Collapse the CC dataframe to one row per bene by: dropping repeated bene_id rows, keeping the first one encountered, resetting the row index

    # ... Build bene-storm (event level) CKD flag ...
    ckd_event = make_ckd_flag(cc, analytical)
    print(f"[INFO] event-level CKD rows = {len(ckd_event):,}")

    # ... Merge back CKD ind ...
    analytical2 = analytical.merge(
        ckd_event,
        on=["event_id", "BENE_ID"],
        how="left",
        indicator=True
    )

    print("[QC] analytical <- ckd_event merge:")
    print(analytical2["_merge"].value_counts(dropna=False).to_string())

    if len(analytical2) != n_rows_before:
        raise ValueError(
            f"{year}: row count changed after CKD merge. Before={n_rows_before:,}, After={len(analytical2):,}"
        )

    n_events_after = analytical2["event_id"].nunique(dropna=True)
    if n_events_after != n_events_before:
        raise ValueError(
            f"{year}: unique event count changed after CKD merge. Before={n_events_before:,}, After={n_events_after:,}"
        )

    new_vars = ["CHRONICKIDNEY_EVER", "chronickidney_ever_dt", "ckd"]

    print("[QC] row-level missingness for CKD-derived variables:")
    miss_rows = pd.DataFrame({
        "variable": new_vars,
        "n_missing_rows": [analytical2[c].isna().sum() for c in new_vars],
        "pct_missing_rows": [analytical2[c].isna().mean() for c in new_vars],
    })
    print(miss_rows.to_string(index=False))

    print(f"[QC] {year} CKD flag frequency:")
    print(analytical2["ckd"].value_counts(dropna=False).sort_index().to_string())

    bad_repeat = qc_event_consistency(analytical2, "event_id", new_vars) # again, more qc's
    if bad_repeat > 0:
        raise ValueError(
            f"{year}: found {bad_repeat:,} event-by-variable inconsistencies in appended CKD values."
        )

    analytical2 = analytical2.drop(columns=["_merge"])

    # ... Reorder ...
    preferred_front = [
        "year", "storm_year", "storm_id", "event_id", "BENE_ID", "fips",
        "week_rel", "hazard_week",
        "ckd", "CHRONICKIDNEY_EVER", "chronickidney_ever_dt",
        "placebo_exposure_start_dt", "exposure_start_dt",
    ]
    front = [c for c in preferred_front if c in analytical2.columns]
    rest = [c for c in analytical2.columns if c not in front]
    analytical2 = analytical2[front + rest]

    # ... Export ...
    odir = out_year_dir(year)
    os.makedirs(odir, exist_ok=True)

    fout = out_file(year)
    analytical2.to_csv(fout, index=False)

    print(f"[WROTE] {year}: {fout}")
    print(f"[QC] final panel rows = {len(analytical2):,}")
    print(f"[QC] final unique events = {analytical2['event_id'].nunique(dropna=True):,}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        process_year(year)

    print(f"\n[DONE] CKD-appended placebo analytical files written under: {OUT_BASE}")
