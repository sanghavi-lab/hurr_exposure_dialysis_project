#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 28, 2026
# Description: This script builds a year-by-year placebo pre-period analytical panel for dialysis beneficiaries by taking
# the analytical cohort, expanding each bene-storm (event) to placebo weeks -6 through -2, and then attaching week-level 
# indicators for dialysis disruption, ED use, and inpatient admission.
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
client = Client("10.50.87.228:38999")
print(client)

# -------------------------
# Paths and spec
# -------------------------
YEAR_MIN, YEAR_MAX = 2011, 2022

# Weeks to build (placebo pre-period only)
PLACEBO_WEEKS = [-6, -5, -4, -3, -2]

# Analytical file (used ONLY to define bene-event cohort + clustering variable)
STEP4_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "04_analytical_panel_hurr_exposure_v05_wkm2_facclust_cumpost_cumdeath"
)

# Dialysis line items (within +/- 2 months of exposure_start_dt)
STEP3_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "bene_storm_hd_services_pm1mo_byyear_v01"
)

def step4_csv(year: int) -> str:
    return os.path.join(STEP4_BASE, f"year_{year}", "analytical_panel.csv")

def step3_detail_dir(year: int) -> str:
    return os.path.join(STEP3_BASE, f"year={year}", "detail_hd_lines")

def medpar_path(year: int) -> str:
    return f"/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/00b_hospital_SL/{year}/"

def ed_path(year: int) -> str:
    return f"/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/00c/{year}/"

# Output
OUT_BASE = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis"
OUT_DIR  = os.path.join(OUT_BASE, "04_placebo_panel_preweeks_v01_wkm2_ref")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Functions
# -------------------------
def _exists(p: str) -> bool: # just used to check if paths exists
    try:
        return os.path.exists(p)
    except Exception:
        return False

def _clean_str_series(s: pd.Series) -> pd.Series: # standardizes a pandas Series into a clean string version
    s = s.astype(str).str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return s

def _week_bounds(k: int) -> tuple[int, int]:
    # converts value into the start and end relative-day numbers for that week. e.g. _week_bounds(-2) -> (-14, -8)
    start = int(k * 7)
    end   = int(start + 6)
    return start, end

def _prep_events_from_step4(df4: pd.DataFrame, year: int) -> pd.DataFrame:
    # Extract unique bene-event cohort from analytical and keep only identifiers and clustering variable
    
    if df4.empty:
        return pd.DataFrame()

    d = df4.copy()
    # Normalize key columns
    for c in ["BENE_ID", "storm_id", "fips", "event_id", "PRVDR_NUM_event"]:
        if c in d.columns:
            d[c] = _clean_str_series(d[c])

    d["exposure_start_dt"] = pd.to_datetime(d["exposure_start_dt"], errors="coerce").dt.normalize()
    d = d[d["exposure_start_dt"].notna()].copy()

    # Make sure the df has all the cols before it subsets
    keep_cols = [
        "year", "storm_year", "storm_id", "event_id", "BENE_ID", "fips",
        "exposure_start_dt", "PRVDR_NUM_event",
        "schedule_type", "stable_3x_weekly"
    ]
    for c in keep_cols:
        if c not in d.columns:
            d[c] = pd.NA

    d["year"] = year

    # Analytical file has two rows per event (wk -2 and wk 0); collapse to unique bene-events
    ev = (
        d[keep_cols]
        .drop_duplicates(subset=["event_id", "BENE_ID"])
        .reset_index(drop=True)
    )

    # Enforce types
    ev["storm_year"] = pd.to_numeric(ev["storm_year"], errors="coerce").astype("Int16")
    ev["stable_3x_weekly"] = pd.to_numeric(ev["stable_3x_weekly"], errors="coerce").fillna(0).astype("int8")

    # Normalize fips/provider formats
    ev["fips"] = _clean_str_series(ev["fips"]).str.zfill(5)
    ev["PRVDR_NUM_event"] = _clean_str_series(ev["PRVDR_NUM_event"]).str.zfill(6)

    return ev

# ... Disruption flags ...
def disruption_from_step3_detail(events: pd.DataFrame, detail_dir: str) -> pd.DataFrame:
    # determine whether the bene had a disruption in each placebo week -6 through -2
    # disrupt = 1 if n_dialysis < 3 else 0

    if events.empty or (not _exists(detail_dir)): # exit if directory does not exist
        return pd.DataFrame(columns=["event_id", "BENE_ID", "week_rel", "n_dialysis", "disrupt"])

    # Read only what we need
    d = dd.read_parquet(detail_dir, engine="pyarrow",
                        columns=["BENE_ID", "storm_id", "fips", "exposure_start_dt", "REV_CNTR_DT"])

    # Clean / types
    d["BENE_ID"] = d["BENE_ID"].astype(str)
    d["storm_id"] = d["storm_id"].astype(str)
    d["fips"] = d["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)

    d["exposure_start_dt"] = dd.to_datetime(d["exposure_start_dt"], errors="coerce").dt.floor("D")
    d["REV_CNTR_DT"] = dd.to_datetime(d["REV_CNTR_DT"], errors="coerce").dt.floor("D")
    d = d[(d["exposure_start_dt"].notnull()) & (d["REV_CNTR_DT"].notnull())]

    # Clean and join
    ev_key = events[["event_id", "BENE_ID", "storm_id", "fips", "exposure_start_dt"]].copy()
    ev_key["BENE_ID"] = _clean_str_series(ev_key["BENE_ID"])
    ev_key["storm_id"] = _clean_str_series(ev_key["storm_id"])
    ev_key["fips"] = _clean_str_series(ev_key["fips"]).str.zfill(5)
    ev_key["exposure_start_dt"] = pd.to_datetime(ev_key["exposure_start_dt"], errors="coerce").dt.normalize()

    ev_dd = dd.from_pandas(ev_key, npartitions=1)
    ev_dd["BENE_ID"] = ev_dd["BENE_ID"].astype(str)
    ev_dd["storm_id"] = ev_dd["storm_id"].astype(str)
    ev_dd["fips"] = ev_dd["fips"].astype(str)

    d = d.merge(ev_dd, on=["BENE_ID", "storm_id", "fips", "exposure_start_dt"], how="inner") # merge happens here

    # Rel day
    d = d.assign(rel_day=(d["REV_CNTR_DT"] - d["exposure_start_dt"]).dt.days)

    # Dedup to dialysis DAYS (keep only one row per dialysis day for each event_id × BENE_ID (i.e. bene-storm))
    d_day = (
        d[["event_id", "BENE_ID", "REV_CNTR_DT", "rel_day"]]
        .drop_duplicates(subset=["event_id", "BENE_ID", "REV_CNTR_DT"])
    )

    out_parts = []
    for wk in PLACEBO_WEEKS: # loop through every placebo week in PLACEBO_WEEKS
        start, end = _week_bounds(wk) # get the rel-day range for that week.
        w = d_day[(d_day["rel_day"] >= start) & (d_day["rel_day"] <= end)] # filter the dialysis-day data to that 7-day window.
        g = ( # Count unique dialysis days in that window
            w.groupby(["event_id", "BENE_ID"])
             .size()
             .rename("n_dialysis") # rename that count to n_dialysis.
             .reset_index()
             .compute()
        )
        g["week_rel"] = wk # add the week label back as week_rel.
        out_parts.append(g) # append the weekly result to out_parts.

    if not out_parts:
        return pd.DataFrame(columns=["event_id", "BENE_ID", "week_rel", "n_dialysis", "disrupt"]) # if nothing was generated, return an empty table with the correct columns. I manually checked and data was generated.

    out = pd.concat(out_parts, ignore_index=True) # stack all the weekly pieces together.
    out["n_dialysis"] = pd.to_numeric(out["n_dialysis"], errors="coerce").fillna(0).astype("int16")
    out["disrupt"] = (out["n_dialysis"] < 3).astype("int8") # create disrupt = 1 when n_dialysis < 3, else 0 for each week
    out["BENE_ID"] = _clean_str_series(out["BENE_ID"])
    out["event_id"] = _clean_str_series(out["event_id"])
    out["week_rel"] = out["week_rel"].astype("int8")
    return out[["event_id", "BENE_ID", "week_rel", "n_dialysis", "disrupt"]]


# ... ED flags ...
def ed_flags_by_week(events: pd.DataFrame, ed_dir: str) -> pd.DataFrame:
    if events.empty or (not _exists(ed_dir)):
        return pd.DataFrame(columns=["event_id", "BENE_ID", "week_rel", "any_ed"])

    # Time boundaries across all placebo week (basically subset to weeks -6 to -2 for efficiently processing of ED claims)
    min_start = min(_week_bounds(w)[0] for w in PLACEBO_WEEKS)
    max_end   = max(_week_bounds(w)[1] for w in PLACEBO_WEEKS)
    tmin = pd.to_datetime(events["exposure_start_dt"].min()) + pd.Timedelta(days=min_start)
    tmax = pd.to_datetime(events["exposure_start_dt"].max()) + pd.Timedelta(days=max_end)

    # Clean and convert to dask df
    benes = pd.DataFrame({"BENE_ID": _clean_str_series(events["BENE_ID"]).unique()})
    benes_dd = dd.from_pandas(benes, npartitions=1)
    benes_dd["BENE_ID"] = benes_dd["BENE_ID"].astype(str)

    ed = dd.read_parquet(ed_dir, columns=["BENE_ID", "REV_CNTR_DT"], engine="pyarrow")
    ed["BENE_ID"] = ed["BENE_ID"].astype(str)
    ed = ed.assign(date=dd.to_datetime(ed["REV_CNTR_DT"], errors="coerce").dt.floor("D"))
    ed = ed[ed["date"].notnull()]
    ed = ed[(ed["date"] >= tmin) & (ed["date"] <= tmax)] # restrict to the date range covering all placebo weeks.
    ed = ed.merge(benes_dd, on="BENE_ID", how="inner") # keep relevant bene

    cohort = events[["event_id", "BENE_ID", "exposure_start_dt"]].copy()
    cohort["BENE_ID"] = _clean_str_series(cohort["BENE_ID"])
    cohort["exposure_start_dt"] = pd.to_datetime(cohort["exposure_start_dt"], errors="coerce").dt.normalize()

    cohort_dd = dd.from_pandas(cohort, npartitions=1)
    cohort_dd["BENE_ID"] = cohort_dd["BENE_ID"].astype(str)
    cohort_dd["event_id"] = cohort_dd["event_id"].astype(str)
    cohort_dd["exposure_start_dt"] = dd.to_datetime(cohort_dd["exposure_start_dt"], errors="coerce").dt.floor("D")

    ed = ed.merge(cohort_dd, on="BENE_ID", how="inner") # merge ED records to the cohort by bene_id.
    ed = ed.assign(rel_day=(ed["date"] - ed["exposure_start_dt"]).dt.days) # calc rel_day for each ED date relative to each event's (bene-storm's) exposure_start_dt.

    out_parts = []
    for wk in PLACEBO_WEEKS: # loop over each placebo week.
        start, end = _week_bounds(wk)
        w = ed[(ed["rel_day"] >= start) & (ed["rel_day"] <= end)] # filter ED rows to that week’s rel-day range.
        g = (
            w.groupby(["event_id", "BENE_ID"])
             .size() # use .size() only to detect presence of at least one ED record.
             .rename("any_ed")
             .reset_index()
             .compute()
        )
        g["any_ed"] = 1 # replace the count with 1, so this becomes a binary indicator any_ed. Only those with ED claims will get this indicator
        g["week_rel"] = wk
        out_parts.append(g[["event_id", "BENE_ID", "week_rel", "any_ed"]])

    if not out_parts:
        return pd.DataFrame(columns=["event_id", "BENE_ID", "week_rel", "any_ed"])

    out = pd.concat(out_parts, ignore_index=True) # stacks the weekly pieces together.
    out["any_ed"] = pd.to_numeric(out["any_ed"], errors="coerce").fillna(0).astype("int8")
    out["week_rel"] = out["week_rel"].astype("int8")
    out["BENE_ID"] = _clean_str_series(out["BENE_ID"])
    out["event_id"] = _clean_str_series(out["event_id"])
    return out


# ... IP flags ...
def ip_flags_by_week(events: pd.DataFrame, medpar_dir: str) -> pd.DataFrame:
    if events.empty or (not _exists(medpar_dir)):
        return pd.DataFrame(columns=["event_id", "BENE_ID", "week_rel", "any_ip"])

    # Time boundaries across all placebo week (basically subset to weeks -6 to -2 for efficiently processing of ED claims)
    min_start = min(_week_bounds(w)[0] for w in PLACEBO_WEEKS)
    max_end   = max(_week_bounds(w)[1] for w in PLACEBO_WEEKS)
    tmin = pd.to_datetime(events["exposure_start_dt"].min()) + pd.Timedelta(days=min_start)
    tmax = pd.to_datetime(events["exposure_start_dt"].max()) + pd.Timedelta(days=max_end)

    # Clean and convert to dask df
    benes = pd.DataFrame({"BENE_ID": _clean_str_series(events["BENE_ID"]).unique()})
    benes_dd = dd.from_pandas(benes, npartitions=1)
    benes_dd["BENE_ID"] = benes_dd["BENE_ID"].astype(str)

    ip = dd.read_parquet(medpar_dir, columns=["BENE_ID", "ADMSN_DT"], engine="pyarrow")
    ip["BENE_ID"] = ip["BENE_ID"].astype(str)
    ip = ip.assign(adm=dd.to_datetime(ip["ADMSN_DT"], errors="coerce").dt.floor("D"))
    ip = ip[ip["adm"].notnull()]
    ip = ip[(ip["adm"] >= tmin) & (ip["adm"] <= tmax)] # restrict to the date range covering all placebo weeks.
    ip = ip.merge(benes_dd, on="BENE_ID", how="inner") # keep relevant bene

    cohort = events[["event_id", "BENE_ID", "exposure_start_dt"]].copy()
    cohort["BENE_ID"] = _clean_str_series(cohort["BENE_ID"])
    cohort["exposure_start_dt"] = pd.to_datetime(cohort["exposure_start_dt"], errors="coerce").dt.normalize()

    cohort_dd = dd.from_pandas(cohort, npartitions=1)
    cohort_dd["BENE_ID"] = cohort_dd["BENE_ID"].astype(str)
    cohort_dd["event_id"] = cohort_dd["event_id"].astype(str)
    cohort_dd["exposure_start_dt"] = dd.to_datetime(cohort_dd["exposure_start_dt"], errors="coerce").dt.floor("D")

    ip = ip.merge(cohort_dd, on="BENE_ID", how="inner") # merge IP records to the cohort by bene_id.
    ip = ip.assign(rel_day=(ip["adm"] - ip["exposure_start_dt"]).dt.days) # calc rel_day for each IP date relative to each event's (bene-storm's) exposure_start_dt.

    out_parts = []
    for wk in PLACEBO_WEEKS: # loop over each placebo week.
        start, end = _week_bounds(wk)
        w = ip[(ip["rel_day"] >= start) & (ip["rel_day"] <= end)] # filter IP rows to that week’s rel-day range.
        g = (
            w.groupby(["event_id", "BENE_ID"])
             .size() # use .size() only to detect presence of at least one IP record.
             .rename("any_ip")
             .reset_index()
             .compute()
        )
        g["any_ip"] = 1 # replace the count with 1, so this becomes a binary indicator any_ip. Only those with IP records will get this indicator
        g["week_rel"] = wk
        out_parts.append(g[["event_id", "BENE_ID", "week_rel", "any_ip"]])

    if not out_parts:
        return pd.DataFrame(columns=["event_id", "BENE_ID", "week_rel", "any_ip"])

    out = pd.concat(out_parts, ignore_index=True) # stacks the weekly pieces together.
    out["any_ip"] = pd.to_numeric(out["any_ip"], errors="coerce").fillna(0).astype("int8")
    out["week_rel"] = out["week_rel"].astype("int8")
    out["BENE_ID"] = _clean_str_series(out["BENE_ID"])
    out["event_id"] = _clean_str_series(out["event_id"])
    return out

# ... Long panel ...
# creates the long panel.
def build_scaffold(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    base = events.copy()
    base["event_id"] = _clean_str_series(base["event_id"])
    base["BENE_ID"] = _clean_str_series(base["BENE_ID"])

    # Cross-join with placebo weeks
    w = pd.DataFrame({"week_rel": PLACEBO_WEEKS}) # create a DF with one row per placebo week.
    w["week_rel"] = w["week_rel"].astype("int8")
    base["__tmp"] = 1 # add a fake column __tmp = 1 to both tables.
    w["__tmp"] = 1 # add a fake column __tmp = 1 to both tables.
    out = base.merge(w, on="__tmp", how="inner").drop(columns="__tmp") # merge on that fake constant, which is a standard trick for a cross join. Thus, every event (bene-storm) row is repeated once for each placebo week. Then drop the fake helper column.

    return out

# -------------------------
# Main
# -------------------------
if __name__ == "__main__": # only run the code below when this file is executed directly as a script. If the file is imported as a module, this block will not run.
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        print(f"\n=== PLACEBO STAGE 1: storm_year={year} ===")

        p4 = step4_csv(year)
        if not _exists(p4):
            print(f"[SKIP] Missing Step4 CSV: {p4}")
            continue

        df4 = pd.read_csv(p4) # analytical
        events = _prep_events_from_step4(df4, year) # collapses to one row per bene-storm (event level)
        if events.empty:
            print(f"[SKIP] {year}: no events from Step4")
            continue

        print(f"[INFO] {year}: events={len(events):,} | unique benes={events['BENE_ID'].nunique():,}")

        # Creates the base long-format panel by cross-joining each event row to every placebo week in PLACEBO_WEEKS. So if one bene-storm (event) exists, it becomes 5 rows: week -6 week -5 week -4 week -3 week -2
        panel = build_scaffold(events)
        if panel.empty:
            print(f"[SKIP] {year}: empty scaffold")
            continue

        # Disruption flags
        ddir = step3_detail_dir(year)
        dis = disruption_from_step3_detail(events, ddir) # return bene-storm (event) with n_dialysis, disrupt indicator
        if dis.empty:
            print(f"[WARN] {year}: disruption table empty (missing detail dir or no rows)")
        panel = panel.merge(dis, on=["event_id", "BENE_ID", "week_rel"], how="left") # merge on long-format panel
        panel["n_dialysis"] = pd.to_numeric(panel.get("n_dialysis", 0), errors="coerce").fillna(0).astype("int16")
        panel["disrupt"] = pd.to_numeric(panel.get("disrupt", 0), errors="coerce").fillna(0).astype("int8")

        # ED flags
        edir = ed_path(year)
        edf = ed_flags_by_week(events, edir) # returns with ed flags
        if edf.empty and _exists(edir):
            print(f"[INFO] {year}: no ED events found in placebo windows (this can be fine).")
        panel = panel.merge(edf, on=["event_id", "BENE_ID", "week_rel"], how="left") # merge on long-format panel
        panel["any_ed"] = pd.to_numeric(panel.get("any_ed", 0), errors="coerce").fillna(0).astype("int8") # fill in na's with 0 (those who did not have ED claims

        # IP flags
        ipdir = medpar_path(year)
        ipf = ip_flags_by_week(events, ipdir) # returns with ip flags
        if ipf.empty and _exists(ipdir):
            print(f"[INFO] {year}: no IP events found in placebo windows (this can be fine).")
        panel = panel.merge(ipf, on=["event_id", "BENE_ID", "week_rel"], how="left") # merge on long-format panel
        panel["any_ip"] = pd.to_numeric(panel.get("any_ip", 0), errors="coerce").fillna(0).astype("int8") # fill in na's with 0 (those who did not have IP claims (medpar stays)

        # For each row’s week_rel, compute the start day of that week. e.g., week -6 -> -42 (start rel day)
        panel["week_start_rel_day"] = panel["week_rel"].apply(lambda k: _week_bounds(int(k))[0]).astype("int16")
        panel["week_end_rel_day"]   = panel["week_rel"].apply(lambda k: _week_bounds(int(k))[1]).astype("int16")

        # Keep tidy column order
        col_order = [
            "year", "storm_year", "storm_id", "event_id", "BENE_ID", "fips",
            "week_rel", "week_start_rel_day", "week_end_rel_day",
            "any_ed", "any_ip", "n_dialysis", "disrupt",
            "schedule_type", "stable_3x_weekly",
            "PRVDR_NUM_event",
            "exposure_start_dt",
        ]
        panel = panel[[c for c in col_order if c in panel.columns]].sort_values(
            ["event_id", "BENE_ID", "week_rel"], kind="mergesort"
        ).reset_index(drop=True)

        # Write year output
        ydir = os.path.join(OUT_DIR, f"year_{year}")
        os.makedirs(ydir, exist_ok=True)
        out_csv = os.path.join(ydir, "analytical_panel_placebo_preweeks.csv")
        panel.to_csv(out_csv, index=False)

        # QC
        n_rows = len(panel)
        n_pairs = panel[["event_id", "BENE_ID"]].drop_duplicates().shape[0]
        print(f"[WROTE] {year}: rows={n_rows:,} | unique event-bene={n_pairs:,} | weeks={PLACEBO_WEEKS} -> {out_csv}")

    print(f"\n[DONE] Outputs under: {OUT_DIR}")