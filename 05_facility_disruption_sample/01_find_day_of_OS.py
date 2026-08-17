#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 21, 2026
# Description: This script identifies when the dialysis facility experience disruption by scanning each facility’s beneficiary 
# attendance data in rolling 7-day windows and flagging periods where a large share of patients received too few treatment 
# days.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
from pathlib import Path
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask import delayed, compute
from collections import Counter  # add near imports at top
from dask.distributed import Client
import dask

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
YEARS = list(range(2011, 2023))

OP_LINES_BASE = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis"
OPB_BASE      = "/gpfs/data/cms-share/data/medicare/{year}/otpt/opb/parquet"

OUT_BASE = "/gpfs/data/cms-share/duas/54200/Jessy/data/derived/facility_rolling_stress_days"

# Thresholds
SESSION_THRESHOLD      = 1.75   # (no longer used for selection, kept for reference/debug)
DAILY_SHARE_THRESHOLD  = 0.15   # This helps ensure the sudden change is likely operational stress. See notes under "find_stress_days_for_facility()" function for more details. This was more relevant when we want to ensure the disruption day was accurate. It is not as relevant anymore because we now use exposure start date (the day when hurricane track is closest to county centroid, but I left it untouched.
MIN_DENOM              = 11     # minimum denominator size

# Window-level disruption definition
MIN_SESSIONS_PER_BENE          = 2     # disruption is if less than this number (initiall tried less than 3 but too many facilities were present)
LOW_SESSION_PCT_THRESHOLD      = 0.33333  # > one third (33%) with <2 sessions means disrupted

# -------------------------
# Functions
# -------------------------
def read_op_lines_year(year: int) -> dd.DataFrame:
    """
    Read dialysis outpatient line items for a year.
    Assumes this directory contains ONLY dialysis OP claims.
    """
    path = f"{OP_LINES_BASE}/{year}/"
    ddf = dd.read_parquet(
        path,
        engine="pyarrow",
        columns=["BENE_ID", "CLM_ID", "REV_CNTR", "REV_CNTR_DT"],
    )
    ddf = ddf.rename(columns={"REV_CNTR_DT": "date"})
    ddf["date"] = dd.to_datetime(ddf["date"])
    return ddf


def read_opb_year(year: int) -> dd.DataFrame:
    """
    Read OPB (base) file for a year, to get facility IDs and ZIPs.
    """
    path = OPB_BASE.format(year=year)
    ddf = dd.read_parquet(
        path,
        engine="pyarrow",
        columns=["CLM_ID", "PRVDR_NUM", "CLM_SRVC_FAC_ZIP_CD"],
    )
    ddf["PRVDR_NUM"] = ddf["PRVDR_NUM"].astype(str).str.zfill(6)
    ddf["CLM_SRVC_FAC_ZIP_CD"] = ddf["CLM_SRVC_FAC_ZIP_CD"].astype(str)
    return ddf


def find_stress_days_for_facility(
    fac_df: pd.DataFrame,
    session_threshold: float = SESSION_THRESHOLD,          # kept but not used
    daily_share_threshold: float = DAILY_SHARE_THRESHOLD,
    min_gap_days: int = 21,   # 3 weeks
) -> pd.DataFrame:
    """
    Summary - For a single facility (fac_df with BENE_ID, date), find all
    stress days that satisfy, for some 7-day window:
      - Among all benes treated at this facility at least once
        in the 7-day window, the fraction with < MIN_SESSIONS_PER_BENE
        visits during that window is > LOW_SESSION_PCT_THRESHOLD.
      - day(-1) (the calendar day before the window start) has
        share >= daily_share_threshold.
      - day0 and day1 (window start and the next day) each have
        share < daily_share_threshold.
      - If day0 is Saturday, we additionally require that day+2
        (Monday) also has share < daily_share_threshold
      - stress_day is the window start (d0), except if d0 is Sunday,
        then stress_day = Monday.

    See below comments for more details
    """
    if fac_df.empty: # if the facility has no data, return an empty result
        return pd.DataFrame(
            columns=[
                "earliest_stress_day",
                "earliest_avg_sessions",
                "earliest_denom",
                "pct_lt3_sessions",
            ]
        )

    df = fac_df.copy() # should be one facility but remember the @delayed allows for multiple pd df's of facilities to run in parallel
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Collapse to one row per bene-date to avoid double counting
    df = df.drop_duplicates(subset=["BENE_ID", "date"])

    # For each date, which beneficiaries showed up at this one facility. Basically, This creates a dictionary where each key is a date and each value is the set of beneficiaries seen on that date at this facility
    bene_by_day = (
        df.groupby("date")["BENE_ID"]
          .agg(lambda s: set(s.tolist()))
          .to_dict()
    )

    if not bene_by_day: # if that map is empty, return nothing
        return pd.DataFrame(
            columns=[
                "earliest_stress_day",
                "earliest_avg_sessions",
                "earliest_denom",
                "pct_lt3_sessions",
            ]
        )

    # This creates every calendar day from facility’s earliest observed date to its latest date. That matters b/c the function wants to slide a 7-day window across every possible start day, including days with no visits.
    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    candidates = []  # store all potential stress days first

    # Slide a 7-day window: start, start+1, ..., start+6. So the function is checking: days 1–7 then days 2–8 then days 3–9 and so on for this facility
    for start in all_dates:
        end = start + pd.Timedelta(days=6)
        if end > all_dates[-1]:
            # Not enough days left for a full 7-day window then stop
            break

        window_dates = [start + pd.Timedelta(days=i) for i in range(7)]

        # Count sessions per bene over the 7-day window. So if a bene has 3 then they showed up 3 times in that rolling week
        bene_counts = Counter() # a counting dictionary
        for d in window_dates:
            for b in bene_by_day.get(d, set()):
                bene_counts[b] += 1

        # denom is the number of unique beneficiaries seen at least once in that 7-day window.
        denom = len(bene_counts)
        if denom == 0:
            continue

        # Percent of benes with < MIN_SESSIONS_PER_BENE sessions in the window
        # e.g. count beneficiaries with fewer than 2 attendance days in the 7-day window then divide by the denominator to get the percentage.
        n_lt3 = sum(
            1 for c in bene_counts.values()
            if c < MIN_SESSIONS_PER_BENE
        )
        pct_lt3 = n_lt3 / denom

        # If that fraction is not above threshold, skip. E.g., keep if more than one-third of beneficiaries had fewer than 2 sessions in that 7-day period
        if pct_lt3 <= LOW_SESSION_PCT_THRESHOLD:
            continue

        # Just for descriptives: average sessions per bene in this window.
        total_sessions = sum(bene_counts.values())
        avg_sessions = total_sessions / denom

        # --- Calculates: attendance share on the day before the window (-1), attendance share on the first day of the window (0), attendance share on the second day of the window (1) ---

        # Daily share checks for day -1, 0, 1
        d0 = window_dates[0]
        d1 = window_dates[1]
        d_prev = d0 - pd.Timedelta(days=1)

        n0 = len(bene_by_day.get(d0, set()))
        n1 = len(bene_by_day.get(d1, set()))
        n_prev = len(bene_by_day.get(d_prev, set()))

        share0 = n0 / denom
        share1 = n1 / denom
        share_prev = n_prev / denom if denom > 0 else 0.0

        # --- Saturday safeguard using day+2 (Monday) ---
        # If the candidate window starts on Saturday (weekday=5),
        # and Monday has a "normal" share (>= threshold),
        # treat this as a regular M/W/F pattern and skip.
        if d0.weekday() == 5:  # Saturday
            d2 = d0 + pd.Timedelta(days=2)
            n2 = len(bene_by_day.get(d2, set()))
            share2 = n2 / denom
            if share2 >= daily_share_threshold:
                # Monday looks normal, so don't call Saturday a stress day. I.e., if the candidate window starts on Saturday, the code checks Monday. If Monday looks normal, it assumes Saturday’s low attendance may just reflect a normal schedule pattern rather than a disruption, so it skips that candidate.
                continue

        # Condition: day(-1) "normal", day0 & day1 "low". This makes sure the window is a sudden operational drop, not just a generally low week.
        if (
            share_prev >= daily_share_threshold
            and share0 < daily_share_threshold
            and share1 < daily_share_threshold
        ):
            # This window is a stress window; define stress day as the start day of the 7 day rolling period.
            stress_day = d0
            # If start day is Sunday, shift label to Monday. No facilities are opened on Sunday
            if stress_day.weekday() == 6:  # Sun
                stress_day = stress_day + pd.Timedelta(days=1)

            # stores the stress day along with summary metrics
            candidates.append(
                {
                    "earliest_stress_day": stress_day,
                    "earliest_avg_sessions": avg_sessions,
                    "earliest_denom": denom,
                    "pct_lt3_sessions": pct_lt3,
                }
            )

    # If none were found, return empty
    if not candidates:
        return pd.DataFrame(
            columns=[
                "earliest_stress_day",
                "earliest_avg_sessions",
                "earliest_denom",
                "pct_lt3_sessions",
            ]
        )

    # Convert to DataFrame and sort by stress day
    cand_df = pd.DataFrame(candidates).sort_values("earliest_stress_day")

    # If multiple windows map to same stress_day (e.g., Sunday→Monday), deduplicate
    # i.e., different rolling windows can sometimes map to the same stress day, especially with the Sunday-to-Monday relabeling. This line keeps only one row per stress day.
    cand_df = cand_df.drop_duplicates(subset=["earliest_stress_day"])

    # Enforce min_gap_days between stress days
    # Basically, this prevents one extended disruption from being counted as many separate events.
    selected_rows = []
    last_kept_day = None # at the beginning, there is no previously accepted stress day

    for _, row in cand_df.iterrows(): # Loop through candidate stress days
        day = row["earliest_stress_day"]
        if last_kept_day is None or (day - last_kept_day).days >= min_gap_days: # min_gap_days set above to 21 days or 3 weeks. Thus it means keep it only if it is at least 21 days later form the last_kept_day
            selected_rows.append(row)
            last_kept_day = day # replace.
        else:
            # Within 3 weeks of a previous stress day: treat as same event, drop
            continue

    # Return an empty table if none.
    if not selected_rows:
        return pd.DataFrame(
            columns=[
                "earliest_stress_day",
                "earliest_avg_sessions",
                "earliest_denom",
                "pct_lt3_sessions",
            ]
        )

    return pd.DataFrame(selected_rows)

@delayed
# ^ this turns a normal Python function into a lazy Dask task. Basically, do not run it until compute so that tasks can be runned in parallel by Dask. 
def process_facility(prvdr: str, fac_df: pd.DataFrame, year: int) -> pd.DataFrame:
    res_df = find_stress_days_for_facility(fac_df)

    if res_df.empty:
        # Return empty with the expected columns so concat works
        res_df = pd.DataFrame(
            columns=["earliest_stress_day", "earliest_avg_sessions", "earliest_denom"]
        )

    # Append columns
    res_df["PRVDR_NUM"] = prvdr
    res_df["year"] = year
    return res_df

def process_year(year: int):
    # Year-level driver:
    #  1) Read OP lines + OPB with Dask.
    #  2) Merge on CLM_ID to add PRVDR_NUM, ZIP.
    #  3) Collapse to facility-bene-day (one row per PRVDR_NUM–BENE_ID–date).
    #  4) For each facility, run rolling 7-day logic in pandas (delayed).
    #  5) Combine outputs and export CSV.

    print(f"Currently on {year}")

    out_dir = Path(OUT_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read OP + OPB
    op_dd = read_op_lines_year(year)
    opb_dd = read_opb_year(year)

    merged_dd = op_dd.merge(opb_dd, on="CLM_ID", how="left")

    # Keep only needed columns
    merged_dd = merged_dd[["PRVDR_NUM", "BENE_ID", "date", "CLM_SRVC_FAC_ZIP_CD"]]

    # Drop rows without identifiable facility
    merged_dd = merged_dd[merged_dd["PRVDR_NUM"].notnull()]

    # 2) Build:
    #   - fac_bene_day_dd: one row per facility–bene–date. It will be used to help answer: who showed up at each facility on each day? This is important to help count the dialysis sessions and see if the facility has disruption/stress
    #   - fac_zip_dd: one row per facility with its ZIP (any). It will be used to append back the ZIP codes.
    fac_bene_day_dd = merged_dd[["PRVDR_NUM", "BENE_ID", "date"]].drop_duplicates()
    fac_zip_dd = merged_dd[["PRVDR_NUM", "CLM_SRVC_FAC_ZIP_CD"]].drop_duplicates(
        subset=["PRVDR_NUM"]
    )

    fac_bene_day, fac_zip = compute(fac_bene_day_dd, fac_zip_dd) # Turn them to pandas DF

    if fac_bene_day.empty: # QC checks
        print(f"No facility-bene-day data for {year}")
        return

    # 3) Group by facility and create delayed tasks - one for each facility.
    tasks = []
    for prvdr, fac_df in fac_bene_day.groupby("PRVDR_NUM"):
        tasks.append(process_facility(prvdr, fac_df, year))

    if not tasks: # QC checks
        print(f"No facilities found for {year}")
        return

    facility_results = compute(*tasks)  # Run the @delayed task now in parallel (meaning check multiple facilities at the same time). tuple/list of PD DataFrames
    out = pd.concat(facility_results, ignore_index=True)

    # Filter: keep facilities with a valid earliest_stress_day and denom >= MIN_DENOM
    out = out[
        out["earliest_stress_day"].notna()
        & (out["earliest_denom"] >= MIN_DENOM)
    ]

    if out.empty:
        print(f"No qualifying stress days for {year}")
        return

    # 4) Attach ZIP codes
    fac_zip = fac_zip.rename(columns={"CLM_SRVC_FAC_ZIP_CD": "zip"})
    out = out.merge(fac_zip, on="PRVDR_NUM", how="left")

    # 5) Sort and preview
    out = out.sort_values(["earliest_stress_day", "zip", "PRVDR_NUM"])
    # print(out.head(30))

    # 6) Export
    out_file = out_dir / f"facility_rolling_stress_days_{year}.csv"
    out.to_csv(out_file, index=False)
    print(f" wrote to {out_file}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    for y in YEARS:
        process_year(y)