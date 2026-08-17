#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 27, 2026
# Description: This script takes the dialysis line items and shifts the exposure date 35 days earlier to create a placebo 
# version of the analysis. It then rebuilds the analytical file around that placebo anchor date, merges in ED, inpatient, 
# and MBSF death information, and outputs a two-row long placebo analytical panel per beneficiary-storm event: a placebo 
# reference week row (week_rel = -7) and a placebo hazard week row (week_rel = -5) for later within-beneficiary analysis.
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
client = Client("10.50.87.31:42109")
print(client)

# -------------------------
# Paths and spec
# -------------------------
YEAR_MIN, YEAR_MAX = 2011, 2022
STRICT_STABLE_SCHEDULE = True
DROP_DUPLICATE_BENE_STORM_WEEK = True

PLACEBO_SHIFT_DAYS = 35  # move anchor 5 weeks earlier

# ... Placebo window def (all RELATIVE to placebo exposure_start_dt which is 5 weeks prior to actual exposure start date]) ...
# Panel rows:
REF_WK_M7_START, REF_WK_M7_END = -14, -8   # week -7
HAZ_WK_M5_START, HAZ_WK_M5_END = 0, 6      # week -5

# Additional weekly placebo outcome windows
WK_M4_START, WK_M4_END = 7, 13     # week -4
WK_M3_START, WK_M3_END = 14, 20    # week -3
WK_M2_START, WK_M2_END = 21, 27    # week -2

# Cumulative placebo post windows
POST_2WK_START, POST_2WK_END = 0, 13   # weeks -5 to -4
POST_3WK_START, POST_3WK_END = 0, 20   # weeks -5 to -3
POST_4WK_START, POST_4WK_END = 0, 27   # weeks -5 to -2

# Schedule classification window (week -8) (for MWF/TTS classify)
SCHED_START, SCHED_END = -21, -15

# Facility PRE window for PRVDR_NUM_event (classify "home" facility for each bene)
FAC_PRE_START, FAC_PRE_END = -21, -8

# 2-week dialysis windows retained for quality check
TWK_PRIOR_START, TWK_PRIOR_END = -21, -8
TWK_POST_START, TWK_POST_END   = 0, 13

# ... Paths ...
STEP3_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "bene_storm_hd_services_pm1mo_byyear_v01"
)

def step3_year_dir(year: int) -> str:
    return os.path.join(STEP3_BASE, f"year={year}")

def step3_detail_dir(year: int) -> str:
    return os.path.join(step3_year_dir(year), "detail_hd_lines")

def step3_summary_dir(year: int) -> str:
    return os.path.join(step3_year_dir(year), "summary_bene_storm_hd")

def medpar_path(year: int) -> str:
    return f"/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/00b_hospital_SL/{year}/"

def ed_path(year: int) -> str:
    return f"/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/00c/{year}/"

def mbsf_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/mbsf/mbsf_abcd/parquet/"

OUT_BASE = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis"
OUT_DIR  = os.path.join(OUT_BASE, "04_placebo_analytical_panel_hurr_exposure_v02_wkm7_vs_m5_facclust_cumpost_cumdeath_cumdisrupt")
os.makedirs(OUT_DIR, exist_ok=True)

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

def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame: # dataframe-level ID cleaning function
    df = df.copy()
    if "BENE_ID" in df.columns:
        df["BENE_ID"] = _as_clean_str(df["BENE_ID"])
    if "storm_id" in df.columns:
        df["storm_id"] = _as_clean_str(df["storm_id"])
    if "fips" in df.columns:
        df["fips"] = _as_clean_str(df["fips"]).str.zfill(5)
    if "PRVDR_NUM" in df.columns:
        df["PRVDR_NUM"] = _as_clean_str(df["PRVDR_NUM"]).str.zfill(6)
    return df

def _parse_mbsf_date_series(s: pd.Series) -> pd.Series: # parses MBSF date fields into normalized pandas datetimes
    ss = _as_clean_str(s)
    dt = pd.to_datetime(ss, format="%Y%m%d", errors="coerce")
    miss = dt.isna() & ss.notna()
    if miss.any():
        dt.loc[miss] = pd.to_datetime(ss.loc[miss], errors="coerce")
    return dt.dt.normalize()

def _eventize_from_step3_summary(summary: pd.DataFrame) -> pd.DataFrame: # creates the event-level cohort from the Step 3 summary file. So one row is BENE_ID × storm_id × fips × exposure_start_dt
    s = _normalize_ids(summary)
    s["exposure_start_dt"] = pd.to_datetime(s["exposure_start_dt"], errors="coerce").dt.normalize()
    s = s[s["exposure_start_dt"].notna()].copy()

    keep = ["BENE_ID", "storm_id", "storm_year", "fips", "exposure_start_dt"]
    extra = [c for c in ["n_hd_lines", "n_hd_days", "has_hd"] if c in s.columns]
    s = s[keep + extra].drop_duplicates().reset_index(drop=True)

    s["event_id"] = np.arange(len(s), dtype=int) # assigns a new sequential event_id
    return s

def _mode_min_tiebreak(series: pd.Series): # avoid random tie behavior.
    s = series.dropna()
    if s.empty:
        return pd.NA
    vc = s.value_counts() # count how many times each remaining value appears
    maxc = vc.max() # find the values with the highest count
    tied = sorted(vc[vc == maxc].index.tolist()) # however, if there is a tie, sort those tied values and pick the smallest string
    return tied[0] if tied else pd.NA

def dialysis_features_from_step3_detail(events: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame: # takes two pandas dataframes
    """
    Turns the line items from the previous script into beneficiary-storm level for identification of placebo dialysis analysis

    Placebo anchor = real exposure_start_dt - 35 days

    Schedule inference window (i.e. MWF or TTS):
        rel_day in [-21, -15]  -> week -8

    Placebo reference week in panel:
        week -7 -> rel_day [-14, -8]

    Early dialysis definition (EarlyA):
      - among stable 3x/wk (MWF or TTS), set earlyA_last_pre_offschedule=1 if
        LAST dialysis day in week -1 (rel_day [-7,-1]) is off-schedule.

    Facility for clustering:
      - PRVDR_NUM_event (the "home" provider of the bene) = modal provider in PRE period dialysis DAYS rel_day [-21,-8].
    """
    if events.empty: # If no events, returns an empty dataframe
        return pd.DataFrame(columns=[
            "event_id","BENE_ID",
            "n_dialysis_wk_m7","n_dialysis_wk_m5","n_dialysis_wk_m4","n_dialysis_wk_m3","n_dialysis_wk_m2",
            "n_dialysis_2wk_prior","n_dialysis_2wk_post",
            "gap_days","no_hazard_dialysis",
            "schedule_type","stable_3x_weekly",
            "earlyA_last_pre_offschedule",
            "PRVDR_NUM_event",
        ])

    d = _normalize_ids(detail)
    d["exposure_start_dt"] = pd.to_datetime(d["exposure_start_dt"], errors="coerce").dt.normalize()
    d["REV_CNTR_DT"] = pd.to_datetime(d["REV_CNTR_DT"], errors="coerce").dt.normalize()
    d = d[(d["exposure_start_dt"].notna()) & (d["REV_CNTR_DT"].notna())].copy()

    if "PRVDR_NUM" not in d.columns:
        d["PRVDR_NUM"] = pd.NA
    else:
        d["PRVDR_NUM"] = _as_clean_str(d["PRVDR_NUM"]).str.zfill(6)

    # "d" is the line items. Merge "d" to the summary event list to obtain event_id
    key = ["BENE_ID", "storm_id", "fips", "exposure_start_dt"]
    ev_key = events[key + ["event_id", "placebo_exposure_start_dt"]].copy()
    d = d.merge(ev_key, on=key, how="inner")

    d["placebo_exposure_start_dt"] = pd.to_datetime(
        d["placebo_exposure_start_dt"], errors="coerce"
    ).dt.normalize()

    d["rel_day"] = (d["REV_CNTR_DT"] - d["placebo_exposure_start_dt"]).dt.days.astype("int16")  # Compute relative day from exposure (placebo)

    # Collapse to one dialysis day per event-beneficiary-date. Important because it creates a dataset with one row per dialysis day
    d = d.sort_values(["event_id", "BENE_ID", "REV_CNTR_DT", "PRVDR_NUM"])
    op_day = d[[
        "event_id", "BENE_ID", "placebo_exposure_start_dt",
        "REV_CNTR_DT", "rel_day", "PRVDR_NUM"
    ]].drop_duplicates(
        subset=["event_id", "BENE_ID", "REV_CNTR_DT"],
        keep="first"
    )

    # Create week/window indicators
    op_day["wk_m7"]  = ((op_day["rel_day"] >= REF_WK_M7_START) & (op_day["rel_day"] <= REF_WK_M7_END)).astype("int8")
    op_day["wk_m5"]  = ((op_day["rel_day"] >= HAZ_WK_M5_START) & (op_day["rel_day"] <= HAZ_WK_M5_END)).astype("int8")
    op_day["wk_m4"]  = ((op_day["rel_day"] >= WK_M4_START) & (op_day["rel_day"] <= WK_M4_END)).astype("int8")
    op_day["wk_m3"]  = ((op_day["rel_day"] >= WK_M3_START) & (op_day["rel_day"] <= WK_M3_END)).astype("int8")
    op_day["wk_m2"]  = ((op_day["rel_day"] >= WK_M2_START) & (op_day["rel_day"] <= WK_M2_END)).astype("int8")
    op_day["prior2"] = ((op_day["rel_day"] >= TWK_PRIOR_START) & (op_day["rel_day"] <= TWK_PRIOR_END)).astype("int8")
    op_day["post2"]  = ((op_day["rel_day"] >= TWK_POST_START) & (op_day["rel_day"] <= TWK_POST_END)).astype("int8")

    op_day["class_week"] = ((op_day["rel_day"] >= SCHED_START) & (op_day["rel_day"] <= SCHED_END)).astype("int8") # Mark the schedule-classification window

    # NOTE: while we do create the MWF/TTS, it is NOT relevant for the placebo analysis for exhibit 6. But the codes are kept in case we want to look at these folks
    op_day["dow"] = op_day["REV_CNTR_DT"].dt.weekday # Derive day-of-week information for each dialysis dates
    op_day["is_mwf_day"] = (op_day["dow"].isin([0, 2, 4])).astype("int8") # MWF
    op_day["is_tts_day"] = (op_day["dow"].isin([1, 3, 5])).astype("int8") # TTS

    # Save dates needed later for gap and early dialysis calculations
    op_day["date_pre_ref"] = op_day["REV_CNTR_DT"].where(op_day["wk_m7"] == 1)
    op_day["date_post"]    = op_day["REV_CNTR_DT"].where(op_day["wk_m5"] == 1)

    # Mark schedule-type days inside the classification window
    op_day["class_MWF"]   = ((op_day["class_week"] == 1) & (op_day["is_mwf_day"] == 1)).astype("int8")
    op_day["class_TTS"]   = ((op_day["class_week"] == 1) & (op_day["is_tts_day"] == 1)).astype("int8")
    op_day["class_other"] = ((op_day["class_week"] == 1) & (op_day["is_mwf_day"] == 0) & (op_day["is_tts_day"] == 0)).astype("int8")

    grp = ( # Collapse to one row per event-beneficiary
        op_day.groupby(["event_id", "BENE_ID"], as_index=False)
        .agg(
            n_dialysis_wk_m7=("wk_m7", "sum"),
            n_dialysis_wk_m5=("wk_m5", "sum"),
            n_dialysis_wk_m4=("wk_m4", "sum"),
            n_dialysis_wk_m3=("wk_m3", "sum"),
            n_dialysis_wk_m2=("wk_m2", "sum"),
            n_dialysis_2wk_prior=("prior2", "sum"),
            n_dialysis_2wk_post=("post2", "sum"),
            date_pre_ref=("date_pre_ref", "max"),
            date_post=("date_post", "min"),
            n_class_total=("class_week", "sum"),
            n_class_MWF=("class_MWF", "sum"),
            n_class_TTS=("class_TTS", "sum"),
            n_class_other=("class_other", "sum"),
        )
    )

    for c in [
        "n_dialysis_wk_m7","n_dialysis_wk_m5","n_dialysis_wk_m4","n_dialysis_wk_m3","n_dialysis_wk_m2",
        "n_dialysis_2wk_prior","n_dialysis_2wk_post",
        "n_class_total","n_class_MWF","n_class_TTS","n_class_other",
    ]:
        grp[c] = grp[c].fillna(0).astype("int16") # For all the count variables, replace missing with 0 and store them as integers.

    # Compute gap days between last pre-storm dialysis and first hazard-week dialysis. Note that we will NOT use this gap-day for exhibit 6 placebo analysis.
    grp["gap_days"] = np.nan
    valid_gap = grp["date_pre_ref"].notna() & grp["date_post"].notna()
    grp.loc[valid_gap, "gap_days"] = (
        grp.loc[valid_gap, "date_post"] - grp.loc[valid_gap, "date_pre_ref"]
    ).dt.days.astype("float")
    grp["no_hazard_dialysis"] = grp["date_post"].isna().astype("int8")

    grp["schedule_type"] = pd.NA
    if STRICT_STABLE_SCHEDULE: # if true, then apply the strict rule (e.g., MWF requires exactly 3 dialysis days in the classification window, all on MWF days. If false, then apply looser rule. We chose to go with the strict rule.
        cond_mwf = (grp["n_class_total"] == 3) & (grp["n_class_MWF"] == 3) & (grp["n_class_TTS"] == 0) & (grp["n_class_other"] == 0)
        cond_tts = (grp["n_class_total"] == 3) & (grp["n_class_TTS"] == 3) & (grp["n_class_MWF"] == 0) & (grp["n_class_other"] == 0)
    else:
        cond_mwf = (grp["n_class_total"] >= 2) & (grp["n_class_MWF"] == grp["n_class_total"]) & (grp["n_class_TTS"] == 0) & (grp["n_class_other"] == 0)
        cond_tts = (grp["n_class_total"] >= 2) & (grp["n_class_TTS"] == grp["n_class_total"]) & (grp["n_class_MWF"] == 0) & (grp["n_class_other"] == 0)

    grp.loc[cond_mwf, "schedule_type"] = "MWF" # Assign the schedule label where the conditions hold
    grp.loc[cond_tts, "schedule_type"] = "TTS"
    grp["stable_3x_weekly"] = grp["schedule_type"].notna().astype("int8") # If the schedule was successfully assigned, mark the beneficiary as stable 3x weekly.

    grp["earlyA_last_pre_offschedule"] = 0 # Initialize early dialysis as 0 for everyone
    grp["pre_ref_dow"] = pd.to_datetime(grp["date_pre_ref"], errors="coerce").dt.weekday

    # These masks restrict the early dialysis (earlyA) logic to beneficiaries who: have a stable schedule, are specifically MWF or TTS, and actually have a last dialysis date in week -1.
    mwf_mask = (grp["stable_3x_weekly"] == 1) & (grp["schedule_type"] == "MWF") & grp["date_pre_ref"].notna()
    tts_mask = (grp["stable_3x_weekly"] == 1) & (grp["schedule_type"] == "TTS") & grp["date_pre_ref"].notna()

    grp.loc[mwf_mask & (~grp["pre_ref_dow"].isin([0, 2, 4])), "earlyA_last_pre_offschedule"] = 1 # if someone is MWF, but their last week -1 dialysis day is not Monday, Wednesday, or Friday, flag them as early/off-schedule
    grp.loc[tts_mask & (~grp["pre_ref_dow"].isin([1, 3, 5])), "earlyA_last_pre_offschedule"] = 1 # if someone is TTS, but their last week -1 dialysis day is not Tuesday, Thursday, or Saturday, flag them too

    # Assign an event-level provider number (home dialysis center)
    pre = op_day[(op_day["rel_day"] >= FAC_PRE_START) & (op_day["rel_day"] <= FAC_PRE_END)].copy()
    pre["PRVDR_NUM"] = _as_clean_str(pre["PRVDR_NUM"]).str.zfill(6)

    prv = (
        pre.groupby(["event_id", "BENE_ID"])["PRVDR_NUM"]
           .apply(_mode_min_tiebreak)
           .reset_index()
           .rename(columns={"PRVDR_NUM": "PRVDR_NUM_event"})
    ) # For each event-beneficiary, find the most common provider in the pre-period. If there is a tie, _mode_min_tiebreak picks the smallest string

    grp = grp.merge(prv, on=["event_id", "BENE_ID"], how="left") # Merge provider back into the grouped output

    return grp[[
        "event_id","BENE_ID",
        "n_dialysis_wk_m7","n_dialysis_wk_m5","n_dialysis_wk_m4","n_dialysis_wk_m3","n_dialysis_wk_m2",
        "n_dialysis_2wk_prior","n_dialysis_2wk_post",
        "gap_days","no_hazard_dialysis",
        "schedule_type","stable_3x_weekly",
        "earlyA_last_pre_offschedule",
        "PRVDR_NUM_event",
    ]]

# ... Outcomes ...
# These are relevant for the placebo analysis

def outcome_flags_from_ed_year(events: pd.DataFrame, ed_pq: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=[
            "event_id", "BENE_ID",
            "any_ed_wk_m7", "any_ed_wk_m5", "any_ed_wk_m4", "any_ed_wk_m3", "any_ed_wk_m2",
            "any_ed_post_2wk", "any_ed_post_3wk", "any_ed_post_4wk",
            "any_ed_2wk_prior", "any_ed_2wk_post",
        ])

    benes = pd.DataFrame({"BENE_ID": _as_clean_str(events["BENE_ID"]).unique()}) # pulls the unique bene ids from the event cohort
    benes_dd = dd.from_pandas(benes, npartitions=1)
    benes_dd["BENE_ID"] = benes_dd["BENE_ID"].astype(str)

    tmin = events["placebo_exposure_start_dt"].min() + pd.Timedelta(days=TWK_PRIOR_START) # earliest exposure date in the cohort, shifted backward to the start of the two-week pre window
    tmax = events["placebo_exposure_start_dt"].max() + pd.Timedelta(days=POST_4WK_END) # latest exposure date in the cohort, shifted forward to the end of the 4-week post window

    ed = dd.read_parquet(ed_pq, columns=["BENE_ID", "REV_CNTR_DT"])
    ed["BENE_ID"] = ed["BENE_ID"].astype(str)
    ed = ed.assign(date=dd.to_datetime(ed["REV_CNTR_DT"], errors="coerce"))
    ed = ed[(ed["date"] >= tmin) & (ed["date"] <= tmax)]
    ed = ed.merge(benes_dd, on="BENE_ID", how="inner") # keep only ED records for beneficiaries who are actually in the event cohort.

    cohort_dd = dd.from_pandas(
        events[["event_id", "BENE_ID", "placebo_exposure_start_dt"]].copy(),
        npartitions=1
    )
    cohort_dd["BENE_ID"] = cohort_dd["BENE_ID"].astype(str)

    ed = ed.merge(cohort_dd, on="BENE_ID", how="inner") # links each ED record to every event row for the same beneficiary. So if a beneficiary has multiple storm events, the same ED visit can temporarily pair with multiple events. That is intentional, because the next step calculates rel_day separately for each event.
    ed = ed.assign(rel_day=(ed["date"] - ed["placebo_exposure_start_dt"]).dt.days) # each ED-event pairing, compute the number of days between the ED date and the storm exposure date.

    ed = ed.assign( # creates window-specific indicators
        flag_wk_m7=((ed["rel_day"] >= REF_WK_M7_START) & (ed["rel_day"] <= REF_WK_M7_END)).astype("int8"),
        flag_wk_m5=((ed["rel_day"] >= HAZ_WK_M5_START) & (ed["rel_day"] <= HAZ_WK_M5_END)).astype("int8"),
        flag_wk_m4=((ed["rel_day"] >= WK_M4_START) & (ed["rel_day"] <= WK_M4_END)).astype("int8"),
        flag_wk_m3=((ed["rel_day"] >= WK_M3_START) & (ed["rel_day"] <= WK_M3_END)).astype("int8"),
        flag_wk_m2=((ed["rel_day"] >= WK_M2_START) & (ed["rel_day"] <= WK_M2_END)).astype("int8"),
        flag_post_2wk=((ed["rel_day"] >= POST_2WK_START) & (ed["rel_day"] <= POST_2WK_END)).astype("int8"),
        flag_post_3wk=((ed["rel_day"] >= POST_3WK_START) & (ed["rel_day"] <= POST_3WK_END)).astype("int8"),
        flag_post_4wk=((ed["rel_day"] >= POST_4WK_START) & (ed["rel_day"] <= POST_4WK_END)).astype("int8"),
        flag_2wk_prior=((ed["rel_day"] >= TWK_PRIOR_START) & (ed["rel_day"] <= TWK_PRIOR_END)).astype("int8"),
        flag_2wk_post=((ed["rel_day"] >= TWK_POST_START) & (ed["rel_day"] <= TWK_POST_END)).astype("int8"),
    ) # So at this stage, each ED row has a set of 0/1 markers for all windows it belongs to.

    grp = ( # group the ED records by event_id and BENE_ID. This means the goal is one row per beneficiary-storm event. This is important: taking the maximum across ED rows turns the visit-level flags into yes/no event-level indicators: if at least one ED visit in that event-window has flag 1, the grouped result is 1 and if none do, it stays 0.
        ed.groupby(["event_id", "BENE_ID"])[[
            "flag_wk_m7","flag_wk_m5","flag_wk_m4","flag_wk_m3","flag_wk_m2",
            "flag_post_2wk","flag_post_3wk","flag_post_4wk",
            "flag_2wk_prior","flag_2wk_post"
        ]]
        .max()
        .rename(columns={
            "flag_wk_m7": "any_ed_wk_m7",
            "flag_wk_m5": "any_ed_wk_m5",
            "flag_wk_m4": "any_ed_wk_m4",
            "flag_wk_m3": "any_ed_wk_m3",
            "flag_wk_m2": "any_ed_wk_m2",
            "flag_post_2wk": "any_ed_post_2wk",
            "flag_post_3wk": "any_ed_post_3wk",
            "flag_post_4wk": "any_ed_post_4wk",
            "flag_2wk_prior": "any_ed_2wk_prior",
            "flag_2wk_post": "any_ed_2wk_post",
        })
        .reset_index()
        .compute()
    )
    grp["BENE_ID"] = _as_clean_str(grp["BENE_ID"])
    return grp

def outcome_flags_from_ip_year(events: pd.DataFrame, medpar_pq: str) -> pd.DataFrame: # This process for IP is VERY similar to ED above. Please see comments above (under function outcome_flags_from_ed_year()) for more details
    if events.empty:
        return pd.DataFrame(columns=[
            "event_id", "BENE_ID",
            "any_ip_wk_m7", "any_ip_wk_m5", "any_ip_wk_m4", "any_ip_wk_m3", "any_ip_wk_m2",
            "any_ip_post_2wk", "any_ip_post_3wk", "any_ip_post_4wk",
            "any_ip_2wk_prior", "any_ip_2wk_post",
        ])

    benes = pd.DataFrame({"BENE_ID": _as_clean_str(events["BENE_ID"]).unique()})
    benes_dd = dd.from_pandas(benes, npartitions=1)
    benes_dd["BENE_ID"] = benes_dd["BENE_ID"].astype(str)

    tmin = events["placebo_exposure_start_dt"].min() + pd.Timedelta(days=TWK_PRIOR_START)
    tmax = events["placebo_exposure_start_dt"].max() + pd.Timedelta(days=POST_4WK_END)

    ip = dd.read_parquet(medpar_pq, columns=["BENE_ID", "MEDPAR_ID", "PRVDR_NUM", "ADMSN_DT", "DSCHRG_DT"])
    ip["BENE_ID"] = ip["BENE_ID"].astype(str)
    ip = ip.assign(ADMSN_DT=dd.to_datetime(ip["ADMSN_DT"], errors="coerce"))
    ip = ip[(ip["ADMSN_DT"] >= tmin) & (ip["ADMSN_DT"] <= tmax)]
    ip = ip.merge(benes_dd, on="BENE_ID", how="inner")

    cohort_dd = dd.from_pandas(
        events[["event_id", "BENE_ID", "placebo_exposure_start_dt"]].copy(),
        npartitions=1
    )
    cohort_dd["BENE_ID"] = cohort_dd["BENE_ID"].astype(str)

    ip = ip.merge(cohort_dd, on="BENE_ID", how="inner")
    ip = ip.assign(rel_day=(ip["ADMSN_DT"] - ip["placebo_exposure_start_dt"]).dt.days)

    ip = ip.assign(
        flag_wk_m7=((ip["rel_day"] >= REF_WK_M7_START) & (ip["rel_day"] <= REF_WK_M7_END)).astype("int8"),
        flag_wk_m5=((ip["rel_day"] >= HAZ_WK_M5_START) & (ip["rel_day"] <= HAZ_WK_M5_END)).astype("int8"),
        flag_wk_m4=((ip["rel_day"] >= WK_M4_START) & (ip["rel_day"] <= WK_M4_END)).astype("int8"),
        flag_wk_m3=((ip["rel_day"] >= WK_M3_START) & (ip["rel_day"] <= WK_M3_END)).astype("int8"),
        flag_wk_m2=((ip["rel_day"] >= WK_M2_START) & (ip["rel_day"] <= WK_M2_END)).astype("int8"),
        flag_post_2wk=((ip["rel_day"] >= POST_2WK_START) & (ip["rel_day"] <= POST_2WK_END)).astype("int8"),
        flag_post_3wk=((ip["rel_day"] >= POST_3WK_START) & (ip["rel_day"] <= POST_3WK_END)).astype("int8"),
        flag_post_4wk=((ip["rel_day"] >= POST_4WK_START) & (ip["rel_day"] <= POST_4WK_END)).astype("int8"),
        flag_2wk_prior=((ip["rel_day"] >= TWK_PRIOR_START) & (ip["rel_day"] <= TWK_PRIOR_END)).astype("int8"),
        flag_2wk_post=((ip["rel_day"] >= TWK_POST_START) & (ip["rel_day"] <= TWK_POST_END)).astype("int8"),
    )

    grp = (
        ip.groupby(["event_id", "BENE_ID"])[[
            "flag_wk_m7","flag_wk_m5","flag_wk_m4","flag_wk_m3","flag_wk_m2",
            "flag_post_2wk","flag_post_3wk","flag_post_4wk",
            "flag_2wk_prior","flag_2wk_post"
        ]]
        .max()
        .rename(columns={
            "flag_wk_m7": "any_ip_wk_m7",
            "flag_wk_m5": "any_ip_wk_m5",
            "flag_wk_m4": "any_ip_wk_m4",
            "flag_wk_m3": "any_ip_wk_m3",
            "flag_wk_m2": "any_ip_wk_m2",
            "flag_post_2wk": "any_ip_post_2wk",
            "flag_post_3wk": "any_ip_post_3wk",
            "flag_post_4wk": "any_ip_post_4wk",
            "flag_2wk_prior": "any_ip_2wk_prior",
            "flag_2wk_post": "any_ip_2wk_post",
        })
        .reset_index()
        .compute()
    )
    grp["BENE_ID"] = _as_clean_str(grp["BENE_ID"])
    return grp

# ... MBSF ...
def bring_mbsf_for_events(events: pd.DataFrame, mbsf_pq: str, year: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["BENE_ID", "BENE_DEATH_DT", "BENE_BIRTH_DT", "SEX_IDENT_CD"])

    benes_dd = dd.from_pandas(pd.DataFrame({"BENE_ID": _as_clean_str(events["BENE_ID"]).unique()}), npartitions=1)
    benes_dd["BENE_ID"] = benes_dd["BENE_ID"].astype(str)

    if year > 2017: # had to read in mbsf conditionally due to how bene_id was indexed in certain years.
        m = dd.read_parquet(mbsf_pq, columns=["BENE_ID", "BENE_DEATH_DT", "BENE_BIRTH_DT", "SEX_IDENT_CD"])
        m["BENE_ID"] = m["BENE_ID"].astype(str)
    else:
        m = dd.read_parquet(mbsf_pq, columns=["BENE_DEATH_DT", "BENE_BIRTH_DT", "SEX_IDENT_CD"])
        if m._meta.index.name == "BENE_ID":
            m = m.reset_index()
        m["BENE_ID"] = m["BENE_ID"].astype(str)

    m = m.merge(benes_dd, on="BENE_ID", how="inner").compute()

    for c in ["BENE_DEATH_DT", "BENE_BIRTH_DT"]:
        m[c] = _parse_mbsf_date_series(m[c])

    m["BENE_ID"] = _as_clean_str(m["BENE_ID"])
    return m[["BENE_ID", "BENE_DEATH_DT", "BENE_BIRTH_DT", "SEX_IDENT_CD"]]

# ... Build long panel (week -2 vs week 0) ...
def build_long_panel(
    year: int,
    events: pd.DataFrame,
    ip_out: pd.DataFrame,
    ed_out: pd.DataFrame,
    dial: pd.DataFrame,
    mbsf: pd.DataFrame
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    df = events.copy()
    df["BENE_ID"] = _as_clean_str(df["BENE_ID"]) # clean

    if not dial.empty:
        dial = dial.copy()
        dial["BENE_ID"] = _as_clean_str(dial["BENE_ID"])
    df = df.merge(dial, on=["event_id", "BENE_ID"], how="left") # Merge the dialysis-derived variables onto the event cohort. The event cohort was created prior: one row per beneficiary-storm exposure event

    for c in [
        "n_dialysis_wk_m7","n_dialysis_wk_m5","n_dialysis_wk_m4","n_dialysis_wk_m3","n_dialysis_wk_m2",
        "n_dialysis_2wk_prior","n_dialysis_2wk_post"
    ]:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype("int16") # make sure numeric and fill na with 0's. This is okay since these are counts of dialysis so NA would = 0

    # Create weekly disruption indicators
    df["disrupt_wk_m7"] = (df["n_dialysis_wk_m7"] < 3).astype("int8") # week -7
    df["disrupt_wk_m5"] = (df["n_dialysis_wk_m5"] < 3).astype("int8") # week of placebo exposure
    df["disrupt_wk_m4"] = (df["n_dialysis_wk_m4"] < 3).astype("int8") # etc...
    df["disrupt_wk_m3"] = (df["n_dialysis_wk_m3"] < 3).astype("int8")
    df["disrupt_wk_m2"] = (df["n_dialysis_wk_m2"] < 3).astype("int8")

    # Cumulative post disruption = any weekly disruption in the post window
    df["disrupt_post_2wk"] = df[["disrupt_wk_m5", "disrupt_wk_m4"]].max(axis=1).astype("int8")
    df["disrupt_post_3wk"] = df[["disrupt_wk_m5", "disrupt_wk_m4", "disrupt_wk_m3"]].max(axis=1).astype("int8")
    df["disrupt_post_4wk"] = df[["disrupt_wk_m5", "disrupt_wk_m4", "disrupt_wk_m3", "disrupt_wk_m2"]].max(axis=1).astype("int8")

    df["disrupt_2wk_prior"] = (df["n_dialysis_2wk_prior"] < 6).astype("int8") # ultimately not used in analysis but left here.
    df["disrupt_2wk_post"]  = (df["n_dialysis_2wk_post"] < 6).astype("int8")

    # Clean and merge IP / ED outcome 
    if not ip_out.empty:
        ip_out = ip_out.copy()
        ip_out["BENE_ID"] = _as_clean_str(ip_out["BENE_ID"])
    if not ed_out.empty:
        ed_out = ed_out.copy()
        ed_out["BENE_ID"] = _as_clean_str(ed_out["BENE_ID"])

    df = df.merge(ip_out, on=["event_id", "BENE_ID"], how="left")
    df = df.merge(ed_out, on=["event_id", "BENE_ID"], how="left")

    outcome_fill_cols = [
        "any_ip_wk_m7","any_ip_wk_m5","any_ip_wk_m4","any_ip_wk_m3","any_ip_wk_m2",
        "any_ip_post_2wk","any_ip_post_3wk","any_ip_post_4wk",
        "any_ip_2wk_prior","any_ip_2wk_post",
        "any_ed_wk_m7","any_ed_wk_m5","any_ed_wk_m4","any_ed_wk_m3","any_ed_wk_m2",
        "any_ed_post_2wk","any_ed_post_3wk","any_ed_post_4wk",
        "any_ed_2wk_prior","any_ed_2wk_post",
    ]
    for c in outcome_fill_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype("int8")

    if not mbsf.empty: # Clean and merge mbsf
        mbsf = mbsf.copy()
        mbsf["BENE_ID"] = _as_clean_str(mbsf["BENE_ID"])
    df = df.merge(mbsf, on="BENE_ID", how="left")

    df["exposure_start_dt"] = pd.to_datetime(df["exposure_start_dt"], errors="coerce").dt.normalize()
    df["placebo_exposure_start_dt"] = pd.to_datetime(df["placebo_exposure_start_dt"], errors="coerce").dt.normalize()
    df["BENE_DEATH_DT"] = pd.to_datetime(df["BENE_DEATH_DT"], errors="coerce").dt.normalize()

    # Drop beneficiaries who died before the placebo exposure start date. This is important because we want to only keep those who has the chance of geting hosp/died in the week of exposure (week -5). Thus, bene who died before has to be dropped
    died_before_placebo = df["BENE_DEATH_DT"].notna() & (df["BENE_DEATH_DT"] < df["placebo_exposure_start_dt"])
    n_before = len(df)
    df = df[~died_before_placebo].copy()
    if n_before - len(df) > 0:
        print(f"Year {year}: dropped {n_before-len(df):,} event rows with death before placebo exposure date.")

    df["event_dow"] = df["placebo_exposure_start_dt"].dt.weekday.astype("int8") # Create the weekday of the storm exposure date: Monday = 0 Tuesday = 1, etc

    # Ensure key dialysis-derived fields exist
    if "schedule_type" not in df.columns:
        df["schedule_type"] = pd.NA
    if "stable_3x_weekly" not in df.columns:
        df["stable_3x_weekly"] = 0
    if "earlyA_last_pre_offschedule" not in df.columns:
        df["earlyA_last_pre_offschedule"] = 0
    if "PRVDR_NUM_event" not in df.columns:
        df["PRVDR_NUM_event"] = pd.NA

    # ... Death flags: cumulative from week 0 onward ...
    death_rel_day = (df["BENE_DEATH_DT"] - df["placebo_exposure_start_dt"]).dt.days

    # Should remain 0 after dropping deaths before placebo exposure date
    df["any_death_wk_m7"] = (
        death_rel_day.notna() &
        (death_rel_day >= REF_WK_M7_START) &
        (death_rel_day <= REF_WK_M7_END)
    ).astype("int8")

    # Cumulative dead-by-week indicators from week 0 onward
    df["any_death_wk_m5"] = (
        death_rel_day.notna() &
        (death_rel_day >= HAZ_WK_M5_START) &
        (death_rel_day <= HAZ_WK_M5_END)
    ).astype("int8")

    df["any_death_wk_m4"] = (
        death_rel_day.notna() &
        (death_rel_day >= HAZ_WK_M5_START) &
        (death_rel_day <= WK_M4_END)
    ).astype("int8")

    df["any_death_wk_m3"] = (
        death_rel_day.notna() &
        (death_rel_day >= HAZ_WK_M5_START) &
        (death_rel_day <= WK_M3_END)
    ).astype("int8")

    df["any_death_wk_m2"] = (
        death_rel_day.notna() &
        (death_rel_day >= HAZ_WK_M5_START) &
        (death_rel_day <= WK_M2_END)
    ).astype("int8")

    # The following are redundant but more of a sanity check
    df["any_death_post_2wk"] = (
        death_rel_day.notna() &
        (death_rel_day >= POST_2WK_START) &
        (death_rel_day <= POST_2WK_END)
    ).astype("int8")

    df["any_death_post_3wk"] = (
        death_rel_day.notna() &
        (death_rel_day >= POST_3WK_START) &
        (death_rel_day <= POST_3WK_END)
    ).astype("int8")

    df["any_death_post_4wk"] = (
        death_rel_day.notna() &
        (death_rel_day >= POST_4WK_START) &
        (death_rel_day <= POST_4WK_END)
    ).astype("int8")

    base_cols = [
        "year","event_id","BENE_ID","storm_id","storm_year","fips",
        "exposure_start_dt","placebo_exposure_start_dt",
        "event_dow","schedule_type","stable_3x_weekly","earlyA_last_pre_offschedule",
        "gap_days","no_hazard_dialysis",
        "PRVDR_NUM_event",
        "BENE_DEATH_DT","BENE_BIRTH_DT","SEX_IDENT_CD",
    ]

    # ... Week -2 row ...
    wk_m7 = df[base_cols].copy()
    wk_m7["week_rel"] = -7  # Mark this row as the placebo reference week 
    wk_m7["hazard_week"] = 0 # Indicate it is not the hazard week (not week of placebo exposure)

    # Populate the outcome columns for the reference week
    wk_m7["any_ip"] = df["any_ip_wk_m7"].values
    wk_m7["any_ed"] = df["any_ed_wk_m7"].values
    wk_m7["any_death"] = df["any_death_wk_m7"].values  # Again, this should be zero since we dropped those who died prior. More of a sanity check.
    wk_m7["n_dialysis"] = df["n_dialysis_wk_m7"].values
    wk_m7["disrupt"] = df["disrupt_wk_m7"].values

    # Intentionally fills all the "_cmp_" variables with the reference-week value. Why? If a model uses, for example, any_ed_cmp_2wk as the dependent variable, then the reference observation contributes to the week -2 ED value. 
    wk_m7["any_ip_cmp_wk"]  = df["any_ip_wk_m7"].values
    wk_m7["any_ip_cmp_2wk"] = df["any_ip_wk_m7"].values
    wk_m7["any_ip_cmp_3wk"] = df["any_ip_wk_m7"].values
    wk_m7["any_ip_cmp_4wk"] = df["any_ip_wk_m7"].values

    wk_m7["any_ed_cmp_wk"]  = df["any_ed_wk_m7"].values
    wk_m7["any_ed_cmp_2wk"] = df["any_ed_wk_m7"].values
    wk_m7["any_ed_cmp_3wk"] = df["any_ed_wk_m7"].values
    wk_m7["any_ed_cmp_4wk"] = df["any_ed_wk_m7"].values

    wk_m7["any_death_cmp_wk"]  = df["any_death_wk_m7"].values
    wk_m7["any_death_cmp_2wk"] = df["any_death_wk_m7"].values
    wk_m7["any_death_cmp_3wk"] = df["any_death_wk_m7"].values
    wk_m7["any_death_cmp_4wk"] = df["any_death_wk_m7"].values

    wk_m7["disrupt_cmp_wk"]  = df["disrupt_wk_m7"].values
    wk_m7["disrupt_cmp_2wk"] = df["disrupt_wk_m7"].values
    wk_m7["disrupt_cmp_3wk"] = df["disrupt_wk_m7"].values
    wk_m7["disrupt_cmp_4wk"] = df["disrupt_wk_m7"].values

    wk_m7["earlyA_last_pre_offschedule"] = 0 # no bene should have gotten early dialysis in reference week

    # These are redundant columns. Initially used as a check but kept for now.
    wk_m7["any_ip_wk_m4"] = df["any_ip_wk_m4"].values
    wk_m7["any_ip_wk_m3"] = df["any_ip_wk_m3"].values
    wk_m7["any_ip_wk_m2"] = df["any_ip_wk_m2"].values
    wk_m7["any_ip_post_2wk"] = df["any_ip_post_2wk"].values
    wk_m7["any_ip_post_3wk"] = df["any_ip_post_3wk"].values
    wk_m7["any_ip_post_4wk"] = df["any_ip_post_4wk"].values

    wk_m7["any_ed_wk_m4"] = df["any_ed_wk_m4"].values
    wk_m7["any_ed_wk_m3"] = df["any_ed_wk_m3"].values
    wk_m7["any_ed_wk_m2"] = df["any_ed_wk_m2"].values
    wk_m7["any_ed_post_2wk"] = df["any_ed_post_2wk"].values
    wk_m7["any_ed_post_3wk"] = df["any_ed_post_3wk"].values
    wk_m7["any_ed_post_4wk"] = df["any_ed_post_4wk"].values

    wk_m7["any_death_wk_m4"] = df["any_death_wk_m4"].values
    wk_m7["any_death_wk_m3"] = df["any_death_wk_m3"].values
    wk_m7["any_death_wk_m2"] = df["any_death_wk_m2"].values
    wk_m7["any_death_post_2wk"] = df["any_death_post_2wk"].values
    wk_m7["any_death_post_3wk"] = df["any_death_post_3wk"].values
    wk_m7["any_death_post_4wk"] = df["any_death_post_4wk"].values

    wk_m7["disrupt_wk_m4"] = df["disrupt_wk_m4"].values
    wk_m7["disrupt_wk_m3"] = df["disrupt_wk_m3"].values
    wk_m7["disrupt_wk_m2"] = df["disrupt_wk_m2"].values
    wk_m7["disrupt_post_2wk"] = df["disrupt_post_2wk"].values
    wk_m7["disrupt_post_3wk"] = df["disrupt_post_3wk"].values
    wk_m7["disrupt_post_4wk"] = df["disrupt_post_4wk"].values

    # --- Week -5 row ---
    # Similar logic but for placebo week of exposure.
    wk_m5 = df[base_cols].copy()
    wk_m5["week_rel"] = -5  # Mark this as week -5 (placebo week of exposure)
    wk_m5["hazard_week"] = 1 # Indicate as placebo week of exposure

    wk_m5["any_ip"] = df["any_ip_wk_m5"].values
    wk_m5["any_ed"] = df["any_ed_wk_m5"].values
    wk_m5["any_death"] = df["any_death_wk_m5"].values
    wk_m5["n_dialysis"] = df["n_dialysis_wk_m5"].values
    wk_m5["disrupt"] = df["disrupt_wk_m5"].values

    # Fill with post cumulative outcomes.
    wk_m5["any_ip_cmp_wk"]  = df["any_ip_wk_m5"].values
    wk_m5["any_ip_cmp_2wk"] = df["any_ip_post_2wk"].values
    wk_m5["any_ip_cmp_3wk"] = df["any_ip_post_3wk"].values
    wk_m5["any_ip_cmp_4wk"] = df["any_ip_post_4wk"].values

    wk_m5["any_ed_cmp_wk"]  = df["any_ed_wk_m5"].values
    wk_m5["any_ed_cmp_2wk"] = df["any_ed_post_2wk"].values
    wk_m5["any_ed_cmp_3wk"] = df["any_ed_post_3wk"].values
    wk_m5["any_ed_cmp_4wk"] = df["any_ed_post_4wk"].values

    wk_m5["any_death_cmp_wk"]  = df["any_death_wk_m5"].values
    wk_m5["any_death_cmp_2wk"] = df["any_death_post_2wk"].values
    wk_m5["any_death_cmp_3wk"] = df["any_death_post_3wk"].values
    wk_m5["any_death_cmp_4wk"] = df["any_death_post_4wk"].values

    wk_m5["disrupt_cmp_wk"]  = df["disrupt_wk_m5"].values
    wk_m5["disrupt_cmp_2wk"] = df["disrupt_post_2wk"].values
    wk_m5["disrupt_cmp_3wk"] = df["disrupt_post_3wk"].values
    wk_m5["disrupt_cmp_4wk"] = df["disrupt_post_4wk"].values

    wk_m5["earlyA_last_pre_offschedule"] = df["earlyA_last_pre_offschedule"].fillna(0).astype("int8").values # Fill if bene received early dialysis. This is not relevant for exh6 but kept for consistency with exh 4's script

    # Again, are redundant columns copied on to week -2 row. Initially used as a check but kept for now.
    wk_m5["any_ip_wk_m4"] = df["any_ip_wk_m4"].values
    wk_m5["any_ip_wk_m3"] = df["any_ip_wk_m3"].values
    wk_m5["any_ip_wk_m2"] = df["any_ip_wk_m2"].values
    wk_m5["any_ip_post_2wk"] = df["any_ip_post_2wk"].values
    wk_m5["any_ip_post_3wk"] = df["any_ip_post_3wk"].values
    wk_m5["any_ip_post_4wk"] = df["any_ip_post_4wk"].values

    wk_m5["any_ed_wk_m4"] = df["any_ed_wk_m4"].values
    wk_m5["any_ed_wk_m3"] = df["any_ed_wk_m3"].values
    wk_m5["any_ed_wk_m2"] = df["any_ed_wk_m2"].values
    wk_m5["any_ed_post_2wk"] = df["any_ed_post_2wk"].values
    wk_m5["any_ed_post_3wk"] = df["any_ed_post_3wk"].values
    wk_m5["any_ed_post_4wk"] = df["any_ed_post_4wk"].values

    wk_m5["any_death_wk_m4"] = df["any_death_wk_m4"].values
    wk_m5["any_death_wk_m3"] = df["any_death_wk_m3"].values
    wk_m5["any_death_wk_m2"] = df["any_death_wk_m2"].values
    wk_m5["any_death_post_2wk"] = df["any_death_post_2wk"].values
    wk_m5["any_death_post_3wk"] = df["any_death_post_3wk"].values
    wk_m5["any_death_post_4wk"] = df["any_death_post_4wk"].values

    wk_m5["disrupt_wk_m4"] = df["disrupt_wk_m4"].values
    wk_m5["disrupt_wk_m3"] = df["disrupt_wk_m3"].values
    wk_m5["disrupt_wk_m2"] = df["disrupt_wk_m2"].values
    wk_m5["disrupt_post_2wk"] = df["disrupt_post_2wk"].values
    wk_m5["disrupt_post_3wk"] = df["disrupt_post_3wk"].values
    wk_m5["disrupt_post_4wk"] = df["disrupt_post_4wk"].values

    long = pd.concat([wk_m7, wk_m5], ignore_index=True) # Concat so each bene-storm event gets two rows

    if (long.loc[long["week_rel"] == -7, "any_death"] == 1).any():
        raise ValueError("Found any_death==1 in placebo week_rel=-7; check placebo death-before-anchor drop logic.")

    if DROP_DUPLICATE_BENE_STORM_WEEK: # final long panel removes duplicate rows for the same beneficiary-storm-week
        long = (
            long.sort_values(["BENE_ID", "storm_id", "week_rel", "fips"])
                .drop_duplicates(subset=["BENE_ID", "storm_id", "storm_year", "week_rel"], keep="first")
                .reset_index(drop=True)
        )

    col_order = [
        "year","storm_year","storm_id","event_id","BENE_ID","fips",
        "week_rel","hazard_week",
        "any_ip","any_ed","any_death","n_dialysis","disrupt",
        "any_ip_cmp_wk","any_ip_cmp_2wk","any_ip_cmp_3wk","any_ip_cmp_4wk",
        "any_ed_cmp_wk","any_ed_cmp_2wk","any_ed_cmp_3wk","any_ed_cmp_4wk",
        "any_death_cmp_wk","any_death_cmp_2wk","any_death_cmp_3wk","any_death_cmp_4wk",
        "disrupt_cmp_wk","disrupt_cmp_2wk","disrupt_cmp_3wk","disrupt_cmp_4wk",
        "earlyA_last_pre_offschedule",
        "gap_days","no_hazard_dialysis",
        "schedule_type","stable_3x_weekly",
        "PRVDR_NUM_event",
        "any_ip_wk_m4","any_ip_wk_m3","any_ip_wk_m2","any_ip_post_2wk","any_ip_post_3wk","any_ip_post_4wk",
        "any_ed_wk_m4","any_ed_wk_m3","any_ed_wk_m2","any_ed_post_2wk","any_ed_post_3wk","any_ed_post_4wk",
        "any_death_wk_m4","any_death_wk_m3","any_death_wk_m2","any_death_post_2wk","any_death_post_3wk","any_death_post_4wk",
        "disrupt_wk_m4","disrupt_wk_m3","disrupt_wk_m2","disrupt_post_2wk","disrupt_post_3wk","disrupt_post_4wk",
        "exposure_start_dt","placebo_exposure_start_dt","event_dow",
        "BENE_DEATH_DT","BENE_BIRTH_DT","SEX_IDENT_CD",
    ]
    long = long[[c for c in col_order if c in long.columns]].sort_values(
        ["event_id","BENE_ID","week_rel"]
    ).reset_index(drop=True)

    return long

# -------------------------
# Main
# -------------------------
# Please see comments in function above for more details on each functions used here

if __name__ == "__main__":
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        print(f"\n=== STEP 4 PLACEBO v02: Processing storm_year={year} ===")

        sdir = step3_summary_dir(year)
        ddir = step3_detail_dir(year)

        if not _exists(sdir) or not _exists(ddir):
            print(f"[SKIP] {year}: missing step3 outputs (summary/detail)")
            continue

        summary = pd.read_parquet(sdir) # bene-storm
        detail  = pd.read_parquet(ddir) # line items

        summary = _normalize_ids(summary)
        detail  = _normalize_ids(detail)

        events = _eventize_from_step3_summary(summary)
        if events.empty:
            print(f"[SKIP] {year}: no events in step3 summary")
            continue
        events["year"] = year
        events["placebo_exposure_start_dt"] = pd.to_datetime(events["exposure_start_dt"], errors="coerce").dt.normalize() - pd.Timedelta(days=PLACEBO_SHIFT_DAYS)

        print(f"[INFO] {year}: events={len(events):,} | unique benes={events['BENE_ID'].nunique():,}")

        dial = dialysis_features_from_step3_detail(events, detail)  # creates some variables like gap days/early dialysis

        ed_pq = ed_path(year)
        ip_pq = medpar_path(year)

        if not _exists(ed_pq):
            print(f"[WARN] {year}: ED path not found: {ed_pq}")
            ed_out = pd.DataFrame(columns=[
                "event_id","BENE_ID",
                "any_ed_wk_m7","any_ed_wk_m5","any_ed_wk_m4","any_ed_wk_m3","any_ed_wk_m2",
                "any_ed_post_2wk","any_ed_post_3wk","any_ed_post_4wk",
                "any_ed_2wk_prior","any_ed_2wk_post",
            ])
        else:
            ed_out = outcome_flags_from_ed_year(events, ed_pq)

        if not _exists(ip_pq):
            print(f"[WARN] {year}: MedPAR path not found: {ip_pq}")
            ip_out = pd.DataFrame(columns=[
                "event_id","BENE_ID",
                "any_ip_wk_m7","any_ip_wk_m5","any_ip_wk_m4","any_ip_wk_m3","any_ip_wk_m2",
                "any_ip_post_2wk","any_ip_post_3wk","any_ip_post_4wk",
                "any_ip_2wk_prior","any_ip_2wk_post",
            ])
        else:
            ip_out = outcome_flags_from_ip_year(events, ip_pq)

        mbsf_pq = mbsf_path(year)
        if not _exists(mbsf_pq):
            print(f"[WARN] {year}: MBSF path not found: {mbsf_pq}")
            mbsf = pd.DataFrame(columns=["BENE_ID","BENE_DEATH_DT","BENE_BIRTH_DT","SEX_IDENT_CD"])
        else:
            mbsf = bring_mbsf_for_events(events, mbsf_pq, year)

        long = build_long_panel(year, events, ip_out, ed_out, dial, mbsf)  # create two rows for each bene - (1) placebo ref row and (2) placebo exposure row
        if long.empty:
            print(f"[SKIP] {year}: produced empty placebo panel")
            continue

        ydir = os.path.join(OUT_DIR, f"year_{year}")
        os.makedirs(ydir, exist_ok=True)
        out_csv = os.path.join(ydir, "analytical_panel.csv")
        long.to_csv(out_csv, index=False)

        n_rows = len(long)
        n_fac = long["PRVDR_NUM_event"].dropna().nunique()
        share_missing_fac = long["PRVDR_NUM_event"].isna().mean()
        print(f"[WROTE] {year}: rows={n_rows:,} | unique PRVDR_NUM_event={n_fac:,} | missing PRVDR_NUM_event={share_missing_fac:.3%}")
        print(f"        -> {out_csv}")

    print(f"\n[DONE] Outputs under: {OUT_DIR}")
