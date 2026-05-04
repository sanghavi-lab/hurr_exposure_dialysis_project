#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: May 1, 2026
# Description: This script looks at each beneficiary in the Sandy-2012 cohort, figures out their usual pre-storm dialysis
# schedule (we focus on MWF bene's here) and their dialysis schedule during Sandy exposure and then classifies them into: 
# (1) regular schedule, (2) regular schedule but transfer, (3) disrupted, (4) early but not disrupted, (5) early disrupted. 
# For example, a beneficiary with a disrupted schedule during Sandy may have a Mon and Fri dialysis but missing Wed. Why? 
# Because this automates the process instead of manually finding and categorizing these beneficiaries based on their 
# dialysis schedule. This is needed for the next script which will select 5 facilities (each with at least 11 beneficiaries 
# and each fitting one of the five classification described).
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import gc
import json
import hashlib
from typing import Dict, List, Tuple

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

client = Client("10.50.87.58:44009")
print(client)


# -------------------------
# Paths and spec
# -------------------------

YEAR = 2012

OPB_PATH = "/gpfs/data/cms-share/data/medicare/2012/otpt/opb/parquet/"
OPREV_PATH = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/2012/"
ED_PATH = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/00c/2012/"
MEDPAR_00B = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/00b/2012/"
COHORT_CSV = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/07a/2012_Sandy-2012.csv"

OUT_DIR = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "derived/sandy_schedule_signature_groups_broad_v01/"
)
os.makedirs(OUT_DIR, exist_ok=True)

SANDY_ANCHORS = pd.to_datetime(["2012-10-29", "2012-10-30"])

# Common Sandy storm week for signature building.
# This week is the one used to summarize each beneficiary's observed dialysis pattern around the storm.
STORM_WEEK_START = pd.Timestamp("2012-10-28")   # Sunday
STORM_WEEK_END   = pd.Timestamp("2012-11-03")   # Saturday

# Pre-storm window used to infer usual facility and usual MWF status.
# In other words, look at the 2 weeks before the storm week to figure out:
#   1) what the beneficiary's usual/home dialysis provider was, and
#   2) whether they looked like a usual M/W/F patient before Sandy.
PRE_START = STORM_WEEK_START - pd.Timedelta(days=14)   # 2012-10-14
PRE_END   = STORM_WEEK_START - pd.Timedelta(days=1)    # 2012-10-27

# Pull window.
# This is the full period read from claims: the pre period plus the Sandy storm week.
PULL_START = PRE_START
PULL_END   = STORM_WEEK_END

# Day labels used throughout the signature-building steps.
DAY_ORDER = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
DAY_MAP = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

# Canonical Monday/Wednesday/Friday pattern.
MWF_DAYS = {"Mon", "Wed", "Fri"}

# Minimum size threshold for a grouped pattern to be highlighted separately.
MIN_GROUP_N = 11

# If True, include inpatient hemodialysis dates when creating daily plotting overlay.
# Classification still uses OP storm-week dialysis schedule (B/Y/.), which matches the
# current conceptual framing. ED/IP are retained as overlays for plotting.
USE_IP_OVERLAY = True


def safe_mode(series: pd.Series):
    # Return the most common non-missing value.
    # If there is a tie, sort values and return the first one
    # so the result is deterministic/reproducible.
    if series.empty:
        return np.nan
    vc = series.astype(str).value_counts(dropna=True)
    if vc.empty:
        return np.nan
    top_n = vc.iloc[0]
    tops = sorted(vc[vc == top_n].index.tolist())
    return tops[0]


def provider_list_compact(series: pd.Series, max_items: int = 20) -> str:
    # Create a compact comma-separated provider list for grouped outputs.
    # If there are many providers, truncate the display string so the table/output
    # stays readable.
    vals = sorted(series.dropna().astype(str).unique().tolist())
    if len(vals) <= max_items:
        return ",".join(vals)
    return ",".join(vals[:max_items]) + f",...(+{len(vals)-max_items} more)"


def hash_group_key(parts: List[str]) -> str:
    # Build a short stable ID for each grouped signature pattern.
    # This makes it easier to reference a group later without relying on long strings.
    raw = "||".join([str(x) for x in parts])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def build_day_signature(day_status: Dict[str, str]) -> str:
    # Convert day-level dialysis statuses into one compact signature string.
    # Example idea: Sun:.|Mon:B|Tue:.|...
    return "|".join([f"{d}:{day_status[d]}" for d in DAY_ORDER])


def build_edip_signature(ed_days: Dict[str, int], ip_days: Dict[str, int]) -> str:
    # Convert day-level ED/IP overlays into one compact signature string.
    return "|".join([f"{d}:ED{ed_days[d]}_IP{ip_days[d]}" for d in DAY_ORDER])


def classify_pattern(
    pre_mwf_flag: int,
    sunday_early: int,
    total_sessions: int,
    day_status: Dict[str, str]
) -> str:
    # First, restrict attention to beneficiaries who looked like pre-storm M/W/F patients.
    # Everyone else is set aside into exclude_non_mwf.
    if pre_mwf_flag != 1:
        return "exclude_non_mwf"

    # present_days = storm-week days where any dialysis was observed,
    # regardless of whether it happened at the usual facility, elsewhere,
    # or at both.
    present_days = {d for d, v in day_status.items() if v in {"B", "Y", "M"}}

    # all_mwf_usual means a very clean "regular" Sandy-week pattern:
    # dialysis only on Mon/Wed/Fri, all at the usual facility,
    # and no dialysis on the other days.
    all_mwf_usual = (
        present_days == MWF_DAYS and
        day_status["Mon"] == "B" and
        day_status["Wed"] == "B" and
        day_status["Fri"] == "B" and
        day_status["Sun"] == "." and
        day_status["Tue"] == "." and
        day_status["Thu"] == "." and
        day_status["Sat"] == "."
    )

    # No Sunday "early" dialysis case.
    if sunday_early == 0:
        if all_mwf_usual:
            return "regular_schedule"
        elif total_sessions >= 3:
            # Still got at least 3 total sessions during storm week,
            # but the pattern was not the clean regular one.
            # This can reflect transfer or rescheduling without clear under-treatment.
            return "not_disrupted_transfer_or_rescheduled"
        else:
            # Fewer than 3 total storm-week sessions and no Sunday early dialysis.
            return "disrupted"
    else:
        # Sunday "early" dialysis case.
        if total_sessions >= 3:
            return "early_not_disrupted"
        else:
            return "early_disrupted"


def extract_group_summary(df: pd.DataFrame, label: str):
    # Convenience printer for a quick look at grouped candidate patterns.
    print(f"\n{label}")
    if df.empty:
        print("  [none]")
        return
    print(df.head(20).to_string(index=False))


def load_broad_sandy_cohort() -> pd.DataFrame:
    """
    Load all beneficiaries from the broader Sandy cohort file.
    We only require BENE_ID here.
    """
    cohort = pd.read_csv(COHORT_CSV, dtype=str)
    if "BENE_ID" not in cohort.columns:
        raise ValueError("COHORT_CSV must contain BENE_ID.")
    cohort = cohort[["BENE_ID"]].drop_duplicates().copy()
    if cohort.empty:
        raise ValueError("No beneficiaries found in COHORT_CSV.")
    return cohort


def build_op_for_broad_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Pull all outpatient dialysis visits for cohort beneficiaries between PRE_START and PULL_END,
    attach provider, and normalize dates.
    """
    # Turn the cohort into a small Dask object so it can be used to filter
    # the large outpatient claims files.
    benes_dd = dd.from_pandas(cohort.copy(), npartitions=1)

    # Read outpatient dialysis revenue-center lines.
    # Then keep only dates inside the pre-period + Sandy storm-week pull window.
    op_rev = dd.read_parquet(
        OPREV_PATH,
        engine="pyarrow",
        columns=["BENE_ID", "CLM_ID", "REV_CNTR_DT"]
    )
    op_rev["REV_CNTR_DT"] = dd.to_datetime(op_rev["REV_CNTR_DT"], errors="coerce")
    op_rev = op_rev[
        (op_rev["REV_CNTR_DT"] >= PULL_START) &
        (op_rev["REV_CNTR_DT"] <= PULL_END)
    ]
    # Keep only claims for beneficiaries in the broad Sandy cohort.
    op_rev = op_rev.merge(benes_dd, on="BENE_ID", how="inner")

    # Read the OP base/header file to attach provider IDs.
    op_bse = dd.read_parquet(
        OPB_PATH,
        engine="pyarrow",
        columns=["CLM_ID", "PRVDR_NUM"]
    )
    op_bse["PRVDR_NUM"] = op_bse["PRVDR_NUM"].astype(str)

    # Merge line-level dates to provider IDs via claim ID.
    op = op_rev.merge(op_bse, on="CLM_ID", how="inner")
    op = op.rename(columns={"PRVDR_NUM": "facility_id", "REV_CNTR_DT": "date"})
    op = op[["BENE_ID", "facility_id", "date"]].drop_duplicates().compute()

    # Normalize types and strip time component so dates are daily.
    op["facility_id"] = op["facility_id"].astype(str)
    op["date"] = pd.to_datetime(op["date"]).dt.normalize()
    return op


def infer_usual_facility_and_prestorm_mwf(op: pd.DataFrame) -> pd.DataFrame:
    """
    For each beneficiary:
      - usual_facility_id = modal provider in 14 days before storm week
      - tie-breaker = last pre-storm provider visited
      - pre_mwf_flag = >=2 Mondays, >=2 Wednesdays, >=2 Fridays, and 0 Sundays
    """
    # Restrict OP visits to the 14-day pre-storm window only.
    pre = op[(op["date"] >= PRE_START) & (op["date"] <= PRE_END)].copy()
    if pre.empty:
        return pd.DataFrame(columns=["BENE_ID", "usual_facility_id", "pre_mwf_flag"])

    # Add day-of-week fields for the pre-storm M/W/F classification.
    pre["dow_num"] = pre["date"].dt.dayofweek
    pre["dow"] = pre["dow_num"].map(DAY_MAP)

    # Counts by beneficiary-provider.
    # This helps identify each beneficiary's modal / most frequently used provider
    # in the 14 days before storm week.
    counts = (
        pre.groupby(["BENE_ID", "facility_id"])
        .size()
        .reset_index(name="n_visits")
    )

    # Modal provider(s).
    # Some beneficiaries can tie across providers, so keep all top-count providers first.
    max_n = counts.groupby("BENE_ID")["n_visits"].transform("max")
    modal = counts[counts["n_visits"] == max_n].copy()

    # Last pre-storm provider tie-breaker.
    # If there is a tie in modal counts, prefer the last provider visited before storm week.
    last_pre = (
        pre.sort_values(["BENE_ID", "date"])
        .groupby("BENE_ID")
        .tail(1)[["BENE_ID", "facility_id"]]
        .rename(columns={"facility_id": "last_pre_facility"})
    )

    # Build one "usual_facility_id" per beneficiary.
    # Sort so the last pre-storm provider wins ties;
    # if still tied, provider_id sort keeps the result deterministic.
    usual = (
        modal.merge(last_pre, on="BENE_ID", how="left")
        .assign(is_last=lambda d: (d["facility_id"].astype(str) == d["last_pre_facility"].astype(str)).astype(int))
        .sort_values(["BENE_ID", "is_last", "facility_id"], ascending=[True, False, True])
        .drop_duplicates("BENE_ID")
        .rename(columns={"facility_id": "usual_facility_id"})
        [["BENE_ID", "usual_facility_id"]]
        .copy()
    )
    usual["usual_facility_id"] = usual["usual_facility_id"].astype(str)

    # Day counts across pre period.
    # Reshape to one row per beneficiary with counts for each weekday.
    day_counts = (
        pre.groupby(["BENE_ID", "dow"])
        .size()
        .rename("n")
        .reset_index()
        .pivot(index="BENE_ID", columns="dow", values="n")
        .fillna(0)
        .reset_index()
    )

    # Make sure all weekday columns exist even if some days never appear in the data.
    for d in DAY_ORDER:
        if d not in day_counts.columns:
            day_counts[d] = 0

    # pre_mwf_flag = beneficiary looked like a usual M/W/F schedule before Sandy:
    # at least 2 Mondays, 2 Wednesdays, 2 Fridays, and no Sundays
    # in the 14-day pre window.
    day_counts["pre_mwf_flag"] = (
        (day_counts["Mon"] >= 2) &
        (day_counts["Wed"] >= 2) &
        (day_counts["Fri"] >= 2) &
        (day_counts["Sun"] == 0)
    ).astype(int)

    out = usual.merge(day_counts[["BENE_ID", "pre_mwf_flag"]], on="BENE_ID", how="left")
    out["pre_mwf_flag"] = out["pre_mwf_flag"].fillna(0).astype(int)
    return out


def build_ed_for_broad_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    # Pull ED daily dates for the broad Sandy cohort over the same pull window.
    # These dates are used as overlays later and not for the core OP schedule classification.
    benes_dd = dd.from_pandas(cohort.copy(), npartitions=1)

    ed = dd.read_parquet(
        ED_PATH,
        engine="pyarrow",
        columns=["BENE_ID", "REV_CNTR_DT"]
    )
    ed["date"] = dd.to_datetime(ed["REV_CNTR_DT"], errors="coerce")
    ed = ed[
        (ed["date"] >= PULL_START) &
        (ed["date"] <= PULL_END)
    ]
    ed = ed.merge(benes_dd, on="BENE_ID", how="inner")
    ed = ed[["BENE_ID", "date"]].drop_duplicates().compute()

    ed["date"] = pd.to_datetime(ed["date"]).dt.normalize()
    return ed


def build_ip_for_broad_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the 00b preprocessed MedPAR output and extracts inpatient hemodialysis dates
    where procedure code == 39.95.
    """
    # Read the preprocessed MedPAR file into pandas.
    mp = dd.read_parquet(MEDPAR_00B, engine="pyarrow").compute()
    if mp.empty:
        return pd.DataFrame(columns=["BENE_ID", "hospital_id", "date"])

    # Match BENE_ID type and keep only the Sandy cohort beneficiaries.
    mp["BENE_ID"] = mp["BENE_ID"].astype(str)
    cohort_benes = set(cohort["BENE_ID"].astype(str).tolist())
    mp = mp[mp["BENE_ID"].isin(cohort_benes)].copy()
    if mp.empty:
        return pd.DataFrame(columns=["BENE_ID", "hospital_id", "date"])

    mp["PRVDR_NUM"] = mp["PRVDR_NUM"].astype(str)

    # Surgical/procedure code columns and their matching performed-date columns.
    # The goal is to scan across the MedPAR procedure slots and pull dates where
    # the inpatient hemodialysis code 39.95 appears.
    cd_cols = [f"SRGCL_PRCDR_{i}_CD" for i in range(1, 26)]
    dt_cols = [f"SRGCL_PRCDR_PRFRM_{i}_DT" for i in range(1, 26)]

    # Convert all procedure date columns to datetimes first.
    for c in dt_cols:
        if c in mp.columns:
            mp[c] = pd.to_datetime(mp[c], errors="coerce")

    def row_ip_dates(row):
        # For one MedPAR stay/row, return all inpatient HD dates recorded
        # under procedure code 39.95.
        out = []
        for i in range(25):
            cd = row.get(cd_cols[i], None)
            if pd.isna(cd):
                continue
            cd_clean = str(cd).replace(".", "").strip().upper()
            if cd_clean == "3995":
                dt = row.get(dt_cols[i], None)
                if pd.notna(dt):
                    out.append(pd.to_datetime(dt).normalize())
        return out

    # Expand row-level procedure dates into one long daily file.
    rows = []
    for _, r in mp.iterrows():
        dates = row_ip_dates(r)
        if dates:
            for dt in dates:
                if PULL_START <= dt <= PULL_END:
                    rows.append((str(r["BENE_ID"]), str(r["PRVDR_NUM"]), dt))

    ip = pd.DataFrame(rows, columns=["BENE_ID", "hospital_id", "date"])
    if ip.empty:
        return pd.DataFrame(columns=["BENE_ID", "hospital_id", "date"])

    ip["date"] = pd.to_datetime(ip["date"]).dt.normalize()
    ip["hospital_id"] = ip["hospital_id"].astype(str)
    ip = ip.drop_duplicates().reset_index(drop=True)
    return ip


def build_bene_storm_signatures(
    op: pd.DataFrame,
    usual_df: pd.DataFrame,
    ed_long: pd.DataFrame,
    ip_long: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Outputs:
      bene_sig: one row per beneficiary with category + signatures
      bene_day: one row per beneficiary x day for plotting later
    """
    # Start from the base file containing one row per beneficiary with:
    # usual facility and pre-storm M/W/F flag.
    base = usual_df.copy()
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Storm-week OP only.
    # This is the actual week used to classify the observed Sandy-week schedule pattern.
    storm_op = op[(op["date"] >= STORM_WEEK_START) & (op["date"] <= STORM_WEEK_END)].copy()
    storm_op["dow_num"] = storm_op["date"].dt.dayofweek
    storm_op["dow"] = storm_op["dow_num"].map(DAY_MAP)

    # Merge usual facility onto each storm-week OP row.
    storm_op = storm_op.merge(base, on="BENE_ID", how="left")
    storm_op["facility_id"] = storm_op["facility_id"].astype(str)
    storm_op["usual_facility_id"] = storm_op["usual_facility_id"].astype(str)

    # Day-level B/Y/M status.
    # For each beneficiary-day:
    #   B = dialysis at the usual facility only
    #   Y = dialysis at a non-usual facility only
    #   M = both usual + non-usual facility on that day
    #   . = no OP dialysis that day
    def day_loc_status(df: pd.DataFrame) -> str:
        facs = set(df["facility_id"].astype(str).tolist())
        usual = str(df["usual_facility_id"].iloc[0])
        has_usual = usual in facs
        has_else = any(f != usual for f in facs)
        if has_usual and has_else:
            return "M"
        elif has_usual:
            return "B"
        elif has_else:
            return "Y"
        else:
            return "."

    if not storm_op.empty:
        # Collapse OP rows to one beneficiary x weekday status.
        day_loc = (
            storm_op.groupby(["BENE_ID", "dow"])
            .apply(day_loc_status, include_groups=False)
            .rename("loc_status")
            .reset_index()
        )
    else:
        day_loc = pd.DataFrame(columns=["BENE_ID", "dow", "loc_status"])

    # Full bene x day grid.
    # This creates a complete 7-day storm-week row set for every beneficiary
    # so missing/no-dialysis days are explicit rather than absent.
    all_days = pd.DataFrame({"dow": DAY_ORDER})
    base2 = base.copy()
    base2["_tmp"] = 1
    all_days["_tmp"] = 1
    bene_day = base2.merge(all_days, on="_tmp", how="outer").drop(columns="_tmp")

    bene_day = bene_day.merge(day_loc, on=["BENE_ID", "dow"], how="left")
    bene_day["loc_status"] = bene_day["loc_status"].fillna(".")

    # ED overlay.
    # Mark whether the beneficiary had any ED date on each storm-week day.
    if not ed_long.empty:
        ed_sw = ed_long[(ed_long["date"] >= STORM_WEEK_START) & (ed_long["date"] <= STORM_WEEK_END)].copy()
        ed_sw["dow"] = ed_sw["date"].dt.dayofweek.map(DAY_MAP)
        ed_day = (
            ed_sw.groupby(["BENE_ID", "dow"])
            .size().rename("ed_any").reset_index()
        )
        ed_day["ed_any"] = 1
    else:
        ed_day = pd.DataFrame(columns=["BENE_ID", "dow", "ed_any"])

    # IP overlay.
    # Same idea as ED, but for inpatient hemodialysis dates.
    if USE_IP_OVERLAY and not ip_long.empty:
        ip_sw = ip_long[(ip_long["date"] >= STORM_WEEK_START) & (ip_long["date"] <= STORM_WEEK_END)].copy()
        ip_sw["dow"] = ip_sw["date"].dt.dayofweek.map(DAY_MAP)
        ip_day = (
            ip_sw.groupby(["BENE_ID", "dow"])
            .size().rename("ip_any").reset_index()
        )
        ip_day["ip_any"] = 1
    else:
        ip_day = pd.DataFrame(columns=["BENE_ID", "dow", "ip_any"])

    # Merge overlays onto the full beneficiary x day grid.
    bene_day = bene_day.merge(ed_day, on=["BENE_ID", "dow"], how="left")
    bene_day = bene_day.merge(ip_day, on=["BENE_ID", "dow"], how="left")
    bene_day["ed_any"] = bene_day["ed_any"].fillna(0).astype(int)
    bene_day["ip_any"] = bene_day["ip_any"].fillna(0).astype(int)

    # Add actual calendar date for plotting later.
    # This makes it easier to export a plotting-ready long file.
    day_to_date = {
        "Sun": pd.Timestamp("2012-10-28"),
        "Mon": pd.Timestamp("2012-10-29"),
        "Tue": pd.Timestamp("2012-10-30"),
        "Wed": pd.Timestamp("2012-10-31"),
        "Thu": pd.Timestamp("2012-11-01"),
        "Fri": pd.Timestamp("2012-11-02"),
        "Sat": pd.Timestamp("2012-11-03"),
    }
    bene_day["date"] = bene_day["dow"].map(day_to_date)

    # Collapse to one row per beneficiary.
    # This is where the daily statuses get summarized into the beneficiary-level
    # category + compact signature strings.
    rows = []
    for bene_id, g in bene_day.groupby("BENE_ID", sort=False):
        g = g.copy().set_index("dow").reindex(DAY_ORDER).reset_index()

        usual_fac = str(g["usual_facility_id"].iloc[0])
        pre_mwf_flag = int(g["pre_mwf_flag"].iloc[0])

        day_status = {d: s for d, s in zip(g["dow"], g["loc_status"])}
        ed_days = {d: int(v) for d, v in zip(g["dow"], g["ed_any"])}
        ip_days = {d: int(v) for d, v in zip(g["dow"], g["ip_any"])}

        # Count total storm-week dialysis days from the OP status pattern.
        total_sessions = int(sum(1 for v in day_status.values() if v in {"B", "Y", "M"}))

        # sunday_early = whether any dialysis was observed on Sunday of storm week.
        sunday_early = int(day_status["Sun"] in {"B", "Y", "M"})

        # Assign the high-level pattern category.
        category = classify_pattern(
            pre_mwf_flag=pre_mwf_flag,
            sunday_early=sunday_early,
            total_sessions=total_sessions,
            day_status=day_status
        )

        # Build compact signature strings that will later be used for grouping.
        dial_sig = build_day_signature(day_status)
        ed_ip_sig = build_edip_signature(ed_days, ip_days)
        nonempty_days = ",".join([d for d in DAY_ORDER if day_status[d] in {"B", "Y", "M"}])
        mwf_pattern_sig = f"Mon:{day_status['Mon']}|Wed:{day_status['Wed']}|Fri:{day_status['Fri']}"

        rows.append({
            "BENE_ID": bene_id,
            "usual_facility_id": usual_fac,
            "pre_mwf_flag": pre_mwf_flag,
            "sunday_early": sunday_early,
            "total_sessions_storm_week": total_sessions,
            "dialysis_signature": dial_sig,
            "ed_ip_signature": ed_ip_sig,
            "nonempty_days": nonempty_days,
            "mwf_pattern_sig": mwf_pattern_sig,
            "category": category,

            # Save the raw 7-day storm-week status columns too
            # so the outputs remain easy to inspect directly.
            "Sun": day_status["Sun"],
            "Mon": day_status["Mon"],
            "Tue": day_status["Tue"],
            "Wed": day_status["Wed"],
            "Thu": day_status["Thu"],
            "Fri": day_status["Fri"],
            "Sat": day_status["Sat"],

            # Save ED overlay flags by day.
            "Sun_ED": ed_days["Sun"], "Mon_ED": ed_days["Mon"], "Tue_ED": ed_days["Tue"],
            "Wed_ED": ed_days["Wed"], "Thu_ED": ed_days["Thu"], "Fri_ED": ed_days["Fri"], "Sat_ED": ed_days["Sat"],

            # Save IP overlay flags by day.
            "Sun_IP": ip_days["Sun"], "Mon_IP": ip_days["Mon"], "Tue_IP": ip_days["Tue"],
            "Wed_IP": ip_days["Wed"], "Thu_IP": ip_days["Thu"], "Fri_IP": ip_days["Fri"], "Sat_IP": ip_days["Sat"],
        })

    bene_sig = pd.DataFrame(rows)

    # Attach category/group key to day-level file later.
    return bene_sig, bene_day


def group_candidate_patterns(bene_sig: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Drop the non-M/W/F beneficiaries first because grouped candidate patterns
    # are only meant to summarize the analytic M/W/F-style population.
    keep = bene_sig[bene_sig["category"] != "exclude_non_mwf"].copy()
    if keep.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Group by category + dialysis signature so beneficiaries with the same
    # observed storm-week OP pattern are summarized together.
    grp = (
        keep.groupby(["category", "dialysis_signature"], dropna=False)
        .agg(
            n_benes=("BENE_ID", "nunique"),
            total_rows=("BENE_ID", "size"),
            total_sessions_mean=("total_sessions_storm_week", "mean"),
            sunday_early_any=("sunday_early", "max"),
            n_usual_providers=("usual_facility_id", "nunique"),
            usual_provider_ids=("usual_facility_id", provider_list_compact),
            sample_nonempty_days=("nonempty_days", safe_mode),
            sample_mwf_pattern=("mwf_pattern_sig", safe_mode),
        )
        .reset_index()
    )

    # Give each grouped pattern a short stable hash ID.
    grp["group_key"] = grp.apply(
        lambda r: hash_group_key([
            r["category"],
            r["dialysis_signature"]
        ]),
        axis=1
    )

    # Sort for readability.
    grp = grp.sort_values(
        ["category", "n_benes", "dialysis_signature"],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    # Separate the subset meeting the minimum size threshold.
    grp_11 = grp[grp["n_benes"] >= MIN_GROUP_N].copy().reset_index(drop=True)
    return grp, grp_11


def build_candidate_members(bene_sig: pd.DataFrame, grp_all: pd.DataFrame) -> pd.DataFrame:
    # Bring the grouped pattern keys back to the beneficiary-level file
    # so each beneficiary can be linked to a grouped candidate pattern.
    if bene_sig.empty or grp_all.empty:
        return pd.DataFrame()

    keys = grp_all[["category", "dialysis_signature", "group_key"]].drop_duplicates()
    mem = bene_sig.merge(
        keys,
        on=["category", "dialysis_signature"],
        how="inner"
    )
    mem = mem.sort_values(["category", "group_key", "usual_facility_id", "BENE_ID"]).reset_index(drop=True)
    return mem


def attach_group_keys_to_day_file(bene_day: pd.DataFrame, bene_sig: pd.DataFrame, grp_all: pd.DataFrame) -> pd.DataFrame:
    # Add the grouped pattern key/category information onto the day-level file
    # so it can be used directly for plotting or inspection by group.
    if bene_day.empty or bene_sig.empty or grp_all.empty:
        return pd.DataFrame()

    tmp = bene_sig.merge(
        grp_all[["category", "dialysis_signature", "group_key"]],
        on=["category", "dialysis_signature"],
        how="left"
    )[
        ["BENE_ID", "usual_facility_id", "pre_mwf_flag", "category", "dialysis_signature", "group_key"]
    ].drop_duplicates()

    out = bene_day.merge(tmp, on=["BENE_ID", "usual_facility_id", "pre_mwf_flag"], how="left")
    out = out.sort_values(["category", "group_key", "BENE_ID", "date"]).reset_index(drop=True)
    return out


# -------------------------
# Main
# -------------------------
def main():
    # -------------------------
    # Read cohort and OP data
    # -------------------------
    print("\nLoading broader Sandy cohort...")
    cohort = load_broad_sandy_cohort()
    cohort["BENE_ID"] = cohort["BENE_ID"].astype(str)
    print(f"Cohort beneficiaries: {len(cohort):,}")

    print("\nPulling OP for broad cohort...")
    op = build_op_for_broad_cohort(cohort)
    op["BENE_ID"] = op["BENE_ID"].astype(str)
    print(f"OP rows: {len(op):,}")
    print(f"Unique OP beneficiaries: {op['BENE_ID'].nunique():,}")

    # -------------------------
    # Infer usual facility + pre-storm schedule type
    # -------------------------
    print("\nInferring usual facility and pre-storm MWF...")
    usual_df = infer_usual_facility_and_prestorm_mwf(op)
    usual_df["BENE_ID"] = usual_df["BENE_ID"].astype(str)
    print(f"Usual-facility rows: {len(usual_df):,}")
    print("\nPre-storm MWF flag counts:")
    print(usual_df["pre_mwf_flag"].value_counts(dropna=False).sort_index())

    # -------------------------
    # Pull ED/IP overlays
    # -------------------------
    print("\nPulling ED overlay...")
    ed_long = build_ed_for_broad_cohort(cohort)
    if not ed_long.empty:
        ed_long["BENE_ID"] = ed_long["BENE_ID"].astype(str)
    print(f"ED rows: {len(ed_long):,}")

    print("\nPulling IP overlay...")
    ip_long = build_ip_for_broad_cohort(cohort)
    if not ip_long.empty:
        ip_long["BENE_ID"] = ip_long["BENE_ID"].astype(str)
    print(f"IP daily rows: {len(ip_long):,}")

    # -------------------------
    # Build beneficiary-level signatures
    # -------------------------
    print("\nBuilding beneficiary storm-week signatures...")
    bene_sig, bene_day = build_bene_storm_signatures(
        op=op,
        usual_df=usual_df,
        ed_long=ed_long,
        ip_long=ip_long
    )

    print(f"Beneficiary signature rows: {len(bene_sig):,}")
    print("\nCategory counts:")
    print(bene_sig["category"].value_counts(dropna=False))

    # -------------------------
    # Group identical patterns
    # -------------------------
    print("\nGrouping candidate patterns...")
    grp_all, grp_11 = group_candidate_patterns(bene_sig)
    mem = build_candidate_members(bene_sig, grp_all)
    bene_day_plot = attach_group_keys_to_day_file(bene_day, bene_sig, grp_all)

    print(f"All grouped patterns: {len(grp_all):,}")
    print(f"Grouped patterns with n >= {MIN_GROUP_N}: {len(grp_11):,}")

    # Quick printed preview of larger grouped patterns.
    extract_group_summary(
        grp_11[[
            "group_key", "category", "dialysis_signature", "n_benes",
            "sample_nonempty_days", "sample_mwf_pattern", "usual_provider_ids"
        ]],
        label=f"Top candidate groups with n >= {MIN_GROUP_N}"
    )

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------
    # Write all core analytic and plotting-ready outputs.
    cohort.to_csv(os.path.join(OUT_DIR, "broad_sandy_cohort.csv"), index=False)
    op.to_csv(os.path.join(OUT_DIR, "broad_cohort_op_rows.csv"), index=False)
    usual_df.to_csv(os.path.join(OUT_DIR, "usual_facility_and_prestorm_mwf.csv"), index=False)
    bene_sig.to_csv(os.path.join(OUT_DIR, "bene_level_signatures.csv"), index=False)
    grp_all.to_csv(os.path.join(OUT_DIR, "grouped_candidate_patterns_all.csv"), index=False)
    grp_11.to_csv(os.path.join(OUT_DIR, f"grouped_candidate_patterns_nge{MIN_GROUP_N}.csv"), index=False)
    mem.to_csv(os.path.join(OUT_DIR, "candidate_group_members.csv"), index=False)
    bene_day_plot.to_csv(os.path.join(OUT_DIR, "bene_day_level_for_plotting.csv"), index=False)

    # Save a small metadata JSON describing the run and the category definitions.
    meta = {
        "year": YEAR,
        "cohort_csv": COHORT_CSV,
        "storm_week_start": str(STORM_WEEK_START.date()),
        "storm_week_end": str(STORM_WEEK_END.date()),
        "pre_start": str(PRE_START.date()),
        "pre_end": str(PRE_END.date()),
        "min_group_n": MIN_GROUP_N,
        "use_ip_overlay": USE_IP_OVERLAY,
        "classification_notes": {
            "regular_schedule": "pre-storm MWF; exact M/W/F at usual facility; no Sunday dialysis",
            "not_disrupted_transfer_or_rescheduled": "pre-storm MWF; no Sunday dialysis; total storm-week sessions >=3; not regular_schedule",
            "disrupted": "pre-storm MWF; no Sunday dialysis; total storm-week sessions <3",
            "early_not_disrupted": "pre-storm MWF; Sunday dialysis present; total storm-week sessions >=3",
            "early_disrupted": "pre-storm MWF; Sunday dialysis present; total storm-week sessions <3"
        }
    }
    with open(os.path.join(OUT_DIR, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Outputs written to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
