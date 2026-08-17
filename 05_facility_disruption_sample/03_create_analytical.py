#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 21, 2026
# Description: This script takes the dialysis line item file and turns it into a two-row analytical file used for 
# modeling: one row for the reference week (week_rel = -2) and one row for the exposure week (week_rel = 0). It merges 
# in dialysis-derived columns (like gap days, early dialysis indicator), ED, inpatient, and MBSF death flags and then 
# reshapes everything into a long beneficiary-storm panel for later within-beneficiary analyses.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
import dask

# -------------------------
# Dask Client
# -------------------------
cust_temp_dir = "/gpfs/data/cms-share/duas/52484/Jessy/temp_space/tmp/"
dask.config.set({"temporary-directory": cust_temp_dir})
dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s"
})

client = Client("10.50.87.98:41255")
print(client)

# -------------------------
# Paths and spec
# -------------------------
STRICT_STABLE_SCHEDULE = True # used to ensure bene must have at least 3 dialysis session during a reference week to be classified into MWF or TTS
DROP_MISSING_EXPOSURE = True # used to drop bene if they could not be linked to exposure start date. Howevever, I checked and was able to link everyone to an exposure date. Thus, this condition is not needed but will be kept here.

# ... Window definitions ...

# Outcomes reference week (week_rel = -2)
REF_WEEK = -2
REF_LO, REF_HI = -14, -8

# Hazard week (week_rel = 0) (i.e. exposure week. Labeled as hazard week since that was the term used when constructing this script)
HAZ_WEEK = 0
HAZ_LO, HAZ_HI = 0, 6

# Post-exposure dialysis window used for the early-dialysis disruption analysis
POST_D1_D7_LO, POST_D1_D7_HI = 1, 7

# Additional post weekly windows
WK1_LO, WK1_HI = 7, 13
WK2_LO, WK2_HI = 14, 20
WK3_LO, WK3_HI = 21, 27

# Cumulative post windows
POST_2WK_LO, POST_2WK_HI = 0, 13   # weeks 0-1
POST_3WK_LO, POST_3WK_HI = 0, 20   # weeks 0-2
POST_4WK_LO, POST_4WK_HI = 0, 27   # weeks 0-3

# EarlyA window: week -1 (latest day in [-7,-1])
EARLY_LO, EARLY_HI = -7, -1

# Canonical schedule classification window: week -3
CLASS_LO, CLASS_HI = -21, -15

# Cohort inclusion window (keep same logic: require dialysis in [-14,+6])
COHORT_LO, COHORT_HI = REF_LO, HAZ_HI

# For OP pulls (need to cover classification + early + outcomes)
OP_PULL_LO = min(CLASS_LO, EARLY_LO, REF_LO)
OP_PULL_HI = POST_4WK_HI

# ... Paths ...

STRESS_PATH = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/derived/"
    "facility_rolling_stress_days/valid_facilities_operational_stress_2011_2022.csv"
)

def op_path(year: int) -> str:
    return f"/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/{year}/"

def opb_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/otpt/opb/parquet/"

def medpar_path(year: int) -> str:
    return f"/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/00b_hospital_SL/{year}/"

def ed_path(year: int) -> str:
    return f"/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/00c/{year}/"

def mbsf_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/mbsf/mbsf_abcd/parquet/"

def out_path(year: int) -> str:
    base = (
        "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/"
        f"dialysis/01_analytical_sample/esrd_crossover_{year}/"
    )
    os.makedirs(base, exist_ok=True)
    return os.path.join(
        base,
        "analytical_simple_case_crossover_anchor_exposure_refwk_m2_early_wkm1_class_wkm3_cumpost_cumdeath_v03.csv"
    )

COUNTY_EXPOSURE_PATH = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "hurricane_county_exposure_start_v02_track64kt_2011_2022/"
    "county_storm_exposure_with_startdate_2011_2022_ms05.csv"
) # This data was created from the hurricane wind-exposed sample pipeline. Used 05 instead of 17 because not all disrupted facilities resided in counties experiencing at least 17 m/s

ZIP_TO_COUNTY_XWALK = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/derived/facility_rolling_stress_days/"
    "derived_hurricane_wind_maps_by_year_2012_2022_dropFeb2021/"
    "zip5_to_county_fips_crosswalk.csv"
)

# ... Specification of storms and earliest landfall dates ...
# This is important. When we created the disrupted sample, we did not assign them a storm. 
# To assign each disrupted facility a storm, we manually created the following dictionary where each date corresponds to a hurricane.
# Each of these dates are the earliest disruption date for each cluster identified and the pair it with a hurricane closest temporally.
# This dictionary will be used in load_stress_table() to map the cluster to the hurricane that caused it's operational stress.
STORM_DATE_TO_NAME = {
    "2012-08-27": "Isaac-2012",
    "2012-10-29": "Sandy-2012",
    "2016-10-06": "Matthew-2016",
    "2017-08-25": "Harvey-2017",
    "2017-09-08": "Irma-2017",
    "2018-09-12": "Florence-2018",
    "2018-10-10": "Michael-2018",
    "2019-09-02": "Dorian-2019",
    "2020-08-24": "Laura-2020",
    "2021-08-28": "Ida-2021",
    "2022-09-27": "Ian-2022",
}

# -------------------------
# Functions
# -------------------------
def load_zip_to_county() -> pd.DataFrame:
    z = pd.read_csv(ZIP_TO_COUNTY_XWALK, dtype={"zip5": str, "fips": str})
    z["zip5"] = z["zip5"].astype(str).str.zfill(5)
    z["fips"] = z["fips"].astype(str).str.zfill(5)
    return z.drop_duplicates(subset=["zip5"]).copy()

def load_county_exposure() -> pd.DataFrame:
    # This data was created from the hurricane wind-exposed sample pipeline. This has the exposure start date for each county-storm
    e = pd.read_csv(COUNTY_EXPOSURE_PATH, dtype={"storm_id": str, "fips": str})
    e["fips"] = e["fips"].astype(str).str.zfill(5)
    e["exposure_start_dt"] = pd.to_datetime(e["exposure_start_dt"], errors="coerce").dt.normalize()
    e = e[["storm_id", "fips", "exposure_start_dt"]].drop_duplicates()
    return e.copy()


# ... Data with disrupted facilities ...
def load_stress_table() -> pd.DataFrame:
    # Data on disrupted facilities
    stress = pd.read_csv(STRESS_PATH, dtype=str)

    stress["earliest_stress_day"] = pd.to_datetime(stress["earliest_stress_day"], errors="coerce").dt.normalize()
    stress["PRVDR_NUM"] = stress["PRVDR_NUM"].astype(str)

    if "year" not in stress.columns:
        stress["year"] = stress["earliest_stress_day"].dt.year
    else:
        stress["year"] = pd.to_numeric(stress["year"], errors="coerce").fillna(
            stress["earliest_stress_day"].dt.year
        ).astype(int)

    if "cluster_id" in stress.columns:
        # Defines each cluster that was affected by a disaster (e.g., hurricane)
        try:
            stress["cluster_id"] = pd.to_numeric(stress["cluster_id"], errors="coerce").astype("Int64")
        except Exception:
            pass

        cluster_dates = ( # find the earliest date for each cluster. Why? Some facilities are disrupted on different days but I would like one date to identify them all and eventually match (temporally) each cluster to the correct hurricane
            stress.dropna(subset=["cluster_id"])
            .groupby(["year", "cluster_id"], as_index=False)["earliest_stress_day"]
            .min()
            .rename(columns={"earliest_stress_day": "cluster_earliest_date"})
        )
        cluster_dates["cluster_label"] = cluster_dates["cluster_earliest_date"].dt.strftime("%Y-%m-%d")

        stress = stress.merge( # every row in the same cluster now gets the same cluster-level earliest date.
            cluster_dates[["year", "cluster_id", "cluster_label"]],
            on=["year", "cluster_id"],
            how="left",
        )
    else:
        stress["cluster_label"] = np.nan

    if "cluster_label" in stress.columns and stress["cluster_label"].notna().any(): # Basically if there is a cluster label (earliest among facilities in cluster, then assign it, if no cluster label then use earliest stress day.
        stress["anchor_date_str"] = stress["cluster_label"].astype(str)
    else:
        stress["anchor_date_str"] = stress["earliest_stress_day"].dt.strftime("%Y-%m-%d")

    stress["storm_id"] = stress["anchor_date_str"].map(STORM_DATE_TO_NAME) # map using dictionary above to the correct hurricane. We also manually checked to ensure each facility was linked to the hurricane that caused it's disruption.
    return stress

# ... Build cohort around EXPOSURE anchor ...
def build_event_cohort_for_year_exposure_anchor(
    stress_year: pd.DataFrame,
    op_pq: str,
    opb_pq: str,
    zip_xw: pd.DataFrame,
    county_exp: pd.DataFrame,
) -> pd.DataFrame:

    # Builds (event_id, facility_id, BENE_ID) cohort anchored on county_exposure_start_dt. event_id is simply bene-storm events
    # Keeps benes with dialysis claims in around anchor_dt. [-14, +6] (wk -2 and wk 0 only) from the disrupted facilities

    cols = ["PRVDR_NUM", "earliest_stress_day", "year", "storm_id"]
    if "cluster_id" in stress_year.columns:
        cols.insert(1, "cluster_id") # if cluster_id exists, insert it into the list of columns.
    if "cluster_label" in stress_year.columns:
        cols.insert(2 if "cluster_id" in cols else 1, "cluster_label") # Ii cluster_label exists, include that too.

    events = stress_year[cols].drop_duplicates().copy()
    events["PRVDR_NUM"] = events["PRVDR_NUM"].astype(str)
    events = events.reset_index(drop=True)
    events["event_id"] = np.arange(len(events), dtype=int) # each unique facility-stress/storm combination in events gets its own event_id

    providers = events["PRVDR_NUM"].unique().tolist()

    op = dd.read_parquet(op_pq, columns=["BENE_ID", "CLM_ID", "REV_CNTR_DT"]) # outpatient dialysis claim-line data
    op = op.assign(REV_CNTR_DT=dd.to_datetime(op["REV_CNTR_DT"], errors="coerce"))

    opb = dd.read_parquet(opb_pq, columns=["CLM_ID", "PRVDR_NUM", "CLM_SRVC_FAC_ZIP_CD"]) # outpatient claim-header data
    opb["PRVDR_NUM"] = opb["PRVDR_NUM"].astype(str)
    opb["facility_zip5"] = (
        opb["CLM_SRVC_FAC_ZIP_CD"]
        .astype(str)
        .str.slice(0, 5)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(5)
    )

    op = op.merge(opb[["CLM_ID", "PRVDR_NUM", "facility_zip5"]], on="CLM_ID", how="left") # merge line with header
    op = op.rename(columns={"PRVDR_NUM": "facility_id"})
    op["facility_id"] = op["facility_id"].astype(str)
    op = op[op["facility_id"].isin(providers)] # removes claims from providers not disrupted (more efficient processing)

    event_cols = ["PRVDR_NUM", "event_id", "storm_id", "earliest_stress_day"]
    if "cluster_id" in events.columns: # same as above but for event columns
        event_cols.insert(1, "cluster_id")
    if "cluster_label" in events.columns:
        event_cols.insert(2 if "cluster_id" in event_cols else 1, "cluster_label")

    events_dd = dd.from_pandas(events[event_cols], npartitions=1).rename(columns={"PRVDR_NUM": "facility_id"}) # the facility_id has to be one of the disrupted facilities that the bene's were present in
    events_dd["facility_id"] = events_dd["facility_id"].astype(str)
    op = op.merge(events_dd, on="facility_id", how="inner") # merges claims to the event table by facility id.

    zip_xw_dd = dd.from_pandas(zip_xw.copy(), npartitions=1) # ZIP-to-county crosswalk
    op["facility_zip5"] = op["facility_zip5"].astype(str).str.zfill(5)
    op = op.merge(zip_xw_dd, left_on="facility_zip5", right_on="zip5", how="left") # merge
    op = op.rename(columns={"fips": "facility_county_fips"}).drop("zip5", axis=1)
    op["facility_county_fips"] = op["facility_county_fips"].astype(str).str.zfill(5)

    exp_dd = dd.from_pandas(county_exp.copy(), npartitions=1) # the county exposure start date (basically the timestamp of when each county centroid was closest to hurricane track (exposure start date))
    op = op.merge(
        exp_dd,
        left_on=["storm_id", "facility_county_fips"],
        right_on=["storm_id", "fips"],
        how="left",
    ).drop("fips", axis=1) # merge on county and storm, near 100% match

    op = op.rename(columns={"exposure_start_dt": "county_exposure_start_dt"})
    op["county_exposure_start_dt"] = dd.to_datetime(op["county_exposure_start_dt"], errors="coerce").dt.normalize()

    op["anchor_dt"] = op["county_exposure_start_dt"] # anchor date is the exposure start date
    if DROP_MISSING_EXPOSURE:
        op = op[op["anchor_dt"].notnull()] # ensure we keep those that matched but the match was near 100% so this is technically not needed but was used to help check.

    op = op.assign(rel_day=(op["REV_CNTR_DT"] - op["anchor_dt"]).dt.days) # computes the relative day of each outpatient claim date compared with the anchor date.
    op_win = op[(op["rel_day"] >= COHORT_LO) & (op["rel_day"] <= COHORT_HI)] # Keeps only rows in the cohort inclusion window (within -14 to 6)

    keep_cols = [
        "event_id", "facility_id", "BENE_ID",
        "storm_id",
        "earliest_stress_day",
        "facility_zip5", "facility_county_fips",
        "county_exposure_start_dt", "anchor_dt",
    ]
    if "cluster_id" in op_win.columns:
        keep_cols.insert(1, "cluster_id")
    if "cluster_label" in op_win.columns:
        keep_cols.insert(2 if "cluster_id" in keep_cols else 1, "cluster_label")

    cohort = op_win[keep_cols].drop_duplicates().compute()

    for c in ["county_exposure_start_dt", "anchor_dt", "earliest_stress_day"]:
        cohort[c] = pd.to_datetime(cohort[c], errors="coerce").dt.normalize() # strips the hour, min, and sec, keep only the date.

    if DROP_MISSING_EXPOSURE: # QC for cohort data
        fac_before = cohort["facility_id"].nunique()
        miss = cohort["anchor_dt"].isna()
        fac_miss = cohort.loc[miss, "facility_id"].nunique()
        cohort = cohort.loc[~miss].copy()
        fac_after = cohort["facility_id"].nunique()

        print(
            "[DROP missing exposure] "
            f"facilities {fac_before:,}->{fac_after:,} (dropped {fac_miss:,})"
        )

    return cohort

# ... Outcome flags (ED/IP) anchored on exposure date ...
def outcome_flags_from_ed_year_anchor(cohort: pd.DataFrame, ed_pq: str) -> pd.DataFrame:
    if cohort.empty:
        return pd.DataFrame(columns=[
            "event_id", "BENE_ID",
            "any_ed_wk_m2", "any_ed_wk0", "any_ed_wk1", "any_ed_wk2", "any_ed_wk3",
            "any_ed_post_2wk", "any_ed_post_3wk", "any_ed_post_4wk",
        ])

    benes_dd = dd.from_pandas(pd.DataFrame({"BENE_ID": cohort["BENE_ID"].unique()}), npartitions=1)

    tmin = cohort["anchor_dt"].min() + pd.Timedelta(days=REF_LO) # two weeks prior to exposure start date
    tmax = cohort["anchor_dt"].max() + pd.Timedelta(days=POST_4WK_HI) # four weeks after

    ed = dd.read_parquet(ed_pq, columns=["BENE_ID", "REV_CNTR_DT"])
    ed = ed.assign(date=dd.to_datetime(ed["REV_CNTR_DT"], errors="coerce"))
    ed = ed[(ed["date"] >= tmin) & (ed["date"] <= tmax)] # filter for efficient processing
    ed = ed.merge(benes_dd, on="BENE_ID", how="inner")

    cohort_dd = dd.from_pandas(cohort[["event_id", "BENE_ID", "anchor_dt"]], npartitions=1)
    ed = ed.merge(cohort_dd, on="BENE_ID", how="inner")
    ed = ed.assign(rel_day=(ed["date"] - ed["anchor_dt"]).dt.days) # each ED-event pairing, compute the number of days between the ED date and the storm exposure date.

    ed = ed.assign( # creates window-specific indicators
        flag_wk_m2=((ed["rel_day"] >= REF_LO) & (ed["rel_day"] <= REF_HI)).astype("int8"),
        flag_wk0=((ed["rel_day"] >= HAZ_LO) & (ed["rel_day"] <= HAZ_HI)).astype("int8"),
        flag_wk1=((ed["rel_day"] >= WK1_LO) & (ed["rel_day"] <= WK1_HI)).astype("int8"),
        flag_wk2=((ed["rel_day"] >= WK2_LO) & (ed["rel_day"] <= WK2_HI)).astype("int8"),
        flag_wk3=((ed["rel_day"] >= WK3_LO) & (ed["rel_day"] <= WK3_HI)).astype("int8"),
        flag_post_2wk=((ed["rel_day"] >= POST_2WK_LO) & (ed["rel_day"] <= POST_2WK_HI)).astype("int8"),
        flag_post_3wk=((ed["rel_day"] >= POST_3WK_LO) & (ed["rel_day"] <= POST_3WK_HI)).astype("int8"),
        flag_post_4wk=((ed["rel_day"] >= POST_4WK_LO) & (ed["rel_day"] <= POST_4WK_HI)).astype("int8"),
    ) # So at this stage, each ED row has a set of 0/1 markers for all windows it belongs to.

    grp = ( # group the ED records by event_id and BENE_ID. This means the goal is one row per beneficiary-storm event. This is important: taking the maximum across ED rows turns the visit-level flags into yes/no event-level indicators: if at least one ED visit in that event-window has flag 1, the grouped result is 1 and if none do, it stays 0.
        ed.groupby(["event_id", "BENE_ID"])[[
            "flag_wk_m2", "flag_wk0", "flag_wk1", "flag_wk2", "flag_wk3",
            "flag_post_2wk", "flag_post_3wk", "flag_post_4wk",
        ]]
        .max()
        .rename(columns={
            "flag_wk_m2": "any_ed_wk_m2",
            "flag_wk0": "any_ed_wk0",
            "flag_wk1": "any_ed_wk1",
            "flag_wk2": "any_ed_wk2",
            "flag_wk3": "any_ed_wk3",
            "flag_post_2wk": "any_ed_post_2wk",
            "flag_post_3wk": "any_ed_post_3wk",
            "flag_post_4wk": "any_ed_post_4wk",
        })
        .reset_index()
        .compute()
    )
    return grp

def outcome_flags_from_ip_year_anchor(cohort: pd.DataFrame, medpar_pq: str) -> pd.DataFrame:
    # This process for IP is VERY similar to ED above. Please see comments above (under function outcome_flags_from_ed_year_anchor()) for more details
    if cohort.empty:
        return pd.DataFrame(columns=[
            "event_id", "BENE_ID",
            "any_ip_wk_m2", "any_ip_wk0", "any_ip_wk1", "any_ip_wk2", "any_ip_wk3",
            "any_ip_post_2wk", "any_ip_post_3wk", "any_ip_post_4wk",
        ])

    benes_dd = dd.from_pandas(pd.DataFrame({"BENE_ID": cohort["BENE_ID"].unique()}), npartitions=1)

    tmin = cohort["anchor_dt"].min() + pd.Timedelta(days=REF_LO)
    tmax = cohort["anchor_dt"].max() + pd.Timedelta(days=POST_4WK_HI)

    ip = dd.read_parquet(medpar_pq, columns=["BENE_ID", "ADMSN_DT"])
    ip = ip.assign(ADMSN_DT=dd.to_datetime(ip["ADMSN_DT"], errors="coerce"))
    ip = ip[(ip["ADMSN_DT"] >= tmin) & (ip["ADMSN_DT"] <= tmax)]
    ip = ip.merge(benes_dd, on="BENE_ID", how="inner")

    cohort_dd = dd.from_pandas(cohort[["event_id", "BENE_ID", "anchor_dt"]], npartitions=1)
    ip = ip.merge(cohort_dd, on="BENE_ID", how="inner")
    ip = ip.assign(rel_day=(ip["ADMSN_DT"] - ip["anchor_dt"]).dt.days)

    ip = ip.assign(
        flag_wk_m2=((ip["rel_day"] >= REF_LO) & (ip["rel_day"] <= REF_HI)).astype("int8"),
        flag_wk0=((ip["rel_day"] >= HAZ_LO) & (ip["rel_day"] <= HAZ_HI)).astype("int8"),
        flag_wk1=((ip["rel_day"] >= WK1_LO) & (ip["rel_day"] <= WK1_HI)).astype("int8"),
        flag_wk2=((ip["rel_day"] >= WK2_LO) & (ip["rel_day"] <= WK2_HI)).astype("int8"),
        flag_wk3=((ip["rel_day"] >= WK3_LO) & (ip["rel_day"] <= WK3_HI)).astype("int8"),
        flag_post_2wk=((ip["rel_day"] >= POST_2WK_LO) & (ip["rel_day"] <= POST_2WK_HI)).astype("int8"),
        flag_post_3wk=((ip["rel_day"] >= POST_3WK_LO) & (ip["rel_day"] <= POST_3WK_HI)).astype("int8"),
        flag_post_4wk=((ip["rel_day"] >= POST_4WK_LO) & (ip["rel_day"] <= POST_4WK_HI)).astype("int8"),
    )

    grp = (
        ip.groupby(["event_id", "BENE_ID"])[[
            "flag_wk_m2", "flag_wk0", "flag_wk1", "flag_wk2", "flag_wk3",
            "flag_post_2wk", "flag_post_3wk", "flag_post_4wk",
        ]]
        .max()
        .rename(columns={
            "flag_wk_m2": "any_ip_wk_m2",
            "flag_wk0": "any_ip_wk0",
            "flag_wk1": "any_ip_wk1",
            "flag_wk2": "any_ip_wk2",
            "flag_wk3": "any_ip_wk3",
            "flag_post_2wk": "any_ip_post_2wk",
            "flag_post_3wk": "any_ip_post_3wk",
            "flag_post_4wk": "any_ip_post_4wk",
        })
        .reset_index()
        .compute()
    )
    return grp

# ... Dialysis outcomes ...
def dialysis_features_from_op_year_anchor(cohort: pd.DataFrame, op_pq: str) -> pd.DataFrame:
    # Computes things like earliest dialysis indicator and gap days

    if cohort.empty:
        return pd.DataFrame(
            columns=[
                "event_id","BENE_ID",
                "n_dialysis_wk_m2","n_dialysis_wk0","n_dialysis_wk1","n_dialysis_wk2","n_dialysis_wk3",
                "n_dialysis_post_d1_d7",
                "gap_days","no_hazard_dialysis",
                "schedule_type","stable_3x_weekly",
                "earlyA_last_pre_offschedule",
            ]
        )
    benes_dd = dd.from_pandas(pd.DataFrame({"BENE_ID": cohort["BENE_ID"].unique()}), npartitions=1)

    tmin = cohort["anchor_dt"].min() + pd.Timedelta(days=OP_PULL_LO) # get the lowest specification from above
    tmax = cohort["anchor_dt"].max() + pd.Timedelta(days=OP_PULL_HI) # get the highest one

    op = dd.read_parquet(op_pq, columns=["BENE_ID", "REV_CNTR_DT"]) # outpatient dialysis claim-line data
    op = op.assign(date=dd.to_datetime(op["REV_CNTR_DT"], errors="coerce"))
    op = op[(op["date"] >= tmin) & (op["date"] <= tmax)] # efficient processing
    op = op.merge(benes_dd, on="BENE_ID", how="inner") # efficient processing

    cohort_dd = dd.from_pandas(cohort[["event_id", "BENE_ID", "anchor_dt"]], npartitions=1)
    op = op.merge(cohort_dd, on="BENE_ID", how="inner")
    op = op.assign(rel_day=(op["date"] - op["anchor_dt"]).dt.days) # each dialysis-event pairing, compute the number of days between the dialysis date of service and the storm exposure date.

    op_day = op[["event_id", "BENE_ID", "anchor_dt", "date", "rel_day"]].drop_duplicates()

    op_day = op_day.assign(
        wk_m2=((op_day["rel_day"] >= REF_LO) & (op_day["rel_day"] <= REF_HI)).astype("int8"),
        wk0=((op_day["rel_day"] >= HAZ_LO) & (op_day["rel_day"] <= HAZ_HI)).astype("int8"),
        wk1=((op_day["rel_day"] >= WK1_LO) & (op_day["rel_day"] <= WK1_HI)).astype("int8"),
        wk2=((op_day["rel_day"] >= WK2_LO) & (op_day["rel_day"] <= WK2_HI)).astype("int8"),
        wk3=((op_day["rel_day"] >= WK3_LO) & (op_day["rel_day"] <= WK3_HI)).astype("int8"),
    
        post_d1_d7=(
            (op_day["rel_day"] >= POST_D1_D7_LO) &
            (op_day["rel_day"] <= POST_D1_D7_HI)
        ).astype("int8"),
    
        wk_m1=((op_day["rel_day"] >= EARLY_LO) & (op_day["rel_day"] <= EARLY_HI)).astype("int8"),
        class_week=((op_day["rel_day"] >= CLASS_LO) & (op_day["rel_day"] <= CLASS_HI)).astype("int8"),
    ) # assign each line item the week it's in relative to exposure date

    # Assign day of the week then create indicator if the dialysis line item is a part of MWF or TTS
    op_day["dow"] = op_day["date"].dt.weekday
    op_day["is_mwf_day"] = ((op_day["dow"] == 0) | (op_day["dow"] == 2) | (op_day["dow"] == 4)).astype("int8")
    op_day["is_tts_day"] = ((op_day["dow"] == 1) | (op_day["dow"] == 3) | (op_day["dow"] == 5)).astype("int8")

    # Assign if date of service is in week -1 or week 0
    op_day["date_pre_m1"] = op_day["date"].where(op_day["wk_m1"] == 1)
    op_day["date_post"] = op_day["date"].where(op_day["wk0"] == 1)

    # Within the week -3 classification window (class_week), this separates observed dialysis dates into dates falling on MWF days, dates falling on TTS days, dates falling on neither pattern
    op_day["class_MWF"] = ((op_day["class_week"] == 1) & (op_day["is_mwf_day"] == 1)).astype("int8")
    op_day["class_TTS"] = ((op_day["class_week"] == 1) & (op_day["is_tts_day"] == 1)).astype("int8")
    op_day["class_other"] = ((op_day["class_week"] == 1) & (op_day["is_mwf_day"] == 0) & (op_day["is_tts_day"] == 0)).astype("int8")

    grp = (
        op_day.groupby(["event_id", "BENE_ID"])
        .agg({
            "wk_m2": "sum",
            "wk0": "sum",
            "wk1": "sum",
            "wk2": "sum",
            "wk3": "sum",
            "post_d1_d7": "sum",
            "date_pre_m1": "max",
            "date_post": "min",
            "class_week": "sum",
            "class_MWF": "sum",
            "class_TTS": "sum",
            "class_other": "sum",
        })
        .reset_index()
        .compute()
    ) # sum of wk_m2, wk0, wk1, wk2, wk3 gives the count of unique dialysis dates in each week and the class-week sums count how many classification-week dialysis days fell into MWF, TTS, or other buckets

    grp = grp.rename(columns={ # rename
        "wk_m2": "n_dialysis_wk_m2",
        "wk0": "n_dialysis_wk0",
        "wk1": "n_dialysis_wk1",
        "wk2": "n_dialysis_wk2",
        "wk3": "n_dialysis_wk3",
        "post_d1_d7": "n_dialysis_post_d1_d7",
        "class_week": "n_class_total",
        "class_MWF": "n_class_MWF",
        "class_TTS": "n_class_TTS",
        "class_other": "n_class_other",
    })

    for c in [
        "n_dialysis_wk_m2","n_dialysis_wk0","n_dialysis_wk1","n_dialysis_wk2","n_dialysis_wk3",
        "n_dialysis_post_d1_d7",
        "n_class_total","n_class_MWF","n_class_TTS","n_class_other"
    ]:
        grp[c] = grp[c].fillna(0).astype("int16")

    # Compute the gap (the gap between the latest dialysis session in week -1 and the earliest dialysis session in week 0)
    grp["gap_days"] = np.nan
    valid_gap = grp["date_pre_m1"].notna() & grp["date_post"].notna()
    grp.loc[valid_gap, "gap_days"] = (grp.loc[valid_gap, "date_post"] - grp.loc[valid_gap, "date_pre_m1"]).dt.days.astype(float)
    grp["no_hazard_dialysis"] = grp["date_post"].isna().astype("int8")

    # Assign the person’s usual schedule using either strict rule or not
    grp["schedule_type"] = pd.NA
    if STRICT_STABLE_SCHEDULE:
        cond_mwf = (grp["n_class_total"] == 3) & (grp["n_class_MWF"] == 3) & (grp["n_class_TTS"] == 0) & (grp["n_class_other"] == 0)
        cond_tts = (grp["n_class_total"] == 3) & (grp["n_class_TTS"] == 3) & (grp["n_class_MWF"] == 0) & (grp["n_class_other"] == 0)
    else:
        cond_mwf = (grp["n_class_total"] >= 2) & (grp["n_class_MWF"] == grp["n_class_total"]) & (grp["n_class_TTS"] == 0) & (grp["n_class_other"] == 0)
        cond_tts = (grp["n_class_total"] >= 2) & (grp["n_class_TTS"] == grp["n_class_total"]) & (grp["n_class_MWF"] == 0) & (grp["n_class_other"] == 0)
    # strict is: MWF only if they had exactly 3 classification-week dialysis dates and all 3 were on MWF days; TTS only if they had exactly 3 classification-week dialysis dates and all 3 were on TTS days

    grp.loc[cond_mwf, "schedule_type"] = "MWF"
    grp.loc[cond_tts, "schedule_type"] = "TTS"
    grp["stable_3x_weekly"] = grp["schedule_type"].notna().astype("int8") # creates a binary indicator for having a classifiable stable schedule

    grp["earlyA_last_pre_offschedule"] = 0 # initializes the earlyA variable to 0
    grp["pre_dow_m1"] = pd.to_datetime(grp["date_pre_m1"], errors="coerce").dt.weekday

    # These masks say: only evaluate early dialysis for people who have a a 3 days a week schedule, have a schedule type of MWF or TTS, actually have a last week -1 dialysis date observed
    mwf_mask = (grp["stable_3x_weekly"] == 1) & (grp["schedule_type"] == "MWF") & grp["date_pre_m1"].notna()
    tts_mask = (grp["stable_3x_weekly"] == 1) & (grp["schedule_type"] == "TTS") & grp["date_pre_m1"].notna()

    grp.loc[mwf_mask & (~grp["pre_dow_m1"].isin([0, 2, 4])), "earlyA_last_pre_offschedule"] = 1 # for bene classified as MWF, flag them as early if their last week -1 dialysis date was not Monday, Wednesday, or Friday
    grp.loc[tts_mask & (~grp["pre_dow_m1"].isin([1, 3, 5])), "earlyA_last_pre_offschedule"] = 1 # for bene classified as TTS, flag them as early if their last week -1 dialysis date was not Tuesday, Thursday, or Saturday

    return grp[
        [
            "event_id","BENE_ID",
            "n_dialysis_wk_m2","n_dialysis_wk0","n_dialysis_wk1","n_dialysis_wk2","n_dialysis_wk3",
            "n_dialysis_post_d1_d7",
            "gap_days","no_hazard_dialysis",
            "schedule_type","stable_3x_weekly",
            "earlyA_last_pre_offschedule",
        ]
    ]

# ... MBSF ...
def bring_mbsf_for_cohort(cohort: pd.DataFrame, mbsf_pq: str, year: int) -> pd.DataFrame:
    if cohort.empty:
        return pd.DataFrame(columns=["BENE_ID", "BENE_DEATH_DT", "BENE_BIRTH_DT", "SEX_IDENT_CD"])

    benes_dd = dd.from_pandas(pd.DataFrame({"BENE_ID": cohort["BENE_ID"].unique()}), npartitions=1)

    if year > 2017: # conditional import due to diff years having bene indexed or not...
        m = dd.read_parquet(mbsf_pq, columns=["BENE_ID", "BENE_DEATH_DT", "BENE_BIRTH_DT", "SEX_IDENT_CD"])
    else:
        m = dd.read_parquet(mbsf_pq, columns=["BENE_DEATH_DT", "BENE_BIRTH_DT", "SEX_IDENT_CD"])
        if m.index.name == "BENE_ID":
            m = m.reset_index()

    m = m.merge(benes_dd, on="BENE_ID", how="inner").compute() # efficent processing
    for c in ["BENE_DEATH_DT", "BENE_BIRTH_DT"]:
        m[c] = pd.to_datetime(m[c], errors="coerce").dt.normalize()

    return m[["BENE_ID", "BENE_DEATH_DT", "BENE_BIRTH_DT", "SEX_IDENT_CD"]]

# ... Long panel (2 rows per event/bene) anchored on exposure date ...
def make_long_panel_exposure_anchor(
    year: int,
    cohort: pd.DataFrame,
    ip_out: pd.DataFrame,
    ed_out: pd.DataFrame,
    dial_feat: pd.DataFrame,
    mbsf: pd.DataFrame,
) -> pd.DataFrame:
    if cohort.empty:
        return pd.DataFrame()

    df = cohort.copy()
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"], errors="coerce").dt.normalize()

    df = df.merge(dial_feat, on=["event_id", "BENE_ID"], how="left") # adds the dialysis-derived variables to each beneficiary-storm pair.

    for col in [
        "n_dialysis_wk_m2",
        "n_dialysis_wk0",
        "n_dialysis_wk1",
        "n_dialysis_wk2",
        "n_dialysis_wk3",
        "n_dialysis_post_d1_d7",
    ]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype("int16")

    # Create weekly and cumulative dialysis disruption indicators
    df["disrupt_wk_m2"] = (df["n_dialysis_wk_m2"] < 3).astype("int8")
    df["disrupt_wk0"] = (df["n_dialysis_wk0"] < 3).astype("int8")
    df["disrupt_wk1"] = (df["n_dialysis_wk1"] < 3).astype("int8")
    df["disrupt_wk2"] = (df["n_dialysis_wk2"] < 3).astype("int8")
    df["disrupt_wk3"] = (df["n_dialysis_wk3"] < 3).astype("int8")
    df["disrupt_post_d1_d7"] = (df["n_dialysis_post_d1_d7"] < 3).astype("int8") # New post-exposure disruption outcome for the early-dialysis analysis

    df["disrupt_post_2wk"] = (
        (df["disrupt_wk0"] == 1) |
        (df["disrupt_wk1"] == 1)
    ).astype("int8")

    df["disrupt_post_3wk"] = (
        (df["disrupt_wk0"] == 1) |
        (df["disrupt_wk1"] == 1) |
        (df["disrupt_wk2"] == 1)
    ).astype("int8")

    df["disrupt_post_4wk"] = (
        (df["disrupt_wk0"] == 1) |
        (df["disrupt_wk1"] == 1) |
        (df["disrupt_wk2"] == 1) |
        (df["disrupt_wk3"] == 1)
    ).astype("int8")

    df = df.merge(ip_out, on=["event_id", "BENE_ID"], how="left")
    df = df.merge(ed_out, on=["event_id", "BENE_ID"], how="left")

    outcome_fill_cols = [
        "any_ip_wk_m2","any_ip_wk0","any_ip_wk1","any_ip_wk2","any_ip_wk3",
        "any_ip_post_2wk","any_ip_post_3wk","any_ip_post_4wk",
        "any_ed_wk_m2","any_ed_wk0","any_ed_wk1","any_ed_wk2","any_ed_wk3",
        "any_ed_post_2wk","any_ed_post_3wk","any_ed_post_4wk",
    ]
    for col in outcome_fill_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype("int8")

    df = df.merge(mbsf, on="BENE_ID", how="left")
    df["BENE_DEATH_DT"] = pd.to_datetime(df["BENE_DEATH_DT"], errors="coerce").dt.normalize()

    # Drop beneficiaries who died before the storm exposure start date. This is important because we want to only keep those who has the chance of geting hosp/died in the week of exposure (week 0). Thus, bene who died before has to be dropped
    died_before_anchor = (df["BENE_DEATH_DT"].notna() & (df["BENE_DEATH_DT"] < df["anchor_dt"])).astype("int8")
    n_before = len(df)
    df = df[died_before_anchor == 0].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Year {year}: dropped {n_dropped:,} (event_id, BENE_ID) rows with death before anchor_dt.")

    df["anchor_dow"] = df["anchor_dt"].dt.weekday.astype("int8") # Compute day-of-week of the anchor
    df["anchor_on_usual_sched_day"] = pd.NA
    df["anchor_on_off_sched_day"] = pd.NA

    mask_stable = (df["stable_3x_weekly"] == 1) & df["schedule_type"].notna() # mask_stable identifies rows where the person has a classifiable stable schedule.
    mask_mwf = mask_stable & (df["schedule_type"] == "MWF") # mask_mwf narrows that to MWF bene.
    mask_tts = mask_stable & (df["schedule_type"] == "TTS") # mask_tts narrows that to TTS bene.

    df.loc[mask_mwf, "anchor_on_usual_sched_day"] = df.loc[mask_mwf, "anchor_dow"].isin([0, 2, 4]).astype("int8") # For stable MWF bene, mark whether the anchor falls on Monday, Wednesday, or Friday.
    df.loc[mask_tts, "anchor_on_usual_sched_day"] = df.loc[mask_tts, "anchor_dow"].isin([1, 3, 5]).astype("int8") # For stable TTS bene, mark whether the anchor falls on Tuesday, Thursday, or Saturday.
    df.loc[mask_stable, "anchor_on_off_sched_day"] = (1 - df.loc[mask_stable, "anchor_on_usual_sched_day"].astype("int8")).astype("int8") # For stable bene, define anchor_on_off_sched_day as the opposite of anchor_on_usual_sched_day.

    # ^ the "anchor_on_usual_sched_day" was important at one point but we no longer need it. We used to use it to see if the disruption date fell on MWF or TTS. We thought that if the disruption fell on MWF, for example, then it would affect MWF more than TTS.

    # Death indicators: cumulative from week 0 onward
    death_rel_day = (df["BENE_DEATH_DT"] - df["anchor_dt"]).dt.days

    df["any_death_wk_m2"] = (
        death_rel_day.notna() &
        (death_rel_day >= REF_LO) &
        (death_rel_day <= REF_HI)
    ).astype("int8")

    df["any_death_wk0"] = (
        death_rel_day.notna() &
        (death_rel_day >= HAZ_LO) &
        (death_rel_day <= HAZ_HI)
    ).astype("int8")

    df["any_death_wk1"] = (
        death_rel_day.notna() &
        (death_rel_day >= HAZ_LO) &
        (death_rel_day <= WK1_HI)
    ).astype("int8")

    df["any_death_wk2"] = (
        death_rel_day.notna() &
        (death_rel_day >= HAZ_LO) &
        (death_rel_day <= WK2_HI)
    ).astype("int8")

    df["any_death_wk3"] = (
        death_rel_day.notna() &
        (death_rel_day >= HAZ_LO) &
        (death_rel_day <= WK3_HI)
    ).astype("int8")

    # The following are redundant but more of a sanity check to make sure I am creating the post death outcomes correctly
    df["any_death_post_2wk"] = (
        death_rel_day.notna() &
        (death_rel_day >= POST_2WK_LO) &
        (death_rel_day <= POST_2WK_HI)
    ).astype("int8")

    df["any_death_post_3wk"] = (
        death_rel_day.notna() &
        (death_rel_day >= POST_3WK_LO) &
        (death_rel_day <= POST_3WK_HI)
    ).astype("int8")

    df["any_death_post_4wk"] = (
        death_rel_day.notna() &
        (death_rel_day >= POST_4WK_LO) &
        (death_rel_day <= POST_4WK_HI)
    ).astype("int8")

    has_cluster_id = "cluster_id" in df.columns # Just checks whether clustering columns exist
    has_cluster_label = "cluster_label" in df.columns

    base_cols = [ # First, define the columns that both week rows will share
        "event_id",
        "BENE_ID",
        "facility_id",
        "facility_county_fips",
        "storm_id",
        "earliest_stress_day",
        "county_exposure_start_dt",
        "anchor_dt",
        "anchor_dow",
        "anchor_on_usual_sched_day",
        "anchor_on_off_sched_day",
        "schedule_type",
        "stable_3x_weekly",
        "earlyA_last_pre_offschedule", # early dialysis indicator
        "gap_days",
        "no_hazard_dialysis",
        "BENE_DEATH_DT",
        "BENE_BIRTH_DT",
        "SEX_IDENT_CD",
    ]
    if has_cluster_id:
        base_cols.insert(1, "cluster_id")
    if has_cluster_label:
        base_cols.insert(2 if has_cluster_id else 1, "cluster_label")

    # --- Week -2 row ---
    wk_m2 = df[base_cols].copy()
    wk_m2["week_rel"] = REF_WEEK # Mark this row as the reference week 
    wk_m2["hazard_week"] = 0 # Indicate it is not the hazard week (not week of exposure)

    # Populate the outcome columns for the reference week
    wk_m2["any_ip"] = df["any_ip_wk_m2"].values
    wk_m2["any_ed"] = df["any_ed_wk_m2"].values
    wk_m2["any_death"] = df["any_death_wk_m2"].values
    wk_m2["n_dialysis"] = df["n_dialysis_wk_m2"].values
    wk_m2["disrupt"] = df["disrupt_wk_m2"].values

    # Intentionally fills all the "_cmp_" variables with the reference-week value. Why? If a model uses, for example, any_ed_cmp_2wk as the dependent variable, then the reference observation contributes to the week -2 ED value. 
    wk_m2["any_ip_cmp_wk"] = df["any_ip_wk_m2"].values
    wk_m2["any_ip_cmp_2wk"] = df["any_ip_wk_m2"].values
    wk_m2["any_ip_cmp_3wk"] = df["any_ip_wk_m2"].values
    wk_m2["any_ip_cmp_4wk"] = df["any_ip_wk_m2"].values

    wk_m2["any_ed_cmp_wk"] = df["any_ed_wk_m2"].values
    wk_m2["any_ed_cmp_2wk"] = df["any_ed_wk_m2"].values
    wk_m2["any_ed_cmp_3wk"] = df["any_ed_wk_m2"].values
    wk_m2["any_ed_cmp_4wk"] = df["any_ed_wk_m2"].values

    wk_m2["any_death_cmp_wk"] = df["any_death_wk_m2"].values
    wk_m2["any_death_cmp_2wk"] = df["any_death_wk_m2"].values
    wk_m2["any_death_cmp_3wk"] = df["any_death_wk_m2"].values
    wk_m2["any_death_cmp_4wk"] = df["any_death_wk_m2"].values

    wk_m2["disrupt_cmp_wk"] = df["disrupt_wk_m2"].values
    wk_m2["disrupt_cmp_2wk"] = df["disrupt_wk_m2"].values
    wk_m2["disrupt_cmp_3wk"] = df["disrupt_wk_m2"].values
    wk_m2["disrupt_cmp_4wk"] = df["disrupt_wk_m2"].values
    wk_m2["disrupt_cmp_post_d1_d7"] = df["disrupt_wk_m2"].values
    wk_m2["n_dialysis_cmp_post_d1_d7"] = df["n_dialysis_wk_m2"].values

    wk_m2["earlyA_last_pre_offschedule"] = 0 # no bene should have gotten early dialysis in reference week

    # These are redundant columns copied on to week -2 row. Initially used as a check but kept for now.
    wk_m2["any_ip_wk1"] = df["any_ip_wk1"].values
    wk_m2["any_ip_wk2"] = df["any_ip_wk2"].values
    wk_m2["any_ip_wk3"] = df["any_ip_wk3"].values
    wk_m2["any_ip_post_2wk"] = df["any_ip_post_2wk"].values
    wk_m2["any_ip_post_3wk"] = df["any_ip_post_3wk"].values
    wk_m2["any_ip_post_4wk"] = df["any_ip_post_4wk"].values

    wk_m2["any_ed_wk1"] = df["any_ed_wk1"].values
    wk_m2["any_ed_wk2"] = df["any_ed_wk2"].values
    wk_m2["any_ed_wk3"] = df["any_ed_wk3"].values
    wk_m2["any_ed_post_2wk"] = df["any_ed_post_2wk"].values
    wk_m2["any_ed_post_3wk"] = df["any_ed_post_3wk"].values
    wk_m2["any_ed_post_4wk"] = df["any_ed_post_4wk"].values

    wk_m2["any_death_wk1"] = df["any_death_wk1"].values
    wk_m2["any_death_wk2"] = df["any_death_wk2"].values
    wk_m2["any_death_wk3"] = df["any_death_wk3"].values
    wk_m2["any_death_post_2wk"] = df["any_death_post_2wk"].values
    wk_m2["any_death_post_3wk"] = df["any_death_post_3wk"].values
    wk_m2["any_death_post_4wk"] = df["any_death_post_4wk"].values

    wk_m2["disrupt_wk1"] = df["disrupt_wk1"].values
    wk_m2["disrupt_wk2"] = df["disrupt_wk2"].values
    wk_m2["disrupt_wk3"] = df["disrupt_wk3"].values
    wk_m2["disrupt_post_2wk"] = df["disrupt_post_2wk"].values
    wk_m2["disrupt_post_3wk"] = df["disrupt_post_3wk"].values
    wk_m2["disrupt_post_4wk"] = df["disrupt_post_4wk"].values

    # --- Week 0 row ---
    # Similar logic but for week of exposure.
    
    wk0 = df[base_cols].copy()
    wk0["week_rel"] = HAZ_WEEK # Mark this as week 0 (week of exposure)
    wk0["hazard_week"] = 1 # Indicate as week of exposure

    wk0["any_ip"] = df["any_ip_wk0"].values
    wk0["any_ed"] = df["any_ed_wk0"].values
    wk0["any_death"] = df["any_death_wk0"].values
    wk0["n_dialysis"] = df["n_dialysis_wk0"].values
    wk0["disrupt"] = df["disrupt_wk0"].values

    # Fill with actual post cumulative outcomes.
    wk0["any_ip_cmp_wk"] = df["any_ip_wk0"].values
    wk0["any_ip_cmp_2wk"] = df["any_ip_post_2wk"].values
    wk0["any_ip_cmp_3wk"] = df["any_ip_post_3wk"].values
    wk0["any_ip_cmp_4wk"] = df["any_ip_post_4wk"].values

    wk0["any_ed_cmp_wk"] = df["any_ed_wk0"].values
    wk0["any_ed_cmp_2wk"] = df["any_ed_post_2wk"].values
    wk0["any_ed_cmp_3wk"] = df["any_ed_post_3wk"].values
    wk0["any_ed_cmp_4wk"] = df["any_ed_post_4wk"].values

    wk0["any_death_cmp_wk"] = df["any_death_wk0"].values
    wk0["any_death_cmp_2wk"] = df["any_death_post_2wk"].values
    wk0["any_death_cmp_3wk"] = df["any_death_post_3wk"].values
    wk0["any_death_cmp_4wk"] = df["any_death_post_4wk"].values

    wk0["disrupt_cmp_wk"] = df["disrupt_wk0"].values
    wk0["disrupt_cmp_2wk"] = df["disrupt_post_2wk"].values
    wk0["disrupt_cmp_3wk"] = df["disrupt_post_3wk"].values
    wk0["disrupt_cmp_4wk"] = df["disrupt_post_4wk"].values
    wk0["disrupt_cmp_post_d1_d7"] = df["disrupt_post_d1_d7"].values
    wk0["n_dialysis_cmp_post_d1_d7"] = df["n_dialysis_post_d1_d7"].values

    # Again, are redundant columns copied on to week -2 row. Initially used as a check but kept for now.
    wk0["any_ip_wk1"] = df["any_ip_wk1"].values
    wk0["any_ip_wk2"] = df["any_ip_wk2"].values
    wk0["any_ip_wk3"] = df["any_ip_wk3"].values
    wk0["any_ip_post_2wk"] = df["any_ip_post_2wk"].values
    wk0["any_ip_post_3wk"] = df["any_ip_post_3wk"].values
    wk0["any_ip_post_4wk"] = df["any_ip_post_4wk"].values

    wk0["any_ed_wk1"] = df["any_ed_wk1"].values
    wk0["any_ed_wk2"] = df["any_ed_wk2"].values
    wk0["any_ed_wk3"] = df["any_ed_wk3"].values
    wk0["any_ed_post_2wk"] = df["any_ed_post_2wk"].values
    wk0["any_ed_post_3wk"] = df["any_ed_post_3wk"].values
    wk0["any_ed_post_4wk"] = df["any_ed_post_4wk"].values

    wk0["any_death_wk1"] = df["any_death_wk1"].values
    wk0["any_death_wk2"] = df["any_death_wk2"].values
    wk0["any_death_wk3"] = df["any_death_wk3"].values
    wk0["any_death_post_2wk"] = df["any_death_post_2wk"].values
    wk0["any_death_post_3wk"] = df["any_death_post_3wk"].values
    wk0["any_death_post_4wk"] = df["any_death_post_4wk"].values

    wk0["disrupt_wk1"] = df["disrupt_wk1"].values
    wk0["disrupt_wk2"] = df["disrupt_wk2"].values
    wk0["disrupt_wk3"] = df["disrupt_wk3"].values
    wk0["disrupt_post_2wk"] = df["disrupt_post_2wk"].values
    wk0["disrupt_post_3wk"] = df["disrupt_post_3wk"].values
    wk0["disrupt_post_4wk"] = df["disrupt_post_4wk"].values

    long = pd.concat([wk_m2, wk0], ignore_index=True)
    long["year"] = year

    if (long.loc[long["week_rel"] == -2, "any_death"] == 1).any():
        raise ValueError("Found any_death==1 in week_rel=-2; check death-before-anchor drop logic.")

    col_order = [
        "year","event_id","BENE_ID","week_rel","hazard_week",
        "any_ip","any_ed","any_death","n_dialysis","disrupt",
        "any_ip_cmp_wk","any_ip_cmp_2wk","any_ip_cmp_3wk","any_ip_cmp_4wk",
        "any_ed_cmp_wk","any_ed_cmp_2wk","any_ed_cmp_3wk","any_ed_cmp_4wk",
        "any_death_cmp_wk","any_death_cmp_2wk","any_death_cmp_3wk","any_death_cmp_4wk",
        "disrupt_cmp_wk","disrupt_cmp_2wk","disrupt_cmp_3wk","disrupt_cmp_4wk",
        "disrupt_cmp_post_d1_d7",
        "n_dialysis_cmp_post_d1_d7",
        "gap_days","no_hazard_dialysis",
        "facility_id",
        "facility_county_fips",
        "storm_id",
        "earliest_stress_day",
        "county_exposure_start_dt",
        "anchor_dt",
        "anchor_dow","anchor_on_usual_sched_day","anchor_on_off_sched_day",
        "schedule_type","stable_3x_weekly",
        "earlyA_last_pre_offschedule",
        "any_ip_wk1","any_ip_wk2","any_ip_wk3","any_ip_post_2wk","any_ip_post_3wk","any_ip_post_4wk",
        "any_ed_wk1","any_ed_wk2","any_ed_wk3","any_ed_post_2wk","any_ed_post_3wk","any_ed_post_4wk",
        "any_death_wk1","any_death_wk2","any_death_wk3","any_death_post_2wk","any_death_post_3wk","any_death_post_4wk",
        "disrupt_wk1","disrupt_wk2","disrupt_wk3","disrupt_post_2wk","disrupt_post_3wk","disrupt_post_4wk",
        "BENE_DEATH_DT","BENE_BIRTH_DT","SEX_IDENT_CD",
    ]
    if has_cluster_id:
        col_order.insert(2, "cluster_id")
    if has_cluster_label:
        col_order.insert(3 if has_cluster_id else 2, "cluster_label")

    long = long[col_order].sort_values(["event_id", "BENE_ID", "week_rel"]).reset_index(drop=True)
    return long

# -------------------------
# Main build loop
# -------------------------
# Please see comments in function above for more details on each functions used here

if __name__ == "__main__":
    print("[LOAD] ZIP->county crosswalk...")
    zip_xw = load_zip_to_county()
    print(f"  rows={len(zip_xw):,}")

    print("[LOAD] County exposure start dates...")
    county_exp = load_county_exposure()
    print(f"  rows={len(county_exp):,} | storms={county_exp['storm_id'].nunique():,} | counties={county_exp['fips'].nunique():,}")

    stress = load_stress_table() # disrupted facilities data

    # Drop this cluster. Feb 2021 was due to Texas 2021 outage in feb. We only want clusters from hurricane impacts
    stress = stress[~((stress["earliest_stress_day"].dt.year == 2021) & (stress["earliest_stress_day"].dt.month == 2))].copy()

    years = sorted(stress["year"].dropna().unique().tolist())

    for year in years:
        stress_year = stress[stress["year"] == year].copy()
        if stress_year.empty:
            continue

        print(f"\n=== Processing year {year} ({len(stress_year)} facility-stress rows) ===")

        op_pq = op_path(year)
        opb_pq = opb_path(year)
        medpar_pq = medpar_path(year)
        ed_pq = ed_path(year)
        mbsf_pq = mbsf_path(year)
        out_csv = out_path(year)

        cohort = build_event_cohort_for_year_exposure_anchor(stress_year, op_pq, opb_pq, zip_xw, county_exp)
        if cohort.empty:
            print(f"Year {year}: no exposure-anchored cohort; skipping.")
            continue

        print(f"Year {year}: events={cohort['event_id'].nunique():,} | benes={cohort['BENE_ID'].nunique():,}")

        ip_out = outcome_flags_from_ip_year_anchor(cohort, medpar_pq)
        ed_out = outcome_flags_from_ed_year_anchor(cohort, ed_pq)
        dial_feat = dialysis_features_from_op_year_anchor(cohort, op_pq)
        mbsf = bring_mbsf_for_cohort(cohort, mbsf_pq, year)

        long = make_long_panel_exposure_anchor(year, cohort, ip_out, ed_out, dial_feat, mbsf)

        if "cluster_id" in long.columns:
            long = long.drop_duplicates(subset=["BENE_ID", "year", "cluster_id", "week_rel", "anchor_dt"])
        else:
            long = long.drop_duplicates(subset=["BENE_ID", "year", "week_rel", "anchor_dt"])

        long.to_csv(out_csv, index=False)
        print(f"Year {year}: wrote {len(long):,} rows to {out_csv}")

