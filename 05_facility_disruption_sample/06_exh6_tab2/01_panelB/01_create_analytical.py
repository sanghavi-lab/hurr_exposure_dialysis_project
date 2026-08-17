#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: July 22, 2026
# Description: This script takes the dialysis line items and shifts the exposure date 35 days earlier to create a placebo 
# version of the analysis. It then rebuilds the analytical file around that placebo anchor date, merges in ED, inpatient, 
# and MBSF death information, and outputs a two-row long placebo analytical panel per beneficiary-storm event: a placebo 
# reference week row (week_rel = -7) and a placebo hazard week row (week_rel = -5) for later within-beneficiary analysis.
#----------------------------------------------------------------------------------------------------------------------#


import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client

# =========================
# Dask Client
# =========================
cust_temp_dir = "/gpfs/data/cms-share/duas/52484/Jessy/temp_space/tmp/"
dask.config.set({"temporary-directory": cust_temp_dir})
dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s"
})
client = Client("10.50.87.31:42109")
print(client)

# =========================
# Config toggles
# =========================
STRICT_STABLE_SCHEDULE = True
DROP_MISSING_EXPOSURE = False  # recommended

# =========================
# Placebo anchor shift
# =========================
PLACEBO_SHIFT_DAYS = 35  # 5 weeks earlier than real county exposure start date

# =========================
# Week label config for final long panel
# =========================
REF_WEEK_LABEL = -7
HAZ_WEEK_LABEL = -5

# =========================
# Window config (relative to placebo_anchor_dt)
# =========================
REF_LO, REF_HI = -14, -8
HAZ_LO, HAZ_HI = 0, 6

WKM4_LO, WKM4_HI = 7, 13
WKM3_LO, WKM3_HI = 14, 20
WKM2_LO, WKM2_HI = 21, 27

POST_2WK_LO, POST_2WK_HI = 0, 13
POST_3WK_LO, POST_3WK_HI = 0, 20
POST_4WK_LO, POST_4WK_HI = 0, 27

EARLY_LO, EARLY_HI = -7, -1
CLASS_LO, CLASS_HI = -21, -15

COHORT_LO, COHORT_HI = REF_LO, HAZ_HI
OP_PULL_LO = min(CLASS_LO, EARLY_LO, REF_LO)
OP_PULL_HI = POST_4WK_HI

# =========================
# Paths
# =========================
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
        "analytical_simple_case_crossover_anchor_exposure_placebo5wk_refwk_m7_hazwk_m5_early_wkm6_class_wkm8_cumpost_cumdeath_cumdisrupt_v02.csv"
    )

# =========================
# Exposure date inputs
# =========================
COUNTY_EXPOSURE_PATH = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "hurricane_county_exposure_start_v02_track64kt_2011_2022/"
    "county_storm_exposure_with_startdate_2011_2022_ms05.csv"
)

ZIP_TO_COUNTY_XWALK = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/derived/facility_rolling_stress_days/"
    "derived_hurricane_wind_maps_by_year_2012_2022_dropFeb2021/"
    "zip5_to_county_fips_crosswalk.csv"
)

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

# =========================
# Helpers (exposure resources)
# =========================
def load_zip_to_county() -> pd.DataFrame:
    z = pd.read_csv(ZIP_TO_COUNTY_XWALK, dtype={"zip5": str, "fips": str})
    z["zip5"] = z["zip5"].astype(str).str.zfill(5)
    z["fips"] = z["fips"].astype(str).str.zfill(5)
    return z.drop_duplicates(subset=["zip5"]).copy()

def load_county_exposure() -> pd.DataFrame:
    e = pd.read_csv(COUNTY_EXPOSURE_PATH, dtype={"storm_id": str, "fips": str})
    e["fips"] = e["fips"].astype(str).str.zfill(5)
    e["exposure_start_dt"] = pd.to_datetime(e["exposure_start_dt"], errors="coerce").dt.normalize()
    e = e[["storm_id", "fips", "exposure_start_dt"]].drop_duplicates()
    return e.copy()

# =========================
# Stress table -> event definitions (storm_id via cluster_label)
# =========================
def load_stress_table() -> pd.DataFrame:
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
        try:
            stress["cluster_id"] = pd.to_numeric(stress["cluster_id"], errors="coerce").astype("Int64")
        except Exception:
            pass

        cluster_dates = (
            stress.dropna(subset=["cluster_id"])
            .groupby(["year", "cluster_id"], as_index=False)["earliest_stress_day"]
            .min()
            .rename(columns={"earliest_stress_day": "cluster_earliest_date"})
        )
        cluster_dates["cluster_label"] = cluster_dates["cluster_earliest_date"].dt.strftime("%Y-%m-%d")

        stress = stress.merge(
            cluster_dates[["year", "cluster_id", "cluster_label"]],
            on=["year", "cluster_id"],
            how="left",
        )
    else:
        stress["cluster_label"] = np.nan

    if "cluster_label" in stress.columns and stress["cluster_label"].notna().any():
        stress["anchor_date_str"] = stress["cluster_label"].astype(str)
    else:
        stress["anchor_date_str"] = stress["earliest_stress_day"].dt.strftime("%Y-%m-%d")

    stress["storm_id"] = stress["anchor_date_str"].map(STORM_DATE_TO_NAME)
    return stress

# =========================
# Build cohort around PLACEBO exposure anchor
# =========================
def build_event_cohort_for_year_placebo_anchor(
    stress_year: pd.DataFrame,
    op_pq: str,
    opb_pq: str,
    zip_xw: pd.DataFrame,
    county_exp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds (event_id, facility_id, BENE_ID) cohort anchored on placebo_anchor_dt,
    where placebo_anchor_dt = county_exposure_start_dt - 35 days.

    Keeps benes with dialysis claims in [COHORT_LO, COHORT_HI] relative to placebo_anchor_dt.
    Here: [-14, +6], which corresponds to real-anchor-relative [-49, -29].
    """
    cols = ["PRVDR_NUM", "earliest_stress_day", "year", "storm_id"]
    if "cluster_id" in stress_year.columns:
        cols.insert(1, "cluster_id")
    if "cluster_label" in stress_year.columns:
        cols.insert(2 if "cluster_id" in cols else 1, "cluster_label")

    events = stress_year[cols].drop_duplicates().copy()
    events["PRVDR_NUM"] = events["PRVDR_NUM"].astype(str)
    events = events.reset_index(drop=True)
    events["event_id"] = np.arange(len(events), dtype=int)

    providers = events["PRVDR_NUM"].unique().tolist()

    op = dd.read_parquet(op_pq, columns=["BENE_ID", "CLM_ID", "REV_CNTR_DT"])
    op = op.assign(REV_CNTR_DT=dd.to_datetime(op["REV_CNTR_DT"], errors="coerce"))

    opb = dd.read_parquet(opb_pq, columns=["CLM_ID", "PRVDR_NUM", "CLM_SRVC_FAC_ZIP_CD"])
    opb["PRVDR_NUM"] = opb["PRVDR_NUM"].astype(str)
    opb["facility_zip5"] = (
        opb["CLM_SRVC_FAC_ZIP_CD"]
        .astype(str)
        .str.slice(0, 5)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(5)
    )

    op = op.merge(opb[["CLM_ID", "PRVDR_NUM", "facility_zip5"]], on="CLM_ID", how="left")
    op = op.rename(columns={"PRVDR_NUM": "facility_id"})
    op["facility_id"] = op["facility_id"].astype(str)
    op = op[op["facility_id"].isin(providers)]

    event_cols = ["PRVDR_NUM", "event_id", "storm_id", "earliest_stress_day"]
    if "cluster_id" in events.columns:
        event_cols.insert(1, "cluster_id")
    if "cluster_label" in events.columns:
        event_cols.insert(2 if "cluster_id" in event_cols else 1, "cluster_label")

    events_dd = dd.from_pandas(events[event_cols], npartitions=1).rename(columns={"PRVDR_NUM": "facility_id"})
    events_dd["facility_id"] = events_dd["facility_id"].astype(str)
    op = op.merge(events_dd, on="facility_id", how="inner")

    zip_xw_dd = dd.from_pandas(zip_xw.copy(), npartitions=1)
    op["facility_zip5"] = op["facility_zip5"].astype(str).str.zfill(5)
    op = op.merge(zip_xw_dd, left_on="facility_zip5", right_on="zip5", how="left")
    op = op.rename(columns={"fips": "facility_county_fips"}).drop("zip5", axis=1)
    op["facility_county_fips"] = op["facility_county_fips"].astype(str).str.zfill(5)

    exp_dd = dd.from_pandas(county_exp.copy(), npartitions=1)
    op = op.merge(
        exp_dd,
        left_on=["storm_id", "facility_county_fips"],
        right_on=["storm_id", "fips"],
        how="left",
    ).drop("fips", axis=1)

    op = op.rename(columns={"exposure_start_dt": "county_exposure_start_dt"})
    op["county_exposure_start_dt"] = dd.to_datetime(op["county_exposure_start_dt"], errors="coerce").dt.normalize()

    op["placebo_anchor_dt"] = op["county_exposure_start_dt"] - pd.Timedelta(days=PLACEBO_SHIFT_DAYS)
    op["anchor_dt"] = op["placebo_anchor_dt"]

    if DROP_MISSING_EXPOSURE:
        op = op[op["anchor_dt"].notnull()]

    op = op.assign(rel_day=(op["REV_CNTR_DT"] - op["anchor_dt"]).dt.days)
    op_win = op[(op["rel_day"] >= COHORT_LO) & (op["rel_day"] <= COHORT_HI)]

    keep_cols = [
        "event_id", "facility_id", "BENE_ID",
        "storm_id",
        "earliest_stress_day",
        "facility_zip5", "facility_county_fips",
        "county_exposure_start_dt",
        "placebo_anchor_dt",
        "anchor_dt",
    ]
    if "cluster_id" in op_win.columns:
        keep_cols.insert(1, "cluster_id")
    if "cluster_label" in op_win.columns:
        keep_cols.insert(2 if "cluster_id" in keep_cols else 1, "cluster_label")

    cohort = op_win[keep_cols].drop_duplicates().compute()

    for c in ["county_exposure_start_dt", "placebo_anchor_dt", "anchor_dt", "earliest_stress_day"]:
        cohort[c] = pd.to_datetime(cohort[c], errors="coerce").dt.normalize()

    if DROP_MISSING_EXPOSURE:
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

# =========================
# Outcome flags (ED/IP) anchored on placebo_anchor_dt
# =========================
def outcome_flags_from_ed_year_placebo_anchor(cohort: pd.DataFrame, ed_pq: str) -> pd.DataFrame:
    if cohort.empty:
        return pd.DataFrame(columns=[
            "event_id", "BENE_ID",
            "any_ed_wk_m7", "any_ed_wkm5", "any_ed_wkm4", "any_ed_wkm3", "any_ed_wkm2",
            "any_ed_post_2wk", "any_ed_post_3wk", "any_ed_post_4wk",
        ])

    benes_dd = dd.from_pandas(pd.DataFrame({"BENE_ID": cohort["BENE_ID"].unique()}), npartitions=1)

    tmin = cohort["anchor_dt"].min() + pd.Timedelta(days=REF_LO)
    tmax = cohort["anchor_dt"].max() + pd.Timedelta(days=POST_4WK_HI)

    ed = dd.read_parquet(ed_pq, columns=["BENE_ID", "REV_CNTR_DT"])
    ed = ed.assign(date=dd.to_datetime(ed["REV_CNTR_DT"], errors="coerce"))
    ed = ed[(ed["date"] >= tmin) & (ed["date"] <= tmax)]
    ed = ed.merge(benes_dd, on="BENE_ID", how="inner")

    cohort_dd = dd.from_pandas(cohort[["event_id", "BENE_ID", "anchor_dt"]], npartitions=1)
    ed = ed.merge(cohort_dd, on="BENE_ID", how="inner")
    ed = ed.assign(rel_day=(ed["date"] - ed["anchor_dt"]).dt.days)

    ed = ed.assign(
        flag_wk_m7=((ed["rel_day"] >= REF_LO) & (ed["rel_day"] <= REF_HI)).astype("int8"),
        flag_wkm5=((ed["rel_day"] >= HAZ_LO) & (ed["rel_day"] <= HAZ_HI)).astype("int8"),
        flag_wkm4=((ed["rel_day"] >= WKM4_LO) & (ed["rel_day"] <= WKM4_HI)).astype("int8"),
        flag_wkm3=((ed["rel_day"] >= WKM3_LO) & (ed["rel_day"] <= WKM3_HI)).astype("int8"),
        flag_wkm2=((ed["rel_day"] >= WKM2_LO) & (ed["rel_day"] <= WKM2_HI)).astype("int8"),
        flag_post_2wk=((ed["rel_day"] >= POST_2WK_LO) & (ed["rel_day"] <= POST_2WK_HI)).astype("int8"),
        flag_post_3wk=((ed["rel_day"] >= POST_3WK_LO) & (ed["rel_day"] <= POST_3WK_HI)).astype("int8"),
        flag_post_4wk=((ed["rel_day"] >= POST_4WK_LO) & (ed["rel_day"] <= POST_4WK_HI)).astype("int8"),
    )

    grp = (
        ed.groupby(["event_id", "BENE_ID"])[[
            "flag_wk_m7", "flag_wkm5", "flag_wkm4", "flag_wkm3", "flag_wkm2",
            "flag_post_2wk", "flag_post_3wk", "flag_post_4wk",
        ]]
        .max()
        .rename(columns={
            "flag_wk_m7": "any_ed_wk_m7",
            "flag_wkm5": "any_ed_wkm5",
            "flag_wkm4": "any_ed_wkm4",
            "flag_wkm3": "any_ed_wkm3",
            "flag_wkm2": "any_ed_wkm2",
            "flag_post_2wk": "any_ed_post_2wk",
            "flag_post_3wk": "any_ed_post_3wk",
            "flag_post_4wk": "any_ed_post_4wk",
        })
        .reset_index()
        .compute()
    )
    return grp

def outcome_flags_from_ip_year_placebo_anchor(cohort: pd.DataFrame, medpar_pq: str) -> pd.DataFrame:
    if cohort.empty:
        return pd.DataFrame(columns=[
            "event_id", "BENE_ID",
            "any_ip_wk_m7", "any_ip_wkm5", "any_ip_wkm4", "any_ip_wkm3", "any_ip_wkm2",
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
        flag_wk_m7=((ip["rel_day"] >= REF_LO) & (ip["rel_day"] <= REF_HI)).astype("int8"),
        flag_wkm5=((ip["rel_day"] >= HAZ_LO) & (ip["rel_day"] <= HAZ_HI)).astype("int8"),
        flag_wkm4=((ip["rel_day"] >= WKM4_LO) & (ip["rel_day"] <= WKM4_HI)).astype("int8"),
        flag_wkm3=((ip["rel_day"] >= WKM3_LO) & (ip["rel_day"] <= WKM3_HI)).astype("int8"),
        flag_wkm2=((ip["rel_day"] >= WKM2_LO) & (ip["rel_day"] <= WKM2_HI)).astype("int8"),
        flag_post_2wk=((ip["rel_day"] >= POST_2WK_LO) & (ip["rel_day"] <= POST_2WK_HI)).astype("int8"),
        flag_post_3wk=((ip["rel_day"] >= POST_3WK_LO) & (ip["rel_day"] <= POST_3WK_HI)).astype("int8"),
        flag_post_4wk=((ip["rel_day"] >= POST_4WK_LO) & (ip["rel_day"] <= POST_4WK_HI)).astype("int8"),
    )

    grp = (
        ip.groupby(["event_id", "BENE_ID"])[[
            "flag_wk_m7", "flag_wkm5", "flag_wkm4", "flag_wkm3", "flag_wkm2",
            "flag_post_2wk", "flag_post_3wk", "flag_post_4wk",
        ]]
        .max()
        .rename(columns={
            "flag_wk_m7": "any_ip_wk_m7",
            "flag_wkm5": "any_ip_wkm5",
            "flag_wkm4": "any_ip_wkm4",
            "flag_wkm3": "any_ip_wkm3",
            "flag_wkm2": "any_ip_wkm2",
            "flag_post_2wk": "any_ip_post_2wk",
            "flag_post_3wk": "any_ip_post_3wk",
            "flag_post_4wk": "any_ip_post_4wk",
        })
        .reset_index()
        .compute()
    )
    return grp

# =========================
# Dialysis outcomes + schedule + earlyA
# =========================
def dialysis_features_from_op_year_placebo_anchor(cohort: pd.DataFrame, op_pq: str) -> pd.DataFrame:
    """
    Computes:
      Outcomes:
        - n_dialysis_wk_m7: count of unique dialysis dates in REF window [-14,-8]
        - n_dialysis_wkm5: count of unique dialysis dates in HAZ window [0,6]
        - n_dialysis_wkm4: count of unique dialysis dates in [7,13]
        - n_dialysis_wkm3: count of unique dialysis dates in [14,20]
        - n_dialysis_wkm2: count of unique dialysis dates in [21,27]
      EarlyA (based on placebo week -1):
        - date_pre_m1: max dialysis date in [-7,-1]
        - earlyA_last_pre_offschedule
      Canonical schedule (based on placebo week -3):
        - schedule_type (MWF/TTS) and stable_3x_weekly using [-21,-15]
      Gap / hazard availability:
        - date_post: min dialysis date in [0,6]
        - gap_days: date_post - date_pre_m1
        - no_hazard_dialysis
    """
    if cohort.empty:
        return pd.DataFrame(
            columns=[
                "event_id","BENE_ID",
                "n_dialysis_wk_m7","n_dialysis_wkm5","n_dialysis_wkm4","n_dialysis_wkm3","n_dialysis_wkm2",
                "gap_days","no_hazard_dialysis",
                "schedule_type","stable_3x_weekly",
                "earlyA_last_pre_offschedule",
            ]
        )

    benes_dd = dd.from_pandas(pd.DataFrame({"BENE_ID": cohort["BENE_ID"].unique()}), npartitions=1)

    tmin = cohort["anchor_dt"].min() + pd.Timedelta(days=OP_PULL_LO)
    tmax = cohort["anchor_dt"].max() + pd.Timedelta(days=OP_PULL_HI)

    op = dd.read_parquet(op_pq, columns=["BENE_ID", "REV_CNTR_DT"])
    op = op.assign(date=dd.to_datetime(op["REV_CNTR_DT"], errors="coerce"))
    op = op[(op["date"] >= tmin) & (op["date"] <= tmax)]
    op = op.merge(benes_dd, on="BENE_ID", how="inner")

    cohort_dd = dd.from_pandas(cohort[["event_id", "BENE_ID", "anchor_dt"]], npartitions=1)
    op = op.merge(cohort_dd, on="BENE_ID", how="inner")
    op = op.assign(rel_day=(op["date"] - op["anchor_dt"]).dt.days)

    op_day = op[["event_id", "BENE_ID", "anchor_dt", "date", "rel_day"]].drop_duplicates()

    op_day = op_day.assign(
        wk_m7=((op_day["rel_day"] >= REF_LO) & (op_day["rel_day"] <= REF_HI)).astype("int8"),
        wkm5=((op_day["rel_day"] >= HAZ_LO) & (op_day["rel_day"] <= HAZ_HI)).astype("int8"),
        wkm4=((op_day["rel_day"] >= WKM4_LO) & (op_day["rel_day"] <= WKM4_HI)).astype("int8"),
        wkm3=((op_day["rel_day"] >= WKM3_LO) & (op_day["rel_day"] <= WKM3_HI)).astype("int8"),
        wkm2=((op_day["rel_day"] >= WKM2_LO) & (op_day["rel_day"] <= WKM2_HI)).astype("int8"),
        wk_m1=((op_day["rel_day"] >= EARLY_LO) & (op_day["rel_day"] <= EARLY_HI)).astype("int8"),
        class_week=((op_day["rel_day"] >= CLASS_LO) & (op_day["rel_day"] <= CLASS_HI)).astype("int8"),
    )

    op_day["dow"] = op_day["date"].dt.weekday
    op_day["is_mwf_day"] = ((op_day["dow"] == 0) | (op_day["dow"] == 2) | (op_day["dow"] == 4)).astype("int8")
    op_day["is_tts_day"] = ((op_day["dow"] == 1) | (op_day["dow"] == 3) | (op_day["dow"] == 5)).astype("int8")

    op_day["date_pre_m1"] = op_day["date"].where(op_day["wk_m1"] == 1)
    op_day["date_post"] = op_day["date"].where(op_day["wkm5"] == 1)

    op_day["class_MWF"] = ((op_day["class_week"] == 1) & (op_day["is_mwf_day"] == 1)).astype("int8")
    op_day["class_TTS"] = ((op_day["class_week"] == 1) & (op_day["is_tts_day"] == 1)).astype("int8")
    op_day["class_other"] = ((op_day["class_week"] == 1) & (op_day["is_mwf_day"] == 0) & (op_day["is_tts_day"] == 0)).astype("int8")

    grp = (
        op_day.groupby(["event_id", "BENE_ID"])
        .agg({
            "wk_m7": "sum",
            "wkm5": "sum",
            "wkm4": "sum",
            "wkm3": "sum",
            "wkm2": "sum",
            "date_pre_m1": "max",
            "date_post": "min",
            "class_week": "sum",
            "class_MWF": "sum",
            "class_TTS": "sum",
            "class_other": "sum",
        })
        .reset_index()
        .compute()
    )

    grp = grp.rename(columns={
        "wk_m7": "n_dialysis_wk_m7",
        "wkm5": "n_dialysis_wkm5",
        "wkm4": "n_dialysis_wkm4",
        "wkm3": "n_dialysis_wkm3",
        "wkm2": "n_dialysis_wkm2",
        "class_week": "n_class_total",
        "class_MWF": "n_class_MWF",
        "class_TTS": "n_class_TTS",
        "class_other": "n_class_other",
    })

    for c in [
        "n_dialysis_wk_m7","n_dialysis_wkm5","n_dialysis_wkm4","n_dialysis_wkm3","n_dialysis_wkm2",
        "n_class_total","n_class_MWF","n_class_TTS","n_class_other"
    ]:
        grp[c] = grp[c].fillna(0).astype("int16")

    grp["gap_days"] = np.nan
    valid_gap = grp["date_pre_m1"].notna() & grp["date_post"].notna()
    grp.loc[valid_gap, "gap_days"] = (grp.loc[valid_gap, "date_post"] - grp.loc[valid_gap, "date_pre_m1"]).dt.days.astype(float)
    grp["no_hazard_dialysis"] = grp["date_post"].isna().astype("int8")

    grp["schedule_type"] = pd.NA
    if STRICT_STABLE_SCHEDULE:
        cond_mwf = (grp["n_class_total"] == 3) & (grp["n_class_MWF"] == 3) & (grp["n_class_TTS"] == 0) & (grp["n_class_other"] == 0)
        cond_tts = (grp["n_class_total"] == 3) & (grp["n_class_TTS"] == 3) & (grp["n_class_MWF"] == 0) & (grp["n_class_other"] == 0)
    else:
        cond_mwf = (grp["n_class_total"] >= 2) & (grp["n_class_MWF"] == grp["n_class_total"]) & (grp["n_class_TTS"] == 0) & (grp["n_class_other"] == 0)
        cond_tts = (grp["n_class_total"] >= 2) & (grp["n_class_TTS"] == grp["n_class_total"]) & (grp["n_class_MWF"] == 0) & (grp["n_class_other"] == 0)

    grp.loc[cond_mwf, "schedule_type"] = "MWF"
    grp.loc[cond_tts, "schedule_type"] = "TTS"
    grp["stable_3x_weekly"] = grp["schedule_type"].notna().astype("int8")

    grp["earlyA_last_pre_offschedule"] = 0
    grp["pre_dow_m1"] = pd.to_datetime(grp["date_pre_m1"], errors="coerce").dt.weekday

    mwf_mask = (grp["stable_3x_weekly"] == 1) & (grp["schedule_type"] == "MWF") & grp["date_pre_m1"].notna()
    tts_mask = (grp["stable_3x_weekly"] == 1) & (grp["schedule_type"] == "TTS") & grp["date_pre_m1"].notna()

    grp.loc[mwf_mask & (~grp["pre_dow_m1"].isin([0, 2, 4])), "earlyA_last_pre_offschedule"] = 1
    grp.loc[tts_mask & (~grp["pre_dow_m1"].isin([1, 3, 5])), "earlyA_last_pre_offschedule"] = 1

    return grp[
        [
            "event_id","BENE_ID",
            "n_dialysis_wk_m7","n_dialysis_wkm5","n_dialysis_wkm4","n_dialysis_wkm3","n_dialysis_wkm2",
            "gap_days","no_hazard_dialysis",
            "schedule_type","stable_3x_weekly",
            "earlyA_last_pre_offschedule",
        ]
    ]

# =========================
# Bring MBSF
# =========================
def bring_mbsf_for_cohort(cohort: pd.DataFrame, mbsf_pq: str, year: int) -> pd.DataFrame:
    if cohort.empty:
        return pd.DataFrame(columns=["BENE_ID", "BENE_DEATH_DT", "BENE_BIRTH_DT", "SEX_IDENT_CD"])

    benes_dd = dd.from_pandas(pd.DataFrame({"BENE_ID": cohort["BENE_ID"].unique()}), npartitions=1)

    if year > 2017:
        m = dd.read_parquet(mbsf_pq, columns=["BENE_ID", "BENE_DEATH_DT", "BENE_BIRTH_DT", "SEX_IDENT_CD"])
    else:
        m = dd.read_parquet(mbsf_pq, columns=["BENE_DEATH_DT", "BENE_BIRTH_DT", "SEX_IDENT_CD"])
        if m.index.name == "BENE_ID":
            m = m.reset_index()

    m = m.merge(benes_dd, on="BENE_ID", how="inner").compute()
    for c in ["BENE_DEATH_DT", "BENE_BIRTH_DT"]:
        m[c] = pd.to_datetime(m[c], errors="coerce").dt.normalize()

    return m[["BENE_ID", "BENE_DEATH_DT", "BENE_BIRTH_DT", "SEX_IDENT_CD"]]

# =========================
# Make long panel (2 rows per event/bene) anchored on placebo_anchor_dt
# =========================
def make_long_panel_placebo_anchor(
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
    df["placebo_anchor_dt"] = pd.to_datetime(df["placebo_anchor_dt"], errors="coerce").dt.normalize()

    df = df.merge(dial_feat, on=["event_id", "BENE_ID"], how="left")

    for col in ["n_dialysis_wk_m7","n_dialysis_wkm5","n_dialysis_wkm4","n_dialysis_wkm3","n_dialysis_wkm2"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype("int16")

    df["disrupt_wk_m7"] = (df["n_dialysis_wk_m7"] < 3).astype("int8")
    df["disrupt_wkm5"] = (df["n_dialysis_wkm5"] < 3).astype("int8")
    df["disrupt_wkm4"] = (df["n_dialysis_wkm4"] < 3).astype("int8")
    df["disrupt_wkm3"] = (df["n_dialysis_wkm3"] < 3).astype("int8")
    df["disrupt_wkm2"] = (df["n_dialysis_wkm2"] < 3).astype("int8")

    df["disrupt_post_2wk"] = df[["disrupt_wkm5", "disrupt_wkm4"]].max(axis=1).astype("int8")
    df["disrupt_post_3wk"] = df[["disrupt_wkm5", "disrupt_wkm4", "disrupt_wkm3"]].max(axis=1).astype("int8")
    df["disrupt_post_4wk"] = df[["disrupt_wkm5", "disrupt_wkm4", "disrupt_wkm3", "disrupt_wkm2"]].max(axis=1).astype("int8")

    df = df.merge(ip_out, on=["event_id", "BENE_ID"], how="left")
    df = df.merge(ed_out, on=["event_id", "BENE_ID"], how="left")

    outcome_fill_cols = [
        "any_ip_wk_m7","any_ip_wkm5","any_ip_wkm4","any_ip_wkm3","any_ip_wkm2",
        "any_ip_post_2wk","any_ip_post_3wk","any_ip_post_4wk",
        "any_ed_wk_m7","any_ed_wkm5","any_ed_wkm4","any_ed_wkm3","any_ed_wkm2",
        "any_ed_post_2wk","any_ed_post_3wk","any_ed_post_4wk",
    ]
    for col in outcome_fill_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype("int8")

    df = df.merge(mbsf, on="BENE_ID", how="left")
    df["BENE_DEATH_DT"] = pd.to_datetime(df["BENE_DEATH_DT"], errors="coerce").dt.normalize()

    died_before_placebo_anchor = (
        df["BENE_DEATH_DT"].notna() & (df["BENE_DEATH_DT"] < df["anchor_dt"])
    ).astype("int8")

    n_before = len(df)
    df = df[died_before_placebo_anchor == 0].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Year {year}: dropped {n_dropped:,} (event_id, BENE_ID) rows with death before placebo_anchor_dt.")

    df["anchor_dow"] = df["anchor_dt"].dt.weekday.astype("int8")
    df["anchor_on_usual_sched_day"] = pd.NA
    df["anchor_on_off_sched_day"] = pd.NA

    mask_stable = (df["stable_3x_weekly"] == 1) & df["schedule_type"].notna()
    mask_mwf = mask_stable & (df["schedule_type"] == "MWF")
    mask_tts = mask_stable & (df["schedule_type"] == "TTS")

    df.loc[mask_mwf, "anchor_on_usual_sched_day"] = df.loc[mask_mwf, "anchor_dow"].isin([0, 2, 4]).astype("int8")
    df.loc[mask_tts, "anchor_on_usual_sched_day"] = df.loc[mask_tts, "anchor_dow"].isin([1, 3, 5]).astype("int8")
    df.loc[mask_stable, "anchor_on_off_sched_day"] = (1 - df.loc[mask_stable, "anchor_on_usual_sched_day"].astype("int8")).astype("int8")

    death_rel_day = (df["BENE_DEATH_DT"] - df["anchor_dt"]).dt.days

    df["any_death_wk_m7"] = (
        death_rel_day.notna() &
        (death_rel_day >= REF_LO) &
        (death_rel_day <= REF_HI)
    ).astype("int8")

    df["any_death_wkm5"] = (
        death_rel_day.notna() &
        (death_rel_day >= HAZ_LO) &
        (death_rel_day <= HAZ_HI)
    ).astype("int8")

    df["any_death_wkm4"] = (
        death_rel_day.notna() &
        (death_rel_day >= HAZ_LO) &
        (death_rel_day <= WKM4_HI)
    ).astype("int8")

    df["any_death_wkm3"] = (
        death_rel_day.notna() &
        (death_rel_day >= HAZ_LO) &
        (death_rel_day <= WKM3_HI)
    ).astype("int8")

    df["any_death_wkm2"] = (
        death_rel_day.notna() &
        (death_rel_day >= HAZ_LO) &
        (death_rel_day <= WKM2_HI)
    ).astype("int8")

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

    has_cluster_id = "cluster_id" in df.columns
    has_cluster_label = "cluster_label" in df.columns

    base_cols = [
        "event_id",
        "BENE_ID",
        "facility_id",
        "storm_id",
        "earliest_stress_day",
        "county_exposure_start_dt",
        "placebo_anchor_dt",
        "anchor_dt",
        "anchor_dow",
        "anchor_on_usual_sched_day",
        "anchor_on_off_sched_day",
        "schedule_type",
        "stable_3x_weekly",
        "earlyA_last_pre_offschedule",
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

    wk_m7 = df[base_cols].copy()
    wk_m7["week_rel"] = REF_WEEK_LABEL
    wk_m7["hazard_week"] = 0

    wk_m7["any_ip"] = df["any_ip_wk_m7"].values
    wk_m7["any_ed"] = df["any_ed_wk_m7"].values
    wk_m7["any_death"] = df["any_death_wk_m7"].values
    wk_m7["n_dialysis"] = df["n_dialysis_wk_m7"].values
    wk_m7["disrupt"] = df["disrupt_wk_m7"].values

    wk_m7["any_ip_cmp_wk"] = df["any_ip_wk_m7"].values
    wk_m7["any_ip_cmp_2wk"] = df["any_ip_wk_m7"].values
    wk_m7["any_ip_cmp_3wk"] = df["any_ip_wk_m7"].values
    wk_m7["any_ip_cmp_4wk"] = df["any_ip_wk_m7"].values

    wk_m7["any_ed_cmp_wk"] = df["any_ed_wk_m7"].values
    wk_m7["any_ed_cmp_2wk"] = df["any_ed_wk_m7"].values
    wk_m7["any_ed_cmp_3wk"] = df["any_ed_wk_m7"].values
    wk_m7["any_ed_cmp_4wk"] = df["any_ed_wk_m7"].values

    wk_m7["any_death_cmp_wk"] = df["any_death_wk_m7"].values
    wk_m7["any_death_cmp_2wk"] = df["any_death_wk_m7"].values
    wk_m7["any_death_cmp_3wk"] = df["any_death_wk_m7"].values
    wk_m7["any_death_cmp_4wk"] = df["any_death_wk_m7"].values

    wk_m7["disrupt_cmp_wk"] = df["disrupt_wk_m7"].values
    wk_m7["disrupt_cmp_2wk"] = df["disrupt_wk_m7"].values
    wk_m7["disrupt_cmp_3wk"] = df["disrupt_wk_m7"].values
    wk_m7["disrupt_cmp_4wk"] = df["disrupt_wk_m7"].values

    wk_m7["earlyA_last_pre_offschedule"] = 0

    wk_m7["any_ip_wkm4"] = df["any_ip_wkm4"].values
    wk_m7["any_ip_wkm3"] = df["any_ip_wkm3"].values
    wk_m7["any_ip_wkm2"] = df["any_ip_wkm2"].values
    wk_m7["any_ip_post_2wk"] = df["any_ip_post_2wk"].values
    wk_m7["any_ip_post_3wk"] = df["any_ip_post_3wk"].values
    wk_m7["any_ip_post_4wk"] = df["any_ip_post_4wk"].values

    wk_m7["any_ed_wkm4"] = df["any_ed_wkm4"].values
    wk_m7["any_ed_wkm3"] = df["any_ed_wkm3"].values
    wk_m7["any_ed_wkm2"] = df["any_ed_wkm2"].values
    wk_m7["any_ed_post_2wk"] = df["any_ed_post_2wk"].values
    wk_m7["any_ed_post_3wk"] = df["any_ed_post_3wk"].values
    wk_m7["any_ed_post_4wk"] = df["any_ed_post_4wk"].values

    wk_m7["any_death_wkm4"] = df["any_death_wkm4"].values
    wk_m7["any_death_wkm3"] = df["any_death_wkm3"].values
    wk_m7["any_death_wkm2"] = df["any_death_wkm2"].values
    wk_m7["any_death_post_2wk"] = df["any_death_post_2wk"].values
    wk_m7["any_death_post_3wk"] = df["any_death_post_3wk"].values
    wk_m7["any_death_post_4wk"] = df["any_death_post_4wk"].values

    wk_m7["disrupt_wkm4"] = df["disrupt_wkm4"].values
    wk_m7["disrupt_wkm3"] = df["disrupt_wkm3"].values
    wk_m7["disrupt_wkm2"] = df["disrupt_wkm2"].values
    wk_m7["disrupt_post_2wk"] = df["disrupt_post_2wk"].values
    wk_m7["disrupt_post_3wk"] = df["disrupt_post_3wk"].values
    wk_m7["disrupt_post_4wk"] = df["disrupt_post_4wk"].values

    wkm5 = df[base_cols].copy()
    wkm5["week_rel"] = HAZ_WEEK_LABEL
    wkm5["hazard_week"] = 1

    wkm5["any_ip"] = df["any_ip_wkm5"].values
    wkm5["any_ed"] = df["any_ed_wkm5"].values
    wkm5["any_death"] = df["any_death_wkm5"].values
    wkm5["n_dialysis"] = df["n_dialysis_wkm5"].values
    wkm5["disrupt"] = df["disrupt_wkm5"].values

    wkm5["any_ip_cmp_wk"] = df["any_ip_wkm5"].values
    wkm5["any_ip_cmp_2wk"] = df["any_ip_post_2wk"].values
    wkm5["any_ip_cmp_3wk"] = df["any_ip_post_3wk"].values
    wkm5["any_ip_cmp_4wk"] = df["any_ip_post_4wk"].values

    wkm5["any_ed_cmp_wk"] = df["any_ed_wkm5"].values
    wkm5["any_ed_cmp_2wk"] = df["any_ed_post_2wk"].values
    wkm5["any_ed_cmp_3wk"] = df["any_ed_post_3wk"].values
    wkm5["any_ed_cmp_4wk"] = df["any_ed_post_4wk"].values

    wkm5["any_death_cmp_wk"] = df["any_death_wkm5"].values
    wkm5["any_death_cmp_2wk"] = df["any_death_post_2wk"].values
    wkm5["any_death_cmp_3wk"] = df["any_death_post_3wk"].values
    wkm5["any_death_cmp_4wk"] = df["any_death_post_4wk"].values

    wkm5["disrupt_cmp_wk"] = df["disrupt_wkm5"].values
    wkm5["disrupt_cmp_2wk"] = df["disrupt_post_2wk"].values
    wkm5["disrupt_cmp_3wk"] = df["disrupt_post_3wk"].values
    wkm5["disrupt_cmp_4wk"] = df["disrupt_post_4wk"].values

    wkm5["any_ip_wkm4"] = df["any_ip_wkm4"].values
    wkm5["any_ip_wkm3"] = df["any_ip_wkm3"].values
    wkm5["any_ip_wkm2"] = df["any_ip_wkm2"].values
    wkm5["any_ip_post_2wk"] = df["any_ip_post_2wk"].values
    wkm5["any_ip_post_3wk"] = df["any_ip_post_3wk"].values
    wkm5["any_ip_post_4wk"] = df["any_ip_post_4wk"].values

    wkm5["any_ed_wkm4"] = df["any_ed_wkm4"].values
    wkm5["any_ed_wkm3"] = df["any_ed_wkm3"].values
    wkm5["any_ed_wkm2"] = df["any_ed_wkm2"].values
    wkm5["any_ed_post_2wk"] = df["any_ed_post_2wk"].values
    wkm5["any_ed_post_3wk"] = df["any_ed_post_3wk"].values
    wkm5["any_ed_post_4wk"] = df["any_ed_post_4wk"].values

    wkm5["any_death_wkm4"] = df["any_death_wkm4"].values
    wkm5["any_death_wkm3"] = df["any_death_wkm3"].values
    wkm5["any_death_wkm2"] = df["any_death_wkm2"].values
    wkm5["any_death_post_2wk"] = df["any_death_post_2wk"].values
    wkm5["any_death_post_3wk"] = df["any_death_post_3wk"].values
    wkm5["any_death_post_4wk"] = df["any_death_post_4wk"].values

    wkm5["disrupt_wkm4"] = df["disrupt_wkm4"].values
    wkm5["disrupt_wkm3"] = df["disrupt_wkm3"].values
    wkm5["disrupt_wkm2"] = df["disrupt_wkm2"].values
    wkm5["disrupt_post_2wk"] = df["disrupt_post_2wk"].values
    wkm5["disrupt_post_3wk"] = df["disrupt_post_3wk"].values
    wkm5["disrupt_post_4wk"] = df["disrupt_post_4wk"].values

    long = pd.concat([wk_m7, wkm5], ignore_index=True)
    long["year"] = year

    if (long.loc[long["week_rel"] == REF_WEEK_LABEL, "any_death"] == 1).any():
        raise ValueError("Found any_death==1 in placebo reference row (week_rel=-7); check death-before-placebo-anchor drop logic.")

    col_order = [
        "year","event_id","BENE_ID","week_rel","hazard_week",
        "any_ip","any_ed","any_death","n_dialysis","disrupt",
        "any_ip_cmp_wk","any_ip_cmp_2wk","any_ip_cmp_3wk","any_ip_cmp_4wk",
        "any_ed_cmp_wk","any_ed_cmp_2wk","any_ed_cmp_3wk","any_ed_cmp_4wk",
        "any_death_cmp_wk","any_death_cmp_2wk","any_death_cmp_3wk","any_death_cmp_4wk",
        "disrupt_cmp_wk","disrupt_cmp_2wk","disrupt_cmp_3wk","disrupt_cmp_4wk",
        "gap_days","no_hazard_dialysis",
        "facility_id",
        "storm_id",
        "earliest_stress_day",
        "county_exposure_start_dt",
        "placebo_anchor_dt",
        "anchor_dt",
        "anchor_dow","anchor_on_usual_sched_day","anchor_on_off_sched_day",
        "schedule_type","stable_3x_weekly",
        "earlyA_last_pre_offschedule",
        "any_ip_wkm4","any_ip_wkm3","any_ip_wkm2","any_ip_post_2wk","any_ip_post_3wk","any_ip_post_4wk",
        "any_ed_wkm4","any_ed_wkm3","any_ed_wkm2","any_ed_post_2wk","any_ed_post_3wk","any_ed_post_4wk",
        "any_death_wkm4","any_death_wkm3","any_death_wkm2","any_death_post_2wk","any_death_post_3wk","any_death_post_4wk",
        "disrupt_wkm4","disrupt_wkm3","disrupt_wkm2","disrupt_post_2wk","disrupt_post_3wk","disrupt_post_4wk",
        "BENE_DEATH_DT","BENE_BIRTH_DT","SEX_IDENT_CD",
    ]
    if has_cluster_id:
        col_order.insert(2, "cluster_id")
    if has_cluster_label:
        col_order.insert(3 if has_cluster_id else 2, "cluster_label")

    long = long[col_order].sort_values(["event_id", "BENE_ID", "week_rel"]).reset_index(drop=True)
    return long

# =========================
# Main build loop
# =========================
if __name__ == "__main__":
    print("[LOAD] ZIP->county crosswalk...")
    zip_xw = load_zip_to_county()
    print(f"  rows={len(zip_xw):,}")

    print("[LOAD] County exposure start dates...")
    county_exp = load_county_exposure()
    print(f"  rows={len(county_exp):,} | storms={county_exp['storm_id'].nunique():,} | counties={county_exp['fips'].nunique():,}")

    stress = load_stress_table()
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

        cohort = build_event_cohort_for_year_placebo_anchor(stress_year, op_pq, opb_pq, zip_xw, county_exp)
        if cohort.empty:
            print(f"Year {year}: no placebo exposure-anchored cohort; skipping.")
            continue

        print(
            f"Year {year}: events={cohort['event_id'].nunique():,} | "
            f"benes={cohort['BENE_ID'].nunique():,}"
        )

        ip_out = outcome_flags_from_ip_year_placebo_anchor(cohort, medpar_pq)
        ed_out = outcome_flags_from_ed_year_placebo_anchor(cohort, ed_pq)
        dial_feat = dialysis_features_from_op_year_placebo_anchor(cohort, op_pq)
        mbsf = bring_mbsf_for_cohort(cohort, mbsf_pq, year)

        long = make_long_panel_placebo_anchor(year, cohort, ip_out, ed_out, dial_feat, mbsf)

        if "cluster_id" in long.columns:
            long = long.drop_duplicates(subset=["BENE_ID", "year", "cluster_id", "week_rel", "anchor_dt"])
        else:
            long = long.drop_duplicates(subset=["BENE_ID", "year", "week_rel", "anchor_dt"])

        long.to_csv(out_csv, index=False)
        print(f"Year {year}: wrote {len(long):,} rows to {out_csv}")
        


