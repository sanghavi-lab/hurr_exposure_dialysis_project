#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 29, 2026
# Description: This script appends MBSF variables to the analytical panel by reading MBSF ABCD, CC, and OTCC data. It 
# creates demographic and enrollment indicators such as race, sex, ESRD, dual eligibility, and Medicare status during the
# exposure month and also turns CC and OTCC *ever date fields into binary indicators based on whether the condition date 
# occurred on or before the exposure date. It also performs some quality checks like reporting missingness.
#----------------------------------------------------------------------------------------------------------------------#

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
    "distributed.comm.timeouts.tcp": "60s",
})
client = Client("10.50.87.26:38381")
print(client)

# -------------------------
# Paths and spec
# -------------------------
YEAR_MIN, YEAR_MAX = 2011, 2022

STEP5E_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05e_analytical_sample_anchor_exposure_plus_comorb_plus_ahrf_v01"
)

OUT_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05f_analytical_sample_anchor_exposure_plus_comorb_plus_ahrf_plus_mbsf_demo_cc_otcc_v01"
)

os.makedirs(OUT_BASE, exist_ok=True)

# -------------------------
# Functions
# -------------------------
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

def _drop_trailing_ever(c: str) -> str:
    c = _normalize_colname(c)
    c = re.sub(r"_ever$", "", c)
    return c

def _safe_unique_names(names):
    # loop through names one by one. The first time a name appears, it keeps it unchanged. If the same name appears again, it appends a suffix like __dup2. If it appears a third time, it becomes __dup3, and so on...
    
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 1
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}__dup{seen[n]}")
    return out

def step5e_file(year: int) -> str:
    return os.path.join(STEP5E_BASE, f"year_{year}", "analytical_panel.csv")

def out_year_dir(year: int) -> str:
    return os.path.join(OUT_BASE, f"year_{year}")

def out_file(year: int) -> str:
    return os.path.join(out_year_dir(year), "analytical_panel.csv")

def mbsf_abcd_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/mbsf/mbsf_abcd/parquet/"

def mbsf_cc_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/mbsf/mbsf_cc/parquet/"


def mbsf_chronic_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/mbsf/mbsf_chronic/parquet/"


def cc_path_and_label(year: int):
    """
    Use 27 CCW CC file through 2021.
    Use 30 CCW CHRONIC file from 2022 onward, then harmonize to 27-CCW-style names.
    """
    if year >= 2022:
        return mbsf_chronic_path(year), "CHRONIC_30CCW_AS_CC_27CCW"
    return mbsf_cc_path(year), "CC_27CCW"


CC_27_EVER_COLS = [
    "ALZH_EVER",
    "ALZH_DEMEN_EVER",
    "AMI_EVER",
    "ANEMIA_EVER",
    "ASTHMA_EVER",
    "ATRIAL_FIB_EVER",
    "CANCER_BREAST_EVER",
    "CANCER_COLORECTAL_EVER",
    "CANCER_ENDOMETRIAL_EVER",
    "CANCER_LUNG_EVER",
    "CANCER_PROSTATE_EVER",
    "CATARACT_EVER",
    "CHF_EVER",
    "CHRONICKIDNEY_EVER",
    "COPD_EVER",
    "DEPRESSION_EVER",
    "DIABETES_EVER",
    "GLAUCOMA_EVER",
    "HIP_FRACTURE_EVER",
    "HYPERL_EVER",
    "HYPERP_EVER",
    "HYPERT_EVER",
    "HYPOTH_EVER",
    "ISCHEMICHEART_EVER",
    "OSTEOPOROSIS_EVER",
    "RA_OA_EVER",
    "STROKE_TIA_EVER",
]


CHRONIC_30_TO_CC_27_RENAME = {
    "HF_EVER": "CHF_EVER",
    "HLP_EVER": "HYPERL_EVER",
    "BPH_EVER": "HYPERP_EVER",
    "HTN_EVER": "HYPERT_EVER",
    "HYPTHYRD_EVER": "HYPOTH_EVER",
}


def mbsf_otcc_path(year: int) -> str:
    return f"/gpfs/data/cms-share/data/medicare/{year}/mbsf/mbsf_otcc/parquet/"

def read_mbsf_subset(year: int, bene_ids: pd.Series) -> pd.DataFrame:
    """
    Read only needed MBSF ABCD columns for rel bene's.
    Handles pre-2018 vs post-2017 differences due to how bene_id was indexed
    """
    pq = mbsf_abcd_path(year)
    if not _exists(pq):
        raise FileNotFoundError(f"MBSF ABCD path not found: {pq}")

    dual_cols = [f"DUAL_STUS_CD_{m:02d}" for m in range(1, 13)]
    mdcr_cols = [f"MDCR_STATUS_CODE_{m:02d}" for m in range(1, 13)]
    needed_cols = ["RTI_RACE_CD", "SEX_IDENT_CD", "ESRD_IND"] + dual_cols + mdcr_cols

    bene_ser = pd.Series(bene_ids)
    bene_ser = _as_clean_str(bene_ser).dropna().drop_duplicates().reset_index(drop=True)
    bene_df = pd.DataFrame({"BENE_ID": bene_ser})

    bene_dd = dd.from_pandas(bene_df, npartitions=1)
    bene_dd["BENE_ID"] = bene_dd["BENE_ID"].astype(str)

    if year > 2017:
        cols = ["BENE_ID"] + needed_cols
        m = dd.read_parquet(pq, columns=cols)
        m["BENE_ID"] = m["BENE_ID"].astype(str)
    else:
        m = dd.read_parquet(pq, columns=needed_cols)
        m = m.reset_index()
        if "BENE_ID" not in m.columns:
            raise ValueError(f"{year}: could not recover BENE_ID from older MBSF ABCD parquet.")
        m["BENE_ID"] = m["BENE_ID"].astype(str)

    m = m.merge(bene_dd, on="BENE_ID", how="inner")
    m = m.compute()

    m["BENE_ID"] = _as_clean_str(m["BENE_ID"])
    for c in ["RTI_RACE_CD", "SEX_IDENT_CD", "ESRD_IND"] + dual_cols + mdcr_cols:
        if c in m.columns:
            m[c] = _as_clean_str(m[c]).str.strip()

    return m

def read_cc_like_subset(year: int, bene_ids: pd.Series, pq: str, label: str) -> pd.DataFrame:
    """
    Read MBSF CC/OTCC subset for rel bene's.
    Keep only BENE_ID + columns ending with 'ever'
    """
    if not _exists(pq):
        print(f"[WARN] {year}: missing {label} path: {pq}")
        return pd.DataFrame(columns=["BENE_ID"])

    meta = dd.read_parquet(pq, rows=0)
    cols_all = list(meta.columns)

    ever_cols = [c for c in cols_all if str(c).lower().endswith("ever")]
    if not ever_cols:
        print(f"[WARN] {year}: no *ever columns found in {label}.")
        return pd.DataFrame(columns=["BENE_ID"])

    bene_ser = pd.Series(bene_ids)
    bene_ser = _as_clean_str(bene_ser).dropna().drop_duplicates().reset_index(drop=True)
    bene_df = pd.DataFrame({"BENE_ID": bene_ser})

    bene_dd = dd.from_pandas(bene_df, npartitions=1)
    bene_dd["BENE_ID"] = bene_dd["BENE_ID"].astype(str)

    if year > 2017:
        cols = ["BENE_ID"] + ever_cols
        x = dd.read_parquet(pq, columns=cols)
        x["BENE_ID"] = x["BENE_ID"].astype(str)
    else:
        x = dd.read_parquet(pq, columns=ever_cols)
        x = x.reset_index()
        if "BENE_ID" not in x.columns:
            raise ValueError(f"{year}: could not recover BENE_ID from older {label} parquet.")
        x["BENE_ID"] = x["BENE_ID"].astype(str)

    x = x.merge(bene_dd, on="BENE_ID", how="inner")
    x = x.compute()

    x["BENE_ID"] = _as_clean_str(x["BENE_ID"])
    for c in ever_cols:
        x[c] = _as_clean_str(x[c]).str.strip()

    print(f"[INFO] {year}: {label} matched bene-year rows = {len(x):,}")
    print(f"[INFO] {year}: {label} ever columns found = {len(ever_cols):,}")

    return x

def _parse_date_series(raw: pd.Series) -> pd.Series:
    # Just converting to datetime
    
    raw = _as_clean_str(raw).str.strip()

    dt = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")

    miss1 = dt.isna() & raw.notna()
    if miss1.any():
        dt.loc[miss1] = pd.to_datetime(raw.loc[miss1], format="%Y%m%d", errors="coerce")

    miss2 = dt.isna() & raw.notna()
    if miss2.any():
        dt.loc[miss2] = pd.to_datetime(raw.loc[miss2], errors="coerce")

    return dt

def _min_date_across_cols(df: pd.DataFrame, cols) -> pd.Series:
    """
    Return the earliest non-missing date across selected date columns.
    Used to approximate 27 CCW ALZH_DEMEN_EVER from 30 CCW ALZH_EVER + NONALZH_DEMEN_EVER.
    """
    present = [c for c in cols if c in df.columns]

    if not present:
        return pd.Series(pd.NaT, index=df.index)

    dates = pd.concat([_parse_date_series(df[c]) for c in present], axis=1)
    return dates.min(axis=1)


def harmonize_30ccw_to_27ccw(chronic_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    For 2022+ MBSF CHRONIC 30 CCW files:
    - rename comparable 30 CCW variables to 27 CCW-style names
    - derive ALZH_DEMEN_EVER as earliest of ALZH_EVER and NONALZH_DEMEN_EVER
    - drop 30 CCW-only conditions to keep output consistent with 27 CCW
    """
    if chronic_df.empty or list(chronic_df.columns) == ["BENE_ID"]:
        return chronic_df

    out = chronic_df.copy()

    # Make matching robust if parquet column case differs.
    out = out.rename(columns={c: str(c).upper() for c in out.columns})

    # 27 CCW has ALZH_DEMEN_EVER = Alzheimer's disease and related disorders / senile dementia.
    # 30 CCW separates Alzheimer's disease and non-Alzheimer's dementia.
    # The closest harmonized approximation is the earliest of the two dates.
    out["ALZH_DEMEN_EVER"] = _min_date_across_cols(
        out,
        ["ALZH_EVER", "NONALZH_DEMEN_EVER"]
    )

    # Rename 30 CCW columns to 27 CCW-style names.
    out = out.rename(columns=CHRONIC_30_TO_CC_27_RENAME)

    # Keep only the 27 CCW ever columns, in 27-style naming.
    keep_cols = ["BENE_ID"] + [c for c in CC_27_EVER_COLS if c in out.columns]

    missing = sorted(set(CC_27_EVER_COLS) - set(out.columns))
    extra_ever = sorted(
        c for c in out.columns
        if c.endswith("EVER") and c not in CC_27_EVER_COLS
    )

    print(
        f"[INFO] {year}: 30 CCW harmonized to 27 CCW-style ever columns = "
        f"{len(keep_cols) - 1:,}"
    )

    if missing:
        print(
            f"[WARN] {year}: missing 27 CCW-style ever columns after harmonization: "
            f"{missing}"
        )

    if extra_ever:
        print(
            f"[INFO] {year}: dropped 30 CCW-only ever columns for consistency with 27 CCW: "
            f"{extra_ever}"
        )

    return out[keep_cols].copy()

def make_demo_indicators(m: pd.DataFrame, year: int, analytical: pd.DataFrame) -> pd.DataFrame:
    """
    Create race / sex / ESRD / dual-at-exposure-month / Medicare status indicators
    at the bene-storm (event) level. Keep only final indicators in output.
    """
    if analytical.empty:
        return pd.DataFrame(columns=["event_id", "BENE_ID"])

    ev = (
        analytical[["event_id", "BENE_ID", "anchor_dt"]]
        .drop_duplicates(subset=["event_id", "BENE_ID"])
        .copy()
        .reset_index(drop=True)
    )

    ev["BENE_ID"] = _as_clean_str(ev["BENE_ID"])
    ev["event_id"] = pd.to_numeric(ev["event_id"], errors="coerce").astype("Int64")
    ev["anchor_dt"] = pd.to_datetime(ev["anchor_dt"], errors="coerce").dt.normalize()
    ev["exposure_month"] = ev["anchor_dt"].dt.month.astype("Int64")

    tmp = ev.merge(m, on="BENE_ID", how="left", indicator=True)

    print("[QC] event-bene crosswalk <- MBSF ABCD merge:")
    print(tmp["_merge"].value_counts(dropna=False).to_string())

    tmp = tmp.drop(columns=["_merge"])

    # Race
    tmp["race_unknown"]  = (tmp["RTI_RACE_CD"] == "0").astype("Int8")
    tmp["race_nh_white"] = (tmp["RTI_RACE_CD"] == "1").astype("Int8")
    tmp["race_black"]    = (tmp["RTI_RACE_CD"] == "2").astype("Int8")
    tmp["race_other"]    = (tmp["RTI_RACE_CD"] == "3").astype("Int8")
    tmp["race_asian_pi"] = (tmp["RTI_RACE_CD"] == "4").astype("Int8")
    tmp["race_hispanic"] = (tmp["RTI_RACE_CD"] == "5").astype("Int8")
    tmp["race_ai_an"]    = (tmp["RTI_RACE_CD"] == "6").astype("Int8")

    race_cols = [
        "race_unknown", "race_nh_white", "race_black", "race_other",
        "race_asian_pi", "race_hispanic", "race_ai_an"
    ]
    valid_race_codes = {"0", "1", "2", "3", "4", "5", "6"}
    bad_race = tmp["RTI_RACE_CD"].isna() | (~tmp["RTI_RACE_CD"].isin(valid_race_codes))
    for c in race_cols:
        tmp.loc[bad_race, c] = pd.NA

    # Sex
    tmp["sex_unknown"] = (tmp["SEX_IDENT_CD"] == "0").astype("Int8")
    tmp["sex_male"]    = (tmp["SEX_IDENT_CD"] == "1").astype("Int8")
    tmp["sex_female"]  = (tmp["SEX_IDENT_CD"] == "2").astype("Int8")

    sex_cols = ["sex_unknown", "sex_male", "sex_female"]
    valid_sex_codes = {"0", "1", "2"}
    bad_sex = tmp["SEX_IDENT_CD"].isna() | (~tmp["SEX_IDENT_CD"].isin(valid_sex_codes))
    for c in sex_cols:
        tmp.loc[bad_sex, c] = pd.NA

    # ESRD
    tmp["esrd"] = pd.NA
    tmp.loc[tmp["ESRD_IND"] == "Y", "esrd"] = 1
    tmp.loc[tmp["ESRD_IND"] == "0", "esrd"] = 0
    tmp["esrd"] = tmp["esrd"].astype("Int8")

    # Dual at exposure month
    tmp["dual_code_at_exposure_month"] = pd.NA
    for mo in range(1, 13):
        col = f"DUAL_STUS_CD_{mo:02d}"
        mask = tmp["exposure_month"] == mo
        if col in tmp.columns:
            tmp.loc[mask, "dual_code_at_exposure_month"] = tmp.loc[mask, col]

    tmp["dual_code_at_exposure_month"] = _as_clean_str(tmp["dual_code_at_exposure_month"]).str.strip()

    print(f"\n[QC] {year} dual_code_at_exposure_month frequency:")
    print(tmp["dual_code_at_exposure_month"].value_counts(dropna=False).sort_index())

    dual_yes = {"01", "02", "03", "04", "05", "06", "08", "09"}
    dual_no  = {"00", "NA"}

    tmp["dual"] = pd.NA
    tmp.loc[tmp["dual_code_at_exposure_month"].isin(dual_yes), "dual"] = 1
    tmp.loc[tmp["dual_code_at_exposure_month"].isin(dual_no) | tmp["dual_code_at_exposure_month"].isna(), "dual"] = 0
    tmp["dual"] = tmp["dual"].astype("Int8")
    tmp.loc[tmp["dual_code_at_exposure_month"] == "99", "dual"] = pd.NA

    # Medicare status at exposure month
    tmp["mdcr_status_code_at_exposure_month"] = pd.NA
    for mo in range(1, 13):
        col = f"MDCR_STATUS_CODE_{mo:02d}"
        mask = tmp["exposure_month"] == mo
        if col in tmp.columns:
            tmp.loc[mask, "mdcr_status_code_at_exposure_month"] = tmp.loc[mask, col]

    tmp["mdcr_status_code_at_exposure_month"] = _as_clean_str(
        tmp["mdcr_status_code_at_exposure_month"]
    ).str.strip()

    print(f"\n[QC] {year} mdcr_status_code_at_exposure_month frequency:")
    print(tmp["mdcr_status_code_at_exposure_month"].value_counts(dropna=False).sort_index())

    valid_mdcr_codes = {"00", "10", "11", "20", "21", "31", "40"}

    tmp["medicare_aged"] = pd.NA
    tmp["medicare_disabled"] = pd.NA
    tmp["medicare_esrd_only"] = pd.NA
    tmp["medicare_with_esrd"] = pd.NA

    known_mdcr = tmp["mdcr_status_code_at_exposure_month"].isin(valid_mdcr_codes)

    for c in ["medicare_aged", "medicare_disabled", "medicare_esrd_only", "medicare_with_esrd"]:
        tmp.loc[known_mdcr, c] = 0

    tmp.loc[tmp["mdcr_status_code_at_exposure_month"].isin({"10", "11"}), "medicare_aged"] = 1
    tmp.loc[tmp["mdcr_status_code_at_exposure_month"].isin({"20", "21"}), "medicare_disabled"] = 1
    tmp.loc[tmp["mdcr_status_code_at_exposure_month"].isin({"31"}), "medicare_esrd_only"] = 1
    tmp.loc[tmp["mdcr_status_code_at_exposure_month"].isin({"11", "21", "31"}), "medicare_with_esrd"] = 1

    for c in ["medicare_aged", "medicare_disabled", "medicare_esrd_only", "medicare_with_esrd"]:
        tmp[c] = tmp[c].astype("Int8")

    out_cols = [
        "event_id", "BENE_ID",
        "race_unknown", "race_nh_white", "race_black", "race_other",
        "race_asian_pi", "race_hispanic", "race_ai_an",
        "sex_unknown", "sex_male", "sex_female",
        "esrd", "dual",
        "medicare_aged", "medicare_disabled", "medicare_esrd_only", "medicare_with_esrd",
    ]

    out = tmp[out_cols].drop_duplicates(subset=["event_id", "BENE_ID"]).reset_index(drop=True)

    dup_pairs = out.duplicated(subset=["event_id", "BENE_ID"]).sum()
    if dup_pairs > 0:
        raise ValueError(f"{year}: event-bene demo output has {dup_pairs:,} duplicated event_id-BENE_ID values.")

    return out

def make_cc_indicators(cc_df: pd.DataFrame, analytical: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Create bene-storm (event) indicators from CC or OTCC *ever date columns.
    indicator = 1 if ever_date <= anchor_dt (i.e. exposure date) else 0
    """
    ev = (
        analytical[["event_id", "BENE_ID", "anchor_dt"]]
        .drop_duplicates(subset=["event_id", "BENE_ID"])
        .copy()
        .reset_index(drop=True)
    )
    ev["BENE_ID"] = _as_clean_str(ev["BENE_ID"])
    ev["event_id"] = pd.to_numeric(ev["event_id"], errors="coerce").astype("Int64")
    ev["anchor_dt"] = pd.to_datetime(ev["anchor_dt"], errors="coerce").dt.normalize()

    if cc_df.empty or (list(cc_df.columns) == ["BENE_ID"]):
        return ev[["event_id", "BENE_ID"]].copy()

    cc_cols = [c for c in cc_df.columns if c != "BENE_ID" and str(c).lower().endswith("ever")]

    tmp = ev.merge(cc_df, on="BENE_ID", how="left", indicator=True)
    print(f"[QC] event-bene crosswalk <- {source_label} merge:")
    print(tmp["_merge"].value_counts(dropna=False).to_string())
    tmp = tmp.drop(columns=["_merge"])

    out = tmp[["event_id", "BENE_ID"]].copy()

    raw_to_clean = {c: _drop_trailing_ever(c) for c in cc_cols}
    clean_names = _safe_unique_names(list(raw_to_clean.values()))
    raw_to_clean = dict(zip(cc_cols, clean_names))

    for raw_col in cc_cols:
        out_col = raw_to_clean[raw_col]
        dt = _parse_date_series(tmp[raw_col])
        ind = ((dt.notna()) & (dt <= tmp["anchor_dt"])).astype("int8")
        out[out_col] = ind

    dup_pairs = out.duplicated(subset=["event_id", "BENE_ID"]).sum()
    if dup_pairs > 0:
        raise ValueError(f"{source_label}: event-bene output has {dup_pairs:,} duplicated event_id-BENE_ID values.")

    print(f"[INFO] {source_label} indicators created = {len([c for c in out.columns if c not in ['event_id','BENE_ID']]):,}")
    return out

def qc_pair_consistency(df: pd.DataFrame, id_cols, varlist):
    bad_total = 0
    for c in varlist:
        chk = df.groupby(id_cols, dropna=False)[c].nunique(dropna=False)
        bad_total += (chk > 1).sum()
    return bad_total

def process_year(year: int):
    print(f"\n=== Processing year {year} ===")

    f_panel = step5e_file(year)
    if not _exists(f_panel):
        print(f"[SKIP] missing analytical input: {f_panel}")
        return

    analytical = pd.read_csv(f_panel, low_memory=False)
    print(f"[INFO] analytical rows = {len(analytical):,}")

    required = ["event_id", "BENE_ID", "anchor_dt"]
    missing_required = [c for c in required if c not in analytical.columns]
    if missing_required:
        raise ValueError(f"{year}: analytical file missing required columns: {missing_required}")

    analytical["event_id"] = pd.to_numeric(analytical["event_id"], errors="coerce").astype("Int64")
    analytical["BENE_ID"] = _as_clean_str(analytical["BENE_ID"])
    analytical["anchor_dt"] = pd.to_datetime(analytical["anchor_dt"], errors="coerce").dt.normalize()

    n_rows_before = len(analytical)
    n_pairs_before = analytical[["event_id", "BENE_ID"]].drop_duplicates().shape[0]

    rows_per_pair = analytical.groupby(["event_id", "BENE_ID"], dropna=False).size()
    bad_pairs = (rows_per_pair != 2).sum()
    print(f"[QC] event_id x BENE_ID with !=2 panel rows = {bad_pairs:,}")

    # -------------------------
    # Read ABCD
    # -------------------------
    abcd = read_mbsf_subset(year, analytical["BENE_ID"])
    print(f"[INFO] MBSF ABCD matched bene-year rows read = {len(abcd):,}")
    dup_bene = abcd.duplicated(subset=["BENE_ID"]).sum()
    if dup_bene > 0:
        print(f"[WARN] {year}: MBSF ABCD subset has {dup_bene:,} duplicated BENE_ID rows; keeping first.")
        abcd = abcd.drop_duplicates(subset=["BENE_ID"], keep="first").reset_index(drop=True)

    demo_pair = make_demo_indicators(abcd, year, analytical)
    print(f"[INFO] event-bene demo rows = {len(demo_pair):,}")

    # -------------------------
    # Read CC + OTCC
    # -------------------------
    cc_path, cc_label = cc_path_and_label(year)
    
    cc = read_cc_like_subset(year, analytical["BENE_ID"], cc_path, cc_label)
    
    if year >= 2022:
        cc = harmonize_30ccw_to_27ccw(cc, year)
    
        cc_ever_cols_after = [
            c for c in cc.columns
            if c != "BENE_ID" and str(c).lower().endswith("ever")
        ]
    
        if len(cc_ever_cols_after) != len(CC_27_EVER_COLS):
            raise ValueError(
                f"{year}: expected {len(CC_27_EVER_COLS)} harmonized 27-CCW-style ever columns "
                f"from MBSF CHRONIC, but found {len(cc_ever_cols_after)}. "
                f"Columns found: {cc_ever_cols_after}"
            )
    
    otcc = read_cc_like_subset(year, analytical["BENE_ID"], mbsf_otcc_path(year), "OTCC")

    if "BENE_ID" in cc.columns:
        dup_bene_cc = cc.duplicated(subset=["BENE_ID"]).sum()
        if dup_bene_cc > 0:
            print(f"[WARN] {year}: CC subset has {dup_bene_cc:,} duplicated BENE_ID rows; keeping first.")
            cc = cc.drop_duplicates(subset=["BENE_ID"], keep="first").reset_index(drop=True)

    if "BENE_ID" in otcc.columns:
        dup_bene_otcc = otcc.duplicated(subset=["BENE_ID"]).sum()
        if dup_bene_otcc > 0:
            print(f"[WARN] {year}: OTCC subset has {dup_bene_otcc:,} duplicated BENE_ID rows; keeping first.")
            otcc = otcc.drop_duplicates(subset=["BENE_ID"], keep="first").reset_index(drop=True)

    cc_pair = make_cc_indicators(cc, analytical, cc_label)
    otcc_pair = make_cc_indicators(otcc, analytical, "OTCC")

    cc_vars = [c for c in cc_pair.columns if c not in ["event_id", "BENE_ID"]]
    otcc_vars = [c for c in otcc_pair.columns if c not in ["event_id", "BENE_ID"]]
    overlap = sorted(set(cc_vars).intersection(set(otcc_vars)))
    if overlap:
        rename_map = {c: f"{c}__otcc" for c in overlap}
        otcc_pair = otcc_pair.rename(columns=rename_map)
        print(f"[WARN] {year}: overlapping CC/OTCC indicator names found and renamed in OTCC:")
        for old, new in rename_map.items():
            print(f"       - {old} -> {new}")

    # -------------------------
    # Merge bene-storm (event) append sets together
    # -------------------------
    pair_all = demo_pair.merge(cc_pair, on=["event_id", "BENE_ID"], how="left")
    pair_all = pair_all.merge(otcc_pair, on=["event_id", "BENE_ID"], how="left")

    # -------------------------
    # Merge back to panel (analytical)
    # -------------------------
    analytical2 = analytical.merge(
        pair_all,
        on=["event_id", "BENE_ID"],
        how="left",
        indicator=True
    )

    print("[QC] analytical <- pair_all merge:")
    print(analytical2["_merge"].value_counts(dropna=False).to_string())

    if len(analytical2) != n_rows_before:
        raise ValueError(
            f"{year}: row count changed after MBSF merge. Before={n_rows_before:,}, After={len(analytical2):,}"
        )

    n_pairs_after = analytical2[["event_id", "BENE_ID"]].drop_duplicates().shape[0]
    if n_pairs_after != n_pairs_before:
        raise ValueError(
            f"{year}: unique event-bene pair count changed after MBSF merge. "
            f"Before={n_pairs_before:,}, After={n_pairs_after:,}"
        )

    # -------------------------
    # QC new vars
    # -------------------------
    new_vars = [c for c in pair_all.columns if c not in ["event_id", "BENE_ID"]]
    new_vars_present = [c for c in new_vars if c in analytical2.columns]

    print("[QC] row-level missingness for new MBSF-derived variables:")
    miss_rows = pd.DataFrame({
        "variable": new_vars_present,
        "n_missing_rows": [analytical2[c].isna().sum() for c in new_vars_present],
        "pct_missing_rows": [analytical2[c].isna().mean() for c in new_vars_present],
    })
    print(miss_rows.to_string(index=False))

    bad_repeat = qc_pair_consistency(analytical2, ["event_id", "BENE_ID"], new_vars_present)
    if bad_repeat > 0:
        raise ValueError(
            f"{year}: found {bad_repeat:,} event-bene-by-variable inconsistencies in appended MBSF values."
        )

    analytical2 = analytical2.drop(columns=["_merge"])

    # -------------------------
    # Reorder
    # -------------------------
    preferred_front = [
        "year", "storm_id", "event_id", "BENE_ID", "facility_county_fips",
        "week_rel", "hazard_week",
        "combinedscore", "combinedscore_missing",
        "ahrf_missing",
        "race_unknown", "race_nh_white", "race_black", "race_other",
        "race_asian_pi", "race_hispanic", "race_ai_an",
        "sex_unknown", "sex_male", "sex_female",
        "esrd", "dual",
        "medicare_aged", "medicare_disabled", "medicare_esrd_only", "medicare_with_esrd",
    ]
    front = [c for c in preferred_front if c in analytical2.columns]
    rest = [c for c in analytical2.columns if c not in front]
    analytical2 = analytical2[front + rest]

    # -------------------------
    # Export
    # -------------------------
    odir = out_year_dir(year)
    os.makedirs(odir, exist_ok=True)

    fout = out_file(year)
    analytical2.to_csv(fout, index=False)

    print(f"[WROTE] {year}: {fout}")
    print(f"[QC] final panel rows = {len(analytical2):,}")
    print(f"[QC] final unique event-bene pairs = {analytical2[['event_id', 'BENE_ID']].drop_duplicates().shape[0]:,}")
    print(f"[QC] total appended indicator variables = {len(new_vars_present):,}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        process_year(year)

    print(f"\n[DONE] MBSF-appended analytical files written under: {OUT_BASE}")