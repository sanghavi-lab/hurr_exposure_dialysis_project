#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 23, 2026
# Description: This script reads the analytical panel, keeps only those with chronic kidney disease, and estimates within 
# beneficiary models of ED, IP, and mortality outcomes on exposure week and early dialysis indicators, using two-way 
# clustered standard errors by facility and storm for the pooled model and one-way facility clustering for the storm-specific 
# models. The script also formats the early-effect coefficient and 95% confidence interval into a .tex file to create a 
# pooled LaTeX table and a by-storm LaTeX table in another script. In other words, it tests whether getting early dialysis 
# before the storm is associated with different post-exposure health outcomes within beneficiaries.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# -------------------------
# Paths and spec
# -------------------------
INPUT_TEMPLATE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/"
    "dialysis/01_analytical_sample/esrd_crossover_{year}/"
    "analytical_simple_case_crossover_anchor_exposure_refwk_m2_early_wkm1_class_wkm3_cumpost_cumdeath_v03.csv"
)

CKD_FLAG_FILE = ( # This was created by a file in exhibit 2 table 1 folder. Basically a dataset with indicators for chronic kidney disease
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "table1_early_vs_nonearly_v01_ckd_only/event_bene_ckd_flag_export.csv"
)

OUTPUT_DIR = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/"
    "dialysis/01_analytical_sample/paper_exhibits/"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUFFIX = "anchor_exposure_refm2_v03_earlyA_aligned_2tables"

# ... Pooled analysis outputs ...
OUT_POOLED_WIDE = os.path.join(
    OUTPUT_DIR,
    f"exhibit_optionA_earlyA_FE_pooled_only_wide_{SUFFIX}.csv",
)
OUT_POOLED_LONG = os.path.join(
    OUTPUT_DIR,
    f"exhibit_optionA_earlyA_FE_pooled_only_long_{SUFFIX}.csv",
)
OUT_POOLED_TEX_BODY = os.path.join(
    OUTPUT_DIR,
    f"exhibit_optionA_earlyA_FE_pooled_only_body_{SUFFIX}.tex",
)
OUT_POOLED_TEX_FULL = os.path.join(
    OUTPUT_DIR,
    f"exhibit_optionA_earlyA_FE_pooled_only_full_{SUFFIX}.tex",
)

# ... storm specific analysis outputs ...
OUT_STORM_WIDE = os.path.join(
    OUTPUT_DIR,
    f"exhibit_optionA_earlyA_FE_by_storm_only_wide_{SUFFIX}.csv",
)
OUT_STORM_LONG = os.path.join(
    OUTPUT_DIR,
    f"exhibit_optionA_earlyA_FE_by_storm_only_long_{SUFFIX}.csv",
)
OUT_STORM_TEX_BODY = os.path.join(
    OUTPUT_DIR,
    f"exhibit_optionA_earlyA_FE_by_storm_only_body_{SUFFIX}.tex",
)
OUT_STORM_TEX_FULL = os.path.join(
    OUTPUT_DIR,
    f"exhibit_optionA_earlyA_FE_by_storm_only_full_{SUFFIX}.tex",
)

REF_WEEK = -2
HAZARD_WEEK = 0

OUTCOMES_ORDER = [ # skipping disruption as an outcome for this model due to discrepancies in disruption definition when considering early dialysis
    ("any_ed", "ED"),
    ("any_ip", "IP"),
    ("any_death", "Death"),
]

BENE_COL = "BENE_ID"
EVENT_COL = "event_id"
WEEK_COL = "week_rel"
HAZ_COL = "hazard_week"
EARLYA_COL = "earlyA_last_pre_offschedule"
CLUSTER_COL = "facility_id" # this is for cluster se but, below, we will use a two way cluster se with STORM_COL
STORM_COL = "storm_id"

ANCHOR_DT_COL = "anchor_dt"
EXPOSURE_COL = "county_exposure_start_dt"

STABLE_COL = "stable_3x_weekly"
SCHED_COL = "schedule_type"

REPORT_TERM = EARLYA_COL
YEARS = list(range(2011, 2023))

ROUND_DECIMALS = 1 # for presentation
REPORT_IN_PERCENT_POINTS = True # multiply coef/CI by 100

# Drop hurricanes not interested in. Both hurricanes do not have available wind data.
DROP_STORM_IDS = {"Maria-2017", "Ian-2022"}
MAX_STORM_YEAR_KEEP = 2021 # Keep if 2021 or less. No wind data for 2022 and onwards

# -------------------------
# Functions
# -------------------------

# ... Formatting ...
def _r(x: float) -> float: # converts x to a float and rounds it to ROUND_DECIMALS
    if pd.isna(x):
        return np.nan
    return round(float(x), ROUND_DECIMALS)


def _pp(x: float) -> float: # converts a coefficient into percentage points.
    if pd.isna(x):
        return np.nan
    return float(x) * 100.0 if REPORT_IN_PERCENT_POINTS else float(x)


def fmt_coef_ci_pp(coef: float, lo: float, hi: float) -> str: # takes a coefficient and its lower and upper confidence limits and turns them into a printable table string
    if pd.isna(coef) or pd.isna(lo) or pd.isna(hi):
        return ""
    c, l, h = _pp(coef), _pp(lo), _pp(hi)
    return f"{_r(c):.1f} [{_r(l):.1f}, {_r(h):.1f}]"


def fmt_int_latex(n: int) -> str: # formats an integer for LaTeX like add commas
    try:
        s = f"{int(n):,}"
    except Exception:
        return ""
    return s.replace(",", "{,}")


def latex_escape(s: str) -> str: # makes text safe to print in LaTex
    if s is None:
        return ""
    s = str(s)
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("_", r"\_")
         .replace("#", r"\#")
         .replace("$", r"\$")
         .replace("{", r"\{")
         .replace("}", r"\}")
         .replace("~", r"\textasciitilde{}")
         .replace("^", r"\textasciicircum{}")
    )


# ... Cleaning/Processing ...
def _clean_str_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"\.0$", "", regex=True) # removes a trailing .0 at the end of a string.
    s = s.replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return s


def _parse_storm_year(storm_id: str) -> Optional[int]: # pull the year out of a storm id like "Sandy-2012"
    if storm_id is None or pd.isna(storm_id):
        return None
    s = str(storm_id)
    if "-" not in s:
        return None
    tail = s.split("-")[-1] # Splits the string on "-" and keeps the last piece which is the year
    try:
        return int(tail)
    except Exception:
        return None


def storm_is_kept(storm_id: str) -> bool:
    if storm_id is None or pd.isna(storm_id): # storm id is missing, do not keep it.
        return False
    sid = str(storm_id)
    if sid in DROP_STORM_IDS: # if storm id is in the predefined set of storms to exclude, drop it. (e.g. maria 2017)
        return False
    storm_year = _parse_storm_year(sid) # extract the year from the storm id
    if storm_year is None: # if no valid year could be extracted, do not keep it.
        return False
    return storm_year <= MAX_STORM_YEAR_KEEP # keep the storm only if its year is less than or equal to the cutoff year 2021


def drop_jan_2011_events(df: pd.DataFrame) -> pd.DataFrame: # removes rows tied to January 2011 events. These are winter storm events and not hurricanes.
    if df.empty:
        return df

    d = df.copy()
    if ANCHOR_DT_COL in d.columns:
        dt = pd.to_datetime(d[ANCHOR_DT_COL], errors="coerce")
        mask = (dt.dt.year == 2011) & (dt.dt.month == 1)
        return d.loc[~mask].copy() # only rows that are not in January 2011

    if "earliest_stress_day" in d.columns:
        dt = pd.to_datetime(d["earliest_stress_day"], errors="coerce")
        mask = (dt.dt.year == 2011) & (dt.dt.month == 1)
        return d.loc[~mask].copy() # only rows that are not in January 2011

    return d


def validate_required_cols(df: pd.DataFrame, year: int) -> None: # checks whether the dataset has the columns needed for the analysis
    must = {
        BENE_COL,
        EVENT_COL,
        WEEK_COL,
        HAZ_COL,
        STORM_COL,
        CLUSTER_COL,
        EARLYA_COL,
    }
    missing = sorted(list(must - set(df.columns))) # Any missing required columns are collected into a sorted list then raise error if missing
    if missing:
        raise KeyError(
            f"Year {year}: missing required columns in disrupted analytical file: {missing}\n"
            f"Expected at least: {sorted(list(must))}"
        )


def load_year_df(year: int) -> pd.DataFrame:
    path = INPUT_TEMPLATE.format(year=year) # builds the file path
    if not os.path.exists(path):
        print(f"[WARN] File not found for year {year}: {path}")
        return pd.DataFrame()

    parse_cols = [ANCHOR_DT_COL, EXPOSURE_COL, "BENE_DEATH_DT", "BENE_BIRTH_DT"] # list of columns that should be parsed as dates
    parse_cols = [c for c in parse_cols]
    df = pd.read_csv(path, parse_dates=parse_cols, low_memory=False)
    df["year"] = year

    if WEEK_COL in df.columns:
        df = df[df[WEEK_COL].isin([REF_WEEK, HAZARD_WEEK])].copy() # if week column exists, keep only the two target weeks, the reference week and hazard week.

    for col in [BENE_COL, EVENT_COL, WEEK_COL, STORM_COL, CLUSTER_COL]: # clean them as strings
        if col in df.columns:
            df[col] = _clean_str_series(df[col])

    if "facility_county_fips" in df.columns: # also clean fips as string
        df["facility_county_fips"] = _clean_str_series(df["facility_county_fips"])

    for col in [BENE_COL, EVENT_COL, WEEK_COL]:
        if col in df.columns:
            df = df[df[col].notna()].copy()

    df = drop_jan_2011_events(df) # remove jan 2011 rows
    validate_required_cols(df, year) # check required columns are present

    num_cols = [HAZ_COL, EARLYA_COL] + [c for c, _ in OUTCOMES_ORDER] # should be numeric
    for col in num_cols:
        if col not in df.columns:
            raise KeyError(f"Year {year}: required outcome/model column missing: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[HAZ_COL] = df[HAZ_COL].fillna(0).astype(int) # more defensive cleaning...
    df[EARLYA_COL] = df[EARLYA_COL].fillna(0).astype(int)
    for col, _ in OUTCOMES_ORDER:
        df[col] = df[col].fillna(0).astype(int)

    df[WEEK_COL] = pd.to_numeric(df[WEEK_COL], errors="coerce").astype(int)

    if STABLE_COL not in df.columns:
        raise KeyError(f"Year {year}: missing required restriction column {STABLE_COL}")
    if SCHED_COL not in df.columns:
        raise KeyError(f"Year {year}: missing required restriction column {SCHED_COL}")

    df[STABLE_COL] = pd.to_numeric(df[STABLE_COL], errors="coerce").fillna(0).astype(int)
    df[SCHED_COL] = _clean_str_series(df[SCHED_COL])

    return df


def restrict_to_two_weeks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    needed = {EVENT_COL, BENE_COL, WEEK_COL} # if any required columns are missing, stop with error.
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"restrict_to_two_weeks missing columns: {sorted(list(missing))}")

    counts = ( # group by event, beneficiary, and week then count rows in each combination
        df.groupby([EVENT_COL, BENE_COL, WEEK_COL])
          .size()
          .unstack(fill_value=0)
    )

    has_ref = counts.get(REF_WEEK, 0) > 0 # a boolean ind for whether each pair has at least one row in the reference week.
    has_haz = counts.get(HAZARD_WEEK, 0) > 0 # a boolean ind for whether each pair has at least one row in the exposure week.
    keep_pairs = counts.index[has_ref & has_haz] # keep only the bene that have both weeks.

    keep_df = pd.DataFrame(list(keep_pairs), columns=[EVENT_COL, BENE_COL]) # turns those kept index pairs into a small df
    return df.merge(keep_df, on=[EVENT_COL, BENE_COL], how="inner") # inner merge to ensure bene has two weeks before analysis


def restrict_to_stable_mwf_tts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    d = df.copy()
    d[STABLE_COL] = pd.to_numeric(d[STABLE_COL], errors="coerce").fillna(0).astype(int)
    d[SCHED_COL] = _clean_str_series(d[SCHED_COL])

    d = d[ # keep only rows where bene has a full week of dialysis during week -3 and schedule type is either mwf or tts. We need a full week of dialysis during the week -3 (relative to exposure week) compare like with like (stronger counterfactual)
        (d[STABLE_COL] == 1) &
        (d[SCHED_COL].isin(["MWF", "TTS"]))
    ].copy()

    return d


def count_paired_event_bene_units(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    d = df.dropna(subset=[EVENT_COL, BENE_COL, WEEK_COL]).copy()
    if d.empty:
        return 0

    tab = ( # build a table of counts by event_id, BENE_ID and week, with week values as columns.
        d.groupby([EVENT_COL, BENE_COL, WEEK_COL])
         .size()
         .unstack(fill_value=0)
    )

    ref_col = tab[REF_WEEK] if REF_WEEK in tab.columns else pd.Series(0, index=tab.index) # Gets the reference-week count column if it exists. If not, creates a zero-filled Series aligned to the same index.
    haz_col = tab[HAZARD_WEEK] if HAZARD_WEEK in tab.columns else pd.Series(0, index=tab.index) # do the same for exposure week count

    return int(((ref_col > 0) & (haz_col > 0)).sum()) # count how many pairs have at least one reference-week row and at least one hazard-week row, then returns that count as an integer. Why? this is for the sample size. We report the number of paired bene's (aka bene's with a ref week and a exposure week row)


# ... Modeling ...
def run_fe_panelols(df: pd.DataFrame, y_col: str, cluster_cols: List[str]):
    """
    PanelOLS within bene:
      y ~ hazard_week (exposure week) + early + FE(BENE_ID)

    SEs:
      - clustered, 2-way facility and storm
    """
    req = [BENE_COL, HAZ_COL, EARLYA_COL, y_col, EVENT_COL, WEEK_COL] + cluster_cols  # a list of cols the function requires in order to run.
    miss = [c for c in req if c not in df.columns] # checks whether any of those required columns are missing
    if miss:
        return None, f"missing_cols({','.join(miss)})"

    d = df.copy()

    for c in [HAZ_COL, EARLYA_COL, y_col]:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)

    d[BENE_COL] = _clean_str_series(d[BENE_COL])
    d[EVENT_COL] = _clean_str_series(d[EVENT_COL])

    d = d.dropna(subset=[BENE_COL, HAZ_COL, EARLYA_COL, y_col]).copy()
    if d.empty:
        return None, "no_data_after_dropna"

    for cc in cluster_cols:
        d[cc] = _clean_str_series(d[cc])

    d = d.dropna(subset=cluster_cols).copy()
    if d.empty:
        return None, "no_data_after_cluster_dropna"

    # Sort by specified cols
    sort_cols = [BENE_COL]
    for c in [STORM_COL, EVENT_COL, WEEK_COL]:
        if c in d.columns:
            sort_cols.append(c)
    d = d.sort_values(sort_cols).copy()

    d["_t"] = d.groupby(BENE_COL).cumcount() # creates a within-beneficiary row counter called _t. It is created so PanelOLS can treat the data as panel data with a two-level index: person and time
    d = d.set_index([BENE_COL, "_t"]) # This sets a two-level panel index: entity = bene id, time = _t
    # ^ Since the model is comparing the paired rows within each beneficiary, this creates _t as an artificial time ordering variable.

    y = d[y_col] # dependent variable (ip, ed, or mortality)
    X = d[[HAZ_COL, EARLYA_COL]]  # the regressor matrix using two predictors: hazard_week and earlyA_last_pre_offschedule

    # This is a rough check for whether the fixed-effects regression has enough degrees of freedom. The facility disruption sample has a lot less sample size than the full sample.
    nobs = int(len(d)) # Counts the number of rows used in the model.
    n_entities = int(d.index.get_level_values(BENE_COL).nunique()) # Counts the number of unique beneficiaries.
    k = int(X.shape[1]) # Counts the number of regressors in X
    approx_df_resid = nobs - n_entities - k # Approximates residual degrees of freedom: usable rows - beneficiary fixed effects - model predictors
    if approx_df_resid <= 0: # If there are no residual degrees of freedom left, the model is not estimable in a meaningful way.
        return None, (
            f"infeasible_fe(df_resid≈{approx_df_resid}, nobs={nobs}, "
            f"n_entities={n_entities}, k={k})"
        )

    mod = PanelOLS(y, X, entity_effects=True) # defines the fixed-effects regression model. entity_effects=True means beneficiary fixed effects are included. So the model compares values within the same beneficiary over time, rather than across different beneficiaries.
    clusters = d[cluster_cols] # facility and storm variables
    try:
        res = mod.fit(cov_type="clustered", clusters=clusters) # fit the linear model with two way clustering
    except ZeroDivisionError: # need this because some denominators are zero (for example, hurricane matthew 2016 [within storm specific analysis] in disrupted sample had no deaths in week of exposure so function threw and error. This will account for that.
        return None, (
            f"infeasible_fe(zero_division_in_fit, nobs={nobs}, "
            f"n_entities={n_entities}, k={k})"
        )
    return res, f"clustered({'+'.join(cluster_cols)})" # returns res (fitted panelols model result) and a readable label saying how the se's were clustered


def extract_term_stats(res, term: str) -> Dict[str, float]: # pulls one coefficient and its confidence interval out of a fitted model result to be put in latex later
    if res is None:
        return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    if term not in res.params.index:
        return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    coef = float(res.params[term])
    ci = res.conf_int()
    ci_lo, ci_hi = ci.loc[term].tolist()
    return {"coef": coef, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}


# ... Latex builders ...
# These functions assist with taking the outputs from the model (e.g., coef, CI, etc) and put's it in a file that can be used to create a table via latex. See latex files for more details

def filter_storm_rows(df_wide: pd.DataFrame) -> pd.DataFrame: # filters and formats the storm-specific output table for appendix.
    if df_wide.empty:
        return df_wide

    out = df_wide.copy()
    out["storm_year"] = out["storm_id"].apply(_parse_storm_year)

    keep = out["storm_id"].apply(storm_is_kept)

    out = out[keep].copy()
    out["storm_display"] = out["storm_id"].astype(str)
    out = out.sort_values(["storm_year", "storm_id"], kind="mergesort").reset_index(drop=True) # sorts the rows chronologically
    return out


def build_latex_body_pooled(df_wide: pd.DataFrame) -> str:
    if df_wide.empty:
        return ""

    r = df_wide.iloc[0].to_dict()

    def get_eff(rowdict, short: str) -> str:
        coef = pd.to_numeric(rowdict.get(f"{short}_coef", np.nan), errors="coerce")
        lo = pd.to_numeric(rowdict.get(f"{short}_ci_lo", np.nan), errors="coerce")
        hi = pd.to_numeric(rowdict.get(f"{short}_ci_hi", np.nan), errors="coerce")
        return fmt_coef_ci_pp(coef, lo, hi)

    row = [
        latex_escape("Pooled"),
        fmt_int_latex(int(r.get("n_pairs", 0))),
        latex_escape(get_eff(r, "ED")),
        latex_escape(get_eff(r, "IP")),
        latex_escape(get_eff(r, "Death")),
    ]
    return " & ".join(row) + r" \\" + "\n"


def build_latex_body_storms(df_wide: pd.DataFrame) -> str:
    if df_wide.empty:
        return ""

    lines: List[str] = []

    def get_eff(rowdict, short: str) -> str:
        coef = pd.to_numeric(rowdict.get(f"{short}_coef", np.nan), errors="coerce")
        lo = pd.to_numeric(rowdict.get(f"{short}_ci_lo", np.nan), errors="coerce")
        hi = pd.to_numeric(rowdict.get(f"{short}_ci_hi", np.nan), errors="coerce")
        return fmt_coef_ci_pp(coef, lo, hi)

    for _, rr in df_wide.iterrows():
        r = rr.to_dict()
        row = [
            latex_escape(r.get("storm_display", "NA")),
            fmt_int_latex(int(r.get("n_pairs", 0))),
            latex_escape(get_eff(r, "ED")),
            latex_escape(get_eff(r, "IP")),
            latex_escape(get_eff(r, "Death")),
        ]
        lines.append(" & ".join(row) + r" \\")

    return "\n".join(lines) + "\n"


def write_latex_full(body: str, out_path: str, note_lines: List[str]) -> None:
    full = []
    full.append(r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.3in} c *{3}{c}}")
    full.append(r"\toprule")
    full.append(
        r"& \multicolumn{1}{c}{Sample size}"
        r"& \multicolumn{1}{c}{ED visit}"
        r"& \multicolumn{1}{c}{IP admission}"
        r"& \multicolumn{1}{c}{Mortality}\\"
    )
    full.append(r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}")
    full.append(
        r"\makecell[l]{Sample} & \makecell[c]{Obs\\$N$\tnote{a}}"
        r"& \makecell[c]{Effect\\(95\% CI)\tnote{b}}"
        r"& \makecell[c]{Effect\\(95\% CI)\tnote{b}}"
        r"& \makecell[c]{Effect\\(95\% CI)\tnote{b}}\\"
    )
    full.append(r"\midrule")
    full.append(body.rstrip("\n"))
    full.append(r"\bottomrule")
    full.append(r"\end{tabular}")
    full.append("")
    full.extend(note_lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full) + "\n")


# -------------------------
# Main
# -------------------------
def main() -> None:
    # Print some info's
    print("[INFO] Using disrupted-sample analytical file (v03 aligned).")
    print(f"[INFO] week_rel kept: {REF_WEEK} and {HAZARD_WEEK}")
    print(f"[INFO] storm label column: {STORM_COL}")
    print(f"[INFO] cluster SE column: {CLUSTER_COL}")
    print(f"[INFO] reporting term: {REPORT_TERM}")
    print(f"[INFO] Restriction: {STABLE_COL}==1 and {SCHED_COL} in {{MWF, TTS}}")

    dfs = []
    for year in YEARS:
        print(f"[LOAD] {year}")
        df_y = load_year_df(year) # load analytical file
        if df_y.empty:
            continue

        df_y = restrict_to_two_weeks(df_y)
        if df_y.empty:
            continue

        df_y = restrict_to_stable_mwf_tts(df_y)
        if df_y.empty:
            continue

        dfs.append(df_y)

    if not dfs:
        raise RuntimeError("No disrupted-sample data loaded after pairing + stable-schedule filter.")

    df_all = pd.concat(dfs, ignore_index=True)

    # ... Merge CKD event-level flags and restrict to CKD == 1 ...
    ckd_df = pd.read_csv(CKD_FLAG_FILE, low_memory=False) # CKD flags created from the pipeline that created exh2 tab1 (pt characteristics table)

    for c in [BENE_COL, STORM_COL, EVENT_COL, CLUSTER_COL]:
        if c in ckd_df.columns:
            ckd_df[c] = _clean_str_series(ckd_df[c])
        if c in df_all.columns:
            df_all[c] = _clean_str_series(df_all[c])

    if "facility_county_fips" in ckd_df.columns:
        ckd_df["facility_county_fips"] = _clean_str_series(ckd_df["facility_county_fips"])
    if "facility_county_fips" in df_all.columns:
        df_all["facility_county_fips"] = _clean_str_series(df_all["facility_county_fips"])

    if "year" in ckd_df.columns:
        ckd_df["year"] = pd.to_numeric(ckd_df["year"], errors="coerce").astype("Int64")

    ckd_df["ckd_ind"] = pd.to_numeric(ckd_df["ckd_ind"], errors="coerce")

    merge_keys = ["year", STORM_COL, EVENT_COL, BENE_COL, CLUSTER_COL] # def cols to merge
    if "facility_county_fips" in df_all.columns and "facility_county_fips" in ckd_df.columns: # use this instead of both files (analytical and ckd file) has it present
        merge_keys = ["year", STORM_COL, EVENT_COL, BENE_COL, "facility_county_fips", CLUSTER_COL]

    ckd_keep = ckd_df[merge_keys + ["ckd_ind"]].drop_duplicates().copy() # Keeps only needed CKD columns and removes duplicate rows.

    before_merge_n = len(df_all)
    df_all = df_all.merge(
        ckd_keep,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    ) # Merges CKD flags onto the analytical file. validate="many_to_one" means each analytical row can match one CKD row, but not multiple CKD rows.

    # Checks
    print(f"[INFO] Rows before CKD merge = {before_merge_n:,}")
    print(f"[INFO] Rows after CKD merge  = {len(df_all):,}")
    print("[QC] ckd_ind after merge:")
    print(df_all["ckd_ind"].value_counts(dropna=False).sort_index())

    before_ckd_n = len(df_all) # count before
    df_all = df_all[df_all["ckd_ind"] == 1].copy() # Keeps only CKD bene's
    print(f"[INFO] kept {len(df_all):,} CKD rows; dropped {before_ckd_n - len(df_all):,}")

    n_unique_ckd_benes_earlyA = (
        df_all.loc[
            (df_all[WEEK_COL] == HAZARD_WEEK) & (df_all[EARLYA_COL] == 1),
            BENE_COL,
        ]
        .dropna()
        .nunique()
    ) # Counts unique CKD beneficiaries across all storms with early == 1 in exposure week.
    
    print(
        f"[INFO] CKD-only: unique beneficiaries across all storms with early dialysis ({EARLYA_COL}=1) = "
        f"{n_unique_ckd_benes_earlyA:,}"
    )

    # ... more QCs ...
    miss_sid = float(df_all[STORM_COL].replace("nan", np.nan).isna().mean()) # Checks missing storm id.
    print(f"[QC] storm_id missing rate: {miss_sid:.3f}")
    print("[QC] storm_id top values:\n", df_all[STORM_COL].value_counts(dropna=False).head(15))

    hw_by_week = df_all.groupby(WEEK_COL)[HAZ_COL].mean().to_dict() # check whether exposure week avgs is 0 in week -2 and 1 in week 0.
    print(f"[QC] mean({HAZ_COL}) by {WEEK_COL}: {hw_by_week}  (expect ~0 for -2 and ~1 for 0)")

    ea_by_week = df_all.groupby(WEEK_COL)[EARLYA_COL].mean().to_dict() # check whether earlyA col is 0 in reference week and positive in exposure week
    print(
        f"[QC] mean({EARLYA_COL}) by {WEEK_COL}: {ea_by_week}  "
        "(expect 0 for -2 because builder forces it to 0 on reference row; >0 possible for 0)"
    )

    pair_n_all = int(count_paired_event_bene_units(df_all)) # count final paired bene in the pooled CKD-only data.
    print(f"[QC] paired ({EVENT_COL}, {BENE_COL}) units in pooled file: {pair_n_all:,}")

    # ... POOLED TABLE (table 2) ...
    pooled_row = { # start one row for the pooled results table.
        "row_type": "POOLED",
        "storm_display": "Pooled",
        "storm_id": np.nan,
        "storm_year": np.nan,
        "n_pairs": pair_n_all,
        "n_rows": int(len(df_all)),
        "n_bene": int(df_all[BENE_COL].nunique()),
    }

    pooled_long_rows = []
    for y_col, y_short in OUTCOMES_ORDER: # loop over outcomes: ed, ip, and mort.
        res, status = run_fe_panelols(df_all, y_col, cluster_cols=[CLUSTER_COL, STORM_COL]) # Runs the pooled fixed-effects model. SEs are clustered by facility and storm.
        stats = extract_term_stats(res, REPORT_TERM) # extract the coef and ci on early variable

        # Store the coef, ci, and se type in the wide pooled row.
        pooled_row[f"{y_short}_coef"] = stats["coef"]
        pooled_row[f"{y_short}_ci_lo"] = stats["ci_lo"]
        pooled_row[f"{y_short}_ci_hi"] = stats["ci_hi"]
        pooled_row[f"{y_short}_se_type"] = status

        pooled_long_rows.append({ # Also store the same result in long format, one row per outcome.
            "row_type": "POOLED",
            "storm_display": "Pooled",
            "storm_id": np.nan,
            "storm_year": np.nan,
            "outcome": y_short,
            "term": REPORT_TERM,
            "coef": stats["coef"],
            "ci_lo": stats["ci_lo"],
            "ci_hi": stats["ci_hi"],
            "se_type": status,
            "n_pairs": pooled_row["n_pairs"],
            "n_rows": pooled_row["n_rows"],
            "n_bene": pooled_row["n_bene"],
        })

    df_pooled_wide = pd.DataFrame([pooled_row]) # create a one-row wide pooled results table.
    for short in ["ED", "IP", "Death"]: # create formatted display columns like: 1.2 [0.3, 2.1]
        df_pooled_wide[f"{short}_effect_pp"] = df_pooled_wide.apply(
            lambda r: fmt_coef_ci_pp(
                pd.to_numeric(r.get(f"{short}_coef", np.nan), errors="coerce"),
                pd.to_numeric(r.get(f"{short}_ci_lo", np.nan), errors="coerce"),
                pd.to_numeric(r.get(f"{short}_ci_hi", np.nan), errors="coerce"),
            ),
            axis=1,
        )

    # Export
    df_pooled_long = pd.DataFrame(pooled_long_rows)
    df_pooled_wide.to_csv(OUT_POOLED_WIDE, index=False)
    print(f"[OK] Saved pooled wide CSV:\n  {OUT_POOLED_WIDE}")
    if not df_pooled_long.empty:
        df_pooled_long.to_csv(OUT_POOLED_LONG, index=False)
        print(f"[OK] Saved pooled long CSV:\n  {OUT_POOLED_LONG}")

    pooled_body = build_latex_body_pooled(df_pooled_wide) # build the latex table body.
    with open(OUT_POOLED_TEX_BODY, "w", encoding="utf-8") as f:
        f.write(pooled_body)
    print(f"Wrote pooled LaTeX body:\n  {OUT_POOLED_TEX_BODY}")

    pooled_notes = [
        r"% Note a: Obs N counts paired (event_id, BENE_ID) units with both week_rel=-2 and week_rel=0.",
        r"% Note b: Effects reported in percentage points (coef and CI multiplied by 100).",
        r"% Note c: Reported effect is the coefficient on earlyA_last_pre_offschedule from y ~ hazard_week + earlyA + FE(BENE_ID).",
        r"% Current builder alignment: week_rel=-2 is reference and week_rel=0 is hazard.",
        r"% Restriction: stable_3x_weekly==1 and schedule_type in {MWF, TTS}.",
        r"% Pooled SEs: two-way clustered by facility_id and storm_id using PanelOLS.",
        rf"% storm_id column used directly: {STORM_COL}",
        rf"% Term reported: {REPORT_TERM}",
    ]
    write_latex_full(pooled_body, OUT_POOLED_TEX_FULL, pooled_notes) # build the latex table (full).
    print(f"Wrote pooled LaTeX full tabular:\n  {OUT_POOLED_TEX_FULL}")

    # ... STORM-SPECIFIC TABLE ...
    # this is for appendix. Storm-by-storm analysis instead of the pooled
    storm_rows = []
    storm_long_rows = []

    storms_present = ( # unique non-missing storm IDs from the final CKD-only data.
        df_all[STORM_COL]
        .replace("nan", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    storms_present = [sid for sid in storms_present if storm_is_kept(sid)] # keep storms based on rule
    storms_present = sorted(storms_present, key=lambda s: (_parse_storm_year(s) or 9999, str(s))) # sort chronologically

    print(f"[INFO] Storms kept for by-storm estimation after display filter = {len(storms_present):,}")

    for sid in storms_present: # loop through one storm at a time and subsets to that storm.
        df_s = df_all[df_all[STORM_COL] == sid].copy()
        if df_s.empty:
            continue

        df_s = restrict_to_two_weeks(df_s)
        if df_s.empty:
            continue

        storm_year = _parse_storm_year(sid)
        row = {
            "row_type": "STORM",
            "storm_id": sid,
            "storm_year": storm_year,
            "storm_display": sid,
            "n_pairs": int(count_paired_event_bene_units(df_s)),
            "n_rows": int(len(df_s)),
            "n_bene": int(df_s[BENE_COL].nunique()),
        }

        for y_col, y_short in OUTCOMES_ORDER: # Runs one storm-specific FE model per outcome. Here SEs are clustered only by facility, not storm, because the subset is already one storm.
            res, status = run_fe_panelols(df_s, y_col, cluster_cols=[CLUSTER_COL])
            stats = extract_term_stats(res, REPORT_TERM) # extract coef on early

            row[f"{y_short}_coef"] = stats["coef"] # stores in wide
            row[f"{y_short}_ci_lo"] = stats["ci_lo"]
            row[f"{y_short}_ci_hi"] = stats["ci_hi"]
            row[f"{y_short}_se_type"] = status

            storm_long_rows.append({ # stores in long
                "row_type": "STORM",
                "storm_display": sid,
                "storm_id": sid,
                "storm_year": storm_year,
                "outcome": y_short,
                "term": REPORT_TERM,
                "coef": stats["coef"],
                "ci_lo": stats["ci_lo"],
                "ci_hi": stats["ci_hi"],
                "se_type": status,
                "n_pairs": row["n_pairs"],
                "n_rows": row["n_rows"],
                "n_bene": row["n_bene"],
            })

        storm_rows.append(row)

    # Create the storm wide table and applies storm filtering.
    df_storm_wide = pd.DataFrame(storm_rows)
    df_storm_wide = filter_storm_rows(df_storm_wide)

    for short in ["ED", "IP", "Death"]: # add formatted percentage-point effect.
        if not df_storm_wide.empty:
            df_storm_wide[f"{short}_effect_pp"] = df_storm_wide.apply(
                lambda r: fmt_coef_ci_pp(
                    pd.to_numeric(r.get(f"{short}_coef", np.nan), errors="coerce"),
                    pd.to_numeric(r.get(f"{short}_ci_lo", np.nan), errors="coerce"),
                    pd.to_numeric(r.get(f"{short}_ci_hi", np.nan), errors="coerce"),
                ),
                axis=1,
            )

    df_storm_long = pd.DataFrame(storm_long_rows) # create long-format storm table
    if not df_storm_long.empty and not df_storm_wide.empty: # Makes sure long-format storm output only includes storms retained in the wide table.
        keep_sids = set(df_storm_wide["storm_id"].tolist())
        df_storm_long = df_storm_long[df_storm_long["storm_id"].isin(keep_sids)].copy()

    # Export
    df_storm_wide.to_csv(OUT_STORM_WIDE, index=False)
    print(f"[OK] Saved storm-specific wide CSV:\n  {OUT_STORM_WIDE}")
    if not df_storm_long.empty:
        df_storm_long.to_csv(OUT_STORM_LONG, index=False)
        print(f"[OK] Saved storm-specific long CSV:\n  {OUT_STORM_LONG}")

    storm_body = build_latex_body_storms(df_storm_wide) # build latex body for storm table.
    with open(OUT_STORM_TEX_BODY, "w", encoding="utf-8") as f:
        f.write(storm_body)
    print(f"[OK] Wrote storm-specific LaTeX body:\n  {OUT_STORM_TEX_BODY}")

    storm_notes = [
        r"% Note a: Obs N counts paired (event_id, BENE_ID) units with both week_rel=-2 and week_rel=0 within each storm.",
        r"% Note b: Effects reported in percentage points (coef and CI multiplied by 100).",
        r"% Note c: Reported effect is the coefficient on earlyA_last_pre_offschedule from y ~ hazard_week + earlyA + FE(BENE_ID).",
        r"% Current builder alignment: week_rel=-2 is reference and week_rel=0 is hazard.",
        r"% Restriction: stable_3x_weekly==1 and schedule_type in {MWF, TTS}.",
        r"% By-storm SEs: one-way clustered by facility_id using PanelOLS.",
        r"% Storm rows that are not estimable after FE absorption are left blank rather than silently falling back to another covariance estimator.",
        rf"% storm_id column used directly: {STORM_COL}",
        rf"% Term reported: {REPORT_TERM}",
    ]
    write_latex_full(storm_body, OUT_STORM_TEX_FULL, storm_notes) # build latex (full) for storm table.
    print(f"[OK] Wrote storm-specific LaTeX full tabular:\n  {OUT_STORM_TEX_FULL}")

    # --------------------
    # Preview
    # --------------------
    print("\n[Preview] Pooled table row:") # show the preview of the table pooled analysis
    show_cols = ["row_type", "storm_display", "n_pairs", "ED_effect_pp", "IP_effect_pp", "Death_effect_pp"]
    print(df_pooled_wide[show_cols].to_string(index=False))

    print("\n[Preview] Storm-specific rows (first 15):") # show preview of storm specific analysis
    if not df_storm_wide.empty:
        print(df_storm_wide[show_cols].head(15).to_string(index=False))
    else:
        print("[INFO] No storm-specific rows after filtering.")


if __name__ == "__main__":
    main()