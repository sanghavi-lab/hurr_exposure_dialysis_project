#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 24, 2026
# Description: This script takes the two-row beneficiary analytical file (the one with paired reference-week and exposure-week)
# restricts the sample to chronic kidney disease (CKD) beneficiaries, and estimates within-beneficiary changes in disruption, 
# ED visits, IP admissions, and mortality after hurricane exposure. It produces a pooled table using two-way clustered standard 
# errors by facility and storm, plus storm-specific appendix results using one-way facility-clustered standard errors, and 
# writes to LaTeX tables
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

# -------------------------
# Paths and spec
# -------------------------
YEARS = list(range(2011, 2023))

IN_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "04_analytical_panel_hurr_exposure_v05_wkm2_facclust_cumpost_cumdeath"
) # path of analytical file

CKD_FLAG_FILE = ( # This was created by a file in exhibit 2 table 1 folder. Basically a dataset with indicators for chronic kidney disease
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "table1_early_vs_nonearly_v04_ckd_only/event_ckd_flag_export.csv"
)

REF_WEEK = -2
HAZ_WEEK = 0

CLUSTER_COL = "PRVDR_NUM_event" # this is for cluster se but, below, we will use a two way cluster se with STORM_COL
STORM_COL = "storm_id"

RESTRICT_DISRUPTION_TO_MWF_TTS = False # Include everyone with dialysis instead of restricting to strict MWF/TTS
SCHED_ALLOWED = {"MWF", "TTS"}
STABLE_COL = "stable_3x_weekly"
SCHEDULE_COL = "schedule_type"

DISRUPTION_FOOTNOTE_MARK = r"\textsuperscript{b}"
OBS_N_FOOTNOTE_MARK = r"\textsuperscript{c}"

ROUND_DECIMALS = 1 # for presentation
REPORT_IN_PERCENT_POINTS = True # multiply coef/CI by 100
Z_975 = 1.959963984540054 # 95% CI -> leaves 2.5% in each tail so use the 97.5th percentile

OUTPUT_DIR = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "paper_exhibits_step5/"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_TEX_POOLED_BODY = os.path.join(
    OUTPUT_DIR, "exhibit_step5_pooled_multihorizon_stacked_body_wkm2.tex"
)
OUT_TEX_POOLED_FULL = os.path.join(
    OUTPUT_DIR, "exhibit_step5_pooled_multihorizon_stacked_full_wkm2.tex"
)

OUT_TEX_STORM_BODY = os.path.join(
    OUTPUT_DIR, "exhibit_step5_bystorm_weekly_body_wkm2.tex"
)
OUT_TEX_STORM_FULL = os.path.join(
    OUTPUT_DIR, "exhibit_step5_bystorm_weekly_full_wkm2.tex"
)

OUT_CSV_DISRUPT = os.path.join(OUTPUT_DIR, "storm_coefs_disruption_pp_weeklyonly_wkm2.csv")
OUT_CSV_ED      = os.path.join(OUTPUT_DIR, "storm_coefs_ed_pp_weeklyonly_wkm2.csv")
OUT_CSV_IP      = os.path.join(OUTPUT_DIR, "storm_coefs_ip_pp_weeklyonly_wkm2.csv")
OUT_CSV_DEATH   = os.path.join(OUTPUT_DIR, "storm_coefs_mortality_pp_weeklyonly_wkm2.csv")


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


def fmt_coef_ci(coef: float, lo: float, hi: float) -> str: # takes a coefficient and its lower and upper confidence limits and turns them into a printable table string
    if pd.isna(coef) or pd.isna(lo) or pd.isna(hi):
        return ""
    coef_pp = _pp(coef)
    lo_pp   = _pp(lo)
    hi_pp   = _pp(hi)
    return f"{_r(coef_pp):.1f} [{_r(lo_pp):.1f}, {_r(hi_pp):.1f}]"


def fmt_int_latex(n) -> str: # formats an integer for LaTeX like add commas
    try:
        s = f"{int(n):,}"
    except Exception:
        return ""
    return s.replace(",", "{,}")


def latex_escape(s: str) -> str: # makes text safe to print in LaTex
    if s is None:
        return ""
    s = str(s)
    return (s.replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#")
            .replace("$", r"\$")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("~", r"\textasciitilde{}")
            .replace("^", r"\textasciicircum{}"))


# ... Cleaning/Processing ...
def _clean_str_series(s: pd.Series) -> pd.Series: # cleaning
    s = s.astype(str).str.replace(r"\.0$", "", regex=True) # removes a trailing .0 at the end of a string.
    s = s.replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return s


def load_step4_year(year: int) -> pd.DataFrame: # loads the analytical file created prior
    csv_path = os.path.join(IN_BASE, f"year_{year}", "analytical_panel.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] missing: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["year"] = year

    for c in ["BENE_ID", "storm_id", "fips", "event_id"]:
        if c in df.columns:
            df[c] = _clean_str_series(df[c])

    if CLUSTER_COL in df.columns:
        df[CLUSTER_COL] = _clean_str_series(df[CLUSTER_COL]).str.zfill(6)

    if "week_rel" in df.columns:
        df = df[df["week_rel"].isin([REF_WEEK, HAZ_WEEK])].copy() # keep only reference week and week of exposure

    cols_to_numeric = [
        "hazard_week", "disrupt",
        "any_ed", "any_ip", "any_death",
        "disrupt_cmp_wk", "disrupt_cmp_2wk", "disrupt_cmp_3wk", "disrupt_cmp_4wk",
        "any_ed_cmp_wk", "any_ed_cmp_2wk", "any_ed_cmp_3wk", "any_ed_cmp_4wk",
        "any_ip_cmp_wk", "any_ip_cmp_2wk", "any_ip_cmp_3wk", "any_ip_cmp_4wk",
        "any_death_cmp_wk", "any_death_cmp_2wk", "any_death_cmp_3wk", "any_death_cmp_4wk",
    ]
    for c in cols_to_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if STABLE_COL in df.columns:
        df[STABLE_COL] = pd.to_numeric(df[STABLE_COL], errors="coerce").fillna(0).astype(int)
    if SCHEDULE_COL in df.columns:
        df[SCHEDULE_COL] = _clean_str_series(df[SCHEDULE_COL])

    return df


def restrict_to_two_weeks(df: pd.DataFrame) -> pd.DataFrame: # keeps only bene that have both rows in the long panel: week -2 and week 0
    needed = {"event_id", "BENE_ID", "week_rel"} # creates a set of column names that must be present for the function to work
    miss = needed - set(df.columns) # calculates which of those required columns are missing from the DataFrame.
    if miss: # if any required columns are missing, the function raises an error and names the missing columns.
        raise KeyError(f"Missing columns for pairing: {sorted(miss)}")

    counts = (
        df.groupby(["event_id", "BENE_ID", "week_rel"])
          .size()
          .unstack(fill_value=0)
    )
    has_ref = counts.get(REF_WEEK, 0) > 0 # creates a boolean indicator for whether each pair has at least one reference-week row, meaning week_rel = -2
    has_haz = counts.get(HAZ_WEEK, 0) > 0 # creates a boolean indicator for whether each pair has at least one hazard-week row, meaning week_rel = 0
    keep_pairs = counts.index[has_ref & has_haz] # This keeps only the index entries where both conditions are true: has week -2 has week 0

    keep_df = pd.DataFrame(list(keep_pairs), columns=["event_id", "BENE_ID"]) # This converts those valid pairs into a small DataFrame with two columns: event_id and bene id
    return df.merge(keep_df, on=["event_id", "BENE_ID"], how="inner")


def restrict_disruption_sample(df: pd.DataFrame) -> pd.DataFrame:
    if not RESTRICT_DISRUPTION_TO_MWF_TTS: # This function is only used if we want to keep MWF/TTS only.
        return df

    required = {STABLE_COL, SCHEDULE_COL}
    if not required.issubset(df.columns):
        return df.iloc[0:0].copy()

    d = df[(df[STABLE_COL] == 1) & (df[SCHEDULE_COL].isin(list(SCHED_ALLOWED)))].copy()
    if d.empty:
        return d

    d = restrict_to_two_weeks(d)
    return d


def n_paired_bene_storm(df: pd.DataFrame) -> int: # returns a count of unique beneficiary-storm combinations
    if df.empty:
        return 0
    required = {"BENE_ID", STORM_COL}
    if not required.issubset(df.columns):
        return 0
    return int(df[["BENE_ID", STORM_COL]].drop_duplicates().shape[0])


# ... Modeling ...
def _cat_codes(series: pd.Series) -> np.ndarray: # converts a pandas Series into numeric category codes becuase, for example, the the two-way clustering function needs cluster id's in numeric-code form.
    s = series.astype("category")
    codes = s.cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError("Found missing values when generating categorical codes.")
    return codes.astype(np.int64, copy=False)


def _ci_from_cov(params: pd.Series, cov: np.ndarray, term: str) -> dict: # create a coefficient and 95% confidence interval from a covariance matrix. This is important to account for the uncertainty of our estimates. The covariance matrix is a structured way to store the standard error for every coefficient in the model at once.
    if term not in params.index:
        return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    idx = list(params.index).index(term) # find the position of the term inside the coefficient list.
    coef = float(params.iloc[idx]) # get the coefficient estimate.
    se = float(np.sqrt(cov[idx, idx])) if cov is not None else np.nan # get the standard error. The diagonal of the covariance matrix gives variances. Standard error is: standard error = square root of variance
    if pd.isna(se) or se <= 0:
        return {"coef": coef, "ci_lo": np.nan, "ci_hi": np.nan} # if the se is missing or invalid, return the coefficient but no ci.

    lo = coef - Z_975 * se # creates the 95% CI
    hi = coef + Z_975 * se
    return {"coef": coef, "ci_lo": float(lo), "ci_hi": float(hi)}


def fit_ols_cluster_one_or_two_way( # runs the OLS model
    y: pd.Series,
    X: pd.DataFrame,
    g1: pd.Series, # first clustering variable (facility)
    g2: pd.Series | None = None, # second clustering variable (storm)
    add_const: bool = False,
):
    y = pd.to_numeric(y, errors="coerce")
    X = X.copy().apply(pd.to_numeric, errors="coerce")

    if add_const: # this is set to false because after demeaning, an intercept is not needed. Intercept captures baseline
        X = sm.add_constant(X, has_constant="add")

    valid = y.notna() & ~X.isna().any(axis=1) & g1.notna() # ensure rows where outcome, predictors, and first cluster are not missing
    if g2 is not None: # if using two-way clustering, also require the second cluster to be not missing.
        valid = valid & g2.notna()

    if valid.sum() == 0 or X.shape[1] == 0: # stop if no valid rows
        return None, None, "no_valid_rows"

    # Subset to valid rows
    y2 = y.loc[valid]
    X2 = X.loc[valid]
    g1v = g1.loc[valid]

    model = sm.OLS(y2, X2) # create the OLS model.
    try:
        res_classic = model.fit() # fits regular OLS first to get the coef estimates. Note that there is no way to do a two way cluster with model.fit(). I get an error like RuntimeError: Pooled disruption (week) was not estimated with clustering: twoway_cluster_fit_error(TypeError). Thus, this will be used to get the coefficient but getting the correct SE will require using the function "cov_cluster_2groups"
    except Exception as e:
        return None, None, f"ols_fit_error({type(e).__name__})"

    n_g1 = int(pd.Series(g1v).nunique(dropna=True)) # count how many unique facilities there are.
    if n_g1 < 2: # if less than two then cannot perform cluster se
        return None, None, f"insufficient_clusters({CLUSTER_COL}={n_g1})"

    # This is for if we want one-way clustering (for storms specific anlysis primarily)
    if g2 is None:
        try:
            g1s = g1v.astype(str)
            res_c = model.fit(cov_type="cluster", cov_kwds={"groups": g1s, "use_correction": True}) # fits the model with clustered SEs by facility
            return res_c, None, f"cluster({CLUSTER_COL})"
        except Exception as e:
            return None, None, f"cluster_fit_error({type(e).__name__})"

    # This is for two-way clustering (pooled analysis)
    g2v = g2.loc[valid] # get the second cluster for valid rows (basically not missing)
    n_g2 = int(pd.Series(g2v).nunique(dropna=True)) # count
    if n_g2 < 2: # if count less than two then cannot perform cluster se
        return None, None, f"insufficient_clusters({STORM_COL}={n_g2})"

    try:
        g1_codes = _cat_codes(g1v) # convert facility and storm clusters into numeric codes.
        g2_codes = _cat_codes(g2v)
        cov_both, cov_g1, cov_g2 = cov_cluster_2groups(res_classic, g1_codes, g2_codes) # compute the two-way clustered covariance matrix. This is the part that accounts for correlation within same facility same storm
        cov = np.asarray(cov_both) # store the two-way covariance matrix
        return res_classic, cov, f"cluster({CLUSTER_COL}+{STORM_COL})" # return OLS coeff, the computed two-way covariance matrix, and label
    except Exception as e:
        return None, None, f"twoway_cluster_error({type(e).__name__})" # Return error instead of try doesn't work (I checked and it worked. No error)


def run_within_bene_demean(
    df: pd.DataFrame,
    y_col: str,
    *,
    use_two_way_if_possible: bool,
):
    required = ["BENE_ID", "hazard_week", y_col] # cols required
    miss = [c for c in required if c not in df.columns]
    if miss:
        return None, None, f"missing_cols({','.join(miss)})"

    d = df.dropna(subset=["BENE_ID", "hazard_week", y_col]).copy()
    if d.empty:
        return None, None, "no_data_after_dropna"

    d["hazard_week"] = pd.to_numeric(d["hazard_week"], errors="coerce").astype(float)
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce").astype(float)
    d = d.dropna(subset=["hazard_week", y_col])
    if d.empty:
        return None, None, "no_data_after_numeric"

    # Demean (more computationally efficient than fitting a bunch of FE. We also don't care about the group effects (only slope coef) so no FE needed. However, I checked and the results are the same either way)
    g = d.groupby("BENE_ID", sort=False) # group the data by beneficiary
    y_dm = d[y_col] - g[y_col].transform("mean") # demean the outcome within each beneficiary.
    x_dm = d["hazard_week"] - g["hazard_week"].transform("mean") # demean the exposure indicator within each beneficiary.

    if np.isclose(x_dm.var(ddof=0), 0.0): # basicaly, a check that shows if exposure week does not vary within beneficiaries, the model cannot estimate the exposure effect.
        return None, None, "no_within_variation_in_hazard"

    if CLUSTER_COL not in d.columns: # check that the facility cluster column exists.
        return None, None, f"missing_cluster_col({CLUSTER_COL})"

    X = pd.DataFrame({"hazard_week": x_dm}, index=d.index) # create the predictor matrix using demeaned exposure week.
    g1 = _clean_str_series(d[CLUSTER_COL]) # clean

    if use_two_way_if_possible: # if pooled model, use two-way clustering.
        if STORM_COL not in d.columns:
            return None, None, f"missing_cluster_col({STORM_COL})"
        g2 = _clean_str_series(d[STORM_COL])
        res, cov, status = fit_ols_cluster_one_or_two_way(y_dm, X, g1=g1, g2=g2, add_const=False) # run OLS using demeaned outcome ~ demeaned hazard_week. See function above for more info
        return res, cov, status

    res, cov, status = fit_ols_cluster_one_or_two_way(y_dm, X, g1=g1, g2=None, add_const=False) # if just doing one way cluster fe. Inititally used to test and compare results and for storm specific analysis
    return res, cov, status


def extract_stats(res, cov: np.ndarray | None, term="hazard_week"): # pulls coefficient and its confidence interval out of a fitted model result to be put in latex later
    if res is None or getattr(res, "params", None) is None:
        return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    params = res.params
    if cov is None:
        if term not in params.index:
            return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
        coef = float(params[term])
        try:
            ci_lo, ci_hi = res.conf_int().loc[term].tolist()
            return {"coef": coef, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}
        except Exception:
            return {"coef": coef, "ci_lo": np.nan, "ci_hi": np.nan}

    out = _ci_from_cov(params, cov, term)
    return {"coef": out["coef"], "ci_lo": out["ci_lo"], "ci_hi": out["ci_hi"]}


# ... Other model helpers ...
def _single_result(df_block: pd.DataFrame, y_col: str, *, pooled: bool) -> dict: # used to run one model for one outcome
    res, cov, se_status = run_within_bene_demean(
        df_block, y_col, use_two_way_if_possible=pooled
    ) # run the within-bene model.
    stats = extract_stats(res, cov, "hazard_week") # extract the coef and correct 95% CI

    n_obs_used = int( # count rows.
        df_block[[y_col, "hazard_week", "BENE_ID"]].dropna().shape[0]
    ) if {"BENE_ID", "hazard_week", y_col}.issubset(df_block.columns) else int(len(df_block))
    n_benes_used = int(df_block["BENE_ID"].nunique()) if "BENE_ID" in df_block.columns else np.nan # count unique bene

    return { # format the result for the table
        "effect": fmt_coef_ci(stats["coef"], stats["ci_lo"], stats["ci_hi"]),
        "se_type": se_status,
        "coef": stats["coef"],
        "ci_lo": stats["ci_lo"],
        "ci_hi": stats["ci_hi"],
        "coef_pp": _pp(stats["coef"]),
        "ci_lo_pp": _pp(stats["ci_lo"]),
        "ci_hi_pp": _pp(stats["ci_hi"]),
        "n_obs_used": n_obs_used,
        "n_benes_used": n_benes_used,
    }


def _require_clustered_result(result: dict, label: str): # check that a model actually used clustered se's
    status = result.get("se_type", "") # get the se status
    if not isinstance(status, str) or not status.startswith("cluster("):
        raise RuntimeError(f"{label} was not estimated with clustering: {status}") # If clustering failed, stop the script. This is a safety check. It prevents silently reporting non-clustered results.


def compute_stacked_pooled_results(df_all: pd.DataFrame) -> list[dict]: # create the pooled table rows
    rows = []

    d_dis = df_all # use the full sample
    if RESTRICT_DISRUPTION_TO_MWF_TTS: # only restrict if true
        d_dis = restrict_disruption_sample(df_all)

    first_row = { # create the first table row comparing week -2 to week 0.
        "comparison": "Week -2 vs week 0",
        "n_obs": n_paired_bene_storm(df_all),
    }

    if d_dis.empty: # if no disruption data, leave blank
        first_row["Disruption"] = ""
    else:
        r_dis = _single_result(d_dis, "disrupt_cmp_wk", pooled=True) # run pooled model for weekly disruption.
        _require_clustered_result(r_dis, "Pooled disruption (week)") # confirm if clustered se's were used.
        eff = r_dis["effect"] # get formatted effect.
        if eff and RESTRICT_DISRUPTION_TO_MWF_TTS: # add footnote if disruption sample was restricted to mwf/tts.
            eff = eff + DISRUPTION_FOOTNOTE_MARK
        first_row["Disruption"] = eff # add disruption estimate to table row.

    # Run pooled models for ED, IP, and death (same process)
    r_ed = _single_result(df_all, "any_ed_cmp_wk", pooled=True)
    r_ip = _single_result(df_all, "any_ip_cmp_wk", pooled=True)
    r_death = _single_result(df_all, "any_death_cmp_wk", pooled=True)
    _require_clustered_result(r_ed, "Pooled ED (week)")
    _require_clustered_result(r_ip, "Pooled IP (week)")
    _require_clustered_result(r_death, "Pooled death (week)")
    first_row["ED"] = r_ed["effect"]
    first_row["IP"] = r_ip["effect"]
    first_row["Death"] = r_death["effect"]
    rows.append(first_row) # add estimates to the first table row.

    # Cumulative windows
    cumulative_specs = [
        ("Week -2 vs cumulative weeks 0--1", "disrupt_cmp_2wk", "any_ed_cmp_2wk", "any_ip_cmp_2wk", "any_death_cmp_2wk"),
        ("Week -2 vs cumulative weeks 0--2", "disrupt_cmp_3wk", "any_ed_cmp_3wk", "any_ip_cmp_3wk", "any_death_cmp_3wk"),
        ("Week -2 vs cumulative weeks 0--3", "disrupt_cmp_4wk", "any_ed_cmp_4wk", "any_ip_cmp_4wk", "any_death_cmp_4wk"),
    ]

    for label, dis_col, ed_col, ip_col, death_col in cumulative_specs:
        row = { # create an empty row.
            "comparison": label,
            "n_obs": "",
            "Disruption": "",
            "ED": "",
            "IP": "",
            "Death": "",
        }

        # Run models for cumulative ED, IP, and death (same process as weekly)
        r_ed = _single_result(df_all, ed_col, pooled=True)
        r_ip = _single_result(df_all, ip_col, pooled=True)
        r_death = _single_result(df_all, death_col, pooled=True)
        _require_clustered_result(r_ed, f"Pooled ED ({label})")
        _require_clustered_result(r_ip, f"Pooled IP ({label})")
        _require_clustered_result(r_death, f"Pooled death ({label})")
        row["ED"] = r_ed["effect"]
        row["IP"] = r_ip["effect"]
        row["Death"] = r_death["effect"]

        if not d_dis.empty: # run cumulative disruption model if data exist (same process as above)
            r_dis = _single_result(d_dis, dis_col, pooled=True)
            _require_clustered_result(r_dis, f"Pooled disruption ({label})")
            eff = r_dis["effect"]
            if eff and RESTRICT_DISRUPTION_TO_MWF_TTS:
                eff = eff + DISRUPTION_FOOTNOTE_MARK
            row["Disruption"] = eff

        # Notice that the disruption is its own seperate chunk of codes. This is because we were going back and forth on whether to include disruption as an outcome due to various factors.

        rows.append(row) # return all pooled table rows.

    return rows


def compute_weekly_results_block(df_block: pd.DataFrame, *, pooled: bool) -> dict: # this is for storm-specific weekly models.
    results = {}

    weekly_map = { # map table labels to outcome columns
        "ED": "any_ed_cmp_wk",
        "IP": "any_ip_cmp_wk",
        "Death": "any_death_cmp_wk",
    }
    for outcome_short, y_col in weekly_map.items(): # run one model each for ED, IP, and mort.
        results[outcome_short] = _single_result(df_block, y_col, pooled=pooled)

    d_dis = df_block # use the same df for disruption.
    if RESTRICT_DISRUPTION_TO_MWF_TTS:
        d_dis = restrict_disruption_sample(df_block)

    if d_dis.empty:
        results["Disruption"] = { # store a blank disruption result if empty
            "effect": "",
            "se_type": "no_data_after_restriction",
            "coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "coef_pp": np.nan, "ci_lo_pp": np.nan, "ci_hi_pp": np.nan,
            "n_obs_used": 0,
            "n_benes_used": 0,
        }
    else: # otherwise, runs disruption model.
        r = _single_result(d_dis, "disrupt_cmp_wk", pooled=pooled) # model
        eff = r["effect"] # get formatted result.
        if eff and RESTRICT_DISRUPTION_TO_MWF_TTS:
            eff = eff + DISRUPTION_FOOTNOTE_MARK # add footnote if needed.
        r["effect"] = eff # store disruption result.
        results["Disruption"] = r

    return results


# ... LaTeX ...
# These functions just take the outputs from the model (e.g., coef, CI, etc) and put's it in a file that can be used to create a table via latex. See latex files for more details
    
def build_pooled_stacked_body(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        line = [
            latex_escape(row.get("comparison", "")),
            fmt_int_latex(row["n_obs"]) if isinstance(row.get("n_obs"), (int, np.integer)) else latex_escape(row.get("n_obs", "")),
            latex_escape(row.get("Disruption", "")),
            latex_escape(row.get("ED", "")),
            latex_escape(row.get("IP", "")),
            latex_escape(row.get("Death", "")),
        ]
        lines.append(" & ".join(line) + r" \\")
    return "\n".join(lines) + "\n"


def build_storm_row(sample_label: str, n_obs: int, results_by_short: dict) -> str:
    row = [
        latex_escape(sample_label),
        fmt_int_latex(n_obs),
        latex_escape(results_by_short.get("Disruption", {}).get("effect", "")),
        latex_escape(results_by_short.get("ED", {}).get("effect", "")),
        latex_escape(results_by_short.get("IP", {}).get("effect", "")),
        latex_escape(results_by_short.get("Death", {}).get("effect", "")),
    ]
    return " & ".join(row) + r" \\\n"


def write_pooled_tex_files(body: str):
    with open(OUT_TEX_POOLED_BODY, "w", encoding="utf-8") as f:
        f.write(body)

    full = []
    full.append(r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.0in} c *{4}{c}}")
    full.append(r"\toprule")
    full.append(
        r"& \multicolumn{1}{c}{Sample size}"
        r"& \multicolumn{1}{c}{Dialysis disruption}"
        r"& \multicolumn{1}{c}{ED visit}"
        r"& \multicolumn{1}{c}{IP admission}"
        r"& \multicolumn{1}{c}{Mortality}\\"
    )
    full.append(r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}")
    full.append(
        r"\makecell[l]{Comparison window}"
        r" & \makecell[c]{Obs\\$N$" + OBS_N_FOOTNOTE_MARK + r"}"
        r" & \makecell[c]{Effect\\(95\% CI)\tnote{a}}"
        r" & \makecell[c]{Effect\\(95\% CI)\tnote{a}}"
        r" & \makecell[c]{Effect\\(95\% CI)\tnote{a}}"
        r" & \makecell[c]{Effect\\(95\% CI)\tnote{a}}\\"
    )
    full.append(r"\midrule")
    full.append(body.rstrip("\n"))
    full.append(r"\bottomrule")
    full.append(r"\end{tabular}")
    full.append("")
    full.append(r"% Note a: Effects reported in percentage points (coef and CI multiplied by 100).")
    full.append(r"% Note b (manual): Disruption estimated on stable MWF/TTS only (stable_3x_weekly==1 and schedule_type in {MWF,TTS}).")
    full.append(rf"% Note c: Obs N is the number of paired bene--storm units (unique (BENE_ID, storm_id)) after enforcing both weeks ({REF_WEEK} and {HAZ_WEEK}). Obs N is shown only on the first pooled row because it is repeated across horizons.")
    full.append(rf"% Pooled table uses two-way clustered SEs at ({CLUSTER_COL}, {STORM_COL}).")
    full.append(rf"% First row compares week {REF_WEEK} vs week {HAZ_WEEK}.")
    full.append(rf"% Second row compares week {REF_WEEK} vs any outcome in post weeks 0-1.")
    full.append(rf"% Third row compares week {REF_WEEK} vs any outcome in post weeks 0-2.")
    full.append(rf"% Fourth row compares week {REF_WEEK} vs any outcome in post weeks 0-3.")
    full.append(rf"% Mortality rows use cumulative death-by-window indicators from week 0 onward.")
    full.append(rf"% Disruption rows use cumulative any-disruption-by-window indicators from week 0 onward.")
    full.append(rf"% Beneficiaries who died before exposure_start_dt were already excluded in Step 4.")
    full.append(r"% No fallback covariance estimator is used in this script.")
    full.append(r"% Blank cells indicate an estimate was not computed because the required clustering structure was unavailable.")

    with open(OUT_TEX_POOLED_FULL, "w", encoding="utf-8") as f:
        f.write("\n".join(full) + "\n")

    print(f"[OK] Wrote pooled TEX:\n  {OUT_TEX_POOLED_BODY}\n  {OUT_TEX_POOLED_FULL}")


def write_storm_tex_files(body: str):
    with open(OUT_TEX_STORM_BODY, "w", encoding="utf-8") as f:
        f.write(body)

    full = []
    full.append(r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.0in} c *{4}{c}}")
    full.append(r"\toprule")
    full.append(
        r"& \multicolumn{1}{c}{Sample size}"
        r"& \multicolumn{1}{c}{Dialysis disruption}"
        r"& \multicolumn{1}{c}{ED visit}"
        r"& \multicolumn{1}{c}{IP admission}"
        r"& \multicolumn{1}{c}{Mortality}\\"
    )
    full.append(r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}")
    full.append(
        r"\makecell[l]{Sample} & \makecell[c]{Obs\\$N$" + OBS_N_FOOTNOTE_MARK + r"}"
        r"& \makecell[c]{Effect\\(95\% CI)\tnote{a}}"
        r"& \makecell[c]{Effect\\(95\% CI)\tnote{a}}"
        r"& \makecell[c]{Effect\\(95\% CI)\tnote{a}}"
        r"& \makecell[c]{Effect\\(95\% CI)\tnote{a}}\\"
    )
    full.append(r"\midrule")
    full.append(body.rstrip("\n"))
    full.append(r"\bottomrule")
    full.append(r"\end{tabular}")
    full.append("")
    full.append(r"% Note a: Effects reported in percentage points (coef and CI multiplied by 100).")
    full.append(r"% Note b (manual): Disruption estimated on stable MWF/TTS only (stable_3x_weekly==1 and schedule_type in {MWF,TTS}).")
    full.append(rf"% Note c: Obs N is the number of paired bene--storm units (unique (BENE_ID, storm_id)) after enforcing both weeks ({REF_WEEK} and {HAZ_WEEK}).")
    full.append(rf"% Storm-specific table uses one-way clustered SEs at {CLUSTER_COL}.")
    full.append(rf"% Storm-specific estimates compare week {REF_WEEK} with week {HAZ_WEEK} only.")
    full.append(rf"% Weekly mortality row uses any_death_cmp_wk (week -2 vs death by end of week 0).")
    full.append(rf"% Weekly disruption row uses disrupt_cmp_wk (week -2 vs week 0).")
    full.append(r"% No fallback covariance estimator is used in this script.")
    full.append(r"% Blank cells indicate an estimate was not computed because the required clustering structure was unavailable.")

    with open(OUT_TEX_STORM_FULL, "w", encoding="utf-8") as f:
        f.write("\n".join(full) + "\n")

    print(f"[OK] Wrote storm TEX:\n  {OUT_TEX_STORM_BODY}\n  {OUT_TEX_STORM_FULL}")


def get_storm_ids_chronological(df_all: pd.DataFrame) -> list[str]: # creates an ordered list of storm IDs
    date_candidates = [
        "storm_start_date", "storm_peak_date", "storm_date",
        "event_date", "week_start", "week_start_date",
        "hazard_week_date"
    ]
    date_col = next((c for c in date_candidates if c in df_all.columns), None) # look for the first date-like column that exists in df_all.

    if date_col is not None: # if a date column exists...
        d0 = df_all[df_all["week_rel"] == HAZ_WEEK].copy() # focus on the exposure week rows, where haz_week =0. Do not need ref and exposure week
        if d0.empty: # if no hazard week, the use all
            d0 = df_all.copy()

        d0[date_col] = pd.to_datetime(d0[date_col], errors="coerce")
        order = ( # for each storm, it finds the earliest date and sorts storms by that date.
            d0.groupby("storm_id")[date_col]
              .min()
              .sort_values(kind="mergesort")
        )
        return order.index.tolist() + [s for s in df_all["storm_id"].dropna().unique() if s not in order.index]

    if "storm_year" in df_all.columns: # If no date column exists...
        order = ( # group by storm and sort by the earliest storm year
            df_all.groupby("storm_id")["storm_year"]
                  .min()
                  .sort_values(kind="mergesort")
        )
        return order.index.tolist()

    # Pull a 4-digit year from the storm name/string
    tmp = pd.DataFrame({"storm_id": df_all["storm_id"].dropna().unique()})
    tmp["yr"] = pd.to_numeric(tmp["storm_id"].str.extract(r"(\d{4})")[0], errors="coerce")
    tmp = tmp.sort_values(["yr", "storm_id"], kind="mergesort") # storms are ordered by year, then alphabetically by storm ID.
    return tmp["storm_id"].tolist()


# -------------------------
# Main
# -------------------------
def main():
    dfs = []
    for y in YEARS:
        df_y = load_step4_year(y)
        if df_y.empty:
            continue
        df_y = restrict_to_two_weeks(df_y)
        if df_y.empty:
            continue
        dfs.append(df_y)

    if not dfs:
        raise RuntimeError("No Step 4 data loaded after pairing filter.")

    df_all = pd.concat(dfs, ignore_index=True)

    # ... Merge CKD event-level flags and restrict to CKD == 1 ...
    ckd_df = pd.read_csv(CKD_FLAG_FILE, low_memory=False) # CKD flags created from the pipeline that created exh2 tab1 (pt characteristics table)

    for c in ["BENE_ID", "storm_id", "fips", "event_id"]:
        if c in ckd_df.columns:
            ckd_df[c] = _clean_str_series(ckd_df[c])

    if CLUSTER_COL in ckd_df.columns:
        ckd_df[CLUSTER_COL] = _clean_str_series(ckd_df[CLUSTER_COL]).str.zfill(6)

    if "year" in ckd_df.columns:
        ckd_df["year"] = pd.to_numeric(ckd_df["year"], errors="coerce").astype("Int64")

    ckd_df["ckd_ind"] = pd.to_numeric(ckd_df["ckd_ind"], errors="coerce")

    before_merge_n = len(df_all) # count

    ckd_keep = ckd_df[
        ["year", "storm_id", "event_id", "BENE_ID", "fips", CLUSTER_COL, "ckd_ind"]
    ].drop_duplicates().copy()

    df_all = df_all.merge(
        ckd_keep,
        on=["year", "storm_id", "event_id", "BENE_ID", "fips", CLUSTER_COL],
        how="left",
        validate="many_to_one"
    )

    print(f"[INFO] Rows before CKD merge = {before_merge_n:,}") # QCs
    print(f"[INFO] Rows after CKD merge  = {len(df_all):,}")
    print("[QC] ckd_ind after merge:")
    print(df_all["ckd_ind"].value_counts(dropna=False).sort_index())

    pre_ckd_unique_benes = df_all["BENE_ID"].dropna().nunique()
    pre_ckd_paired_bene_storm = n_paired_bene_storm(df_all)
    print(f"[INFO] Before CKD restriction: unique beneficiaries across all storms = {pre_ckd_unique_benes:,}")
    print(f"[INFO] Before CKD restriction: paired bene-storm N = {pre_ckd_paired_bene_storm:,}")
    print(f"[INFO] Before CKD restriction: unique storms = {df_all[STORM_COL].nunique():,}")

    before_ckd_n = len(df_all)
    df_all = df_all[df_all["ckd_ind"] == 1].copy() # Keep only CKDs

    print(f"[INFO] kept {len(df_all):,} CKD rows; dropped {before_ckd_n - len(df_all):,}")
    print(
        f"[INFO] CKD-only pooled rows={len(df_all):,} | "
        f"paired bene-storm N={n_paired_bene_storm(df_all):,} | "
        f"unique benes={df_all['BENE_ID'].nunique():,} | storms={df_all[STORM_COL].nunique():,}"
    )

    if STORM_COL not in df_all.columns: # check clustering cols
        raise KeyError(f"{STORM_COL} not found in Step 4 panel.")

    if CLUSTER_COL not in df_all.columns: # check clustering cols
        raise KeyError(f"{CLUSTER_COL} not found in Step 4 panel.")

    # Counts unique facility and missing facility id's.
    n_clusters = df_all[CLUSTER_COL].dropna().nunique()
    share_miss = df_all[CLUSTER_COL].isna().mean()
    print(f"[INFO] Pooled: unique {CLUSTER_COL}={n_clusters:,} | missing={share_miss:.3%}")

    print(
        f"[INFO] Total pooled rows={len(df_all):,} | "
        f"paired bene-storm N={n_paired_bene_storm(df_all):,} | "
        f"unique benes={df_all['BENE_ID'].nunique():,} | storms={df_all[STORM_COL].nunique():,} | "
        f"weeks={sorted(df_all['week_rel'].unique().tolist()) if 'week_rel' in df_all.columns else 'NA'}"
    )

    # ... POOLED TABLE (table 2) ...
    pooled_rows = compute_stacked_pooled_results(df_all) # run the pooled within-bene models.
    pooled_body = build_pooled_stacked_body(pooled_rows) # turn model results into latex table rows.
    write_pooled_tex_files(pooled_body) # write the pooled latex table files.

    # ... STORM-SPECIFIC TABLE ...
    # this is for appendix. Storm-by-storm analysis instead of the pooled

    # Prepares containers for storm-specific table rows and CSV output.
    lines = []
    csv_rows = {"Disruption": [], "ED": [], "IP": [], "Death": []}

    storm_ids = get_storm_ids_chronological(df_all) # get storms in chronological order.

    for sid in storm_ids: # loop through each storm
        d = df_all[df_all[STORM_COL] == sid].copy()
        if d.empty:
            continue

        d = restrict_to_two_weeks(d)
        if d.empty:
            continue

        storm_results = compute_weekly_results_block(d, pooled=False) # run storm-specific weekly models only.
        lines.append(build_storm_row(str(sid), n_paired_bene_storm(d), storm_results)) # create one latex row for that storm.

        # Used to assign a year label to the storm-specific output. (w/in storm loop)
        storm_year = np.nan
        if "storm_year" in d.columns:
            try:
                storm_year = float(pd.to_numeric(d["storm_year"], errors="coerce").dropna().min())
            except Exception:
                storm_year = np.nan

        for outcome in ["Disruption", "ED", "IP", "Death"]: # for each outcome, extract coef, CI, SE, and sample size.
            r = storm_results.get(outcome, {})
            csv_rows[outcome].append({ # store storm-specific estimates in a dictionary.
                "storm_id": str(sid),
                "storm_year": storm_year,
                "coef_pp": r.get("coef_pp", np.nan),
                "ci_lo_pp": r.get("ci_lo_pp", np.nan),
                "ci_hi_pp": r.get("ci_hi_pp", np.nan),
                "se_type": r.get("se_type", ""),
                "n_rows_block": int(len(d)),
                "n_obs_used": r.get("n_obs_used", np.nan),
                "n_benes_used": r.get("n_benes_used", np.nan),
                "note": "weekly only; pp=coef*100" if REPORT_IN_PERCENT_POINTS else "weekly only",
            })

    # Write storm latex and csv files
    storm_body = "".join(lines)
    write_storm_tex_files(storm_body)

    # Func to write csv
    def _write_csv(out_path: str, rows: list[dict]):
        out_df = pd.DataFrame(rows)
        if out_df.empty:
            out_df.to_csv(out_path, index=False)
            print(f"[OK] Wrote empty CSV: {out_path}")
            return
        if "storm_year" in out_df.columns:
            out_df = out_df.sort_values(["storm_year", "storm_id"], kind="mergesort")
        else:
            out_df = out_df.sort_values(["storm_id"], kind="mergesort")
        out_df.to_csv(out_path, index=False)
        print(f"[OK] Wrote CSV: {out_path} (n={len(out_df):,})")

    _write_csv(OUT_CSV_DISRUPT, csv_rows["Disruption"])
    _write_csv(OUT_CSV_ED,      csv_rows["ED"])
    _write_csv(OUT_CSV_IP,      csv_rows["IP"])
    _write_csv(OUT_CSV_DEATH,   csv_rows["Death"])


if __name__ == "__main__":
    main()

