#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 28, 2026
# Description: This script takes the placebo pre-period panels analytical file, keeps only those with chronic kidney 
# disease, and estimates within-beneficiary placebo effects by comparing each earlier pre-exposure week (-3, -4, -5, and 
# -6) with reference week -2. For each, it fits demeaned linear models for dialysis disruption, ED visits, and inpatient 
# admissions, uses two-way clustered standard errors by provider and storm, and writes the formatted results to a LaTeX 
# table.
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
YEAR_MIN, YEAR_MAX = 2011, 2022
YEARS = list(range(YEAR_MIN, YEAR_MAX + 1))

CKD_FLAG_FILE = ( # created from exh2 scripts directory
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "table1_early_vs_nonearly_v04_ckd_only/event_ckd_flag_export.csv"
)

IN_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "04_placebo_panel_preweeks_v01_wkm2_ref"
)

OUTPUT_DIR = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "paper_exhibits_placebo/"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_TEX_BODY = os.path.join(OUTPUT_DIR, "exhibit_placebo_preweeks_body.tex") # LaTex
OUT_TEX_FULL = os.path.join(OUTPUT_DIR, "exhibit_placebo_preweeks_full.tex")
META_OUT_CSV = os.path.join(OUTPUT_DIR, "placebo_preweeks_model_meta.csv")

REF_WEEK = -2
PLACEBO_TEST_WEEKS = [-3, -4, -5, -6]  # closest first

OUTCOMES = [ # notice no mortality here. Since we are comparing week -2 with -3, it doesn't make sense to add mortality.
    ("disrupt", "Dialysis disruption"),
    ("any_ed", "ED visit"),
    ("any_ip", "IP admission"),
]

CLUSTER1_COL = "PRVDR_NUM_event"
CLUSTER2_COL = "storm_id" # for two way cluster
MIN_CLUSTERS = 2 # min clusters (i.e., # facility needed to do a clustering SE)
FALLBACK_COV_TYPE = "HC1" # used to compare and in case two way cluster failed. However, I checked and we didn't need to use fallback

Z_975 = 1.959963984540054 # 95% CI -> leaves 2.5% in each tail so use the 97.5th percentile
ROUND_DECIMALS = 1 # presentation
REPORT_IN_PERCENT_POINTS = True # multiply coef/CI by 100

# -------------------------
# Functions
# -------------------------
def _clean_str_series(s: pd.Series) -> pd.Series: # clean
    s = s.astype(str).str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return s

def _cat_codes(series: pd.Series) -> np.ndarray: # convert a clustering variable into integer category codes so it can be used in the two-way clustered standard error calculation.
    s = series.astype("category")
    codes = s.cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError("Missing values in clustering variable.")
    return codes.astype(np.int64, copy=False)

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
    return f"{_r(_pp(coef)):.1f} [{_r(_pp(lo)):.1f}, {_r(_pp(hi)):.1f}]"

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

def load_year(y: int) -> pd.DataFrame: # read and clean
    p = os.path.join(IN_BASE, f"year_{y}", "analytical_panel_placebo_preweeks.csv")
    if not os.path.exists(p):
        print(f"[SKIP] missing: {p}")
        return pd.DataFrame()

    df = pd.read_csv(p)
    df["year"] = y

    for c in ["BENE_ID", "storm_id", "event_id", "fips", CLUSTER1_COL]:
        if c in df.columns:
            df[c] = _clean_str_series(df[c])

    if CLUSTER1_COL in df.columns:
        df[CLUSTER1_COL] = _clean_str_series(df[CLUSTER1_COL]).str.zfill(6)
    if "week_rel" in df.columns:
        df["week_rel"] = pd.to_numeric(df["week_rel"], errors="coerce").astype("Int16")

    for c, _ in OUTCOMES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df

def restrict_to_two_weeks(df: pd.DataFrame, wk_a: int, wk_b: int) -> pd.DataFrame: # take the full placebo panel and keeps only the rows for two specific weeks
    d = df[df["week_rel"].isin([wk_a, wk_b])].copy() # e.g., wk_a: one placebo week, like -3 and wk_b: the reference week, here -2
    if d.empty:
        return d

    counts = ( # each pair-week should usually have one row. These counts will be used to keep those with two rows of info
        d.groupby(["event_id", "BENE_ID", "week_rel"])
         .size()
         .unstack(fill_value=0)
    )
    has_a = counts.get(wk_a, 0) > 0 # create a Boolean indicator for whether each bene-storm (event) pair has at least one row in wk_a.
    has_b = counts.get(wk_b, 0) > 0 # create a Boolean indicator for whether each bene-storm (event) pair has at least one row in wk_b.
    keep_pairs = counts.index[has_a & has_b] # keep only those that have both weeks.
    keep_df = pd.DataFrame(list(keep_pairs), columns=["event_id", "BENE_ID"])

    return d.merge(keep_df, on=["event_id", "BENE_ID"], how="inner")

def n_paired_bene_storm(df: pd.DataFrame) -> int: # counts the number for Obs N sample size
    if df.empty or ("BENE_ID" not in df.columns) or ("storm_id" not in df.columns):
        return 0
    return int(df[["BENE_ID", "storm_id"]].drop_duplicates().shape[0])


# ... Modeling ...
# within-BENE demeaning + clustered SE (2-way)

def _ci_from_cov(params: pd.Series, cov: np.ndarray, term: str) -> dict: # create a coefficient and 95% confidence interval from a covariance matrix. This is important to account for the uncertainty of our estimates. The covariance matrix is a structured way to store the standard error for every coefficient in the model at once.
    if term not in params.index:
        return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    idx = list(params.index).index(term) # find the position of the term inside the coefficient list.
    coef = float(params.iloc[idx]) # get the coefficient estimate.
    se = float(np.sqrt(cov[idx, idx])) if cov is not None else np.nan # get the standard error. The diagonal of the covariance matrix gives variances. Standard error is: standard error = square root of variance
    if pd.isna(se) or se <= 0: # if the se is missing or invalid, return the coefficient but no ci.
        return {"coef": coef, "ci_lo": np.nan, "ci_hi": np.nan}
    return {"coef": coef, "ci_lo": coef - Z_975 * se, "ci_hi": coef + Z_975 * se} # creates the 95% CI

def run_within_bene_demean_twocluster(d: pd.DataFrame, y_col: str) -> dict:
    # Compare placebo week vs ref week using demeaned bene (within bene)
    # Two-way cluster on facility and storm
    
    req = {"BENE_ID", "fake_hazard", y_col} # In main, I created fake_hazard. Specifically, in a two-week comparison like -3 vs -2: week -3 gets fake_hazard = 1 week -2 gets fake_hazard = 0. Basically, designate one as reference week and the other as "fake" exposure
    if not req.issubset(d.columns):
        return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "se_type": "missing_cols"}

    dd2 = d.dropna(subset=["BENE_ID", "fake_hazard", y_col]).copy()
    if dd2.empty:
        return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "se_type": "no_data"}

    dd2["BENE_ID"] = _clean_str_series(dd2["BENE_ID"])
    dd2[y_col] = pd.to_numeric(dd2[y_col], errors="coerce").astype(float)
    dd2["fake_hazard"] = pd.to_numeric(dd2["fake_hazard"], errors="coerce").astype(float)
    dd2 = dd2.dropna(subset=[y_col, "fake_hazard"])
    if dd2.empty:
        return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "se_type": "no_numeric"}

    # fixed-effects transformation via demeaning (more computationally efficient and we don't care about the group level effects)
    g = dd2.groupby("BENE_ID", sort=False)
    y_dm = dd2[y_col] - g[y_col].transform("mean") # outcome minus that bene’s mean outcome across the two weeks
    x_dm = dd2["fake_hazard"] - g["fake_hazard"].transform("mean") # fake_hazard minus that person’s mean fake_hazard. Because each bene in this sample should have exactly two rows, one with fake_hazard=1 and one with 0, their mean fake_hazard is 0.5. So after demeaning: placebo-week row becomes +0.5 reference-week row becomes -0.5

    if np.isclose(x_dm.var(ddof=0), 0.0): # check whether the demeaned regressor has any variance. If x_dm is constant, the model cannot estimate a coefficient. That would happen if the sample somehow did not actually contain variation in fake_hazard.
        return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "se_type": "no_within_var"}

    X = pd.DataFrame({"fake_hazard": x_dm}, index=dd2.index) # build the reg design matrix using only the demeaned fake_hazard.
    model = sm.OLS(y_dm, X) # define an OLS model of y_dm x_dm. Notice there is no intercept. That is b/c the within transformation already removed the bene-specific mean.

    # If no provider cluster col -> HC1. Used for testing. We use the two way cluster SE
    if CLUSTER1_COL not in dd2.columns:
        res = model.fit(cov_type=FALLBACK_COV_TYPE)
        cov = np.asarray(res.cov_params())
        out = _ci_from_cov(res.params, cov, "fake_hazard")
        out["se_type"] = f"fallback_{FALLBACK_COV_TYPE}(no_{CLUSTER1_COL})"
        return out

    # Create g1, the facility clustering variable, and counts how many unique provider clusters exist.
    g1 = _clean_str_series(dd2[CLUSTER1_COL])
    n_g1 = int(pd.Series(g1).nunique(dropna=True))

    # If there are too few provider clusters, clustering is not reliable.
    if n_g1 < MIN_CLUSTERS:
        res = model.fit(cov_type=FALLBACK_COV_TYPE)
        cov = np.asarray(res.cov_params())
        out = _ci_from_cov(res.params, cov, "fake_hazard")
        out["se_type"] = f"fallback_{FALLBACK_COV_TYPE}(n_g1={n_g1})"
        return out

    # Checks whether the second cluster variable, storm_id, exists. If not, two-way clustering is impossible so fall back to one way. Function not relevant since we will do two way. Function left here for sanity checks
    if CLUSTER2_COL not in dd2.columns:
        # one-way fallback
        try:
            res_c = model.fit(cov_type="cluster", cov_kwds={"groups": g1.astype(str), "use_correction": True})
            cov = np.asarray(res_c.cov_params())
            out = _ci_from_cov(res_c.params, cov, "fake_hazard")
            out["se_type"] = f"cluster({CLUSTER1_COL})_no_{CLUSTER2_COL}"
            return out
        except Exception:
            res = model.fit(cov_type=FALLBACK_COV_TYPE)
            cov = np.asarray(res.cov_params())
            out = _ci_from_cov(res.params, cov, "fake_hazard")
            out["se_type"] = f"fallback_{FALLBACK_COV_TYPE}(cluster_fail)"
            return out

    # Another fall back to one way if not enough storm clusters but there are more than 2 hurricanes so will not fall back to one way cluster. Again, kept for sanity checks
    g2 = _clean_str_series(dd2[CLUSTER2_COL])
    n_g2 = int(pd.Series(g2).nunique(dropna=True))
    if n_g2 < MIN_CLUSTERS:
        # one-way on provider fallback
        try:
            res_c = model.fit(cov_type="cluster", cov_kwds={"groups": g1.astype(str), "use_correction": True})
            cov = np.asarray(res_c.cov_params())
            out = _ci_from_cov(res_c.params, cov, "fake_hazard")
            out["se_type"] = f"twoway_fallback_oneway(n_g2={n_g2})"
            return out
        except Exception:
            res = model.fit(cov_type=FALLBACK_COV_TYPE)
            cov = np.asarray(res.cov_params())
            out = _ci_from_cov(res.params, cov, "fake_hazard")
            out["se_type"] = f"fallback_{FALLBACK_COV_TYPE}(n_g2={n_g2})"
            return out

    # Finally, this is what we will do: two-way cluster using cov_cluster_2groups function (storm and facility)
    try:
        res_classic = model.fit() # fit OLS based on y_dm and X above
        g1_codes = _cat_codes(g1) # convert the provider (g1) and storm labels (g2) into integer category codes. That is needed because cov_cluster_2groups expects group arrays.
        g2_codes = _cat_codes(g2)
        cov = np.asarray(cov_cluster_2groups(res_classic, g1_codes, g2_codes)) # compute the two-way clustered covariance matrix using provider and storm 
        out = _ci_from_cov(res_classic.params, cov, "fake_hazard") # pull the coefficient and CI for fake_hazard
        out["se_type"] = f"cluster({CLUSTER1_COL}+{CLUSTER2_COL})" # label the SE type as two-way clustered
        return out
    except Exception as e:
        # one-way fallback. I manually checked to confirm results produced were two way. This code is not necessary but I left it as is
        try:
            res_c = model.fit(cov_type="cluster", cov_kwds={"groups": g1.astype(str), "use_correction": True})
            cov = np.asarray(res_c.cov_params())
            out = _ci_from_cov(res_c.params, cov, "fake_hazard")
            out["se_type"] = f"twoway_failed_fallback_oneway(err={type(e).__name__})"
            return out
        except Exception:
            res = model.fit(cov_type=FALLBACK_COV_TYPE)
            cov = np.asarray(res.cov_params())
            out = _ci_from_cov(res.params, cov, "fake_hazard")
            out["se_type"] = f"fallback_{FALLBACK_COV_TYPE}(twoway_fail)"
            return out

# ... LaTeX ...
# These functions just take the outputs from the model (e.g., coef, CI, etc) and put's it in a file that can be used to create a table via latex. See latex files for more details
def build_row(label: str, n_obs: int, effects: dict) -> str:
    row = [latex_escape(label), fmt_int_latex(n_obs)]
    for _, col_name in OUTCOMES:
        row.append(latex_escape(effects.get(col_name, "")))
    return " & ".join(row) + r" \\" + "\n"

def write_tex(body: str):
    with open(OUT_TEX_BODY, "w", encoding="utf-8") as f:
        f.write(body)

    full = []
    full.append(r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.6in} c *{3}{c}}")
    full.append(r"\toprule")
    full.append(
        r"& \multicolumn{1}{c}{Obs $N$}"
        r"& \multicolumn{1}{c}{Dialysis disruption}"
        r"& \multicolumn{1}{c}{ED visit}"
        r"& \multicolumn{1}{c}{IP admission}\\"
    )
    full.append(r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}")
    full.append(
        r"\makecell[l]{Week} & \makecell[c]{Obs\\N\tnote{a}}"
        r"& \makecell[c]{Effect\\(95\% CI)\tnote{b}}"
        r"& \makecell[c]{Effect\\(95\% CI)\tnote{b}}"
        r"& \makecell[c]{Effect\\(95\% CI)\tnote{b}}\\"
    )
    full.append(r"\midrule")
    full.append(body.rstrip("\n"))
    full.append(r"\bottomrule")
    full.append(r"\end{tabular}")
    full.append("")
    full.append(r"% Note a: Effects reported in percentage points (coef and 95\% CI multiplied by 100).")
    full.append(rf"% Model: within-beneficiary FE via demeaning; comparing placebo week k to reference week {REF_WEEK}.")
    full.append(rf"% SEs: two-way clustered at ({CLUSTER1_COL}, {CLUSTER2_COL}) when feasible; else one-way at {CLUSTER1_COL}; else {FALLBACK_COV_TYPE}.")
    with open(OUT_TEX_FULL, "w", encoding="utf-8") as f:
        f.write("\n".join(full) + "\n")

    print(f"[OK] Wrote:\n  {OUT_TEX_BODY}\n  {OUT_TEX_FULL}")

# -------------------------
# Main
# -------------------------
def main():
    dfs = []
    for y in YEARS:
        df_y = load_year(y)
        if not df_y.empty:
            dfs.append(df_y)

    if not dfs:
        raise RuntimeError("No placebo panel files loaded.")

    df_all = pd.concat(dfs, ignore_index=True)

    # ... Merge CKD event-level flags and restrict to CKD == 1 ...
    ckd_df = pd.read_csv(CKD_FLAG_FILE, low_memory=False) # CKD flags created from the pipeline that created exh2 tab1 (pt characteristics table)
    
    # clean merge keys to match main panel formatting
    for c in ["BENE_ID", "storm_id", "fips", "event_id"]:
        if c in ckd_df.columns:
            ckd_df[c] = _clean_str_series(ckd_df[c])
    
    if "PRVDR_NUM_event" in ckd_df.columns:
        ckd_df["PRVDR_NUM_event"] = _clean_str_series(ckd_df["PRVDR_NUM_event"]).str.zfill(6)
    
    if "year" in ckd_df.columns:
        ckd_df["year"] = pd.to_numeric(ckd_df["year"], errors="coerce").astype("Int64")
    
    ckd_df["ckd_ind"] = pd.to_numeric(ckd_df["ckd_ind"], errors="coerce")
    
    # keep one row per event-level key in the CKD flag file
    ckd_keep = ckd_df[
        ["year", "storm_id", "event_id", "BENE_ID", "fips", "PRVDR_NUM_event", "ckd_ind"]
    ].drop_duplicates().copy()
    
    before_merge_n = len(df_all) # count
    
    df_all = df_all.merge(
        ckd_keep,
        on=["year", "storm_id", "event_id", "BENE_ID", "fips", "PRVDR_NUM_event"],
        how="left",
        validate="many_to_one"
    )
    
    print(f"[INFO] Rows before CKD merge = {before_merge_n:,}") # QCs
    print(f"[INFO] Rows after CKD merge  = {len(df_all):,}")
    print("[QC] ckd_ind after merge:")
    print(df_all["ckd_ind"].value_counts(dropna=False).sort_index())
    
    before_ckd_n = len(df_all)
    df_all = df_all[df_all["ckd_ind"] == 1].copy() # Keep only CKDs
    
    print(f"[INFO] kept {len(df_all):,} CKD rows; dropped {before_ckd_n - len(df_all):,}")
    print(
        f"[INFO] CKD-only pooled rows={len(df_all):,} | "
        f"paired bene-storm N={n_paired_bene_storm(df_all):,} | "
        f"unique benes={df_all['BENE_ID'].nunique():,} | storms={df_all['storm_id'].nunique():,}"
    )

    if "week_rel" not in df_all.columns:
        raise KeyError("week_rel missing from placebo panel.")

    lines = [] # hold the LaTeX table rows
    meta_rows = [] # hold model diagnostics for a CSV output

    for wk in PLACEBO_TEST_WEEKS: # Loops through each placebo week I want to compare against the ref week: wk = -3 wk = -4 wk = -5 wk = -6
        d = restrict_to_two_weeks(df_all, wk, REF_WEEK) # if wk = -3, then this keeps only: week -3 week -2
        if d.empty:
            effects = {name: "" for _, name in OUTCOMES}
            lines.append(build_row(f"Week {wk} vs {REF_WEEK}", 0, effects))
            continue

        d = d.copy()
        d["fake_hazard"] = (pd.to_numeric(d["week_rel"], errors="coerce") == wk).astype(int)

        n_obs = n_paired_bene_storm(d) # for the Obs N in the table.

        effects_fmt = {} # create an empty dictionary that will hold the formatted effect strings for each outcome.
        for y_col, y_name in OUTCOMES: # loop through each outcome
            stats = run_within_bene_demean_twocluster(d, y_col=y_col) # run the within-bene placebo reg for that outcome.
            effects_fmt[y_name] = fmt_coef_ci(stats["coef"], stats["ci_lo"], stats["ci_hi"]) # format the result into a string for the LaTeX table.

            meta_rows.append({
                "contrast": f"{wk} vs {REF_WEEK}",
                "outcome": y_name,
                "se_type": stats.get("se_type", ""),
                "n_rows_block": int(len(d)),
                "n_obs_paired_bene_storm": int(n_obs),
                "n_benes": int(d["BENE_ID"].nunique()) if "BENE_ID" in d.columns else np.nan,
                "n_provider_clusters": int(d[CLUSTER1_COL].dropna().nunique()) if CLUSTER1_COL in d.columns else np.nan,
                "n_storm_clusters": int(d[CLUSTER2_COL].dropna().nunique()) if CLUSTER2_COL in d.columns else np.nan,
            })

        lines.append(build_row(f"Week {wk} vs {REF_WEEK}", n_obs, effects_fmt)) # After all outcomes are modeled, build the final LaTeX row and append it to lines.

    body = "".join(lines)
    write_tex(body)

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(META_OUT_CSV, index=False)
    print(f"[OK] Wrote meta CSV: {META_OUT_CSV} (n={len(meta_df):,})")

if __name__ == "__main__":
    main()