#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 22, 2026
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
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS # using PanelOLS here for a cleaner look. Processing doesn't take that long so PanelOLS is feasible.

# -------------------------
# Paths and spec
# -------------------------

IN_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "04_analytical_panel_hurr_exposure_v05_wkm2_facclust_cumpost_cumdeath"
)

CKD_FLAG_FILE = ( # This was created by a file in exhibit 2 table 1 folder. Basically a dataset with indicators for chronic kidney disease
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "table1_early_vs_nonearly_v04_ckd_only/event_ckd_flag_export.csv"
)

YEARS = list(range(2011, 2023))

REF_WEEK = -2
HAZ_WEEK = 0

EARLY_COL   = "earlyA_last_pre_offschedule"
HAZ_COL     = "hazard_week"
BENE_COL    = "BENE_ID"
STORM_COL   = "storm_id"
CLUSTER_COL = "PRVDR_NUM_event" # this is for cluster se but, below, we will use a two way cluster se with STORM_COL

OUTCOME_DISPLAY_ORDER = ["ED", "IP", "Death"] # skipping disruption as an outcome for this model due to discrepancies in disruption definition when considering early dialysis
OUTCOME_MAP = {
    "ED": "any_ed",
    "IP": "any_ip",
    "Death": "any_death",
}

ROUND_DECIMALS = 1 # for presentation
REPORT_IN_PERCENT_POINTS = True  # multiply coef/CI by 100

OUTPUT_DIR = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "paper_exhibits_step5_split_tables/"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ... Output files: pooled ...
OUT_POOLED_TEX_BODY = os.path.join(
    OUTPUT_DIR, "exhibit_step5_early_FE_pooled_body.tex"
)
OUT_POOLED_TEX_FULL = os.path.join(
    OUTPUT_DIR, "exhibit_step5_early_FE_pooled_full.tex"
)

# ... Output files: by-storm ...
OUT_STORM_TEX_BODY = os.path.join(
    OUTPUT_DIR, "exhibit_step5_early_FE_by_storm_body.tex"
)
OUT_STORM_TEX_FULL = os.path.join(
    OUTPUT_DIR, "exhibit_step5_early_FE_by_storm_full.tex"
)

# ... storm-level CSV format...
OUT_STORM_CSV = os.path.join(
    OUTPUT_DIR, "storm_coefs_early_pp.csv"
)

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

# ... Cleaning/Processing ...

def _clean_str_series(s: pd.Series) -> pd.Series: # cleaning
    s = s.astype(str).str.replace(r"\.0$", "", regex=True) # removes a trailing .0 at the end of a string.
    s = s.replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return s

def restrict_to_stable_schedule(df: pd.DataFrame) -> pd.DataFrame: # among early dialyzed, keep only those with a full week of dialysis (3 dialysis sessions). Important since we want to compare "stable" bene's with early dialysis vs those without.
    if df.empty:
        return df

    if "stable_3x_weekly" not in df.columns:
        raise KeyError("stable_3x_weekly not found in Step4 panel. Can't restrict to stable schedule.")

    d = df.copy()
    d["stable_3x_weekly"] = pd.to_numeric(d["stable_3x_weekly"], errors="coerce").fillna(0).astype(int)
    d = d[d["stable_3x_weekly"] == 1].copy()

    if "schedule_type" in d.columns:
        d["schedule_type"] = d["schedule_type"].astype(str)
        d = d[d["schedule_type"].isin(["MWF", "TTS"])].copy() # Keep only MWF/TTS

    return d

def load_step4_year(year: int) -> pd.DataFrame: # loads the analytical file created prior
    path = os.path.join(IN_BASE, f"year_{year}", "analytical_panel.csv")
    if not os.path.exists(path):
        print(f"[SKIP] missing: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["year"] = year

    for c in [BENE_COL, STORM_COL, "fips"]:
        if c in df.columns:
            df[c] = _clean_str_series(df[c])

    if CLUSTER_COL in df.columns:
        df[CLUSTER_COL] = _clean_str_series(df[CLUSTER_COL]).str.zfill(6)

    if "week_rel" in df.columns:
        df = df[df["week_rel"].isin([REF_WEEK, HAZ_WEEK])].copy() # keep only reference week and week of exposure

    for c in [HAZ_COL, EARLY_COL] + list(OUTCOME_MAP.values()):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df

def restrict_to_two_weeks(df: pd.DataFrame) -> pd.DataFrame: # keeps only bene that have both rows in the long panel: week -2 and week 0
    needed = {"event_id", BENE_COL, "week_rel"} # creates a set of column names that must be present for the function to work
    miss = needed - set(df.columns) # calculates which of those required columns are missing from the DataFrame.
    if miss: # If any required columns are missing, the function raises an error and names the missing columns.
        raise KeyError(f"Missing columns for pairing: {sorted(miss)}")

    counts = (
        df.groupby(["event_id", BENE_COL, "week_rel"])
          .size()
          .unstack(fill_value=0)
    )

    has_ref = counts.get(REF_WEEK, 0) > 0 # creates a boolean indicator for whether each pair has at least one reference-week row, meaning week_rel = -2
    has_haz = counts.get(HAZ_WEEK, 0) > 0 # creates a boolean indicator for whether each pair has at least one hazard-week row, meaning week_rel = 0
    keep_pairs = counts.index[has_ref & has_haz] # This keeps only the index entries where both conditions are true: has week -2 has week 0

    keep_df = pd.DataFrame(list(keep_pairs), columns=["event_id", BENE_COL]) # This converts those valid pairs into a small DataFrame with two columns: event_id and bene id
    return df.merge(keep_df, on=["event_id", BENE_COL], how="inner")

def n_paired_bene_storm(df: pd.DataFrame) -> int: # returns a count of unique beneficiary-storm combinations
    if df.empty or not {BENE_COL, STORM_COL}.issubset(df.columns):
        return 0
    return int(df[[BENE_COL, STORM_COL]].drop_duplicates().shape[0])

# ... Modeling ...

def run_fe_panelols(df: pd.DataFrame, y_col: str, cluster_cols: list[str]):
    """
    PanelOLS within bene:
      y ~ hazard_week (exposure week) + early + FE(BENE_ID)

    SEs:
      - clustered, 2-way facility and storm
    """
    req = [BENE_COL, HAZ_COL, EARLY_COL, y_col] + cluster_cols # a list of cols the function requires in order to run.
    miss = [c for c in req if c not in df.columns] # checks whether any of those required columns are missing
    if miss:
        return None, f"missing_cols({','.join(miss)})"

    d = df.copy()

    for c in [HAZ_COL, EARLY_COL, y_col]:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)

    d = d.dropna(subset=[BENE_COL, HAZ_COL, EARLY_COL, y_col]).copy()
    if d.empty:
        return None, "no_data_after_dropna"

    for cc in cluster_cols:
        d[cc] = _clean_str_series(d[cc])

    d = d.dropna(subset=cluster_cols).copy()
    if d.empty:
        return None, "no_data_after_cluster_dropna"

    # Sort by specified cols
    sort_cols = [BENE_COL]
    for c in [STORM_COL, "event_id", "week_rel"]:
        if c in d.columns:
            sort_cols.append(c)
    d = d.sort_values(sort_cols).copy()

    d["_t"] = d.groupby(BENE_COL).cumcount() # creates a within-beneficiary row counter called _t. It is created so PanelOLS can treat the data as panel data with a two-level index: person and time
    d = d.set_index([BENE_COL, "_t"]) # This sets a two-level panel index: entity = bene id, time = _t
    # ^ Since the model is comparing the paired rows within each beneficiary, this creates _t as an artificial time ordering variable.

    y = d[y_col] # dependent variable (ip, ed, or mortality)
    X = d[[HAZ_COL, EARLY_COL]] # the regressor matrix using two predictors: hazard_week and earlyA_last_pre_offschedule

    mod = PanelOLS(y, X, entity_effects=True) # defines the fixed-effects regression model. entity_effects=True means beneficiary fixed effects are included. So the model compares values within the same beneficiary over time, rather than across different beneficiaries.
    clusters = d[cluster_cols] # facility and storm variables

    res = mod.fit(cov_type="clustered", clusters=clusters) # fit the linear model with two way clustering
    return res, f"clustered({'+'.join(cluster_cols)})"

def extract_term_stats(res, term: str): # pulls one coefficient and its confidence interval out of a fitted model result to be put in latex later
    if res is None:
        return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    if term not in res.params.index:
        return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    coef = float(res.params[term])
    ci = res.conf_int()
    ci_lo, ci_hi = ci.loc[term].tolist()
    return {"coef": coef, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}

# ... Latex builders ...
# These functions just take the outputs from the model (e.g., coef, CI, etc) and put's it in a file that can be used to create a table via latex. See latex files for more details

def build_row(sample_label: str, n_obs: int, effects_by_outcome: dict) -> str:
    row = [latex_escape(sample_label), fmt_int_latex(n_obs)]
    for k in OUTCOME_DISPLAY_ORDER:
        row.append(latex_escape(effects_by_outcome.get(k, "")))
    return " & ".join(row) + r" \\" + "\n"

def build_full_table(body: str, sample_header: str, note_se: str) -> str:
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
        rf"\makecell[l]{{{latex_escape(sample_header)}}} & \makecell[c]{{Obs\\$N$}}"
        r"& \makecell[c]{Early effect\\(95\% CI)\tnote{a}}"
        r"& \makecell[c]{Early effect\\(95\% CI)\tnote{a}}"
        r"& \makecell[c]{Early effect\\(95\% CI)\tnote{a}}\\"
    )
    full.append(r"\midrule")
    full.append(body.rstrip("\n"))
    full.append(r"\bottomrule")
    full.append(r"\end{tabular}")
    full.append("")
    full.append(r"% Note a: Effects reported in percentage points (coef and CI multiplied by 100).")
    full.append(r"% Model: y ~ hazard_week + earlyA_last_pre_offschedule + FE(BENE) (entity FE via PanelOLS).")
    full.append(rf"% {note_se}")
    full.append(r"% The early coefficient is interpreted as the hazard-period differential effect for the early group because earlyA is only carried on the week 0 row.")
    return "\n".join(full) + "\n"

def write_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def get_storm_ids_chronological(df_all: pd.DataFrame) -> list[str]: # creates an ordered list of storm IDs
    if "storm_year" in df_all.columns:
        order = (
            df_all.groupby(STORM_COL)["storm_year"] # STORM_COL is storm_id
                  .min()
                  .sort_values(kind="mergesort")
        )
        return order.index.tolist()
    return df_all[STORM_COL].dropna().unique().tolist()

# -------------------------
# Main
# -------------------------

def main():
    dfs = []
    for y in YEARS:
        d = load_step4_year(y) # analytical file
        if d.empty:
            continue
        d = restrict_to_two_weeks(d)
        if d.empty:
            continue
        d = restrict_to_stable_schedule(d)
        if d.empty:
            continue
        dfs.append(d)

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
    
    # Clean event_id in df_all too, since this script does not currently do that upstream
    if "event_id" in df_all.columns:
        df_all["event_id"] = _clean_str_series(df_all["event_id"])
    
    ckd_keep = ckd_df[
        ["year", "storm_id", "event_id", "BENE_ID", "fips", CLUSTER_COL, "ckd_ind"]
    ].drop_duplicates().copy()
    
    before_merge_n = len(df_all)
    
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
    
    before_ckd_n = len(df_all)
    df_all = df_all[df_all["ckd_ind"] == 1].copy() # Keep only CKDs
    
    print(f"[INFO] kept {len(df_all):,} CKD rows; dropped {before_ckd_n - len(df_all):,}")

    print(
        f"[INFO] Total pooled rows={len(df_all):,} | "
        f"paired bene-storm N={n_paired_bene_storm(df_all):,} | "
        f"unique benes={df_all[BENE_COL].nunique():,} | "
        f"storms={df_all[STORM_COL].nunique():,}"
    )

    # ... POOLED TABLE (table 3) ...
    pooled_effects = {}
    pooled_se_types = {}

    for out_name in OUTCOME_DISPLAY_ORDER: # order if ED, IP, then mortality
        y_col = OUTCOME_MAP[out_name]
        res, se_status = run_fe_panelols( # run the model
            df_all,
            y_col,
            cluster_cols=[CLUSTER_COL, STORM_COL] # two way cluster se
        )
        stats = extract_term_stats(res, EARLY_COL) # get coef and ci
        pooled_effects[out_name] = fmt_coef_ci(stats["coef"], stats["ci_lo"], stats["ci_hi"])
        pooled_se_types[out_name] = se_status # just to tell me how the SE status was computed (two way? one way? etc...). I checked and it successfully did two way.

    # ... Latex builder ...
    pooled_body = build_row("Pooled", n_paired_bene_storm(df_all), pooled_effects)

    write_text(OUT_POOLED_TEX_BODY, pooled_body)
    write_text(
        OUT_POOLED_TEX_FULL,
        build_full_table(
            pooled_body,
            sample_header="Sample",
            note_se="Pooled SEs: two-way clustered by facility (PRVDR_NUM_event) and storm (storm_id)."
        )
    )

    print(f"Wrote pooled LaTeX:\n  {OUT_POOLED_TEX_BODY}\n  {OUT_POOLED_TEX_FULL}")

    # ... STORM-SPECIFIC TABLE ...
    # this is for appendix. Storm-by-storm analysis instead of the pooled
    
    storm_lines = []
    storm_csv_rows = []

    for sid in get_storm_ids_chronological(df_all): # an ordered list of storm IDs
        d = df_all[df_all[STORM_COL] == sid].copy() # get a storm
        if d.empty:
            continue

        d = restrict_to_two_weeks(d)
        if d.empty:
            continue

        # ... Runs the model just like the pooled but looping through each storm ...
        effects = {}
        se_types = {}

        for out_name in OUTCOME_DISPLAY_ORDER:
            y_col = OUTCOME_MAP[out_name]
            res, se_status = run_fe_panelols(
                d,
                y_col,
                cluster_cols=[CLUSTER_COL]
            )
            stats = extract_term_stats(res, EARLY_COL)

            effects[out_name] = fmt_coef_ci(stats["coef"], stats["ci_lo"], stats["ci_hi"])
            se_types[out_name] = se_status

            storm_csv_rows.append({
                "storm_id": str(sid),
                "outcome": out_name,
                "coef_pp": _pp(stats["coef"]),
                "ci_lo_pp": _pp(stats["ci_lo"]),
                "ci_hi_pp": _pp(stats["ci_hi"]),
                "n_paired_bene_storm": n_paired_bene_storm(d),
                "n_rows_block": int(len(d)),
                "se_type": se_status,
            })

        storm_lines.append(
            build_row(str(sid), n_paired_bene_storm(d), effects) # str(sid) converts the storm id to a string so it can be used as the row label, n_paired_bene_storm(d) computes the sample size shown for that storm row, and "effects" is a dictionary holding the formatted results for that storm
        )

    # ... Latex builder ...
    storm_body = "".join(storm_lines) # takes the storm by storm results and joins them so that it's readable by latex

    write_text(OUT_STORM_TEX_BODY, storm_body)
    write_text(
        OUT_STORM_TEX_FULL,
        build_full_table(
            storm_body,
            sample_header="Storm",
            note_se="By-storm SEs: one-way clustered by facility (PRVDR_NUM_event)."
        )
    )

    print(f"[OK] Wrote storm-specific LaTeX:\n  {OUT_STORM_TEX_BODY}\n  {OUT_STORM_TEX_FULL}")

    pd.DataFrame(storm_csv_rows).to_csv(OUT_STORM_CSV, index=False) # csv style of outputs
    print(f"[OK] Wrote storm CSV: {OUT_STORM_CSV} (n={len(storm_csv_rows):,})")

if __name__ == "__main__":
    main()
