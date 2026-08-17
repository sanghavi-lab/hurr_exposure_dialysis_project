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

# Note: Need to update the CKD file from exh 2 folder before running this.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import re
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

# Highest Saffir-Simpson U.S. category for each storm.
# Keys are lowercase so matching is not case-sensitive.
STORM_CATEGORY = {
    "irene-2011": 1,
    "isaac-2012": 1,
    "sandy-2012": 1,
    "arthur-2014": 2,
    "hermine-2016": 1,
    "matthew-2016": 2,
    "harvey-2017": 4,
    "irma-2017": 4,
    "nate-2017": 1,
    "florence-2018": 1,
    "michael-2018": 5,
    "barry-2019": 1,
    "dorian-2019": 2,
    "delta-2020": 2,
    "hanna-2020": 1,
    "isaias-2020": 1,
    "laura-2020": 4,
    "sally-2020": 2,
    "zeta-2020": 3,
    "ida-2021": 4,
    "nicholas-2021": 1,
    "ian-2022": 4,
}

OUTCOME_DISPLAY_ORDER = ["ED", "IP", "Death", "Disruption"]

OUTCOME_MAP = {
    "ED": "any_ed",
    "IP": "any_ip",
    "Death": "any_death",
    "Disruption": "disrupt_cmp_post_d1_d7",
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

# ... Stata-ready analysis dataset ...
OUT_STATA_ANALYSIS = os.path.join(
    OUTPUT_DIR,
    "early_dialysis_two_stage_analysis_ready.dta"
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
        df = df[df["week_rel"].isin([REF_WEEK, HAZ_WEEK])].copy()
    
    required_outcomes = list(OUTCOME_MAP.values())
    missing_outcomes = [
        c for c in required_outcomes
        if c not in df.columns
    ]
    
    if missing_outcomes:
        raise KeyError(
            f"Missing required outcome columns in {path}: {missing_outcomes}. "
            "Confirm that Script 1 v06 was run successfully."
        )
    
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

def run_fe_panelols(
    df: pd.DataFrame,
    y_col: str,
    cluster_cols: list[str],
):
    """
    PanelOLS within beneficiary-storm:

        y ~ hazard_week + earlyA_last_pre_offschedule
            + FE(BENE_ID × storm_id)

    Standard errors:
      - pooled analyses: clustered by facility and storm
      - storm-specific analyses: clustered by facility
    """

    # Use dict.fromkeys to remove duplicate column names while
    # preserving their original order.
    req = list(dict.fromkeys(
        [
            BENE_COL,
            STORM_COL,
            HAZ_COL,
            EARLY_COL,
            y_col,
        ] + cluster_cols
    ))

    miss = [c for c in req if c not in df.columns]
    if miss:
        return None, f"missing_cols({','.join(miss)})"

    d = df.copy()

    for c in [HAZ_COL, EARLY_COL, y_col]:
        d[c] = pd.to_numeric(
            d[c], errors="coerce"
        ).astype(float)

    d[BENE_COL] = _clean_str_series(d[BENE_COL])
    d[STORM_COL] = _clean_str_series(d[STORM_COL])

    d = d.dropna(
        subset=[
            BENE_COL,
            STORM_COL,
            HAZ_COL,
            EARLY_COL,
            y_col,
        ]
    ).copy()

    if d.empty:
        return None, "no_data_after_dropna"

    for cc in cluster_cols:
        d[cc] = _clean_str_series(d[cc])

    d = d.dropna(subset=cluster_cols).copy()

    if d.empty:
        return None, "no_data_after_cluster_dropna"

    # ----------------------------------------------------------
    # Create one entity for each beneficiary-storm pair.
    # ngroup() generates a unique numeric ID for each pair.
    # ----------------------------------------------------------
    d["_bene_storm_id"] = (
        d.groupby(
            [BENE_COL, STORM_COL],
            sort=False,
        )
        .ngroup()
    )

    # Sort the two observations within each beneficiary-storm pair.
    sort_cols = ["_bene_storm_id"]

    for c in ["event_id", "week_rel"]:
        if c in d.columns:
            sort_cols.append(c)

    d = d.sort_values(sort_cols).copy()

    # Artificial time index within each beneficiary-storm entity.
    d["_t"] = (
        d.groupby("_bene_storm_id", sort=False)
        .cumcount()
    )

    # First index: beneficiary-storm entity
    # Second index: time/row within the entity
    d = d.set_index(["_bene_storm_id", "_t"])

    y = d[y_col]

    X = d[
        [
            HAZ_COL,
            EARLY_COL,
        ]
    ]

    mod = PanelOLS(
        y,
        X,
        entity_effects=True,
    )

    clusters = d[cluster_cols]

    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
    )

    return res, f"clustered({'+'.join(cluster_cols)})"

def extract_term_stats(res, term: str) -> dict:
    """
    Extract the coefficient and 95% confidence interval for one model term.
    Returns missing values when the model was not estimated or the requested
    term is not present.
    """
    if res is None:
        return {
            "coef": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
        }

    if term not in res.params.index:
        return {
            "coef": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
        }

    coef = float(res.params[term])

    try:
        ci = res.conf_int()
        ci_lo, ci_hi = ci.loc[term].tolist()

        return {
            "coef": coef,
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
        }

    except Exception:
        return {
            "coef": coef,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
        }
    
# ... Latex builders ...
# These functions just take the outputs from the model (e.g., coef, CI, etc) and put's it in a file that can be used to create a table via latex. See latex files for more details

def build_row(sample_label: str, n_obs: int, effects_by_outcome: dict) -> str:
    row = [latex_escape(sample_label), fmt_int_latex(n_obs)]
    for k in OUTCOME_DISPLAY_ORDER:
        row.append(latex_escape(effects_by_outcome.get(k, "")))
    return " & ".join(row) + r" \\" + "\n"

def build_full_table(body: str, sample_header: str, note_se: str) -> str:
    full = []
    full.append(r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.7in} c *{4}{c}}")
    full.append(r"\toprule")
    full.append(
        r"& \multicolumn{1}{c}{Sample size}"
        r"& \multicolumn{1}{c}{ED visit}"
        r"& \multicolumn{1}{c}{IP admission}"
        r"& \multicolumn{1}{c}{Mortality}"
        r"& \multicolumn{1}{c}{Dialysis disruption}\\"
    )
    full.append(
        r"\cmidrule(lr){2-2}"
        r"\cmidrule(lr){3-3}"
        r"\cmidrule(lr){4-4}"
        r"\cmidrule(lr){5-5}"
        r"\cmidrule(lr){6-6}"
    )
    full.append(
        rf"\makecell[l]{{{latex_escape(sample_header)}}} & \makecell[c]{{Obs\\$N$}}"
        r"& \makecell[c]{Early effect\\(95\% CI)\tnote{a}}"
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
    full.append(
        r"% Model: y ~ hazard_week + earlyA_last_pre_offschedule "
        r"+ FE(BENE-STORM) (entity FE via PanelOLS)."
    )
    full.append(rf"% {note_se}")
    full.append(r"% The early coefficient is interpreted as the hazard-period differential effect for the early group because earlyA is only carried on the week 0 row.")
    full.append(
        r"% Dialysis disruption is defined as fewer than 3 outpatient dialysis "
        r"days during relative days 1--7 after exposure."
    )
    full.append(
        r"% The reference disruption outcome is fewer than 3 outpatient dialysis "
        r"days during relative days -14 through -8."
    )
    full.append(
        r"% Dialysis received on or before the exposure date is not counted in "
        r"the post-exposure days 1--7 disruption outcome."
    )
    full.append(
        r"% The days 1--7 window applies to dialysis disruption only; ED, IP, "
        r"and mortality retain their existing outcome windows."
    )
    return "\n".join(full) + "\n"

def write_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def get_storm_category_and_year(
    storm_id: str,
) -> tuple[float, float]:
    """
    Return the highest U.S. Saffir-Simpson category and year
    for a storm.

    Category is obtained from STORM_CATEGORY.
    Year is extracted from storm_id, such as Matthew-2016.
    """
    sid = str(storm_id).strip()

    category = STORM_CATEGORY.get(
        sid.lower(),
        np.nan,
    )

    year_match = re.search(r"(?:19|20)\d{2}", sid)
    year = (
        int(year_match.group(0))
        if year_match
        else np.nan
    )

    return category, year


def get_storm_ids_by_category(
    df_all: pd.DataFrame,
) -> list[str]:
    """
    Order storms by:
      1. Lowest Saffir-Simpson U.S. category first
      2. Earliest year within category
      3. Alphabetically within category and year
    """
    order = pd.DataFrame({
        STORM_COL: (
            df_all[STORM_COL]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .tolist()
        )
    })

    metadata = order[STORM_COL].apply(
        get_storm_category_and_year
    )

    order["storm_category"] = metadata.apply(
        lambda x: x[0]
    )

    order["storm_year"] = metadata.apply(
        lambda x: x[1]
    )

    unmatched = order.loc[
        order["storm_category"].isna(),
        STORM_COL,
    ].tolist()

    if unmatched:
        print(
            "[WARN] No hurricane category was found for: "
            + ", ".join(unmatched)
            + ". These storms will be placed at the end."
        )

    order = order.sort_values(
        [
            "storm_category",
            "storm_year",
            STORM_COL,
        ],
        ascending=[True, True, True],
        na_position="last",
        kind="mergesort",
    )

    return order[STORM_COL].tolist()

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

    # --------------------------------------
    # Export analysis-ready dataset for Stata
    # --------------------------------------
    
    stata_df = df_all.copy()
    
    # Preserve identifiers as strings.
    # This prevents loss of leading zeros in provider and FIPS codes
    # and avoids precision problems with long beneficiary identifiers.
    stata_id_cols = [
        "BENE_ID",
        "storm_id",
        "event_id",
        "fips",
        "PRVDR_NUM_event",
    ]
    
    for c in stata_id_cols:
        if c in stata_df.columns:
            stata_df[c] = _clean_str_series(stata_df[c])
    
    # Stata variable names cannot exceed 32 characters.
    # This check will stop the script if a variable name is too long.
    long_stata_names = [c for c in stata_df.columns if len(c) > 32]
    
    if long_stata_names:
        raise ValueError(
            "The following variable names exceed Stata's 32-character limit: "
            f"{long_stata_names}"
        )
    
    # Convert pandas extension data types that may not export cleanly to Stata.
    for c in stata_df.columns:
        dtype_name = str(stata_df[c].dtype)
    
        # Nullable integer columns
        if dtype_name.startswith("Int"):
            if stata_df[c].isna().any():
                stata_df[c] = stata_df[c].astype(float)
            else:
                stata_df[c] = stata_df[c].astype(np.int64)
    
        # Nullable Boolean columns
        elif dtype_name == "boolean":
            stata_df[c] = stata_df[c].astype(float)
    
        # Pandas string columns
        elif dtype_name == "string":
            stata_df[c] = stata_df[c].astype(object)
    
    # Export using a modern Stata file format.
    stata_df.to_stata(
        OUT_STATA_ANALYSIS,
        write_index=False,
        version=118
    )
    
    print(
        f"[OK] Wrote Stata analysis dataset:\n"
        f"  {OUT_STATA_ANALYSIS}\n"
        f"  rows={len(stata_df):,}\n"
        f"  beneficiary-storm pairs={n_paired_bene_storm(stata_df):,}\n"
        f"  unique beneficiaries={stata_df[BENE_COL].nunique():,}\n"
        f"  storms={stata_df[STORM_COL].nunique():,}"
    )
    
    # --------------------------------------
    # Count CKD-only beneficiaries with early dialysis
    # --------------------------------------
    early_haz = df_all.loc[
        (df_all["week_rel"] == HAZ_WEEK) &
        (df_all[EARLY_COL] == 1)
    ].copy()
    
    n_unique_benes_early = early_haz[BENE_COL].dropna().nunique()
    
    n_unique_bene_storm_early = (
        early_haz[[BENE_COL, STORM_COL]]
        .drop_duplicates()
        .shape[0]
    )
    
    print(
        f"[INFO] CKD-only: unique beneficiaries with early dialysis "
        f"({EARLY_COL}=1 in week_rel={HAZ_WEEK}) = {n_unique_benes_early:,}"
    )
    
    print(
        f"[INFO] CKD-only: unique bene-storm pairs with early dialysis "
        f"({EARLY_COL}=1 in week_rel={HAZ_WEEK}) = {n_unique_bene_storm_early:,}"
    )

    # ------- #
    
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

    storm_ids = get_storm_ids_by_category(
        df_all
    )  # lowest category, then year, then storm name

    for sid in storm_ids:
        storm_category, storm_year = (
            get_storm_category_and_year(sid)
        )

        d = df_all[
            df_all[STORM_COL] == sid
        ].copy()  # get one storm

        if d.empty:
            continue

        d = restrict_to_two_weeks(d)

        if d.empty:
            continue

        # ... Runs the model just like the pooled but looping
        # through each storm ...
        effects = {}
        se_types = {}

        for out_name in OUTCOME_DISPLAY_ORDER:
            y_col = OUTCOME_MAP[out_name]

            res, se_status = run_fe_panelols(
                d,
                y_col,
                cluster_cols=[CLUSTER_COL],
            )

            stats = extract_term_stats(
                res,
                EARLY_COL,
            )

            effects[out_name] = fmt_coef_ci(
                stats["coef"],
                stats["ci_lo"],
                stats["ci_hi"],
            )

            se_types[out_name] = se_status

            storm_csv_rows.append({
                "storm_id": str(sid),
                "storm_category": (
                    int(storm_category)
                    if pd.notna(storm_category)
                    else np.nan
                ),
                "storm_year": storm_year,
                "outcome": out_name,
                "coef_pp": _pp(stats["coef"]),
                "ci_lo_pp": _pp(stats["ci_lo"]),
                "ci_hi_pp": _pp(stats["ci_hi"]),
                "n_paired_bene_storm": (
                    n_paired_bene_storm(d)
                ),
                "n_rows_block": int(len(d)),
                "se_type": se_status,
            })

        # Show the category directly beside the storm name.
        if pd.notna(storm_category):
            storm_label = (
                f"{sid} (Cat. {int(storm_category)})"
            )
        else:
            storm_label = str(sid)

        storm_lines.append(
            build_row(
                storm_label,
                n_paired_bene_storm(d),
                effects,
            )
        )

    # ... Latex builder ...
    storm_body = "".join(storm_lines)

    write_text(
        OUT_STORM_TEX_BODY,
        storm_body,
    )

    write_text(
        OUT_STORM_TEX_FULL,
        build_full_table(
            storm_body,
            sample_header="Storm",
            note_se=(
                "By-storm SEs: one-way clustered by facility "
                "(PRVDR_NUM_event). Storms are ordered from "
                "the lowest to the highest U.S. "
                "Saffir-Simpson category, then by year and "
                "storm name."
            ),
        ),
    )

    print(
        f"[OK] Wrote storm-specific LaTeX:\n"
        f"  {OUT_STORM_TEX_BODY}\n"
        f"  {OUT_STORM_TEX_FULL}"
    )

    # Keep the storm CSV in the same order as the LaTeX table.
    storm_csv_df = pd.DataFrame(
        storm_csv_rows
    )

    if not storm_csv_df.empty:
        outcome_order = {
            outcome: position
            for position, outcome
            in enumerate(OUTCOME_DISPLAY_ORDER)
        }

        storm_csv_df["_outcome_order"] = (
            storm_csv_df["outcome"]
            .map(outcome_order)
        )

        storm_csv_df = storm_csv_df.sort_values(
            [
                "storm_category",
                "storm_year",
                "storm_id",
                "_outcome_order",
            ],
            ascending=[True, True, True, True],
            na_position="last",
            kind="mergesort",
        ).drop(
            columns="_outcome_order"
        )

    storm_csv_df.to_csv(
        OUT_STORM_CSV,
        index=False,
    )

    print(
        f"[OK] Wrote storm CSV: "
        f"{OUT_STORM_CSV} "
        f"(n={len(storm_csv_df):,})"
    )

if __name__ == "__main__":
    main()
