#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: July 22, 2026
# Description: This script takes the two-row beneficiary placebo analytical file (the one with paired placebo reference-week 
# and placebo exposure-week) restricts the sample to chronic kidney disease (CKD) beneficiaries, and estimates within-
# beneficiary-storm changes in disruption, ED visits, IP admissions, and mortality after hurricane exposure. It produces a pooled 
# table using two-way clustered standard errors by facility and storm and writes to LaTeX tables.
#----------------------------------------------------------------------------------------------------------------------#

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

# ====================
# Config
# ====================
YEARS = list(range(2011, 2023))

REF_WEEK = -7
HAZ_WEEK = -5

CLUSTER_COL = "facility_id"
STORM_COL = "storm_id"

RESTRICT_DISRUPTION_TO_MWF_TTS = False
SCHED_ALLOWED = {"MWF", "TTS"}
STABLE_COL = "stable_3x_weekly"
SCHEDULE_COL = "schedule_type"

DISRUPTION_FOOTNOTE_MARK = r"\textsuperscript{b}"
OBS_N_FOOTNOTE_MARK = r"\textsuperscript{c}"

ROUND_DECIMALS = 1
REPORT_IN_PERCENT_POINTS = True
Z_975 = 1.959963984540054

INPUT_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/"
    "dialysis/01_analytical_sample/"
)

OUTPUT_DIR = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "paper_exhibits_step5_disrupted_sample_placebo5wk/"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def input_path(year: int) -> str:
    return os.path.join(
        INPUT_BASE,
        f"esrd_crossover_{year}",
        "analytical_simple_case_crossover_anchor_exposure_placebo5wk_refwk_m7_hazwk_m5_early_wkm6_class_wkm8_cumpost_cumdeath_cumdisrupt_plus_ckd_v01.csv",
    )

# --------------------
# Output files
# --------------------
OUT_TEX_POOLED_BODY = os.path.join(
    OUTPUT_DIR, "exhibit_step5_disrupted_placebo5wk_pooled_multihorizon_stacked_body_refm7_hazm5_cumdisrupt.tex"
)
OUT_TEX_POOLED_FULL = os.path.join(
    OUTPUT_DIR, "exhibit_step5_disrupted_placebo5wk_pooled_multihorizon_stacked_full_refm7_hazm5_cumdisrupt.tex"
)

OUT_TEX_STORM_BODY = os.path.join(
    OUTPUT_DIR, "exhibit_step5_disrupted_placebo5wk_bystorm_weekly_body_refm7_hazm5_cumdisrupt.tex"
)
OUT_TEX_STORM_FULL = os.path.join(
    OUTPUT_DIR, "exhibit_step5_disrupted_placebo5wk_bystorm_weekly_full_refm7_hazm5_cumdisrupt.tex"
)

OUT_CSV_DISRUPT = os.path.join(
    OUTPUT_DIR, "storm_coefs_disrupted_placebo5wk_disruption_pp_weeklyonly_refm7_hazm5_cumdisrupt.csv"
)
OUT_CSV_ED = os.path.join(
    OUTPUT_DIR, "storm_coefs_disrupted_placebo5wk_ed_pp_weeklyonly_refm7_hazm5_cumdisrupt.csv"
)
OUT_CSV_IP = os.path.join(
    OUTPUT_DIR, "storm_coefs_disrupted_placebo5wk_ip_pp_weeklyonly_refm7_hazm5_cumdisrupt.csv"
)
OUT_CSV_DEATH = os.path.join(
    OUTPUT_DIR, "storm_coefs_disrupted_placebo5wk_mortality_pp_weeklyonly_refm7_hazm5_cumdisrupt.csv"
)

# ====================
# Formatting helpers
# ====================
def _r(x: float) -> float:
    if pd.isna(x):
        return np.nan
    return round(float(x), ROUND_DECIMALS)

def _pp(x: float) -> float:
    if pd.isna(x):
        return np.nan
    return float(x) * 100.0 if REPORT_IN_PERCENT_POINTS else float(x)

def fmt_coef_ci(coef: float, lo: float, hi: float) -> str:
    if pd.isna(coef) or pd.isna(lo) or pd.isna(hi):
        return ""
    coef_pp = _pp(coef)
    lo_pp   = _pp(lo)
    hi_pp   = _pp(hi)
    return f"{_r(coef_pp):.1f} [{_r(lo_pp):.1f}, {_r(hi_pp):.1f}]"

def fmt_int_latex(n) -> str:
    try:
        s = f"{int(n):,}"
    except Exception:
        return ""
    return s.replace(",", "{,}")

def latex_escape(s: str) -> str:
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

# ====================
# Data helpers
# ====================
def _clean_str_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return s

def load_step4_year(year: int) -> pd.DataFrame:
    csv_path = input_path(year)
    if not os.path.exists(csv_path):
        print(f"[SKIP] missing: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["year"] = year

    for c in ["BENE_ID", "storm_id", "event_id", "facility_id"]:
        if c in df.columns:
            df[c] = _clean_str_series(df[c])

    if "week_rel" in df.columns:
        df = df[df["week_rel"].isin([REF_WEEK, HAZ_WEEK])].copy()

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

def restrict_to_two_weeks(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"event_id", "BENE_ID", "week_rel"}
    miss = needed - set(df.columns)
    if miss:
        raise KeyError(f"Missing columns for pairing: {sorted(miss)}")

    counts = (
        df.groupby(["event_id", "BENE_ID", "week_rel"])
          .size()
          .unstack(fill_value=0)
    )
    has_ref = counts.get(REF_WEEK, 0) > 0
    has_haz = counts.get(HAZ_WEEK, 0) > 0
    keep_pairs = counts.index[has_ref & has_haz]

    keep_df = pd.DataFrame(list(keep_pairs), columns=["event_id", "BENE_ID"])
    return df.merge(keep_df, on=["event_id", "BENE_ID"], how="inner")

def restrict_disruption_sample(df: pd.DataFrame) -> pd.DataFrame:
    if not RESTRICT_DISRUPTION_TO_MWF_TTS:
        return df

    required = {STABLE_COL, SCHEDULE_COL}
    if not required.issubset(df.columns):
        return df

    d = df[(df[STABLE_COL] == 1) & (df[SCHEDULE_COL].isin(list(SCHED_ALLOWED)))].copy()
    if d.empty:
        return d

    d = restrict_to_two_weeks(d)
    return d

def n_paired_event_bene(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    required = {"event_id", "BENE_ID"}
    if not required.issubset(df.columns):
        return 0
    return int(df[["event_id", "BENE_ID"]].drop_duplicates().shape[0])

# ====================
# Modeling helpers
# ====================
def _cat_codes(series: pd.Series) -> np.ndarray:
    s = series.astype("category")
    codes = s.cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError("Found missing values when generating categorical codes.")
    return codes.astype(np.int64, copy=False)

def _ci_from_cov(params: pd.Series, cov: np.ndarray, term: str) -> dict:
    if term not in params.index:
        return {"coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    idx = list(params.index).index(term)
    coef = float(params.iloc[idx])
    se = float(np.sqrt(cov[idx, idx])) if cov is not None else np.nan
    if pd.isna(se) or se <= 0:
        return {"coef": coef, "ci_lo": np.nan, "ci_hi": np.nan}

    lo = coef - Z_975 * se
    hi = coef + Z_975 * se
    return {"coef": coef, "ci_lo": float(lo), "ci_hi": float(hi)}

def fit_ols_cluster_one_or_two_way(
    y: pd.Series,
    X: pd.DataFrame,
    g1: pd.Series,
    g2: pd.Series | None = None,
):
    """Fit OLS with the required clustered covariance estimator.

    Pooled models use two-way clustering by facility and storm.
    Storm-specific models use one-way clustering by facility.
    No heteroskedasticity-robust or one-way fallback is used for pooled models.
    If the requested clustering is not feasible, no estimate is returned.
    """
    y = pd.to_numeric(y, errors="coerce")
    X = X.copy().apply(pd.to_numeric, errors="coerce")

    valid = y.notna() & ~X.isna().any(axis=1) & g1.notna()
    if g2 is not None:
        valid = valid & g2.notna()

    if valid.sum() == 0 or X.shape[1] == 0:
        return None, None, "no_valid_rows"

    y2 = y.loc[valid]
    X2 = X.loc[valid]
    g1v = g1.loc[valid]

    try:
        model = sm.OLS(y2, X2)
        res_classic = model.fit()
    except Exception:
        return None, None, "ols_fit_error"

    n_g1 = int(pd.Series(g1v).nunique(dropna=True))
    if n_g1 < 2:
        return None, None, f"insufficient_{CLUSTER_COL}_clusters(n_g1={n_g1})"

    # Storm-specific analysis: one-way facility clustering only.
    if g2 is None:
        try:
            g1s = g1v.astype(str)
            res_c = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": g1s, "use_correction": True},
            )
            cov = np.asarray(res_c.cov_params())
            return res_c, cov, f"cluster({CLUSTER_COL})"
        except Exception as e:
            return None, None, f"cluster_fit_error({type(e).__name__})"

    # Pooled analysis: two-way facility and storm clustering only.
    g2v = g2.loc[valid]
    n_g2 = int(pd.Series(g2v).nunique(dropna=True))
    if n_g2 < 2:
        return None, None, f"insufficient_{STORM_COL}_clusters(n_g2={n_g2})"

    try:
        g1_codes = _cat_codes(g1v)
        g2_codes = _cat_codes(g2v)
        cov_both, _, _ = cov_cluster_2groups(res_classic, g1_codes, g2_codes)
        cov = np.asarray(cov_both)
        return res_classic, cov, f"cluster({CLUSTER_COL}+{STORM_COL})"
    except Exception as e:
        return None, None, f"twoway_cluster_error({type(e).__name__})"


def run_within_bene_demean(
    df: pd.DataFrame,
    y_col: str,
    *,
    use_two_way_if_possible: bool,
):
    """
    Estimate a within-beneficiary-storm model by demeaning the outcome
    and hazard-week indicator within each year × event_id × BENE_ID unit (basically within bene-storm).

    Standard errors:
      - pooled analyses: two-way clustered by facility and storm
      - storm-specific analyses: clustered by facility
    """

    required = [
        "year",
        "event_id",
        "BENE_ID",
        "hazard_week",
        y_col,
    ]

    miss = [c for c in required if c not in df.columns]
    if miss:
        return None, None, f"missing_cols({','.join(miss)})"

    d = df.dropna(
        subset=[
            "year",
            "event_id",
            "BENE_ID",
            "hazard_week",
            y_col,
        ]
    ).copy()

    if d.empty:
        return None, None, "no_data_after_dropna"

    # Clean identifiers.
    d["event_id"] = _clean_str_series(d["event_id"])
    d["BENE_ID"] = _clean_str_series(d["BENE_ID"])

    d["year"] = pd.to_numeric(
        d["year"],
        errors="coerce",
    )

    d["hazard_week"] = pd.to_numeric(
        d["hazard_week"],
        errors="coerce",
    ).astype(float)

    d[y_col] = pd.to_numeric(
        d[y_col],
        errors="coerce",
    ).astype(float)

    d = d.dropna(
        subset=[
            "year",
            "event_id",
            "BENE_ID",
            "hazard_week",
            y_col,
        ]
    ).copy()

    if d.empty:
        return None, None, "no_data_after_numeric"

    # ----------------------------------------------------------
    # Within-beneficiary-storm transformation
    # ----------------------------------------------------------
    g = d.groupby(
        [
            "year",
            "event_id",
            "BENE_ID",
        ],
        sort=False,
    )

    y_dm = d[y_col] - g[y_col].transform("mean")

    x_dm = (
        d["hazard_week"]
        - g["hazard_week"].transform("mean")
    )

    if np.isclose(x_dm.var(ddof=0), 0.0):
        return (
            None,
            None,
            "no_within_bene_event_variation_in_hazard",
        )

    if CLUSTER_COL not in d.columns:
        return None, None, f"missing_{CLUSTER_COL}"

    X = pd.DataFrame(
        {"hazard_week": x_dm},
        index=d.index,
    )

    g1 = _clean_str_series(d[CLUSTER_COL])

    if use_two_way_if_possible:
        if STORM_COL not in d.columns:
            return None, None, f"missing_{STORM_COL}"

        g2 = _clean_str_series(d[STORM_COL])

        return fit_ols_cluster_one_or_two_way(
            y_dm,
            X,
            g1=g1,
            g2=g2,
        )

    return fit_ols_cluster_one_or_two_way(
        y_dm,
        X,
        g1=g1,
        g2=None,
    )


def extract_stats(res, cov: np.ndarray | None, term="hazard_week"):
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

# ====================
# Computation blocks
# ====================
def _single_result(df_block: pd.DataFrame, y_col: str, *, pooled: bool) -> dict:
    res, cov, se_status = run_within_bene_demean(
        df_block, y_col, use_two_way_if_possible=pooled
    )
    stats = extract_stats(res, cov, "hazard_week")

    n_obs_used = int(
        df_block[[y_col, "hazard_week", "BENE_ID"]].dropna().shape[0]
    ) if {"BENE_ID", "hazard_week", y_col}.issubset(df_block.columns) else int(len(df_block))
    n_benes_used = int(df_block["BENE_ID"].nunique()) if "BENE_ID" in df_block.columns else np.nan

    return {
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

def compute_stacked_pooled_results(df_all: pd.DataFrame) -> list[dict]:
    rows = []

    d_dis = df_all
    if RESTRICT_DISRUPTION_TO_MWF_TTS:
        d_dis = restrict_disruption_sample(df_all)

    first_row = {
        "comparison": "Week -7 vs week -5",
        "n_obs": n_paired_event_bene(df_all),
    }

    if d_dis.empty:
        first_row["Disruption"] = ""
    else:
        r_dis = _single_result(d_dis, "disrupt_cmp_wk", pooled=True)
        eff = r_dis["effect"]
        if eff and RESTRICT_DISRUPTION_TO_MWF_TTS:
            eff = eff + DISRUPTION_FOOTNOTE_MARK
        first_row["Disruption"] = eff

    first_row["ED"] = _single_result(df_all, "any_ed_cmp_wk", pooled=True)["effect"]
    first_row["IP"] = _single_result(df_all, "any_ip_cmp_wk", pooled=True)["effect"]
    first_row["Death"] = _single_result(df_all, "any_death_cmp_wk", pooled=True)["effect"]
    rows.append(first_row)

    cumulative_specs = [
        ("Week -7 vs cumulative weeks -5 to -4", "disrupt_cmp_2wk", "any_ed_cmp_2wk", "any_ip_cmp_2wk", "any_death_cmp_2wk"),
        ("Week -7 vs cumulative weeks -5 to -3", "disrupt_cmp_3wk", "any_ed_cmp_3wk", "any_ip_cmp_3wk", "any_death_cmp_3wk"),
        ("Week -7 vs cumulative weeks -5 to -2", "disrupt_cmp_4wk", "any_ed_cmp_4wk", "any_ip_cmp_4wk", "any_death_cmp_4wk"),
    ]

    for label, disrupt_col, ed_col, ip_col, death_col in cumulative_specs:
        if d_dis.empty:
            disrupt_eff = ""
        else:
            r_dis = _single_result(d_dis, disrupt_col, pooled=True)
            disrupt_eff = r_dis["effect"]
            if disrupt_eff and RESTRICT_DISRUPTION_TO_MWF_TTS:
                disrupt_eff = disrupt_eff + DISRUPTION_FOOTNOTE_MARK

        row = {
            "comparison": label,
            "n_obs": "",
            "Disruption": disrupt_eff,
            "ED": _single_result(df_all, ed_col, pooled=True)["effect"],
            "IP": _single_result(df_all, ip_col, pooled=True)["effect"],
            "Death": _single_result(df_all, death_col, pooled=True)["effect"],
        }
        rows.append(row)

    return rows

def compute_weekly_results_block(df_block: pd.DataFrame, *, pooled: bool) -> dict:
    results = {}

    weekly_map = {
        "ED": "any_ed_cmp_wk",
        "IP": "any_ip_cmp_wk",
        "Death": "any_death_cmp_wk",
    }
    for outcome_short, y_col in weekly_map.items():
        results[outcome_short] = _single_result(df_block, y_col, pooled=pooled)

    d_dis = df_block
    if RESTRICT_DISRUPTION_TO_MWF_TTS:
        d_dis = restrict_disruption_sample(df_block)

    if d_dis.empty:
        results["Disruption"] = {
            "effect": "",
            "se_type": "no_data_after_restriction",
            "coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "coef_pp": np.nan, "ci_lo_pp": np.nan, "ci_hi_pp": np.nan,
            "n_obs_used": 0,
            "n_benes_used": 0,
        }
    else:
        r = _single_result(d_dis, "disrupt_cmp_wk", pooled=pooled)
        eff = r["effect"]
        if eff and RESTRICT_DISRUPTION_TO_MWF_TTS:
            eff = eff + DISRUPTION_FOOTNOTE_MARK
        r["effect"] = eff
        results["Disruption"] = r

    return results

# ====================
# LaTeX builders
# ====================
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
    return " & ".join(row) + r" \\" + "\n"

def write_pooled_tex_files(body: str):
    with open(OUT_TEX_POOLED_BODY, "w", encoding="utf-8") as f:
        f.write(body)

    full = []
    full.append(r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.2in} c *{4}{c}}")
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
    full.append(rf"% Pooled table uses two-way clustered SEs at ({CLUSTER_COL}, {STORM_COL}); estimates are left blank when two-way clustering is not feasible.")
    full.append(rf"% First row compares placebo week {REF_WEEK} vs placebo week {HAZ_WEEK}.")
    full.append(rf"% Second row compares placebo week {REF_WEEK} vs any outcome in placebo weeks -5 to -4.")
    full.append(rf"% Third row compares placebo week {REF_WEEK} vs any outcome in placebo weeks -5 to -3.")
    full.append(rf"% Fourth row compares placebo week {REF_WEEK} vs any outcome in placebo weeks -5 to -2.")
    full.append(rf"% Mortality rows use cumulative death-by-window indicators beginning from placebo week {HAZ_WEEK}.")
    full.append(rf"% Disruption rows use cumulative any-disrupted-week-by-window indicators beginning from placebo week {HAZ_WEEK}.")
    full.append(rf"% Beneficiaries who died before placebo_anchor_dt were already excluded in Step 4.")

    with open(OUT_TEX_POOLED_FULL, "w", encoding="utf-8") as f:
        f.write("\n".join(full) + "\n")

    print(f"[OK] Wrote pooled TEX:\n  {OUT_TEX_POOLED_BODY}\n  {OUT_TEX_POOLED_FULL}")

def write_storm_tex_files(body: str):
    with open(OUT_TEX_STORM_BODY, "w", encoding="utf-8") as f:
        f.write(body)

    full = []
    full.append(r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.2in} c *{4}{c}}")
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
        r"\makecell[l]{Storm} & \makecell[c]{Obs\\$N$" + OBS_N_FOOTNOTE_MARK + r"}"
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
    full.append(rf"% Storm-specific table uses one-way clustered SEs at {CLUSTER_COL}; estimates are left blank when facility clustering is not feasible.")
    full.append(rf"% Storm-specific estimates compare placebo week {REF_WEEK} with placebo week {HAZ_WEEK} only.")
    full.append(rf"% Weekly mortality row uses any_death_cmp_wk (placebo week -7 vs death by end of placebo week -5).")
    full.append(rf"% Weekly disruption row uses disrupt_cmp_wk (placebo week -7 vs placebo week -5).")

    with open(OUT_TEX_STORM_FULL, "w", encoding="utf-8") as f:
        f.write("\n".join(full) + "\n")

    print(f"[OK] Wrote storm TEX:\n  {OUT_TEX_STORM_BODY}\n  {OUT_TEX_STORM_FULL}")

def get_storm_ids_chronological(df_all: pd.DataFrame) -> list[str]:
    date_candidates = [
        "county_exposure_start_dt", "placebo_anchor_dt", "anchor_dt", "earliest_stress_day"
    ]
    date_col = next((c for c in date_candidates if c in df_all.columns), None)

    if date_col is not None:
        d0 = df_all[df_all["week_rel"] == HAZ_WEEK].copy()
        if d0.empty:
            d0 = df_all.copy()

        d0[date_col] = pd.to_datetime(d0[date_col], errors="coerce")
        order = (
            d0.groupby("storm_id")[date_col]
              .min()
              .sort_values(kind="mergesort")
        )
        return order.index.tolist() + [s for s in df_all["storm_id"].dropna().unique() if s not in order.index]

    tmp = pd.DataFrame({"storm_id": df_all["storm_id"].dropna().unique()})
    tmp = tmp.sort_values(["storm_id"], kind="mergesort")
    return tmp["storm_id"].tolist()

# ====================
# Main
# ====================
def main():
    dfs = []
    for y in YEARS:
        df_y = load_step4_year(y)
        if df_y.empty:
            continue

        df_y = restrict_to_two_weeks(df_y)
        if df_y.empty:
            continue

        if "ckd" not in df_y.columns:
            raise KeyError(f"ckd not found in disrupted placebo panel for year {y}.")

        df_y["ckd"] = pd.to_numeric(df_y["ckd"], errors="coerce").fillna(0).astype(int)
        df_y = df_y[df_y["ckd"] == 1].copy()
        if df_y.empty:
            print(f"[SKIP] no CKD sample after pairing filter for year {y}")
            continue

        dfs.append(df_y)

    if not dfs:
        raise RuntimeError("No placebo Step 4 data loaded after pairing filter.")

    df_all = pd.concat(dfs, ignore_index=True)

    if STORM_COL not in df_all.columns:
        raise KeyError(f"{STORM_COL} not found in placebo Step 4 panel.")

    if CLUSTER_COL not in df_all.columns:
        raise KeyError(
            f"{CLUSTER_COL} not found in placebo Step 4 panel; "
            "clustered standard errors cannot be estimated."
        )

    n_clusters = df_all[CLUSTER_COL].dropna().nunique()
    share_miss = df_all[CLUSTER_COL].isna().mean()
    print(f"[INFO] Pooled: unique {CLUSTER_COL}={n_clusters:,} | missing={share_miss:.3%}")

    print(
        f"[INFO] Total pooled rows={len(df_all):,} | "
        f"paired bene-storm N={n_paired_event_bene(df_all):,} | "
        f"unique benes={df_all['BENE_ID'].nunique():,} | storms={df_all[STORM_COL].nunique():,} | "
        f"weeks={sorted(df_all['week_rel'].unique().tolist()) if 'week_rel' in df_all.columns else 'NA'}"
    )

    pooled_rows = compute_stacked_pooled_results(df_all)
    pooled_body = build_pooled_stacked_body(pooled_rows)
    write_pooled_tex_files(pooled_body)

    lines = []
    csv_rows = {"Disruption": [], "ED": [], "IP": [], "Death": []}

    storm_ids = get_storm_ids_chronological(df_all)

    for sid in storm_ids:
        d = df_all[df_all[STORM_COL] == sid].copy()
        if d.empty:
            continue

        d = restrict_to_two_weeks(d)
        if d.empty:
            continue

        storm_results = compute_weekly_results_block(d, pooled=False)
        lines.append(build_storm_row(str(sid), n_paired_event_bene(d), storm_results))

        for outcome in ["Disruption", "ED", "IP", "Death"]:
            r = storm_results.get(outcome, {})
            csv_rows[outcome].append({
                "storm_id": str(sid),
                "coef_pp": r.get("coef_pp", np.nan),
                "ci_lo_pp": r.get("ci_lo_pp", np.nan),
                "ci_hi_pp": r.get("ci_hi_pp", np.nan),
                "se_type": r.get("se_type", ""),
                "n_rows_block": int(len(d)),
                "n_obs_used": r.get("n_obs_used", np.nan),
                "n_benes_used": r.get("n_benes_used", np.nan),
                "note": "placebo weekly only; pp=coef*100" if REPORT_IN_PERCENT_POINTS else "placebo weekly only",
            })

    storm_body = "".join(lines)
    write_storm_tex_files(storm_body)

    def _write_csv(out_path: str, rows: list[dict]):
        out_df = pd.DataFrame(rows)
        if out_df.empty:
            out_df.to_csv(out_path, index=False)
            print(f"[OK] Wrote empty CSV: {out_path}")
            return
        out_df = out_df.sort_values(["storm_id"], kind="mergesort")
        out_df.to_csv(out_path, index=False)
        print(f"[OK] Wrote CSV: {out_path} (n={len(out_df):,})")

    _write_csv(OUT_CSV_DISRUPT, csv_rows["Disruption"])
    _write_csv(OUT_CSV_ED,      csv_rows["ED"])
    _write_csv(OUT_CSV_IP,      csv_rows["IP"])
    _write_csv(OUT_CSV_DEATH,   csv_rows["Death"])

if __name__ == "__main__":
    main()





    
