#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 30, 2026
# Description: This script builds Table 1 (bene characteristics) by filtering and cleaning relevant columns and outputs 
# a tex file for LaTeX to generate the table. It also computes p-values and appends asterisks to denote statistical 
# significance.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# -------------------------
# Paths and spec
# -------------------------
YEAR_MIN, YEAR_MAX = 2011, 2022

INPUT_BASE = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "05g_analytical_sample_anchor_exposure_plus_comorb_plus_ahrf_plus_mbsf_demo_cc_otcc_plus_wind_v01"
)

OUT_DIR = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "table1_early_vs_nonearly_v01_ckd_only"
)
os.makedirs(OUT_DIR, exist_ok=True)

OUT_MACRO_TEX = os.path.join(OUT_DIR, "table1_disrupted_macros.tex")
OUT_EVENT_CSV = os.path.join(OUT_DIR, "table1_event_bene_level_dataset.csv")
OUT_CKD_FLAG_CSV = os.path.join(OUT_DIR, "event_bene_ckd_flag_export.csv")

GROUP_VAR = "earlyA_last_pre_offschedule"
TABLE_PREFIX = "Disrupted"

DISPLAY_DECIMALS_CONT = 2
DISPLAY_DECIMALS_PCT = 2

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

def input_file(year: int) -> str:
    return os.path.join(INPUT_BASE, f"year_{year}", "analytical_panel.csv")

def fmt_num(x, digits):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"

def fmt_int(x):
    if pd.isna(x):
        return ""
    return f"{int(round(x)):,}"

def star_from_p(p): # stars for significance
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def mean_ignore_na(x: pd.Series): # take average
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return np.nan
    return x.mean()

def prop_percent(x: pd.Series): # convert to percents
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return np.nan
    return x.mean() * 100.0

def ttest_pvalue(x1: pd.Series, x0: pd.Series):
    a = pd.to_numeric(x1, errors="coerce").dropna()
    b = pd.to_numeric(x0, errors="coerce").dropna()

    if len(a) == 0 or len(b) == 0: # if either group has no usable values, it returns NaN
        return np.nan

    if a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]: # if both groups are constant and have the exact same value (e.g., if group "a" has [10, 10, 10] and "b" has [10, 10]), it returns p-value of 1.0 because there is clearly no difference
        return 1.0

    try:
        res = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit") # two-sample t-test
        return float(res.pvalue) # returns just the p-value
    except Exception:
        return np.nan

def prop_test_pvalue(x1: pd.Series, x0: pd.Series):
    a = pd.to_numeric(x1, errors="coerce")
    b = pd.to_numeric(x0, errors="coerce")

    a = a[a.notna()]
    b = b[b.notna()]

    if len(a) == 0 or len(b) == 0: # if either group has no usable values, it returns NaN
        return np.nan

    count = np.array([int((a == 1).sum()), int((b == 1).sum())], dtype=float) # Counts how many 1s are in each group.
    nobs = np.array([int(a.notna().sum()), int(b.notna().sum())], dtype=float) # Counts how many total non-missing observations are in each group.

    if np.any(nobs == 0): # Checks again whether either group has zero observations.
        return np.nan

    if count[0] == count[1] and nobs[0] == nobs[1]: # If both groups have the same number of 1s and the same total number of observations then they have the exact same proportion so return p value of 1
        return 1.0

    try:
        _, pval = proportions_ztest(count=count, nobs=nobs) # Runs a two-sample proportion z-test. This compares the proportion of 1s in the two groups.
        return float(pval)
    except Exception:
        return np.nan

def nice_label(var: str) -> str:
    raw = str(var).replace("__otcc", "").replace("_medicare", "").strip()
    raw_lower = raw.lower() # lower case

    if raw_lower in { # if name matches, return "Chronic Kidney Disease (%)"
        "chronickidney",
        "chronickidneydisease",
        "chronic_kidney",
        "chronic_kidney_disease",
        "ckd",
    }:
        return "Chronic Kidney Disease (%)"

    label = raw.replace("_", " ") # replace underscores with spaces.
    label = " ".join([
        w.upper() if w.lower() in {"ami", "copd", "chf", "esrd", "hiv", "aids", "oud", "pvd", "tia", "oa", "ra", "ckd"} # if a word is one of the listed abbreviations, it converts it to all caps
        else w.capitalize() # otherwise, it capitalizes the word normally
        for w in label.split() # splits the label into words
    ]) # e.g., "ami" becomes "AMI", "stroke tia" becomes "Stroke TIA"

    label = label.replace("Alzh", "Alzheimer's") # "Alzh" -> "Alzheimer's"
    label = label.replace("Alzh Demen", "Alzheimer's / Dementia") # "Alzh Demen" -> "Alzheimer's / Dementia"
    label = label.replace("Ischemicheart", "Ischemic Heart Disease") # etc...
    label = label.replace("Stroke Tia", "Stroke / TIA")
    label = label.replace("Ra Oa", "Rheumatoid Arthritis / Osteoarthritis")
    label = label.replace("Hyperl", "Hyperlipidemia")
    label = label.replace("Hyperp", "Hyperplasia")
    label = label.replace("Hypert", "Hypertension")
    label = label.replace("Hypoth", "Hypothyroidism")

    return label + " (%)"

def find_ckd_variable(event_df: pd.DataFrame, cc_vars: list) -> str: # this searches for the ckd variable. It is not hard-coded because the CKD variable name may vary across datasets or processing steps prior. Ultimately, it will be used to restrict the analysis to CKD-only. Note that I also manually checked to ensure we have 100% CKD bene's.
    norm_map = {c: re.sub(r"[^a-z0-9]+", "", str(c).lower()) for c in event_df.columns} # create a dict called norm_map. For every col name in event_df.columns, converts it to lowercase and removes anything that is not a letter or number

    exact_targets = {
        "chronickidney",
        "chronickidneydisease",
        "chronickidneydiseasemedicare",
        "chronickidneydiseasecc",
        "ckd",
    }

    exact_hits = [c for c, n in norm_map.items() if n in exact_targets] # build list of col names if the norm_map matches one of above exact_targets. Basically finding ckd col.
    if len(exact_hits) == 1:
        return exact_hits[0] # return the col if there is exaclty one match.
    if len(exact_hits) > 1:
        print(f"[WARN] Multiple exact CKD-like matches found: {exact_hits}. Using first.") # if there is more than one CKD column, prints a warning
        return exact_hits[0]

    kidney_hits = [] # an empty list for the second search strategy
    for c in cc_vars: # loop through each variable name in cc_vars.
        cl = str(c).lower()
        if "kidney" in cl or "ckd" in cl: # if the CC variable name contains the text "kidney" or "ckd", add it to kidney_hits.
            kidney_hits.append(c)

    if len(kidney_hits) == 1: # return the col if there is exaclty one match.
        return kidney_hits[0]
    if len(kidney_hits) > 1: # If there is more than one kidney-like CC variable, print a warning
        print(f"[WARN] Multiple kidney-like CC vars found: {kidney_hits}. Using first.")
        return kidney_hits[0]

    raise ValueError( # If neither search strategy found a usable CKD variable, raise an error.
        "Could not identify a CKD indicator column. "
        "Please inspect cc_vars / event_df.columns and set ckd_var manually."
    )

def safe_cmd_suffix(text: str) -> str: # takes a row key and turns it into a LaTeX-safe command-name piece.
    s = re.sub(r"[^A-Za-z0-9]+", "", str(text)) # convert text to a string, then removes everything that is not a letter or number.
    if not s:
        s = "Row" # if the cleaned string becomes empty, use "Row" instead.
    if s[0].isdigit():
        s = "Row" + s # if the cleaned string starts with a number, prepend "Row".
    return s

def macro_name(key: str) -> str: # creates the actual LaTeX command name the table file will define.
    return f"\\{TABLE_PREFIX}{safe_cmd_suffix(key)}" # returns a string that looks like a LaTeX command.

def macro_line(key: str, value: str, label: str = None) -> str: # a function that creates one full line of output for the .tex macro file.
    pieces = []
    if label is not None:
        clean_label = str(label).replace("\n", " ").strip()
        pieces.append(f"% {clean_label}")
    pieces.append(rf"\providecommand{{{macro_name(key)}}}{{{value}}}")
    return "\n".join(pieces)

# ... Read the yearly files and reduce them to one row per bene-storm (event) ...
# Goal: clean up analytical (do not need two rows) before building Table 1

frames = [] # store one cleaned df per year in this list, then combine them later.

for year in range(YEAR_MIN, YEAR_MAX + 1):
    f = input_file(year)
    if not _exists(f):
        print(f"[SKIP] missing input: {f}")
        continue

    df = pd.read_csv(f, low_memory=False)
    df["year"] = year

    if "week_rel" not in df.columns:
        raise ValueError(f"{year}: missing week_rel in input file.")

    df = df[df["week_rel"] == 0].copy() # keep only rows where week_rel == 0.

    required = ["event_id", "BENE_ID"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"{year}: missing required columns after read: {missing_required}")

    dup_pair = df.duplicated(subset=["event_id", "BENE_ID"]).sum() # QC check. If week_rel == 0 truly gives one row per event, then each event_id should appear only once.
    if dup_pair > 0:
        raise ValueError(
            f"{year}: found {dup_pair:,} duplicated event_id-BENE_ID rows after restricting to week_rel == 0."
        )

    frames.append(df)

if not frames:
    raise ValueError("No yearly input files found.")

event_df = pd.concat(frames, axis=0, ignore_index=True) # stack all the yearly df in frames vertically into one combined df called event_df.
print(f"[INFO] event-bene-level rows after stacking all years = {len(event_df):,}")
print(f"[INFO] unique event_id-BENE_ID pairs = {event_df[['event_id', 'BENE_ID']].drop_duplicates().shape[0]:,}")

# ... Clean / derive some analysis variables (like age) for Table 1...

for c in ["BENE_ID", "storm_id", "facility_county_fips"]:
    if c in event_df.columns:
        event_df[c] = _as_clean_str(event_df[c])

event_df["event_id"] = pd.to_numeric(event_df["event_id"], errors="coerce").astype("Int64")
event_df["anchor_dt"] = pd.to_datetime(event_df["anchor_dt"], errors="coerce")
event_df["BENE_BIRTH_DT"] = pd.to_datetime(event_df["BENE_BIRTH_DT"], errors="coerce")

event_df["age"] = ( # create age
    (event_df["anchor_dt"] - event_df["BENE_BIRTH_DT"]).dt.days / 365.25
)

event_df[GROUP_VAR] = pd.to_numeric(event_df[GROUP_VAR], errors="coerce")

before_n = len(event_df)
event_df = event_df[event_df[GROUP_VAR].isin([0, 1])].copy() # Keeps only rows where the grouping variable (e.g., early dialysis indicator) is either 0 or 1. If early indicator is missing, or some value like 2, that row gets dropped. This is important because the later comparison is specifically between two groups: 1 = early 0 = non-early
print(f"[INFO] kept {len(event_df):,} event-bene rows with {GROUP_VAR} in {{0,1}}; dropped {before_n - len(event_df):,}")

# CBSA indicators
if "cbsa_indicator_2020_ahrf" in event_df.columns: # check whether the AHRF CBSA variable exists in the dataset (metro vs micro)
    cbsa_num = pd.to_numeric(event_df["cbsa_indicator_2020_ahrf"], errors="coerce")
    event_df["cbsa_metro"] = np.where(cbsa_num.isna(), np.nan, (cbsa_num == 1).astype(float))
    event_df["cbsa_micro"] = np.where(cbsa_num.isna(), np.nan, (cbsa_num == 2).astype(float))
    event_df["cbsa_neither"] = np.where(cbsa_num.isna(), np.nan, (cbsa_num == 0).astype(float))
else:
    event_df["cbsa_metro"] = np.nan
    event_df["cbsa_micro"] = np.nan
    event_df["cbsa_neither"] = np.nan

# ... Harmonize year-specific AHRF variables ...
# Goal: creates one harmonized variable from many year-specific AHRF cols.
# e.g., So many year-specific cols like "median_hh_income_2011_ahrf", "median_hh_income_2012_ahrf", "median_hh_income_2013_ahrf" into one single column "median_hh_income"

def make_harmonized_year_specific(prefix: str, outcol: str):
    year_cols = {y: f"{prefix}_{y}_ahrf" for y in range(YEAR_MIN, YEAR_MAX + 1)} # build a dict mapping each year to the matching AHRF column name. E.g., if prefix = "median_hh_income" then 2011 -> median_hh_income_2011_ahrf
    event_df[outcol] = np.nan # start with missing values
    for y, c in year_cols.items(): # loop through each year and its matching year-specific column name.
        if c in event_df.columns: # check whether that year-specific AHRF column actually exists in the dataset.
            mask = event_df["year"] == y # create a row for year
            event_df.loc[mask, outcol] = pd.to_numeric(event_df.loc[mask, c], errors="coerce") # for rows in year y, copies the values from that year’s AHRF column into the harmonized output column, converting them to numeric.

# Apply above function. Why? This makes the later table code simpler b/c it can refer to one column like median_hh_income instead of writing year-by-year logic everywhere.
make_harmonized_year_specific("median_hh_income", "median_hh_income")
make_harmonized_year_specific("pct_below_poverty", "pct_below_poverty")
make_harmonized_year_specific("pct_population_65plus", "pct_population_65plus")
make_harmonized_year_specific("pct_gen_pract_md_patientcare", "pct_gen_pract_md_patientcare")
make_harmonized_year_specific("pct_st_gen_hospital_beds_of_all_hospital_beds", "pct_st_gen_hospital_beds_of_all_hospital_beds")
make_harmonized_year_specific("st_gen_hospital_beds_per_1000_pop", "st_gen_hospital_beds_per_1000_pop")

# ... Variable specification ...
# Define row specifications for the table. 
# Tuples: demo_rows, clinical_rows, and county_rows. 
# Each tuple gives: a macro key, a readable label for table, the col name in event_df, and if the variable should be continuous or binary
# e.g., ("FemalePct", "Female (%)", "sex_female", "binary")

demo_rows = [
    ("Age", "Age", "age", "continuous"),
    ("FemalePct", "Female (%)", "sex_female", "binary"),
    ("NonHispanicWhitePct", "Non-Hispanic White (%)", "race_nh_white", "binary"),
    ("BlackPct", "Black (%)", "race_black", "binary"),
    ("OtherRacePct", "Other race (%)", "race_other", "binary"),
    ("AsianPacificIslanderPct", "Asian / Pacific Islander (%)", "race_asian_pi", "binary"),
    ("HispanicPct", "Hispanic (%)", "race_hispanic", "binary"),
    ("AmericanIndianAlaskaNativePct", "American Indian / Alaska Native (%)", "race_ai_an", "binary"),
]

clinical_rows = [
    ("Combinedscore", "Combinedscore", "combinedscore", "continuous"),
    ("NumCCConditions", "Number of CC conditions", "n_cc_conditions", "continuous"),
    ("NumOTCCConditions", "Number of OTCC conditions", "n_otcc_conditions", "continuous"),
    ("DualPct", "Dual (%)", "dual", "binary"),
    ("MedicareESRDOnlyPct", "Medicare ESRD only (%)", "medicare_esrd_only", "binary"),
]

county_rows = [
    ("VmaxSustainedWindSpeed", "Vmax sustained wind speed", "vmax_sust", "continuous"),
    ("MedianHouseholdIncome", "Median household income", "median_hh_income", "continuous"),
    ("BelowPovertyPct", "Below poverty (%)", "pct_below_poverty", "continuous"),
    ("PopulationAgeSixtyFivePlusPct", "Population age 65+ (%)", "pct_population_65plus", "continuous"),
    ("MetroPct", "Metro (%)", "cbsa_metro", "binary"),
    ("MicroPct", "Micro (%)", "cbsa_micro", "binary"),
    ("NeitherPct", "Neither (%)", "cbsa_neither", "binary"),
    ("GeneralPracticeMDsPct", "General practice MDs (%)", "pct_gen_pract_md_patientcare", "continuous"),
    ("ShortTermGeneralHospitalBedsPerThousandPop", "Short-term general hospital beds per 1,000 pop", "st_gen_hospital_beds_per_1000_pop", "continuous"),
]

exclude_cols = { # set of variables that should not be scanned as candidate binary condition indicators.
    # identifiers / structure
    "year", "storm_year", "storm_id", "event_id", "BENE_ID", "facility_county_fips",
    "week_rel", "hazard_week",

    # appended / summary flags
    "combinedscore", "combinedscore_missing",
    "ahrf_missing", "wind_missing",
    "vmax_sust",
    "gap_days", "no_hazard_dialysis",
    "schedule_type", "stable_3x_weekly", GROUP_VAR,
    "anchor_dt", "anchor_dow", "anchor_on_usual_sched_day", "anchor_on_off_sched_day",
    "BENE_DEATH_DT", "BENE_BIRTH_DT",
    "cbsa_indicator_2020_ahrf", "cbsa_indicator_2020_label_ahrf",
    "cbsa_metro", "cbsa_micro", "cbsa_neither",
    "age",
    "sex_unknown", "sex_male", "sex_female",
    "race_unknown", "race_nh_white", "race_black", "race_other",
    "race_asian_pi", "race_hispanic", "race_ai_an",
    "dual", "esrd",
    "medicare_aged", "medicare_disabled", "medicare_esrd_only", "medicare_with_esrd",

    # summary counts handled explicitly
    "n_cc_conditions", "n_otcc_conditions", "n_all_conditions",

    # outcomes / process vars
    "any_ip", "any_ed", "any_death", "n_dialysis", "disrupt",
    "any_ip_cmp_wk", "any_ip_cmp_2wk", "any_ip_cmp_3wk", "any_ip_cmp_4wk",
    "any_ed_cmp_wk", "any_ed_cmp_2wk", "any_ed_cmp_3wk", "any_ed_cmp_4wk",
    "any_death_cmp_wk", "any_death_cmp_2wk", "any_death_cmp_3wk", "any_death_cmp_4wk",
    "any_ip_wk1", "any_ip_wk2", "any_ip_wk3",
    "any_ip_post_2wk", "any_ip_post_3wk", "any_ip_post_4wk",
    "any_ed_wk1", "any_ed_wk2", "any_ed_wk3",
    "any_ed_post_2wk", "any_ed_post_3wk", "any_ed_post_4wk",
    "any_death_wk1", "any_death_wk2", "any_death_wk3",
    "any_death_post_2wk", "any_death_post_3wk", "any_death_post_4wk",
    "n_dialysis_wk_m2", "n_dialysis_wk0",
}

known_county_vars = { # a set of all the year-specific AHRF variable names, like median_hh_income_2017_ahrf or pct_below_poverty_2020_ahrf. These are excluded from the CC/OTCC scan too, because they not binary condition flags used for table 1.
    f"median_hh_income_{y}_ahrf" for y in range(YEAR_MIN, YEAR_MAX + 1)
}.union({
    f"pct_below_poverty_{y}_ahrf" for y in range(YEAR_MIN, YEAR_MAX + 1)
}).union({
    f"pct_population_65plus_{y}_ahrf" for y in range(YEAR_MIN, YEAR_MAX + 1)
}).union({
    f"pct_gen_pract_md_patientcare_{y}_ahrf" for y in range(YEAR_MIN, YEAR_MAX + 1)
}).union({
    f"pct_st_gen_hospital_beds_of_all_hospital_beds_{y}_ahrf" for y in range(YEAR_MIN, YEAR_MAX + 1)
}).union({
    f"st_gen_hospital_beds_per_1000_pop_{y}_ahrf" for y in range(YEAR_MIN, YEAR_MAX + 1)
})

candidate_binary = []
for c in event_df.columns: # loop over all columns in event_df
    if c in exclude_cols or c in known_county_vars: # # skip columns that are excluded or known county variables
        continue

    cname = str(c).lower()
    if (
        cname.startswith("any_ip") or
        cname.startswith("any_ed") or
        cname.startswith("any_death") or
        cname.startswith("disrupt") or
        cname == "n_dialysis" or
        cname.startswith("n_dialysis_")
    ):
        continue

    vals = pd.to_numeric(event_df[c], errors="coerce") # convert the remaining column that will be used for table 1 to numeric
    uniq = set(pd.Series(vals.dropna()).unique().tolist()) # check its non-missing unique values
    if len(uniq) > 0 and uniq.issubset({0, 1}): # if all observed values are only 0 and 1, treat it as a binary candidate
        candidate_binary.append(c)

otcc_vars = sorted([c for c in candidate_binary if c.endswith("_medicare") or "__otcc" in c]) # splits the binary candidates into otcc_vars: variables that look like OTCC indicators
cc_vars = sorted([c for c in candidate_binary if c not in otcc_vars]) # splits the binary candidates into: cc_vars: variables that look like CC indicators

# ... Create summary condition counts ...
# sums all CC indicators across each row to create n_cc_conditions. Same for n_otcc_conditions and n_all_conditions
if cc_vars:
    event_df["n_cc_conditions"] = event_df[cc_vars].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
else:
    event_df["n_cc_conditions"] = np.nan

if otcc_vars:
    event_df["n_otcc_conditions"] = event_df[otcc_vars].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
else:
    event_df["n_otcc_conditions"] = np.nan

if cc_vars or otcc_vars:
    event_df["n_all_conditions"] = (
        pd.to_numeric(event_df["n_cc_conditions"], errors="coerce").fillna(0) +
        pd.to_numeric(event_df["n_otcc_conditions"], errors="coerce").fillna(0)
    )
else:
    event_df["n_all_conditions"] = np.nan

# Turn the individual CC and OTCC variables into row for the table
cc_rows = [(f"CC_{v}", nice_label(v), v, "binary") for v in cc_vars]
otcc_rows = [(f"OTCC_{v}", nice_label(v), v, "binary") for v in otcc_vars]

ckd_var = find_ckd_variable(event_df, cc_vars) # identify which col is the CKD flag and prints the chosen variable name. That CKD variable is used for export and for restricting the sample later.
print(f"[INFO] using CKD restriction variable: {ckd_var}")

# ... Build CKD flag export BEFORE restricting to CKD-only events ...
event_ckd_export = event_df[
    ["year", "storm_id", "event_id", "BENE_ID", "facility_county_fips"]
].copy() # creates a smaller export dataset with some col

if "facility_id" in event_df.columns: # facility_id col not relevant for bene characteristic table 1. But, left unchanged for now
    event_ckd_export["facility_id"] = event_df["facility_id"]
else:
    event_ckd_export["facility_id"] = pd.NA

event_ckd_export["ckd_ind"] = pd.to_numeric(event_df[ckd_var], errors="coerce")
event_ckd_export["ckd_ind"] = np.where(event_ckd_export["ckd_ind"] == 1, 1, 0) # basically forces any NaN into 0's

for c in ["BENE_ID", "storm_id", "facility_county_fips", "facility_id"]:
    if c in event_ckd_export.columns:
        event_ckd_export[c] = _as_clean_str(event_ckd_export[c])

event_ckd_export = event_ckd_export.drop_duplicates().copy()

print("[QC] CKD export counts:")
print(event_ckd_export["ckd_ind"].value_counts(dropna=False).sort_index())

event_ckd_export.to_csv(OUT_CKD_FLAG_CSV, index=False)
print(f"[WROTE] CKD flag export: {OUT_CKD_FLAG_CSV}")

# ... Restrict to CKD-only events ...
before_ckd_n = len(event_df)
event_df = event_df[pd.to_numeric(event_df[ckd_var], errors="coerce") == 1].copy() # keep only rows where the CKD indicator equals 1.
print(f"[INFO] kept {len(event_df):,} CKD event-bene rows; dropped {before_ckd_n - len(event_df):,}")

if len(event_df) == 0:
    raise ValueError("After restricting to CKD == 1, no rows remain.")

print("[QC] CKD percent after restriction:")
print(f"      overall = {prop_percent(event_df[ckd_var]):.6f}")

event_df.to_csv(OUT_EVENT_CSV, index=False)
print(f"[WROTE] event-bene-level table dataset: {OUT_EVENT_CSV}")

# ... Create the two comparison groups ...
g1 = event_df[event_df[GROUP_VAR] == 1].copy() # early group
g0 = event_df[event_df[GROUP_VAR] == 0].copy() # non-early group

if len(g1) == 0 or len(g0) == 0: # check that both groups are non-empty
    raise ValueError(
        f"After CKD restriction, one group is empty. early_n={len(g1):,}, nonearly_n={len(g0):,}"
    )

print(f"[INFO] CKD-only sample sizes: overall={len(event_df):,}, early={len(g1):,}, nonearly={len(g0):,}")
print(f"[QC] CKD in early group = {prop_percent(g1[ckd_var]):.6f}")
print(f"[QC] CKD in non-early group = {prop_percent(g0[ckd_var]):.6f}")

qc_vars = [
    ckd_var,
    "sex_female", "combinedscore", "dual", "medicare_esrd_only",
    "n_cc_conditions", "n_otcc_conditions"
]
print("\n[QC] Unrounded group means / percents:")
for v in qc_vars: # For each var: if it is binary, it prints percentages using prop_percent() if it is continuous, it prints means using mean_ignore_na()
    if v not in event_df.columns:
        continue

    if v in [ckd_var, "sex_female", "dual", "medicare_esrd_only"]:
        overall = prop_percent(event_df[v])
        early = prop_percent(g1[v])
        nonearly = prop_percent(g0[v])
        print(f"{v:22s} overall={overall:.6f} early={early:.6f} nonearly={nonearly:.6f}")
    else:
        overall = mean_ignore_na(event_df[v])
        early = mean_ignore_na(g1[v])
        nonearly = mean_ignore_na(g0[v])
        print(f"{v:22s} overall={overall:.6f} early={early:.6f} nonearly={nonearly:.6f}")

# ... Compute LaTeX macros for table ...
rows_tex = [] # store the LaTeX macro lines for the table, one row at a time

rows_tex.append(macro_line("N", f"{fmt_int(len(g1))} & {fmt_int(len(g0))}", "N")) # sample sizes

def add_stat_macro(key: str, label: str, var: str, kind: str): # builds table rows
    if var not in event_df.columns:
        print(f"[WARN] missing variable for table: {var}")
        rows_tex.append(macro_line(key, " & ", f"{label} [missing variable: {var}]"))
        return

    if kind == "continuous":
        early = mean_ignore_na(g1[var])
        nonearly = mean_ignore_na(g0[var])
        p = ttest_pvalue(g1[var], g0[var])

        early_str = fmt_num(early, DISPLAY_DECIMALS_CONT) + star_from_p(p)
        nonearly_str = fmt_num(nonearly, DISPLAY_DECIMALS_CONT)
        rows_tex.append(macro_line(key, f"{early_str} & {nonearly_str}", label))

    elif kind == "binary":
        early = prop_percent(g1[var])
        nonearly = prop_percent(g0[var])
        p = prop_test_pvalue(g1[var], g0[var])

        early_str = fmt_num(early, DISPLAY_DECIMALS_PCT) + star_from_p(p)
        nonearly_str = fmt_num(nonearly, DISPLAY_DECIMALS_PCT)
        rows_tex.append(macro_line(key, f"{early_str} & {nonearly_str}", label))

# Build the table rows for: demographic, clinical, CC condition, OTCC condition, and county.
for spec in demo_rows:
    add_stat_macro(*spec)

for spec in clinical_rows:
    add_stat_macro(*spec)

for spec in cc_rows:
    add_stat_macro(*spec)

for spec in otcc_rows:
    add_stat_macro(*spec)

for spec in county_rows:
    add_stat_macro(*spec)

# ... Write tex file or LaTeX to create table ...
with open(OUT_MACRO_TEX, "w", encoding="utf-8") as f:
    f.write("% Auto-generated LaTeX macros for disrupted beneficiary sample\n")
    f.write("% Each macro expands to: <early value> & <non-early value>\n\n")
    f.write("\n\n".join(rows_tex))
    f.write("\n")

print(f"[WROTE] macro tex: {OUT_MACRO_TEX}")
print("[DONE]")

