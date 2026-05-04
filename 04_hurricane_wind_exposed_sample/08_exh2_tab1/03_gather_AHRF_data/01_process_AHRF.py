#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 29, 2026
# Description: This script builds a county-level AHRF characteristics file by reading selected variables from the 2019–2020 
# and 2021–2022 AHRF fixed-width files plus the 2023–2024 AHRF CSV, cleaning and standardizing FIPS and numeric fields, 
# and merging the three sources into one county dataset. It then constructs annual characteristics for median household 
# income, poverty, percent age 65+, CBSA status, the share of patient-care physicians who are general practitioners, 
# and short-term general hospital bed capacity. Note that not every year of AHRF are available. Thus, for each missing year
# we will use the closest year (e.g., using 2011-2015 AHRF data for 2012 or using 2014-2018 AHRF data for 2016.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import pandas as pd
from pathlib import Path

# -------------------------
# Paths
# -------------------------

file1_path = Path("/gpfs/data/public/AHRF/2020/DATA/AHRF2020.asc")
file2_path = Path("/gpfs/data/public/AHRF/2022/DATA/ahrf2022.asc")
file3_path = Path("/gpfs/data/public/AHRF/2024/AHRF_2023-2024_CSV/ahrf2024_Feb2025.csv")

out_dir = Path("/gpfs/data/public/AHRF/derived")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "ahrf_controls_pct_only_final_phys_optionA_stgenbeds.csv"

# -------------------------
# Functions
# -------------------------
def doc_to_colspec(start, end):
    # AHRF docs are 1-based inclusive
    # pandas fixed-width uses 0-based, end-exclusive
    return (start - 1, end)

def clean_string_df(df):
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].str.strip()
    return df.replace({
        "": pd.NA,
        ".": pd.NA,
        "NA": pd.NA,
        "nan": pd.NA,
    })

def force_fips(df, col="fips"):
    df[col] = df[col].astype("string").str.strip().str.zfill(5)
    return df

def convert_numeric_except(df, exclude=("fips", "cbsa_indicator_2020_label")):
    for c in df.columns:
        if c not in exclude:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def apply_interval_fill(df, source_prefix, target_years, mapping):
    """
    Creates columns like {source_prefix}_filled_{year}
    by borrowing from benchmark years according to mapping.
    """
    out = df.copy()
    for yr in target_years:
        src_yr = mapping.get(yr)
        src_col = f"{source_prefix}_{src_yr}"
        out[f"{source_prefix}_filled_{yr}"] = out[src_col] if src_col in out.columns else pd.NA
    return out

def print_duplicate_fips(df, name):
    dup_n = df["fips"].duplicated().sum()
    print(f"{name} duplicate fips: {dup_n}")

# =========================================================
# FILE 1: AHRF 2019-2020 ASCII
# =========================================================
# Some notes on physician/hosp num/denom:

# Physician variables:
# numerator   = MD's, Tot Gen Pract, Tot Ptn Cr
# denominator = M.D.'s, Total Ptn Care Non-Fed
#
# Hospital bed variables:
# numerator   = Short Term General Hosp Beds
# denominator = Hospital Beds
# =========================================================
var_specs_1 = [
    {"name": "fips", "start": 2, "end": 6},

    # Median household income
    {"name": "median_hh_income_2018", "start": 24300, "end": 24305},
    {"name": "median_hh_income_2017", "start": 24306, "end": 24311},
    {"name": "median_hh_income_2016", "start": 24312, "end": 24317},
    {"name": "median_hh_income_2015", "start": 24318, "end": 24323},
    {"name": "median_hh_income_2014", "start": 24324, "end": 24329},
    {"name": "median_hh_income_2013", "start": 24330, "end": 24335},
    {"name": "median_hh_income_2012", "start": 24336, "end": 24341},
    {"name": "median_hh_income_2011", "start": 24342, "end": 24347},
    {"name": "median_hh_income_2010", "start": 24348, "end": 24353},

    # Poverty windows
    {"name": "pct_below_poverty_2014_2018", "start": 25424, "end": 25427},
    {"name": "pct_below_poverty_2011_2015", "start": 25428, "end": 25431},

    # Population
    {"name": "population_2019", "start": 16860, "end": 16867},
    {"name": "population_2018", "start": 16868, "end": 16875},
    {"name": "population_2017", "start": 16876, "end": 16883},
    {"name": "population_2016", "start": 16884, "end": 16891},
    {"name": "population_2015", "start": 16892, "end": 16899},
    {"name": "population_2014", "start": 16900, "end": 16907},
    {"name": "population_2013", "start": 16908, "end": 16915},
    {"name": "population_2012", "start": 16916, "end": 16923},
    {"name": "population_2011", "start": 16924, "end": 16931},

    # Population 65+
    {"name": "population_65plus_2018", "start": 19318, "end": 19324},
    {"name": "population_65plus_2017", "start": 19325, "end": 19331},
    {"name": "population_65plus_2016", "start": 19332, "end": 19338},
    {"name": "population_65plus_2015", "start": 19339, "end": 19345},
    {"name": "population_65plus_2014", "start": 19346, "end": 19352},
    {"name": "population_65plus_2013", "start": 19353, "end": 19359},
    {"name": "population_65plus_2012", "start": 19360, "end": 19366},
    {"name": "population_65plus_2011", "start": 19367, "end": 19373},

    # CBSA
    {"name": "cbsa_indicator_2020", "start": 222, "end": 222},

    # Option A denominator: M.D.'s, Total Ptn Care Non-Fed
    {"name": "md_total_patient_care_2018", "start": 921, "end": 925},
    {"name": "md_total_patient_care_2017", "start": 926, "end": 930},
    {"name": "md_total_patient_care_2016", "start": 931, "end": 935},
    {"name": "md_total_patient_care_2015", "start": 936, "end": 940},
    {"name": "md_total_patient_care_2014", "start": 941, "end": 945},
    {"name": "md_total_patient_care_2013", "start": 946, "end": 950},
    {"name": "md_total_patient_care_2012", "start": 951, "end": 955},
    {"name": "md_total_patient_care_2011", "start": 956, "end": 960},
    {"name": "md_total_patient_care_2010", "start": 961, "end": 965},

    # Option A numerator: MD's, Tot Gen Pract, Tot Ptn Cr
    {"name": "md_gen_pract_patient_care_2018", "start": 1318, "end": 1321},
    {"name": "md_gen_pract_patient_care_2015", "start": 1322, "end": 1325},
    {"name": "md_gen_pract_patient_care_2010", "start": 1326, "end": 1329},

    # Hospital beds (staffed)
    {"name": "hospital_beds_2018", "start": 12145, "end": 12150},
    {"name": "hospital_beds_2015", "start": 12151, "end": 12156},
    {"name": "hospital_beds_2010", "start": 12157, "end": 12162},

    # Short-term general hospital beds (staffed)
    {"name": "st_gen_hospital_beds_2018", "start": 12163, "end": 12168},
    {"name": "st_gen_hospital_beds_2015", "start": 12169, "end": 12174},
    {"name": "st_gen_hospital_beds_2010", "start": 12175, "end": 12180},
]

colspecs_1 = [doc_to_colspec(v["start"], v["end"]) for v in var_specs_1]
colnames_1 = [v["name"] for v in var_specs_1]

with open(file1_path, "r", encoding="latin1", errors="replace") as f:
    df1 = pd.read_fwf(f, colspecs=colspecs_1, names=colnames_1, dtype=str)

df1 = clean_string_df(df1)
df1 = force_fips(df1, "fips")
df1 = convert_numeric_except(df1, exclude=("fips",))
print_duplicate_fips(df1, "df1")

# =========================================================
# FILE 2: AHRF 2021-2022 ASCII
# =========================================================
# Some notes on physician/hosp num/denom:

# Physician vars:
# numerator   = MD's, Tot Gen Pract, Tot Ptn Cr
# denominator = M.D.'s, Total Ptn Care Non-Fed
#
# Hospital bed variables:
# numerator   = Short Term General Hosp Beds
# denominator = Hospital Beds
# =========================================================
var_specs_2 = [
    {"name": "fips", "start": 2, "end": 6},

    # Median household income
    {"name": "median_hh_income_2020", "start": 23975, "end": 23980},
    {"name": "median_hh_income_2019", "start": 23981, "end": 23986},

    # Poverty window
    {"name": "pct_below_poverty_2016_2020", "start": 25017, "end": 25020},

    # Population
    {"name": "population_2021", "start": 17450, "end": 17457},
    {"name": "population_2020", "start": 17466, "end": 17473},

    # Population 65+
    {"name": "population_65plus_2020", "start": 18878, "end": 18884},
    {"name": "population_65plus_2019", "start": 18885, "end": 18891},

    # CBSA
    {"name": "cbsa_indicator_2020", "start": 222, "end": 222},

    # Option A denominator: M.D.'s, Total Ptn Care Non-Fed
    {"name": "md_total_patient_care_2020", "start": 991, "end": 995},
    {"name": "md_total_patient_care_2019", "start": 996, "end": 1000},

    # Option A numerator: MD's, Tot Gen Pract, Tot Ptn Cr
    {"name": "md_gen_pract_patient_care_2020", "start": 1464, "end": 1467},

    # Hospital beds (staffed)
    {"name": "hospital_beds_2020", "start": 12713, "end": 12718},
    {"name": "hospital_beds_2015", "start": 12719, "end": 12724},
    {"name": "hospital_beds_2010", "start": 12725, "end": 12730},

    # Short-term general hospital beds (staffed)
    {"name": "st_gen_hospital_beds_2020", "start": 12731, "end": 12736},
    {"name": "st_gen_hospital_beds_2015", "start": 12737, "end": 12742},
    {"name": "st_gen_hospital_beds_2010", "start": 12743, "end": 12748},
]

colspecs_2 = [doc_to_colspec(v["start"], v["end"]) for v in var_specs_2]
colnames_2 = [v["name"] for v in var_specs_2]

with open(file2_path, "r", encoding="latin1", errors="replace") as f:
    df2 = pd.read_fwf(f, colspecs=colspecs_2, names=colnames_2, dtype=str)

df2 = clean_string_df(df2)
df2 = force_fips(df2, "fips")
df2 = convert_numeric_except(df2, exclude=("fips",))
print_duplicate_fips(df2, "df2")

# =========================================================
# FILE 3: AHRF 2023-2024 CSV
# =========================================================
# Some notes on physician/hosp num/denom:

# Physician var:
# numerator   = md_nf_all_gp_all_pc_21 / _22
# denominator = md_nf_all_pc_21 / _22
#
# Hospital bed variables:
# numerator   = stgh_hosp_beds_21 / _22
# denominator = hosp_beds_21 / _22
# =========================================================
usecols_3 = [
    "fips_st_cnty",

    # income / poverty / pop
    "medn_hhi_saipe_22",
    "medn_hhi_saipe_21",
    "pers_lt_fpl_pct_22",
    "popn_est_23",
    "popn_est_22",
    "popn_est_ge65_22",
    "popn_est_ge65_21",

    # Option A denominator
    "md_nf_all_pc_22",
    "md_nf_all_pc_21",

    # Option A numerator
    "md_nf_all_gp_all_pc_22",
    "md_nf_all_gp_all_pc_21",

    # staffed hospital beds
    "hosp_beds_22",
    "hosp_beds_21",
    "stgh_hosp_beds_22",
    "stgh_hosp_beds_21",
]

rename_map_3 = {
    "fips_st_cnty": "fips",

    # income / poverty / pop
    "medn_hhi_saipe_22": "median_hh_income_2022",
    "medn_hhi_saipe_21": "median_hh_income_2021",
    "pers_lt_fpl_pct_22": "pct_below_poverty_2018_2022",
    "popn_est_23": "population_2023",
    "popn_est_22": "population_2022",
    "popn_est_ge65_22": "population_65plus_2022",
    "popn_est_ge65_21": "population_65plus_2021",

    # Option A denominator
    "md_nf_all_pc_22": "md_total_patient_care_2022",
    "md_nf_all_pc_21": "md_total_patient_care_2021",

    # Option A numerator
    "md_nf_all_gp_all_pc_22": "md_gen_pract_patient_care_2022",
    "md_nf_all_gp_all_pc_21": "md_gen_pract_patient_care_2021",

    # staffed hospital beds
    "hosp_beds_22": "hospital_beds_2022",
    "hosp_beds_21": "hospital_beds_2021",
    "stgh_hosp_beds_22": "st_gen_hospital_beds_2022",
    "stgh_hosp_beds_21": "st_gen_hospital_beds_2021",
}

df3 = pd.read_csv(file3_path, usecols=usecols_3, dtype=str, low_memory=False)
df3 = df3.rename(columns=rename_map_3)
df3 = clean_string_df(df3)
df3 = force_fips(df3, "fips")
df3 = convert_numeric_except(df3, exclude=("fips",))
print_duplicate_fips(df3, "df3")

# =========================================================
# Merge all files
# =========================================================
df12 = df1.merge(df2, on="fips", how="outer", suffixes=("_f1", "_f2"), validate="one_to_one")

# resolve CBSA overlap
if "cbsa_indicator_2020_f1" in df12.columns and "cbsa_indicator_2020_f2" in df12.columns:
    df12["cbsa_indicator_2020"] = df12["cbsa_indicator_2020_f2"].combine_first(df12["cbsa_indicator_2020_f1"])
    df12 = df12.drop(columns=["cbsa_indicator_2020_f1", "cbsa_indicator_2020_f2"])

# resolve overlapping bed variables (prefer newer file2)
bed_overlap_years = [2010, 2015]
for yr in bed_overlap_years:
    hb_f1 = f"hospital_beds_{yr}_f1"
    hb_f2 = f"hospital_beds_{yr}_f2"
    st_f1 = f"st_gen_hospital_beds_{yr}_f1"
    st_f2 = f"st_gen_hospital_beds_{yr}_f2"

    if hb_f1 in df12.columns and hb_f2 in df12.columns:
        df12[f"hospital_beds_{yr}"] = df12[hb_f2].combine_first(df12[hb_f1])
        df12 = df12.drop(columns=[hb_f1, hb_f2])

    if st_f1 in df12.columns and st_f2 in df12.columns:
        df12[f"st_gen_hospital_beds_{yr}"] = df12[st_f2].combine_first(df12[st_f1])
        df12 = df12.drop(columns=[st_f1, st_f2])

df = df12.merge(df3, on="fips", how="outer", validate="one_to_one")

# =========================================================
# CBSA label
# =========================================================
cbsa_map = {0: "neither", 1: "metro", 2: "micro"}
df["cbsa_indicator_2020_label"] = df["cbsa_indicator_2020"].map(cbsa_map)

# =========================================================
# Poverty scaling
# Older fixed-width poverty fields have one implied decimal.
# New CSV-era 2018-2022 field is already a real percent.
# Basically, need to convert due to how data was stored
# =========================================================
poverty_implied_decimal_cols = [
    "pct_below_poverty_2011_2015",
    "pct_below_poverty_2014_2018",
    "pct_below_poverty_2016_2020",
]

for c in poverty_implied_decimal_cols:
    if c in df.columns:
        df[c] = df[c] / 10.0

# =========================================================
# Create assigned annual poverty series, 2011-2022
# =========================================================

# AHRF does not have data for each year so we will use certain available years for the missing years

poverty_fill_map = {
    2011: "pct_below_poverty_2011_2015", # using 2011-2015 data for 2011
    2012: "pct_below_poverty_2011_2015", # using 2011-2015 data for 2012
    2013: "pct_below_poverty_2011_2015", # etc...
    2014: "pct_below_poverty_2011_2015",
    2015: "pct_below_poverty_2014_2018", # using 2014-2018 data for 2015
    2016: "pct_below_poverty_2014_2018", # using 2014-2018 data for 2016
    2017: "pct_below_poverty_2014_2018", # etc...
    2018: "pct_below_poverty_2016_2020",
    2019: "pct_below_poverty_2016_2020",
    2020: "pct_below_poverty_2016_2020",
    2021: "pct_below_poverty_2018_2022",
    2022: "pct_below_poverty_2018_2022",
}

for yr, src_col in poverty_fill_map.items():
    df[f"pct_below_poverty_{yr}"] = df[src_col] if src_col in df.columns else pd.NA

# =========================================================
# Percent population 65+, 2011-2022
# =========================================================
years_2010_2022 = list(range(2010, 2023))
years_2011_2022 = list(range(2011, 2023))

for yr in years_2011_2022:
    pop_col = f"population_{yr}"
    pop65_col = f"population_65plus_{yr}"
    out_col = f"pct_population_65plus_{yr}"

    if pop_col in df.columns and pop65_col in df.columns:
        df[out_col] = pd.NA
        valid = df[pop_col].notna() & (df[pop_col] > 0) & df[pop65_col].notna()
        df.loc[valid, out_col] = (df.loc[valid, pop65_col] / df.loc[valid, pop_col]) * 100
    else:
        df[out_col] = pd.NA

# =========================================================
# Physician percent
# =========================================================
phys_benchmark_years = [2010, 2015, 2018, 2020, 2021, 2022]

for yr in phys_benchmark_years:
    num_col = f"md_gen_pract_patient_care_{yr}"
    den_col = f"md_total_patient_care_{yr}"
    out_col = f"pct_gen_pract_md_patientcare_benchmark_{yr}"

    if num_col in df.columns and den_col in df.columns:
        df[out_col] = pd.NA
        valid = df[den_col].notna() & (df[den_col] > 0) & df[num_col].notna()
        df.loc[valid, out_col] = (df.loc[valid, num_col] / df.loc[valid, den_col]) * 100
    else:
        df[out_col] = pd.NA

phys_ratio_fill_map = {
    2010: 2010,
    2011: 2010,
    2012: 2010,
    2013: 2015,
    2014: 2015,
    2015: 2015,
    2016: 2015,
    2017: 2018,
    2018: 2018,
    2019: 2018,
    2020: 2020,
    2021: 2021,
    2022: 2022,
}

for yr, src_yr in phys_ratio_fill_map.items():
    src_col = f"pct_gen_pract_md_patientcare_benchmark_{src_yr}"
    out_col = f"pct_gen_pract_md_patientcare_{yr}"
    df[out_col] = df[src_col] if src_col in df.columns else pd.NA

# =========================================================
# Short-term general hospital bed measures
# =========================================================
bed_fill_map = {
    2010: 2010,
    2011: 2010,
    2012: 2010,
    2013: 2015,
    2014: 2015,
    2015: 2015,
    2016: 2015,
    2017: 2015,
    2018: 2020,
    2019: 2020,
    2020: 2020,
    2021: 2021,
    2022: 2022,
}
bed_years = list(range(2010, 2023))

df = apply_interval_fill(df, "hospital_beds", bed_years, bed_fill_map)
df = apply_interval_fill(df, "st_gen_hospital_beds", bed_years, bed_fill_map)

# Percent short-term general hospital beds out of all hospital beds
for yr in bed_years:
    num_col = f"st_gen_hospital_beds_filled_{yr}"
    den_col = f"hospital_beds_filled_{yr}"
    out_col = f"pct_st_gen_hospital_beds_of_all_hospital_beds_{yr}"

    df[out_col] = pd.NA
    valid = df[den_col].notna() & (df[den_col] > 0) & df[num_col].notna()
    df.loc[valid, out_col] = (df.loc[valid, num_col] / df.loc[valid, den_col]) * 100

# Short-term general hospital beds per 1,000 population
for yr in bed_years:
    beds_col = f"st_gen_hospital_beds_filled_{yr}"
    pop_col = f"population_{yr}"
    out_col = f"st_gen_hospital_beds_per_1000_pop_{yr}"

    df[out_col] = pd.NA

    if beds_col in df.columns and pop_col in df.columns:
        valid = df[beds_col].notna() & df[pop_col].notna() & (df[pop_col] > 0)
        df.loc[valid, out_col] = (df.loc[valid, beds_col] / df.loc[valid, pop_col]) * 1000

# =========================================================
# Keep only final columns
# =========================================================
income_cols = [
    f"median_hh_income_{yr}"
    for yr in years_2010_2022
    if f"median_hh_income_{yr}" in df.columns
]

poverty_annual_cols = [
    f"pct_below_poverty_{yr}"
    for yr in range(2011, 2023)
    if f"pct_below_poverty_{yr}" in df.columns
]

pct65_cols = [
    f"pct_population_65plus_{yr}"
    for yr in years_2011_2022
    if f"pct_population_65plus_{yr}" in df.columns
]

phys_pct_cols = [
    f"pct_gen_pract_md_patientcare_{yr}"
    for yr in years_2010_2022
    if f"pct_gen_pract_md_patientcare_{yr}" in df.columns
]

stgen_bed_pct_cols = [
    f"pct_st_gen_hospital_beds_of_all_hospital_beds_{yr}"
    for yr in years_2010_2022
    if f"pct_st_gen_hospital_beds_of_all_hospital_beds_{yr}" in df.columns
]

stgen_bed_rate_cols = [
    f"st_gen_hospital_beds_per_1000_pop_{yr}"
    for yr in range(2011, 2023)
    if f"st_gen_hospital_beds_per_1000_pop_{yr}" in df.columns
]

keep_cols = (
    ["fips"] +
    income_cols +
    poverty_annual_cols +
    pct65_cols +
    [c for c in ["cbsa_indicator_2020", "cbsa_indicator_2020_label"] if c in df.columns] +
    phys_pct_cols +
    stgen_bed_pct_cols +
    stgen_bed_rate_cols
)

df_final = df[keep_cols].copy()

# =========================================================
# Round percentages / rates
# =========================================================
for c in poverty_annual_cols + pct65_cols + phys_pct_cols + stgen_bed_pct_cols + stgen_bed_rate_cols:
    df_final[c] = pd.to_numeric(df_final[c], errors="coerce").round(4)

# =========================================================
# QCs
# =========================================================
print("Final shape:", df_final.shape)
print("Unique FIPS:", df_final["fips"].nunique(dropna=True))

preview_cols = [c for c in [
    "fips",
    "median_hh_income_2018",
    "median_hh_income_2019",
    "median_hh_income_2020",
    "median_hh_income_2021",
    "median_hh_income_2022",
    "pct_below_poverty_2018",
    "pct_below_poverty_2019",
    "pct_below_poverty_2020",
    "pct_below_poverty_2021",
    "pct_below_poverty_2022",
    "pct_population_65plus_2018",
    "pct_population_65plus_2019",
    "pct_population_65plus_2020",
    "pct_population_65plus_2021",
    "pct_population_65plus_2022",
    "cbsa_indicator_2020",
    "cbsa_indicator_2020_label",
    "pct_gen_pract_md_patientcare_2018",
    "pct_gen_pract_md_patientcare_2019",
    "pct_gen_pract_md_patientcare_2020",
    "pct_gen_pract_md_patientcare_2021",
    "pct_gen_pract_md_patientcare_2022",
    "pct_st_gen_hospital_beds_of_all_hospital_beds_2018",
    "pct_st_gen_hospital_beds_of_all_hospital_beds_2019",
    "pct_st_gen_hospital_beds_of_all_hospital_beds_2020",
    "pct_st_gen_hospital_beds_of_all_hospital_beds_2021",
    "pct_st_gen_hospital_beds_of_all_hospital_beds_2022",
    "st_gen_hospital_beds_per_1000_pop_2018",
    "st_gen_hospital_beds_per_1000_pop_2019",
    "st_gen_hospital_beds_per_1000_pop_2020",
    "st_gen_hospital_beds_per_1000_pop_2021",
    "st_gen_hospital_beds_per_1000_pop_2022",
] if c in df_final.columns]

print(df_final[preview_cols].head(10).to_string(index=False))

# Check whether physician percent exceeds 100 anywhere
phys_check_cols = [c for c in phys_pct_cols if c in df_final.columns]
if phys_check_cols:
    over_100 = pd.DataFrame({
        "column": phys_check_cols,
        "n_over_100": [(df_final[c] > 100).sum(skipna=True) for c in phys_check_cols],
        "max_value": [df_final[c].max(skipna=True) for c in phys_check_cols],
    })
    print("\nPhysician percent QC:")
    print(over_100.to_string(index=False))

# Check whether short-term general bed share exceeds 100 anywhere
bed_pct_check_cols = [c for c in stgen_bed_pct_cols if c in df_final.columns]
if bed_pct_check_cols:
    over_100_beds = pd.DataFrame({
        "column": bed_pct_check_cols,
        "n_over_100": [(df_final[c] > 100).sum(skipna=True) for c in bed_pct_check_cols],
        "max_value": [df_final[c].max(skipna=True) for c in bed_pct_check_cols],
    })
    print("\nShort-term general hospital bed percent QC:")
    print(over_100_beds.to_string(index=False))

# Check whether bed rate is negative anywhere
bed_rate_check_cols = [c for c in stgen_bed_rate_cols if c in df_final.columns]
if bed_rate_check_cols:
    negative_bed_rates = pd.DataFrame({
        "column": bed_rate_check_cols,
        "n_negative": [(df_final[c] < 0).sum(skipna=True) for c in bed_rate_check_cols],
        "max_value": [df_final[c].max(skipna=True) for c in bed_rate_check_cols],
    })
    print("\nShort-term general hospital bed rate QC:")
    print(negative_bed_rates.to_string(index=False))

# =========================================================
# Export
# =========================================================
df_final.to_csv(out_file, index=False)
print(f"\nSaved final file to: {out_file}")

pd.set_option("display.max_columns", None)
print(df_final.head(15))

df = pd.read_csv("/gpfs/data/public/AHRF/derived/ahrf_controls_pct_only_final_phys_optionA_stgenbeds.csv")
print(df[[
    "fips",
    "pct_population_65plus_2021",
    "pct_population_65plus_2022",
    "st_gen_hospital_beds_per_1000_pop_2021",
    "st_gen_hospital_beds_per_1000_pop_2022"
]].isna().mean())

