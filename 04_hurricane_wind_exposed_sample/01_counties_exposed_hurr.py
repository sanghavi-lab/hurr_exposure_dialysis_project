#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 20, 2026
# Description: This code builds county-hurricane exposure files for a set of NOAA storms from 2011–2022 (while the script 
# goes up to 2022, the wind data for 2022 was missing. Thus, we will ultimately use only until 2021). It identifies counties 
# exposed to at least 17 m/s and at least 5 m/s using the county-level wind file, then assigns each exposed county an exposure 
# start date based on the timestamp of the hurricane track point closest to that county’s centroid. It also appends a few 
# descriptive variables to each county–storm row, including the county’s maximum sustained wind, the wind at the nearest 
# track point, the distance to that track point, and the storm’s overall peak wind.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import pandas as pd
import geopandas as gpd

# -------------------------
# Paths and other spec
# -------------------------

BASE = "/gpfs/data/cms-share/duas/52484/Jessy/data/public_data/data/brooke_hurricane/update_ryanzomorrodi"
WIND_PATH  = f"{BASE}/storm_winds.csv"
TRACK_PATH = f"{BASE}/hurr_tracks.csv"

COUNTY_SHP = "/gpfs/data/cms-share/duas/52484/Jessy/data/public_data/data/shp_files/cb_2018_us_county_500k/"

OUT_BASE = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis"
OUT_DIR  = os.path.join(OUT_BASE, "hurricane_county_exposure_start_v02_track64kt_2011_2022")
os.makedirs(OUT_DIR, exist_ok=True)

YEAR_MIN, YEAR_MAX = 2011, 2022

# Two thresholds in m/s
TS_MS = 17.0
PROX_MS = 5.0

# Shapefile exclusions
DROP_STATEFP = ['02','15','66','72','60','69','78']  # AK, HI, territories, PR

# CRS
CRS_WGS84 = "EPSG:4326"
CRS_CONUS = "EPSG:5070"

# -------------------------
# NOAA landfall storm list
# -------------------------
# These are list of hurricanes that made landfall in the US according to NOAA (see appendix for more details)

NOAA_STORMS = [
    # 2011
    "Irene-2011",

    # 2012
    "Isaac-2012",
    "Sandy-2012",

    # 2014
    "Arthur-2014",

    # 2016
    "Hermine-2016",
    "Matthew-2016",

    # 2017
    "Harvey-2017",
    "Irma-2017",
    "Nate-2017",

    # 2018
    "Florence-2018",
    "Michael-2018",

    # 2019
    "Barry-2019",
    "Dorian-2019",

    # 2020
    "Hanna-2020",
    "Isaias-2020",
    "Laura-2020",
    "Sally-2020",
    "Delta-2020",
    "Zeta-2020",

    # 2021
    "Ida-2021",
    "Nicholas-2021",

    # 2022
    "Ian-2022",
]

NOAA_STORMS_NORM = sorted({s.strip().lower() for s in NOAA_STORMS})

# -------------------------
# Functions
# -------------------------

def parse_track_datetime(s: str):
    # date like "198808051800" -> YYYYMMDDHHMM
    try:
        return pd.to_datetime(str(s), format="%Y%m%d%H%M")
    except Exception:
        return pd.NaT

def norm_storm_id(x):
    # cleans by forcing lower case
    return str(x).strip().lower()

def build_exposure_with_startdate(
    wind_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    centroids_gdf: gpd.GeoDataFrame,
    storm_peak_df: pd.DataFrame,
    threshold_ms: float,
) -> pd.DataFrame:
    # Build county-storm exposure start date for counties with vmax_sust >= threshold_ms. (17 or 5 m/s)
    # Returns exp_out with columns:
      # storm_id, storm_year, fips, vmax_sust,
      # track_wind_kt_at_nearest, dist_m,
      # storm_peak_wind_kt,
      # exposure_start_trackdate, exposure_start_dt

    # Keep only county–storm rows that are on the approved NOAA storm list and whose county-level sustained wind meets the threshold.
    exp = wind_df[(wind_df["storm_id_norm"].isin(NOAA_STORMS_NORM)) & (wind_df["vmax_sust"] >= threshold_ms)].copy()
    exp = exp[["storm_id", "storm_id_norm", "storm_year", "fips", "vmax_sust"]].drop_duplicates()

    # Prints showing how many qualifying county–storm rows there are and how many storms they come from
    print(f"threshold={threshold_ms} m/s | county–storm rows: {len(exp):,} | storms: {exp['storm_id'].nunique():,}")

    # For each storm, find nearest track point to each county centroid. i.e. hurricane data has track points that track the date of eye of the storm. This whole process would find what time is the county centroid closest to the hurricane's track.
    storm_to_fips = exp.groupby("storm_id")["fips"].apply(list).to_dict() # reorganize wind data as dictionary
    out_rows = [] # empty list to collect results

    for sid, fips_list in storm_to_fips.items(): # For each storm ID, pull the exposed counties for that storm
        if not fips_list: # skip if empty
            continue

        c_sub = centroids_gdf[centroids_gdf["fips"].isin(fips_list)].copy() # Subsets the county centroids to only the counties exposed to that storm
        if c_sub.empty: # skip if empty
            continue

        t_sub = tracks_df[tracks_df["storm_id"] == sid].copy() # subsets the track data to only that storm
        if t_sub.empty: # robustness by trying again with lower case letters.
            sid_norm = norm_storm_id(sid)
            t_sub = tracks_df[tracks_df["storm_id_norm"] == sid_norm].copy()
            if t_sub.empty:
                print(f"WARNING: No track rows found for storm_id={sid} (norm={sid_norm})")
                continue

        t_gdf = gpd.GeoDataFrame( # turns the storm track rows into a GeoDataFrame of points to calc distance
            t_sub[["date", "wind", "latitude", "longitude"]].copy(),
            geometry=gpd.points_from_xy(t_sub["longitude"], t_sub["latitude"]),
            crs=CRS_WGS84
        ).to_crs(CRS_CONUS)

        # This finds, for each county centroid, the nearest hurricane track point for that storm. It brings over the track point’s date and wind, and it saves the distance in meters as dist_m
        joined = gpd.sjoin_nearest(
            c_sub,
            t_gdf[["date", "wind", "geometry"]],
            how="left",
            distance_col="dist_m"
        )

        joined = joined.sort_values("dist_m").drop_duplicates(subset=["fips"], keep="first") # this ensures the nearest match per county
        joined = joined.drop(columns=["index_right"], errors="ignore") # clean by removing index col
        joined["storm_id"] = sid # adds back storm id explicitly
        out_rows.append(joined[["storm_id", "fips", "date", "wind", "dist_m"]]) # keep only the columns needed and appends those county-level nearest-track results to out_rows list

    # Once the loop finishes, combines all storm-level results into one table
    nearest_df = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(
        columns=["storm_id", "fips", "date", "wind", "dist_m"]
    )
    nearest_df = nearest_df.rename(columns={"wind": "track_wind_kt_at_nearest"})

    # Merges that nearest-track information back onto the original exposed county–storm table
    exp2 = exp.merge(nearest_df, on=["storm_id", "fips"], how="left")

    # Attach max track wind
    exp2["storm_id_norm"] = exp2["storm_id"].apply(norm_storm_id)
    storm_peak_small = storm_peak_df[["storm_id_norm", "storm_peak_wind_kt"]].drop_duplicates()
    exp2 = exp2.merge(storm_peak_small, on="storm_id_norm", how="left")

    # Create the exposure start date. Basically, the date of when county centroid was closest to hurr track.
    exp2["exposure_start_trackdate"] = exp2["date"]
    exp2["exposure_start_dt"] = exp2["exposure_start_trackdate"].apply(parse_track_datetime)

    exp2 = exp2.drop(columns=["date"], errors="ignore")
    return exp2

# -------------------------
# 1) Load tracks and restrict to NOAA storms (and years)
# -------------------------

# Import
tracks = pd.read_csv(TRACK_PATH, dtype={"storm_id": str, "date": str})
tracks["storm_id_norm"] = tracks["storm_id"].apply(norm_storm_id)

# Filter relevant storms
tracks["storm_year"] = tracks["storm_id"].str[-4:].astype(int) # take the year from storm_id
tracks = tracks[(tracks["storm_year"] >= YEAR_MIN) & (tracks["storm_year"] <= YEAR_MAX)].copy()
tracks = tracks[tracks["storm_id_norm"].isin(NOAA_STORMS_NORM)].copy()

# Convert to num
tracks["wind"] = pd.to_numeric(tracks["wind"], errors="coerce")
tracks["latitude"] = pd.to_numeric(tracks["latitude"], errors="coerce")
tracks["longitude"] = pd.to_numeric(tracks["longitude"], errors="coerce")

# Drop any missing data. We only want counties that were able to estimate at least 17 m/s windspeeds. I checked and found no missing data but left code here.
tracks = tracks.dropna(subset=["storm_id", "storm_year", "wind", "latitude", "longitude", "date"]).copy()

# Sort
present_tracks = sorted(tracks["storm_id_norm"].unique().tolist())
missing_in_tracks = sorted(set(NOAA_STORMS_NORM) - set(present_tracks))

print(f"NOAA storms: {len(NOAA_STORMS_NORM)}")
print(f"NOAA storms present in hurricane data: {len(present_tracks)}")
if missing_in_tracks: # Quality checks
    print("[WARN] NOAA storms missing in TRACKS:", missing_in_tracks)
    # Basically, did all the NOAA storms I asked for actually show up in the data? If not, then will appear here...

# Keep max wind estimates of each storm. Possible uses: assign category of storm based on max windspeed? Compare again NOAA's?
storm_peak = (
    tracks.groupby("storm_id")["wind"]
    .max()
    .reset_index(name="storm_peak_wind_kt")
)
storm_peak["storm_id_norm"] = storm_peak["storm_id"].apply(norm_storm_id)

# -------------------------
# 2) Load storm_winds
# -------------------------
# keep NOAA storms and keep ALL counties for those storms

# Import, clean, and filter the wind data
wind = pd.read_csv(WIND_PATH, dtype={"fips": str, "storm_id": str})
wind["storm_id_norm"] = wind["storm_id"].apply(norm_storm_id)
wind["fips"] = wind["fips"].astype(str).str.zfill(5)
wind["storm_year"] = wind["storm_id"].str[-4:].astype(int)
wind = wind[(wind["storm_year"] >= YEAR_MIN) & (wind["storm_year"] <= YEAR_MAX)].copy()
wind = wind[wind["storm_id_norm"].isin(NOAA_STORMS_NORM)].copy()

present_wind = sorted(wind["storm_id_norm"].unique().tolist())
missing_in_wind = sorted(set(NOAA_STORMS_NORM) - set(present_wind))
print(f"NOAA storms present in STORM_WINDS: {len(present_wind)}")
if missing_in_wind:
    print("[WARN] NOAA storms missing in STORM_WINDS:", missing_in_wind)
    # Same quality check, did all the NOAA storms I asked for actually show up in the data? If not, then will appear here...

# Ensure num vmax_sust
wind["vmax_sust"] = pd.to_numeric(wind["vmax_sust"], errors="coerce")
wind = wind.dropna(subset=["vmax_sust", "fips", "storm_id", "storm_year"]).copy() # Again, keep nonmissing data. Checked and found no county missing data but left code here.

# -------------------------
# 3) County centroids
# -------------------------

# Import shape file and clean
counties = gpd.read_file(COUNTY_SHP)
counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2) # e.g. 1 -> 01
counties["COUNTYFP"] = counties["COUNTYFP"].astype(str).str.zfill(3) # e.g. 7 -> 007

# Keeping only contiguous states (i.e. drop HI, AK, US territories)
counties = counties[~counties["STATEFP"].isin(DROP_STATEFP)].copy()
counties["fips"] = (counties["STATEFP"] + counties["COUNTYFP"]).astype(str).str.zfill(5)

# Set the correct CRS
counties = counties.set_crs(CRS_WGS84, allow_override=True)
counties_proj = counties.to_crs(CRS_CONUS)

centroids = counties_proj[["fips", "geometry"]].copy()
centroids["geometry"] = counties_proj.geometry.centroid # replaces each county polygon geometry with its centroid point

# -------------------------
# 4) Build TWO exposure tables (>=17 m/s and >=5 m/s)
# -------------------------
# the 5 m/s will be used mainly to construct a table in the appendix. The 17 m/s will be used to help identify bene's in counties experiencing at least 17 m/s windspeeds

# Counties experiencing at least 17 m/s wind speeds from hurricanes, find when their centroid is closest to a track.
exp_ts17 = build_exposure_with_startdate(
    wind_df=wind,
    tracks_df=tracks,
    centroids_gdf=centroids,
    storm_peak_df=storm_peak,
    threshold_ms=TS_MS,
)

# Counties experiencing at least 5 m/s wind speeds from hurricanes, find when their centroid is closest to a track.
exp_ms05 = build_exposure_with_startdate(
    wind_df=wind,
    tracks_df=tracks,
    centroids_gdf=centroids,
    storm_peak_df=storm_peak,
    threshold_ms=PROX_MS,
)

# Quality Checks [QC]
print("\n[QC] TS17: missing exposure_start_dt:", int(exp_ts17["exposure_start_dt"].isna().sum()))
print("[QC] MS05: missing exposure_start_dt:", int(exp_ms05["exposure_start_dt"].isna().sum()))
print("[QC] TS17 storms:", exp_ts17["storm_id"].nunique(), "| counties:", exp_ts17["fips"].nunique())
print("[QC] MS05 storms:", exp_ms05["storm_id"].nunique(), "| counties:", exp_ms05["fips"].nunique())
print("[QC] TS17 min vmax_sust:", float(exp_ts17["vmax_sust"].min()) if len(exp_ts17) else None)
print("[QC] MS05 min vmax_sust:", float(exp_ms05["vmax_sust"].min()) if len(exp_ms05) else None)

# -------------------------
# 5) Save outputs
# -------------------------

storm_list_path = os.path.join(OUT_DIR, "hurricane_storm_ids_track64kt_2011_2022.csv")
storm_peak_path = os.path.join(OUT_DIR, "hurricane_storm_peak_wind_kt_2011_2022.csv")

# File names
exp_out_csv_ts17 = os.path.join(OUT_DIR, "county_storm_exposure_with_startdate_2011_2022_ts17ms.csv")
exp_out_csv_ms05 = os.path.join(OUT_DIR, "county_storm_exposure_with_startdate_2011_2022_ms05.csv")

# Export list of hurricanes
pd.DataFrame({"storm_id": NOAA_STORMS}).to_csv(storm_list_path, index=False)

# Export max wind speed of each hurr
storm_peak_out = storm_peak.copy()
storm_peak_out = storm_peak_out.sort_values(["storm_peak_wind_kt", "storm_id"], ascending=[False, True])
storm_peak_out.to_csv(storm_peak_path, index=False)

# Drop helper norm column
exp_ts17_out = exp_ts17.drop(columns=["storm_id_norm"], errors="ignore")
exp_ms05_out = exp_ms05.drop(columns=["storm_id_norm"], errors="ignore")

# Export
exp_ts17_out.to_csv(exp_out_csv_ts17, index=False)
exp_ms05_out.to_csv(exp_out_csv_ms05, index=False)

print("\nWhere I wrote the data")
print(" -", storm_list_path)
print(" -", storm_peak_path)
print(" -", exp_out_csv_ts17)
print(" -", exp_out_csv_ms05)

print("\n[QCs]")
print("TS17 rows:", len(exp_ts17_out), "| missing exposure_start_dt:", int(exp_ts17_out["exposure_start_dt"].isna().sum()))
print("MS05 rows:", len(exp_ms05_out), "| missing exposure_start_dt:", int(exp_ms05_out["exposure_start_dt"].isna().sum()))



