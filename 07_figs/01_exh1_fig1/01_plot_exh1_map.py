#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: May 4, 2026
# Description: This script creates a two-panel map figure to show the relationship between hurricane paths and dialysis 
# facility disruptions. It first plots the tracks of 21 hurricanes over the continental U.S. then separately maps the 
# ZIP-based locations of facilities with operational stress and assigns each facility to the hurricane associated with
# its disruption (I used the timing of the facility disruption and when the hurricane made landfall to match them). The 
# overall goal is to visually compare where major hurricanes traveled with where disrupted dialysis facilities were located.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, GeometryCollection
from pathlib import Path
from matplotlib.lines import Line2D

# -------------------
# Import paths
# -------------------
HURR_TRACKS_CSV = "/gpfs/data/cms-share/duas/52484/Jessy/data/public_data/data/brooke_hurricane/update_ethan_2025/hurr_tracks.csv"
STATES_SHP_DIR  = "/gpfs/data/cms-share/duas/52484/Jessy/data/public_data/data/shp_files/cb_2018_us_state_500k/"

BASE_STRESS = "/gpfs/data/cms-share/duas/54200/Jessy/data/derived/facility_rolling_stress_days"
FAC_PATH = f"{BASE_STRESS}/valid_facilities_operational_stress_2011_2022.csv"

ZCTA_PATH   = "/gpfs/data/cms-share/duas/52484/Jessy/data/public_data/data/shp_files/cb_2013_us_zcta_zip_500k/"
COUNTY_PATH = "/gpfs/data/cms-share/duas/52484/Jessy/data/public_data/data/shp_files/cb_2018_us_county_500k/"
STATES_PATH = "/gpfs/data/cms-share/duas/52484/Jessy/data/public_data/data/shp_files/cb_2018_us_state_500k/"

OUT_DIR = Path("/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/fig/hurricane_tracks_multi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Drop non-CONUS states / territories because the map is focused on the continental US
DROP_STATEFP = ["02", "15", "66", "72", "60", "69", "78"]


# Buffer CONUS outward a bit so the track lines can include some ocean area right before landfall.
# In other words, do not cut the track exactly at the coastline; allow some near-shore lead-in.
BUFFER_KM = 200

# -------------------
# Storm metadata (Panel A = exactly this set/order)
# -------------------
# These are the 21 hurricanes used for the tracks panel. For each storm, store its category
# and the label that should appear in the legend.
storm_meta = {
    "Irene-2011":     {"cat": 1, "label_cat": "Cat 1"},
    "Isaac-2012":     {"cat": 1, "label_cat": "Cat 1"},
    "Sandy-2012":     {"cat": 1, "label_cat": "Cat 1"},
    "Arthur-2014":    {"cat": 2, "label_cat": "Cat 2"},
    "Hermine-2016":   {"cat": 1, "label_cat": "Cat 1"},
    "Matthew-2016":   {"cat": 2, "label_cat": "Cat 2"},
    "Harvey-2017":    {"cat": 4, "label_cat": "Cat 4"},
    "Irma-2017":      {"cat": 4, "label_cat": "Cat 4"},
    "Nate-2017":      {"cat": 1, "label_cat": "Cat 1"},
    "Florence-2018":  {"cat": 1, "label_cat": "Cat 1"},
    "Michael-2018":   {"cat": 5, "label_cat": "Cat 5"},
    "Barry-2019":     {"cat": 1, "label_cat": "Cat 1"},
    "Dorian-2019":    {"cat": 2, "label_cat": "Cat 2"},
    "Hanna-2020":     {"cat": 1, "label_cat": "Cat 1"},
    "Isaias-2020":    {"cat": 1, "label_cat": "Cat 1"},
    "Laura-2020":     {"cat": 4, "label_cat": "Cat 4"},
    "Sally-2020":     {"cat": 2, "label_cat": "Cat 2"},
    "Delta-2020":     {"cat": 2, "label_cat": "Cat 2"},
    "Zeta-2020":      {"cat": 3, "label_cat": "Cat 3"},
    "Ida-2021":       {"cat": 4, "label_cat": "Cat 4"},
    "Nicholas-2021":  {"cat": 1, "label_cat": "Cat 1"},
    "Ian-2022":       {"cat": 4, "label_cat": "Cat 4"},
}
storms_all = list(storm_meta.keys())

# Line width mapping so stronger storms appear visually thicker on the map
cat_to_lw = {1: 1.6, 2: 2.4, 3: 3.2, 4: 4.0, 5: 4.8}

# These are the storms shown in Panel B when plotting disrupted dialysis facilities.
# So Panel A uses all 21 storms, whereas Panel B uses a smaller storm subset.
STORMS_RESTRICT = [
    "Isaac-2012",
    "Sandy-2012",
    "Matthew-2016",
    "Harvey-2017",
    "Irma-2017",
    "Florence-2018",
    "Michael-2018",
    "Dorian-2019",
    "Laura-2020",
    "Ida-2021",
    "Ian-2022",
]

# Anchor dates used to assign disrupted facilities to a hurricane based on timing.
# Later the code compares each facility's earliest stress day to these dates and assigns
# the closest storm within a small allowable window.
storm_dates = {
    "Isaac-2012":    "2012-08-25",
    "Sandy-2012":    "2012-10-27",
    "Matthew-2016":  "2016-10-06",
    "Harvey-2017":   "2017-08-25",
    "Irma-2017":     "2017-09-08",
    "Florence-2018": "2018-09-12",
    "Michael-2018":  "2018-10-10",
    "Dorian-2019":   "2019-08-30",
    "Laura-2020":    "2020-08-27",
    "Ida-2021":      "2021-08-29",
    "Ian-2022":      "2022-09-28",
}
storm_dates = {k: pd.to_datetime(v) for k, v in storm_dates.items()}



def to_zip5(s: pd.Series) -> pd.Series:
    # Clean ZIPs down to 5-digit strings.
    # Basically, keep the first 5 digits, remove non-numeric characters, and left-pad if needed.
    s = s.astype(str).str[:5].str.replace(r"\D", "", regex=True)
    return s.str.zfill(5)

def make_county_fips(counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Build a standard 5-digit county FIPS code from state + county components.
    counties = counties.copy()
    counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)
    counties["COUNTYFP"] = counties["COUNTYFP"].astype(str).str.zfill(3)
    counties["fips"] = counties["STATEFP"] + counties["COUNTYFP"]
    return counties

def assign_storm_by_date(df: pd.DataFrame, date_col: str, window_days: int = 21) -> pd.Series:
    # For each row, assign the closest storm in the same year based on calendar timing.
    # The storm is only assigned if the closest anchor date falls within window_days.
    dt = pd.to_datetime(df[date_col], errors="coerce")
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    years = dt.dt.year

    for yr in sorted(years.dropna().unique()):
        # Work year by year so a facility in one year cannot be matched to a storm in another year
        idx = df.index[years == yr]
        candidates = [s for s in STORMS_RESTRICT if storm_dates[s].year == int(yr)]
        if not candidates or len(idx) == 0:
            continue

        # For each candidate storm in that year, calculate how many days away the facility stress date is
        delta_df = pd.concat(
            [(dt.loc[idx] - storm_dates[s]).abs().dt.days.rename(s) for s in candidates],
            axis=1
        )
        # Pick the nearest storm and keep it only if it is close enough in time
        best = delta_df.idxmin(axis=1)
        best_delta = delta_df.min(axis=1)
        out.loc[idx] = best.where(best_delta <= window_days, pd.NA)

    return out

def jitter_points_lonlat(gdf: gpd.GeoDataFrame, seed: int = 7, degrees: float = 0.05) -> gpd.GeoDataFrame:
    # Apply a very small amount of random jitter to overlapping facility points.
    # This is just for readability on the map so multiple facilities do not sit exactly on top of one another.
    import numpy as np
    rng = np.random.default_rng(seed)
    gdf = gdf.copy()
    x = gdf.geometry.x.to_numpy()
    y = gdf.geometry.y.to_numpy()
    x = x + rng.normal(0, degrees, size=len(gdf))
    y = y + rng.normal(0, degrees, size=len(gdf))
    gdf["geometry"] = gpd.points_from_xy(x, y, crs=gdf.crs)
    return gdf

def clip_track_to_conus(g: pd.DataFrame, conus_buffer_5070):
    """
    Summary - Build a hurricane track line from the point sequence and clip it to a buffered
    continental US polygon. This keeps the portion of the track that overlaps CONUS plus some
    near-shore ocean space.

    Returns geometry in EPSG:4326, or None if no overlap.
    """
    g = g.dropna(subset=["longitude", "latitude"]).copy()
    if len(g) < 2:
        # Need at least two points to draw a line
        return None

    # Convert point sequence to projected coordinates first so the clipping/buffering geometry behaves better
    pts_4326 = gpd.GeoSeries(
        gpd.points_from_xy(g["longitude"].astype(float), g["latitude"].astype(float)),
        crs="EPSG:4326"
    )
    pts_5070 = pts_4326.to_crs("EPSG:5070")

    # Connect the storm points in order into one path line
    line_5070 = LineString([(p.x, p.y) for p in pts_5070.geometry])
    if line_5070.is_empty:
        return None

    # Intersect the full track with the buffered CONUS shape
    clipped = line_5070.intersection(conus_buffer_5070)
    if clipped.is_empty:
        return None

    # The intersection can come back as one line, multiple lines, or a geometry collection.
    # Keep the longest line piece so the storm still has one clean representative track.
    if isinstance(clipped, LineString):
        keep = clipped
    elif isinstance(clipped, MultiLineString):
        keep = max(clipped.geoms, key=lambda geom: geom.length, default=None)
    elif isinstance(clipped, GeometryCollection):
        lines = [geom for geom in clipped.geoms if isinstance(geom, LineString)]
        if not lines:
            return None
        keep = max(lines, key=lambda geom: geom.length, default=None)
    else:
        return None

    if keep is None or keep.is_empty:
        return None

    # Convert back to lon/lat for plotting
    keep_4326 = gpd.GeoSeries([keep], crs="EPSG:5070").to_crs("EPSG:4326").iloc[0]
    return keep_4326

def storm_year_from_id(storm_id: str):
    # Pull the year from a storm ID like "Sandy-2012"
    try:
        return int(str(storm_id)[-4:])
    except Exception:
        return 9999

def legend_order_cat_then_time(storm_ids: list[str]) -> list[str]:
    """
    Order storms by category first (Cat 1 -> Cat 5), then by year, then by name.
    This changes legend order only. It does not change the drawing order or color mapping.
    """
    def key(sid: str):
        cat = storm_meta.get(sid, {}).get("cat", 99)
        yr = storm_year_from_id(sid)
        return (cat, yr, sid)
    return sorted(storm_ids, key=key)

# -------------------
# 1) Load hurr tracks for ALL 21 (for extent + clipping)
# -------------------
tracks = pd.read_csv(HURR_TRACKS_CSV)

needed = {"storm_id", "longitude", "latitude"}
missing = needed - set(tracks.columns)
if missing:
    raise ValueError(f"Missing columns in hurr_tracks.csv: {sorted(missing)}")

# Keep only the 21 storms used in this figure
tracks = tracks[tracks["storm_id"].isin(storms_all)].copy()
tracks["longitude"] = pd.to_numeric(tracks["longitude"], errors="coerce")
tracks["latitude"]  = pd.to_numeric(tracks["latitude"],  errors="coerce")
tracks = tracks.dropna(subset=["longitude", "latitude"])

# If a time column exists, use it so each storm's path is drawn in true chronological order
time_candidates = ["datetime", "date_time", "time", "date", "timestamp", "iso_time", "ISO_TIME"]
time_col = next((c for c in time_candidates if c in tracks.columns), None)
if time_col is not None:
    tracks[time_col] = pd.to_datetime(tracks[time_col], errors="coerce")
    tracks = tracks.sort_values(["storm_id", time_col])
else:
    tracks = tracks.sort_values(["storm_id"])

# -------------------
# 2) Set the map boundaries for both panels
# -------------------
# The plotting window is based on the raw points from all 21 storms before clipping.
# That way Panel A and Panel B share the same geographic extent and match the reference figure.

# Find the full longitude/latitude range of all 21 hurricane track points
lon_min = float(tracks["longitude"].min())
lon_max = float(tracks["longitude"].max())
lat_min = float(tracks["latitude"].min())
lat_max = float(tracks["latitude"].max())

# Add padding around that range
pad_lon = 3.5
pad_lat = 2.5
x0, x1 = lon_min - pad_lon, lon_max + pad_lon
y0, y1 = lat_min - pad_lat, lat_max + pad_lat

# Limit the map to a reasonable continental U.S. box
x0 = max(x0, -125.0)
x1 = min(x1,  -60.0)
y0 = max(y0,   20.0)
y1 = min(y1,   52.0)

print(f"[Extent matched to 21-storm map] x=[{x0:.2f}, {x1:.2f}] y=[{y0:.2f}, {y1:.2f}]")

# -------------------
# 3) Load state boundaries for plotting (EPSG:4326)
# -------------------
states = gpd.read_file(STATES_SHP_DIR)
if "STATEFP" in states.columns:
    states["STATEFP"] = states["STATEFP"].astype(str).str.zfill(2)
states = states[~states["STATEFP"].isin(DROP_STATEFP)].copy()

# Keep the plotting boundaries in standard lon/lat coordinates
if states.crs is None:
    states = states.set_crs("EPSG:4326")
else:
    states = states.to_crs("EPSG:4326")

# -------------------
# 4) Build CONUS buffered polygon for clipping (EPSG:5070)
# -------------------
# Use a projected CRS for the buffering/clipping work so the geometry operations behave more sensibly.
states_poly = gpd.read_file(STATES_PATH)
if "STATEFP" in states_poly.columns:
    states_poly["STATEFP"] = states_poly["STATEFP"].astype(str).str.zfill(2)
    states_poly = states_poly[~states_poly["STATEFP"].isin(DROP_STATEFP)].copy()

if states_poly.crs is None:
    states_poly = states_poly.set_crs("EPSG:4326")
else:
    states_poly = states_poly.to_crs("EPSG:4326")

# Dissolve all kept states into one land polygon and buffer it outward
conus_land_4326 = states_poly.dissolve().geometry.iloc[0]
conus_land_4326 = conus_land_4326.buffer(0)  # helps fix common invalid geometries

conus_land_5070 = gpd.GeoSeries([conus_land_4326], crs="EPSG:4326").to_crs("EPSG:5070").iloc[0]
conus_buffer_5070 = conus_land_5070.buffer(BUFFER_KM * 1000.0)

# -------------------
# 5) Build CLIPPED LineString per storm (ALL 21)
# -------------------
storm_lines_all = []
for sid, g in tracks.groupby("storm_id", sort=False):
    g = g.dropna(subset=["longitude", "latitude"]).copy()
    if len(g) < 2:
        continue

    if time_col is not None and time_col in g.columns:
        g = g.sort_values(time_col)

    # Build one clipped line per storm and keep only storms that overlap buffered CONUS
    geom_4326 = clip_track_to_conus(g, conus_buffer_5070)
    if geom_4326 is None:
        continue

    storm_lines_all.append({"storm_id": sid, "geometry": geom_4326})

tracks_gdf_all = gpd.GeoDataFrame(storm_lines_all, crs="EPSG:4326")
if tracks_gdf_all.empty:
    raise ValueError("After clipping, no storms had overlap with buffered CONUS to draw tracks.")

# -------------------
# 6) Build storm->color mapping FIXED to storms_all order
# -------------------
# Assign one fixed color per storm so the tracks panel and facilities panel stay visually aligned.
tab20  = list(plt.colormaps["tab20"].colors)
tab20b = list(plt.colormaps["tab20b"].colors)
colors_all = (tab20 + tab20b)[:len(storms_all)]
storm_color = dict(zip(storms_all, colors_all))

# -------------------
# 7) Facilities geocoding (ZIP representative points) + storm assignment
# -------------------
# Load ZIP polygons and keep a clean ZIP5 identifier
zcta = gpd.read_file(ZCTA_PATH)
if "ZCTA5CE10" in zcta.columns:
    zcta["zip5"] = zcta["ZCTA5CE10"].astype(str).str.zfill(5)
elif "ZCTA5CE20" in zcta.columns:
    zcta["zip5"] = zcta["ZCTA5CE20"].astype(str).str.zfill(5)
else:
    raise ValueError("Could not find ZCTA5CE10 or ZCTA5CE20 in ZCTA shapefile.")

# Load counties so ZIP representative points can be linked to county FIPS
counties = gpd.read_file(COUNTY_PATH)
if "STATEFP" in counties.columns:
    counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)
if "COUNTYFP" in counties.columns:
    counties["COUNTYFP"] = counties["COUNTYFP"].astype(str).str.zfill(3)

counties = make_county_fips(counties)
counties = counties[~counties["STATEFP"].isin(DROP_STATEFP)].copy()

# Put ZIPs and counties into the same CRS before the spatial join
target_crs = zcta.crs or "EPSG:4269"
zcta = zcta.to_crs(target_crs)
counties = counties.to_crs(target_crs)

# Use ZIP representative points as the facility location proxy
zpts = zcta[["zip5", "geometry"]].copy()
zpts["geometry"] = zpts.geometry.representative_point()
zpts = gpd.GeoDataFrame(zpts, geometry="geometry", crs=target_crs)

# Spatially assign each ZIP representative point to a county FIPS
zip_to_county = gpd.sjoin(
    zpts,
    counties[["fips", "geometry"]],
    how="left",
    predicate="within"
)[["zip5", "fips"]].dropna(subset=["fips"]).drop_duplicates(subset=["zip5"])

# Load disrupted facilities and keep study years 2012-2022
fac = pd.read_csv(FAC_PATH, dtype=str)
fac["year"] = pd.to_numeric(fac.get("year"), errors="coerce")
fac = fac[fac["year"].between(2012, 2022)].copy()

# Parse earliest facility stress date and drop the February 2021 event I had chosen to exclude
fac["earliest_stress_day_dt"] = pd.to_datetime(fac["earliest_stress_day"], errors="coerce")
drop_feb2021 = (
    fac["earliest_stress_day_dt"].notna()
    & (fac["earliest_stress_day_dt"].dt.year == 2021)
    & (fac["earliest_stress_day_dt"].dt.month == 2)
)
fac = fac.loc[~drop_feb2021].copy()

# Clean ZIP and attach county FIPS
fac["zip5"] = to_zip5(fac["zip"])
fac = fac.merge(zip_to_county, on="zip5", how="left", validate="m:1")

# Attach ZIP representative-point geometry
fac_geo = fac.merge(zpts[["zip5", "geometry"]], on="zip5", how="left")
fac_geo = gpd.GeoDataFrame(fac_geo, geometry="geometry", crs=target_crs)
fac_geo["has_point"] = fac_geo["geometry"].notna()

# Assign each disrupted facility to the nearest restricted storm in time.
# The matching window here is 7 days, which is stricter than the helper's default.
fac_geo["storm_id_assigned"] = assign_storm_by_date(fac_geo, "earliest_stress_day_dt", window_days=7)

# Keep only facilities that have a map point and were assigned to one of the restricted storms
fac_plot = fac_geo.loc[
    fac_geo["has_point"] & fac_geo["storm_id_assigned"].isin(STORMS_RESTRICT)
].copy()

# Keep only one point per facility so a provider is not plotted multiple times.
# Since the point is really a ZIP representative point, this gives one centroid-like proxy per facility.
if "PRVDR_NUM" in fac_plot.columns:
    fac_plot = fac_plot.drop_duplicates(subset=["PRVDR_NUM"]).copy()

fac_plot = fac_plot.to_crs("EPSG:4326")
fac_plot = jitter_points_lonlat(fac_plot, degrees=0.05)

# -------------------
# 8) Plot Figure 1 with two panels
# -------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)

# ---- Panel A: ALL 21 storms (clipped to buffered CONUS) ----
states.boundary.plot(ax=axA, linewidth=0.8, color="black", zorder=1)

for row in tracks_gdf_all.itertuples(index=False):
    sid = row.storm_id
    cat_num = storm_meta.get(sid, {}).get("cat", 1)
    lw = cat_to_lw.get(cat_num, 2.0)
    label_cat = storm_meta.get(sid, {}).get("label_cat", f"Cat {cat_num}")
    label = f"{sid} ({label_cat})"

    # Plot each storm path with its fixed color and category-specific line width
    gpd.GeoSeries([row.geometry], crs="EPSG:4326").plot(
        ax=axA,
        linewidth=lw,
        color=storm_color.get(sid, "gray"),
        zorder=3,
        label=label
    )

axA.set_xlim([x0, x1])
axA.set_ylim([y0, y1])
axA.set_axis_off()
# axA.set_title("A. Hurricane Paths (21 cyclones)", fontsize=16, pad=10)

# ---- Panel A legend: Cat 1 -> Cat 5, and within cat chronological ----
ordered_for_legend = legend_order_cat_then_time(storms_all)

handlesA = []
labelsA = []
for sid in ordered_for_legend:
    cat_num = storm_meta.get(sid, {}).get("cat", 1)
    lw = cat_to_lw.get(cat_num, 2.0)
    label_cat = storm_meta.get(sid, {}).get("label_cat", f"Cat {cat_num}")
    label = f"{sid} ({label_cat})"

    handlesA.append(Line2D([0], [0], color=storm_color.get(sid, "gray"), linewidth=lw))
    labelsA.append(label)

axA.legend(
    handlesA,
    labelsA,
    loc="center left",
    bbox_to_anchor=(0.92, 0.5),
    frameon=True,
    title="Hurricane (category)",
    fontsize=9,
    title_fontsize=10
)

# ---- Panel B: FACILITY CENTROIDS ONLY (no tracks) ----
states.boundary.plot(ax=axB, linewidth=0.8, color="black", zorder=1)

# Plot facilities as colored points based on the storm they were assigned to
for sid in STORMS_RESTRICT:
    d = fac_plot.loc[fac_plot["storm_id_assigned"] == sid]
    if d.empty:
        continue
    d.plot(ax=axB, markersize=26, color=storm_color[sid], alpha=0.9, zorder=5)

axB.set_xlim([x0, x1])
axB.set_ylim([y0, y1])
axB.set_axis_off()
# axB.set_title("B. Dialysis facilities disrupted by hurricanes", fontsize=16, pad=10)

# ---- Panel B legend: marker-only, Cat 1 -> Cat 5, within-cat chronological ----
orderedB = legend_order_cat_then_time(STORMS_RESTRICT)

handlesB = []
labelsB = []
for sid in orderedB:
    cat_num = storm_meta.get(sid, {}).get("cat", 1)
    label_cat = storm_meta.get(sid, {}).get("label_cat", f"Cat {cat_num}")

    # Marker-only handle because Panel B is showing facility points, not track lines
    handlesB.append(
        Line2D([0], [0],
               linestyle="None",
               marker="o",
               markersize=7,
               markerfacecolor=storm_color[sid],
               markeredgecolor="none")
    )
    labelsB.append(f"{sid} ({label_cat})")

axB.legend(
    handlesB,
    labelsB,
    loc="center left",
    bbox_to_anchor=(0.80, 0.5),
    frameon=True,
    title="Hurricane associated\nwith closures",
    fontsize=9,
    title_fontsize=10
).get_title().set_multialignment("center")

plt.show()

# -------------------
# Save
# -------------------
out_png = OUT_DIR / "Figure1_two_panels_A_tracks_B_facility_centroids_only.png"
out_pdf = OUT_DIR / "Figure1_two_panels_A_tracks_B_facility_centroids_only.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.2)
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)

print("Saved:", out_png)
print("Saved:", out_pdf)
print("Panel B facilities plotted (unique facilities):", fac_plot.shape[0])
print("BUFFER_KM used:", BUFFER_KM)
