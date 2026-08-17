#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: April 21, 2026
# Description: This script takes the previously identified facility stress-day events and looks for larger operational 
# stress clusters by linking facilities that had stress events close together in both time and geography. These clusters
# are important since we want to know which facilities were disrupted near each other in space and time suggesting that
# a disaster (like a hurricane) occurred.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
import calendar
from matplotlib.colors import to_hex

# --------------------------
# Paths and spec
# --------------------------
YEARS = list(range(2011, 2023))

# Rolling-stress-day
BASE_STRESS = "/gpfs/data/cms-share/duas/54200/Jessy/data/derived/facility_rolling_stress_days"

ZCTA_PATH   = "/gpfs/data/cms-share/duas/52484/Jessy/data/public_data/data/shp_files/cb_2013_us_zcta_zip_500k/"
STATES_PATH = "/gpfs/data/cms-share/duas/52484/Jessy/data/public_data/data/shp_files/cb_2018_us_state_500k/"
OUT_DIR = Path(f"{BASE_STRESS}/figures_clustered")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cluster filter parameters
CLUSTER_MIN        = 5    # minimum number of facilities per spatio-temporal cluster (greater than 5)
TEMPORAL_LINK_DAYS = 2    # events within this many days can be linked (meaning if a facility has a disruption within that many days then it is part of that cluster
PROX_RADIUS_KM     = 200  # facilities within this distance are neighbors (meaning if at least two facilities are within 200km, then it is part of that cluster)
PROX_RADIUS_M      = PROX_RADIUS_KM * 1000.0 # takes the KM and converts it. Need to use M and not KM since projection is "EPSG:5070"

# --------------------------
# Functions
# --------------------------
def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    # returns the date of the nth occurrence of a given weekday in a month. Will help with identifying dates for holidays (see holiday_days() before)
    # This is needed since, for example, thanksgiving is always the fourth occurence of thursday and I don't want to be manually looking this up every time.
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    days = [
        d for d in c.itermonthdates(year, month)
        if d.month == month and d.weekday() == weekday
    ]
    return pd.Timestamp(days[n - 1])


def last_weekday_of_month(year: int, month: int, weekday: int) -> pd.Timestamp:
    # returns the last occurrence of the specified weekday within that month. Needed for Memorial day since Memorial day is the last monday of May.
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    days = [
        d for d in c.itermonthdates(year, month)
        if d.month == month and d.weekday() == weekday
    ]
    return pd.Timestamp(days[-1])


def _observed(date: pd.Timestamp) -> pd.Timestamp:
    if date.weekday() == 5:   # Saturday -> Friday
        return date - pd.Timedelta(days=1)
    if date.weekday() == 6:   # Sunday -> Monday
        return date + pd.Timedelta(days=1)
    return date


def holiday_days(year: int,
                 include_observed: bool = True,
                 include_dec24: bool = True) -> set:
    memorial  = last_weekday_of_month(year, 5, 0) # function to return the last monday of May.
    labor     = nth_weekday_of_month(year, 9, 0, 1) # Return the date of labor day in that particular year. (first monday)
    thanks    = nth_weekday_of_month(year, 11, 3, 4) # Return the date of thanksgiving day in that particular year. (fourth thursday)
    new_years = pd.Timestamp(year, 1, 1) # don't use nth_weekday_of_month() because new years is always on the 1st
    july4     = pd.Timestamp(year, 7, 4) # don't use nth_weekday_of_month() because july 4th is always on the 4th
    xmas      = pd.Timestamp(year, 12, 25) # don't use nth_weekday_of_month() because xmas is always on the 25th

    base = {memorial, thanks, xmas}

    # Extra dates leading up to christmas 
    if include_dec24:
        base.update({
            pd.Timestamp(year, 12, 22),
            pd.Timestamp(year, 12, 23),
            pd.Timestamp(year, 12, 24),
            pd.Timestamp(year, 5, 31), # Included to account for when memorial day occurred on 31st of May (e.g., 2021). This is okay to include since hurricanes typically do not occur during spring time. However, I manually checked to ensure I did not accidentally dropped a hurricane from NOAA's list.
        })

    if include_observed: # True would include ny and july 4th for exclusion.
        for d in [new_years, july4]:
            base.add(_observed(d))

    return {pd.Timestamp(b.date()) for b in base}


def exclude_holiday_days(df: pd.DataFrame,
                         date_col: str,
                         year: int,
                         include_observed: bool = True,
                         include_dec24: bool = True) -> pd.DataFrame:
    hd = holiday_days(year, include_observed=include_observed,
                      include_dec24=include_dec24)
    tmp = df.copy()
    tmp["_day"] = tmp[date_col].dt.normalize()
    out = tmp[~tmp["_day"].isin(hd)].drop(columns=["_day"]) # Exclude any if date falls on a holiday
    return out

def to_zip5(s: pd.Series) -> pd.Series:
    # Cleaning
    s = s.astype(str).str[:5].str.replace(r"\D", "", regex=True)
    return s.str.zfill(5)


def build_state_adjacency(states_gdf: gpd.GeoDataFrame) -> dict:
    # Build adjacency dict: state -> set(neighbor_states).
    # Two states are neighbors if their polygons touch.
    # Basically, for each state (in the dictionary), what other states touch them.

    # Start an empty adjacency dictionary. Set() is used so adjacent states only appear once
    adj = {abbrev: set() for abbrev in states_gdf["STUSPS"]}
    sidx = states_gdf.sindex # creates a spatial index. Will assist in finding likely states whose bounds overlap the current state’s bounding box

    # Loop through each state
    for i, row in states_gdf.iterrows():
        geom = row.geometry # This is the polygon for the current state.
        cand_idx = list(sidx.intersection(geom.bounds)) # Find potential candidate neighboring states using bounding boxes
        for j in cand_idx: # Loop through the candidate states
            if i == j: # Skip comparing the state to itself
                continue
            other = states_gdf.iloc[j] # Pull out the other state
            if geom.touches(other.geometry): # Check whether the two polygons actually touch
                adj[row["STUSPS"]].add(other["STUSPS"]) # Add the neighboring state abbreviation

    return adj


def connected_components_fac(nodes: set, adj: dict) -> list[set]:
    # Group together facilities that are in a cluster. E.g., if facilities are nodes = {0, 1, 2, 3, 4, 5} and we get adj = {0: {1}, 1: {0, 2}, 2: {1}, 3: {4}, 4: {3}, 5: set()}, then the clusters would be facilities 0, 1, 2 then 3, 4 then 5.
    
    seen = set()
    comps = []
    for n in nodes: # Go through every facility
        if n in seen: # Skip facilities already assigned
            continue
        stack = [n] # stack holds facilities still waiting to be explored
        comp = set() # comp is the current connected component being built
        while stack: # As long as there are still facilities left to explore in this current component, keep going.
            u = stack.pop() # This grabs the last facility that was added.
            if u in seen: # Skip if already visited
                continue
            seen.add(u) # u is officially explored
            comp.add(u) # u belongs to the current connected component
            for v in adj.get(u, []): # This looks up the neighbors of facility u in the adjacency dictionary.
                if v not in seen: # If a neighbor has not yet been visited, the neighboring facilities gets added to the stack so the search can continue from there.
                    stack.append(v) # appends the neighboring facility from the adj (adjacency dictionary)
        comps.append(comp)
    return comps


# --------------------------
# Load geography data only once
# --------------------------
print("Reading in ZCTA and state shapefiles...")

zcta = gpd.read_file(ZCTA_PATH)
if "ZCTA5CE10" in zcta.columns:
    zcta["zip5"] = zcta["ZCTA5CE10"]
elif "ZCTA5CE20" in zcta.columns:
    zcta["zip5"] = zcta["ZCTA5CE20"]
else:
    raise ValueError("Could not find ZCTA5CE10 or ZCTA5CE20 in ZCTA shapefile.")

states = gpd.read_file(STATES_PATH)
if "STATEFP" in states.columns:
    # drop AK, HI, territories
    states = states[~states["STATEFP"].isin(['02', '15', '69', '78', '66', '60'])].copy()

# Specify projection
target_crs = zcta.crs or "EPSG:4269"
zcta = zcta.to_crs(target_crs)
states = states.to_crs(target_crs)

# Get ZIP centroids
zpts = zcta[["zip5", "geometry"]].copy()
zpts["geometry"] = zpts.geometry.representative_point() # Replace each polygon with one point
zpts = gpd.GeoDataFrame(zpts, geometry="geometry", crs=zcta.crs)

# Assigning each ZIP point to a state
z_to_state = gpd.sjoin(
    zpts, states[["STUSPS", "geometry"]],
    how="left", predicate="within"
)[["zip5", "STUSPS"]]
z_to_state = z_to_state.dropna(subset=["STUSPS"]).drop_duplicates(subset=["zip5"])

# Get set of states that neighbors each state (e.g., CA neighbors OR, NV, AZ, etc...)
state_adj = build_state_adjacency(states[["STUSPS", "geometry"]])

# Copy for mapping merge
g_zip = zpts.copy()

# Collect final facilities across years
all_valid_facilities = []


# --------------------------
# Main
# --------------------------
for YEAR in YEARS:
    print(f"\n{YEAR}")

    # Rolling-stress-day file for this year
    df_path = f"{BASE_STRESS}/facility_rolling_stress_days_{YEAR}.csv"
    df = pd.read_csv(df_path, dtype="str")

    # Cleaning
    df["date"] = pd.to_datetime(df["earliest_stress_day"], errors="coerce") # earliest stress date is d0 (i.e., beginning of disruption)
    df = df.dropna(subset=["date"]).copy()
    df["zip5"] = to_zip5(df["zip"])
    df["earliest_denom"] = pd.to_numeric(df["earliest_denom"], errors="coerce")

    # Require denominator >= 11 (cell supression rule)
    mask_safe = df["earliest_denom"].ge(11)
    df_supp = df.loc[mask_safe].copy()

    # Remove holidays
    df_clean = exclude_holiday_days(
        df_supp, "date", YEAR,
        include_observed=True,
        include_dec24=True
    )

    # Attach state
    df_clean = df_clean.merge(z_to_state, on="zip5", how="left", validate="m:1")
    df_clean = df_clean.dropna(subset=["STUSPS"]).copy()

    if df_clean.empty:
        print(f"No non-holiday, non-suppressed, geo-locatable facilities for {YEAR}.")
        continue

    # --------------------------
    # Cluster assignment (temporal + spatial proximity)
    # --------------------------

    # Cleaning
    dfc = df_clean.copy().reset_index(drop=True)
    dfc["d0"] = dfc["date"].dt.normalize() # earliest stress date
    dfc["row_id"] = dfc.index
    dfc["cluster_id"] = pd.Series(index=dfc.index, dtype="Int64")

    # Attach geometry
    dfc_geo = dfc.merge(g_zip[["zip5", "geometry"]], on="zip5", how="left")
    dfc_geo = gpd.GeoDataFrame(dfc_geo, geometry="geometry", crs=zcta.crs)
    dfc_geo = dfc_geo.dropna(subset=["geometry"]).copy()

    if dfc_geo.empty:
        print(f"No mappable facilities for {YEAR}.")
        continue

    # Project for distance calculations
    proj_crs = "EPSG:5070"  # US Albers (meters)
    dfc_geo_proj = dfc_geo.to_crs(proj_crs)

    # Build spatio-temporal adjacency
    nodes = set(dfc_geo_proj.index) # each row is a facility stress-event
    fac_adj = {i: set() for i in nodes} # This makes a dictionary where each node starts with an empty set of neighbors.
    sidx = dfc_geo_proj.sindex # uses the spatial index to quickly find possible nearby neighbors

    for i, row in dfc_geo_proj.iterrows(): # Loop through each facility-event row
        geom_i = row.geometry # pull location of the facility ZIP point
        date_i = row["d0"] # pull event date
        state_i = row["STUSPS"] # pull state abbreviation

        # candidate neighbors in space
        buf_bounds = geom_i.buffer(PROX_RADIUS_M).bounds # creates a buffer of radius PROX_RADIUS_M around the current point
        cand_idx = list(sidx.intersection(buf_bounds)) # returns the indices of rows whose geometries intersect that bounding box (candidate neighbors)

        for j in cand_idx: # examines each candidate row j more carefully.
            if i == j: # A row should not be linked to itself.
                continue
            other = dfc_geo_proj.iloc[j] # other is just the other candidate row being compared to the current row i
            date_j = other["d0"]
            state_j = other["STUSPS"]

            # temporal proximity (<= TEMPORAL_LINK_DAYS apart)
            if abs((date_j - date_i).days) > TEMPORAL_LINK_DAYS: # if the two facility dates are more than TEMPORAL_LINK_DAYS apart, do not link them
                continue

            # If they are not in the same state, and the second state is not adjacent to the first state, then do not link them
            if state_j != state_i and state_j not in state_adj.get(state_i, set()):
                continue

            # Checks whether the two facility-event points are actually within 200 km of each other.
            if geom_i.distance(other.geometry) <= PROX_RADIUS_M:
                fac_adj[i].add(j) # i is a neighbor of j
                fac_adj[j].add(i) # j is a neighbor of i (do it in both direction)

    # Connected components in spatio-temporal graph
    comps = connected_components_fac(nodes, fac_adj)

    # Start cluster IDs and metadata storage
    cid = 1
    cluster_meta = {}  # cluster_id -> (start_date, end_date)

    for comp in comps: # Loop through each connected component
        comp = list(comp)
        # Map component rows back to original dfc rows
        row_ids = dfc_geo_proj.loc[comp, "row_id"].unique() # get the original row IDs in dfc corresponding to the component nodes
        facility_ids = dfc.loc[row_ids, "PRVDR_NUM"].unique() # get the distinct provider numbers represented by those rows

        # require enough distinct facilities
        if len(facility_ids) < CLUSTER_MIN: # This is the minimum cluster size rule. Need to have at least 5 facilities in a cluster. If not, then skip.
            continue

        # cluster time span: find the earliest event date and latest event date of each cluster (some facilities were disrupted on Monday and others on Tuesday but they are temporally close (and should be spatially close too), so we want to min and max of the date within these clusters.
        comp_dates = dfc.loc[row_ids, "d0"]
        start_d = comp_dates.min()
        end_d   = comp_dates.max()

        # Assign the cluster ID to the rows
        dfc.loc[row_ids, "cluster_id"] = cid
        cluster_meta[cid] = (start_d, end_d)
        cid += 1

    # Drop rows that never got a cluster ID (i.e., they weren't a part of the cluster)
    kept = dfc.dropna(subset=["cluster_id"]).copy()
    if kept.empty:
        print(
            f"No spatio-temporal clusters with >={CLUSTER_MIN} facilities "
            f"(<={PROX_RADIUS_KM} km, <={TEMPORAL_LINK_DAYS} days) for {YEAR}."
        )
        continue

    # --------------------------
    # Attach geometry again, then enforce ≥5 plotted facilities per cluster
    # --------------------------
    kept_geo = kept.merge(g_zip[["zip5", "geometry"]], on="zip5", how="left") # merge ZIP geometry back onto the kept rows. This is to visual map them later.
    kept_geo = gpd.GeoDataFrame(kept_geo, geometry="geometry", crs=zcta.crs)
    kept_geo = kept_geo.dropna(subset=["geometry"])

    if kept_geo.empty:
        print(f"No mappable facilities for clustered events in {YEAR}.")
        continue

    # This counts how many distinct facilities each cluster still has after the geometry join.
    geo_cluster_sizes = (
        kept_geo.groupby("cluster_id")["PRVDR_NUM"]
        .nunique()
        .rename("n_facilities_geo")
    )
    valid_geo_clusters = geo_cluster_sizes[geo_cluster_sizes >= CLUSTER_MIN].index # This picks the cluster IDs that still have at least 5 unique facilities

    # Filter both tables to those valid geo clusters
    kept_geo = kept_geo[kept_geo["cluster_id"].isin(valid_geo_clusters)].copy()
    kept = kept[kept["cluster_id"].isin(valid_geo_clusters)].copy()

    if kept_geo.empty:
        print(
            f"All clusters dropped after geo filter for {YEAR} "
            f"(no cluster with >={CLUSTER_MIN} plotted facilities)."
        )
        continue

    # --------------------------
    # Window summary (for printing + legend)
    # --------------------------
    # Not relevant to identifying the valid facilities in a cluster. Simply to turn cluster ids into readable date-window labels, assigning colors, and producing a printed summary table for the map
    
    cluster_ids = sorted(kept["cluster_id"].unique())
    cluster_window = {int(k): cluster_meta[int(k)] for k in cluster_ids}

    # dedupe windows, chrono order
    unique_windows = sorted(
        {cluster_window[cid] for cid in cluster_ids},
        key=lambda w: (w[0], w[1])
    )

    from matplotlib import cm # Set up a color palette
    cmap = plt.get_cmap("tab20")
    color_by_window = { # Assign a color to each unique time window (e.g., dec 1 to 4 is a color)
        win: cmap(i % 20) for i, win in enumerate(unique_windows)
    }

    def window_label(sd, ed): # Define a readable label for each window
        if sd.month == ed.month:
            return f"{sd.strftime('%b %d')}–{ed.strftime('%d')}"
        else:
            return f"{sd.strftime('%b %d')}–{ed.strftime('%b %d')}"

    # Add start and end dates to each row in kept
    kept["_sd"] = kept["cluster_id"].map(lambda cid: cluster_window[int(cid)][0])
    kept["_ed"] = kept["cluster_id"].map(lambda cid: cluster_window[int(cid)][1])
    kept["_win"] = list(zip(kept["_sd"], kept["_ed"]))
    kept["_label"] = kept.apply(lambda r: window_label(r["_sd"], r["_ed"]), axis=1)
    kept["_color"] = kept.apply(
        lambda r: to_hex(color_by_window[(r["_sd"], r["_ed"])]), axis=1
    )

    # Build the summary table
    summary = (
        kept.groupby(["_win", "_label", "_color"])
            .agg(
                n_events=("PRVDR_NUM", "size"),
                n_facilities=("PRVDR_NUM", "nunique"),
                states=("STUSPS", lambda s: ",".join(sorted(set(s)))),
            )
            .reset_index()
            .rename(columns={"_label": "window", "_color": "color_hex"})
    )

    summary["win_start"] = summary["_win"].apply(lambda w: w[0])
    summary["win_end"] = summary["_win"].apply(lambda w: w[1])
    summary = summary.sort_values(["win_start", "win_end"]).drop(columns=["_win"])

    print(
        f"\n=== Cluster summary for {YEAR} "
        f"(each cluster has >={CLUSTER_MIN} facilities; "
        f"≤{PROX_RADIUS_KM} km & ≤{TEMPORAL_LINK_DAYS} days) ==="
    )
    print(summary[["window", "n_facilities", "n_events", "states", "color_hex"]].to_string(index=False))

    # --------------------------
    # Collect valid facilities for output CSV. Will be used to identify bene's from these disrupted facilities
    # --------------------------
    fac_year = (
        kept[[
            "PRVDR_NUM", "earliest_stress_day", "earliest_avg_sessions",
            "earliest_denom", "zip", "zip5", "STUSPS", "cluster_id"
        ]]
        .drop_duplicates()
        .copy()
    )
    fac_year["year"] = YEAR
    all_valid_facilities.append(fac_year)

    # --------------------------
    # Map for this year
    # --------------------------
    # This is for visual checks of each cluster of facilities.
    
    states_map = states.to_crs(target_crs)
    kept_geo   = kept_geo.to_crs(target_crs)

    print(kept_geo.head(60))
    print(kept_geo.tail(60))
    print(kept_geo['zip5'].to_list())

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    states_map.plot(ax=ax, color="white", edgecolor="black", linewidth=0.3, zorder=1)

    # Plot facilities, colored by cluster date window
    for cid_i, grp in kept_geo.groupby("cluster_id"):
        sd, ed = cluster_window[int(cid_i)]
        col = color_by_window[(sd, ed)]
        grp.plot(ax=ax, markersize=18, color=col, alpha=0.9, zorder=2)

    # Legend
    handles, labels = [], []
    for sd, ed in unique_windows:
        win = (sd, ed)
        if win not in color_by_window:
            continue
        col = color_by_window[win]
        handles.append(
            plt.Line2D(
                [], [], marker="o", linestyle="", markersize=6,
                markerfacecolor=col, markeredgecolor="none"
            )
        )
        labels.append(window_label(sd, ed))

    xmin, ymin, xmax, ymax = states_map.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if labels and len(labels) <= 12:
        ax.legend(
            handles, labels, title="Cluster date windows",
            loc="lower left", frameon=True, fontsize=8,
        )

    ax.set_title(
        f"Operational stress clusters (>={CLUSTER_MIN} facilities; "
        f"<={PROX_RADIUS_KM} km & <={TEMPORAL_LINK_DAYS} days)\n"
        f"Facility-level proximity — {YEAR}"
    )
    ax.set_axis_off()
    fig.tight_layout()

    out_fig = OUT_DIR / (
        f"map_clusters_{YEAR}_clusters_ge{CLUSTER_MIN}"
        f"_prox{PROX_RADIUS_KM}km_t{TEMPORAL_LINK_DAYS}d.png"
    )
    fig.savefig(out_fig, dpi=200)
    plt.show()
    plt.close(fig)
    print(f"Saved map to {out_fig}")


# --------------------------
# After loop: combined valid facilities and export
# --------------------------
if all_valid_facilities:
    valid_facilities_all_years = pd.concat(all_valid_facilities, ignore_index=True)
    valid_facilities_all_years["earliest_stress_day"] = pd.to_datetime(
        valid_facilities_all_years["earliest_stress_day"]
    )
    valid_facilities_all_years = valid_facilities_all_years.sort_values(
        ["earliest_stress_day", "PRVDR_NUM"]
    )

    print(valid_facilities_all_years.head(20))
    print("Total rows:", valid_facilities_all_years.shape[0])
    print("Unique clusters:", valid_facilities_all_years.cluster_id.nunique())
    print(valid_facilities_all_years.cluster_id.value_counts())

    out_path = BASE_STRESS + "/valid_facilities_operational_stress_2011_2022.csv"
    valid_facilities_all_years.to_csv(out_path, index=False)
    print(f"Saved valid facilities to: {out_path}")


