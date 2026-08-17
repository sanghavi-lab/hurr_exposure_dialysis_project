#----------------------------------------------------------------------------------------------------------------------#
# Project: Hurricane impacts on dialysis Medicare beneficiaries
# Author: Jessy Nguyen
# Last Updated: May 4, 2026
# Description: This script takes the 5 final Sandy dialysis schedule groups that were chosen [(1) regular schedule, 
# (2) regular schedule but transfer, (3) disrupted, (4) early but not disrupted, (5) early disrupted], selects 11 
# beneficiaries from a unique facility for each group (cell suppression policy), and turns them into a representative 
# heatmap showing the overall dialysis pattern across storm week. This representative heatmap will be manually overlayed 
# (as a legend) on a map that plots the location of the 5 selected facilities, each resembling one unique dialysis schedule 
# group. The borders of the legend was also manually created for convenience.
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------
# Import modules
# -------------------------

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

# -------------------------
# Dask setup
# -------------------------
from dask.distributed import Client
import dask

cust_temp_dir = '/gpfs/data/cms-share/duas/52484/Jessy/temp_space/tmp/'
dask.config.set({"temporary-directory": cust_temp_dir})
dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s"
})

client = Client("10.50.87.74:43773")

# -------------------------
# Shared panel palette and display label
# -------------------------
# These are the fixed display labels / colors used across the final heatmaps and the clean
# provider map. This helps keep the schedule panels visually consistent across outputs.
PANEL_PALETTE = {
    "A. Regular schedule": "#4d4d4d",
    "B. Not disrupted (with a transfer)": "#9467bd",
    "C. Disrupted": "#d95f02",
    "D. Early dialysis, not disrupted": "#2ca02c",
    "E. Early dialysis, disrupted": "#d62728",
}

DISPLAY_LABELS = {
    "A. Regular schedule": "Regular schedule",
    "B. Not disrupted (with a transfer)": "Not disrupted (with a transfer)",
    "C. Disrupted": "Disrupted",
    "D. Early dialysis, not disrupted": "Early dialysis, not disrupted",
    "E. Early dialysis, disrupted": "Early dialysis, disrupted",
}

SHORT_DISPLAY_LABELS = {
    "A. Regular schedule": "Regular schedule",
    "B. Not disrupted (with a transfer)": "Not disrupted w/ transfer",
    "C. Disrupted": "Disrupted",
    "D. Early dialysis, not disrupted": "Early / not disrupted",
    "E. Early dialysis, disrupted": "Early / disrupted",
}

# =========================================================
# PART 1: FINAL HEATMAPS
# =========================================================
# Summary - This first part of the script takes the already-defined Sandy schedule groups,
# chooses a final set of representative beneficiaries for each of the 5 panels, and then
# creates the heatmap figure(s) used for the final display.

# -------------------------
# Paths
# -------------------------
IN_DIR = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "derived/sandy_schedule_signature_groups_broad_v01/"
)

GROUPS_CSV = os.path.join(IN_DIR, "grouped_candidate_patterns_nge11.csv")
MEMBERS_CSV = os.path.join(IN_DIR, "candidate_group_members.csv")
DAY_CSV = os.path.join(IN_DIR, "bene_day_level_for_plotting.csv")
OP_ROWS_CSV = os.path.join(IN_DIR, "broad_cohort_op_rows.csv")

# Write to a new derived folder so outputs are versioned rather than overwriting earlier figures/files
OUT_DIR = (
    "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/"
    "derived/sandy_schedule_signature_groups_broad_v01/final_heatmaps_v02_clean_map/"
)
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Selected groups
# -------------------------
# These 5 rows are the final schedule groups/panels already chosen for the figure.
# The groups are: (1) regular schedule, (2) regular schedule but transfer, (3) disrupted, 
# (4) early but not disrupted, (5) early disrupted
# It is taking these 5 specific category + group_key + dialysis_signature combinations and building
# the final display outputs from them.
SELECTED = [
    {
        "panel_order": 1,
        "panel_label": "A. Regular schedule",
        "category": "regular_schedule",
        "group_key": "5e3a8ac6a371",
        "dialysis_signature": "Sun:.|Mon:B|Tue:.|Wed:B|Thu:.|Fri:B|Sat:."
    },
    {
        "panel_order": 2,
        "panel_label": "B. Not disrupted (with a transfer)",
        "category": "not_disrupted_transfer_or_rescheduled",
        "group_key": "121419572a1e",
        "dialysis_signature": "Sun:.|Mon:B|Tue:.|Wed:Y|Thu:.|Fri:B|Sat:."
    },
    {
        "panel_order": 3,
        "panel_label": "C. Disrupted",
        "category": "disrupted",
        "group_key": "1bb497c78da0",
        "dialysis_signature": "Sun:.|Mon:.|Tue:.|Wed:B|Thu:.|Fri:B|Sat:."
    },
    {
        "panel_order": 4,
        "panel_label": "D. Early dialysis, not disrupted",
        "category": "early_not_disrupted",
        "group_key": "6dcef853d9c6",
        "dialysis_signature": "Sun:B|Mon:.|Tue:.|Wed:B|Thu:.|Fri:B|Sat:."
    },
    {
        "panel_order": 5,
        "panel_label": "E. Early dialysis, disrupted",
        "category": "early_disrupted",
        "group_key": "03bef5afa684",
        "dialysis_signature": "Sun:B|Mon:.|Tue:.|Wed:.|Thu:.|Fri:B|Sat:."
    },
]

# This will be used to force one of the providers to change to the below specified provider. 
# I changed it because the previous one that was automatically selected wasn't near the other
# facilities so plotting it on the map looks too spread out
PREFERRED_PROVIDER_BY_PANEL = {
    3: "332670",
}

# Each panel will show 11 beneficiary rows in the detailed selection step
N_ROWS_PER_PANEL = 11

# If True, also create provider-week context heatmaps for each provider represented in the
# selected rows. These are additional reference figures, not the main collapsed article figure.
MAKE_PROVIDER_REFERENCE_PLOTS = True

# If True, display figures interactively in addition to saving them
SHOW_PLOTS = True

# Fixed day order / axis labels used throughout the heatmaps
DAY_ORDER = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
DATE_LABELS = ["Sun\n10/28", "Mon\n10/29", "Tue\n10/30", "Wed\n10/31", "Thu\n11/1", "Fri\n11/2", "Sat\n11/3"]
SHORT_DATE_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]



def load_inputs():
    # Read the grouped schedule outputs created upstream:
    #   - groups: one row per grouped schedule pattern
    #   - members: beneficiary members belonging to each group
    #   - day: beneficiary x day plotting file
    #   - op_rows: OP dialysis rows used later for provider-week context plots
    groups = pd.read_csv(GROUPS_CSV, dtype=str)
    members = pd.read_csv(MEMBERS_CSV, dtype=str)
    day = pd.read_csv(DAY_CSV, dtype=str)
    op_rows = pd.read_csv(OP_ROWS_CSV, dtype=str)

    # Normalize common ID/text fields so merges behave cleanly downstream
    for df in [groups, members, day, op_rows]:
        if "BENE_ID" in df.columns:
            df["BENE_ID"] = df["BENE_ID"].astype(str)
        if "usual_facility_id" in df.columns:
            df["usual_facility_id"] = df["usual_facility_id"].astype(str)
        if "facility_id" in df.columns:
            df["facility_id"] = df["facility_id"].astype(str)
        if "group_key" in df.columns:
            df["group_key"] = df["group_key"].astype(str)
        if "category" in df.columns:
            df["category"] = df["category"].astype(str)
        if "loc_status" in df.columns:
            df["loc_status"] = df["loc_status"].astype(str)

    # Convert numeric plotting flags from strings to integer indicators
    for col in ["ed_any", "ip_any", "pre_mwf_flag"]:
        if col in day.columns:
            day[col] = pd.to_numeric(day[col], errors="coerce").fillna(0).astype(int)

    # Normalize dates for later filtering / plotting
    if "date" in day.columns:
        day["date"] = pd.to_datetime(day["date"])
    if "date" in op_rows.columns:
        op_rows["date"] = pd.to_datetime(op_rows["date"])

    return groups, members, day, op_rows


def select_top_provider_heavy(
    group_members: pd.DataFrame,
    n_select: int = 11,
    preferred_provider: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Summary - From one selected schedule group, choose the final beneficiaries to display
    # in the panel. The logic is provider-heavy, meaning it prioritizes beneficiaries from
    # the most represented usual provider(s), with an optional hard-coded preferred provider.
    gm = group_members.copy()
    gm["usual_facility_id"] = gm["usual_facility_id"].astype(str)
    gm["BENE_ID"] = gm["BENE_ID"].astype(str)

    # Count how many unique beneficiaries belong to each usual provider inside this group.
    # This helps rank providers from most represented to least represented.
    provider_counts = (
        gm.groupby("usual_facility_id")["BENE_ID"]
        .nunique()
        .reset_index(name="n_benes_provider")
        .sort_values(["n_benes_provider", "usual_facility_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    selected_rows = [] # store the final selected beneficiary rows
    selected_benes = set() # helps avoid selecting the same beneficiary twice

    # If a preferred provider was specified for this panel, pull from that provider first
    if preferred_provider is not None:
        preferred_provider = str(preferred_provider)
        sub_pref = (
            gm[gm["usual_facility_id"] == preferred_provider]
            .drop_duplicates(subset=["BENE_ID"])
            .sort_values(["BENE_ID"])
            .copy()
        )

        # Add preferred-provider beneficiaries first until the panel is filled or we run out
        for _, r in sub_pref.iterrows():
            bene = r["BENE_ID"]
            if bene in selected_benes:
                continue
            selected_rows.append(r)
            selected_benes.add(bene)
            if len(selected_rows) == n_select:
                break

    # If the preferred provider did not fully fill the panel, continue filling from the
    # remaining providers in descending provider-size order
    if len(selected_rows) < n_select:
        for _, prow in provider_counts.iterrows():
            prov = prow["usual_facility_id"]

            if preferred_provider is not None and prov == preferred_provider:
                continue

            sub = (
                gm[gm["usual_facility_id"] == prov]
                .drop_duplicates(subset=["BENE_ID"])
                .sort_values(["BENE_ID"])
                .copy()
            )

            # Keep adding beneficiaries provider by provider until n_select rows are filled
            for _, r in sub.iterrows():
                bene = r["BENE_ID"]
                if bene in selected_benes:
                    continue
                selected_rows.append(r)
                selected_benes.add(bene)
                if len(selected_rows) == n_select:
                    break

            if len(selected_rows) == n_select:
                break

    selected_rows = selected_rows[:n_select]
    selected = pd.DataFrame(selected_rows).copy()

    # QC check - the panel is expected to have exactly n_select beneficiaries
    if len(selected) != n_select:
        raise ValueError(
            f"Expected {n_select} selected beneficiaries, got {len(selected)} "
            f"for group_key={gm['group_key'].iloc[0] if not gm.empty else 'NA'}"
        )

    selected = selected.reset_index(drop=True)

    # Summarize how many of the finally selected rows came from each usual provider.
    # This table is later used for figure notes / provider footnotes / map selection.
    provider_summary = (
        selected.groupby("usual_facility_id")["BENE_ID"]
        .nunique()
        .reset_index(name="n_selected")
        .sort_values(["n_selected", "usual_facility_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return selected, provider_summary


def derive_plot_value(row) -> int:
    # Convert the day-level status into one plotting code used by the detailed heatmap. These are mainly for QCs to ensure the correct facilities with at least 11 bene's of a particular schedule was selected.
    # Priority goes:
    #   4 = IP overlay
    #   3 = ED overlay
    #   2 = dialysis at usual facility or mixed day (B/M)
    #   1 = dialysis at another facility only (Y)
    #   0 = no dialysis / no overlay
    ip_any = int(row["ip_any"])
    ed_any = int(row["ed_any"])
    loc = str(row["loc_status"])

    if ip_any == 1:
        return 4
    if ed_any == 1:
        return 3
    if loc in {"B", "M"}:
        return 2
    if loc == "Y":
        return 1
    return 0


def derive_collapsed_plot_value_from_loc_status(loc: str) -> int:
    # Simpler plotting code for the collapsed figure that will be used as an exhibit.
    # Here ED/IP overlays are intentionally not carried forward. The collapsed figure is
    # meant to show the shared OP dialysis schedule only.
    loc = str(loc)
    if loc in {"B", "M"}:
        return 2
    if loc == "Y":
        return 1
    return 0


def build_panel_matrix(
    selected_benes: pd.DataFrame,
    day_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Summary - Build the row x day matrix for one detailed heatmap panel.
    # Each row is one beneficiary and each column is one day of Sandy week.
    use = day_df.merge(
        selected_benes[["BENE_ID", "usual_facility_id"]],
        on=["BENE_ID", "usual_facility_id"],
        how="inner"
    ).copy()

    # Rank providers by how many selected rows they contribute, so the displayed rows can be
    # ordered provider-first rather than purely alphabetically
    provider_rank = (
        selected_benes.groupby("usual_facility_id")["BENE_ID"]
        .nunique()
        .reset_index(name="n_provider")
        .sort_values(["n_provider", "usual_facility_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    provider_rank["provider_sort"] = np.arange(len(provider_rank))
    use = use.merge(provider_rank[["usual_facility_id", "provider_sort"]], on="usual_facility_id", how="left")

    # Build the final beneficiary row order used in the heatmap
    bene_order = (
        use[["BENE_ID", "usual_facility_id", "provider_sort"]]
        .drop_duplicates()
        .sort_values(["provider_sort", "usual_facility_id", "BENE_ID"])
        .reset_index(drop=True)
    )
    bene_order["row_num"] = np.arange(1, len(bene_order) + 1)

    use = use.merge(bene_order[["BENE_ID", "row_num"]], on="BENE_ID", how="left")

    # Translate each bene-day row into a single heatmap plotting value
    use["plot_value"] = use.apply(derive_plot_value, axis=1)
    use["dow"] = pd.Categorical(use["dow"], categories=DAY_ORDER, ordered=True)

    # Pivot to the matrix format expected by imshow:
    # rows = beneficiaries, columns = Sun-Sat, values = plotting code
    matrix = (
        use.pivot_table(index="row_num", columns="dow", values="plot_value", aggfunc="max")
        .reindex(columns=DAY_ORDER)
        .sort_index()
    )

    matrix.index = [f"{i}" for i in matrix.index.tolist()]

    # Keep row metadata too in case we need to know which beneficiary landed on which row
    row_meta = bene_order.copy()
    row_meta["display_label"] = row_meta["row_num"].astype(str)
    return matrix, row_meta


def build_collapsed_panel_row(
    selected_benes: pd.DataFrame,
    day_df: pd.DataFrame,
    panel_label: str
) -> pd.DataFrame:
    # Summary - Collapse one panel down to a single row, but only if all selected
    # beneficiaries truly share the same OP schedule across Sandy week.
    use = day_df.merge(
        selected_benes[["BENE_ID", "usual_facility_id"]],
        on=["BENE_ID", "usual_facility_id"],
        how="inner"
    ).copy()

    if use.empty:
        raise ValueError(f"No day-level rows found for collapsed panel: {panel_label}")

    use["dow"] = pd.Categorical(use["dow"], categories=DAY_ORDER, ordered=True)

    # QC check - within each beneficiary/day there should only be one loc_status value
    chk = (
        use.groupby(["BENE_ID", "dow"], observed=False)["loc_status"]
        .agg(lambda x: sorted(set(x.astype(str))))
        .reset_index(name="loc_status_values")
    )

    bad = chk[chk["loc_status_values"].apply(len) != 1].copy()
    if not bad.empty:
        raise ValueError(
            f"Found inconsistent loc_status within bene/day for collapsed panel {panel_label}."
        )

    chk["loc_status"] = chk["loc_status_values"].str[0]

    # Reshape to one row per beneficiary across Sun-Sat
    bene_sched = (
        chk.pivot(index="BENE_ID", columns="dow", values="loc_status")
        .reindex(columns=DAY_ORDER)
        .copy()
    )

    if bene_sched.empty:
        raise ValueError(f"No beneficiary schedules built for collapsed panel: {panel_label}")

    # The collapsed figure only works if every selected beneficiary in the panel shares the
    # same OP schedule. If not, suppressing the panel to one row would be misleading.
    unique_sched = bene_sched.drop_duplicates().copy()
    if len(unique_sched) != 1:
        raise ValueError(
            f"Collapsed panel {panel_label} is not suppressible as one row: "
            f"selected beneficiaries do not all share the same OP schedule."
        )

    one_sched = unique_sched.iloc[0]
    collapsed_vals = [derive_collapsed_plot_value_from_loc_status(one_sched[d]) for d in DAY_ORDER]

    collapsed_matrix = pd.DataFrame(
        [collapsed_vals],
        index=[panel_label],
        columns=DAY_ORDER
    )
    return collapsed_matrix


def provider_footnote_string(provider_summary: pd.DataFrame) -> str:
    # Convert the provider summary table to a compact text string like:
    # 332670 (n=7); 123456 (n=4)
    parts = [
        f"{r['usual_facility_id']} (n={int(r['n_selected'])})"
        for _, r in provider_summary.iterrows()
    ]
    return "; ".join(parts)


def draw_single_heatmap_panel(
    ax,
    matrix: pd.DataFrame,
    panel_title: str,
    n_total_group: int,
    provider_note: str,
    y_label: str = "Selected beneficiaries"
):
    # Draw one detailed heatmap panel where rows are individual beneficiaries and columns
    # are Sandy-week days. Colors reflect no dialysis / another facility / usual facility
    # / ED overlay / IP overlay.
    cmap = ListedColormap([
        "#ffffff",
        "#f1c40f",
        "#1f77b4",
        "#8e44ad",
        "#c0392b",
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=cmap.N)

    arr = matrix.values
    ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(np.arange(len(DAY_ORDER)))
    ax.set_xticklabels(DATE_LABELS, rotation=0, fontsize=9)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=8)

    ax.set_title(
        f"{panel_title}\nSelected 11 beneficiaries from pooled group (group n={n_total_group})",
        fontsize=10,
        loc="left",
        pad=10
    )
    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=9)

    # Minor-grid lines help non-coders visually follow rows and days more easily
    ax.set_xticks(np.arange(-.5, len(DAY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Dashed line marks Monday 10/29/2012
    zero_col = DAY_ORDER.index("Mon")
    ax.axvline(zero_col - 0.5, color="black", linestyle="--", linewidth=0.8)

    # Provider note goes below the panel so readers can see which usual providers are represented
    ax.text(
        0.0, -0.25,
        f"Usual provider IDs among displayed rows: {provider_note}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8
    )


def draw_collapsed_heatmap(
    ax,
    matrix: pd.DataFrame,
    use_short_labels: bool = False,
    use_short_dates: bool = False,
    tick_fontsize: int = 10,
    grid_lw: float = 0.5
):
    # Draw the simpler article-safe figure where each row is one shared OP schedule pattern
    # rather than one individual beneficiary.
    cmap = ListedColormap([
        "#ffffff",
        "#f1c40f",
        "#1f77b4",
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=cmap.N)

    arr = matrix.values
    ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(np.arange(len(DAY_ORDER)))
    if use_short_dates:
        ax.set_xticklabels(SHORT_DATE_LABELS, rotation=0, fontsize=tick_fontsize)
    else:
        ax.set_xticklabels(DATE_LABELS, rotation=0, fontsize=tick_fontsize)

    ax.set_yticks(np.arange(matrix.shape[0]))
    row_labels = matrix.index.tolist()

    if use_short_labels:
        display_row_labels = [SHORT_DISPLAY_LABELS.get(x, x) for x in row_labels]
    else:
        display_row_labels = [DISPLAY_LABELS.get(x, x) for x in row_labels]

    ax.set_yticklabels(display_row_labels, fontsize=tick_fontsize)
    for lab in ax.get_yticklabels():
        # White box helps the row label remain readable even when extra annotations are added nearby
        lab.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.2))

    ax.set_xlabel("")

    ax.set_xticks(np.arange(-0.5, len(DAY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=grid_lw)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Dashed line marks Monday 10/29/2012
    zero_col = DAY_ORDER.index("Mon")
    ax.axvline(zero_col - 0.5, color="black", linestyle="--", linewidth=0.8)


def add_dynamic_label_squares(ax, row_keys, x_offset_px=-8, y_offset_px=-3, fontsize=18):
    # Add colored squares next to row labels after the figure is rendered.
    # This is done in figure coordinates because the exact text bounding boxes are only known
    # after matplotlib draws the axis.
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ticks = ax.get_yticklabels()

    for tick, row_key in zip(ticks, row_keys):
        color = PANEL_PALETTE.get(row_key, "#777777")

        bbox = tick.get_window_extent(renderer=renderer)
        x_disp = bbox.x0 + x_offset_px
        y_disp = (bbox.y0 + bbox.y1) / 2.0 + y_offset_px

        x_fig, y_fig = fig.transFigure.inverted().transform((x_disp, y_disp))

        fig.text(
            x_fig,
            y_fig,
            "■",
            color=color,
            ha="right",
            va="center",
            fontsize=fontsize
        )


def make_collapsed_article_safe_figure(
    panel_objects: List[Dict],
    day_df: pd.DataFrame,
    out_png: str,
    out_pdf: str
):
    # Summary - Create the final article-safe figure where each of the 5 chosen panels is
    # collapsed to one row representing the shared OP schedule for that panel.
    collapsed_rows = []

    for pobj in panel_objects:
        collapsed_row = build_collapsed_panel_row(
            selected_benes=pobj["selected_benes"],
            day_df=day_df,
            panel_label=pobj["panel_label"]
        )
        collapsed_rows.append(collapsed_row)

    collapsed_matrix = pd.concat(collapsed_rows, axis=0).reindex(columns=DAY_ORDER)

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    draw_collapsed_heatmap(ax=ax, matrix=collapsed_matrix)

    fig.suptitle(
        "Dialysis schedules during Sandy with each row corresponding to\na unique schedule shared by >=11 beneficiaries",
        fontsize=13, y=0.98
    )

    # Legend for the collapsed OP-only display
    legend_handles = [
        mpatches.Patch(color="#ffffff", ec="black", label="No dialysis"),
        mpatches.Patch(color="#1f77b4", label="Dialysis at facility"),
        mpatches.Patch(color="#f1c40f", label="Dialysis at another facility"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.02)
    )

    # For each panel, note the top provider represented among the 11 selected rows.
    # These are the providers later carried into the clean provider map.
    provider_lines = []
    for pobj in panel_objects:
        top_provider = (
            pobj["provider_summary"]
            .sort_values(["n_selected", "usual_facility_id"], ascending=[False, True])
            .iloc[0]["usual_facility_id"]
        )
        row_letter = pobj["panel_label"].split(".")[0]
        provider_lines.append(f"Row {row_letter} ({top_provider})")

    fig.text(
        0.20, -0.09,
        "Each row represents the shared outpatient dialysis schedule across at least 11 beneficiaries.\n"
        "Colored squares at left match the provider colors used in the provider map.\n"
        f"Providers: {', '.join(provider_lines)}.\n"
        "Dashed line marks Monday 10/29/2012.",
        ha="left", va="bottom", fontsize=8
    )

    plt.tight_layout(rect=[0.10, 0.08, 1, 0.93])

    # Add the colored squares after layout so they align with the final row label positions
    add_dynamic_label_squares(
        ax=ax,
        row_keys=collapsed_matrix.index.tolist(),
        x_offset_px=-8,
        y_offset_px=+2.5,
        fontsize=18
    )

    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=250, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def make_provider_reference_plots(
    panel_objects: List[Dict],
    day_df: pd.DataFrame,
    op_rows_df: pd.DataFrame
):
    # Summary - Make optional provider-week context heatmaps. These show all beneficiaries
    # who dialyzed at a given provider during Sandy week, so the reader can see the broader
    # provider-week context around the selected panel rows.
    storm_start = pd.Timestamp("2012-10-28")
    storm_end = pd.Timestamp("2012-11-03")

    # Restrict OP rows to Sandy week only
    op_sw = op_rows_df[
        (op_rows_df["date"] >= storm_start) &
        (op_rows_df["date"] <= storm_end)
    ].copy()

    for pobj in panel_objects:
        # Safe file-name version of the panel label
        panel_label_safe = (
            pobj["panel_label"]
            .replace(".", "")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        )

        # Count how many selected rows in the main panel came from each usual provider
        selected_provider_counts = (
            pobj["selected_benes"]
            .groupby("usual_facility_id")["BENE_ID"]
            .nunique()
            .reset_index(name="n_selected_in_main_panel")
            .sort_values(["n_selected_in_main_panel", "usual_facility_id"], ascending=[False, True])
        )

        for _, prow in selected_provider_counts.iterrows():
            prov = prow["usual_facility_id"]

            # Find all beneficiaries who dialyzed at this provider during Sandy week, not just
            # the ones selected into the main 11 displayed rows
            prov_benes = (
                op_sw.loc[op_sw["facility_id"] == prov, "BENE_ID"]
                .drop_duplicates()
                .astype(str)
                .tolist()
            )

            if len(prov_benes) == 0:
                continue

            sub_day = day_df[day_df["BENE_ID"].isin(prov_benes)].copy()
            if sub_day.empty:
                continue

            sub_sel = (
                sub_day[["BENE_ID", "usual_facility_id"]]
                .drop_duplicates()
                .sort_values(["usual_facility_id", "BENE_ID"])
                .reset_index(drop=True)
            )

            matrix, _ = build_panel_matrix(sub_sel, day_df)

            # Make the figure taller when more beneficiaries are shown
            fig_height = max(3.0, 1.8 + 0.18 * len(sub_sel))
            fig, ax = plt.subplots(figsize=(8.5, fig_height))

            n_selected_main = int(prow["n_selected_in_main_panel"])
            provider_note = (
                f"{prov} | all beneficiaries who dialyzed at this provider during Sandy week "
                f"(n={len(sub_sel)}); of these, n={n_selected_main} were used in the main panel"
            )

            draw_single_heatmap_panel(
                ax=ax,
                matrix=matrix,
                panel_title=f"{pobj['panel_label']} | Provider-week context heatmap",
                n_total_group=pobj["group_n"],
                provider_note=provider_note,
                y_label="All beneficiaries seen at this provider during Sandy week"
            )

            plt.tight_layout()

            out_file = os.path.join(
                OUT_DIR,
                f"reference_PROVIDERWEEK_{panel_label_safe}_provider_{prov}.png"
            )
            fig.savefig(out_file, dpi=220, bbox_inches="tight")
            if SHOW_PLOTS:
                plt.show()
            plt.close(fig)


def run_final_heatmap_pipeline():
    # Summary - Main driver for Part 1.
    # For each of the 5 final schedule groups:
    #   1) verify the chosen group exists in the grouped CSV,
    #   2) pull its member beneficiaries,
    #   3) select 11 representative beneficiaries,
    #   4) build supporting tables/metadata,
    #   5) create the collapsed final figure and optional provider context plots.
    groups, members, day, op_rows = load_inputs()

    panel_objects = []    # store all panel-level objects needed for later plotting
    selection_rows = []   # store the final 11 selected beneficiaries per panel
    footnote_rows = []    # store provider summaries used in notes / map selection

    for spec in SELECTED:
        gk = spec["group_key"]
        cat = spec["category"]
        sig = spec["dialysis_signature"]

        # Pull the exact chosen group row and confirm it exists
        g_row = groups[
            (groups["group_key"] == gk) &
            (groups["category"] == cat) &
            (groups["dialysis_signature"] == sig)
        ].copy()

        if g_row.empty:
            raise ValueError(f"Selected group not found in grouped CSV: {spec}")

        g_row = g_row.iloc[0]
        group_n = int(float(g_row["n_benes"]))

        # Pull the member beneficiaries belonging to this chosen group
        gm = members[
            (members["group_key"] == gk) &
            (members["category"] == cat) &
            (members["dialysis_signature"] == sig)
        ].copy()

        # QC check - each chosen panel should have at least 11 unique beneficiaries available
        if gm["BENE_ID"].nunique() < N_ROWS_PER_PANEL:
            raise ValueError(
                f"Group {gk} has fewer than {N_ROWS_PER_PANEL} unique beneficiaries."
            )

        preferred_provider = PREFERRED_PROVIDER_BY_PANEL.get(spec["panel_order"], None)

        # Select the final displayed beneficiaries and summarize their providers
        selected_benes, provider_summary = select_top_provider_heavy(
            gm,
            n_select=N_ROWS_PER_PANEL,
            preferred_provider=preferred_provider
        )
        provider_note = provider_footnote_string(provider_summary)

        # Build row numbers / display ordering for the selected beneficiaries
        _, row_meta = build_panel_matrix(selected_benes, day)

        # Store all panel-level pieces together for later figure creation
        panel_objects.append({
            "panel_order": spec["panel_order"],
            "panel_label": spec["panel_label"],
            "category": cat,
            "group_key": gk,
            "group_n": group_n,
            "provider_note": provider_note,
            "selected_benes": selected_benes.copy(),
            "provider_summary": provider_summary.copy(),
            "row_meta": row_meta.copy(),
        })

        # Save beneficiary-level selection file for transparency / auditing
        tmp = selected_benes.merge(
            row_meta[["BENE_ID", "usual_facility_id", "row_num"]],
            on=["BENE_ID", "usual_facility_id"],
            how="left"
        ).copy()
        tmp["panel_order"] = spec["panel_order"]
        tmp["panel_label"] = spec["panel_label"]
        tmp["group_n"] = group_n
        selection_rows.append(tmp)

        # Save provider summary file. This is the file later used in Part 2 to identify
        # the final top provider for each of the 5 panels.
        ps = provider_summary.copy()
        ps["panel_order"] = spec["panel_order"]
        ps["panel_label"] = spec["panel_label"]
        ps["group_key"] = gk
        ps["category"] = cat
        ps["dialysis_signature"] = sig
        ps["group_n"] = group_n
        footnote_rows.append(ps)

    panel_objects = sorted(panel_objects, key=lambda x: x["panel_order"])

    selection_df = pd.concat(selection_rows, ignore_index=True)
    footnote_df = pd.concat(footnote_rows, ignore_index=True)

    selection_df.to_csv(os.path.join(OUT_DIR, "selected_11_beneficiaries_per_panel.csv"), index=False)
    footnote_df.to_csv(os.path.join(OUT_DIR, "selected_provider_footnotes.csv"), index=False)

    # Build the main collapsed article-safe figure
    collapsed_png = os.path.join(OUT_DIR, "final_sandy_5panel_collapsed_schedule.png")
    collapsed_pdf = os.path.join(OUT_DIR, "final_sandy_5panel_collapsed_schedule.pdf")
    make_collapsed_article_safe_figure(
        panel_objects=panel_objects,
        day_df=day,
        out_png=collapsed_png,
        out_pdf=collapsed_pdf
    )

    # Optionally create the extra provider-week context heatmaps
    if MAKE_PROVIDER_REFERENCE_PLOTS:
        make_provider_reference_plots(panel_objects, day, op_rows)

    # Save a plain-text summary of the final panel selections
    summary_lines = []
    for pobj in panel_objects:
        summary_lines.append(
            f"{pobj['panel_label']} | group_key={pobj['group_key']} | "
            f"group_n={pobj['group_n']} | providers={pobj['provider_note']}"
        )
    with open(os.path.join(OUT_DIR, "panel_selection_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    print("\nDone with final heatmap pipeline.")
    print(f"Collapsed PNG: {collapsed_png}")
    print(f"Collapsed PDF: {collapsed_pdf}")
    print(f"Selection CSV: {os.path.join(OUT_DIR, 'selected_11_beneficiaries_per_panel.csv')}")
    print(f"Footnote CSV: {os.path.join(OUT_DIR, 'selected_provider_footnotes.csv')}")


# =========================================================
# PART 2: CLEAN MAP OF THE 5 FINAL PROVIDERS
# =========================================================
# Summary - This second part takes the provider summary from Part 1, chooses the top provider
# from each of the 5 final panels, attaches ZIPs, maps them to ZCTA representative points,
# and then creates the clean provider map.

FINAL_DIR = OUT_DIR
FOOTNOTE_CSV = os.path.join(FINAL_DIR, "selected_provider_footnotes.csv")

OPB_PATH = "/gpfs/data/cms-share/data/medicare/2012/otpt/opb/parquet/"
OPREV_PATH = "/gpfs/data/cms-share/duas/54200/Jessy/data/climate_change/dialysis/2012/"

ZCTA_SHP = (
    "/gpfs/data/cms-share/duas/52484/Jessy/data/public_data/data/shp_files/"
    "cb_2013_us_zcta_zip_500k/"
)

STATES_SHP = (
    "/gpfs/data/cms-share/duas/52484/Jessy/data/public_data/data/shp_files/"
    "cb_2018_us_state_500k/"
)

OUT_CSV = os.path.join(FINAL_DIR, "final_5_panel_providers_with_zip_v02.csv")
OUT_PNG_ALL = os.path.join(FINAL_DIR, "map_final_5_providers_clean_v02.png")


def extract_zip5(x):
    # Pull the first 5-digit ZIP out of a ZIP-like field
    if pd.isna(x):
        return np.nan
    m = re.search(r"(\d{5})", str(x))
    return m.group(1) if m else np.nan


def add_place_labels(ax):
    """
    Manual location labels for readability on the NYC/NJ map.
    Coordinates are approximate lon/lat placements.
    """
    place_labels = [
        ("Manhattan", -73.98, 40.78),
        ("Bronx", -73.87, 40.87),
        ("Queens", -73.82, 40.73),
        ("Brooklyn", -73.95, 40.64),
        ("Staten Island", -74.14, 40.60),
        ("Newark city", -74.18, 40.74),
    ]

    for name, x, y in place_labels:
        ax.text(
            x, y, name,
            fontsize=9,
            ha="center",
            va="center",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="#666666",
                boxstyle="round,pad=0.20",
                alpha=0.96
            ),
            zorder=6
        )


def run_final_provider_map():
    # Summary - Read the provider summary from Part 1, keep the top provider from each panel,
    # merge in the most common ZIP observed for that provider, attach ZCTA geometry, and create
    # the clean provider map.
    foot = pd.read_csv(FOOTNOTE_CSV, dtype=str)
    foot["usual_facility_id"] = foot["usual_facility_id"].astype(str)
    foot["n_selected"] = pd.to_numeric(foot["n_selected"], errors="coerce").fillna(0).astype(int)
    foot["panel_order"] = pd.to_numeric(foot["panel_order"], errors="coerce").fillna(999).astype(int)

    # Keep one top provider per panel. Because there are 5 panels, this produces the final 5 providers.
    final5 = (
        foot.sort_values(
            ["panel_order", "n_selected", "usual_facility_id"],
            ascending=[True, False, True]
        )
        .drop_duplicates(subset=["panel_order"], keep="first")
        .sort_values("panel_order")
        .reset_index(drop=True)
    )

    providers_keep = final5["usual_facility_id"].drop_duplicates().tolist()

    print(f"Final providers kept: {len(providers_keep)}")
    print(providers_keep)

    print("\nTop provider per panel before ZIP merge:")
    print(
        final5[["panel_order", "panel_label", "usual_facility_id", "n_selected"]]
        .to_string(index=False)
    )

    # Read OPREV just for dates so we can restrict OPB facility ZIP rows to the relevant study window
    op_rev_dates = dd.read_parquet(
        OPREV_PATH,
        engine="pyarrow",
        columns=["CLM_ID", "REV_CNTR_DT"]
    )
    op_rev_dates["REV_CNTR_DT"] = dd.to_datetime(op_rev_dates["REV_CNTR_DT"], errors="coerce")
    op_rev_dates = op_rev_dates[
        (op_rev_dates["REV_CNTR_DT"] >= "2012-10-14") &
        (op_rev_dates["REV_CNTR_DT"] <= "2012-11-03")
    ]

    # Read OPB claims headers for the selected providers and merge with the date-restricted OP rows
    opb = dd.read_parquet(
        OPB_PATH,
        engine="pyarrow",
        columns=["CLM_ID", "PRVDR_NUM", "CLM_SRVC_FAC_ZIP_CD"]
    )
    opb["PRVDR_NUM"] = opb["PRVDR_NUM"].astype(str)
    opb = opb[opb["PRVDR_NUM"].isin(providers_keep)]

    opb_use = opb.merge(op_rev_dates, on="CLM_ID", how="inner").compute()
    opb_use["ZIP5"] = opb_use["CLM_SRVC_FAC_ZIP_CD"].apply(extract_zip5)

    # Assign each provider its modal ZIP across the relevant OP rows
    prov_zip = (
        opb_use.dropna(subset=["ZIP5"])
        .groupby("PRVDR_NUM")["ZIP5"]
        .agg(lambda s: s.value_counts().idxmax() if not s.empty else np.nan)
        .reset_index()
        .rename(columns={"PRVDR_NUM": "usual_facility_id"})
    )

    prov_tbl = final5.merge(prov_zip, on="usual_facility_id", how="left").copy()
    prov_tbl = prov_tbl.sort_values(["panel_order"]).reset_index(drop=True)

    print("\nFinal 5 providers with ZIP5:")
    print(
        prov_tbl[["panel_order", "panel_label", "usual_facility_id", "n_selected", "ZIP5"]]
        .to_string(index=False)
    )

    prov_tbl.to_csv(OUT_CSV, index=False)

    # Read ZIP polygon file and attach geometry via ZIP5
    zcta = gpd.read_file(ZCTA_SHP)
    zcta = zcta.to_crs(epsg=4326)
    zcta["ZIP5"] = zcta["ZCTA5CE10"].astype(str).str.zfill(5)

    prov_geo = prov_tbl.merge(
        zcta[["ZIP5", "geometry"]],
        on="ZIP5",
        how="left"
    ).copy()

    missing_zip = prov_geo["geometry"].isna().sum()
    if missing_zip > 0:
        print(f"\nWarning: {missing_zip} provider rows did not match a ZCTA geometry.")

    prov_geo = prov_geo.loc[prov_geo["geometry"].notna()].copy()
    prov_gdf = gpd.GeoDataFrame(prov_geo, geometry="geometry", crs=zcta.crs)

    # Convert ZIP polygons to representative points so the final map shows one point per provider
    prov_gdf["geometry"] = prov_gdf["geometry"].representative_point()

    if prov_gdf.empty:
        raise ValueError("No provider points available after ZIP/ZCTA merge.")

    prov_gdf["lon"] = prov_gdf.geometry.x
    prov_gdf["lat"] = prov_gdf.geometry.y

    xmin, ymin, xmax, ymax = prov_gdf.total_bounds

    # Build a tighter map extent around the provider points so the final image stays focused
    pad_x = max((xmax - xmin) * 0.12, 0.12)
    pad_y = max((ymax - ymin) * 0.12, 0.10)

    x0, x1 = xmin - pad_x, xmax + pad_x
    y0, y1 = ymin - pad_y, ymax + pad_y

    # Clip the ZIP/state layers to just the local map extent for a cleaner display
    zcta_clip = zcta.cx[x0:x1, y0:y1].copy()
    states = gpd.read_file(STATES_SHP).to_crs(epsg=4326)
    states_clip = states.cx[x0:x1, y0:y1].copy()

    prov_gdf["color"] = prov_gdf["panel_label"].map(PANEL_PALETTE).fillna("#777777")

    fig, ax = plt.subplots(figsize=(9, 7))

    # Light water background just improves readability / aesthetics on the final map
    water_rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        facecolor="#d9eef7",
        edgecolor="none",
        alpha=0.80,
        zorder=0
    )
    ax.add_patch(water_rect)

    # Base ZIP polygons + state boundaries
    zcta_clip.plot(
        ax=ax,
        color="#f3f3f3",
        edgecolor="#d0d0d0",
        linewidth=0.35,
        zorder=1
    )
    states_clip.boundary.plot(
        ax=ax,
        color="#555555",
        linewidth=0.8,
        zorder=2
    )

    # Provider points sized by how many selected beneficiaries from that panel/provider were used
    sizes = 135 + prov_gdf["n_selected"].clip(lower=1).astype(float) * 32

    prov_gdf.plot(
        ax=ax,
        color=prov_gdf["color"],
        markersize=sizes,
        alpha=0.95,
        edgecolor="black",
        linewidth=0.6,
        zorder=4
    )

    # Add city/borough labels for orientation
    add_place_labels(ax)

    # Final map formatting
    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)   # increase bottom white space
    plt.savefig(OUT_PNG_ALL, dpi=240)
    plt.show()
    plt.close(fig)

    print(f"\nSaved final 5-provider ZIP table to:\n{OUT_CSV}")
    print(f"Saved clean provider map to:\n{OUT_PNG_ALL}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    run_final_heatmap_pipeline()
    run_final_provider_map()
