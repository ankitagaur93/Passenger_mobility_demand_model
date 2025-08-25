# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 23:37:16 2025

@author: agaur
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 17:02:58 2025

@author: agaur
"""
import pandas as pd
import plotnine as p9
from genno import Quantity, computations
import numpy as np
import patchworklib as pw  # pip install patchworklib
from pathlib import Path

folder = Path(r"C:\Users\agaur\Passenger_mobility_demand_model")
scenarios = ["Base", "NP", "BP", "TECH", "TOD"]
i_range = range(1, 3)
j_range = range(1, 4)

dat_all = pd.DataFrame()

for scen in scenarios:
    for i in i_range:
        for j in j_range:
            file_path = folder / f"total_pdt_{scen}_{i}_{j}.csv"
            # Read the CSV
            dat = pd.read_csv(file_path)
            dat["scen"] = scen
            dat["urb"] = i
            dat["mode_s"] = j

            dat_all = pd.concat([dat_all, dat])

conditions = [
    dat_all["trip_dist"].isin(["00_01", "02_05", "06_10"]),
    dat_all["trip_dist"].isin(["11_20", "21_30"]),
    dat_all["trip_dist"].isin(["31_50", "51+"]),
]
choices = ["Short", "Medium", "Long"]
dat_all["type"] = np.select(conditions, choices, default="Other")
dat_all = dat_all[
    (dat_all["y"].isin([2011, 2050]))
    & (dat_all["urb"] == 1)
    & (dat_all["type"] != "Other")
]

dat_all["scen"] = pd.Categorical(
    dat_all["scen"],
    categories=["Base", "NP", "BP", "TOD", "TECH"],
    ordered=True,
)

mapping = {1: "BaU", 2: "COF", 3: "SF"}

# Replace values in the column
dat_all["mode_s"] = dat_all["mode_s"].map(mapping)  #

# Map original values to display names
mode_mapping = {
    "tw_share": "2-Wheeler",
    "rail_share": "Rail",
    "nmt_share": "NMT",
    "ldv_share": "LDV",
    "ipt_share": "IPT",
    "bus_share": "Bus",
}
dat_all["mode"] = dat_all["mode"].replace(mode_mapping)

# Assign ordered categorical
dat_all["mode"] = pd.Categorical(
    dat_all["mode"],
    categories=["2-Wheeler", "Rail", "NMT", "LDV", "IPT", "Bus"],
    ordered=True,
)

dat_all = (
    dat_all.groupby(["n", "scen", "mode", "mode_s", "y"])["value"]
    .sum()
    .reset_index()
)

dat_all["share"] = (
    dat_all["value"]
    * 100
    / dat_all.groupby(["n", "scen", "y", "mode_s"])["value"].transform("sum")
)


# Ensure categorical ordering
scen_order = ["Base", "BP", "NP", "TOD", "TECH"]
mode_s_order = ["BaU", "COF", "SF"]

dat_all["scen"] = pd.Categorical(
    dat_all["scen"], categories=scen_order, ordered=True
)
dat_all["mode_s"] = pd.Categorical(
    dat_all["mode_s"], categories=mode_s_order, ordered=True
)

# ---- 2011 plot ----
dat_2011 = dat_all[
    (dat_all["y"] == 2011)
    & (dat_all["scen"] == "Base")
    & (dat_all["mode_s"] == "BaU")
    & (dat_all["n"] == "India")
]
p2011 = (
    p9.ggplot(dat_2011, p9.aes(x="mode_s", y="share", fill="mode"))
    + p9.geom_bar(stat="identity", position="stack", width=0.7)
    + p9.scale_fill_manual(
        values={
            "2-Wheeler": "#76C893",
            "Rail": "#E07A5F",
            "NMT": "#FFB703",
            "LDV": "#778DA9",
            "IPT": "#023047",
            "Bus": "#219EBC",
        }
    )
    + p9.theme_light()
    + p9.labs(y="Share of modes in daily travel PDT", x="", fill="")
    + p9.theme(
        axis_text_x=p9.element_blank(),
        legend_position="right",
        figure_size=(10, 4),
    )
)

# ---- 2050 plot ----
dat_2050 = dat_all[(dat_all["y"] == 2050) & (dat_all["n"] == "India")]
p2050 = (
    p9.ggplot(dat_2050, p9.aes(x="mode_s", y="share", fill="mode"))
    + p9.geom_bar(stat="identity", position="stack", width=0.7)
    + p9.facet_wrap("~scen", scales="free_x", nrow=1)
    + p9.scale_fill_manual(
        values={
            "2-Wheeler": "#76C893",
            "Rail": "#E07A5F",
            "NMT": "#FFB703",
            "LDV": "#778DA9",
            "IPT": "#023047",
            "Bus": "#219EBC",
        }
    )
    + p9.theme_bw()
    + p9.labs(y="", x="", fill="")
    + p9.theme(
        axis_text_x=p9.element_text(rotation=0),
        axis_text_y=p9.element_blank(),
        legend_position="none",  # legend only on first plot
        figure_size=(10, 4),
    )  # Grey rectangle across full x range (just below axis)
    + p9.annotate(
        "rect",
        xmin=-0.5,
        xmax=len(dat_2050["scen"].cat.categories)
        * len(dat_2050["mode_s"].cat.categories)
        - 0.5,
        ymin=-12,
        ymax=-2,  # adjust based on axis range
        fill="lightgrey",
        alpha=0.8,
    )
    # Text "2050" centered below plot
    + p9.annotate(
        "text",
        x=len(dat_2050["scen"].cat.categories)
        * len(dat_2050["mode_s"].cat.categories)
        / 2,
        y=-7,
        label="2050",
        size=12,
        fontweight="bold",
    )
)

import matplotlib.pyplot as plt


fig = p2050.draw()

fig.text(
    0.5,
    -0.05,
    "2050",  # center horizontally, slightly below plot
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.3", facecolor="lightgrey", edgecolor="none"
    ),
)
plt.tight_layout()
plt.show()

# ---- Combine side by side ----
# fig = pw.load_ggplot(p2011, figsize=(10, 4)) | pw.load_ggplot(
#     p2050, figsize=(10, 4)
# )
# fig.savefig("afghanistan_mode_share.png", dpi=300)

import matplotlib.pyplot as plt
import seaborn as sns

# Create combined x = (scenario, mode_s)
# Filter 2050 subset
dat_2050 = dat_2050

# Ensure order
scen_order = ["Base", "BP", "NP", "TOD", "TECH"]
mode_s_order = ["BaU", "COF", "SF"]

dat_2050["scen"] = pd.Categorical(
    dat_2050["scen"], categories=scen_order, ordered=True
)
dat_2050["mode_s"] = pd.Categorical(
    dat_2050["mode_s"], categories=mode_s_order, ordered=True
)

# Colors
palette = {
    "2-Wheeler": "#76C893",
    "Rail": "#E07A5F",
    "NMT": "#FFB703",
    "LDV": "#778DA9",
    "IPT": "#023047",
    "Bus": "#219EBC",
}

# Create FacetGrid by scenario
g = sns.FacetGrid(
    dat_2050,
    col="scen",
    col_order=scen_order,
    sharey=True,
    sharex=False,
    height=4,
    aspect=0.7,
)


def stacked_bar(data, color=None, **kwargs):
    ax = kwargs.get("ax", plt.gca())

    # Group and pivot
    df = (
        data.groupby(["mode_s", "mode"])["share"]
        .sum()
        .unstack()
        .reindex(index=mode_s_order, fill_value=0)
    )

    # Keep only palette-defined modes that are present in the data
    present_modes = [m for m in palette if m in df.columns]
    df = df.reindex(columns=present_modes, fill_value=0)

    # Plot
    bottom = np.zeros(len(df))
    for mode in present_modes:
        ax.bar(
            df.index, df[mode], bottom=bottom, color=palette[mode], label=mode
        )
        bottom += df[mode].values


g = sns.FacetGrid(
    dat_2050,
    col="scen",
    col_order=scen_order,
    sharey=True,
    sharex=False,
    height=4,
    aspect=0.7,
)
g.map_dataframe(stacked_bar)

# Clean up axes
for ax in g.axes.flatten():
    ax.set_xlabel("")
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, dat_2050["share"].max() * 1.1)

# Add legend
handles, labels = g.axes[0][0].get_legend_handles_labels()
g.fig.legend(handles, labels, loc="center right", title="Mode")

# Add grey strip for year
g.fig.subplots_adjust(bottom=0.2, right=0.85)
g.fig.text(
    0.5,
    -0.05,
    "2050",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.3", facecolor="lightgrey", edgecolor="none"
    ),
)

plt.show()
