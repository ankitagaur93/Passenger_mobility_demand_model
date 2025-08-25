# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:15:53 2025

@author: agaur
"""

import pandas as pd
import plotnine as p9
from genno import Quantity, computations
import numpy as np
import patchworklib as pw  # pip install patchworklib
from pathlib import Path
import matplotlib.pyplot as plt


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

dat_2050 = dat_all[(dat_all["y"] == 2050) & (dat_all["n"] == "India")]


p2050 = (
    p9.ggplot(dat_2050, p9.aes(x="mode_s", y="share", fill="mode"))
    + p9.geom_bar(stat="identity", position="stack", width=0.4)
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
    + p9.theme_bw(base_size=18)
    + p9.labs(y="", x="", fill="")
    + p9.theme(
        axis_text_x=p9.element_text(rotation=0),
        axis_text_y=p9.element_blank(),
        legend_position="right",  # legend only on first plot
        figure_size=(10, 4),
    )
)


# Draw plotnine figure
fig = p2050.draw()  # this is already a matplotlib Figure

# Add a grey rectangle across the entire x-axis
fig.patches.append(
    plt.Rectangle(
        (0, -0.05),  # y slightly below the plot area
        0.85,  # width: fraction of figure
        0.05,  # height: fraction of figure
        transform=fig.transFigure,
        color="lightgrey",
        zorder=1,
    )
)

# Add centered text on top of rectangle
fig.text(
    0.5,
    -0.025,
    "2050",  # adjust y for vertical position
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    zorder=1,
)

dat_2011 = dat_all[
    (dat_all["y"] == 2011)
    & (dat_all["n"] == "India")
    & (dat_all["scen"] == "Base")
    & (dat_all["mode_s"] == "BaU")
]


p2011 = (
    p9.ggplot(dat_2011, p9.aes(x="mode_s", y="share", fill="mode"))
    + p9.geom_bar(stat="identity", position="stack", width=0.2)
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
    + p9.theme_bw(base_size=18)
    + p9.labs(y="Share of modes in daily travel PDT", x="", fill="")
    + p9.theme(
        axis_text_x=p9.element_text(color="white"),
        legend_position="none",
        figure_size=(2, 4),
    )
)

fig_2 = p2011.draw()  # this is already a matplotlib Figure

# Add a grey rectangle across the entire x-axis
fig_2.patches.append(
    plt.Rectangle(
        (0.2, -0.055),  # y slightly below the plot area
        0.8,  # width: fraction of figure
        0.05,  # height: fraction of figure
        transform=fig_2.transFigure,
        color="lightgrey",
        zorder=1,
    )
)

# Add centered text on top of rectangle
fig_2.text(
    0.6,
    -0.035,
    "2011",  # adjust y for vertical position
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    zorder=1,
)

countries = dat_all["n"].unique()

for country in countries:
    # Filter data for 2050
    dat_2050 = dat_all[(dat_all["y"] == 2050) & (dat_all["n"] == country)]
    p2050 = (
        p9.ggplot(dat_2050, p9.aes(x="mode_s", y="share", fill="mode"))
        + p9.geom_bar(stat="identity", position="stack", width=0.4)
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
        + p9.theme_bw(base_size=18)
        + p9.labs(y="", x="", fill="")
        + p9.theme(
            axis_text_x=p9.element_text(rotation=0),
            axis_text_y=p9.element_blank(),
            legend_position="right",
            figure_size=(10, 4),
        )
    )
    fig_2050 = p2050.draw()
    # Grey rectangle + text for 2050
    fig_2050.patches.append(
        plt.Rectangle(
            (0, -0.05),
            0.85,
            0.05,
            transform=fig_2050.transFigure,
            color="lightgrey",
            zorder=1,
        )
    )
    fig_2050.text(
        0.5,
        -0.025,
        "2050",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        zorder=2,
    )

    # Filter data for 2011
    dat_2011 = dat_all[
        (dat_all["y"] == 2011)
        & (dat_all["n"] == country)
        & (dat_all["scen"] == "Base")
        & (dat_all["mode_s"] == "BaU")
    ]
    p2011 = (
        p9.ggplot(dat_2011, p9.aes(x="mode_s", y="share", fill="mode"))
        + p9.geom_bar(stat="identity", position="stack", width=0.2)
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
        + p9.theme_bw(base_size=18)
        + p9.labs(y="Share of modes in daily travel PDT", x="", fill="")
        + p9.theme(
            axis_text_x=p9.element_text(color="white"),
            legend_position="none",
            figure_size=(2, 4),
        )
    )
    fig_2011 = p2011.draw()
    # Grey rectangle + text for 2011
    fig_2011.patches.append(
        plt.Rectangle(
            (0.2, -0.055),
            0.8,
            0.05,
            transform=fig_2011.transFigure,
            color="lightgrey",
            zorder=1,
        )
    )
    fig_2011.text(
        0.6,
        -0.035,
        "2011",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        zorder=2,
    )

    # Save figures
    fig_2011.savefig(f"fig_2011_{country}.png", bbox_inches="tight", dpi=300)
    fig_2050.savefig(f"fig_2050_{country}.png", bbox_inches="tight", dpi=300)

    plt.close(fig_2011)
    plt.close(fig_2050)


# other figure with just 1 scenario
import pandas as pd
import plotnine as p9
from genno import Quantity, computations
import numpy as np
import patchworklib as pw  # pip install patchworklib
from pathlib import Path
import matplotlib.pyplot as plt


folder = Path(r"C:\Users\agaur\Passenger_mobility_demand_model")
scenarios = ["Base", "NP", "BP", "TECH", "TOD"]
i_range = range(1, 3)


dat_all = pd.DataFrame()

for scen in scenarios:
    for i in i_range:

        file_path = folder / f"total_pdt_{scen}_{i}_1.csv"
        # Read the CSV
        dat = pd.read_csv(file_path)
        dat["scen"] = scen
        dat["urb"] = i

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


# Replace values in the column

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
    dat_all.groupby(["n", "scen", "mode", "y"])["value"].sum().reset_index()
)

dat_all["share"] = (
    dat_all["value"]
    * 100
    / dat_all.groupby(["n", "scen", "y"])["value"].transform("sum")
)


# Ensure categorical ordering
scen_order = ["Base", "BP", "NP", "TOD", "TECH"]

dat_all["scen"] = pd.Categorical(
    dat_all["scen"], categories=scen_order, ordered=True
)
countries = dat_all["n"].unique()

for country in countries:
    # Filter data for 2050
    dat_2050 = dat_all[(dat_all["y"] == 2050) & (dat_all["n"] == country)]
    p2050 = (
        p9.ggplot(dat_2050, p9.aes(x="scen", y="share", fill="mode"))
        + p9.geom_bar(stat="identity", position="stack", width=0.4)
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
        + p9.theme_bw(base_size=20)
        + p9.labs(y="", x="", fill="")
        + p9.theme(
            axis_text_x=p9.element_text(rotation=0, face="bold"),
            axis_text_y=p9.element_blank(),
            legend_position="right",
            figure_size=(5, 4),
        )
    )
    fig_2050 = p2050.draw()
    # Grey rectangle + text for 2050
    fig_2050.patches.append(
        plt.Rectangle(
            (0, -0.05),
            0.75,
            0.05,
            transform=fig_2050.transFigure,
            color="lightgrey",
            zorder=1,
        )
    )
    fig_2050.text(
        0.4,
        -0.025,
        "2050",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        zorder=2,
    )

    # Filter data for 2011
    dat_2011 = dat_all[
        (dat_all["y"] == 2011)
        & (dat_all["n"] == country)
        & (dat_all["scen"] == "Base")
    ]
    p2011 = (
        p9.ggplot(dat_2011, p9.aes(x="scen", y="share", fill="mode"))
        + p9.geom_bar(stat="identity", position="stack", width=0.2)
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
        + p9.theme_bw(base_size=20)
        + p9.labs(y="Share of modes in total PDT", x="", fill="")
        + p9.theme(
            axis_text_x=p9.element_text(color="white"),
            legend_position="none",
            figure_size=(2, 4),
        )
    )
    fig_2011 = p2011.draw()
    # Grey rectangle + text for 2011
    fig_2011.patches.append(
        plt.Rectangle(
            (0.3, -0.055),
            0.75,
            0.05,
            transform=fig_2011.transFigure,
            color="lightgrey",
            zorder=1,
        )
    )
    fig_2011.text(
        0.65,
        -0.035,
        "2011",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        zorder=2,
    )

    # Save figures
    fig_2011.savefig(
        f"fig_2011_{country}_ms.png", bbox_inches="tight", dpi=300
    )
    fig_2050.savefig(
        f"fig_2050_{country}_ms.png", bbox_inches="tight", dpi=300
    )

    plt.close(fig_2011)
    plt.close(fig_2050)
