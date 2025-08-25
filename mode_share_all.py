# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 17:02:58 2025

@author: agaur
"""
import pandas as pd
import plotnine as p9
from genno import Quantity, computations
import numpy as np

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
dat_all = dat_all[dat_all["y"] == 2050]

dat_all = (
    dat_all.groupby(["scen", "mode", "type", "urb", "mode_s"])["value"]
    .sum()
    .reset_index()
)

dat_all["share"] = (
    dat_all["value"]
    * 100
    / dat_all.groupby(["scen", "type", "urb", "mode_s"])["value"].transform(
        "sum"
    )
)

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

short_dist = dat_all[(dat_all["type"] == "Short") & (dat_all["urb"] == 1)]


plot_s = (
    p9.ggplot(short_dist, p9.aes(x="mode_s", y="share", fill="mode"))
    + p9.geom_bar(stat="identity", position="stack", width=0.4)
    + p9.facet_wrap("scen", nrow=1)
    + p9.scale_fill_manual(
        values={
            "2-Wheeler": "#76C893",
            "Rail": "#E07A5F",
            "NMT": "#FFB703",
            "LDV": "#778DA9",
            "IPT": "#023047",
            "Bus": "#219EBC",
        },
        name="",
    )
    + p9.labs(y=" Mode share", x="")
    + p9.theme_light(base_size=24)
    + p9.theme(
        text=p9.element_text(size=24),
        legend_title=p9.element_blank(),
        axis_title=p9.element_text(face="bold", size=24),
        strip_text=p9.element_text(face="bold", color="black", size=24),
    )
    + p9.theme(legend_position="right", panel_spacing_x=0.025)
)

# plot_s.save("mode_share_scens_short.pdf", width=20, height=8)

med_dist = dat_all[(dat_all["type"] == "Medium") & (dat_all["urb"] == 1)]


plot_s = (
    p9.ggplot(med_dist, p9.aes(x="mode_s", y="share", fill="mode"))
    + p9.geom_bar(stat="identity", position="stack", width=0.4)
    + p9.facet_wrap("scen", nrow=1)
    + p9.scale_fill_manual(
        values={
            "2-Wheeler": "#76C893",
            "Rail": "#E07A5F",
            "NMT": "#FFB703",
            "LDV": "#778DA9",
            "IPT": "#023047",
            "Bus": "#219EBC",
        },
        name="",
    )
    + p9.labs(y=" Mode share", x="")
    + p9.theme_light(base_size=24)
    + p9.theme(
        text=p9.element_text(size=24),
        legend_title=p9.element_blank(),
        axis_title=p9.element_text(face="bold", size=24),
        strip_text=p9.element_text(face="bold", color="black", size=24),
    )
    + p9.theme(legend_position="right", panel_spacing_x=0.025)
)

# plot_s.save("mode_share_scens_med.pdf", width=20, height=8)

long_dist = dat_all[(dat_all["type"] == "Long") & (dat_all["urb"] == 1)]


plot_s = (
    p9.ggplot(long_dist, p9.aes(x="mode_s", y="share", fill="mode"))
    + p9.geom_bar(stat="identity", position="stack", width=0.4)
    + p9.facet_wrap("scen", nrow=1)
    + p9.scale_fill_manual(
        values={
            "2-Wheeler": "#76C893",
            "Rail": "#E07A5F",
            "NMT": "#FFB703",
            "LDV": "#778DA9",
            "IPT": "#023047",
            "Bus": "#219EBC",
        },
        name="",
    )
    + p9.labs(y=" Mode share", x="")
    + p9.theme_light(base_size=24)
    + p9.theme(
        text=p9.element_text(size=24),
        legend_title=p9.element_blank(),
        axis_title=p9.element_text(face="bold", size=24),
        strip_text=p9.element_text(face="bold", color="black", size=24),
    )
    + p9.theme(legend_position="right", panel_spacing_x=0.025)
)

# plot_s.save("mode_share_scens_long.pdf", width=20, height=8)
