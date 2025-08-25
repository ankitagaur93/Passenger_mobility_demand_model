# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 15:14:23 2025

@author: agaur
"""

import pandas as pd
import plotnine as p9
from genno import Quantity, computations
from growth_development import pop
from pathlib import Path

folder = Path(r"C:\Users\agaur\Passenger_mobility_demand_model")
i_range = range(1, 3)
j_range = range(1, 4)

dat_tech = pd.DataFrame()


for i in i_range:
    for j in j_range:
        file_path = folder / f"total_pdt_TECH_{i}_{j}.csv"
        file_path_2 = folder / f"total_pdt_TECH_{i}_{j}_elifefalse.csv"
        # Read the CSV
        dat = pd.read_csv(file_path)
        dat["urb"] = i
        dat["mode_s"] = j
        dat["elife"] = "yes"
        dat_2 = pd.read_csv(file_path_2)
        dat_2["urb"] = i
        dat_2["mode_s"] = j
        dat_2["elife"] = "no"

        dat_tech = pd.concat([dat_tech, dat])
        dat_tech = pd.concat([dat_tech, dat_2])
dat_tech = dat_tech[dat_tech["y"] <= 2050]
dat_tech = (
    dat_tech.groupby(["mode", "urb", "mode_s", "elife"])["value"]
    .sum()
    .reset_index()
)

dat_tech = dat_tech.pivot_table(
    index=[
        "mode",
        "urb",
        "mode_s",
    ],
    columns="elife",
    values="value",
).reset_index()

dat_tech["diff"] = (dat_tech["no"] - dat_tech["yes"]) * 100 / dat_tech["yes"]

dat_tech["urb"] = dat_tech["urb"].replace({1: "Balanced", 2: "Concentrated"})
mapping = {1: "BaU", 2: "COF", 3: "SF"}

dat_tech["mode_s"] = dat_tech["mode_s"].map(mapping)  #
mode_mapping = {
    "tw_share": "2-Wheeler",
    "rail_share": "Rail",
    "nmt_share": "NMT",
    "ldv_share": "LDV",
    "ipt_share": "IPT",
    "bus_share": "Bus",
}
dat_tech["mode"] = dat_tech["mode"].replace(mode_mapping)
plot_p = (
    (
        p9.ggplot(dat_tech, p9.aes(x="mode", y="diff", fill="mode_s"))
        + p9.geom_bar(stat="identity", position="dodge")
        + p9.facet_grid("urb")
    )
    + p9.scale_fill_manual(
        values={"BaU": "#e5989b", "COF": "#666a86", "SF": "#b2c9ab"}
    )
    + p9.theme_bw(base_size=18)
    + p9.labs(x="", y=" Percentage change", fill="Mode share traj")
    + p9.theme(
        axis_text_x=p9.element_text(rotation=0),
        legend_position="right",  # legend only on first plot
        figure_size=(10, 8),
    )
)
plot_p.save("elife_compare.pdf")
