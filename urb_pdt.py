# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 23:10:40 2025

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

dat_all = dat_all[
    (dat_all["scen"] == "Base")
    & (dat_all["mode_s"] == 1)
    & (dat_all["area_type"] != "rural")
    & (dat_all["y"] <= 2050)
]


dat_all = (
    dat_all.groupby(["n", "y", "scen", "urb", "area_type"])["value"]
    .sum()
    .reset_index()
)

dat_all["pattern"] = np.where(dat_all["urb"] == 1, "stripe", "solid")

dat_all = dat_all.replace(
    {
        "city": "City",
        "large_city": "Large city",
        "town": "Town",
        "rural": "Rural",
    }
)

dat_all["value"] /= 10**3


plot_urb = (
    p9.ggplot(
        dat_all,
        p9.aes(x="y", y="value", color="area_type", linetype="pattern"),
    )
    + p9.geom_line(stat="identity", size=2)
    + p9.facet_wrap("n", scales="free")
    + p9.scale_linetype_manual(
        values={"solid": "solid", "stripe": "dotted"},
        labels={"solid": "Concentrated", "stripe": "Balanced"},
        name="Urbanization type",
    )
    + p9.scale_color_manual(
        values={"City": "#14213d", "Large city": "#fca311", "Town": "#fb6f92"},
        name="Urban area type",
    )
    + p9.labs(y=" Billion Passenger Kilometres(bpkm)", x="Year")
    + p9.theme_light()
    + p9.theme(
        text=p9.element_text(size=16),
        axis_title=p9.element_text(face="bold"),
        strip_text=p9.element_text(face="bold", color="black"),
    )
    + p9.theme(
        legend_title=p9.element_text(face="bold"),
        legend_position=(1, 0.1),
        legend_background=p9.element_blank(),  # remove legend box
        legend_key=p9.element_blank(),
        legend_box="vertical",
    )
)
plot_urb.save("urbanization.pdf", width=14, height=10)
