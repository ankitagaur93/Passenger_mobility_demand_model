# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 22:43:22 2025

@author: agaur
"""

import pandas as pd
import plotnine as p9
from genno import Quantity, computations

from growth_development import pop, gdp_cap


total_pdt = pd.read_csv("total_pdt_Base_1_1.csv")

total_pdt = total_pdt.groupby(["n", "y"])["value"].sum().reset_index()
total_pdt = Quantity(total_pdt.set_index(["n", "y"])["value"])

total_pdt_cap = (total_pdt / pop(2)).reset_index()

total_pdt_cap = total_pdt_cap[total_pdt_cap["y"] <= 2050]

gdp_cap_2 = gdp_cap(2).reset_index()

intensity = pd.merge(total_pdt_cap, gdp_cap_2, on=["n", "y"]).rename(
    columns={"value_x": "PDT_cap", "value_y": "GDP_cap, 2017 US$ at PPP"}
)

intensity["PDT_cap/GDP_cap"] = (
    intensity["PDT_cap"] / intensity["GDP_cap, 2017 US$ at PPP"]
)

intensity = pd.melt(
    intensity,
    id_vars=["n", "y"],
    value_vars=["PDT_cap", "GDP_cap, 2017 US$ at PPP", "PDT_cap/GDP_cap"],
    var_name="parameter",
    value_name="value",
)

intensity = Quantity(intensity.set_index(["n", "y", "parameter"])["value"])
intensity_indexed = computations.index_to(intensity, {"y": 2011}).reset_index()

plot_c = (
    (
        p9.ggplot(
            intensity_indexed, p9.aes(x="y", y="value", color="parameter")
        )
        + p9.geom_line(stat="identity", size=2)
        + p9.facet_wrap("n", scales="free")
        + p9.scale_color_manual(
            values={
                "GDP_cap, 2017 US$ at PPP": "#219ebc",
                "PDT_cap": "#c1121f",
                "PDT_cap/GDP_cap": "#e9c46a",
            },
            name="",
        )
    )
    + p9.theme_light()
    + p9.labs(y="Values indexed to 2011", x="")
    + p9.theme(
        text=p9.element_text(size=16),
        legend_title=p9.element_blank(),
        axis_title=p9.element_text(face="bold"),
        strip_text=p9.element_text(face="bold", color="black"),
    )
    + p9.theme(
        legend_position=(1, 0.1),
        legend_background=p9.element_blank(),  # remove legend box
        legend_key=p9.element_blank(),
    )
)

plot_c.save("pdt_gdp_inten.pdf", width=14, height=10)
