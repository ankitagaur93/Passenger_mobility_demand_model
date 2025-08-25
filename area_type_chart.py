# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 18:47:38 2025

@author: agaur
"""

import pandas as pd
import plotnine as p9
from genno import Quantity, computations
import numpy as np

total_pdt = pd.read_csv("total_pdt_Base_1_1.csv")

total_pdt_area = (
    total_pdt.groupby(["n", "y", "area_type"])["value"].sum().reset_index()
)

total_pdt_area["value"] /= 1000

total_pdt_area = total_pdt_area[total_pdt_area["y"] <= 2050]

total_pdt_area = total_pdt_area.replace(
    {
        "city": "City",
        "large_city": "Large city",
        "town": "Town",
        "rural": "Rural",
    }
)

total_pdt_area["area_type"] = pd.Categorical(
    total_pdt_area["area_type"],
    categories=["Large city", "City", "Town", "Rural"],
    ordered=True,
)

plot_at = (
    p9.ggplot(total_pdt_area, p9.aes(x="y", y="value", fill="area_type"))
    + p9.geom_area(stat="identity")
    + p9.facet_wrap("n", scales="free")
    + p9.scale_fill_manual(
        values={
            "City": "#14213d",
            "Large city": "#fca311",
            "Town": "#fb6f92",
            "Rural": "#2a9d8f",
        },
        name="",
    )
    + p9.labs(y=" Billion Passenger Kilometres(bpkm)", x="Year")
    + p9.theme_light()
    + p9.theme(
        text=p9.element_text(size=16),
        legend_title=p9.element_blank(),
        axis_title=p9.element_text(face="bold"),
        strip_text=p9.element_text(face="bold", color="black"),
    )
    + p9.theme(legend_position=(0.8, 0.1))
)
plot_at.save("pdt_area_type.pdf", width=14, height=10)
