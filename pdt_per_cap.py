# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:49:46 2025

@author: agaur
"""

import pandas as pd
import plotnine as p9
from genno import Quantity, computations
from growth_development import pop


total_pdt = pd.read_csv("total_pdt_Base_2_1.csv")

total_pdt = total_pdt.groupby(["n", "y"])["value"].sum().reset_index()
total_all_c = total_pdt.groupby(["y"])["value"].sum().reset_index()


total_pdt = Quantity(total_pdt.set_index(["n", "y"])["value"])

total_pdt_cap = (total_pdt / pop(2)).reset_index()

total_pdt_cap = total_pdt_cap[total_pdt_cap["y"] <= 2050]


plot_pdt_cap = (
    p9.ggplot(total_pdt_cap, p9.aes(x="y", y="value", color="n"))
    + p9.geom_line(stat="identity", size=1.5)
    + p9.labs(y=" Passenger Kilometres per capita", x="Year")
    + p9.theme_light()
    + p9.scale_color_manual(
        values={
            "Afghanistan": "#e9c46a",
            "Bangladesh": "#e63946",
            "Bhutan": "#748cab",
            "India": "#00b4d8",
            "Maldives": "#fb8500",
            "Nepal": "#e4c1f9",
            "Pakistan": "#99d98c",
            "Sri Lanka": "#7209b7",
        },
        name="Countries",
    )
    + p9.theme(
        legend_key=p9.element_blank(), axis_title=p9.element_text(face="bold")
    )
)
plot_pdt_cap.save("pdt_cap_base.pdf")

# x = total_pdt_cap[total_pdt_cap["y"].isin([2011, 2025, 2050])]
