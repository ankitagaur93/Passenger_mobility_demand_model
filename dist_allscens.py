# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 12:18:26 2025

@author: agaur
"""

import pandas as pd
import plotnine as p9
from genno import Quantity, computations

scen = ["Base", "NP", "BP", "TOD", "TECH"]
path = "C:/Users/agaur/Passenger_mobility_demand_model/"

all_data = []

for i in scen:
    filename = f"{path}total_pdt_{i}_1_1.csv"
    dat = pd.read_csv(filename)
    dat = dat[dat["y"].isin([2011, 2050])]
    dat["scen"] = i
    all_data.append(dat)

new_dat = pd.concat(all_data, ignore_index=True)
new_dat = new_dat[
    ~(
        (new_dat["y"] == 2011)
        & (new_dat["scen"].isin(["BP", "NP", "TOD", "TECH"]))
    )
]
new_dat = (
    new_dat.groupby(["y", "area_type", "scen"])["value"].sum().reset_index()
)


plot_scen = (
    p9.ggplot(new_dat, p9.aes(x="scen", y="value", fill="area_type"))
    + p9.geom_bar(stat="identity", position="stack")
    + p9.facet_wrap("y")
)

df=new_dat
# Use empty label for 2011 (only one bar), actual scen name for 2050
df['scen_label'] = df.apply(lambda row: row['scen'] if row['y'] == 2050 else '', axis=1)

# Convert year for faceting
df['year'] = df['y'].astype(str)

# Narrower bar width for 2011
df['bar_width'] = df['y'].apply(lambda y: 0.1 if y == 2011 else 0.4)

# Plot
plot_scen = (
    p9.ggplot(df, p9.aes(x="scen_label", y="value", fill="area_type"))
    + p9.geom_bar(p9.aes(width='bar_width'), stat="identity", position="stack")
    + p9.facet_wrap("~year", scales="free_x")
    + p9.labs(x="", y="Passenger Kilometers (in trillion)", fill="Area Type")
    + p9.scale_fill_manual(values={
        'large_city': '#F4A300',
        'city': '#101A33',
        'town': '#F68BB5',
        'rural': '#39B29D'
    })
    
)

plot_scen
