# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 16:49:10 2025

@author: agaur
"""
import pandas as pd
import plotnine as p9
from genno import Quantity, computations
import numpy as np
import patchworklib as pw  # pip install patchworklib
from pathlib import Path

folder = Path(r"C:\Users\agaur\Passenger_mobility_demand_model")
modes = ["bus", "ipt", "nmt", "ldv", "rail", "tw"]
m_range = range(1, 4)

mode_all = pd.DataFrame()

for m in modes:
    for i in m_range:
        file_path = folder / f"{m}_{i}_2.csv"
        dat_mode = pd.read_csv(file_path)
        dat_mode["m"] = i
        dat_mode["mode"] = m
        mode_all = pd.concat([mode_all, dat_mode])

dat_afg = mode_all[(mode_all["n"] == "Nepal") & (mode_all["y"] <= 2050)]

mapping = {1: "BaU", 2: "COF", 3: "SF"}

# Replace values in the column
dat_afg["m"] = dat_afg["m"].map(mapping)  #

# Map original values to display names
mode_mapping = {
    "tw": "2-Wheeler",
    "rail": "Rail",
    "nmt": "NMT",
    "ldv": "LDV",
    "ipt": "IPT",
    "bus": "Bus",
}
dat_afg["mode"] = dat_afg["mode"].replace(mode_mapping)

# Assign ordered categorical
dat_afg["mode"] = pd.Categorical(
    dat_afg["mode"],
    categories=["2-Wheeler", "Rail", "NMT", "LDV", "IPT", "Bus"],
    ordered=True,
)

p1 = p9.ggplot(
    dat_afg[dat_afg["m"] == "SF"], p9.aes(x="y", y="value", color="mode")
) + p9.geom_line(stat="identity")
