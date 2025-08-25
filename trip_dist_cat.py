# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 19:14:19 2025

@author: agaur
"""

import pandas as pd
import plotnine as p9
from genno import Quantity, computations
import numpy as np

total_pdt_mode_1 = pd.read_csv("total_pdt_Base_1_1.csv")


conditions = [
    total_pdt_mode_1["trip_dist"].isin(["00_01", "02_05", "06_10"]),
    total_pdt_mode_1["trip_dist"].isin(["11_20", "21_30"]),
    total_pdt_mode_1["trip_dist"].isin(["31_50", "51+"]),
]
choices = ["Short", "Medium", "Long"]
total_pdt_mode_1["type"] = np.select(conditions, choices, default="Other")

total_pdt_mode_1 = (
    total_pdt_mode_1.groupby(["y", "mode", "type"])["value"]
    .sum()
    .reset_index()
)
total_pdt_mode_1 = total_pdt_mode_1[total_pdt_mode_1["y"].isin([2011, 2050])]


total_pdt_mode_2 = pd.read_csv("total_pdt_Base_1_2.csv")


conditions = [
    total_pdt_mode_2["trip_dist"].isin(["00_01", "02_05", "06_10"]),
    total_pdt_mode_2["trip_dist"].isin(["11_20", "21_30"]),
    total_pdt_mode_2["trip_dist"].isin(["31_50", "51+"]),
]
choices = ["Short", "Medium", "Long"]
total_pdt_mode_2["type"] = np.select(conditions, choices, default="Other")

total_pdt_mode_2 = (
    total_pdt_mode_2.groupby(["y", "mode", "type"])["value"]
    .sum()
    .reset_index()
)
total_pdt_mode_2 = total_pdt_mode_2[total_pdt_mode_2["y"].isin([2050])]

total_pdt_mode_3 = pd.read_csv("total_pdt_Base_1_3.csv")


conditions = [
    total_pdt_mode_3["trip_dist"].isin(["00_01", "02_05", "06_10"]),
    total_pdt_mode_3["trip_dist"].isin(["11_20", "21_30"]),
    total_pdt_mode_3["trip_dist"].isin(["31_50", "51+"]),
]
choices = ["Short", "Medium", "Long"]
total_pdt_mode_3["type"] = np.select(conditions, choices, default="Other")

total_pdt_mode_3 = (
    total_pdt_mode_3.groupby(["y", "mode", "type"])["value"]
    .sum()
    .reset_index()
)
total_pdt_mode_3 = total_pdt_mode_3[total_pdt_mode_3["y"].isin([2050])]
