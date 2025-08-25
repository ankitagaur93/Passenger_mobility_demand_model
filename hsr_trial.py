# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 17:42:23 2025

@author: agaur
"""

import os
import numpy as np
import pandas as pd
from demand_model import Daily_travel
from genno import Quantity, computations, Key


x = Daily_travel(2, 1, 1)
long_dist_pdt = x[x.index.get_level_values("trip_dist").isin(["31_50", "51+"])]


long_dist_pdt = computations.group_sum(
    group=["y", "n", "trip_dist", "area_type"],
    sum="mode",
    qty=long_dist_pdt,
)

long_dist_rail = x[
    x.index.get_level_values("mode").isin(["rail_share"])
    & (x.index.get_level_values("y") == 2011)
    & x.index.get_level_values("trip_dist").isin(["31_50", "51+"])
]

# long_dist_rail = computations.group_sum(
#     group=["n"],
#     sum="area_type",
#     qty=long_dist_rail,
# )


long_dist_rail_share = long_dist_rail / long_dist_pdt

countries = [
    "India",
    "Pakistan",
    "Bangladesh",
    "Sri Lanka",
    "Afghanistan",
    "Bhutan",
    "Nepal",
    "Maldives",
]

rail_share_2050 = [0.6, 0.25, 0.25, 0.4, 0, 0, 0, 0]
treip_dist = ["31_50", "51+"]
area_type = ["large_city", "city", "town", "rural"]
new_df = pd.DataFrame(
    {"n": countries, "y": 2050, "mode": "rail_share", "value": rail_share_2050}
)

hsr = computations.concat(
    long_dist_rail_share,
    Quantity(
        pd.Series(
            pd.DataFrame(
                {
                    "n": countries,
                    "mode": "rail_share",
                    "y": 2050,
                    0: rail_share_2050,
                }
            ).set_index(["n", "mode", "y"])[0]
        )
    ),  # set share of rail to 52.7% in 2050 for all regions
    Quantity(
        pd.Series(
            pd.DataFrame(
                {
                    "n": np.nan,
                    "mode": np.nan,
                    "y": range(2051, 2101),
                    0: 0,
                }
            ).set_index(["n", "mode", "y"])[0]
        )
    ),  # keep the share constant beyond 2050
)

hsr = hsr.ffill("y")
# Interpolate
years = list(range(2011, 2101))
hsr = computations.interpolate(hsr, dict(y=years))
hsr = hsr * long_dist_pdt

other_modes = x[
    (x.index.get_level_values("trip_dist").isin(["31_50", "51+"]))
    & (~x.index.get_level_values("mode").isin(["rail_share"]))
]
other_modes_pdt = computations.group_sum(
    group=["y", "n"],
    sum="mode",
    qty=other_modes.drop(["area_type", "trip_dist"]),
)
other_modes_share = (
    computations.group_sum(
        group=["y", "n"],
        sum="trip_dist",
        qty=other_modes.drop(["area_type"]),
    )
    / other_modes_pdt
)

other_modes_total = (long_dist_pdt - hsr.drop("mode")) * other_modes_share

hsr = computations.concat(hsr, other_modes_total)
