# -*- coding: utf-8 -*-
"""
For each SSP; this module produces:
    - population (total and by area type)
    - urbanisation rate (by area type) 
    - GDP per capita

"""

import pandas as pd
import numpy as np
from pathlib import Path
from genno import Quantity, computations

# Load SSP data on growth and development
# - Population projections
# - GDP projections
# - Future urbanisation rates
data_gnd = (
    pd.read_csv("ssp_snapshot.csv")
    .query("`2050`.notna()")
    .melt(
        id_vars=["Model", "Scenario", "Region", "Variable", "Unit"],
        value_vars=[str(year) for year in range(2010, 2105, 5)],
        var_name="y",
        value_name="value",
    )
    .query("y >= '2025'")
    .rename(columns={"Region": "n"})
)

# Load historical data on growth and development
hist_gnd = (
    pd.read_csv("historical_gnd.csv")
    .melt(
        id_vars=["Region", "Variable", "Unit"],
        value_vars=[str(year) for year in range(2011, 2021, 1)],
        var_name="y",
        value_name="value",
    )
    .rename(columns={"Region": "n"})
)

# Define a dictionary with 'k' referring to SSP scenario number
scenario_mapping = {
    1: "SSP1 - Review Phase 1",
    2: "SSP2 - Review Phase 1",
    3: "SSP3 - Review Phase 1",
    4: "SSP4 - Review Phase 1",
    5: "SSP5 - Review Phase 1",
}


# Function that returns population (in millions) for each SSP scenario
def pop(k) -> Quantity:
    # Get the scenario string based on k
    scenario = scenario_mapping.get(k, "")
    # Get population based on k
    pop = data_gnd.query(
        # scenario is used within the query string
        "Scenario == @scenario and Variable == 'Population'"
    )
    # Concat historical values with SSP projections
    pop_hist = hist_gnd.query("Variable == 'Population'")
    pop = pd.concat(
        [pop_hist[["n", "y", "value"]], pop[["n", "y", "value"]]]
    )
    pop["y"] = pop["y"].astype(int)
    # Convert to genno
    pop = Quantity(pop.set_index(["n", "y"])["value"])
    return pop


# Funtion that defines the distribution of urbanisation between
# area types: large city, city, & town
def area_type_share(j) -> Quantity:
    area_share = pd.read_csv("urban_area_type.csv")

    ## define future distribution of urbanisation
    if j == 0:
        future_urb = [
            ["large_city", 0.05, 0.02],
            ["city", -0.025, -0.01],
            ["town", -0.025, -0.01],
        ]
    if j == 1:
        future_urb = [
            ["large_city", 0.10, 0.07],
            ["city", -0.05, -0.035],
            ["town", -0.05, -0.035],
        ]

    future_urb_df = pd.DataFrame(
        future_urb, columns=["area_type", 2030, 2050]
    )
    ## Merge old and new values in a single dataframe
    area_share = pd.merge(
        area_share, future_urb_df, on="area_type", how="left"
    )
    area_share[2030] = area_share[2030] + area_share["2011"]
    area_share[2050] = area_share[2030] + area_share[2050]
    area_share = pd.melt(
        area_share,
        id_vars=["Country", "area_type"],
        var_name="year",
        value_name="value",
    )

    ## Add new rows beyond 2050
    years = list(range(2051, 2101))
    new_rows = pd.DataFrame(
        [(np.nan, np.nan, year, 0) for year in years],
        columns=["Country", "area_type", "year", "value"],
    )
    area_share = pd.concat([area_share, new_rows])

    ## - Rename columns
    ## - set n, y as dimensions and convert "value" to genno quantity
    area_share = area_share.rename(
        columns={"Country": "n", "year": "y"}
    ).astype({"y": int})
    area_share = Quantity(
        area_share.set_index(["n", "area_type", "y"])["value"]
    )

    ## - 2050 value used until 2100 (Assume no change in urbanisation pattern after 2050)
    years_int = list(range(2011, 2101))
    area_share = area_share.ffill("y")

    ## - interpolate values between 2011, 2030 and 2050
    area_share = computations.interpolate(
        area_share,
        dict(y=years_int),
    )
    computations.write_report(area_share, Path(f"area_share_{j}.csv"))

    return area_share


# Function that returns population for each area type in million
def pop_area_type(k, j) -> Quantity:
    # Get the scenario string based on k
    scenario = scenario_mapping.get(k, "")
    # Get urbanisation rate based on k
    urb = data_gnd.query(
        "Scenario == @scenario and Variable == 'Population|Urban|Share'"
    )
    # Concat historical values with SSP projections
    urb_hist = hist_gnd.query("Variable == 'Population|Urban|Share'")
    urb = pd.concat(
        [urb_hist[["n", "y", "value"]], urb[["n", "y", "value"]]]
    )
    urb["y"] = urb["y"].astype(int)
    # Convert to genno and convert percent to share
    urb = Quantity(urb.set_index(["n", "y"])["value"]) / 100
    # Calculate total urban population (in million) by area type
    urb_pop = computations.mul(urb, pop(k), area_type_share(j))
    # Calculate rural population
    rural_pop = pop(k) - computations.group_sum(
        group=["n", "y"], sum="area_type", qty=urb_pop
    )

    # Add area_type dimension to match urb_pop format
    rural_pop = rural_pop.expand_dims(area_type={"rural": len(rural_pop)})

    # Concat quantities to get total population by area type
    total_pop = computations.concat(urb_pop, rural_pop)
    computations.write_report(total_pop, f"total_pop_{k}_{j}.csv")

    return total_pop


# Function that returns gdp per capita (US $) for each SSP scenario
def gdp_cap(k) -> Quantity:
    # Get the scenario string based on k
    scenario = scenario_mapping.get(k, "")
    # Get GDP based on k
    # scenario is used within the query string
    gdp = data_gnd.query("Scenario == @scenario and Variable == 'GDP|PPP'")
    # Concat historical values with SSP projections
    gdp_hist = hist_gnd.query("Variable == 'GDP|PPP'")
    gdp = pd.concat(
        [gdp_hist[["n", "y", "value"]], gdp[["n", "y", "value"]]]
    )
    gdp["y"] = gdp["y"].astype(int)
    # Convert to genno
    gdp = Quantity(gdp.set_index(["n", "y"])["value"])
    # Calculate GDP per capita for each SSP scenario
    gdp_cap = gdp * 1000 / pop(k)

    return gdp_cap
