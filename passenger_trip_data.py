# -*- coding: utf-8 -*-
"""
This module produces passenger trip data which includes:
    - trip distance
    - trip share
    - trip rate (projections for each SSP)
    - mode share for each distance category
    - long distance travel by mode

"""

import pandas as pd
from pathlib import Path
from growth_development import gdp_cap
from genno import Quantity, computations, Key
from functools import reduce
import numpy as np

# Load the spreadsheet model
demand_model = pd.read_excel(
    "demand_spreadsheet_model.xlsx", sheet_name="India"
)

# Select trip distance to convert to genno
trip_dist = demand_model[
    ["trip_dist", "Typical distance"]
].drop_duplicates()
trip_dist = trip_dist.rename(columns={"Typical distance": "value"})
trip_dist = trip_dist.to_csv(index=False, lineterminator="\n")
trip_dist = trip_dist.replace(",", ", ")

data_info = """# Typical trip distance for each distance catgeory
#
# These are assumed values
#
#
"""
trip_dist = f"{data_info}{trip_dist}"
file_path = Path("trip_distance.csv")
file_path.write_text(trip_dist)


# Select trip share for each area type to convert to genno
trip_share = demand_model[["area_type", "trip_dist", "trip_share"]]
trip_share = trip_share.rename(columns={"trip_share": "value"})
trip_share = trip_share.to_csv(index=False, lineterminator="\n")
trip_share = trip_share.replace(",", ", ")

data_info = data_info = """# Trip share for each area type and trip distance catgeory
#
# Calculated values from Indian Census 2011
#
#
"""
trip_share = f"{data_info}{trip_share}"
file_path = Path("trip_share.csv")
file_path.write_text(trip_share)


def Trip_rate(k) -> Quantity:
    # select trip rate from demand model for each distance category and area type
    trip_rate = demand_model[
        ["trip_dist", "area_type", "trip_rate_adjusted"]
    ].drop_duplicates()
    trip_rate = trip_rate.rename(columns={"trip_rate_adjusted": "value"})
    # add 'y' dimension
    trip_rate["y"] = 2011

    years = [2011, 2030, 2050, 2100]
    ## Get GDP per capita for target years and SSP scenario 'k'

    gdp_cap_list = [
        (
            gdp_cap(k)
            .loc[:, year]
            .to_dataframe()
            .reset_index()
            .rename(columns={"value": str(year)})
        )
        for year in years
    ]

    # Merge values for all years into one dataframe

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on="n", how="outer"),
        gdp_cap_list,
    )
    merged_df.fillna(0, inplace=True)

    # Calulate % growth in GDP per capita- s1, s2 and s3
    s1 = pd.DataFrame(
        {
            "n": merged_df["n"],
            "value": ((merged_df["2030"] / merged_df["2011"]) - 1),
        }
    )

    s2 = pd.DataFrame(
        {
            "n": merged_df["n"],
            "value": ((merged_df["2050"] / merged_df["2030"]) - 1),
        }
    )

    s3 = pd.DataFrame(
        {
            "n": merged_df["n"],
            "value": ((merged_df["2100"] / merged_df["2050"]) - 1),
        }
    )

    ## add country as 'n' and assign same trip rates to all n for 2011
    countries = s1["n"]
    trip_rate = pd.concat(
        [trip_rate.assign(n=country) for country in countries],
        ignore_index=True,
    )

    # Function that returns trip rate trajectories for each distance category
    # Trajectories are defined based on slope from s1, s2, and s3 for each distance category
    def calculate_trip_rate(row, growth_rate, prev_value):
        calculated_value = prev_value * (1 + (growth_rate / 8))
        return calculated_value

    # def calculate_trip_rate(row, growth_rate, prev_value):
    #     calculated_value = (
    #         -0.5 * growth_rate + prev_value
    #         if row["trip_dist"] == "No_travel"
    #         else -0.25 * growth_rate + prev_value
    #         if row["trip_dist"] == "00_01"
    #         else -0.1 * growth_rate + prev_value
    #         if row["trip_dist"] == "02_05"
    #         else -0.05 * growth_rate + prev_value
    #         if row["trip_dist"] == "06_10"
    #         else 0.25 * growth_rate + prev_value
    #         if row["trip_dist"] == "11_20"
    #         else 0.5 * growth_rate + prev_value
    #         if row["trip_dist"] == "21_30"
    #         else 1 * growth_rate + prev_value
    #         if row["trip_dist"] == "31_50"
    #         else 1.5 * growth_rate + prev_value
    #         if row["trip_dist"] == "51+"
    #         else 1
    #     )  # Condition that if calculated_value is less than zero, then use previous value
    #     return (
    #         max(calculated_value, prev_value)
    #         if calculated_value < 0
    #         else calculated_value
    #     )

    trip_rate["2030"] = trip_rate.apply(
        lambda row: calculate_trip_rate(
            row,
            s1.loc[s1["n"] == row["n"], "value"].values[0],
            row["value"],
        ),
        axis=1,
    )

    trip_rate["2050"] = trip_rate.apply(
        lambda row: calculate_trip_rate(
            row,
            s2.loc[s1["n"] == row["n"], "value"].values[0],
            row["2030"],
        ),
        axis=1,
    )

    trip_rate["2100"] = trip_rate.apply(
        lambda row: calculate_trip_rate(
            row,
            s3.loc[s1["n"] == row["n"], "value"].values[0],
            row["2050"],
        ),
        axis=1,
    )

    # reshape dataframe
    new_df = trip_rate[
        ["trip_dist", "area_type", "2030", "2050", "2100", "n"]
    ]
    new_df = pd.melt(
        new_df,
        id_vars=["trip_dist", "area_type", "n"],
        var_name="y",
        value_name="value",
    )

    new_df["y"] = new_df["y"].astype(int)
    trip_rate = pd.concat(
        [trip_rate[["trip_dist", "area_type", "y", "value", "n"]], new_df]
    )

    # convert to genno Quantity
    trip_rate = Quantity(
        trip_rate.set_index(["trip_dist", "area_type", "y", "n"])["value"]
    )

    # interpolate values
    years_int = list(range(2011, 2101))
    trip_rate = computations.interpolate(trip_rate, dict(y=years_int))

    # save as csv file
    computations.write_report(trip_rate, Path(f"trip_rate_{k}.csv"))
    return trip_rate


# Mode share
file_list = [
    "ldv_share",
    "bus_share",
    "nmt_share",
    "tw_share",
    "rail_share",
    "ipt_share",
]


# A function that defines mode share trajectories for daily travel
def Mode_shares(m) -> Quantity:
    modes = {}
    new_modes = {}
    mode_share = Key("mode_share", ["mode", "y", "n"])
    if m == 1:  # BAU mode shares
        for i in file_list:
            modes[i] = demand_model[["area_type", "trip_dist", f"{i}"]]
            modes[i] = modes[i].rename(columns={f"{i}": "value"})
            modes[i] = modes[i].to_csv(index=False, lineterminator="\n")
            modes[i] = modes[i].replace(",", ", ")

            file_path = Path(f"{i}_{m}.csv")
            file_path.write_text(modes[i])
    elif m in {2, 3}:  # m=2: Car-oriented; m=3: Sustainable future
        file_name = (
            "car_oriented_modeshare.csv"
            if m == 2
            else "sustainable_modeshare.csv"
        )
        future_modes = pd.read_csv(file_name)

        for i in file_list:
            modes[i] = demand_model[["area_type", "trip_dist", f"{i}"]]
            modes[i]["y"] = 2011
            new_modes[i] = future_modes[
                ["area_type", "trip_dist", "y", f"{i}"]
            ]
            # Concatenate the original dataframe and the new_rows dataframe
            modes[i] = pd.concat([modes[i], new_modes[i]])
            new_rows = pd.DataFrame(
                [
                    (np.nan, np.nan, year, 0)
                    for year in list(range(2051, 2101))
                ],
                columns=["area_type", "trip_dist", "y", f"{i}"],
            )
            modes[i] = pd.concat([modes[i], new_rows]).rename(
                columns={f"{i}": "value"}
            )
            modes[i] = Quantity(
                modes[i].set_index(["area_type", "trip_dist", "y"])[
                    "value"
                ]
            )
            modes[i] = modes[i].ffill("y")
            modes[i] = computations.interpolate(
                modes[i], dict(y=list(range(2011, 2101)))
            )
            computations.write_report(modes[i], Path(f"{i}_{m}.csv"))
            modes[i] = modes[i].expand_dims(mode={f"{i}": len(modes[i])})
            mode_share = computations.concat(mode_share, modes[i])

        return mode_share


# A function that defined mode share trajcetories for long distance travel
def Long_dist_mode(m) -> Quantity:
    long_dist_data = pd.read_excel(
        "demand_spreadsheet_model.xlsx", sheet_name="long_dist"
    )
    long_dist_data = Quantity(
        long_dist_data.set_index(["n", "mode"])["value"]
    )
    # calculate total pkm for 2011
    long_dist_travel = computations.group_sum(
        group=["n"], sum="mode", qty=long_dist_data
    )
    # save as csv
    # Long distance passenger kilometers in billion
    # BAU assumption: bus-84% and ldv-16% of road
    computations.write_report(long_dist_travel, "long_dist_pkm.csv")
    # Calculate mode share in 2011
    long_dist_data = long_dist_data / computations.group_sum(
        group=["n"], sum="mode", qty=long_dist_data
    )
    m = 1
    if m == 1:  # BAU mode shares
        long_dist_data = long_dist_data.expand_dims(
            y={2011: len(long_dist_data)}
        )
        new_rows = Quantity(
            pd.DataFrame(
                [
                    (year, np.nan, np.nan, 0)
                    for year in list(range(2011, 2101))
                ],
                columns=["y", "n", "mode", "value"],
            ).set_index(["y", "n", "mode"])["value"]
        )
        long_dist_data = computations.concat(long_dist_data, new_rows)
        long_dist_data = long_dist_data.ffill("y")

        computations.write_report(
            long_dist_data, f"long_dist_modes_{m}.csv"
        )
        long_dist_modes = long_dist_data

    elif m in {2, 3}:  # m=2: Car-oriented; m=3: Sustainable future
        # Define changes in mode share in 2050
        mode_changes = {
            "ldv_share": 0.15 if m == 2 else -0.05,
            "bus_share": -0.15 if m == 2 else -0.05,
            "rail_share": 0.10 if m == 3 else 0,
        }
        # Future mode share
        future_mode = long_dist_data
        long_dist_modes = long_dist_data.expand_dims(
            y={2011: len(long_dist_data)}
        )
        for mode, change in mode_changes.items():
            future_mode.loc[
                future_mode.index.get_level_values(1).isin([mode])
            ] = (
                future_mode.loc[
                    future_mode.index.get_level_values(1).isin([mode])
                ]
                + change
            )
        # Exapand dimensions to set y=2050
        future_mode = future_mode.expand_dims(y={2050: len(future_mode)})
        # Add new rows beyond 2050
        new_rows = Quantity(
            pd.DataFrame(
                [
                    (year, np.nan, np.nan, 0)
                    for year in list(range(2051, 2101))
                ],
                columns=["y", "n", "mode", "value"],
            ).set_index(["y", "n", "mode"])["value"]
        )
        # Concat Values for 2011, 2050 and >2050

        long_dist_modes = computations.concat(
            long_dist_modes, future_mode, new_rows
        )
        long_dist_modes = long_dist_modes.ffill("y")
        long_dist_modes = computations.interpolate(
            long_dist_modes, dict(y=list(range(2011, 2101)))
        )
        computations.write_report(
            long_dist_modes, f"long_dist_modes_{m}.csv"
        )

    return long_dist_modes


# ## Long distance travel
# ##Unit: billion passenger kilometres
# # Assumption- bus-84% and ldv-16% of road
# file_path = Path("long_dist.csv")
# file_path.write_text(
#     """# Long distance travel by rail and road
# #
# #
# #
# #
# n, mode, value
# India, rail_share, 1150.74
# India, bus_share, 5477.514
# India, ldv_share, 1043.336
# Bangladesh, rail_share, 14.33
# Bangladesh, bus_share, 88.3008
# Bangladesh, ldv_share, 16.8192
# Pakistan, rail_share, 29.6
# Pakistan, bus_share, 223.7424
# Pakistan, ldv_share, 42.6176
# Sri Lanka, rail_share, 7.41
# Sri Lanka, bus_share, 45.6288
# Sri Lanka, ldv_share, 8.6912
# Nepal, bus_share, 24.7884
# Nepal, ldv_share, 4.7216
# Bhutan, bus_share, 0.6636
# Bhutan, ldv_share, 0.1264
# Afghanistan, bus_share, 23.5452
# Afghanistan, ldv_share, 4.4848
# Afghanistan, rail_share, 1.48
# Maldives, rail_share, 0
# Maldives, ldv_share, 0
# Maldives, bus_share, 0
# Maldives, nmt_share, 0
# Maldives, ipt_share, 0
# Maldives, tw_share, 0
# India, nmt_share, 0
# India, ipt_share, 0
# India, tw_share, 0
# Bangladesh, nmt_share, 0
# Bangladesh, ipt_share, 0
# Bangladesh, tw_share, 0
# Pakistan, nmt_share, 0
# Pakistan, ipt_share, 0
# Pakistan, tw_share, 0
# Sri Lanka, nmt_share, 0
# Sri Lanka, ipt_share, 0
# Sri Lanka, tw_share, 0
# Nepal, nmt_share, 0
# Nepal, ipt_share, 0
# Nepal, tw_share, 0
# Bhutan, nmt_share, 0
# Bhutan, ipt_share, 0
# Bhutan, tw_share, 0
# Afghanistan, nmt_share, 0
# Afghanistan, ipt_share, 0
# Afghanistan, tw_share, 0

# """
# )
# # ## define trajectories for changes in trip rates for each distance category
# trip_rate["2030"] = trip_rate.apply(
#     lambda row: -1 * s1.loc[s1["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "No_travel"
#     else -0.75 * s1.loc[s1["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "00_01"
#     else -0.5 * s1.loc[s1["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "02_05"
#     else -0.25 * s1.loc[s1["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "06_10"
#     else 0.5 * s1.loc[s1["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "11_20"
#     else 1 * s1.loc[s1["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "21_30"
#     else 1.5 * s1.loc[s1["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "31_50"
#     else 2 * s1.loc[s1["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "51+"
#     else 1,
#     axis=1,
# )

# trip_rate["2050"] = trip_rate.apply(
#     lambda row: -1 * s2.loc[s2["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "No_travel"
#     else -0.75 * s2.loc[s2["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "00_01"
#     else -0.5 * s2.loc[s2["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "02_05"
#     else -0.25 * s2.loc[s2["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "06_10"
#     else 0.5 * s2.loc[s2["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "11_20"
#     else 1 * s2.loc[s2["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "21_30"
#     else 1.5 * s2.loc[s2["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "31_50"
#     else 2 * s2.loc[s2["n"] == row["n"], "value"].values[0] / 8
#     + row["value"]
#     if row["trip_dist"] == "51+"
#     else 1,
#     axis=1,
# )
