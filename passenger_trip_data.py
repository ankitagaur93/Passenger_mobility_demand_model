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

# # Select trip distance to convert to genno
trip_dist = (
    demand_model[["trip_dist", "Typical distance"]]
    .drop_duplicates()
    .rename(columns={"Typical distance": "value"})
)
trip_dist["y"] = 2011

data_2050 = pd.DataFrame(
    {
        "trip_dist": [
            "No_travel",
            "00_01",
            "02_05",
            "06_10",
            "11_20",
            "21_30",
            "31_50",
            "51+",
        ],
        "value": [0, 1, 4, 9.5, 18, 27, 48, 80],
        "y": [2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050],
    }
)

# Concatenate the two DataFrames
trip_dist = pd.concat([trip_dist, data_2050], ignore_index=True)

new_rows = pd.DataFrame(
    [(np.nan, year, 0) for year in list(range(2051, 2101))],
    columns=["trip_dist", "y", "value"],
)
trip_dist = Quantity(
    pd.concat([trip_dist, new_rows], ignore_index=True).set_index(
        ["trip_dist", "y"]
    )["value"]
)

trip_dist = trip_dist.ffill("y")
trip_dist = computations.interpolate(
    trip_dist, dict(y=list(range(2011, 2101)))
)
computations.write_report(trip_dist, Path("trip_distance.csv"))

# Select trip distance to convert to genno
# trip_dist = demand_model[
#     ["trip_dist", "Typical distance"]
# ].drop_duplicates()
# trip_dist = trip_dist.rename(columns={"Typical distance": "value"})
# trip_dist = trip_dist.to_csv(index=False, lineterminator="\n")
# trip_dist = trip_dist.replace(",", ", ")

# data_info = """# Typical trip distance for each distance catgeory
# #
# # These are assumed values
# #
# #
# """
# trip_dist = f"{data_info}{trip_dist}"
# file_path = Path("trip_distance.csv")
# file_path.write_text(trip_dist)


# # Select trip share for each area type to convert to genno
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

# Dataframe that contains GDP value converter
# Source: IMF (~30 year avg considered, assume same value throughout)
gdp_conv = pd.DataFrame(
    {
        "n": [
            "Afghanistan",
            "Bangladesh",
            "Bhutan",
            "India",
            "Sri Lanka",
            "Maldives",
            "Nepal",
            "Pakistan",
        ],
        "value": [
            4.33626287,
            4.683377012,
            4.991630661,
            4.412238944,
            5.280631338,
            2.929762585,
            6.368285382,
            5.14564764,
        ],
    }
)

# 2-Wheeler Kuznets curve: https://doi.org/10.1016/j.jtrangeo.2014.03.008
# tw_kc defines the GDP per capita value at which share of 2-Wheelers starts to decrease
tw_kc = 3500

# Convert GDP at current prices to GDP |PPP @ 2017 prices
gdp_conv["value"] = tw_kc * gdp_conv["value"]
gdp_conv = Quantity(gdp_conv.set_index(["n"])["value"])


def Mode_shares(k, m) -> Quantity:
    # Calculate the minimum difference to find year of inflection for 2-wheelers
    diff = abs(gdp_cap(k) - gdp_conv)
    min_values = diff.groupby("n").idxmin().values

    # Define log(GDP_cap)
    log_GDP_cap = np.log(gdp_cap(k))
    log_GDP_cap = computations.index_to(log_GDP_cap, dim_or_selector="y")

    # Define a monotonically decreasing fucntion between NMT share and GDP per cap
    # decay rate controls slope of the curve
    def nmt_share_func(x, decay_rate):
        return Quantity(
            np.exp(-decay_rate * x) / np.exp(-decay_rate * x[0])
        )

    # Constant function for IPT
    def ipt_share_func(x):
        return x / x

    def bus_share_func(x, rate):
        # Monotonically increasing function
        return np.exp(rate * x) / np.exp(rate * x[0])

    def rail_share_func(x, rate):
        # Monotonically increasing function
        return np.exp(rate * x) / np.exp(rate * x[0])

    # Function that defined growth of LDVs
    def ldv_share_func(x, rate):
        return np.exp(rate * x) / np.exp(rate * x[0])

    # Inverted U function for 2-wheelers
    def tw_share_func(x, peak_point, peak_value):
        return np.exp(-((x - peak_point) ** 2) / peak_value)

    if m == 1:  # Baseline mode shares
        bus_rate = 0.8
        rail_rate = 0.7
        ldv_rate = 2
        tw_peak = 0.08

    elif m == 2:  # Car-oriented future
        bus_rate = 0.6
        rail_rate = 0.5
        ldv_rate = 3
        tw_peak = 0.08

    elif m == 3:  # Sustainable future
        bus_rate = 2.2
        rail_rate = 2.1
        ldv_rate = 1.5
        tw_peak = 0.25

    nmt = nmt_share_func(log_GDP_cap, 0.3)
    bus = bus_share_func(log_GDP_cap, bus_rate)
    rail = rail_share_func(log_GDP_cap, rail_rate)
    ldv = ldv_share_func(log_GDP_cap, ldv_rate)
    ipt = ipt_share_func(log_GDP_cap)

    tw_all = pd.DataFrame()

    # Loop calculates 2-wheeler trajectory for all countries
    for n, y in min_values:
        peak_value = (
            tw_peak  # Assuming peak_value is constant for all countries
        )
        peak_point = log_GDP_cap.loc[n, y]
        tw = tw_share_func(log_GDP_cap[n], peak_point, peak_value)
        tw_all[n] = tw

    tw_concat = pd.concat(
        [tw_all[n] for n in tw_all.columns], axis=1
    ).reset_index()

    tw_concat = pd.melt(
        tw_concat, id_vars="y", var_name="n", value_name="value"
    ).set_index(["n", "y"])["value"]

    tw = Quantity(tw_concat)

    ldv = computations.index_to(ldv, dim_or_selector="y")
    bus = computations.index_to(bus, dim_or_selector="y")
    nmt = computations.index_to(nmt, dim_or_selector="y")
    rail = computations.index_to(rail, dim_or_selector="y")
    tw = computations.index_to(tw, dim_or_selector="y")
    ipt = computations.index_to(ipt, dim_or_selector="y")

    # Adjust mode shares to avoid excessive PDT due to mode shares
    # Tw not included to avoid shifting of inflection point
    total_share = (ldv + bus + rail + nmt + ipt) / 5
    ldv /= total_share
    nmt /= total_share
    bus /= total_share
    rail /= total_share
    ipt /= total_share

    # ldv = ldv.to_dataframe().rename(columns={"value": "LDV"})
    # nmt = nmt.to_dataframe().rename(columns={"value": "NMT"})
    # tw = tw.to_dataframe().rename(columns={"value": "2-Wheeler"})
    # rail = rail.to_dataframe().rename(columns={"value": "Rail"})
    # bus = bus.to_dataframe().rename(columns={"value": "Bus"})
    # ipt = ipt.to_dataframe().rename(columns={"value": "IPT"})

    # merged_df = pd.concat(
    #     [
    #         ldv["LDV"],
    #         nmt["NMT"],
    #         bus["Bus"],
    #         rail["Rail"],

    #         ipt["IPT"],
    #         tw["2-Wheeler"],

    #     ],
    #     axis=1,
    # )

    modes = {}
    mode_share = Key("mode_share", ["mode", "y", "n"])

    # Multiply mode share growth with demand_model

    for i in file_list:
        modes[i] = demand_model[["area_type", "trip_dist", f"{i}"]]
        modes[i] = modes[i].rename(columns={f"{i}": "value"})
        modes[i] = Quantity(
            modes[i].set_index(["area_type", "trip_dist"])["value"]
        )
        for mode in [
            "ldv",
            "bus",
            "nmt",
            "tw",
            "rail",
            "ipt",
        ]:
            if mode in i:
                modes[i] = computations.mul(modes[i], locals()[mode])
                break  # Exit the loop once a match is found
        computations.write_report(modes[i], Path(f"{i}_{m}_{k}.csv"))
        modes[i] = modes[i].expand_dims(mode={f"{i}": len(modes[i])})
        mode_share = computations.concat(mode_share, modes[i])

    return mode_share


# # A function that defines mode share trajectories for daily travel
# def Mode_shares(m) -> Quantity:
#     modes = {}
#     new_modes = {}
#     mode_share = Key("mode_share", ["mode", "y", "n"])
#     if m == 1:  # BAU mode shares
#         for i in file_list:
#             modes[i] = demand_model[["area_type", "trip_dist", f"{i}"]]
#             modes[i] = modes[i].rename(columns={f"{i}": "value"})
#             modes[i] = modes[i].to_csv(index=False, lineterminator="\n")
#             modes[i] = modes[i].replace(",", ", ")

#             file_path = Path(f"{i}_{m}.csv")
#             file_path.write_text(modes[i])
#     elif m in {2, 3}:  # m=2: Car-oriented; m=3: Sustainable future
#         file_name = (
#             "car_oriented_modeshare.csv"
#             if m == 2
#             else "sustainable_modeshare.csv"
#         )
#         future_modes = pd.read_csv(file_name)

#         for i in file_list:
#             modes[i] = demand_model[["area_type", "trip_dist", f"{i}"]]
#             modes[i]["y"] = 2011
#             new_modes[i] = future_modes[
#                 ["area_type", "trip_dist", "y", f"{i}"]
#             ]
#             # Concatenate the original dataframe and the new_rows dataframe
#             modes[i] = pd.concat([modes[i], new_modes[i]])
#             new_rows = pd.DataFrame(
#                 [
#                     (np.nan, np.nan, year, 0)
#                     for year in list(range(2051, 2101))
#                 ],
#                 columns=["area_type", "trip_dist", "y", f"{i}"],
#             )
#             modes[i] = pd.concat([modes[i], new_rows]).rename(
#                 columns={f"{i}": "value"}
#             )
#             modes[i] = Quantity(
#                 modes[i].set_index(["area_type", "trip_dist", "y"])[
#                     "value"
#                 ]
#             )
#             modes[i] = modes[i].ffill("y")
#             modes[i] = computations.interpolate(
#                 modes[i], dict(y=list(range(2011, 2101)))
#             )
#             computations.write_report(modes[i], Path(f"{i}_{m}.csv"))
#             modes[i] = modes[i].expand_dims(mode={f"{i}": len(modes[i])})
#             mode_share = computations.concat(mode_share, modes[i])

#         return mode_share


# A function that defined mode share trajcetories for long distance travel
def Long_dist_mode(k, m) -> Quantity:
    # Define log(GDP_cap)
    log_GDP_cap = np.log(gdp_cap(2))
    log_GDP_cap = computations.index_to(log_GDP_cap, dim_or_selector="y")

    # Function that defined growth of LDVs
    def ldv_share_func(x, ldv_rate):
        return np.exp(ldv_rate * x) / np.exp(ldv_rate * x[0])

    def rail_share_func(x, rail__rate):
        return Quantity(
            np.exp(-rail__rate * x) / np.exp(-rail__rate * x[0])
        )

    # Constant function for IPT
    def bus_share_func(x):
        return x / x

    if m == 1:  # BaU mode share
        ldv_rate = 1.5
        rail_rate = 0.3
    elif m == 2:  # Car-oriented future
        ldv_rate = 2.5
        rail_rate = 0.5
    elif m == 3:  # Sustainable future
        ldv_rate = 0.9
        rail_rate = -1

    ldv_share = ldv_share_func(log_GDP_cap, ldv_rate)
    rail_share = rail_share_func(log_GDP_cap, rail_rate)
    bus_share = bus_share_func(log_GDP_cap)

    # Load the spreadsheet model
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

    ldv_share = computations.mul(
        ldv_share.rename("value"),
        long_dist_data.iloc[
            long_dist_data.index.get_level_values("mode").isin(
                ["ldv_share"]
            )
        ],
    )

    bus_share = computations.mul(
        bus_share.rename("value"),
        long_dist_data.iloc[
            long_dist_data.index.get_level_values("mode").isin(
                ["bus_share"]
            )
        ],
    )

    rail_share = computations.mul(
        rail_share.rename("value"),
        long_dist_data.iloc[
            long_dist_data.index.get_level_values("mode").isin(
                ["rail_share"]
            )
        ],
    )

    total_share = (
        ldv_share.drop("mode")
        + rail_share.drop("mode")
        + bus_share.drop("mode")
    )

    # Adjust shares so that sum is 1
    ldv_share /= total_share
    bus_share /= total_share
    rail_share /= total_share

    # account for modes with no contribution in long distance
    # this is needed for later steps
    years = list(range(2011, 2021)) + list(range(2025, 2101, 5))

    modes_un = Quantity(
        pd.DataFrame({"y": years, "value": [0] * len(years)}).set_index(
            ["y"]
        )["value"]
    )

    modes_un = computations.mul(
        modes_un,
        long_dist_data.iloc[
            long_dist_data.index.get_level_values("mode").isin(
                ["nmt_share", "ipt_share", "tw_share"]
            )
        ],
    )

    modes = computations.concat(ldv_share, rail_share, bus_share, modes_un)
    computations.write_report(modes, f"long_dist_modes_{k}_{m}.csv")

    return modes
