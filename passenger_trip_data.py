# -*- coding: utf-8 -*-
"""
This module produces passenger trip data which includes:
    - trip distance
    - trip share
    - trip rate (projections for each SSP)
    - daily travel mode share trajcetories


"""

import pandas as pd
from pathlib import Path
from growth_development import gdp_cap
from genno import Quantity, computations, Key
from functools import reduce
import numpy as np

# Load the spreadsheet model
demand_model = pd.read_excel(
    "demand_spreadsheet_model_up.xlsx", sheet_name="Sheet1"
)

# # Select trip distance for commute and convert to genno
trip_dist_c = (
    demand_model[["Country", "trip_dist", "Typical_distance_c"]]
    .drop_duplicates()
    .rename(columns={"Typical_distance_c": "value", "Country": "n"})
)
trip_dist_c["y"] = 2011

# # Select trip distance for non-commute and convert to genno
trip_dist_nc = (
    demand_model[["Country", "trip_dist", "Typical_dist_nc"]]
    .drop_duplicates()
    .rename(columns={"Typical_dist_nc": "value", "Country": "n"})
)
trip_dist_nc["y"] = 2011

## Define typical distance for future
# commute
data_2050_c = pd.DataFrame(
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
# non-commute
data_2050_nc = pd.DataFrame(
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
        "value": [0, 1, 4, 9.5, 18, 27, 48, 100],
        "y": [2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050],
    }
)
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

data_2050_c = pd.merge(
    pd.DataFrame({"n": countries}), data_2050_c, how="cross"
)

data_2050_nc = pd.merge(
    pd.DataFrame({"n": countries}), data_2050_nc, how="cross"
)

# Concatenate the two DataFrames
trip_dist_c = pd.concat([trip_dist_c, data_2050_c], ignore_index=True)
trip_dist_nc = pd.concat([trip_dist_nc, data_2050_nc], ignore_index=True)


new_rows = pd.DataFrame(
    [(np.nan, np.nan, year, 0) for year in list(range(2051, 2101))],
    columns=["n", "trip_dist", "y", "value"],
)

trip_dist_c = Quantity(
    pd.concat([trip_dist_c, new_rows], ignore_index=True).set_index(
        ["n", "trip_dist", "y"]
    )["value"]
)

trip_dist_nc = Quantity(
    pd.concat([trip_dist_nc, new_rows], ignore_index=True).set_index(
        ["n", "trip_dist", "y"]
    )["value"]
)

trip_dist_c = trip_dist_c.ffill("y")

trip_dist_c = computations.interpolate(
    trip_dist_c, dict(y=list(range(2011, 2101)))
)

trip_dist_nc = trip_dist_nc.ffill("y")
trip_dist_nc = computations.interpolate(
    trip_dist_nc, dict(y=list(range(2011, 2101)))
)


computations.write_report(trip_dist_c, Path("trip_distance_c.csv"))
computations.write_report(trip_dist_nc, Path("trip_distance_nc.csv"))


# # Select trip share for each area type to convert to genno
trip_share = demand_model[["area_type", "trip_dist", "trip_share"]]
trip_share = trip_share.rename(
    columns={"trip_share": "value"}
).drop_duplicates()
trip_share = trip_share.to_csv(index=False, lineterminator="\n")
trip_share = trip_share.replace(",", ", ")

data_info = (
    data_info
) = """# Trip share for each area type and trip distance catgeory
#
# Calculated values from Indian Census 2011
#
#
"""
trip_share = f"{data_info}{trip_share}"
file_path = Path("trip_share.csv")
file_path.write_text(trip_share)


def Trip_rate(k) -> Quantity:
    # select commute trip rate from demand model for each distance category and area type
    trip_rate = demand_model[
        ["trip_dist", "area_type", "trip_rate_adjusted_c"]
    ].drop_duplicates()
    trip_rate = trip_rate.rename(columns={"trip_rate_adjusted_c": "value"})
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
    trip_rate_c = computations.interpolate(trip_rate, dict(y=years_int))

    # compute non-commute trip rate
    # - Commute accounts for 28.74%- find total passenger distance in km (daily)
    trip_rate_nc = trip_rate_c / 0.32

    # save as csv file
    computations.write_report(trip_rate_c, Path(f"trip_rate_c_{k}.csv"))
    computations.write_report(trip_rate_nc, Path(f"trip_rate_nc_{k}.csv"))

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
    # Countries without rail
    no_rail_countries = ["Afghanistan", "Bhutan", "Maldives", "Nepal"]

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
        """
        Monotonically increasing function for rail share.
        Returns 1 for countries without rail, normal exponential growth for others.
        """
        no_rail_countries = ["Afghanistan", "Bhutan", "Maldives", "Nepal"]
        y = pd.Series(index=x.index, dtype=float)

        for n in x.index.get_level_values("n").unique():
            mask = x.index.get_level_values("n") == n
            if n in no_rail_countries:
                y[mask] = 1.0  # fixed value for no-rail countries
            else:
                x_n = x[mask]
                y[mask] = np.exp(rate * x_n) / np.exp(
                    rate * x_n.iloc[0]
                )  # normal growth

        return Quantity(y)

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

    # List of countries without rail
    no_rail_countries = ["Afghanistan", "Bhutan", "Maldives", "Nepal"]

    # Create mask
    mask_no_rail = ldv.index.get_level_values("n").isin(no_rail_countries)

    # Initialize total_share as same shape
    total_share = pd.Series(index=ldv.index, dtype=float)

    # For countries with rail
    total_share[~mask_no_rail] = (ldv + bus + rail + nmt + ipt)[
        ~mask_no_rail
    ] / 5

    # For countries without rail
    total_share[mask_no_rail] = (ldv + bus + nmt + ipt)[mask_no_rail] / 4

    total_share = Quantity(
        (total_share.reset_index()).set_index(["n", "y"])
    )
    # Adjust mode shares to avoid excessive PDT due to mode shares
    # Tw not included to avoid shifting of inflection point
    # total_share = (ldv + bus + rail + nmt + ipt) / 5
    ldv /= total_share
    nmt /= total_share
    bus /= total_share
    rail /= total_share
    ipt /= total_share

    computations.write_report(ldv, Path(f"ldv_{m}_{k}.csv"))
    computations.write_report(bus, Path(f"bus_{m}_{k}.csv"))
    computations.write_report(tw, Path(f"tw_{m}_{k}.csv"))
    computations.write_report(rail, Path(f"rail_{m}_{k}.csv"))
    computations.write_report(ipt, Path(f"ipt_{m}_{k}.csv"))
    computations.write_report(nmt, Path(f"nmt_{m}_{k}.csv"))

    modes = {}
    mode_share = Key("mode_share", ["mode", "y", "n"])

    # Multiply mode share growth with demand_model

    for i in file_list:
        modes[i] = demand_model[
            ["Country", "area_type", "trip_dist", f"{i}"]
        ]
        modes[i] = modes[i].rename(
            columns={f"{i}": "value", "Country": "n"}
        )
        modes[i] = Quantity(
            modes[i].set_index(["n", "area_type", "trip_dist"])["value"]
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
