# -*- coding: utf-8 -*-
"""
This is the demand model that produces pdt by mode for travel type:
    -Daily travel
    -Long distance travel

Produces outcomes for every:
    k- SSP scenario
    j- Type of urbanisation (daily travel only)
    m- mode share trajectory
    
Depdendent on:
    - growth_development.py 
    - passenger_trip_data.py
    
"""

from genno import computations, Quantity, Key
from pathlib import Path
from growth_development import pop_area_type, gdp_cap, pop
from passenger_trip_data import Mode_shares, Trip_rate, Long_dist_mode

# Specifiying base path for stored files
base_path = Path.cwd()

# Function that returns daily travel passenger kilometers by mode
# k : {1,2,3,4,5}- SSP scenario
# j : {0,1}- type of urbanisation
# m : {1,2,3}- mode share trajcetory


def Daily_travel(k, j, m) -> Quantity:
    Mode_shares(m)
    Trip_rate(k)
    data_files = [
        f"ipt_share_{m}.csv",
        f"tw_share_{m}.csv",
        f"bus_share_{m}.csv",
        f"ldv_share_{m}.csv",
        f"nmt_share_{m}.csv",
        f"rail_share_{m}.csv",
        "trip_distance.csv",
        f"trip_rate_{k}.csv",
        "trip_share.csv",
    ]

    ## setting up path objects
    path_objs = [base_path / filename for filename in data_files]
    csv_names = [
        i.stem.replace(f"_{m}", "")
        if "trip_rate" not in i.stem or m != 2
        else i.stem
        for i in path_objs
    ]

    ## gathers all trip data needed to calculate per capita values
    trip_data = {}
    for path, name in zip(path_objs, csv_names):
        trip_data[name] = computations.load_file(path)

    ## daily_travel is a dictionary containing per capita daily distance travelled by mode for
    ## each distance category and area_type
    daily_travel = {}
    daily_travel_pdt = Key(
        "daily_travel_pdt", ["mode", "area_type", "trip_dist", "y", "n"]
    )

    for i in [
        "ldv_share",
        "ipt_share",
        "tw_share",
        "bus_share",
        "nmt_share",
        "rail_share",
    ]:
        daily_travel[i] = computations.mul(
            trip_data[f"{i}"],
            trip_data["trip_share"],
            trip_data["trip_distance"],
            trip_data[f"trip_rate_{k}"],
        )
        # - Convert daily commute to annual total travel
        # - multiplying by 260- no. of working days in a year
        # - Commute accounts for 28.74%- find total passenger distance in km (daily)
        daily_travel[i] = daily_travel[i] * 260 / 0.2874

        ## multiply by population
        daily_travel[i] = daily_travel[i] * pop_area_type(k, j)
        daily_travel[i] = daily_travel[i].expand_dims(
            mode={f"{i}": len(daily_travel[i])}
        )
        daily_travel_pdt = computations.concat(
            daily_travel_pdt, daily_travel[i]
        )

    computations.write_report(
        daily_travel_pdt, Path(f"daily_travel_{k}_{j}_{m}.csv")
    )

    return daily_travel


# Function that returns long distance travel passenger kilometers by mode
# k : {1,2,3,4,5}- SSP scenario
# m : {1,2,3}- mode share trajcetory


def Long_dist_travel(k, m) -> Quantity:
    Long_dist_mode(m)
    long_dist_pkm = computations.load_file(Path("long_dist_pkm.csv"))
    long_dist_pkm = long_dist_pkm.expand_dims(y={2011: len(long_dist_pkm)})
    long_dist_mode = computations.load_file(
        Path(f"long_dist_modes_{m}.csv")
    )

    # index GDP per cap to 2011 (SSP scenario "k")
    gdp_cap_index = computations.index_to(gdp_cap(k), dim_or_selector="y")

    # Calibrate long dist travel to historical values
    # -Define list of countries which have historical rail pkm
    countries = ["India", "Bangladesh", "Pakistan", "Sri Lanka"]
    long_dist_calibrated = Key("long_dist_calibrated", ["y", "n"])
    # -Use established linear relationships to project future long distance travel
    # -See long_dist_calibrations.R for a and b values
    # -c is share of rail in long distance pkm
    for country, a, b, c in zip(
        countries,
        [242717, 3999, 11136, 11753],
        [789761, 3974, 8075, -7314],
        [0.15, 0.12, 0.1, 0.12],
    ):
        eq = (
            (
                a
                * gdp_cap_index[
                    gdp_cap_index.index.get_level_values("n") == country
                ]
                + b
            )
            / 1000
            / c
        )
        eq = eq[eq.index.get_level_values("y") != 2011]
        long_dist_calibrated = computations.concat(
            long_dist_calibrated, eq
        )

    long_dist_calibrated = computations.concat(
        long_dist_calibrated, long_dist_pkm
    )

    # Define list of countries that are not calibrated
    countries_uncalibrated = ["Maldives", "Nepal", "Bhutan", "Afghanistan"]

    # Calculate long distance travel for uncalibrated countries
    long_dist_uncalibrated = computations.mul(
        long_dist_pkm.drop("y"),
        gdp_cap_index[
            gdp_cap_index.index.get_level_values("n").isin(
                countries_uncalibrated
            )
        ],
    )
    # concat
    long_dist_travel = computations.concat(
        long_dist_calibrated, long_dist_uncalibrated
    )
    # Calculate total long distance travel by mode
    long_dist_travel = computations.mul(long_dist_travel, long_dist_mode)

    computations.write_report(
        long_dist_travel, Path(f"long_dist_travel_{k}_{m}.csv")
    )

    return long_dist_travel, long_dist_mode


# # Set SSP scenario- check passenger_trip_data.py for k
# k = 2

# # Set urbanisation type- check area_type_share(j) in growth_development.py
# j = 0


# # Set the mode share trajectory- check passenger_trip_data.py for m
# m = 1
# # Call the function that generates mode shares for different m's
# Mode_shares(m)

# # CSV data files to be read
# data_files = [
#     f"ipt_share_{m}.csv",
#     f"tw_share_{m}.csv",
#     f"bus_share_{m}.csv",
#     f"ldv_share_{m}.csv",
#     f"nmt_share_{m}.csv",
#     f"rail_share_{m}.csv",
#     "trip_distance.csv",
#     "trip_rate.csv",
#     "trip_share.csv",
# ]


# ## setting up path objects
# path_objs = [base_path / filename for filename in data_files]
# csv_names = [i.stem.replace(f"_{m}", "") for i in path_objs]

# ## gathers all trip data needed to calculate per capita values
# trip_data = {}
# for path, name in zip(path_objs, csv_names):
#     trip_data[name] = computations.load_file(path)

# ## pc_dt is a dictionary containing per capita daily distance travelled by mode for
# ## each distance category and area_type
# pc_dt = {}
# total_pdt_dt = 0
# mode_share = Key(
#     "mode_share", ["mode", "y", "n"]
# )  # create a quantity to concat later
# pdt_mode_at = Key("pdt_mode_at", ["mode", "area_type", "y", "n"])

# for i in [
#     "ldv_share",
#     "ipt_share",
#     "tw_share",
#     "bus_share",
#     "nmt_share",
#     "rail_share",
# ]:
#     pc_dt[i] = computations.mul(
#         trip_data[f"{i}"],
#         trip_data["trip_share"],
#         trip_data["trip_distance"],
#         trip_data["trip_rate"],
#     )
#     # - Convert daily commute to annual total travel
#     # - multiplying by 260- no. of working days in a year
#     # - Commute accounts for 28.74%- find total passenger distance in km (daily)
#     pc_dt[i] = pc_dt[i] * 260 / 0.2874

#     ## multiply by population
#     pc_dt[i] = pc_dt[i] * pop_area_type(k, j)

#     # Calculate total passenger distance travelled (million pkm)
#     total_pdt_dt += pc_dt[i]
#     # Calculate mode share
#     # - expand dimensions to include mode
#     # - sum by "trip_dist" and "area_type"
#     pdt_mode = pc_dt[i].expand_dims(mode={f"{i}": len(pc_dt[i])})
#     pdt_mode = computations.group_sum(
#         group=["y"], sum="trip_dist", qty=pdt_mode
#     )
#     # PDT by mode for each area type
#     pdt_mode_at = computations.concat(pdt_mode_at, pdt_mode)

#     pdt_mode = computations.group_sum(
#         group=["y"], sum="area_type", qty=pdt_mode
#     )
#     # concat all values
#     mode_share = computations.concat(mode_share, pdt_mode)


# # billion pkm
# mode_pdt = mode_share / 10**3

# # Calculate mode shares
# mode_share = mode_share / computations.group_sum(
#     group=["y"], sum="mode", qty=mode_share
# )
# computations.write_report(mode_share, Path("mode_shares.csv"))


# # Sum across trip_dist for total pdt by area type
# total_pdt_dt = computations.group_sum(
#     group=["y"], sum="trip_dist", qty=total_pdt_dt
# )


# # Calculate per capita pdt for all area types
# per_cap_daily = total_pdt_dt / pop_area_type(k, j)

# # Calculate total pdt in billion pkm
# total_pdt_dt = (
#     computations.group_sum(group=["y"], sum="area_type", qty=total_pdt_dt)
#     / 10**3
# )

# per_capita_national = (total_pdt_dt * 1000 / pop(k)).expand_dims(
#     area_type={"National avg.": len(total_pdt_dt)}
# )
# per_cap_daily = computations.concat(per_cap_daily, per_capita_national)
# computations.write_report(per_cap_daily, Path("pdt_cap_dt.csv"))


# # long-distance travel growth
# long_dist_path = Path("long_dist.csv")
# long_dist_travel = computations.load_file(long_dist_path)

# # index GDP per cap to 2011 (SSP scenario "k")
# gdp_cap_index = computations.index_to(gdp_cap(k), dim_or_selector="y")

# ## Total long distance travel in billion passenger kilometers- by rail and road
# long_dist_travel = computations.mul(long_dist_travel, gdp_cap_index)

# ## long distance travel by mode
# ldt_mode = long_dist_travel / computations.group_sum(
#     group=["y"], sum="mode", qty=long_dist_travel
# )

# # Total annual passenger kilometres (in billion) by mode in South Asia
# total_passenger_dist_mode = long_dist_travel + mode_pdt
# computations.write_report(
#     total_passenger_dist_mode, Path("mode_share_total.csv")
# )

# total_mode_share = total_passenger_dist_mode / computations.group_sum(
#     group=["y"], sum="mode", qty=total_passenger_dist_mode
# )

# total_passenger_dist = computations.group_sum(
#     group=["y"], sum="mode", qty=total_passenger_dist_mode
# )

# # Per capita passenger kilometres

# per_cap_dist = total_passenger_dist * 1000 / pop(k)

# computations.write_report(per_cap_dist, Path("pdt_cap.csv"))
