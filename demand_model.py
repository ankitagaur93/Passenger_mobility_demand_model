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
from growth_development import pop_area_type, gdp_cap
from passenger_trip_data import Mode_shares, Trip_rate, Long_dist_mode

# Specifiying base path for stored files
base_path = Path.cwd()

# Function that returns daily travel passenger kilometers by mode
# k : {1,2,3,4,5}- SSP scenario
# j : {0,1}- type of urbanisation
# m : {1,2,3}- mode share trajcetory


def Daily_travel(k, j, m) -> Quantity:
    Mode_shares(k, m)
    Trip_rate(k)
    data_files = [
        f"ipt_share_{m}_{k}.csv",
        f"tw_share_{m}_{k}.csv",
        f"bus_share_{m}_{k}.csv",
        f"ldv_share_{m}_{k}.csv",
        f"nmt_share_{m}_{k}.csv",
        f"rail_share_{m}_{k}.csv",
        "trip_distance.csv",
        f"trip_rate_{k}.csv",
        "trip_share.csv",
    ]

    ## setting up path objects
    path_objs = [base_path / filename for filename in data_files]
    csv_names = [
        i.stem.replace(f"_{m}_{k}", "")
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
        daily_travel_pdt, Path(f"daily_travel_{k}_{j}_{m}_new.csv")
    )

    return daily_travel


# Function that returns long distance travel passenger kilometers by mode
# k : {1,2,3,4,5}- SSP scenario
# m : {1,2,3}- mode share trajcetory


def Long_dist_travel(k, m) -> Quantity:
    Long_dist_mode(k, m)
    long_dist_pkm = computations.load_file(Path("long_dist_pkm.csv"))
    long_dist_pkm = long_dist_pkm.expand_dims(y={2011: len(long_dist_pkm)})
    long_dist_mode = computations.load_file(
        Path(f"long_dist_modes_{k}_{m}.csv")
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
        [242717, 3999, 11136, 7449],
        [789761, 3974, 8075, -2441],
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
