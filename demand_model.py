# -*- coding: utf-8 -*-
"""
This is the demand model that produces pdt by mode for travel type:
    -Commute travel
    -Non commute travel

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
from growth_development import pop_area_type
from passenger_trip_data import Mode_shares, Trip_rate

# Specifiying base path for stored files
base_path = Path.cwd()

# Function that returns daily travel passenger kilometers by mode
# k : {1,2,3,4,5}- SSP scenario
# j : {1,2}- type of urbanisation
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
        "trip_distance_c.csv",
        f"trip_rate_c_{k}.csv",
        "trip_distance_nc.csv",
        f"trip_rate_nc_{k}.csv",
        "trip_share.csv",
    ]

    ## setting up path objects
    path_objs = [base_path / filename for filename in data_files]
    csv_names = [
        (
            i.stem.replace(f"_{m}_{k}", "")
            if "trip_rate" not in i.stem or m != 2
            else i.stem
        )
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
        "daily_travel_pdt",
        ["mode", "area_type", "trip_dist", "y", "n"],
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
            (
                trip_data["trip_distance_c"] * trip_data[f"trip_rate_c_{k}"]
                + trip_data["trip_distance_nc"]
                * trip_data[f"trip_rate_nc_{k}"]
            ),
        )
        # - Convert daily commute to annual total travel
        # - multiplying by 260- no. of working days in a year

        daily_travel[i] = daily_travel[i] * 260

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

    return daily_travel_pdt
