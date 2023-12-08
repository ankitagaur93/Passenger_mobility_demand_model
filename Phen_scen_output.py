# -*- coding: utf-8 -*-
"""
This module handles the definition and modelling of each phenomenon 
and all the scenarios. 

It produces pdt by mode for each scenario, 
and other MESSAGEix-Transport parameters.

Alternate outcomes for each phenomenon can be defined within functions
"""

import os
import numpy as np
import pandas as pd
from demand_model import Daily_travel, Long_dist_travel
from genno import Quantity, computations, Key

os.getcwd()

# Load phenomena settings
phen_setting = pd.read_csv("phen_settings.csv")

# Get unique combinations of phen, area_type, and parameter
phen_set_unique = phen_setting[
    ["phen", "area_type", "parameter", "trip_dist"]
].drop_duplicates()


# Create a DataFrame with NaN values for the specified columns
new_rows = pd.DataFrame(
    {
        "phen": np.nan,
        "parameter": np.nan,
        "trip_dist": np.nan,
        "area_type": np.nan,
        "y": list(range(2051, 2101)),
        "value": 0.0,
    }
)

# Append the new rows to the DataFrame
phen_setting = pd.concat(
    [pd.DataFrame(new_rows, columns=phen_setting.columns), phen_setting],
    ignore_index=True,
)

# Convert to genno
phen_setting = Quantity(
    phen_setting.set_index(
        ["phen", "parameter", "trip_dist", "area_type", "y"]
    )["value"]
)
# For years > 2050, fill value same as the one in 2050
phen_setting = phen_setting.ffill("y")

# Values to be interpolated
phen_setting = computations.interpolate(
    phen_setting, dict(y=list(range(2011, 2101)))
)


# functions to handle each phenomenon individually
# i is an integer which can be used to define alternate outcomes for each phenomenon
def URB(i) -> Quantity:
    """Urbanisation-
    representation: increase in pdt"""
    urb = phen_setting["URB"].drop("trip_dist") + 1
    return urb


def AV(i) -> Quantity:
    """Autonomous vehicle-
    representation: increase in LDV vkt
                    decrease in pt share (both long run)"""
    av = phen_setting["AV"].drop("trip_dist") + 1
    occ_av = 1.5  # assumtpion
    ldv_share = av["vdt"] * occ_av  # calculate pdt
    ldv_share = ldv_share.expand_dims(
        parameter={"ldv_share": len(ldv_share)}
    )
    av = computations.concat(av, ldv_share)
    return av


def ELIFE(i) -> Quantity:
    """Elife ("Tele-X")-
    representation: reduction in trip_rate"""
    elf = phen_setting["ELIFE"].drop("trip_dist") + 1
    return elf


def ENV(i) -> Quantity:
    """Environmental awareness-
    representation: high share of EVs"""
    env = phen_setting["ENV"].drop("trip_dist")
    return env


def HSR(i, m) -> Quantity:
    """High speed rail-
    representation: mode shift to rail in long distance travel"""
    long_dist_travel, ldt_mode = Long_dist_travel(2, m)
    # ldt_mode = ldt_mode.expand_dims(y={2011: len(ldt_mode)})
    hsr = ldt_mode[
        (ldt_mode.index.get_level_values("mode") == "rail_share")
        & (ldt_mode.index.get_level_values("y") == 2011)
    ]  # get share of rail in 2011

    hsr = computations.concat(
        hsr,
        Quantity(
            pd.Series(
                pd.DataFrame(
                    {
                        "n": hsr.index.get_level_values("n"),
                        "mode": hsr.index.get_level_values("mode"),
                        "y": 2050,
                        0: 0.527,
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

    # Calculate total pdt by rail
    pdt_rail = (
        computations.group_sum(
            group=["y"], sum="mode", qty=long_dist_travel
        )
        * hsr
    )

    # Calculate road_based pdt
    # - Calculate share of bus and LDV
    # - Multiply by remaining pdt (not from rail)
    pdt_bus_car = (
        long_dist_travel[
            long_dist_travel.index.get_level_values("mode").isin(
                ["ldv_share", "bus_share"]
            )
        ]
        / computations.group_sum(
            group=["y"],
            sum="mode",
            qty=long_dist_travel[
                long_dist_travel.index.get_level_values("mode").isin(
                    ["ldv_share", "bus_share"]
                )
            ],
        )
        * (
            computations.group_sum(
                group=["y"], sum="mode", qty=long_dist_travel
            )
            - pdt_rail.drop("mode")
        )
    )

    # Concat all quantities
    hsr = computations.concat(
        long_dist_travel[
            long_dist_travel.index.get_level_values("mode").isin(
                ["nmt_share", "tw_share", "ipt_share"]
            )
        ],
        pdt_rail,
        pdt_bus_car,
    )

    return hsr


def HSL(i) -> Quantity:
    """High standard of living-
    representation: increase in LDV ownerhsip"""
    hsl = phen_setting["HSL"].drop("trip_dist") + 1
    return hsl


def NMT(i) -> Quantity:
    """Non-motorised transport infrastructure development-
    representation: shift from 2-wheelers and bus to nmt (long tun)
                    for trips <5km"""
    nmt = phen_setting["NMT"] + 1
    # find increase in share of nmt due to mode switch
    nmt_share = 3 - (nmt["bus_share"] + nmt["tw_share"])
    # Expand dims to include parameter
    nmt_share = nmt_share.expand_dims(
        parameter={"nmt_share": len(nmt_share)}
    )
    # Concat
    nmt = computations.concat(nmt, nmt_share)
    # Expand dimensions to include trip distance
    # nmt = nmt.expand_dims(trip_dist={"00_01": len(nmt), "02_05": len(nmt)})
    # nmt = nmt.transpose("parameter", "trip_dist", "area_type", "y")

    return nmt


def PT(i) -> Quantity:
    """Public transport infrastructure:
    representation: increase in pt share in short run
                    mode shift from 2-wheelers and LDVs in long run"""

    pt = phen_setting["PT"].drop("trip_dist") + 1
    # Delete pt_share values for y> 2030 to avoid double counting
    pt = pt[
        ~(
            (pt.index.get_level_values("parameter") == "bus_share")
            & (pt.index.get_level_values("y") > 2030)
        )
    ]
    # Calculate new bus shares due to shift from two-wheelers and LDVS
    bus_share = (
        1
        - pt[
            pt.index.get_level_values("parameter").isin(
                ["tw_share", "ldv_share"]
            )
        ]
    )
    bus_share = 1 + computations.group_sum(
        bus_share, group=["area_type", "y"], sum="parameter"
    )
    bus_share = bus_share.expand_dims(
        parameter={"bus_share": len(bus_share)}
    )
    bus_share = bus_share[bus_share.index.get_level_values("y") > 2030]

    pt = computations.concat(pt, bus_share)

    return pt


def TODU(i) -> Quantity:
    """Transit orineted development in urban areas-
    representations: decrease in pdt,
                     increase in nmt and pt shares,
                     decrease in vehicle ownership in long run"""
    todu = phen_setting["TOD"].drop("trip_dist") + 1
    return todu


def RH(i) -> Quantity:
    """Ride haling-
    representation: increase in vdt associated with taxis,
                    decrease in nmt and pt share,
                    decrease in vehicle ownerhsip,
                    changes in average LDV occupancy"""
    rh = phen_setting["RH"].drop("trip_dist") + 1
    # assume a time series increase in taxi pdt in base
    taxi_pdt = {
        "area_type": [
            "large_city",
            "large_city",
            "large_city",
            "city",
            "city",
            "city",
            "town",
            "town",
            "town",
            "rural",
            "rural",
            "rural",
        ],
        "y": [
            2011,
            2030,
            2050,
            2011,
            2030,
            2050,
            2011,
            2030,
            2050,
            2011,
            2030,
            2050,
        ],
        "value": [
            0.01,
            0.07,
            0.10,
            0.01,
            0.05,
            0.07,
            0.01,
            0.03,
            0.05,
            0.00,
            0.005,
            0.01,
        ],
    }
    taxi_pdt = pd.DataFrame(taxi_pdt)
    years_list = list(range(2051, 2101))
    new_rows = pd.DataFrame(
        {
            "area_type": np.nan,
            "y": years_list,
            "value": 0.0,
        }
    )

    # Concatenate the existing DataFrame and the new DataFrame
    taxi_pdt = pd.concat(
        [taxi_pdt, pd.DataFrame(new_rows, columns=taxi_pdt.columns)],
        ignore_index=True,
    )
    # Convert to genno quantity
    taxi_pdt = Quantity(taxi_pdt.set_index(["area_type", "y"])["value"])
    # Forward fill values
    taxi_pdt = taxi_pdt.ffill("y")
    # list of years for interpolation
    years = list(range(2011, 2101))
    # Interpolate values
    taxi_pdt = computations.interpolate(taxi_pdt, dict(y=years))
    # convert pdt to vdt
    # occupancy of private cars- https://www.adb.org/sites/default/files/project-documents/39578-reg-dpta.pdf
    occ_ldv = 2.1
    occ_taxi = 1.5  # assumption
    # calculate vehicle distance travelled (vdt)
    taxi_vdt = taxi_pdt / occ_taxi
    # Calculate new vdt due to Ride hailing
    rh_vdt = computations.mul(taxi_vdt, rh["vdt"])
    # Calculate pdt associated ith ride hailing
    rh_pdt = rh_vdt * occ_taxi
    # Calulate share of private LDV in base
    ldv_pdt = 1 - taxi_pdt
    # Calulate total share of LDV = private LDV in base + ride hailing
    ldv_share = ldv_pdt + rh_pdt
    # Calculate weighted occupancy
    occ = ldv_share / (rh_vdt + (ldv_pdt / occ_ldv))
    # Expand dims to concat
    occ = occ.expand_dims(parameter={"ldv_occ": len(occ)})
    ldv_share = ldv_share.expand_dims(
        parameter={"ldv_share": len(ldv_share)}
    )
    rh = computations.concat(rh, ldv_share, occ)
    return rh


def RS(i) -> Quantity:
    """Ride sharing-
    representation: decrease in vdt associated with taxis in short run
                    but rebound effect in long run,
                    changes in average LDV occupancy"""
    rs = phen_setting["RS"].drop("trip_dist") + 1
    # assume a time series increase in taxi shares in base
    taxi_pdt = {
        "area_type": [
            "large_city",
            "large_city",
            "large_city",
            "city",
            "city",
            "city",
            "town",
            "town",
            "town",
            "rural",
            "rural",
            "rural",
        ],
        "y": [
            2011,
            2030,
            2050,
            2011,
            2030,
            2050,
            2011,
            2030,
            2050,
            2011,
            2030,
            2050,
        ],
        "value": [
            0.01,
            0.07,
            0.10,
            0.01,
            0.05,
            0.07,
            0.01,
            0.03,
            0.05,
            0.00,
            0.005,
            0.01,
        ],
    }
    taxi_pdt = pd.DataFrame(taxi_pdt)
    years_list = list(range(2051, 2101))
    new_rows = pd.DataFrame(
        {
            "area_type": np.nan,
            "y": years_list,
            "value": 0.0,
        }
    )

    # Concatenate the existing DataFrame and the new DataFrame
    taxi_pdt = pd.concat(
        [taxi_pdt, pd.DataFrame(new_rows, columns=taxi_pdt.columns)],
        ignore_index=True,
    )
    # Convert to genno quantity
    taxi_pdt = Quantity(taxi_pdt.set_index(["area_type", "y"])["value"])
    # Forward fill values
    taxi_pdt = taxi_pdt.ffill("y")
    # list of years for interpolation
    years = list(range(2011, 2101))
    # Interpolate values
    taxi_pdt = computations.interpolate(taxi_pdt, dict(y=years))
    # convert pdt to vdt
    # occupancy of private cars- https://www.adb.org/sites/default/files/project-documents/39578-reg-dpta.pdf
    occ_ldv = 2.1
    occ_taxi = 1.5  # assumption
    occ_rs = 1.9  # assumption
    # Calculate vdt
    taxi_vdt = taxi_pdt / occ_taxi
    # Calculate change in vdt due to ride sharing
    rs_vdt = computations.mul(taxi_vdt, rs["vdt"])
    # Calculate pdt associated with ride sharing
    rs_pdt = rs_vdt * occ_rs

    ldv_pdt = 1 - taxi_pdt

    ldv_share = ldv_pdt + rs_pdt
    # Calculate weighted occupancy
    occ = ldv_share / (rs_vdt + (ldv_pdt / occ_ldv))
    # Expand dims to concat
    occ = occ.expand_dims(parameter={"ldv_occ": len(occ)})
    ldv_share = ldv_share.expand_dims(
        parameter={"ldv_share": len(ldv_share)}
    )
    rs = computations.concat(rs, ldv_share, occ)
    return rs


def MAAS(i) -> Quantity:
    """Mobility as a Service-
    representation: Higher vdt due to ride sharing and ride hailing,
                    changes in average LDV occupancy"""
    maas = 1.25 * RH(i)["vdt"] * RS(i)["vdt"]
    maas = maas.expand_dims(parameter={"vdt": len(maas)})
    ldv_occ = (RH(i)["ldv_occ"] + RS(i)["ldv_occ"]) / 2
    ldv_occ = ldv_occ.expand_dims(parameter={"ldv_occ": len(ldv_occ)})
    ldv_share = 1.25 * RH(1)["ldv_share"] * RS(1)["ldv_share"]
    ldv_share = ldv_share.expand_dims(parameter={"ldv_share": len(maas)})
    maas = computations.concat(maas, ldv_occ, ldv_share)
    return maas


# define a new class to call the above functions in each scenario
class Scenario:
    def __init__(self, name):
        self.name = name
        self.phen_status = {}
        self.quantities = {}

    def flags(self, phen_name, status):
        self.phen_status[phen_name] = status

    def call_function(self, i, m):
        for phen_name, status in self.phen_status.items():
            if status:
                function = globals()[phen_name]
                if phen_name == "HSR":
                    result = function(i, m)
                else:
                    result = function(i)
                self.quantities[phen_name] = result


# Define  scenarios
NP = Scenario("NP")
BP = Scenario("BP")
TOD = Scenario("TOD")
TECH = Scenario("TECH")
Base = Scenario("Base")

# define i
i = 0
# Deinfm- mode share trajectory - {1,2,3}
for m in range(1, 4):
    for j in range(0, 2):
        # populate scenario with phenomenon status and quantities
        for p_id, flags in (
            ("URB", [False, False, True, False, False]),
            ("AV", [False, False, False, True, False]),
            ("ENV", [False, False, True, True, False]),
            ("HSL", [True, True, True, True, False]),
            ("NMT", [True, True, True, False, False]),
            ("RH", [False, True, False, False, False]),
            ("RS", [False, True, False, False, False]),
            ("PT", [True, True, True, True, False]),
            ("TODU", [True, True, True, False, False]),
            ("ELIFE", [False, False, False, True, False]),
            ("HSR", [False, True, False, True, False]),
            ("MAAS", [False, False, True, True, False]),
        ):
            for flag, scen in zip(flags, [NP, BP, TOD, TECH, Base]):
                scen.flags(p_id, flag)
                scen.call_function(i, m)

        def scen_info(scen: Scenario, j, m):
            """Return various calculated parameters for each scenario."""
            # Default total PDT (mode, n, area_type, trip_dist, y): values
            # in `pc_dt` (generated by demand_model.py)
            total_pdt = Daily_travel(2, j, m)
            ldt_factor = 1

            # Calculate total_pdt given any impacts from `pdt_dt` factors
            for phen_id, value in scen.quantities.items():
                # print(f"{phen_id = }")

                y = value.get("pdt", 1.0)
                # Trip rate; impacts total PDT
                trip_rate = value.get(
                    "trip_rate", 1.0
                )  # Non-default for ELF
                # Adjust LDV, NMT, and PT shares
                ldv_share = value.get(
                    "ldv_share", 1.0
                )  # Non-default for RH
                nmt_share = value.get(
                    "nmt_share", 1.0
                )  # Non-default for TODU
                bus_share = value.get(
                    "bus_share", 1.0
                )  # Non-default for RH
                tw_share = value.get(
                    "tw_share", 1.0
                )  # Non-default for NMT, PT
                # NB not currently used
                ldv_occ = value.get(
                    "ldv_occ", 2.1
                )  # Non-default for RS, RH
                ev_share = value.get("ev_share", 1)  # Non-default for ENV
                ldv_own = value.get(
                    "ldv_own", 1
                )  # non-default for HSL, TODU

                if phen_id in ["URB", "TODU"]:
                    # Phenomena where the `pdt` value has multiplicative effect
                    pdt_mul = y
                else:
                    pdt_mul = Quantity(1.0)

                for k in total_pdt:
                    # print(f"{k = }")
                    # print(f"{total_pdt[k] = }")
                    # print(f"{pdt_mul = }")
                    # print(pdt_mul.to_frame().to_string())
                    # print(f"{trip_rate = }")
                    total_pdt[k] *= pdt_mul * trip_rate

                total_pdt["ldv_share"] *= ldv_share
                total_pdt["bus_share"] *= bus_share
                total_pdt["nmt_share"] *= nmt_share
                total_pdt["tw_share"] *= tw_share

                if phen_id in "ELIFE":
                    print(f"{phen_id=}")
                    factor = 0.9
                else:
                    factor = 1

                ldt_factor *= factor

                if phen_id in "HSR":
                    print(f"{phen_id=}")
                    ldt = value * ldt_factor
                    break
            else:
                ldt = Long_dist_travel(2, m)[0] * ldt_factor

            ldt = ldt.expand_dims(
                trip_dist={"None": len(ldt)},
                area_type={"Long_dist": len(ldt)},
            )

            daily_travel_pdt = Key(
                "daily_travel_pdt",
                ["mode", "area_type", "trip_dist", "y", "n"],
            )

            for i in total_pdt.keys():
                daily_travel_pdt = computations.concat(
                    daily_travel_pdt, total_pdt[i]
                )

            total_pdt = computations.concat(daily_travel_pdt, ldt * 1000)

            computations.write_report(
                total_pdt, f"total_pdt_{scen.name}_{j}_{m}.csv"
            )

            return total_pdt

        scen_info(NP, j, m)
        scen_info(BP, j, m)
        scen_info(TOD, j, m)
        scen_info(TECH, j, m)
        scen_info(Base, j, m)
