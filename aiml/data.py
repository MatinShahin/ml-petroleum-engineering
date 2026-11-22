"""Loaders for the three Part I course datasets.

asphaltene.csv     Asphaltene precipitation (regression, 167 tests)
eor_screening.csv  EOR method screening (classification, 455 field projects)
vfm_well_rates.csv Virtual flow metering (regression, 2,100 well tests)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

ASPH_FEATURES = ["temperature_K", "alkane_carbon_number", "dilution_ratio"]
ASPH_TARGET = "asphaltene_wt_pct"

EOR_FEATURES = ["permeability_md", "depth_ft", "api_gravity", "viscosity_cp", "temperature_F", "oil_saturation_pct"]
EOR_TARGET = "eor_method"
EOR_CLASSES = ["CO2", "Combustion", "Hydrocarbon", "Nitrogen", "Polymer", "Steam Injection"]

VFM_FEATURES = ["choke_in", "bhp_psi", "whp_psi", "wht_F", "psep_psi", "tsep_F"]
VFM_TARGETS = ["oil_rate_stb_d", "water_rate_bbl_d", "gas_rate_mmscf_d"]


def _path(name, folder=None):
    return os.path.join(folder or DATA, name)


def asphaltene_suspect(df):
    """The 17 tests at 332 K with n-C6 to n-C8 report 10x more precipitate than every other series.
    They are kept in the file but flagged: check them before modeling (Lab 0)."""
    return (df["temperature_K"] == 332) & (df["alkane_carbon_number"] <= 8)


def load_asphaltene(drop_suspect=False, add_features=False, folder=None):
    df = pd.read_csv(_path("asphaltene.csv", folder))
    df["suspect"] = asphaltene_suspect(df)
    if drop_suspect:
        df = df[~df.suspect].reset_index(drop=True)
    if add_features:
        df["log_dilution"] = np.log(df["dilution_ratio"])
        df["inv_temperature"] = 1000.0 / df["temperature_K"]
    df["series"] = df["temperature_K"].astype(str) + "K_C" + df["alkane_carbon_number"].astype(str)
    return df


def load_eor(log_viscosity=False, folder=None):
    df = pd.read_csv(_path("eor_screening.csv", folder))
    if log_viscosity:
        df["log_viscosity"] = np.log10(df["viscosity_cp"].clip(lower=0.1))
        df["log_permeability"] = np.log10(df["permeability_md"].clip(lower=0.1))
    return df


def load_vfm(folder=None):
    """Each physical well test appears once per separator pressure (Psep does not change the well),
    so the 2,100 rows hold fewer than 1,000 different tests. `case_id` groups the copies: split by it."""
    df = pd.read_csv(_path("vfm_well_rates.csv", folder))
    key = df[["choke_in", "bhp_psi", "whp_psi", "wht_F", "tsep_F"]].round(2).astype(str).agg("|".join, axis=1)
    df["case_id"] = pd.factorize(key)[0]
    df["liquid_rate_bbl_d"] = df["oil_rate_stb_d"] + df["water_rate_bbl_d"]
    return df

