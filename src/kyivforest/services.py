"""Conversion of biomass to carbon, CO2 and O2, and comparison with regional reference values.

Regional reference values are taken from the scanned monograph chapter on Kyiv region forests
(forest inventory database of «Укрдержліспроект» as of 01.01.2011). They describe the whole
region and serve only as an order-of-magnitude check for the demonstration area.
"""
from . import config as C

REGION_2011 = {
    "forest_covered_ha": 655_415.9,
    "phytomass_Mt": 111.0,
    "phytomass_t_ha": 180.0,
    "carbon_plant_Mt": 61.8,
    "carbon_seq_kt_yr": 3_153.4,
    "oxygen_kt_yr": 9_126.0,
    "o2_net_t_ha_yr": 6.8,
}


def biomass_to_pools(agb_Mg, cf=C.CARBON_FRACTION):
    """Above-ground biomass (Mg) to carbon, CO2-equivalent and stoichiometric O2 (Mg)."""
    c = agb_Mg * cf
    return {"agb_Mg": agb_Mg, "carbon_Mg": c, "co2_Mg": c * C.CO2_PER_C, "o2_Mg": c * C.O2_PER_C}


def o2_from_sequestration(c_seq_Mg_yr):
    """Net O2 release that corresponds stoichiometrically to net carbon uptake."""
    return c_seq_Mg_yr * C.O2_PER_C


def compare_with_region(mean_agb_Mg_ha):
    """Ratio of the demo-area mean AGB to the regional mean phytomass density.

    Phytomass in the monograph includes stems, branches, foliage, roots and understorey,
    so the ratio is expected to be below 1.
    """
    return mean_agb_Mg_ha / REGION_2011["phytomass_t_ha"]
