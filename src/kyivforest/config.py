"""Constants for the Kyiv region demonstration.

All data sources are open. No National Forest Inventory plot coordinates are used.
"""

# Demonstration area around the Boyarka Forest Research Station (Kyiv region), about 21 x 22 km.
AOI_BBOX = [30.15, 50.22, 30.45, 50.42]  # lon_min, lat_min, lon_max, lat_max (WGS 84)
AOI_NAME = "Боярська ЛДС, демонстраційна ділянка"

UTM_EPSG = 32636           # WGS 84 / UTM zone 36N (Kyiv region)
BLOCK_SIZE_M = 2000        # spatial block for cross-validation, m

# Imagery window for predictors (leaf-on season)
S2_START, S2_END = "2022-06-01", "2022-08-31"
S1_START, S1_END = "2022-06-01", "2022-08-31"

# GEDI L4A (footprint biomass) window
GEDI_START, GEDI_END = "2019-04-01", "2023-03-31"

# Biomass to carbon and CO2 (IPCC default carbon fraction 0.47 for forest biomass)
CARBON_FRACTION = 0.47
CO2_PER_C = 44.0 / 12.0
O2_PER_C = 32.0 / 12.0

# Deterministic acceptance thresholds used by the checks (not universal norms)
CHECKS = {
    "min_samples": 300,          # GEDI footprints after QA
    "agbd_range": (0.0, 600.0),  # Mg/ha, plausible range for temperate forests
    "max_nan_share": 0.05,
    "min_r2_spatial": 0.20,
    "max_cv_gap": 0.25,          # R2(random CV) - R2(spatial CV)
}

GEE_PROJECT = "gen-lang-client-0200925587"
