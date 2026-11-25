"""Earth Engine data access: predictor stack, GEDI L4A samples, biomass map.

Requires an initialised `ee` session (see notebooks/00_setup.ipynb).
"""
import ee

from . import config as C

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def aoi():
    return ee.Geometry.Rectangle(C.AOI_BBOX)


def _mask_s2(img):
    # Cloud Score+ (cs_cdf >= 0.6 treated as clear)
    return img.updateMask(img.select("cs_cdf").gte(0.6))


def sentinel2(region=None):
    region = region or aoi()
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region).filterDate(C.S2_START, C.S2_END)
           .linkCollection(csp, ["cs_cdf"]).map(_mask_s2))
    med = col.select(S2_BANDS).median().divide(10000)
    ndvi = med.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = med.normalizedDifference(["B8", "B12"]).rename("NBR")
    ndre = med.normalizedDifference(["B8A", "B5"]).rename("NDRE")
    return med.addBands([ndvi, nbr, ndre])


def sentinel1(region=None):
    region = region or aoi()
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(region).filterDate(C.S1_START, C.S1_END)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .select(["VV", "VH"]))
    med = col.median()
    return med.addBands(med.select("VH").subtract(med.select("VV")).rename("VH_VV"))


def terrain():
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().rename("elev")
    dem = dem.setDefaultProjection("EPSG:4326", None, 30)
    return dem.addBands(ee.Terrain.slope(dem).rename("slope"))


def forest_mask():
    wc = ee.ImageCollection("ESA/WorldCover/v200").first()
    return wc.eq(10).rename("forest")


def predictors(region=None):
    return sentinel2(region).addBands(sentinel1(region)).addBands(terrain()).float()


def gedi_agbd():
    def qa(im):
        return (im.updateMask(im.select("l4_quality_flag").eq(1))
                  .updateMask(im.select("degrade_flag").eq(0)))
    col = (ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY")
           .filterDate(C.GEDI_START, C.GEDI_END).map(qa).select("agbd"))
    return col.mosaic().rename("agbd")


def training_samples(region=None, max_n=4000, seed=42):
    """GEDI footprints inside forest (ESA WorldCover) with predictor values."""
    region = region or aoi()
    stack = gedi_agbd().addBands(predictors(region)).updateMask(forest_mask())
    return (stack.sample(region=region, scale=25, geometries=True, tileScale=4,
                         dropNulls=True, seed=seed)
                 .filter(ee.Filter.rangeContains("agbd", *C.CHECKS["agbd_range"]))
                 .randomColumn("rnd", seed).sort("rnd").limit(max_n)
                 .select(stack.bandNames()))


def ee_rf_map(samples, bands, region=None, n_trees=200):
    """Train an Earth Engine random forest on the samples and map AGB in forest."""
    region = region or aoi()
    rf = (ee.Classifier.smileRandomForest(numberOfTrees=n_trees, minLeafPopulation=3, seed=1)
            .setOutputMode("REGRESSION")
            .train(features=samples, classProperty="agbd", inputProperties=bands))
    agb = predictors(region).select(bands).classify(rf).rename("agb")
    return agb.updateMask(forest_mask()).clip(region)


def area_stats(agb, region=None, scale=30):
    """Mean AGB (Mg/ha), forest area (ha) and total AGB (Mg) inside the region."""
    region = region or aoi()
    px_ha = ee.Image.pixelArea().divide(1e4)
    st = (agb.addBands(px_ha.updateMask(agb.mask()).rename("ha"))
             .addBands(agb.multiply(px_ha).rename("agb_tot"))
             .reduceRegion(ee.Reducer.mean().combine(ee.Reducer.sum(), sharedInputs=True),
                           region, scale, maxPixels=1e10, tileScale=4))
    return st


def export_map(agb, folder="Kyiv_LandShift_AgenticAI_2026", name="boyarka_agb_rf_30m"):
    """GeoTIFF in WGS 84 / UTM 36N (EPSG:32636), 30 m. Areas and maps for print use this projection;
    interactive maps (geemap, Code Editor) are Web Mercator and serve only for viewing."""
    task = ee.batch.Export.image.toDrive(image=agb.toFloat(), description=name, folder=folder,
                                         region=aoi(), scale=30, crs=f"EPSG:{C.UTM_EPSG}", maxPixels=1e10)
    task.start()
    return task


def preview_url(image, dimensions=640):
    """PNG preview in UTM 36N, so that proportions are not distorted."""
    utm = ee.Projection(f"EPSG:{C.UTM_EPSG}")
    reg = aoi().transform(utm, 1).bounds(1, utm)
    return image.getThumbURL({"region": reg, "crs": f"EPSG:{C.UTM_EPSG}", "dimensions": dimensions, "format": "png"})
