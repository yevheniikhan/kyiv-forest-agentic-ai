/**
 * Надземна біомаса лісу на демонстраційній ділянці Боярської ЛДС (Київська область).
 * Above-ground forest biomass, Boyarka Forest Research Station demo area (Kyiv region).
 *
 * Google Earth Engine Code Editor. Відкриті дані, без координат пробних площ НІЛ.
 * Еталон: відбитки GEDI L4A (AGBD, Мг/га). Предиктори: Sentinel-2, Sentinel-1, Copernicus DEM.
 * Модель: random forest (регресія). Валідація: просторові блоки 2 км, 5 груп.
 *
 * Скрипт є демонстрацією методу, а не офіційною оцінкою запасів.
 */

// ---------- 1. Ділянка і параметри ----------
var aoi = ee.Geometry.Rectangle([30.15, 50.22, 30.45, 50.42]);
var S2_START = '2022-06-01', S2_END = '2022-08-31';
var GEDI_START = '2019-04-01', GEDI_END = '2023-03-31';
var BLOCK_DEG = 0.03;      // ≈ 2 км по широті (просторовий блок для крос-валідації)
var N_FOLDS = 5;
var CARBON_FRACTION = 0.47; // частка вуглецю в сухій біомасі (IPCC, типове значення)

Map.centerObject(aoi, 11);
Map.addLayer(ee.Image().paint(aoi, 0, 2), {palette: ['000000']}, 'Демонстраційна ділянка');

// ---------- 2. Маска лісу ----------
var forest = ee.ImageCollection('ESA/WorldCover/v200').first().eq(10).rename('forest');

// ---------- 3. Предиктори ----------
var csp = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi).filterDate(S2_START, S2_END)
  .linkCollection(csp, ['cs_cdf'])
  .map(function (im) { return im.updateMask(im.select('cs_cdf').gte(0.6)); })
  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
  .median().divide(10000);
s2 = s2.addBands([
  s2.normalizedDifference(['B8', 'B4']).rename('NDVI'),
  s2.normalizedDifference(['B8', 'B12']).rename('NBR'),
  s2.normalizedDifference(['B8A', 'B5']).rename('NDRE')
]);

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi).filterDate(S2_START, S2_END)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH']).median();
s1 = s1.addBands(s1.select('VH').subtract(s1.select('VV')).rename('VH_VV'));

var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic()
  .setDefaultProjection('EPSG:4326', null, 30).rename('elev');
var terrain = dem.addBands(ee.Terrain.slope(dem).rename('slope'));

var stack = s2.addBands(s1).addBands(terrain).float();
var BANDS = stack.bandNames();

// ---------- 4. Еталон GEDI L4A ----------
var gedi = ee.ImageCollection('LARSE/GEDI/GEDI04_A_002_MONTHLY')
  .filterDate(GEDI_START, GEDI_END)
  .map(function (im) {
    return im.updateMask(im.select('l4_quality_flag').eq(1))
             .updateMask(im.select('degrade_flag').eq(0));
  })
  .select('agbd').mosaic();

var samples = gedi.addBands(stack).updateMask(forest)
  .sample({region: aoi, scale: 25, geometries: true, tileScale: 4, dropNulls: true, seed: 42})
  .filter(ee.Filter.lte('agbd', 600))
  .limit(4000);

// Номер просторового блоку і групи (fold) для кожного відбитка
samples = samples.map(function (f) {
  var c = f.geometry().coordinates();
  var bx = ee.Number(c.get(0)).divide(BLOCK_DEG).floor();
  var by = ee.Number(c.get(1)).divide(BLOCK_DEG).floor();
  var fold = bx.multiply(7).add(by.multiply(13)).mod(N_FOLDS);
  return f.set({fold: fold});
});
print('Кількість відбитків GEDI після контролю якості:', samples.size());

// ---------- 5. Просторова крос-валідація ----------
function trainRF(fc) {
  return ee.Classifier.smileRandomForest({numberOfTrees: 200, minLeafPopulation: 3, seed: 1})
    .setOutputMode('REGRESSION')
    .train({features: fc, classProperty: 'agbd', inputProperties: BANDS});
}

var oof = ee.FeatureCollection(ee.List.sequence(0, N_FOLDS - 1).map(function (k) {
  var train = samples.filter(ee.Filter.neq('fold', k));
  var test = samples.filter(ee.Filter.eq('fold', k));
  return test.classify(trainRF(train), 'pred');
})).flatten();

var stats = oof.map(function (f) {
  var e = ee.Number(f.get('pred')).subtract(f.get('agbd'));
  return f.set({err: e, err2: e.pow(2)});
});
var n = stats.size();
var meanObs = stats.aggregate_mean('agbd');
var rmse = ee.Number(stats.aggregate_mean('err2')).sqrt();
var bias = stats.aggregate_mean('err');
var ssTot = stats.map(function (f) {
  return f.set('d2', ee.Number(f.get('agbd')).subtract(meanObs).pow(2));
}).aggregate_sum('d2');
var r2 = ee.Number(1).subtract(ee.Number(stats.aggregate_sum('err2')).divide(ssTot));
print('Просторова CV: n, R², RMSE (Мг/га), rRMSE (%), зміщення (Мг/га)',
      n, r2, rmse, rmse.divide(meanObs).multiply(100), bias);

print(ui.Chart.feature.byFeature(oof.limit(1500), 'agbd', 'pred')
  .setChartType('ScatterChart')
  .setOptions({title: 'GEDI L4A проти прогнозу (просторова CV)',
               hAxis: {title: 'GEDI AGBD, Мг/га'}, vAxis: {title: 'Прогноз, Мг/га'},
               pointSize: 2, legend: 'none', colors: ['14524A']}));

// ---------- 6. Карта біомаси і запас вуглецю ----------
var rf = trainRF(samples);
var agb = stack.select(BANDS).classify(rf).rename('agb').updateMask(forest).clip(aoi);
var vis = {min: 0, max: 350, palette: ['f2efe6', 'c9d8c5', '8fb49a', '4f8a6f', '14524a', '0c3a34']};
Map.addLayer(agb, vis, 'Надземна біомаса, Мг/га (RF)');
Map.addLayer(gedi.updateMask(forest).clip(aoi), vis, 'GEDI L4A, відбитки', false);

// Порівняння з грідованим продуктом GEDI L4B (1 км, середня AGBD)
var l4b = ee.Image('LARSE/GEDI/GEDI04_B_002').select('MU').clip(aoi);
Map.addLayer(l4b, vis, 'GEDI L4B, 1 км', false);

var ha = ee.Image.pixelArea().divide(1e4);
var tot = agb.multiply(ha).rename('agb_t').addBands(ha.updateMask(agb.mask()).rename('ha'))
  .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi, scale: 30, maxPixels: 1e10, tileScale: 4});
var agbT = ee.Number(tot.get('agb_t'));
var forestHa = ee.Number(tot.get('ha'));
print('Площа лісу (WorldCover), га:', forestHa);
print('Середня надземна біомаса, Мг/га:', agbT.divide(forestHa));
print('Надземний запас вуглецю, тис. т C:', agbT.multiply(CARBON_FRACTION).divide(1000));
print('Еквівалент CO₂, тис. т:', agbT.multiply(CARBON_FRACTION).multiply(44 / 12).divide(1000));

// ---------- 7. Експорт (запускає користувач у вкладці Tasks) ----------
Export.image.toDrive({image: agb, description: 'boyarka_agb_rf_30m', folder: 'Kyiv_LandShift_AgenticAI_2026',
                      region: aoi, scale: 30, crs: 'EPSG:32636', maxPixels: 1e10});
Export.table.toDrive({collection: oof, description: 'boyarka_gedi_spatial_cv',
                      folder: 'Kyiv_LandShift_AgenticAI_2026', fileFormat: 'CSV'});
