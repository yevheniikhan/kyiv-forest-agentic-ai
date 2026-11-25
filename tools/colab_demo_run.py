# Demonstration run used to produce the figures and numbers for the workshop slides.
# Executed in Colab after ee.Initialize(); writes results to /content/results/.
import os, sys, json, time
sys.path.insert(0, '/content/kyiv-forest-agentic-ai/src')
import numpy as np, pandas as pd, ee
from kyivforest import config as C, ee_data as D, validate as V, services as S

os.makedirs('/content/results', exist_ok=True)
t0 = time.time()
fc = D.training_samples(max_n=4000)
rows = fc.map(lambda f: f.set({'lon': f.geometry().coordinates().get(0),
                               'lat': f.geometry().coordinates().get(1)})).getInfo()['features']
df = pd.DataFrame([r['properties'] for r in rows]).dropna()
FEATURES = [c for c in df.columns if c not in ('agbd', 'lon', 'lat')]
df = V.add_blocks(df)

res = []
for kind in ('rf', 'gbm'):
    for scheme in ('random', 'spatial'):
        oof, m = V.cross_validate(df, FEATURES, kind=kind, scheme=scheme)
        res.append({'model': kind, 'cv': scheme, **m})
        df[f'pred_{kind}_{scheme}'] = oof
imp = V.importance(df, FEATURES)

agb = D.ee_rf_map(fc, FEATURES)
st = D.area_stats(agb).getInfo()
l4b = ee.Image('LARSE/GEDI/GEDI04_B_002').select('MU').updateMask(D.forest_mask())
l4b_mean = l4b.reduceRegion(ee.Reducer.mean(), D.aoi(), 1000).getInfo()['MU']
pools = S.biomass_to_pools(st['agb_tot_sum'])
vis = {'min': 0, 'max': 350, 'palette': ['f2efe6', 'c9d8c5', '8fb49a', '4f8a6f', '14524a', '0c3a34']}
bg = ee.Image(1).visualize(palette=['ffffff']).clip(D.aoi())
# previews in UTM 36N (EPSG:32636) so that proportions are correct
s2rgb = D.preview_url(D.sentinel2().visualize(bands=['B4', 'B3', 'B2'], min=0.0, max=0.15))
thumb = None  # AGB preview: export GeoTIFF in UTM 36N with D.export_map(agb)

summary = {
    'n_samples': int(len(df)), 'n_blocks': int(df.block.nunique()),
    'agbd_mean': float(df.agbd.mean()), 'agbd_median': float(df.agbd.median()),
    'cv': res, 'importance': imp.round(4).to_dict(),
    'forest_ha': st['ha_sum'], 'mean_agb_map': st['agb_tot_sum'] / st['ha_sum'],
    'agb_total_Mg': st['agb_tot_sum'], 'carbon_Mg': pools['carbon_Mg'], 'co2_Mg': pools['co2_Mg'],
    'o2_equiv_Mg': pools['o2_Mg'], 'l4b_mean': l4b_mean,
    'ratio_to_region': S.compare_with_region(st['agb_tot_sum'] / st['ha_sum']),
    'thumb_agb': thumb, 'thumb_s2': s2rgb, 'seconds': round(time.time() - t0, 1),
}
json.dump(summary, open('/content/results/summary.json', 'w'), ensure_ascii=False, indent=1)
df[['lon', 'lat', 'block', 'agbd'] + [c for c in df.columns if c.startswith('pred_')]].round(2).to_csv(
    '/content/results/oof.csv', index=False)
print(json.dumps({k: v for k, v in summary.items() if not k.startswith('thumb')}, ensure_ascii=False, indent=1))
