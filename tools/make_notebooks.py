"""Build the Colab notebooks from cell lists (run: python tools/make_notebooks.py)."""
import nbformat as nbf
from pathlib import Path

REPO = "https://github.com/yevheniikhan/kyiv-forest-agentic-ai"
OUT = Path(__file__).resolve().parents[1] / "notebooks"


def badge(name):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/yevheniikhan/kyiv-forest-agentic-ai/blob/main/notebooks/{name})")


SETUP = f"""# Код репозиторію і бібліотеки
import os, sys
if not os.path.exists('/content/kyiv-forest-agentic-ai'):
    !git clone -q {REPO} /content/kyiv-forest-agentic-ai
sys.path.insert(0, '/content/kyiv-forest-agentic-ai/src')
!pip -q install geemap pyproj

import ee
PROJECT = 'gen-lang-client-0200925587'   # замініть на свій проєкт Google Cloud з увімкненим Earth Engine
try:
    ee.Initialize(project=PROJECT)
except Exception:
    ee.Authenticate()                     # вікно авторизації підтверджує користувач
    ee.Initialize(project=PROJECT)
print('Earth Engine:', ee.String('готово').getInfo())"""


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c)
               for c in cells]
    nbf.write(n, OUT / name)


# ---------------------------------------------------------------- 01
nb("01_biomass_gedi_model.ipynb", [
    "md:" + f"""# 01. Модель надземної біомаси за GEDI L4A і Sentinel-1/2

{badge('01_biomass_gedi_model.ipynb')}

**Ділянка:** Боярська лісова дослідна станція та околиці (Київська область), близько 21 × 22 км.
**Еталон:** відбитки лідара GEDI L4A (AGBD, Мг/га; діаметр відбитка 25 м; 2019–2023).
**Предиктори:** медіанний композит Sentinel-2 (червень–серпень 2022), Sentinel-1 VV/VH, рельєф Copernicus DEM.
**Модель:** random forest і градієнтний бустинг.
**Валідація:** випадкова та просторова блокова крос-валідація (блоки 2 × 2 км, 5 груп).

Записник показує, як навчати й чесно перевіряти модель. Відбитки GEDI самі мають похибку моделі L4A,
тому отримані метрики описують узгодженість із GEDI, а не з наземною таксацією.
Координати пробних площ НІЛ не використовуються.""",
    SETUP,
    "md:## 1. Вибірка відбитків GEDI з предикторами",
    """from kyivforest import config as C, ee_data as D, validate as V
import pandas as pd, numpy as np

fc = D.training_samples(max_n=4000)
rows = fc.map(lambda f: f.set({'lon': f.geometry().coordinates().get(0),
                               'lat': f.geometry().coordinates().get(1)})).getInfo()['features']
df = pd.DataFrame([r['properties'] for r in rows])
df = df[(df.agbd >= C.CHECKS['agbd_range'][0]) & (df.agbd <= C.CHECKS['agbd_range'][1])].dropna()
FEATURES = [c for c in df.columns if c not in ('agbd', 'lon', 'lat')]
df = V.add_blocks(df)
print(f'Відбитків: {len(df)}, просторових блоків: {df.block.nunique()}')
df.agbd.describe().round(1)""",
    "md:## 2. Випадкова і просторова крос-валідація\n\nВипадкова CV перемішує сусідні відбитки між навчанням і перевіркою, тому завищує точність. Просторова CV тримає цілі блоки поза навчанням і ближча до умов перенесення моделі на нову територію.",
    """res = []
for kind in ('rf', 'gbm'):
    for scheme in ('random', 'spatial'):
        oof, m = V.cross_validate(df, FEATURES, kind=kind, scheme=scheme)
        res.append({'модель': kind, 'схема CV': scheme, **{k: round(v, 3) for k, v in m.items()}})
        if kind == 'rf' and scheme == 'spatial':
            df['pred_spatial'] = oof
metrics = pd.DataFrame(res)
metrics.to_csv('metrics_cv.csv', index=False)
metrics""",
    """import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
ax[0].scatter(df.agbd, df.pred_spatial, s=4, alpha=.35, color='#14524A')
lim = [0, np.percentile(df.agbd, 99)]
ax[0].plot(lim, lim, color='#9a6b2f', lw=1); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel('GEDI L4A AGBD, Мг/га'); ax[0].set_ylabel('Прогноз RF, Мг/га')
ax[0].set_title('Просторова CV')
imp = V.importance(df, FEATURES).head(10)[::-1]
ax[1].barh(imp.index, imp.values, color='#14524A'); ax[1].set_title('Важливість предикторів (RF)')
plt.tight_layout(); plt.savefig('fig_cv_importance.png', dpi=200); plt.show()""",
    "md:## 3. Карта біомаси і підсумки для ділянки",
    """import geemap
agb = D.ee_rf_map(fc, FEATURES)
st = D.area_stats(agb).getInfo()
forest_ha, agb_tot = st['ha_sum'], st['agb_tot_sum']
print(f'Площа лісу: {forest_ha:,.0f} га; середня AGB: {agb_tot/forest_ha:.1f} Мг/га; '
      f'надземна біомаса: {agb_tot/1e6:.2f} млн т')
vis = {'min': 0, 'max': 350, 'palette': ['f2efe6', 'c9d8c5', '8fb49a', '4f8a6f', '14524a', '0c3a34']}
m = geemap.Map(); m.centerObject(D.aoi(), 11)
m.addLayer(agb, vis, 'AGB, Мг/га'); m.add_colorbar(vis, label='AGB, Мг/га'); m""",
    "md:### Проєкція\n\nІнтерактивна карта geemap показана в Web Mercator лише для перегляду. Площі обчислено через `ee.Image.pixelArea()` (фактична площа пікселя). Для друку і ГІС карту експортують у WGS 84 / UTM 36N (EPSG:32636), 30 м.",
    """from IPython.display import Image
print(D.export_map(agb).status()['state'], '– експорт GeoTIFF у UTM 36N запущено (вкладка Tasks / Google Drive)')
Image(url=D.preview_url(D.sentinel2().visualize(bands=['B4', 'B3', 'B2'], min=0, max=0.15)), width=480)""",
    "md:## 4. Що перевіряє дослідник\n\n- чи не завеликий розрив між випадковою і просторовою CV (ознака перенавчання на локальну автокореляцію);\n- чи немає систематичного зміщення на високих запасах (насичення оптичних індексів);\n- чи узгоджується середнє значення з GEDI L4B і з регіональними даними (записник 02);\n- для звітності потрібні наземні дані: GEDI є допоміжним еталоном.",
])

# ---------------------------------------------------------------- 02
nb("02_carbon_oxygen_services.ipynb", [
    "md:" + f"""# 02. Вуглець, CO₂ і кисень: від карти біомаси до показників екосистемних послуг

{badge('02_carbon_oxygen_services.ipynb')}

Записник переводить надземну біомасу в запас вуглецю, еквівалент CO₂ і стехіометричний еквівалент кисню,
порівнює результат із грідованим продуктом GEDI L4B та з регіональними значеннями для Київської області
(база даних «Укрдержліспроекту» станом на 01.01.2011).

Обмеження: карта охоплює лише надземну деревну біомасу. Корені, підлісок, мертва деревина і ґрунт не враховані,
тому загальний запас вуглецю буде більшим.""",
    SETUP,
    """from kyivforest import config as C, ee_data as D, services as S
import pandas as pd
fc = D.training_samples(max_n=4000)
bands = D.predictors().bandNames().getInfo()
agb = D.ee_rf_map(fc, bands)
st = D.area_stats(agb).getInfo()
forest_ha, agb_tot = st['ha_sum'], st['agb_tot_sum']
pools = S.biomass_to_pools(agb_tot)
l4b = ee.Image('LARSE/GEDI/GEDI04_B_002').select('MU').updateMask(D.forest_mask())
l4b_mean = l4b.reduceRegion(ee.Reducer.mean(), D.aoi(), 1000).getInfo()['MU']
tab = pd.DataFrame({
    'показник': ['площа лісу, га', 'середня AGB (RF), Мг/га', 'середня AGBD GEDI L4B, Мг/га',
                 'надземна біомаса, тис. т', 'вуглець, тис. т C', 'еквівалент CO₂, тис. т',
                 'частка від регіональної щільності фітомаси 2011 р.'],
    'значення': [forest_ha, agb_tot / forest_ha, l4b_mean, pools['agb_Mg'] / 1e3,
                 pools['carbon_Mg'] / 1e3, pools['co2_Mg'] / 1e3, S.compare_with_region(agb_tot / forest_ha)]})
tab['значення'] = tab['значення'].round(2)
tab.to_csv('services_summary.csv', index=False); tab""",
    "md:## Регіональний контекст (Київська область, 2011 р.)\n\n| Показник | Значення |\n|---|---|\n| Площа, вкрита лісовою рослинністю | 655,4 тис. га |\n| Фітомаса | 111,0 млн т (180 т/га) |\n| Вуглець рослинної біомаси | 61,8 млн т |\n| Щорічне депонування вуглецю | 3,15 млн т C/рік |\n| Продукування кисню | 9,13 млн т O₂/рік |\n\nКарта з космічних даних дає просторовий розподіл, а таксаційні дані дають повні запаси за фракціями. Для звітності їх поєднують, а не замінюють одне одним.",
    "md:## Як це пов'язати з Living Lab LandShift у Київській області\n\n- моніторинг змін землекористування за відкритими даними: щорічне оновлення карти біомаси і втрат лісу;\n- зелена інфраструктура й агролісомеліорація: оцінка вуглецю в полезахисних смугах і нових насадженнях;\n- водно-болотні угіддя: маски і тренди за Sentinel-1/2;\n- воркшопи з громадами: карти й таблиці у форматах, які місцеві фахівці можуть перевірити самостійно.",
])

# ---------------------------------------------------------------- 03
nb("03_agentic_workflow.ipynb", [
    "md:" + f"""# 03. Агентний процес під наглядом дослідника

{badge('03_agentic_workflow.ipynb')}

Записник показує будову агентної системи на прикладі оцінювання біомаси:

1. **реєстр інструментів**: типізовані функції з описом, лише їх може викликати планувальник;
2. **планувальник**: пропонує послідовність кроків (за замовчуванням сценарний, без ключа API; можна підключити мовну модель);
3. **детерміновані перевірки**: правила дослідника виконуються після кожного кроку незалежно від мовної моделі;
4. **точки рішення дослідника**: план, дані й результат підтверджує людина;
5. **журнал запуску**: кожен крок, аргументи, перевірки й рішення записуються у JSON.

Роль мовної моделі обмежена плануванням і написанням коду. Методику, пороги перевірок і остаточні висновки визначає дослідник.""",
    SETUP,
    "md:## 1. Інструменти",
    """from kyivforest import config as C, ee_data as D, validate as V, services as S
from kyivforest.agent import Registry, CheckResult, run, scripted_planner, llm_planner
import pandas as pd

reg = Registry()

@reg.register('load_samples', 'відбитки GEDI L4A з предикторами Sentinel-1/2 і рельєфу в межах лісу')
def load_samples(state):
    fc = D.training_samples(max_n=4000)
    rows = fc.map(lambda f: f.set({'lon': f.geometry().coordinates().get(0),
                                   'lat': f.geometry().coordinates().get(1)})).getInfo()['features']
    df = V.add_blocks(pd.DataFrame([r['properties'] for r in rows]).dropna())
    state['df'], state['fc'] = df, fc
    state['features'] = [c for c in df.columns if c not in ('agbd', 'lon', 'lat', 'x', 'y', 'block')]
    return {'n': len(df), 'nan_share': float(df.isna().mean().max()),
            'agbd_min': float(df.agbd.min()), 'agbd_max': float(df.agbd.max())}

@reg.register('train_validate', 'навчання RF і крос-валідація; аргумент scheme: random або spatial')
def train_validate(state, scheme='spatial'):
    _, m = V.cross_validate(state['df'], state['features'], kind='rf', scheme=scheme)
    return m

@reg.register('map_and_totals', 'карта AGB у GEE, площа лісу, середня AGB, запас вуглецю')
def map_and_totals(state):
    st = D.area_stats(D.ee_rf_map(state['fc'], state['features'])).getInfo()
    p = S.biomass_to_pools(st['agb_tot_sum'])
    return {'forest_ha': st['ha_sum'], 'mean_agb': st['agb_tot_sum'] / st['ha_sum'],
            'carbon_kt': p['carbon_Mg'] / 1e3, 'co2_kt': p['co2_Mg'] / 1e3}

@reg.register('compare_region', 'порівняння середньої AGB з регіональною щільністю фітомаси 2011 р.')
def compare_region(state):
    return {'ratio_to_region': S.compare_with_region(state['map_and_totals']['mean_agb'])}

print(reg.catalogue())""",
    "md:## 2. Перевірки дослідника\n\nПороги задано в `config.CHECKS`. Вони відповідають цій демонстрації і не є загальними нормативами.",
    """K = C.CHECKS
def chk_samples(state, o):
    return [CheckResult('обсяг вибірки', o['n'] >= K['min_samples'], f"n = {o['n']} (мін. {K['min_samples']})"),
            CheckResult('пропуски', o['nan_share'] <= K['max_nan_share'], f"{o['nan_share']:.1%}"),
            CheckResult('діапазон AGBD', K['agbd_range'][0] <= o['agbd_min'] and o['agbd_max'] <= K['agbd_range'][1],
                        f"{o['agbd_min']:.0f}–{o['agbd_max']:.0f} Мг/га")]

def chk_validate(state, o):
    out = [CheckResult('R² більше порогу', o['r2'] >= K['min_r2_spatial'], f"R² = {o['r2']:.2f}")]
    if 'train_validate:random' in state and 'train_validate:spatial' in state:
        gap = state['train_validate:random']['r2'] - state['train_validate:spatial']['r2']
        out.append(CheckResult('розрив random/spatial CV', gap <= K['max_cv_gap'], f'ΔR² = {gap:.2f}'))
    return out

def chk_totals(state, o):
    return [CheckResult('правдоподібна середня AGB', 20 <= o['mean_agb'] <= 400, f"{o['mean_agb']:.1f} Мг/га")]

CHECKS = {'load_samples': chk_samples, 'train_validate': chk_validate, 'map_and_totals': chk_totals}""",
    "md:## 3. Запуск\n\nПід час запуску записник зупиняється і чекає відповіді дослідника. Для демонстрації без зупинок задайте `AUTO = True`.",
    """AUTO = None   # None: питати дослідника; True: автоматичне підтвердження (лише тест)
state = run('оцінити надземну біомасу й запас вуглецю лісів демонстраційної ділянки',
            reg, CHECKS, planner=scripted_planner, auto=AUTO, log_path='run_log.json')""",
    """import json
log = json.load(open('run_log.json', encoding='utf-8'))
pd.DataFrame([{k: e.get(k) for k in ('t', 'kind', 'tool', 'name', 'passed', 'question', 'approved')} for e in log])""",
    "md:## 4. Планувальник на основі мовної моделі (необов'язково)\n\nЯкщо у Secrets Colab є `ANTHROPIC_API_KEY`, план можна отримати від моделі. Відповідь розбирається як JSON, і кожен інструмент звіряється з реєстром до виконання.",
    """# from google.colab import userdata
# !pip -q install anthropic
# import anthropic
# client = anthropic.Anthropic(api_key=userdata.get('ANTHROPIC_API_KEY'))
# plan = llm_planner('оцінити біомасу і перевірити модель просторово', reg, client)
# for s in plan: print(s)""",
])
print("ok")
