"""Offline tests: validation and agent loop on synthetic data (no Earth Engine)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from kyivforest import validate as V, services as S
from kyivforest.agent import Registry, CheckResult, run


def synthetic(n=600, seed=0):
    r = np.random.default_rng(seed)
    lon, lat = r.uniform(30.15, 30.45, n), r.uniform(30.22 + 20, 50.42, n)
    ndvi = r.uniform(.4, .9, n); vh = r.normal(-15, 2, n)
    agbd = 300 * (ndvi - .35) + 5 * (vh + 15) + 40 * np.sin(lon * 40) + r.normal(0, 25, n)
    return pd.DataFrame({"lon": lon, "lat": lat, "NDVI": ndvi, "VH": vh, "agbd": agbd.clip(1)})


def test_cv():
    df = V.add_blocks(synthetic())
    _, mr = V.cross_validate(df, ["NDVI", "VH"], scheme="random")
    _, ms = V.cross_validate(df, ["NDVI", "VH"], scheme="spatial")
    assert 0 < ms["r2"] <= mr["r2"] + 0.05 and ms["n"] == len(df)


def test_agent():
    reg = Registry()
    @reg.register("load_samples", "synthetic")
    def a(state):
        state["df"] = V.add_blocks(synthetic()); return {"n": len(state["df"])}
    @reg.register("train_validate", "cv")
    def b(state, scheme="spatial"):
        return V.cross_validate(state["df"], ["NDVI", "VH"], scheme=scheme)[1]
    @reg.register("map_and_totals", "totals")
    def c(state):
        return {"mean_agb": 150.0, "forest_ha": 1000.0}
    @reg.register("compare_region", "ratio")
    def d(state):
        return {"ratio": S.compare_with_region(150)}
    checks = {"train_validate": lambda s, o: [CheckResult("r2", o["r2"] > 0.2, f"{o['r2']:.2f}")]}
    st = run("test", reg, checks, auto=True, log_path="/tmp/run_log.json")
    assert "compare_region" in st


if __name__ == "__main__":
    test_cv(); test_agent(); print("all passed")
