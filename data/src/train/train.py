
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.loader import load_data
from src.features.calendar import make_calendar_index
from src.models.shape import hub_hourly_shape, basis_stats, volume_bootstrap_table
from src.models.price import shape_monthly_to_hourly, simulate_prices
from src.models.volume import bootstrap_generation
from src.pricing.solve_price import solve_product_prices, expected_generation_monthly
from src.utils.persistence import save_json, save_model

def _read_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_all(config_path: str = "configs/project.yaml"):
    cfg = _read_config(config_path)

    # Load historical + forwards using selected format
    hist, fwd = load_data(cfg)

    start_year = cfg['start_year']
    end_year = cfg['end_year']
    n_scenarios = cfg['n_scenarios']
    seed = cfg.get('random_seed', 42)
    p_level = cfg.get('p_level', 0.75)
    neg_rule = cfg.get('negative_price_rule', 'include')

    # Fit simple components
    shp = hub_hourly_shape(hist)
    bs_rt = basis_stats(hist, price_col_hub='rt_hub', price_col_node='rt_node')
    bs_da = basis_stats(hist, price_col_hub='da_hub', price_col_node='da_node')
    vol_tbl = volume_bootstrap_table(hist)

    # Save artifacts
    save_json({'p_level': p_level, 'negative_price_rule': neg_rule}, "models/config_summary.json")
    save_model(shp, "models/hub_shape.pkl")
    save_model(bs_rt, "models/basis_rt.pkl")
    save_model(bs_da, "models/basis_da.pkl")
    save_model(vol_tbl, "models/volume_bootstrap.pkl")

    # Build calendar
    cal = make_calendar_index(start_year, end_year)

    # Hourly hub forwards
    hh = shape_monthly_to_hourly(fwd, shp, cal)

    # Simulate RT/DA hub & node
    sims_rt = simulate_prices(hh, bs_rt, n_scenarios, seed=seed)

    # Generation scenarios
    rng = np.random.default_rng(seed)
    gen = bootstrap_generation(cal, vol_tbl, n_scenarios, rng)

    # Merge and solve fixed prices
    products = ["RT_HUB","RT_NODE","DA_HUB","DA_NODE"]
    prices = solve_product_prices(sims_rt, gen, products, p_level, neg_rule)

    # Expected generation by month/bucket
    eg = expected_generation_monthly(gen)

    # Save outputs
    prices.to_csv("results/fixed_prices.csv", index=False)
    eg.to_csv("results/expected_generation.csv", index=False)

    # Simple risk breakdown (merchant RT node revenue per scenario)
    df = gen.merge(sims_rt, on=['s','ts','market'])
    df['merch'] = df['gen_mwh'] * df['node_rt']
    agg = df.groupby(['asset','s'])['merch'].sum().reset_index()
    rb = agg.groupby('asset')['merch'].agg(['mean','std',lambda x: x.quantile(0.05)]).reset_index()
    rb = rb.rename(columns={'<lambda_0>':'p5'})
    rb.to_csv("results/risk_breakdown.csv", index=False)

    return {
        "artifacts": ["models/hub_shape.pkl","models/basis_rt.pkl","models/basis_da.pkl","models/volume_bootstrap.pkl"],
        "results": ["results/fixed_prices.csv","results/expected_generation.csv","results/risk_breakdown.csv"]
    }
