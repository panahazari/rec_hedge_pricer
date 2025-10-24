import yaml
import numpy as np
import pandas as pd

from src.data.loader import load_data
from src.features.calendar import make_calendar_index
from src.models.shape import hub_hourly_shape
from src.models.price import shape_monthly_to_hourly, basis_stats, da_spread_stats, simulate_prices
from src.models.volume import bootstrap_generation
from src.pricing.solve_price import (
    solve_product_prices,
    expected_generation_monthly,
    compute_price_breakdown,
    solve_prices_for_levels,
)
from src.utils.persistence import save_json, save_model

def _read_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_all(config_path: str = "configs/project.yaml"):
    cfg = _read_config(config_path)

    # Load data (CSV or Excel based on cfg['data_format'])
    hist, fwd = load_data(cfg)

    start_year = cfg['start_year']
    end_year = cfg['end_year']
    n_scenarios = cfg['n_scenarios']
    seed = cfg.get('random_seed', 42)
    p_level = cfg.get('p_level', 0.75)
    neg_rule = cfg.get('negative_price_rule', 'include')

    # Fit shape/basis/spreads + volume bootstrap
    shp = hub_hourly_shape(hist)
    bs_rt = basis_stats(hist, price_col_hub='rt_hub', price_col_node='rt_node')
    spr_hub = da_spread_stats(hist, level='hub')
    spr_node = da_spread_stats(hist, level='node')
    vol_tbl = (hist.assign(month=hist['date'].dt.to_period('M').dt.to_timestamp(),
                           hour=hist['he'])
                    .groupby(['asset','market','month','hour'])['gen_mwh']
                    .apply(list).reset_index(name='samples'))

    # Save artifacts
    save_json({'p_level': p_level, 'negative_price_rule': neg_rule}, "models/config_summary.json")
    save_model(shp, "models/hub_shape.pkl")
    save_model(bs_rt, "models/basis_rt.pkl")
    save_model(spr_hub, "models/da_spread_hub.pkl")
    save_model(spr_node, "models/da_spread_node.pkl")
    save_model(vol_tbl, "models/volume_bootstrap.pkl")

    # Calendar & hourly hub forward
    cal = make_calendar_index(start_year, end_year)
    hh = shape_monthly_to_hourly(fwd, shp, cal)

    # Simulate hourly prices
    sims = simulate_prices(hh, bs_rt, spr_hub, spr_node, n_scenarios=n_scenarios, seed=seed)

    # Generation scenarios
    rng = np.random.default_rng(seed)
    gen = bootstrap_generation(cal, vol_tbl, n_scenarios, rng)

    # Solve fixed prices (P*)
    products = ["RT_HUB","RT_NODE","DA_HUB","DA_NODE"]
    prices = solve_product_prices(sims, gen, products, p_level, neg_rule)
    prices.to_csv("results/fixed_prices.csv", index=False)

    # Expected generation by month/bucket
    eg = expected_generation_monthly(gen)
    eg.to_csv("results/expected_generation.csv", index=False)

    # Price breakdown (A,B,C,D,E) using expectations + solver deltas
    breakdown = compute_price_breakdown(
        hourly_hub_fwd=hh,
        basis_mean_rt=bs_rt[['market','month','hour','mean']],
        da_spr_mean_hub=spr_hub[['market','month','hour','mean']],
        da_spr_mean_node=spr_node[['market','month','hour','mean']],
        sims=sims,
        gen_df=gen,
        products=products,
        p_level=p_level
    )
    breakdown.to_csv("results/price_breakdown.csv", index=False)

    # Simple risk breakdown for merchant (scenario distribution)
    dfm = gen.merge(sims, on=['s','ts','market'])
    dfm['merchant_rt_node'] = dfm['gen_mwh'] * dfm['node_rt']
    agg = dfm.groupby(['asset','s'])['merchant_rt_node'].sum().reset_index()
    rb = agg.groupby('asset')['merchant_rt_node'].agg(['mean','std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.05)]).reset_index()
    rb = rb.rename(columns={'<lambda_0>':'p25','<lambda_1>':'p5'})
    rb.to_csv("results/risk_breakdown.csv", index=False)

    # P-level sweep (optional grid)
    p_grid = cfg.get('p_level_grid', [0.50, 0.75, 0.90])
    sweep = solve_prices_for_levels(sims, gen, products, p_grid, neg_rule)
    sweep.to_csv("results/fixed_prices_Pgrid.csv", index=False)

    return {
        "artifacts": [
            "models/hub_shape.pkl","models/basis_rt.pkl",
            "models/da_spread_hub.pkl","models/da_spread_node.pkl",
            "models/volume_bootstrap.pkl"
        ],
        "results": [
            "results/fixed_prices.csv",
            "results/fixed_prices_Pgrid.csv",
            "results/expected_generation.csv",
            "results/price_breakdown.csv",
            "results/risk_breakdown.csv"
        ]
    }
