# scripts/run_all_assets.py
# Run end-to-end valuation for all assets and save results for the portfolio notebook.

import os
import json
import numpy as np
import pandas as pd

# --- Project imports (existing code you already have) ---
from src.data.loader import load_data
from src.models.shape import hub_hourly_shape, shape_monthly_to_hourly, build_calendar
from src.models.price import simulate_prices
from src.models.volume import build_volume_table, bootstrap_generation
from src.pricing.solve_price import (
    solve_product_prices,
    solve_prices_for_levels,
    compute_price_breakdown,
)

# -------------------------
# User-adjustable settings
# -------------------------
RESULTS_DIR = "results"
SEED        = 42
SCENARIOS   = 500           # keep moderate for speed; raise for final runs
P_LEVEL     = 0.75
NEG_RULE    = "zero"        # 'zero' or 'all'
P_GRID      = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
PRODUCTS    = ['RT_HUB', 'RT_NODE', 'DA_HUB', 'DA_NODE']

# Input format config (matches how you placed your CSVs)
CONFIG = {
    "format": "per_asset_csv",
    "assets": [
        {
            "name": "Wind1",
            "market": "ERCOT",
            "hist_csv": "data/raw/hist_wind1_ercot.csv",
            "forward_csv": "data/raw/forward_wind1_ercot.csv",
        },
        {
            "name": "Wind2",
            "market": "MISO",
            "hist_csv": "data/raw/hist_wind2_miso.csv",
            "forward_csv": "data/raw/forward_wind2_miso.csv",
        },
        {
            "name": "Solar",
            "market": "CAISO",
            "hist_csv": "data/raw/hist_solar_caiso.csv",
            "forward_csv": "data/raw/forward_solar_caiso.csv",
        },
    ],
    # forward column names in your CSVs
    "forwards_cols": {"month": "Month", "peak": "Peak", "offpeak": "Off Peak"},
    # historical column rename map (adjust if your headers differ)
    "rename_map": {
        "date": "date",          # or 'Date'
        "he": "he",              # hour-ending 1..24
        "rt_hub": "RT Hub",      # or your exact column header
        "rt_node": "RT Busbar",  # busbar/node RT
        "da_hub": "DA Hub",      # day-ahead hub
        "da_node": "DA Busbar",  # day-ahead node
        "gen_mwh": "Gen",        # generation MWh (historical)
    },
}

# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_df(df: pd.DataFrame, path_csv: str):
    ensure_dir(os.path.dirname(path_csv))
    df.to_csv(path_csv, index=False)

# -------------------------
# Main
# -------------------------
def main():
    ensure_dir(RESULTS_DIR)

    # 1) Load all historical + forwards (per-asset CSVs)
    hist_all, fwd_all = load_data(CONFIG)

    # 2) Stats from history (common)
    shape_tbl = hub_hourly_shape(hist_all)  # market, mon, hour, shape
    bs_rt     = ( # RT hub->node basis stats
        hist_all.assign(mon=pd.to_datetime(hist_all['date']).dt.month,
                        hour=hist_all['he'])
                .groupby(['market','mon','hour'])
                .apply(lambda g: pd.Series({
                    'mean': (pd.to_numeric(g['rt_hub'], errors='coerce') - pd.to_numeric(g['rt_node'], errors='coerce')).mean(),
                    'std':  (pd.to_numeric(g['rt_hub'], errors='coerce') - pd.to_numeric(g['rt_node'], errors='coerce')).std(),
                })).reset_index()
    )

    # DA–RT spreads (hub)
    spr_hub = (
        hist_all.assign(mon=pd.to_datetime(hist_all['date']).dt.month,
                        hour=hist_all['he'])
                .groupby(['market','mon','hour'])
                .apply(lambda g: pd.Series({
                    'mean': (pd.to_numeric(g['da_hub'], errors='coerce') - pd.to_numeric(g['rt_hub'], errors='coerce')).mean(),
                    'std':  (pd.to_numeric(g['da_hub'], errors='coerce') - pd.to_numeric(g['rt_hub'], errors='coerce')).std(),
                })).reset_index()
    )
    # DA–RT spreads (node)
    spr_node = (
        hist_all.assign(mon=pd.to_datetime(hist_all['date']).dt.month,
                        hour=hist_all['he'])
                .groupby(['market','mon','hour'])
                .apply(lambda g: pd.Series({
                    'mean': (pd.to_numeric(g['da_node'], errors='coerce') - pd.to_numeric(g['rt_node'], errors='coerce')).mean(),
                    'std':  (pd.to_numeric(g['da_node'], errors='coerce') - pd.to_numeric(g['rt_node'], errors='coerce')).std(),
                })).reset_index()
    )

    # 3) Horizon calendar (2026-01-01 .. 2030-12-31)
    cal = build_calendar(start="2026-01-01 00:00:00", end="2030-12-31 23:00:00", tz=None)

    # 4) Turn monthly hub forwards into hourly hub curves
    hh_all = shape_monthly_to_hourly(fwd_all, shape_tbl, cal)
    # Columns: ['market','ts','hub_forward_hourly']

    # Prepare portfolio outputs
    portfolio_rows = []
    all_prices_p75 = []
    all_breakdowns = []
    all_pgrid      = []

    rng = np.random.default_rng(SEED)

    for a in CONFIG["assets"]:
        asset = a["name"]
        market = a["market"]

        # Subsets for the asset/market
        hist = hist_all[hist_all["market"] == market].copy()
        hh   = hh_all[hh_all["market"] == market].copy()

        # 5) Simulate hourly prices (RT hub/node, DA hub/node)
        sims_all = simulate_prices(
            hourly_hub=hh,
            basis_rt=bs_rt,          # expects ['market','mon','hour','mean','std']
            da_spr_hub=spr_hub,
            da_spr_node=spr_node,
            n_scenarios=SCENARIOS,
            seed=SEED
        )
        sims = sims_all[sims_all["market"] == market].copy()

        # 6) Generation model: bootstrap by (mon, hour)
        vol_tbl = build_volume_table(hist_all, asset, market)  # returns samples per (asset, market, mon, hour)
        gen_all = bootstrap_generation(cal, vol_tbl, n_scenarios=SCENARIOS, rng=rng)
        gen = gen_all[(gen_all["asset"] == asset) & (gen_all["market"] == market)].copy()

        # 7) Solve risk-adjusted fixed prices at P_LEVEL and price grid
        prices_p75 = solve_product_prices(sims, gen, PRODUCTS, p_level=P_LEVEL, negative_rule=NEG_RULE)
        prices_p75 = prices_p75[prices_p75["asset"] == asset].copy()

        prices_grid = solve_prices_for_levels(sims, gen, PRODUCTS, P_GRID, negative_rule=NEG_RULE)
        prices_grid = prices_grid[prices_grid["asset"] == asset].copy()

        # 8) Component breakdown (expected ref, risk premium, etc.)
        breakdown = compute_price_breakdown(
            hourly_hub_fwd=hh,
            basis_mean_rt=bs_rt[['market','mon','hour','mean']],
            da_spr_mean_hub=spr_hub[['market','mon','hour','mean']],
            da_spr_mean_node=spr_node[['market','mon','hour','mean']],
            sims=sims,
            gen_df=gen,
            products=PRODUCTS,
            p_level=P_LEVEL
        )
        breakdown = breakdown[breakdown["asset"] == asset].copy()

        # --- Save per-asset outputs ---
        save_df(prices_p75, f"{RESULTS_DIR}/{asset}_fixed_prices_p75.csv")
        save_df(prices_grid, f"{RESULTS_DIR}/{asset}_fixed_prices_pgrid.csv")
        save_df(breakdown,  f"{RESULTS_DIR}/{asset}_price_breakdown_p75.csv")

        # small samples for plots (to avoid huge files)
        # sample one scenario's generation and 3 days of hourly hub forward
        try:
            s0 = int(gen['s'].min())
            gen_s0 = gen[gen['s'] == s0].copy()
            gen_s0_3d = gen_s0.sort_values('ts').iloc[:24*3].copy()
            save_df(gen_s0_3d, f"{RESULTS_DIR}/{asset}_gen_sample_3d.csv")
        except Exception:
            pass

        hh_3d = hh.sort_values('ts').iloc[:24*3].copy()
        save_df(hh_3d, f"{RESULTS_DIR}/{asset}_hub_fwd_sample_3d.csv")

        # collect for portfolio-level tables
        all_prices_p75.append(prices_p75.assign(asset=asset))
        all_breakdowns.append(breakdown.assign(asset=asset))
        all_pgrid.append(prices_grid.assign(asset=asset))

        # for quick portfolio bar chart
        for _, r in prices_p75.iterrows():
            portfolio_rows.append({
                "asset": asset, "product": r["product"],
                "p_level": r["p_level"], "fixed_price": r["fixed_price"]
            })

    # --- Portfolio summaries ---
    df_port = pd.DataFrame(portfolio_rows)
    save_df(df_port, f"{RESULTS_DIR}/portfolio_fixed_prices_p75.csv")

    if len(all_pgrid):
        df_pgrid_all = pd.concat(all_pgrid, ignore_index=True)
        save_df(df_pgrid_all, f"{RESULTS_DIR}/portfolio_fixed_prices_pgrid.csv")

    if len(all_breakdowns):
        df_break_all = pd.concat(all_breakdowns, ignore_index=True)
        save_df(df_break_all, f"{RESULTS_DIR}/portfolio_breakdown_p75.csv")

    if len(all_prices_p75):
        df_prices_all = pd.concat(all_prices_p75, ignore_index=True)
        save_df(df_prices_all, f"{RESULTS_DIR}/_all_assets_fixed_prices_p75.csv")

    # Save run meta
    meta = {
        "seed": SEED,
        "scenarios": SCENARIOS,
        "p_level": P_LEVEL,
        "negative_rule": NEG_RULE,
        "p_grid": P_GRID,
        "products": PRODUCTS
    }
    with open(f"{RESULTS_DIR}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Results saved to 'results/'.")

if __name__ == "__main__":
    main()
