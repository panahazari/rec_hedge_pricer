
import numpy as np
import pandas as pd
from typing import Dict, List
from .hedge import merchant_revenue, pnl_hedged

def find_flat_price(gen, sell_price, ref_price, p_level: float, negative_rule: str) -> float:
    # Solve for P where percentile(hedge - merchant) >= 0 at q = 1 - p_level
    # Since hedge P&L is linear in fixed price, use a quick bracket + bisection.
    q = 1.0 - p_level
    def diff_for(P):
        d = (pnl_hedged(gen, sell_price, ref_price, P, negative_rule) - merchant_revenue(gen, sell_price))
        # aggregate across horizon per scenario if multi-index
        if hasattr(d, "groupby"):
            g = d.groupby('s').sum()
            return np.percentile(g.values, q*100)
        return np.percentile(d, q*100)
    lo, hi = -200.0, 200.0
    for _ in range(40):
        mid = 0.5*(lo+hi)
        val = diff_for(mid)
        if val >= 0:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)

def solve_product_prices(sim_prices: pd.DataFrame, gen_df: pd.DataFrame, products: List[str], p_level: float, negative_rule: str) -> pd.DataFrame:
    # sim_prices: [s, ts, market, hub_rt, node_rt, hub_da, node_da]
    # gen_df:     [s, ts, asset, market, gen_mwh]
    df = gen_df.merge(sim_prices, on=['s','ts','market'], how='left')
    out_rows = []
    for asset in df['asset'].unique():
        sub = df[df['asset']==asset]
        for prod in products:
            if prod == "RT_HUB":
                sell = sub['node_rt']  # merchant sells at node RT
                ref = sub['hub_rt']    # hedge settles vs hub RT
            elif prod == "RT_NODE":
                sell = sub['node_rt']
                ref = sub['node_rt']
            elif prod == "DA_HUB":
                sell = sub['node_rt']  # keep merchant at RT node baseline
                ref = sub['hub_da']
            elif prod == "DA_NODE":
                sell = sub['node_rt']
                ref = sub['node_da']
            else:
                continue
            P = find_flat_price(sub['gen_mwh'], sell, ref, p_level, negative_rule)
            out_rows.append({'asset': asset, 'product': prod, 'fixed_price': P})
    return pd.DataFrame(out_rows)

def expected_generation_monthly(gen_df: pd.DataFrame) -> pd.DataFrame:
    tmp = gen_df.copy()
    tmp['month'] = tmp['ts'].dt.to_period('M').dt.to_timestamp()
    tmp['peak'] = (tmp['ts'].dt.weekday <= 4) & (tmp['ts'].dt.hour+1>=7) & (tmp['ts'].dt.hour+1<=22)
    tmp['bucket'] = np.where(tmp['peak'], 'Peak', 'Off-Peak')
    out = tmp.groupby(['asset','month','bucket'])['gen_mwh'].mean().reset_index(name='expected_mwh')  # scenario mean
    return out
