import numpy as np
import pandas as pd
from typing import List
from .hedge import merchant_revenue, pnl_hedged

def find_flat_price(gen, sell_price, ref_price, p_level: float, negative_rule: str) -> float:
    """
    Solve for flat P where percentile_q(hedged - merchant) >= 0 with q = 1 - p_level.
    Uses bisection on P; inputs should be aligned pandas Series, optionally multi-scenario.
    """
    q = 1.0 - p_level

    def diff_for(P):
        d = (pnl_hedged(gen, sell_price, ref_price, P, negative_rule) - merchant_revenue(gen, sell_price))
        if hasattr(d, "groupby"):
            g = d.groupby('s').sum()
            return np.percentile(g.values, q*100.0)
        return np.percentile(d, q*100.0)

    lo, hi = -200.0, 200.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        val = diff_for(mid)
        if val >= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def solve_product_prices(sim_prices: pd.DataFrame, gen_df: pd.DataFrame,
                         products: List[str], p_level: float, negative_rule: str) -> pd.DataFrame:
    """
    Compute P* for each product at requested p_level and negative price rule.
    sim_prices: ['s','ts','market','hub_rt','node_rt','hub_da','node_da']
    gen_df    : ['s','ts','asset','market','gen_mwh']
    """
    df = gen_df.merge(sim_prices, on=['s','ts','market'], how='left')
    out_rows = []
    for asset in df['asset'].unique():
        sub = df[df['asset'] == asset]
        for prod in products:
            if prod == "RT_HUB":
                sell = sub['node_rt']  # merchant realized at node RT
                ref = sub['hub_rt']    # hedge settles at hub RT
            elif prod == "RT_NODE":
                sell = sub['node_rt']
                ref = sub['node_rt']
            elif prod == "DA_HUB":
                sell = sub['node_rt']
                ref = sub['hub_da']
            elif prod == "DA_NODE":
                sell = sub['node_rt']
                ref = sub['node_da']
            else:
                continue
            P = find_flat_price(sub['gen_mwh'], sell, ref, p_level, negative_rule)
            out_rows.append({'asset': asset, 'product': prod, 'p_level': p_level,
                             'negative_rule': negative_rule, 'fixed_price': P})
    return pd.DataFrame(out_rows)

# ---------- Price breakdown (A, B, C, D, E) ----------

def _gen_weighted_avg(series: pd.Series, gen: pd.Series) -> float:
    w = gen.fillna(0.0).values
    x = series.fillna(0.0).values
    den = w.sum()
    return float((w @ x) / den) if den > 0 else float('nan')

def compute_price_breakdown(hourly_hub_fwd: pd.DataFrame,
                            basis_mean_rt: pd.DataFrame,
                            da_spr_mean_hub: pd.DataFrame,
                            da_spr_mean_node: pd.DataFrame,
                            sims: pd.DataFrame,
                            gen_df: pd.DataFrame,
                            products: List[str],
                            p_level: float) -> pd.DataFrame:
    """
    Build waterfall components (per asset/product):
      A: Hub capture (gen-wtd mean of hourly hub forward)
      B: Basis conversion (gen-wtd expected node-hub difference) for node-settled products; 0 for hub-settled
      C: DA-RT spread at settlement point (gen-wtd expected)
      D: Negative-price rule delta = P*(zero) - P*(include)  (computed via solver, scenario-based)
      E: Risk add-on so that A+B+C(+D if applicable) + E = selected P*
    Returns: ['asset','product','A_hub_capture','B_basis','C_da_rt','D_neg_rule','E_risk','P_star','neg_rule']
    """
    # Prepare deterministic expectations aligned to gen hours
    cal = hourly_hub_fwd.copy()
    cal['month'] = cal['ts'].to_period('M').dt.to_timestamp()
    cal['hour'] = cal['ts'].dt.hour + 1

    # Merge expected basis & spreads
    exp = (cal
           .merge(basis_mean_rt.rename(columns={'mean':'basis_mean'})[['market','month','hour','basis_mean']],
                  on=['market','month','hour'], how='left')
           .merge(da_spr_mean_hub.rename(columns={'mean':'spr_hub_mean'})[['market','month','hour','spr_hub_mean']],
                  on=['market','month','hour'], how='left')
           .merge(da_spr_mean_node.rename(columns={'mean':'spr_node_mean'})[['market','month','hour','spr_node_mean']],
                  on=['market','month','hour'], how='left'))
    for col in ['basis_mean','spr_hub_mean','spr_node_mean']:
        exp[col] = exp[col].fillna(0.0)

    # Join gen and expectations
    g = gen_df.merge(exp, on=['ts','market'], how='left')

    rows = []
    df = gen_df.merge(sims, on=['s','ts','market'], how='left')

    for asset in g['asset'].unique():
        g_asset = g[g['asset'] == asset]

        # A: hub capture (gen-weighted mean of hub forward)
        A = _gen_weighted_avg(g_asset['hub_forward_hourly'], g_asset['gen_mwh'])

        # Expected B and C per settlement later
        # Precompute gen-weighted expected basis and spreads
        B_exp = -_gen_weighted_avg(g_asset['basis_mean'], g_asset['gen_mwh'])  # node = hub - basis -> add (-basis)
        C_hub = _gen_weighted_avg(g_asset['spr_hub_mean'], g_asset['gen_mwh'])
        C_node = _gen_weighted_avg(g_asset['spr_node_mean'], g_asset['gen_mwh'])

        # Solve prices for include/zero to compute D and E cleanly
        sub = df[df['asset'] == asset]
        # common sell legs per product defined below
        for prod in products:
            if prod == "RT_HUB":
                sell = sub['node_rt']; ref_inc = sub['hub_rt']; ref_zero = sub['hub_rt']
                B = 0.0; C = 0.0
            elif prod == "RT_NODE":
                sell = sub['node_rt']; ref_inc = sub['node_rt']; ref_zero = sub['node_rt']
                B = B_exp; C = 0.0
            elif prod == "DA_HUB":
                sell = sub['node_rt']; ref_inc = sub['hub_da']; ref_zero = sub['hub_da']
                B = 0.0; C = C_hub
            elif prod == "DA_NODE":
                sell = sub['node_rt']; ref_inc = sub['node_da']; ref_zero = sub['node_da']
                B = B_exp; C = C_node
            else:
                continue

            P_incl = find_flat_price(sub['gen_mwh'], sell, ref_inc, p_level, negative_rule="include")
            P_zero = find_flat_price(sub['gen_mwh'], sell, ref_zero, p_level, negative_rule="zero")
            D = P_zero - P_incl  # uplift due to zeroing negatives

            # Choose P* according to the portfolio negative rule (you can set per-config if needed)
            # For breakdown we report both effects cleanly:
            # E_incl = residual risk premium under include; E_zero = under zero
            E_incl = P_incl - (A + B + C)
            E_zero = P_zero - (A + B + C) - D  # equals E_incl numerically

            rows.append({
                'asset': asset,
                'product': prod,
                'A_hub_capture': A,
                'B_basis': B,
                'C_da_rt': C,
                'D_neg_rule': D,
                'E_risk': E_incl,     # report include-version (same as zero-version residual)
                'P_star_include': P_incl,
                'P_star_zero': P_zero
            })

    out = pd.DataFrame(rows)
    return out

def expected_generation_monthly(gen_df: pd.DataFrame) -> pd.DataFrame:
    tmp = gen_df.copy()
    tmp['month'] = tmp['ts'].dt.to_period('M').dt.to_timestamp()
    tmp['peak'] = (tmp['ts'].dt.weekday <= 4) & (tmp['ts'].dt.hour + 1 >= 7) & (tmp['ts'].dt.hour + 1 <= 22)
    tmp['bucket'] = np.where(tmp['peak'], 'Peak', 'Off-Peak')
    out = tmp.groupby(['asset','month','bucket'])['gen_mwh'].mean().reset_index(name='expected_mwh')
    return out

def solve_prices_for_levels(sim_prices: pd.DataFrame, gen_df: pd.DataFrame,
                            products: List[str], p_levels: List[float],
                            negative_rule: str) -> pd.DataFrame:
    """
    Sweep a list of p_levels (e.g., [0.5,0.75,0.9]) and return stacked results.
    """
    frames = []
    for pl in p_levels:
        frames.append(solve_product_prices(sim_prices, gen_df, products, pl, negative_rule))
    return pd.concat(frames, ignore_index=True)
