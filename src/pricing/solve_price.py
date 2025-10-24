import numpy as np
import pandas as pd
from typing import List
from .hedge import merchant_revenue, pnl_hedged

# src/pricing/solve_price.py


import numpy as np
import pandas as pd

def _coerce_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        sel = out[c]
        # If duplicate column labels exist, sel can be a DataFrame. Take first.
        if isinstance(sel, pd.DataFrame):
            sel = sel.iloc[:, 0]
        out[c] = pd.to_numeric(sel, errors='coerce').fillna(0.0)
    return out

def _sum_by_s(keys_s: np.ndarray, values: np.ndarray) -> pd.Series:
    """
    Fast per-s sum using numpy. Works even if 's' is not 0..N-1.
    Returns a Series indexed by unique s values.
    """
    uniq, inv = np.unique(keys_s, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    return pd.Series(sums, index=uniq)

def merchant_revenue(df: pd.DataFrame, sell_col: str) -> pd.Series:
    """
    Per-scenario merchant revenue: sum_t gen_mwh * sell_price.
    df must have ['s','gen_mwh', sell_col]
    """
    dfn = _coerce_numeric_cols(df, ['gen_mwh', sell_col])
    vals = (dfn['gen_mwh'].to_numpy() * dfn[sell_col].to_numpy())
    return _sum_by_s(dfn['s'].to_numpy(), vals)

def pnl_hedged(df: pd.DataFrame, sell_col: str, ref_col: str, P: float, negative_rule: str) -> pd.Series:
    """
    Per-scenario hedge PnL: sum_t (P - ref_price) * vol, with vol rule on negative ref prices.
    df must have ['s','gen_mwh', ref_col]
    """
    dfn = _coerce_numeric_cols(df, ['gen_mwh', ref_col])
    ref = dfn[ref_col].to_numpy()
    vol = dfn['gen_mwh'].to_numpy()
    if negative_rule == 'zero':
        vol = np.where(ref < 0.0, 0.0, vol)
    pnl = (P - ref) * vol
    return _sum_by_s(dfn['s'].to_numpy(), pnl)

def find_flat_price(df: pd.DataFrame, sell_col: str, ref_col: str,
                    p_level: float, negative_rule: str) -> float:
    q = 1.0 - float(p_level)

    # Merchant is not needed for the quantile target if pnl_hedged returns only CFD payoff
    def q_val(P):
        hedge_by_s = pnl_hedged(df[['s','gen_mwh', ref_col]].copy(),
                                sell_col=None, ref_col=ref_col,
                                P=P, negative_rule=negative_rule)
        # We want Quantile_{1-p}( CFD payoff ) = 0
        return np.percentile(hedge_by_s.values, q * 100.0)

    lo, hi = -1000.0, 1000.0
    for _ in range(50):
        mid = 0.5*(lo + hi)
        if q_val(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo + hi)


def solve_product_prices(sim_prices: pd.DataFrame,
                         gen_df: pd.DataFrame,
                         products: list[str],
                         p_level: float,
                         negative_rule: str) -> pd.DataFrame:
    """
    Expects sim_prices with ['s','ts','market','hub_rt','node_rt','hub_da','node_da']
            gen_df     with ['s','ts','asset','market','gen_mwh']
    """
    # Merge and drop duplicate columns by name to avoid DataFrame selection on ref/sell cols
    df = gen_df.merge(sim_prices, on=['s','ts','market'], how='inner')
    df = df.loc[:, ~df.columns.duplicated()].copy()

    need = {'s','ts','asset','market','gen_mwh','hub_rt','node_rt','hub_da','node_da'}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"solve_product_prices: missing columns {sorted(miss)}")

    out = []
    for asset in df['asset'].unique():
        sub_a = df[df['asset'] == asset].copy()

        for prod in products:
            if   prod == 'RT_HUB':
                sell_col, ref_col = 'node_rt', 'hub_rt'
            elif prod == 'RT_NODE':
                sell_col, ref_col = 'node_rt', 'node_rt'
            elif prod == 'DA_HUB':
                sell_col, ref_col = 'node_rt', 'hub_da'
            elif prod == 'DA_NODE':
                sell_col, ref_col = 'node_rt', 'node_da'
            else:
                continue

            cols_req = ['s', 'gen_mwh', sell_col, ref_col]
            cols_req = list(dict.fromkeys(cols_req))  # preserve order, drop dups
            sub_cols = sub_a[cols_req].copy()

            sub_cols = _coerce_numeric_cols(sub_cols, [c for c in cols_req if c != 's'])

            P = find_flat_price(sub_cols, sell_col, ref_col, p_level, negative_rule)
            out.append({
                'asset': asset,
                'product': prod,
                'p_level': p_level,
                'negative_rule': negative_rule,
                'fixed_price': P
            })

    return pd.DataFrame(out)


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
                            products: list[str],
                            p_level: float) -> pd.DataFrame:
    """
    Build expected components used to explain fixed prices:
      - H_t            : hourly hub forward (from hourly_hub_fwd)
      - basis_mean     : E[RT hub - RT node] by (market, mon, hour)
      - spr_hub_mean   : E[DA hub - RT hub]   by (market, mon, hour)
      - spr_node_mean  : E[DA node - RT node] by (market, mon, hour)

    Keys are normalized to (market, mon=1..12, hour=HE 1..24).
    Returns one row per (asset, product) with component averages and P* at p_level.
    """

    # --- Calendar from hourly hub fwd ---
    cal = hourly_hub_fwd[['market', 'ts', 'hub_forward_hourly']].copy()
    cal['ts'] = pd.to_datetime(cal['ts'])
    cal['mon'] = cal['ts'].dt.month
    cal['hour'] = cal['ts'].dt.hour + 1  # HE 1..24

    # --- Normalize stats inputs to (market, mon, hour, value) ---
    def _norm(df, col_name):
        out = df.copy()
        if 'mon' not in out.columns:
            # allow 'month' (timestamp) -> mon
            if 'month' in out.columns:
                out['mon'] = pd.to_datetime(out['month']).dt.month
            else:
                raise KeyError(f"{col_name}: requires 'mon' or 'month' column")
        if 'hour' not in out.columns:
            raise KeyError(f"{col_name}: requires 'hour' column")
        out = out[['market', 'mon', 'hour', col_name]].copy()
        return out

    b_rt = basis_mean_rt.rename(columns={'mean': 'basis_mean'})
    s_h  = da_spr_mean_hub.rename(columns={'mean': 'spr_hub_mean'})
    s_n  = da_spr_mean_node.rename(columns={'mean': 'spr_node_mean'})

    b_rt = _norm(b_rt, 'basis_mean')
    s_h  = _norm(s_h,  'spr_hub_mean')
    s_n  = _norm(s_n,  'spr_node_mean')

    # --- Expected hourly components table keyed by ts ---
    exp = (cal
           .merge(b_rt, on=['market', 'mon', 'hour'], how='left')
           .merge(s_h,  on=['market', 'mon', 'hour'], how='left')
           .merge(s_n,  on=['market', 'mon', 'hour'], how='left'))
    for col in ['basis_mean', 'spr_hub_mean', 'spr_node_mean']:
        exp[col] = pd.to_numeric(exp[col], errors='coerce').fillna(0.0)

    # --- Helper: expected reference price per product (hourly) ---
    # Using means only (no stochastic shocks) to explain components
    def _ref_cols(df):
        df = df.copy()
        df['RT_HUB_ref'] = df['hub_forward_hourly']
        df['RT_NODE_ref'] = df['hub_forward_hourly'] - df['basis_mean']
        df['DA_HUB_ref'] = df['hub_forward_hourly'] + df['spr_hub_mean']
        df['DA_NODE_ref'] = df['hub_forward_hourly'] - df['basis_mean'] + df['spr_node_mean']
        return df

    exp = _ref_cols(exp)

    # --- Compute P* via scenarios (re-use find_flat_price on merged df) ---
    # Merge gen with sims to get scenario-level prices and volumes
    df_all = gen_df.merge(sims, on=['s', 'ts', 'market'], how='inner')
    df_all = df_all.loc[:, ~df_all.columns.duplicated()].copy()

    out_rows = []
    for asset in df_all['asset'].unique():
        sub = df_all[df_all['asset'] == asset].copy()

        # Join expected (non-stochastic) components for weighted means
        sub = sub.merge(
            exp[['market','ts','hub_forward_hourly',
                'basis_mean','spr_hub_mean','spr_node_mean',
                'RT_HUB_ref','RT_NODE_ref','DA_HUB_ref','DA_NODE_ref']],
            on=['market','ts'], how='left'
        )
        sub = sub.loc[:, ~sub.columns.duplicated()].copy()  # <— add this


        # Generation weights per hour (sum over scenarios later if needed)
        # For expected averages we’ll use scenario-mean gen per hour to avoid bias.
        gen_mean = sub.groupby('ts', as_index=False)['gen_mwh'].mean().rename(columns={'gen_mwh': 'gen_mean'})
        sub = sub.merge(gen_mean, on='ts', how='left')

        # Expected component (gen-weighted) over horizon
        w = sub[['ts', 'gen_mean']].drop_duplicates()
        w_sum = w['gen_mean'].sum() if w['gen_mean'].sum() != 0 else 1.0

        def wavg(col):
            tmp = sub[['ts', col]].drop_duplicates().merge(w, on='ts', how='left')
            return float((tmp[col] * tmp['gen_mean']).sum() / w_sum)

        exp_hub   = wavg('hub_forward_hourly')
        exp_basis = wavg('basis_mean')
        exp_spr_h = wavg('spr_hub_mean')
        exp_spr_n = wavg('spr_node_mean')

        # Solve P* per product using scenario paths
        prods = products or ['RT_HUB', 'RT_NODE', 'DA_HUB', 'DA_NODE']
        for prod in prods:
            if   prod == 'RT_HUB':  sell_col, ref_col = 'node_rt', 'hub_rt'
            elif prod == 'RT_NODE': sell_col, ref_col = 'node_rt', 'node_rt'
            elif prod == 'DA_HUB':  sell_col, ref_col = 'node_rt', 'hub_da'
            elif prod == 'DA_NODE': sell_col, ref_col = 'node_rt', 'node_da'
            else:                   continue


            cols_req = ['s', 'gen_mwh', sell_col, ref_col]
            cols_req = list(dict.fromkeys(cols_req))  # drop duplicates but preserve order
            df_p = sub[cols_req].copy()

            # coerce to numeric defensively (skip 's')
            df_p = _coerce_numeric_cols(df_p, [c for c in cols_req if c != 's'])

            P_star = find_flat_price(df_p, sell_col, ref_col, p_level, negative_rule='zero')


            # Expected reference (gen-weighted) from means for reporting
            if   prod == 'RT_HUB':  exp_ref = exp_hub
            elif prod == 'RT_NODE': exp_ref = exp_hub - exp_basis
            elif prod == 'DA_HUB':  exp_ref = exp_hub + exp_spr_h
            elif prod == 'DA_NODE': exp_ref = exp_hub - exp_basis + exp_spr_n

            out_rows.append({
                'asset': asset,
                'product': prod,
                'p_level': p_level,
                'hub_mean': exp_hub,
                'basis_mean': exp_basis,
                'da_spr_hub_mean': exp_spr_h,
                'da_spr_node_mean': exp_spr_n,
                'expected_ref_price': exp_ref,
                'fixed_price': P_star,
                'risk_premium': P_star - exp_ref
            })

    return pd.DataFrame(out_rows)


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
