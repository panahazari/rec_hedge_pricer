
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

def shape_monthly_to_hourly(forwards: pd.DataFrame, shape: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    forwards: ['month','market','peak','offpeak']  (month = month-start TS)
    shape:    ['market','moy','hour','shape']      (moy=1..12, hour=1..24)
    calendar: ['ts','he','weekday']                (hourly rows)
    """
    # Normalize forwards
    fwd = forwards.copy()
    fwd['market'] = fwd['market'].astype(str).str.strip().str.upper()
    fwd['month']  = pd.to_datetime(fwd['month']).dt.to_period('M').dt.to_timestamp()
    fwd['moy']    = fwd['month'].dt.month

    # Normalize shape
    shp = shape.copy()
# normalize shape
    shp = shape.copy()

    # NEW: derive moy if not present (for backward compatibility)
    if 'moy' not in shp.columns:
        if 'month' in shp.columns:
            shp['moy'] = pd.to_datetime(shp['month']).dt.month
        else:
            raise ValueError("shape must have either 'moy' or 'month' to derive month-of-year")

    shp['market'] = shp['market'].astype(str).str.strip().str.upper()
    shp['moy']    = pd.to_numeric(shp['moy'], errors='coerce').astype('Int64')
    shp['hour']   = pd.to_numeric(shp['hour'], errors='coerce').astype('Int64')
    shp['shape']  = pd.to_numeric(shp['shape'], errors='coerce')


    # Normalize calendar
    cal = calendar.copy()
    cal['ts']    = pd.to_datetime(cal['ts'])
    cal['month'] = cal['ts'].dt.to_period('M').dt.to_timestamp()
    cal['moy']   = cal['month'].dt.month
    he = pd.to_numeric(cal['he'], errors='coerce').fillna(0).astype(int)
    cal['hour']  = ((he - 1) % 24) + 1

    # 1) Attach forwards by MONTH (calendar has no market)

    out = cal.merge(fwd[['month','market','peak','offpeak']],
                    on='month', how='left', validate='many_to_many')

    # 2) Base level: Peak vs Off-Peak
    is_peak = (out['weekday'] <= 4) & (out['he'].between(7, 22))
    out['_base'] = np.where(is_peak, out['peak'], out['offpeak'])

    # 3) Merge shape by (market, moy, hour)  <-- year-agnostic
    out = out.merge(shp[['market','moy','hour','shape']],
                    on=['market','moy','hour'], how='left')

    # 4) Fallback shape = 1.0 where missing
    out['shape'] = out['shape'].fillna(1.0)

    # 5) Hourly hub forward
    out['hub_forward_hourly'] = out['_base'] * out['shape']

    return out[['ts','market','hub_forward_hourly']]


def basis_stats(hist: pd.DataFrame, price_col_hub: str, price_col_node: str) -> pd.DataFrame:
    """
    Expected hub-node basis by (market, month, hour) with std for simulation.
    Returns: ['market','month','hour','mean','std']
    """
    df = hist.copy()
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['hour'] = df['he']
    df['basis'] = df[price_col_hub] - df[price_col_node]
    out = df.groupby(['market','month','hour'])['basis'].agg(['mean', 'std']).reset_index()
    return out

def da_spread_stats(hist: pd.DataFrame, level: str = 'hub') -> pd.DataFrame:
    """
    Gen-agnostic expected DA-RT spreads by (market, month, hour) with std.
    level: 'hub' or 'node' -> uses (da_hub-rt_hub) or (da_node-rt_node)
    Returns: ['market','month','hour','mean','std']
    """
    df = hist.copy()
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['hour'] = df['he']
    if level == 'hub':
        df['spr'] = df['da_hub'] - df['rt_hub']
    else:
        df['spr'] = df['da_node'] - df['rt_node']
    out = df.groupby(['market','month','hour'])['spr'].agg(['mean','std']).reset_index()
    return out

def simulate_prices(hourly_hub: pd.DataFrame,
                    basis_rt: pd.DataFrame,
                    da_spr_hub: pd.DataFrame,
                    da_spr_node: pd.DataFrame,
                    n_scenarios: int,
                    seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hh = hourly_hub.copy()
    # ✅ ensure ts is datetime, then use .dt.to_period(...)
    hh['ts'] = pd.to_datetime(hh['ts'])
    hh['month'] = hh['ts'].dt.to_period('M').dt.to_timestamp()
    hh['hour'] = hh['ts'].dt.hour + 1


    # Merge expected stats
    df = (hh
          .merge(basis_rt.rename(columns={'mean':'basis_mean','std':'basis_std'}),
                 on=['market','month','hour'], how='left')
          .merge(da_spr_hub.rename(columns={'mean':'spr_hub_mean','std':'spr_hub_std'}),
                 on=['market','month','hour'], how='left')
          .merge(da_spr_node.rename(columns={'mean':'spr_node_mean','std':'spr_node_std'}),
                 on=['market','month','hour'], how='left'))

    for col in ['basis_mean','basis_std','spr_hub_mean','spr_hub_std','spr_node_mean','spr_node_std']:
        df[col] = df[col].fillna(0.0)

    rows = []
    n = len(df)
    for s in range(n_scenarios):
        z_basis = rng.standard_normal(n)
        z_spr_h = rng.standard_normal(n)
        z_spr_n = rng.standard_normal(n)

        hub_rt = df['hub_forward_hourly'].values  # deterministic for clarity/audit
        basis  = df['basis_mean'].values + df['basis_std'].values * z_basis
        node_rt = hub_rt - basis

        hub_da = hub_rt + (df['spr_hub_mean'].values + df['spr_hub_std'].values * z_spr_h)
        node_da = node_rt + (df['spr_node_mean'].values + df['spr_node_std'].values * z_spr_n)

        rows.append(pd.DataFrame({
            's': s,
            'ts': df['ts'].values,
            'market': df['market'].values,
            'hub_rt': hub_rt,
            'node_rt': node_rt,
            'hub_da': hub_da,
            'node_da': node_da
        }))

    return pd.concat(rows, ignore_index=True)