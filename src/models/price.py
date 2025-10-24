
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

#
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
    Expected hub-node basis by (market, mon(1..12), hour) with std for simulation.
    Returns: ['market','mon','hour','mean','std']
    """
    df = hist.copy()
    df['hour'] = df['he']                      # HE 1..24
    df['mon']  = df['date'].dt.month          # 1..12
    df['basis'] = df[price_col_hub] - df[price_col_node]
    out = df.groupby(['market','mon','hour'])['basis'].agg(['mean','std']).reset_index()
    return out

def da_spread_stats(hist: pd.DataFrame, level: str = 'hub') -> pd.DataFrame:
    """
    Expected DA-RT spreads by (market, mon(1..12), hour) with std.
    Returns: ['market','mon','hour','mean','std']
    """
    df = hist.copy()
    df['hour'] = df['he']             # HE 1..24
    df['mon']  = df['date'].dt.month  # 1..12
    if level == 'hub':
        df['spr'] = df['da_hub'] - df['rt_hub']
    else:
        df['spr'] = df['da_node'] - df['rt_node']
    out = df.groupby(['market','mon','hour'])['spr'].agg(['mean','std']).reset_index()
    return out


import numpy as np
import pandas as pd

def simulate_prices(hourly_hub: pd.DataFrame,
                    basis_rt: pd.DataFrame,
                    da_spr_hub: pd.DataFrame,
                    da_spr_node: pd.DataFrame,
                    n_scenarios: int,
                    seed: int = 42) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    ['s','ts','market','hub_rt','node_rt','hub_da','node_da']
    """
    # --- Basic input checks ---
    for name, df, cols in [
        ("hourly_hub", hourly_hub, ["ts", "market", "hub_forward_hourly"]),
        ("basis_rt", basis_rt, ["market", "mon", "hour", "mean", "std"]),
        ("da_spr_hub", da_spr_hub, ["market", "mon", "hour", "mean", "std"]),
        ("da_spr_node", da_spr_node, ["market", "mon", "hour", "mean", "std"]),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} is missing columns {missing}. Found {list(df.columns)}")

    rng = np.random.default_rng(seed)

    # Prepare hourly table with (mon, hour)
    hh = hourly_hub.copy()
    hh['ts'] = pd.to_datetime(hh['ts'])
    hh['hour'] = hh['ts'].dt.hour + 1   # HE 1..24
    hh['mon']  = hh['ts'].dt.month      # 1..12

    # Merge expected stats on (market, mon, hour)
    df = (hh
          .merge(basis_rt.rename(columns={'mean':'basis_mean','std':'basis_std'}),
                 on=['market','mon','hour'], how='left')
          .merge(da_spr_hub.rename(columns={'mean':'spr_hub_mean','std':'spr_hub_std'}),
                 on=['market','mon','hour'], how='left')
          .merge(da_spr_node.rename(columns={'mean':'spr_node_mean','std':'spr_node_std'}),
                 on=['market','mon','hour'], how='left'))

    # Coverage diagnostics (fail fast if no matches)
    basis_match_rate = 1.0 - df['basis_mean'].isna().mean()
    spr_hub_match    = 1.0 - df['spr_hub_mean'].isna().mean()
    spr_node_match   = 1.0 - df['spr_node_mean'].isna().mean()
    if basis_match_rate < 0.50:
        raise ValueError(f"Basis match rate too low ({basis_match_rate:.1%}). "
                         f"Check that basis_rt uses keys (market, mon, hour).")
    if (spr_hub_match < 0.50) or (spr_node_match < 0.50):
        raise ValueError(f"DA–RT spread match rate too low "
                         f"(hub={spr_hub_match:.1%}, node={spr_node_match:.1%}). "
                         f"Check da_spr_* keys (market, mon, hour).")

    # Fill gaps (leftovers only)
    for col in ['basis_mean','basis_std','spr_hub_mean','spr_hub_std','spr_node_mean','spr_node_std']:
        df[col] = df[col].fillna(0.0)

    n = len(df)
    if n == 0:
        # Avoid returning None; always return a valid empty frame
        return pd.DataFrame(columns=['s','ts','market','hub_rt','node_rt','hub_da','node_da'])

    # Scenario generation
    rows = []
    for s in range(int(n_scenarios)):
        z_basis = rng.standard_normal(n)
        z_spr_h = rng.standard_normal(n)
        z_spr_n = rng.standard_normal(n)

        hub_rt = df['hub_forward_hourly'].values  # deterministic (forward-consistent)
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

    # Concatenate and RETURN (never None)
    return pd.concat(rows, ignore_index=True)
