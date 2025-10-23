
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


def simulate_prices(hourly_hub: pd.DataFrame, basis_stats: pd.DataFrame, n_scenarios: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hh = hourly_hub.copy()
    hh['ts']   = pd.to_datetime(hh['ts'])
    hh['moy']  = hh['ts'].dt.month
    hh['hour'] = hh['ts'].dt.hour + 1

    bs = basis_stats.copy()

    # Join on (market, moy, hour) so historical basis applies across years
    df = hh.merge(bs, on=['market','moy','hour'], how='left')
    df['mean'] = df['mean'].fillna(0.0)
    df['std']  = df['std'].fillna(0.0)

    sims = []
    for s in range(n_scenarios):
        z = rng.standard_normal(len(df))
        basis = df['mean'].values + df['std'].values * z

        hub = df['hub_forward_hourly'].values
        node = hub - basis

        tmp = pd.DataFrame({
            's': s,
            'ts': df['ts'].values,
            'market': df['market'].values,
            'hub_rt': hub,
            'node_rt': node,
        })
        # simple DA proxy
        tmp['hub_da']  = hub * 0.98 + 0.02 * tmp['hub_rt']
        tmp['node_da'] = node * 0.98 + 0.02 * tmp['node_rt']

        sims.append(tmp)

    return pd.concat(sims, ignore_index=True)

