
import numpy as np
import pandas as pd

def shape_monthly_to_hourly(forwards: pd.DataFrame, shape: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    # Merge monthly Peak/OP to calendar, then apply average shape per (market,month,hour)
    fwd = forwards.copy()
    cal = calendar.copy()
    cal['month'] = cal['ts'].to_period('M').dt.to_timestamp()
    cal['hour'] = cal['he']
    out = cal.merge(fwd, on=['market','month'], how='left').merge(shape, on=['market','month','hour'], how='left')
    # Use peak/offpeak flag to choose base level before shape. Assume P/OP provided in historical used to fit shape; here we only need level.
    # Simple: start from (peak if weekday&HE in 7..22, else offpeak)
    is_peak = (out['weekday']<=4) & (out['he'].between(7,22))
    base = np.where(is_peak, out['peak'], out['offpeak'])
    out['hub_forward_hourly'] = base * out['shape'].fillna(1.0)
    return out[['ts','market','hub_forward_hourly']]
##
def simulate_prices(hourly_hub: pd.DataFrame, basis_stats: pd.DataFrame, n_scenarios: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hh = hourly_hub.copy()
    hh['month'] = hh['ts'].to_period('M').dt.to_timestamp()
    hh['hour'] = hh['ts'].dt.hour + 1
    bs = basis_stats.copy()
    df = hh.merge(bs, on=['market','month','hour'], how='left')
    df['mean'] = df['mean'].fillna(0.0)
    df['std'] = df['std'].fillna(0.0)
    sims = []
    for s in range(n_scenarios):
        z = rng.standard_normal(len(df))
        basis = df['mean'] + df['std'] * z
        hub = df['hub_forward_hourly']  # keep simple: deterministic hourly hub from forwards
        node = hub - basis
        tmp = pd.DataFrame({'s': s, 'ts': df['ts'], 'market': df['market'], 'hub_rt': hub.values, 'node_rt': node.values})
        # DA simple proxy: smooth version of RT (e.g., 80% of deviations removed)
        tmp['hub_da'] = hub * 0.98 + 0.02 * tmp['hub_rt']
        tmp['node_da'] = node * 0.98 + 0.02 * tmp['node_rt']
        sims.append(tmp)
    sims = pd.concat(sims, ignore_index=True)
    return sims
