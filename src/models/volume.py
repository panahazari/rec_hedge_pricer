
import numpy as np
import pandas as pd

def bootstrap_generation(cal: pd.DataFrame, vol_tbl: pd.DataFrame, n_scenarios: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _, r in cal.iterrows():
        month = r['month']
        hour = r['he']
        for _, vr in vol_tbl[(vol_tbl['month']==month) & (vol_tbl['hour']==hour)].iterrows():
            samples = vr['samples']
            if len(samples)==0:
                g = 0.0
            else:
                g = rng.choice(samples)
            rows.append({'asset': vr['asset'], 'market': vr['market'], 'ts': r['ts'], 'gen_mwh': g})
    df = pd.DataFrame(rows)
    # repeat for scenarios by stacking
    df = pd.concat([df.assign(s=sc) for sc in range(n_scenarios)], ignore_index=True)
    return df
