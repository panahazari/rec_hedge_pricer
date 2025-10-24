# src/models/volume.py
import numpy as np
import pandas as pd

def build_volume_table(hist_all: pd.DataFrame, asset: str, market: str) -> pd.DataFrame:
    """
    Build historical generation samples keyed by (asset, market, mon=1..12, hour=HE 1..24).
    Returns: ['asset','market','mon','hour','samples'] where 'samples' is a numpy array.
    """
    df = hist_all.loc[(hist_all['asset'] == asset) & (hist_all['market'] == market)].copy()
    df['gen_mwh'] = pd.to_numeric(df['gen_mwh'], errors='coerce').fillna(0.0)
    df['mon']  = df['date'].dt.month            # 1..12
    df['hour'] = df['he']                       # HE 1..24 (ensure your 'he' is 1..24)

    tbl = (df.groupby(['asset','market','mon','hour'])['gen_mwh']
             .apply(lambda s: np.asarray(s.values, dtype='float32'))
             .reset_index(name='samples'))
    return tbl

def bootstrap_generation(cal: pd.DataFrame,
                         vol_tbl: pd.DataFrame,
                         n_scenarios: int,
                         rng: np.random.Generator) -> pd.DataFrame:
    base = cal.copy()
    base['ts']   = pd.to_datetime(base['ts'])
    base['mon']  = base['ts'].dt.month
    base['hour'] = base['ts'].dt.hour + 1

    need = {'asset','market','mon','hour','samples'}
    miss = need - set(vol_tbl.columns)
    if miss:
        raise KeyError(f"vol_tbl missing {sorted(miss)}; has {list(vol_tbl.columns)}")

    keys = vol_tbl[['asset','market']].drop_duplicates()
    if len(keys) != 1:
        raise ValueError(f"vol_tbl should contain exactly one (asset, market); got {len(keys)}")
    asset = keys['asset'].iloc[0]
    market = keys['market'].iloc[0]
    base = base.assign(asset=asset, market=market)

    base = base.merge(vol_tbl, on=['asset','market','mon','hour'], how='left')

    S = int(n_scenarios)
    N = len(base)
    base = base.assign(bucket_id=(base['mon'].astype('uint8') - 1) * 24 + (base['hour'].astype('uint8') - 1))

    # ✅ fixed: indices is already ndarray
    bucket_to_rows = {b: np.asarray(idx, dtype=np.int64)
                      for b, idx in base.groupby('bucket_id').indices.items()}

    s_out  = np.repeat(np.arange(S, dtype=np.int32), N)
    ts_out = np.tile(base['ts'].values, S)
    gen_out = np.empty(S * N, dtype='float32')

    for b, rows in bucket_to_rows.items():
        samples = base.loc[rows, 'samples'].iloc[0]
        if not isinstance(samples, np.ndarray) or samples.size == 0:
            idx_flat = np.concatenate([rows + k*N for k in range(S)])
            gen_out[idx_flat] = 0.0
            continue
        K = rows.size
        idx = rng.integers(0, samples.size, size=(S, K), dtype=np.int64)
        draws = samples[idx]
        for k in range(S):
            gen_out[k*N + rows] = draws[k]

    return pd.DataFrame({
        's': s_out,
        'ts': ts_out,
        'asset': asset,
        'market': market,
        'gen_mwh': gen_out
    })
