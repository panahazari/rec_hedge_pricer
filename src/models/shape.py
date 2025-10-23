
import pandas as pd
import numpy as np


def hub_hourly_shape(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    df['rt_hub'] = pd.to_numeric(df['rt_hub'], errors='coerce')
    df['market'] = df['market'].astype(str).str.strip().str.upper()

    # Normalize time keys
    df['date']  = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['moy']   = df['month'].dt.month  # <-- month-of-year only

    he = pd.to_numeric(df['he'], errors='coerce').fillna(0).astype(int)
    df['hour'] = ((he - 1) % 24) + 1    # 1..24

    # Monthly (calendar month) average then normalize to get shape
    m_avg = df.groupby(['market','month'])['rt_hub'].mean().rename('m_avg')
    df = df.join(m_avg, on=['market','month'])
    df['shape'] = np.where(df['m_avg'] != 0, df['rt_hub'] / df['m_avg'], 1.0)

    # Average shape by (market, month-of-year, hour)
    shp = (df.groupby(['market','moy','hour'])['shape']
             .mean()
             .reset_index())

    # returns: market, moy (1..12), hour (1..24), shape
    return shp


def basis_stats(hist: pd.DataFrame, price_col_hub: str = 'rt_hub', price_col_node: str = 'rt_node') -> pd.DataFrame:
    df = hist.copy()
    df['market'] = df['market'].astype(str).str.strip().str.upper()
    df['date']   = pd.to_datetime(df['date'])
    df['month']  = df['date'].dt.to_period('M').dt.to_timestamp()
    df['moy']    = df['month'].dt.month
    he = pd.to_numeric(df['he'], errors='coerce').fillna(0).astype(int)
    df['hour']   = ((he - 1) % 24) + 1

    # basis = hub - node
    for c in [price_col_hub, price_col_node]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['basis'] = df[price_col_hub] - df[price_col_node]

    bs = (df.groupby(['market','moy','hour'])['basis']
            .agg(mean='mean', std='std')
            .reset_index())

    # returns: market, moy, hour, mean, std
    return bs


def volume_bootstrap_table(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['hour'] = df['he']
    tbl = df.groupby(['asset','market','month','hour'])['gen_mwh'].apply(list).reset_index(name='samples')
    return tbl
