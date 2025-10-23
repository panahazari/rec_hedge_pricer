
import pandas as pd
import numpy as np

def hub_hourly_shape(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    # --- NEW: enforce numeric in case upstream files change ---
    for col in ['rt_hub']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['hour'] = df['he']
    g = df.groupby(['market','month'])['rt_hub'].mean().rename('m_avg')
    df = df.join(g, on=['market','month'])
    df['shape'] = np.where(df['m_avg'] != 0, df['rt_hub'] / df['m_avg'], 1.0)
    shp = df.groupby(['market','month','hour'])['shape'].mean().reset_index()
    return shp

def basis_stats(hist: pd.DataFrame, price_col_hub: str, price_col_node: str) -> pd.DataFrame:
    df = hist.copy()
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['hour'] = df['he']
    df['basis'] = df[price_col_hub] - df[price_col_node]
    out = df.groupby(['market','hour','month'])['basis'].agg(['mean','std']).reset_index()
    return out

def volume_bootstrap_table(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['hour'] = df['he']
    tbl = df.groupby(['asset','market','month','hour'])['gen_mwh'].apply(list).reset_index(name='samples')
    return tbl
