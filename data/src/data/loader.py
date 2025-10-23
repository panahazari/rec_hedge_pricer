
import os
import pandas as pd
from typing import Dict, List

EXPECTED_HIST_COLS = ["Date","HE","P/OP","Gen","RT Busbar","RT Hub","DA Busbar","DA Hub"]
EXPECTED_FWD_COLS  = ["Month","Peak","Off Peak"]

def _resolve_path(path: str) -> str:
    # Try given path; if not found, swap data/raw <-> data/; then try basenames in both
    if os.path.exists(path):
        return path
    alt = path.replace("data/raw/","data/") if "data/raw/" in path else path.replace("data/","data/raw/")
    if os.path.exists(alt):
        return alt
    base = os.path.basename(path)
    for candidate in [os.path.join("data", base), os.path.join("data","raw", base)]:
        if os.path.exists(candidate):
            return candidate
    return path

def _apply_hist_renames(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    cols = {v:k for k,v in rename_map.items()}
    df = df.rename(columns=cols)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'he' in df.columns:
        df['he'] = df['he'].astype(int)
    return df

def _validate_columns(df: pd.DataFrame, expected: List[str], context: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing columns {missing}. Found {list(df.columns)}")

def load_historical_csvs(assets: List[dict], rename_map: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for a in assets:
        path = _resolve_path(a['historical_csv'])
        df = pd.read_csv(path)
        needed = [rename_map[k] for k in ['date','he','peak_flag','gen_mwh','rt_node','rt_hub','da_node','da_hub']]
        _validate_columns(df, needed, f"Historical CSV {path}")
        df = _apply_hist_renames(df, rename_map)
        if 'asset' not in df.columns:
            df['asset'] = a['name']
        else:
            df['asset'] = df['asset'].fillna(a['name'])
        if 'market' not in df.columns:
            df['market'] = a['market']
        else:
            df['market'] = df['market'].fillna(a['market'])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def load_forwards_csvs(assets: List[dict], cols_map: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for a in assets:
        path = _resolve_path(a['forwards_csv'])
        df = pd.read_csv(path)
        needed = [cols_map['month'], cols_map['peak'], cols_map['offpeak']]
        _validate_columns(df, needed, f"Forwards CSV {path}")
        df = df.rename(columns={cols_map['month']: 'month',
                                cols_map['peak']: 'peak',
                                cols_map['offpeak']: 'offpeak'})
        df['month'] = pd.to_datetime(df['month']).dt.to_period('M').dt.to_timestamp()
        if 'market' not in df.columns:
            df['market'] = a['market']
        else:
            df['market'] = df['market'].fillna(a['market'])
        frames.append(df[['month','market','peak','offpeak']])
    return pd.concat(frames, ignore_index=True)

def load_data(config: dict):
    fmt = config.get('data_format', 'excel')
    if fmt == 'per_asset_csv':
        hist = load_historical_csvs(config['assets'], config['rename_map'])
        fwd  = load_forwards_csvs(config['assets'], config['forwards_cols'])
    else:
        # Excel back-compat
        hist = pd.read_excel(config['data_excel'], sheet_name=config['historical_sheet'])
        hist = _apply_hist_renames(hist, config['rename_map'])
        fwd = pd.read_excel(config['data_excel'], sheet_name=config['forwards_sheet'])
        fwd = fwd.rename(columns={config['forwards_cols']['month']: 'month',
                                  config['forwards_cols']['market']: 'market',
                                  config['forwards_cols']['peak']: 'peak',
                                  config['forwards_cols']['offpeak']: 'offpeak'})
        fwd['month'] = pd.to_datetime(fwd['month']).dt.to_period('M').dt.to_timestamp()
    return hist, fwd
