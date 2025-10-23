
import os
import pandas as pd
from typing import Dict, List
import numpy as np
import re

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

def _parse_month_col(s: pd.Series) -> pd.Series:
    # strip spaces and trailing punctuation (commas)
    s = s.astype(str).str.strip().str.replace(r"[,\s]+$", "", regex=True)
    # try common formats
    parsed = pd.to_datetime(s, format="%b-%y", errors="coerce")         # Jan-26 -> 2026-01 (uses 1969-2068 window)
    parsed = parsed.fillna(pd.to_datetime(s, format="%b-%Y", errors="coerce"))
    parsed = parsed.fillna(pd.to_datetime(s, format="%Y-%m", errors="coerce"))
    parsed = parsed.fillna(pd.to_datetime(s, errors="coerce"))          # last resort
    if parsed.isna().any():
        bad = s[parsed.isna()].unique().tolist()[:5]
        raise ValueError(f"Unparseable Month values (e.g.): {bad}")
    return parsed.dt.to_period("M").dt.to_timestamp()

def _clean_money(s: pd.Series) -> pd.Series:
    # remove $, commas, spaces and cast to float
    return (
        s.astype(str)
         .str.replace(r"[^\d\.\-]", "", regex=True)
         .replace({"": np.nan})
         .astype(float)
    )

_number_pat = re.compile(r"\(?-?\d+(?:\.\d+)?\)?")

def _extract_first_number(val) -> float:
    """
    Parse the FIRST numeric token from a messy string like:
      "$56.43 "    -> 56.43
      "($0.05)"    -> -0.05
      "12.3 14.5"  -> 12.3   (takes first)
    Returns np.nan if none found.
    """
    s = str(val)
    m = _number_pat.search(s)
    if not m:
        return np.nan
    tok = m.group(0)
    neg = tok.startswith("(") and tok.endswith(")")
    tok = tok.strip("()")
    try:
        x = float(tok)
        return -x if neg else x
    except ValueError:
        return np.nan

def _to_numeric_first(series: pd.Series) -> pd.Series:
    return series.apply(_extract_first_number).astype(float)

def _parse_month_col(s: pd.Series) -> pd.Series:
    # Trim trailing commas/spaces (e.g., "Jan-26, ")
    s = s.astype(str).str.strip().str.replace(r"[,\s]+$", "", regex=True)
    # Try common formats, coercing then error if still NaT
    parsed = pd.to_datetime(s, format="%b-%y", errors="coerce")       # Jan-26
    parsed = parsed.fillna(pd.to_datetime(s, format="%b-%Y", errors="coerce"))
    parsed = parsed.fillna(pd.to_datetime(s, format="%Y-%m", errors="coerce"))
    parsed = parsed.fillna(pd.to_datetime(s, errors="coerce"))
    if parsed.isna().any():
        bad = s[parsed.isna()].unique().tolist()[:5]
        raise ValueError(f"Unparseable Month values (e.g.): {bad}")
    return parsed.dt.to_period("M").dt.to_timestamp()

def _clean_money(s: pd.Series) -> pd.Series:
    # "$66.49 " -> 66.49 ; "($0.05)" -> -0.05 ; " 1,234.50" -> 1234.50
    out = (
        s.astype(str)
         .str.replace(r"[^\d\.\-\(\)]", "", regex=True)
         .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    out = out.replace({"": np.nan})
    return out.astype(float)


def load_historical_csvs(assets: List[dict], rename_map: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for a in assets:
        path = _resolve_path(a['historical_csv'])
        df = pd.read_csv(path)

        needed = [rename_map[k] for k in ['date','he','peak_flag','gen_mwh','rt_node','rt_hub','da_node','da_hub']]
        _validate_columns(df, needed, f"Historical CSV {path}")

        # rename to canonical keys
        df = _apply_hist_renames(df, rename_map)

        # fill asset/market if absent
        if 'asset' not in df.columns:
            df['asset'] = a['name']
        else:
            df['asset'] = df['asset'].fillna(a['name'])
        if 'market' not in df.columns:
            df['market'] = a['market']
        else:
            df['market'] = df['market'].fillna(a['market'])

        # --- NEW: clean numerics ---
        for col in ['gen_mwh', 'rt_hub', 'rt_node', 'da_hub', 'da_node']:
            if col in df.columns:
                df[col] = _to_numeric_first(df[col])

        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_forwards_csvs(assets: List[dict], cols_map: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for a in assets:
        path = _resolve_path(a['forwards_csv'])
        df = pd.read_csv(path)

        # flexible header rename
        ren = {}
        for key in ['month','peak','offpeak']:
            expect = cols_map[key]
            if expect in df.columns:
                ren[expect] = key
            else:
                # case-insensitive fallback
                matches = [c for c in df.columns if c.strip().lower() == expect.strip().lower()]
                if matches:
                    ren[matches[0]] = key
        if ren:
            df = df.rename(columns=ren)

        _validate_columns(df, ['month','peak','offpeak'], f"Forwards CSV {path}")

        # ⇩⇩ NEW: clean & parse ⇩⇩
        df['month'] = _parse_month_col(df['month'])
        df['peak'] = _clean_money(df['peak'])
        df['offpeak'] = _clean_money(df['offpeak'])

        # attach market if missing
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
        hist = pd.read_excel(config['data_excel'], sheet_name=config['historical_sheet'])
        hist = _apply_hist_renames(hist, config['rename_map'])

        # clean numerics for historical (Excel path)
        for col in ['gen_mwh', 'rt_hub', 'rt_node', 'da_hub', 'da_node']:
            if col in hist.columns:
                hist[col] = _to_numeric_first(hist[col])

        fwd = pd.read_excel(config['data_excel'], sheet_name=config['forwards_sheet'])
        fwd = fwd.rename(columns={config['forwards_cols']['month']: 'month',
                                  config['forwards_cols']['market']: 'market',
                                  config['forwards_cols']['peak']: 'peak',
                                  config['forwards_cols']['offpeak']: 'offpeak'})
        # (use your robust month + money cleaners for forwards here)
        fwd['month'] = _parse_month_col(fwd['month'])
        fwd['peak'] = _clean_money(fwd['peak'])
        fwd['offpeak'] = _clean_money(fwd['offpeak'])
    return hist, fwd
