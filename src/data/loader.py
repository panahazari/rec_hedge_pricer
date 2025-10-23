
import pandas as pd
from typing import Dict

def load_historical(excel_path: str, sheet: str, rename_map: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    cols = {v:k for k,v in rename_map.items()}
    df = df.rename(columns=cols)
    # Basic cleanup
    df['date'] = pd.to_datetime(df['date'])
    df['he'] = df['he'].astype(int)
    if 'P/OP' in df.columns:
        pass
    return df

def load_forwards(excel_path: str, sheet: str, cols_map: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    df = df.rename(columns={cols_map['month']: 'month',
                            cols_map['market']: 'market',
                            cols_map['peak']: 'peak',
                            cols_map['offpeak']: 'offpeak'})
    df['month'] = pd.to_datetime(df['month']).dt.to_period('M').dt.to_timestamp()
    return df
