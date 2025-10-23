
import pandas as pd

NERC_HOLIDAYS = None  # Keep simple, user data already flags Peak/OP

def make_calendar_index(start_year: int, end_year: int, tz: str = None) -> pd.DataFrame:
    idx = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31 23:00:00", freq="H")
    df = pd.DataFrame({'ts': idx})
    df['date'] = df['ts'].dt.date
    df['he'] = df['ts'].dt.hour + 1
    df['month'] = df['ts'].dt.to_period('M').dt.to_timestamp()
    df['weekday'] = df['ts'].dt.weekday
    return df
