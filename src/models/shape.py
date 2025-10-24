
import numpy as np
import pandas as pd



def build_calendar(start: str | pd.Timestamp,
                   end: str | pd.Timestamp,
                   tz: str = "America/Chicago",
                   markets: list[str] = None) -> pd.DataFrame:
    """
    Build an hourly valuation calendar between start and end (inclusive),
    timezone-aware (DST-safe), replicated across the given markets.

    Parameters
    ----------
    start : str or pd.Timestamp
        e.g., "2026-01-01 00:00"
    end   : str or pd.Timestamp
        e.g., "2030-12-31 23:00"
    tz    : str
        IANA timezone (e.g., "America/Chicago", "America/Los_Angeles")
    markets : list[str]
        e.g., ["ERCOT", "MISO", "CAISO"] or ["ERCOT"]

    Returns
    -------
    pd.DataFrame with columns:
        - market : str
        - ts     : pd.Timestamp (tz-aware)
        - mon    : int month-of-year (1..12)
        - hour   : int hour-ending (1..24)
    Notes
    -----
    * Uses tz-aware hourly frequency so DST spring/fall transitions are handled
      (23/25-hour days will be represented correctly).
    * `shape_monthly_to_hourly` only requires ['market','ts']; it will compute
      Peak/Off-peak and other fields itself. `mon` and `hour` are provided for
      convenience in other steps.
    """
    if markets is None or len(markets) == 0:
        markets = ["ERCOT"]

    # Hourly range with timezone; inclusive of both endpoints
    hours = pd.date_range(start=pd.to_datetime(start).tz_localize(tz) if pd.to_datetime(start).tzinfo is None else pd.to_datetime(start).tz_convert(tz),
                          end=pd.to_datetime(end).tz_localize(tz) if pd.to_datetime(end).tzinfo is None else pd.to_datetime(end).tz_convert(tz),
                          freq="H")

    # Cartesian product: markets x hours
    idx = pd.MultiIndex.from_product([markets, hours], names=["market", "ts"])
    cal = idx.to_frame(index=False)

    # Convenience keys
    cal["mon"]  = cal["ts"].dt.month
    cal["hour"] = cal["ts"].dt.hour + 1  # HE 1..24

    return cal.sort_values(["market", "ts"]).reset_index(drop=True)
# ---------------------------------------------------------------------
# Helper: parse money-like strings to float
def _to_float(s):
    if pd.api.types.is_numeric_dtype(getattr(s, "dtype", None)):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str)
    s = s.str.replace(r"[\$,]", "", regex=True)      # drop $ and commas
    s = s.str.replace(r"\((.*?)\)", r"-\1", regex=True)  # (123.45) -> -123.45
    return pd.to_numeric(s, errors="coerce")

# Helper: NERC holiday calendar for a given year (dates only)
def _nerc_holidays(year: int) -> set:
    # New Year's Day (Jan 1)
    ny = pd.Timestamp(year=year, month=1, day=1).date()
    # Memorial Day (last Monday of May)
    may = pd.date_range(f"{year}-05-01", f"{year}-05-31", freq="D")
    memorial = may[may.weekday == 0][-1].date()
    # Labor Day (first Monday of September)
    sep = pd.date_range(f"{year}-09-01", f"{year}-09-30", freq="D")
    labor = sep[sep.weekday == 0][0].date()
    # Thanksgiving (4th Thursday of November)
    nov = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
    thx = nov[nov.weekday == 3][3].date()
    # Christmas Day (Dec 25)
    xmas = pd.Timestamp(year=year, month=12, day=25).date()
    return {ny, memorial, labor, thx, xmas}

def _compute_is_peak_if_missing(cal: pd.DataFrame) -> pd.DataFrame:
    cal = cal.copy()
    if "is_peak" in cal.columns:
        return cal
    # Compute HE (1..24)
    cal["hour"] = cal["ts"].dt.hour + 1
    # Weekday & holiday flags by local date
    cal["date"] = cal["ts"].dt.date
    years = sorted(pd.Series(cal["ts"].dt.year.unique(), dtype=int).tolist())
    hol = set()
    for y in years:
        hol |= _nerc_holidays(int(y))
    is_weekday = cal["ts"].dt.weekday.between(0, 4)  # Mon–Fri
    is_he_peak = cal["hour"].between(7, 22)          # HE 7..22
    is_holiday = cal["date"].isin(hol)
    cal["is_peak"] = is_weekday & is_he_peak & (~is_holiday)
    cal = cal.drop(columns=["date"])
    return cal

# ---------------------------------------------------------------------
def shape_monthly_to_hourly(fwd_all: pd.DataFrame,
                            shape_tbl: pd.DataFrame,
                            cal: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monthly Peak/Off-Peak hub forwards into an hourly hub curve H_t
    using historical hourly shape factors S[mon,h], preserving each month’s
    Peak and Off-Peak average prices (Mon–Fri HE 7–22 excl. NERC holidays).

    Inputs
    ------
    fwd_all: DataFrame with at least
        ['market', 'month', 'peak', 'offpeak']  (column names case/space-insensitive)
        - 'month' is month bucket (timestamp or string like 'Jan-26')
        - 'peak' and 'offpeak' are monthly means in $/MWh
    shape_tbl: DataFrame with
        ['market','mon','hour','shape']  where mon=1..12, hour=1..24
    cal: DataFrame with future hourly calendar, needs
        ['market','ts'] and optionally 'is_peak'
        (if missing, it will be computed using NERC holiday rules)

    Returns
    -------
    DataFrame with columns:
        ['market','ts','mon','hour','is_peak','hub_forward_hourly']
    """
    # --- Normalize forwards columns ---
    fwd = fwd_all.copy()
    # make lowercase/strip spaces for flexible rename
    fwd.columns = [c.strip().lower().replace(" ", "") for c in fwd.columns]
    colmap = {}
    if "peak" in fwd.columns: colmap["peak"] = "peak"
    if "offpeak" in fwd.columns: colmap["offpeak"] = "offpeak"
    if "offpeak" not in colmap and "off_peak" in fwd.columns:
        colmap["off_peak"] = "offpeak"
    if "month" not in fwd.columns:
        raise KeyError("Forwards must include a 'month' column (e.g., 'Jan-26').")
    fwd = fwd.rename(columns=colmap)
    # Parse 'month' to first-of-month timestamp
    fwd["month"] = pd.to_datetime(fwd["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    # Coerce prices to float
    for c in ["peak", "offpeak"]:
        if c not in fwd.columns:
            raise KeyError(f"Forwards missing '{c}' column.")
        fwd[c] = _to_float(fwd[c])

    # Require market column
    if "market" not in fwd.columns:
        if "mkt" in fwd.columns:
            fwd = fwd.rename(columns={"mkt": "market"})
        else:
            raise KeyError("Forwards must include 'market' column.")

    # --- Calendar keys ---
    cal = cal.copy()
    if "market" not in cal.columns:
        if "mkt" in cal.columns:
            cal = cal.rename(columns={"mkt": "market"})
        else:
            raise KeyError("Calendar must include 'market' column.")
    cal["ts"]  = pd.to_datetime(cal["ts"])
    cal["mon"] = cal["ts"].dt.month
    cal["hour"] = cal["ts"].dt.hour + 1
    cal["month"] = cal["ts"].dt.to_period("M").dt.to_timestamp()

    # is_peak (compute if missing)
    cal = _compute_is_peak_if_missing(cal)

    # --- Merge shape by (market, mon, hour) ---
    need_shape = {"market","mon","hour","shape"}
    if not need_shape.issubset(set(shape_tbl.columns)):
        raise KeyError(f"shape_tbl must have columns {sorted(need_shape)}")
    shaped = cal.merge(shape_tbl[["market","mon","hour","shape"]],
                       on=["market","mon","hour"], how="left")
    shaped["shape"] = shaped["shape"].fillna(1.0)

    # --- Attach monthly forwards by (market, month) ---
    fwd2 = fwd[["market","month","peak","offpeak"]].copy()
    shaped = shaped.merge(fwd2, on=["market","month"], how="left")

    # If some months missing, fill with nearest available (optional)
    # Here we leave NaNs; users should ensure full coverage.

    # --- Compute scale factors per (market, month, bucket) to preserve means ---
    # mean S over each bucket within the month
    grp = (shaped
           .groupby(["market","month","is_peak"], as_index=False)["shape"]
           .mean()
           .rename(columns={"shape":"shape_mean"}))
    # pick forward by bucket
    grp = grp.merge(fwd2, on=["market","month"], how="left")
    grp["fwd_bucket"] = np.where(grp["is_peak"], grp["peak"], grp["offpeak"])
    # k = forward_mean / mean(shape)
    grp["k"] = np.where(grp["shape_mean"] != 0, grp["fwd_bucket"] / grp["shape_mean"], 0.0)
    grp = grp[["market","month","is_peak","k"]]

    shaped = shaped.merge(grp, on=["market","month","is_peak"], how="left")

    # --- Hourly hub forward ---
    shaped["hub_forward_hourly"] = shaped["shape"] * shaped["k"]
    shaped["hub_forward_hourly"] = shaped["hub_forward_hourly"].fillna(0.0)

    return shaped[["market","ts","mon","hour","is_peak","hub_forward_hourly"]].sort_values(["market","ts"]).reset_index(drop=True)


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
