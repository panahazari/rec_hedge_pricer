
import pandas as pd
import numpy as np

def merchant_revenue(gen: pd.Series, price: pd.Series) -> pd.Series:
    return gen * price

def cfd_as_generated(gen: pd.Series, ref_price: pd.Series, fixed_price: float, negative_rule: str = "include") -> pd.Series:
    ref = ref_price.copy()
    if negative_rule == "zero":
        ref = ref.where(ref >= 0, 0.0)
    return (fixed_price - ref) * gen

def pnl_hedged(gen: pd.Series, sell_price: pd.Series, ref_price: pd.Series, fixed_price: float, negative_rule: str = "include") -> pd.Series:
    m = merchant_revenue(gen, sell_price)
    h = cfd_as_generated(gen, ref_price, fixed_price, negative_rule)
    return m + h
