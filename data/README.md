
# Merchant Hedge Pricer

Valuation toolkit to compute risk‑adjusted 5‑year fixed offtake prices for three merchant assets (2 wind, 1 solar) across ERCOT, MISO, CAISO. 
Focus: energy value only. Ignore capex/O&M, taxes, capacity, RECs.

## What this does
- Loads historical hourly generation and prices (RT/DA, busbar & hub) from Excel.
- Shapes monthly Peak/Off‑Peak hub forwards to hourly.
- Models hub→node basis and volume using simple empirical distributions.
- Simulates RT/DA price paths and generation.
- Compares merchant vs fixed‑price offtake (as‑generated CfD) under four products:
  - RT Hub‑settled, RT Busbar‑settled, DA Hub‑settled, DA Busbar‑settled.
- Solves for a **5‑year flat price** such that **P(hedge ≥ merchant) ≥ P‑level** (default P75).
- Exports tables: expected generation by month/Peak-OP, fixed prices, risk breakdown.

## Repo layout
```
merchant-hedge-pricer/
  configs/
    project.yaml              # edit paths and parameters here
  data/
    raw/input_data.xlsx       # put your Excel here (copied if you uploaded)
    processed/                # generated
  models/                     # saved model artifacts
  notebooks/                  # notebooks will be added later
  reports/                    # slides/figures
  results/                    # csv outputs
  src/
    data/loader.py
    features/calendar.py
    models/shape.py
    models/volume.py
    models/price.py
    pricing/hedge.py
    pricing/solve_price.py
    train/train.py
    utils/persistence.py
  tests/
  requirements.txt
  README.md
  LICENSE
  .gitignore
```

## Quick start
1) **Install**
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) **Configure** in `configs/project.yaml` (paths, assets, parameters).

3) **Train** simple empirical models and save artifacts:
```python
from src.train.train import train_all
train_all(config_path="configs/project.yaml")
```

4) **Generate prices and reports** (from a notebook later):
```python
from src.pricing.solve_price import solve_all_assets
solve_all_assets(config_path="configs/project.yaml")
```

Artifacts and CSVs land in `models/` and `results/`.

## Inputs (Excel)
Expected columns in the historical tab:
- `Date`, `HE`, `P/OP`, `Asset`, `Market`, `Gen`, 
- `RT Busbar`, `RT Hub`, `DA Busbar`, `DA Hub`

Forward curve tab:
- `Month` (YYYY-MM), `Market`, `Peak`, `Off Peak`

If your file differs, adjust `configs/project.yaml` or loader mappings.

## Methods (short)
- **Hourly shaping:** Monthly Peak/OP forwards are split to hourly using historical hub shapes by (month, weekday, HE). 
- **Basis:** Empirical hub−node by (month, HE). 
- **Volume:** Bootstrap generation by (month, HE).
- **Simulation:** Draw residuals and basis; produce RT/DA & node/hub hourly paths for 2026–2030.
- **Decision rule:** Find flat price P where the 25th percentile of (Hedge − Merchant) ≥ 0 (P75).

## Outputs
- `results/expected_generation.csv` (by month, P/OP)
- `results/fixed_prices.csv` (four products per asset)
- `results/risk_breakdown.csv` (mean/std/p5/p25 expected shortfall proxies)

## Notes
- Negative‑price rule configurable: include or zero settlements when LMP<0.
- Settlement point: hub or node.
- Price set: RT or DA.
- Extend with alternative structures (block hedges) in `pricing/hedge.py`.



## Using per-asset CSV files (6 files)
You can split the Excel into 6 CSVs (per asset: one historical + one forwards). Point the paths in `configs/project.yaml`:
```yaml
data_format: "per_asset_csv"

assets:
  - name: "Wind_A"
    market: "ERCOT"
    historical_csv: "data/raw/wind_a_historical.csv"
    forwards_csv:   "data/raw/wind_a_forwards.csv"
  - name: "Wind_B"
    market: "MISO"
    historical_csv: "data/raw/wind_b_historical.csv"
    forwards_csv:   "data/raw/wind_b_forwards.csv"
  - name: "Solar_C"
    market: "CAISO"
    historical_csv: "data/raw/solar_c_historical.csv"
    forwards_csv:   "data/raw/solar_c_forwards.csv"
```

**Historical CSV expected columns:**
- `Date`, `HE`, `P/OP`, `Gen`, `RT Busbar`, `RT Hub`, `DA Busbar`, `DA Hub`  
  (If `Asset`/`Market` columns are missing, the loader fills them from the config.)

**Forwards CSV expected columns:**
- `Month` (YYYY-MM), `Peak`, `Off Peak`  
  (If `Market` is missing, the loader fills it from the config.)

> Backwards compatible: leaving `data_format` unset (or `excel`) uses the single-Excel loader with `historical_sheet`/`forwards_sheet` as before.
