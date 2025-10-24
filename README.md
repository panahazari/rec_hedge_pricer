# Pricing & Hedging — Wind A, Wind B, Solar

This project prices as-generated fixed-for-floating swaps (VPPAs) for **Wind A**, **Wind B**, and **Solar** across four settlement choices: **RT Hub**, **RT Node**, **DA Hub**, and **DA Node**. It also answers practical questions about volume/price risk, negative prices, market choice, and data needs.

---

## Methodology (what we did and how)

We start from **monthly hub forwards** \(F_m\) and turn them into realistic **hourly** price paths at both **hub** and **node** in **RT** and **DA**. We then simulate hourly **generation** for each asset and compute the **generation‑weighted** settlement price. Finally, we set the fixed price \(K_p\) so that the hedge breaks even at protection level \(p\) (e.g., \(p=0.75\)).

### 1) From monthly forwards to hourly hub prices
For hour \(h\) in month \(m\):
\[
P^{\mathrm{RT,hub}}_{m,h} = F_m \cdot s_{m,h} + \omega_{m,h},
\quad \text{with } \frac{1}{H_m}\sum_{h\in m} s_{m,h} = 1,\ \ \mathbb{E}[\omega_{m,h}] = 0.
\]
- \(s_{m,h}\): normalized intra‑month shape (keeps seasons and hour‑of‑day).
- \(\omega_{m,h}\): residual volatility drawn from month×hour histories (can go negative).

Day‑ahead hub adds a DA–RT spread:
\[
P^{\mathrm{DA,hub}}_{m,h} = P^{\mathrm{RT,hub}}_{m,h} + \Delta_{m,h}.
\]

### 2) From hub to node (basis)
Add node–hub basis \(B_{m,h}\) to get node prices:
\[
P^{\mathrm{RT,node}}_{m,h} = P^{\mathrm{RT,hub}}_{m,h} + B_{m,h}, 
\qquad
P^{\mathrm{DA,node}}_{m,h} = P^{\mathrm{DA,hub}}_{m,h} + B_{m,h}.
\]
We co‑sample \(\omega,\Delta,B\) by month×hour so that bad basis and DA–RT spread tend to show up at the right times.

### 3) Forecasting hourly generation
- **Solar**: draw irradiance/clear‑sky state and temperature, map to capacity factor \(\mathrm{CF}^{\mathrm{solar}}_t\) (bounded \([0,1]\)), then \(G_t = \mathrm{CF}^{\mathrm{solar}}_t \times \text{AC rating}\).
- **Wind**: draw wind speed \(V_t\) (Weibull by month×hour), push through the power curve to get \(\mathrm{CF}^{\mathrm{wind}}_t\), then \(G_t = \eta_{\mathrm{avail}} \cdot \mathrm{CF}^{\mathrm{wind}}_t \times \text{AC rating}\).

Where relevant, we co‑sample weather and basis in the same month×hour bucket so high‑output hours can coincide with congestion/negative prices.

### 4) Pricing the as‑generated swap
For market \(X \in \{\text{RT Hub}, \text{RT Node}, \text{DA Hub}, \text{DA Node}\}\) with hourly price \(P^X_t\):
\[
\bar{P}^X = \frac{\sum_t G_t\, P^X_t}{\sum_t G_t}.
\]
With negative prices **included**, the fixed price at protection level \(p\) is the \(p\)-quantile of \(\bar{P}^X\):
\[
K_p = Q_p\!\left(\bar{P}^X\right), 
\qquad 
\mu^X = \mathbb{E}[\bar{P}^X], 
\qquad 
\text{Risk Premium } \mathrm{RP}^X = K_p - \mu^X.
\]
If a contract **excludes negative prices**, replace \(G_t\) by \(G_t\mathbf{1}\{P^X_t\ge 0\}\) in all formulas (removes worst tail → increases \(K_p\)).

---

## Core Results (p = 0.75, negative prices included)

**Generation‑weighted expected price \(\mu\)**, **fixed price \(K_{0.75}\)**, and **risk premium \(RP\)** in \$/MWh (rounded).

### Solar
| Market  | \(\mu\) | \(K_{0.75}\) | RP |
|---|---:|---:|---:|
| RT Hub  | 36.09 | 36.12 | 0.03 |
| RT Node | 36.47 | 37.62 | 1.15 |
| DA Hub  | 36.86 | 41.44 | 4.59 |
| DA Node | 36.85 | 43.24 | 6.39 |

### Wind A
| Market  | \(\mu\) | \(K_{0.75}\) | RP |
|---|---:|---:|---:|
| RT Hub  | 57.30 | 57.38 | 0.09 |
| RT Node | 48.33 | 54.50 | 6.17 |
| DA Hub  | 64.75 | 72.59 | 7.84 |
| DA Node | 55.51 | 70.03 | 14.52 |

### Wind B
| Market  | \(\mu\) | \(K_{0.75}\) | RP |
|---|---:|---:|---:|
| RT Hub  | 45.25 | 45.30 | 0.04 |
| RT Node | 39.95 | 41.02 | 1.06 |
| DA Hub  | 45.31 | 48.10 | 2.79 |
| DA Node | 36.77 | 42.95 | 6.19 |

### Sensitivity (RT Hub only, for brevity)

| Asset  | K₀.₅₀ | K₀.₇₅ | K₀.₉₀ |
|---|---:|---:|---:|
| Wind A | 57.31 | 57.38 | 57.47 |
| Wind B | 45.25 | 45.30 | 45.41 |
| Solar  | 36.09 | 36.12 | 36.15 |

---

## What the numbers mean (plain language)

We handle **price** and **volume** by splitting the problem. The hub‑level market price is cheap to hedge: the RT Hub risk premium is near zero for all three assets, so locking an **RT Hub** as‑generated swap efficiently covers the market level. The hard (expensive) risks are **basis** (node–hub) and **DA–RT spread**, which show up as large risk premia at **DA Node**—especially for **Wind A**—because congestion and negative prices tend to hit exactly when the asset produces.

Negative prices widen the left tail and raise premia at node/DA. That’s why **DA Node** is the priciest place to lock a fixed price. If your PPA **won’t take generation when prices are negative**, we just drop those hours from settlement; the left tail disappears and the **fair fixed price goes up** (you’re buying a safer hedge).

Some markets are simply more hedge‑friendly. **RT Hub** is consistently the easiest (lowest RP). **DA Node** is the hardest (highest RP). In a node that looks like **Wind A**, a 5‑year node‑settled price can be expensive enough that it’s reasonable to (a) hedge **at the hub** and manage basis separately (CRRs/FTRs or shorter‑dated basis swaps), or (b) keep a **merchant slice** while you wait for cheaper basis cover.

---

## Reproduce in short

1. **Prepare inputs**: monthly hub forwards \(F_m\); hourly shape \(s_{m,h}\); hourly DA–RT spreads \(\Delta_{m,h}\); hourly basis \(B_{m,h}\); generation model parameters.
2. **Simulate** hourly hub/node, RT/DA prices allowing negatives.
3. **Simulate** hourly generation for Solar, Wind A, Wind B.
4. **Compute** \(\bar{P}^X\), then \(K_p=Q_p(\bar{P}^X)\), \(\mu=\mathbb{E}[\bar{P}^X]\), \(\mathrm{RP}=K_p-\mu\).
5. **(Optional)** Re‑run with negatives excluded by applying \(\mathbf{1}\{P^X_t\ge 0\}\) to settlement hours.

---

## Repository layout (suggested)

```
.
├─ data/
│  ├─ forwards_monthly.csv
│  ├─ shapes_hourly.csv           # s_{m,h}
│  ├─ da_rt_spread_hourly.csv     # Δ_{m,h}
│  ├─ basis_hourly.csv            # B_{m,h}
│  ├─ generation_params/          # solar & wind model params
├─ src/
│  ├─ simulate_prices.py
│  ├─ simulate_generation.py
│  ├─ price_to_node.py
│  ├─ compute_fixed_price.py      # computes μ, K_p, RP
│  └─ utils.py
├─ notebooks/
│  └─ 01_run_scenarios.ipynb
├─ output/
│  ├─ results_p075.csv
│  └─ figures/
├─ README.md
└─ LICENSE
```

---

## Data used in this README

- Fixed prices and breakdowns at \(p=0.75\) (negative prices included) for **Wind A**, **Wind B**, **Solar** as provided.
- Sensitivities for \(p \in \{0.50, 0.75, 0.90\}\) at **RT Hub**.
