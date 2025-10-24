## Assumptions & Method (concise)

- **HPFC shaping:** Monthly Peak/OP hub forwards → hourly via normalized historical shapes by (market, month, hour). Bucket averages match the monthly forwards exactly.
- **Basis:** Expected hub→node basis = mean of (hub−node) by (market, month, hour) from history. Basis volatility is simulated (Gaussian with historical std).
- **DA–RT spreads:** Expected spread by (market, month, hour) from history; DA = RT + spread + noise.
- **Generation:** Volume risk via bootstrap of historical hourly generation by (asset, month, hour). (Implicitly captures curtailment/outages embedded in history.)
- **Settlement structures:** Four products: RT/DA × Hub/Node, modeled as **as-generated CfDs**.
- **Negative prices:** Configurable rule: `"include"` or `"zero"`. Breakdown exports both `P* (include)` and `P* (zero)`; `D` = delta between them.
- **Risk target:** Solve flat price `P*` so that **P(hedged ≥ merchant) ≥ P-level** (default P75). A **P-grid** (P50/P75/P90) is exported for sensitivity.

## Outputs added

- `results/price_breakdown.csv` — per asset/product waterfall:
  - `A_hub_capture`, `B_basis`, `C_da_rt`, `D_neg_rule`, `E_risk`, `P_star_include`, `P_star_zero`
- `results/fixed_prices_Pgrid.csv` — 5-year flat prices across P-levels (e.g., 0.50, 0.75, 0.90).

## Interpreting the breakdown

- `A` = gen-weighted hub value from shaped forwards  
- `B` = expected hub→node conversion (0 for hub-settled)  
- `C` = expected DA–RT spread at settlement point (0 for RT products)  
- `D` = uplift from excluding negative prices (zeroing)  
- `E` = residual risk add-on so that `A+B+C(+D) + E = P*` (P-level target)
