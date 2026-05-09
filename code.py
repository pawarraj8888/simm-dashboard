import numpy as np
import pandas as pd

# ============================================================
# STEP 1: PORTFOLIO SENSITIVITIES
# These come from your CRIF file / representative portfolio
# ============================================================

# Interest Rate sensitivities (PV01 — dollars per 1bp move)
# USD is a "regular volatility" currency
ir_sensitivities = {
    '2Y':  125000,   # long 2Y — gain if 2Y rates rise
    '5Y': -90000,    # short 5Y — lose if 5Y rates rise
    '10Y': 160000    # long 10Y — gain if 10Y rates rise
}

# FX sensitivities (dollars per 1% move in currency pair)
# EUR, GBP, JPY are all "regular" FX volatility currencies
# Calculation currency is USD (also regular)
fx_sensitivities = {
    'EURUSD':  1200000,
    'GBPUSD': -850000,
    'USDJPY':  950000
}

# Equity sensitivities (dollars per 1% move in equity)
equity_sensitivities = {
    'Bucket5':   1500000,   # Dev large cap consumer/utilities
    'Bucket8':  -1100000,   # Dev large cap financials/tech
    'Bucket11':  2000000    # Index/ETF
}

print("Portfolio loaded successfully.")


# ============================================================
# STEP 2: ISDA RISK WEIGHTS
# Taken directly from ISDA SIMM v2.8 document
# ============================================================

# From Section D, Table 1 — USD is regular volatility currency
# Tenors: 2w, 1m, 3m, 6m, 1y, 2y, 3y, 5y, 10y, 15y, 20y, 30y
# We only need 2y=69, 5y=61, 10y=60
ir_risk_weights = {
    '2Y':  69,   # 69 basis points — the stressed 10-day move at 2Y tenor
    '5Y':  61,   # 61 basis points
    '10Y': 60    # 60 basis points
}

# From Section I — regular currency vs regular calculation currency
# Risk weight = 7.1% for regular/regular pair
fx_risk_weights = {
    'EURUSD': 0.071,
    'GBPUSD': 0.071,
    'USDJPY': 0.071   # JPY is low vol but USDJPY pair uses 7.1 vs USD
}

# From Section G, Table — bucket-specific risk weights
equity_risk_weights = {
    'Bucket5':  0.23,   # 23% for Dev large cap consumer/utilities
    'Bucket8':  0.29,   # 29% for Dev large cap financials/tech
    'Bucket11': 0.17    # 17% for Indexes/ETFs
}

print("\nISDA v2.8 Risk weights loaded.")
print(f"IR weights — 2Y: {ir_risk_weights['2Y']}bp, "
      f"5Y: {ir_risk_weights['5Y']}bp, "
      f"10Y: {ir_risk_weights['10Y']}bp")
print(f"FX weight (regular/regular): {fx_risk_weights['EURUSD']*100}%")
print(f"Equity — B5: {equity_risk_weights['Bucket5']*100}%, "
      f"B8: {equity_risk_weights['Bucket8']*100}%, "
      f"B11: {equity_risk_weights['Bucket11']*100}%")


# ============================================================
# STEP 3: ISDA CORRELATION MATRICES
# Taken directly from ISDA SIMM v2.8 document
# ============================================================

# From Section D.2 — IR tenor correlations (regular currency)
# We need 2Y vs 5Y, 2Y vs 10Y, 5Y vs 10Y
# From the table: 2y-5y=92%, 2y-10y=86%, 5y-10y=96%
ir_correlations = np.array([
    #   2Y    5Y    10Y
    [1.00, 0.92, 0.86],   # 2Y vs 2Y, 5Y, 10Y
    [0.92, 1.00, 0.96],   # 5Y vs 2Y, 5Y, 10Y
    [0.86, 0.96, 1.00]    # 10Y vs 2Y, 5Y, 10Y
])

# From Section I.2 — FX correlations
# Regular calculation currency (USD), all pairs are regular currencies
# Regular/Regular = 50%
fx_correlations = np.array([
    #   EUR   GBP   JPY
    [1.00, 0.50, 0.50],   # EUR vs EUR, GBP, JPY
    [0.50, 1.00, 0.50],   # GBP vs EUR, GBP, JPY
    [0.50, 0.50, 1.00]    # JPY vs EUR, GBP, JPY
])

# From Section G.2 — Equity cross-bucket correlations
# Bucket 5 vs 8 = 31%, Bucket 5 vs 11 = 29%, Bucket 8 vs 11 = 37%
equity_correlations = np.array([
    #   B5    B8    B11
    [1.00, 0.31, 0.29],   # B5 vs B5, B8, B11
    [0.31, 1.00, 0.37],   # B8 vs B5, B8, B11
    [0.29, 0.37, 1.00]    # B11 vs B5, B8, B11
])

print("\nISDA v2.8 Correlation matrices loaded.")
print("\nIR Correlation Matrix (2Y, 5Y, 10Y):")
print(pd.DataFrame(ir_correlations,
                   index=['2Y','5Y','10Y'],
                   columns=['2Y','5Y','10Y']))

print("\nFX Correlation Matrix (EUR, GBP, JPY):")
print(pd.DataFrame(fx_correlations,
                   index=['EUR','GBP','JPY'],
                   columns=['EUR','GBP','JPY']))

print("\nEquity Correlation Matrix (B5, B8, B11):")
print(pd.DataFrame(equity_correlations,
                   index=['B5','B8','B11'],
                   columns=['B5','B8','B11']))


# ============================================================
# STEP 4: CONCENTRATION RISK FACTOR
# From Section J of ISDA SIMM v2.8
# CR = max(1, sqrt(|sensitivity| / threshold))
# If below threshold, CR = 1 (no concentration add-on)
# ============================================================

def compute_CR(sensitivity, threshold):
    """
    Concentration Risk Factor from ISDA Section J.
    If your position is below the threshold — CR = 1, no add-on.
    If above — CR > 1, meaning you get penalized for concentration.
    
    For IR: threshold is in USD mm/bp
    For FX: threshold is in USD mm/%
    For Equity: threshold is in USD mm/%
    """
    return max(1.0, np.sqrt(abs(sensitivity) / threshold))

# IR concentration threshold for USD (regular vol, well-traded) = 210 mm/bp
# Convert to dollars: 210 * 1,000,000 = 210,000,000
IR_THRESHOLD = 210 * 1_000_000  # USD mm/bp

# Sum of absolute IR sensitivities for CR calculation
ir_total_abs = sum(abs(v) for v in ir_sensitivities.values())
ir_CR = compute_CR(ir_total_abs, IR_THRESHOLD)

# FX threshold for EUR, GBP (Category 1 currencies) = 3,100 mm/%
# JPY is also Category 1
FX_THRESHOLD = 3100 * 1_000_000

fx_CRs = {}
for pair, sens in fx_sensitivities.items():
    fx_CRs[pair] = compute_CR(sens, FX_THRESHOLD)

# Equity thresholds from Section J.3
# Bucket 5-8 (Dev large cap) = 14 mm/%
# Bucket 11-12 (Indexes/ETFs) = 730 mm/%
eq_thresholds = {
    'Bucket5':  14  * 1_000_000,
    'Bucket8':  14  * 1_000_000,
    'Bucket11': 730 * 1_000_000
}
eq_CRs = {}
for bucket, sens in equity_sensitivities.items():
    eq_CRs[bucket] = compute_CR(sens, eq_thresholds[bucket])

print("\n--- Concentration Risk Factors ---")
print(f"IR CR (total position vs 210mm threshold): {ir_CR:.4f}")
for pair, cr in fx_CRs.items():
    print(f"FX CR ({pair}): {cr:.4f}")
for bucket, cr in eq_CRs.items():
    print(f"Equity CR ({bucket}): {cr:.4f}")


# ============================================================
# STEP 5: WEIGHTED SENSITIVITIES
# WS = sensitivity × risk_weight × CR
# This is the stressed dollar loss for each position
# ============================================================

def compute_weighted_sensitivities(sensitivities, risk_weights, CRs):
    weighted = {}
    print(f"\n{'Factor':<12} {'Sensitivity':>15} {'RW':>8} {'CR':>6} {'WS':>15}")
    print("-" * 60)
    for factor, sens in sensitivities.items():
        rw  = risk_weights[factor]
        cr  = CRs[factor] if isinstance(CRs, dict) else CRs
        ws  = sens * rw * cr
        weighted[factor] = ws
        print(f"{factor:<12} {sens:>15,.0f} {rw:>8.4f} {cr:>6.4f} {ws:>15,.0f}")
    return weighted

print("\n=== IR Weighted Sensitivities ===")
ir_CRs_dict = {k: ir_CR for k in ir_sensitivities}  # same CR for all IR tenors
ir_ws = compute_weighted_sensitivities(
    ir_sensitivities, ir_risk_weights, ir_CRs_dict)

print("\n=== FX Weighted Sensitivities ===")
fx_ws = compute_weighted_sensitivities(
    fx_sensitivities, fx_risk_weights, fx_CRs)

print("\n=== Equity Weighted Sensitivities ===")
eq_ws = compute_weighted_sensitivities(
    equity_sensitivities, equity_risk_weights, eq_CRs)


# ============================================================
# STEP 6: SIMM AGGREGATION FORMULA
# IM = sqrt( sum_i sum_j rho_ij * WS_i * WS_j )
# This is the core formula from Section B of the document
# ============================================================

def compute_delta_margin(weighted_sensitivities, correlation_matrix, label):
    """
    SIMM Delta Margin aggregation formula.
    
    For every pair of positions (i, j):
      - Multiply their weighted sensitivities together
      - Multiply by their correlation
      - Sum everything up
      - Take the square root
    
    When i == j, correlation = 1.0 (a position with itself)
    When i != j, correlation comes from ISDA's matrix
    
    The result: positions that offset each other reduce the total.
    Positions that are independent stack up more fully.
    """
    ws_values = np.array(list(weighted_sensitivities.values()))
    factors   = list(weighted_sensitivities.keys())
    n = len(ws_values)

    print(f"\n--- {label} Aggregation Detail ---")
    total = 0.0
    for i in range(n):
        for j in range(n):
            rho = correlation_matrix[i, j]
            contribution = rho * ws_values[i] * ws_values[j]
            total += contribution
            print(f"  {factors[i]} × {factors[j]}: "
                  f"ρ={rho:.2f} × {ws_values[i]:,.0f} × {ws_values[j]:,.0f} "
                  f"= {contribution:,.0f}")

    margin = np.sqrt(abs(total))
    print(f"\n  Sum before sqrt: {total:,.0f}")
    print(f"  Delta Margin = sqrt({total:,.0f}) = ${margin:,.0f}")
    return margin

ir_margin  = compute_delta_margin(ir_ws,  ir_correlations,     "IR")
fx_margin  = compute_delta_margin(fx_ws,  fx_correlations,     "FX")
eq_margin  = compute_delta_margin(eq_ws,  equity_correlations, "Equity")


# ============================================================
# STEP 7: CROSS ASSET CLASS AGGREGATION
# From Section K of ISDA v2.8
# IR vs FX correlation = 10%
# IR vs Equity = 12%
# FX vs Equity = 24%
# ============================================================

# Cross risk class correlations from Section K
# IR=0, FX=1, Equity=2
cross_class_corr = np.array([
    #   IR    FX    EQ
    [1.00, 0.10, 0.12],   # IR vs IR, FX, Equity
    [0.10, 1.00, 0.24],   # FX vs IR, FX, Equity
    [0.12, 0.24, 1.00]    # Equity vs IR, FX, Equity
])

class_margins = np.array([ir_margin, fx_margin, eq_margin])
class_names   = ['IR', 'FX', 'Equity']

print("\n--- Cross Asset Class Aggregation ---")
total_sq = 0.0
for i in range(3):
    for j in range(3):
        psi = cross_class_corr[i, j]
        contribution = psi * class_margins[i] * class_margins[j]
        total_sq += contribution
        print(f"  {class_names[i]} × {class_names[j]}: "
              f"ψ={psi:.2f} × {class_margins[i]:,.0f} × {class_margins[j]:,.0f} "
              f"= {contribution:,.0f}")

total_margin = np.sqrt(abs(total_sq))


# ============================================================
# STEP 8: FINAL RESULTS
# ============================================================

print("\n" + "="*55)
print("FINAL SIMM DELTA MARGIN RESULTS (ISDA v2.8)")
print("="*55)
print(f"  Interest Rate Delta Margin:  ${ir_margin:>12,.0f}")
print(f"  FX Delta Margin:             ${fx_margin:>12,.0f}")
print(f"  Equity Delta Margin:         ${eq_margin:>12,.0f}")
print(f"  {'─'*40}")
print(f"  Total Delta Margin:          ${total_margin:>12,.0f}")
print("="*55)


# ============================================================
# STEP 9: SANITY CHECK
# Compare against naive sum (no correlation offsets at all)
# Shows you how much the correlation structure matters
# ============================================================

naive_ir  = sum(abs(v) for v in ir_ws.values())
naive_fx  = sum(abs(v) for v in fx_ws.values())
naive_eq  = sum(abs(v) for v in eq_ws.values())
naive_total = naive_ir + naive_fx + naive_eq

reduction    = naive_total - total_margin
reduction_pct = (reduction / naive_total) * 100

print("\n--- Sanity Check: SIMM vs Naive Sum ---")
print(f"Naive sum (ignore all offsets):  ${naive_total:>12,.0f}")
print(f"SIMM (with ISDA correlations):   ${total_margin:>12,.0f}")
print(f"Reduction from diversification:  ${reduction:>12,.0f}")
print(f"That is a {reduction_pct:.1f}% reduction")
print()
print("This is why SIMM uses correlations.")
print("A firm with a balanced book should not post the same")
print("margin as a firm with a purely directional book.")
print("The correlation structure recognizes the difference.")