"""
ISDA SIMM Stress Test — FRE-7801 NYU  (FIXED VERSION)
Downloads real historical data:
  IR  — FRED via pandas_datareader (fallback: manual CSV instructions)
  FX  — Yahoo Finance
  EQ  — Yahoo Finance (sector ETF proxies)

KEY FIX: sensitivity is $/1% move, pct_change() returns decimals.
  CORRECT:  pnl = sensitivity * pct_change * 100
  WRONG:    pnl = sensitivity * pct_change / 100   ← old bug, 10000x too small

Run:
  pip install pandas pandas_datareader yfinance matplotlib numpy requests
  python stress_test.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAVE_YF = True
except ImportError:
    HAVE_YF = False
    print("Install yfinance: pip install yfinance")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

START_DATE  = '2005-01-01'
END_DATE    = '2025-01-01'
WINDOW      = 10   # SIMM 10-day closeout window

# SIMM Delta Margin outputs from the project
SIMM = {'IR': 12_280_000, 'FX': 10_950_000, 'Equity': 51_880_000}

# Portfolio sensitivities
# IR:  PV01 — dollars gained/lost per 1 basis point move
# FX:  dollars gained/lost per 1 PERCENT move
# EQ:  dollars gained/lost per 1 PERCENT move
IR_SENS = {'DGS2': 125_000, 'DGS5': -90_000, 'DGS10': 160_000}
FX_SENS = {'EURUSD=X': 1_200_000, 'GBPUSD=X': -850_000, 'JPY=X': 950_000}
EQ_SENS = {'XLU': 1_500_000, 'XLF': -1_100_000, 'SPY': 2_000_000}

# ─── IR DATA — FRED ────────────────────────────────────────────────────────────

def get_ir_data_datareader():
    """Try pandas_datareader FRED connection."""
    try:
        import pandas_datareader.data as web
        print("  Trying pandas_datareader FRED...")
        frames = {}
        for s in IR_SENS.keys():
            print(f"    Fetching {s}...")
            df = web.DataReader(s, 'fred', START_DATE, END_DATE)
            frames[s] = df[s]
        rates = pd.DataFrame(frames).ffill().dropna()
        rates_bp = rates * 100   # FRED stores as % (e.g. 4.25 = 4.25%), convert to bp
        print(f"  Got {len(rates_bp)} days via pandas_datareader")
        return rates_bp
    except Exception as e:
        print(f"  pandas_datareader failed: {e}")
        return None

def get_ir_data_requests():
    """Direct FRED CSV download."""
    try:
        import requests
        frames = {}
        for s in IR_SENS.keys():
            print(f"    Fetching {s} from FRED...")
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
            r = requests.get(url, timeout=45,
                             headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
            r.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
            df.columns = [s]
            df = df.replace('.', np.nan).astype(float)
            df = df.loc[START_DATE:END_DATE]
            frames[s] = df[s]
        rates = pd.DataFrame(frames).ffill().dropna()
        rates_bp = rates * 100
        print(f"  Got {len(rates_bp)} days via direct request")
        return rates_bp
    except Exception as e:
        print(f"  Direct FRED request failed: {e}")
        return None

def get_ir_data_yfinance():
    """
    Use Treasury ETF proxies from Yahoo Finance as IR data fallback.
    SHY  ≈ 2Y rates  (iShares 1-3 Year Treasury)
    IEF  ≈ 5Y rates  (iShares 7-10 Year Treasury, best proxy)
    TLT  ≈ 10Y rates (iShares 20+ Year Treasury)
    
    These move INVERSELY to rates — price falls when rates rise.
    We compute rate moves from price moves using approximate duration.
    Duration: SHY≈1.8y, IEF≈7.5y, TLT≈17y → bp = -price_pct_change / duration / 0.01
    """
    if not HAVE_YF:
        return None
    print("  Using Treasury ETF proxies from Yahoo Finance (IR fallback)...")
    try:
        etfs = {'SHY': ('DGS2', 1.8), 'IEF': ('DGS5', 7.5), 'TLT': ('DGS10', 17.0)}
        prices = yf.download(list(etfs.keys()), start=START_DATE, end=END_DATE,
                             auto_adjust=True, progress=False)['Close']
        prices = prices.ffill().dropna()
        rates_bp = pd.DataFrame()
        for etf, (series, dur) in etfs.items():
            if etf in prices.columns:
                # price moves inversely to rates
                # price_pct_change ≈ -duration * rate_change (in decimal)
                # rate_change_bp = -price_pct_change / duration * 10000
                price_pct = prices[etf].pct_change()
                rate_bp_daily = -price_pct / dur * 10000
                rates_bp[series] = rate_bp_daily
        rates_bp = rates_bp.ffill().dropna()
        # Now rates_bp contains daily bp changes — we need LEVELS to take diff
        # So we accumulate to get levels then take 10-day diff
        rates_level = rates_bp.cumsum()  # approximate level in bp
        print(f"  Got {len(rates_level)} days of implied rate levels")
        return rates_level
    except Exception as e:
        print(f"  Treasury ETF fallback failed: {e}")
        return None

def get_ir_data():
    print("\n[1/3] Downloading interest rate data...")
    # Try in order: pandas_datareader → direct request → yfinance ETF proxy
    result = get_ir_data_datareader()
    if result is not None:
        return result
    result = get_ir_data_requests()
    if result is not None:
        return result
    print("  Both FRED methods failed. Using Treasury ETF proxy from Yahoo Finance.")
    return get_ir_data_yfinance()

# ─── FX + EQUITY FROM YAHOO FINANCE ───────────────────────────────────────────

def get_fx_equity_data():
    if not HAVE_YF:
        return None, None
    print("\n[2/3] Downloading FX and equity data from Yahoo Finance...")
    all_tickers = list(FX_SENS.keys()) + list(EQ_SENS.keys())
    try:
        raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)['Close']
        raw = raw.ffill().dropna()
        fx  = raw[[t for t in FX_SENS if t in raw.columns]]
        eq  = raw[[t for t in EQ_SENS if t in raw.columns]]
        print(f"  {len(raw)} days  ({raw.index[0].date()} to {raw.index[-1].date()})")
        return fx, eq
    except Exception as e:
        print(f"  Yahoo Finance failed: {e}")
        return None, None

# ─── P&L COMPUTATION ──────────────────────────────────────────────────────────

def ir_pnl(rates_bp):
    """
    IR PnL = PV01 × 10-day rate move in basis points
    rates_bp contains levels in bp → diff(10) gives 10-day move in bp
    """
    moves_bp = rates_bp.diff(WINDOW)    # bp move over 10 days
    pnl = pd.Series(0.0, index=moves_bp.index)
    for col, sens in IR_SENS.items():
        if col in moves_bp.columns:
            # PV01: +sens means gain $sens per +1bp move
            # e.g. long 2Y: if 2Y rates rise 1bp, gain $125,000
            pnl += moves_bp[col] * sens
    return pnl.dropna()

def fx_pnl(fx_prices):
    """
    FX PnL = sensitivity × 10-day % move
    Sensitivity is $/1% move.
    pct_change(10) returns decimal (0.01 = 1%).
    CORRECT formula: pnl = sens × pct_change × 100
    Because: sens × (decimal_move × 100) converts decimal to percentage.
    """
    pct = fx_prices.pct_change(WINDOW)   # decimal e.g. 0.03 = 3%
    pnl = pd.Series(0.0, index=pct.index)
    for col, sens in FX_SENS.items():
        if col in pct.columns:
            pnl += pct[col] * 100 * sens    # ← KEY FIX: × 100 converts decimal to %
    return pnl.dropna()

def eq_pnl(eq_prices):
    """
    Equity PnL = sensitivity × 10-day % move
    Same unit fix as FX.
    """
    pct = eq_prices.pct_change(WINDOW)
    pnl = pd.Series(0.0, index=pct.index)
    for col, sens in EQ_SENS.items():
        if col in pct.columns:
            pnl += pct[col] * 100 * sens    # ← KEY FIX
    return pnl.dropna()

# ─── ANALYSIS ─────────────────────────────────────────────────────────────────

def analyze(pnl, margin, label):
    losses = (-pnl[pnl < 0]).dropna()
    if len(losses) == 0:
        print(f"\n{label}: No loss windows found.")
        return {}

    p95        = np.percentile(losses, 95)
    p99        = np.percentile(losses, 99)
    worst      = losses.max()
    worst_date = losses.idxmax()
    exceptions = int((losses > margin).sum())
    exc_rate   = exceptions / len(losses) * 100
    cov_99     = margin / p99
    cov_worst  = worst / margin * 100   # % of margin consumed by worst loss

    # Find top 5 worst loss dates
    top5 = losses.nlargest(5)

    print(f"\n{'─'*52}")
    print(f"  {label} Stress Test Results")
    print(f"{'─'*52}")
    print(f"  10-day windows analyzed    : {len(pnl):,}")
    print(f"  Loss windows               : {len(losses):,}")
    print(f"  SIMM Delta Margin          : ${margin/1e6:.2f}M")
    print(f"  Mean 10-day loss           : ${losses.mean()/1e6:.2f}M")
    print(f"  95th percentile loss       : ${p95/1e6:.2f}M")
    print(f"  99th percentile loss       : ${p99/1e6:.2f}M")
    print(f"  Worst ever loss            : ${worst/1e6:.2f}M  ({worst_date.date()})")
    print(f"  Coverage ratio (99th pct)  : {cov_99:.2f}×")
    print(f"  Worst case (% of margin)   : {cov_worst:.1f}%")
    print(f"  Exceptions (loss > margin) : {exceptions} days  ({exc_rate:.2f}%)")
    print(f"\n  Top 5 worst loss dates:")
    for dt, val in top5.items():
        pct_margin = val/margin*100
        print(f"    {dt.date()}  ${val/1e6:.2f}M  ({pct_margin:.1f}% of margin)")
    if cov_99 >= 1.0:
        print(f"\n  ✓ ADEQUATE at 99th percentile ({cov_99:.2f}× coverage)")
    else:
        print(f"\n  ✗ BREACH at 99th percentile ({cov_99:.2f}×)")
    if cov_worst > 100:
        print(f"  ✗ TAIL BREACH — worst loss consumed {cov_worst:.1f}% of margin")
    else:
        print(f"  ✓ No tail breach — worst loss {cov_worst:.1f}% of margin")

    return dict(label=label, losses=losses, margin=margin,
                p95=p95, p99=p99, worst=worst, worst_date=worst_date,
                cov_99=cov_99, cov_worst=cov_worst,
                exceptions=exceptions, exc_rate=exc_rate)

# ─── CHARTS ───────────────────────────────────────────────────────────────────

def plot_results(results):
    valid = [r for r in results if r and 'losses' in r]
    if not valid:
        print("Nothing to plot.")
        return

    n = len(valid)
    fig = plt.figure(figsize=(7*n, 13))
    fig.patch.set_facecolor('#f0f0ff')
    gs = gridspec.GridSpec(3, n, figure=fig, hspace=0.42, wspace=0.28)

    palette = {'IR': '#6366f1', 'FX': '#10b981', 'Equity': '#f59e0b'}
    RED     = '#ef4444'
    ORANGE  = '#f97316'
    BLACK   = '#1a1a2e'

    for ci, r in enumerate(valid):
        color  = palette.get(r['label'], '#6366f1')
        losses = r['losses'] / 1e6
        margin = r['margin'] / 1e6
        p99    = r['p99']    / 1e6
        p95    = r['p95']    / 1e6
        worst  = r['worst']  / 1e6

        # ── Row 0: Loss distribution histogram ─────────────────────
        ax = fig.add_subplot(gs[0, ci])
        ax.set_facecolor('white')
        counts, bins, patches = ax.hist(losses, bins=60, color=color,
                                        alpha=0.72, edgecolor='white', lw=0.3)
        ax.axvline(margin, color=RED,    lw=2.0,       label=f'SIMM margin ${margin:.1f}M')
        ax.axvline(p99,    color=ORANGE, lw=1.5, ls='--', label=f'99th pct ${p99:.1f}M')
        ax.axvline(worst,  color=BLACK,  lw=1.2, ls=':',  label=f'Worst ${worst:.1f}M')
        for patch, left in zip(patches, bins[:-1]):
            if left >= margin:
                patch.set_facecolor(RED); patch.set_alpha(0.85)
        ax.set_title(f'{r["label"]} — Loss Distribution', fontsize=12,
                     fontweight='500', pad=8, color=BLACK)
        ax.set_xlabel('10-day loss ($M)', fontsize=9, color=BLACK)
        ax.set_ylabel('Frequency',        fontsize=9, color=BLACK)
        ax.legend(fontsize=7.5, framealpha=0.8)
        ax.spines[['top','right']].set_visible(False)

        # ── Row 1: Running worst loss vs margin ─────────────────────
        ax2 = fig.add_subplot(gs[1, ci])
        ax2.set_facecolor('white')
        running_max = r['losses'].expanding().max() / 1e6
        ax2.plot(running_max.index, running_max.values,
                 color=color, lw=1.4, label='Running worst loss')
        ax2.fill_between(running_max.index, running_max.values,
                         alpha=0.10, color=color)
        ax2.axhline(margin, color=RED, lw=1.8, label=f'SIMM margin ${margin:.1f}M')
        # mark worst date
        ax2.axvline(r['worst_date'], color=BLACK, lw=0.8, ls=':', alpha=0.5)
        ax2.annotate(f"Worst\n{r['worst_date'].strftime('%b %Y')}",
                     xy=(r['worst_date'], worst),
                     xytext=(10, -20), textcoords='offset points',
                     fontsize=7, color=BLACK,
                     arrowprops=dict(arrowstyle='->', color=BLACK, lw=0.8))
        ax2.set_title(f'{r["label"]} — Running Worst Loss vs Margin',
                      fontsize=11, fontweight='500', pad=8, color=BLACK)
        ax2.set_xlabel('Date',      fontsize=9, color=BLACK)
        ax2.set_ylabel('Loss ($M)', fontsize=9, color=BLACK)
        ax2.legend(fontsize=7.5)
        ax2.spines[['top','right']].set_visible(False)

        # ── Row 2: Coverage ratio bars ──────────────────────────────
        ax3 = fig.add_subplot(gs[2, ci])
        ax3.set_facecolor('white')
        metrics = ['95th pct\ncoverage ×', '99th pct\ncoverage ×',
                   'Worst case\n% of margin']
        values  = [r['margin']/r['p95'],
                   r['cov_99'],
                   r['cov_worst']/100]
        bar_clrs = ['#10b981' if v >= 1.0 else '#ef4444' for v in values]
        bars = ax3.bar(metrics, values, color=bar_clrs, alpha=0.82,
                       edgecolor='white', width=0.5)
        ax3.axhline(1.0, color=BLACK, lw=1.0, ls='--', alpha=0.4,
                    label='Adequacy threshold')
        for bar, val, raw in zip(bars, values,
                                 [r['margin']/r['p95'], r['cov_99'], r['cov_worst']]):
            label_str = f'{val:.1f}×' if raw < 50 else f'{raw:.0f}%'
            ax3.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+max(values)*0.02,
                     label_str, ha='center', va='bottom',
                     fontsize=9, fontweight='500', color=BLACK)
        ax3.set_title(f'{r["label"]} — Coverage Ratios',
                      fontsize=11, fontweight='500', pad=8, color=BLACK)
        ax3.set_ylabel('Ratio', fontsize=9, color=BLACK)
        ax3.legend(fontsize=7.5)
        ax3.spines[['top','right']].set_visible(False)
        ax3.set_ylim(0, max(values) * 1.3)

    fig.suptitle(
        'ISDA SIMM Delta Margin — Historical Stress Test\n'
        'FRE-7801 Independent Validation · NYU · ISDA SIMM v2.8',
        fontsize=14, fontweight='500', y=0.99, color=BLACK)

    plt.savefig('simm_stress_test.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    print("\n  ✓ Chart saved: simm_stress_test.png")
    plt.show()

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("="*55)
    print("  ISDA SIMM Stress Test — FRE-7801 NYU")
    print("  Fixed version — correct unit math")
    print("="*55)

    results = []

    # Interest Rates
    rates_bp = get_ir_data()
    if rates_bp is not None:
        pnl = ir_pnl(rates_bp)
        results.append(analyze(pnl, SIMM['IR'], 'IR'))
    else:
        print("  Skipping IR — all data sources failed")
        print("  Manual fix: download DGS2,DGS5,DGS10 CSV from fred.stlouisfed.org")
        print("  and place as fred_rates.csv in this folder")

    # FX + Equity from Yahoo Finance
    fx_prices, eq_prices = get_fx_equity_data()

    if fx_prices is not None:
        pnl = fx_pnl(fx_prices)
        results.append(analyze(pnl, SIMM['FX'], 'FX'))
    else:
        print("  Skipping FX")

    if eq_prices is not None:
        pnl = eq_pnl(eq_prices)
        results.append(analyze(pnl, SIMM['Equity'], 'Equity'))
    else:
        print("  Skipping Equity")

    print("\n[3/3] Generating charts...")
    plot_results(results)

    print("\n" + "="*55)
    print("  FINAL SUMMARY")
    print("="*55)
    for r in results:
        if r and 'label' in r:
            s1 = "✓ PASS" if r['cov_99'] >= 1.0 else "✗ FAIL"
            s2 = "✗ TAIL BREACH" if r['cov_worst'] > 100 else "✓ OK"
            print(f"  {r['label']:<8}  "
                  f"99th pct: {r['cov_99']:.2f}×  {s1}   "
                  f"Worst: {r['cov_worst']:.1f}% of margin  {s2}  "
                  f"Exceptions: {r['exceptions']} days")

    print()
    print("  THE FIX from the broken version:")
    print("  pct_change() returns decimals (0.01 = 1%)")
    print("  Sensitivity is $/1% move")
    print("  CORRECT:  pnl = sensitivity × pct_change × 100")
    print("  WRONG:    pnl = sensitivity × pct_change / 100  ← 10,000× too small")
    print()
    print("  DATA SOURCES")
    print("  IR  : FRED — DGS2, DGS5, DGS10 (US Treasury Constant Maturity)")
    print("  FX  : Yahoo Finance — EURUSD=X, GBPUSD=X, JPY=X")
    print("  EQ  : Yahoo Finance — XLU (B5), XLF (B8), SPY (B11)")
    print(f"  Period: {START_DATE} to {END_DATE}  |  10-day windows")

if __name__ == '__main__':
    main()