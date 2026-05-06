"""
Ratio Computation Tools
=======================
All tools are self-contained: they call internal EDGAR helpers and return
computed ratios with benchmarks and trend signals.

Mapped to subagents:
  Financial Health  → compute_liquidity_ratios, compute_leverage_ratios,
                       compute_altman_zscore, compute_piotroski_fscore
  Profitability     → compute_profitability_ratios, compute_growth_metrics,
                       compute_dupont_analysis
"""

import yfinance as yf
from langchain_core.tools import tool
from tools.fundamental.edgar_tools import (
    _fetch_income_stmt,
    _fetch_balance_sheet,
    _fetch_cash_flow,
)


# ── Utility helpers ───────────────────────────────────────────────────────────

def _div(a, b) -> float | None:
    """Safe division — returns None if denominator is zero or either value is None."""
    if a is None or b is None or b == 0:
        return None
    return round(a / b, 4)


def _pct(val) -> float | None:
    """Convert decimal to percentage, rounded to 2dp."""
    return round(val * 100, 2) if val is not None else None


def _trend(values: dict) -> str:
    """Return 'improving', 'deteriorating', or 'stable' based on sorted year values."""
    sorted_vals = [v for _, v in sorted(values.items()) if v is not None]
    if len(sorted_vals) < 2:
        return "insufficient data"
    delta = sorted_vals[-1] - sorted_vals[0]
    pct_change = delta / abs(sorted_vals[0]) if sorted_vals[0] != 0 else 0
    if pct_change > 0.05:
        return "improving"
    elif pct_change < -0.05:
        return "deteriorating"
    return "stable"


# ── Financial Health Tools ────────────────────────────────────────────────────

@tool
def compute_liquidity_ratios(cik: str, num_years: int = 3) -> dict:
    """
    Compute liquidity ratios for the company over num_years fiscal years.

    Metrics:
      - Current Ratio  = Current Assets / Current Liabilities
      - Quick Ratio    = (Cash + Receivables) / Current Liabilities
      - Cash Ratio     = Cash / Current Liabilities
      - OCF Ratio      = Operating Cash Flow / Current Liabilities

    Benchmarks:
      Current Ratio: >2.0 strong, 1.0-2.0 adequate, <1.0 concern
      Quick Ratio:   >1.0 healthy, <0.5 potential risk
      OCF Ratio:     >0.4 indicates strong operating cash coverage

    Args:
        cik:       SEC Central Index Key
        num_years: Fiscal years to analyse (default 3)
    """
    bs = _fetch_balance_sheet(cik, num_years)
    cf = _fetch_cash_flow(cik, num_years)

    years = sorted(bs["years_covered"], reverse=True)
    values = {}

    for yr in years:
        ca  = bs["current_assets"].get(yr)
        cl  = bs["current_liabilities"].get(yr)
        csh = bs["cash"].get(yr, 0) or 0
        ar  = bs["accounts_receivable"].get(yr, 0) or 0
        ocf = cf["operating_cf"].get(yr)

        values[yr] = {
            "current_ratio": _div(ca, cl),
            "quick_ratio":   _div(csh + ar, cl),
            "cash_ratio":    _div(csh, cl),
            "ocf_ratio":     _div(ocf, cl),
        }

    current_ratios = {yr: v["current_ratio"] for yr, v in values.items() if v["current_ratio"]}

    return {
        "metric":      "liquidity_ratios",
        "years":       years,
        "values":      values,
        "trend":       _trend(current_ratios),
        "benchmarks": {
            "current_ratio": ">2.0 strong | 1.0–2.0 adequate | <1.0 concern",
            "quick_ratio":   ">1.0 healthy | <0.5 potential liquidity risk",
            "ocf_ratio":     ">0.4 strong cash coverage of current liabilities",
        },
    }


@tool
def compute_leverage_ratios(cik: str, num_years: int = 3) -> dict:
    """
    Compute leverage and solvency ratios over num_years fiscal years.

    Metrics:
      - Debt / Equity            = Total Debt / Total Equity
      - Net Debt / EBITDA        = (Total Debt - Cash) / EBITDA
      - Interest Coverage (EBIT) = Operating Income / Interest Expense
      - Debt / Total Assets      = Total Debt / Total Assets
      - Net Debt                 = Total Debt - Cash

    Benchmarks:
      Net Debt/EBITDA: <2x conservative | 2–4x moderate | >5x highly leveraged
      Interest Coverage: >3x healthy | 1.5–3x watch | <1.5x distress signal

    Args:
        cik:       SEC Central Index Key
        num_years: Fiscal years to analyse (default 3)
    """
    bs  = _fetch_balance_sheet(cik, num_years)
    is_ = _fetch_income_stmt(cik, num_years)

    years = sorted(bs["years_covered"], reverse=True)
    values = {}

    for yr in years:
        total_debt = bs["total_debt"].get(yr, 0) or 0
        equity     = bs["total_equity"].get(yr)
        ta         = bs["total_assets"].get(yr)
        cash       = bs["cash"].get(yr, 0) or 0
        ebitda     = is_["ebitda"].get(yr)
        op_income  = is_["operating_income"].get(yr)
        int_exp    = is_["interest_expense"].get(yr)
        net_debt   = total_debt - cash

        values[yr] = {
            "debt_to_equity":     _div(total_debt, equity),
            "net_debt_to_ebitda": _div(net_debt, ebitda),
            "interest_coverage":  _div(op_income, int_exp),
            "debt_to_assets":     _div(total_debt, ta),
            "net_debt":           net_debt,
        }

    leverage_trend = {yr: v["net_debt_to_ebitda"] for yr, v in values.items() if v["net_debt_to_ebitda"]}

    return {
        "metric":  "leverage_ratios",
        "years":   years,
        "values":  values,
        "trend":   _trend({yr: -v for yr, v in leverage_trend.items()}),  # inverse: lower = improving
        "benchmarks": {
            "net_debt_to_ebitda": "<2x conservative | 2–4x moderate | >5x highly leveraged",
            "interest_coverage":  ">3x healthy | <1.5x distress signal",
        },
    }


@tool
def compute_altman_zscore(ticker: str, cik: str) -> dict:
    """
    Compute the Altman Z-Score — a quantitative bankruptcy prediction model.

    Formula: Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5
      X1 = Working Capital / Total Assets
      X2 = Retained Earnings / Total Assets
      X3 = EBIT / Total Assets
      X4 = Market Cap / Total Liabilities
      X5 = Revenue / Total Assets

    Zones:
      Z > 2.99  → Safe Zone     (low bankruptcy risk)
      1.81–2.99 → Grey Zone    (monitor closely)
      Z < 1.81  → Distress Zone (high bankruptcy risk)

    Args:
        ticker: Stock ticker (needed for current market cap via yfinance)
        cik:    SEC Central Index Key
    """
    bs  = _fetch_balance_sheet(cik, 1)
    is_ = _fetch_income_stmt(cik, 1)

    yr = bs["years_covered"][0] if bs["years_covered"] else None
    if not yr:
        return {"error": "No balance sheet data found."}

    ta  = bs["total_assets"].get(yr) or 1
    ca  = bs["current_assets"].get(yr, 0) or 0
    cl  = bs["current_liabilities"].get(yr, 0) or 0
    re  = bs["retained_earnings"].get(yr, 0) or 0
    tl  = bs["total_liabilities"].get(yr, 0) or 0
    ebit = is_["operating_income"].get(yr, 0) or 0
    rev  = is_["revenue"].get(yr, 0) or 0

    try:
        market_cap = yf.Ticker(ticker).info.get("marketCap") or 0
    except Exception:
        market_cap = 0

    wc = ca - cl

    x1 = wc / ta
    x2 = re / ta
    x3 = ebit / ta
    x4 = market_cap / tl if tl else 0
    x5 = rev / ta

    z = round(1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5, 3)

    if z > 2.99:
        zone = "Safe Zone"
        interpretation = "Low bankruptcy risk. Financial structure is sound."
    elif z > 1.81:
        zone = "Grey Zone"
        interpretation = "Moderate risk. Warrants monitoring."
    else:
        zone = "Distress Zone"
        interpretation = "Elevated bankruptcy risk. Significant financial weakness detected."

    return {
        "metric":          "altman_z_score",
        "fiscal_year":     yr,
        "z_score":         z,
        "zone":            zone,
        "interpretation":  interpretation,
        "components": {
            "X1_working_capital_to_assets":      round(x1, 4),
            "X2_retained_earnings_to_assets":    round(x2, 4),
            "X3_ebit_to_assets":                 round(x3, 4),
            "X4_market_cap_to_total_liabilities": round(x4, 4),
            "X5_revenue_to_assets":              round(x5, 4),
        },
    }


@tool
def compute_piotroski_fscore(cik: str) -> dict:
    """
    Compute the Piotroski F-Score (0–9) measuring year-over-year financial quality improvement.

    9 binary criteria (1 = pass, 0 = fail):
      Profitability (4): ROA positive, ROA improving, Positive OCF, OCF > ROA (accruals)
      Leverage (3):      Leverage decreasing, Liquidity improving, No share dilution
      Efficiency (2):    Gross margin improving, Asset turnover improving

    Score: 8–9 Strong | 5–7 Neutral | 0–2 Weak

    Args:
        cik: SEC Central Index Key (requires 2 years of data)
    """
    bs  = _fetch_balance_sheet(cik, 2)
    is_ = _fetch_income_stmt(cik, 2)
    cf  = _fetch_cash_flow(cik, 2)

    yrs = sorted(bs["years_covered"], reverse=True)
    if len(yrs) < 2:
        return {"error": "Need at least 2 fiscal years of data for Piotroski F-Score."}

    yr1, yr0 = yrs[0], yrs[1]  # yr1=current, yr0=prior

    def g(data, yr):
        return data.get(yr, 0) or 0

    ta1 = g(bs["total_assets"], yr1) or 1
    ta0 = g(bs["total_assets"], yr0) or 1

    # ── Profitability ─────────────────────────────────────────────────────────
    ni1  = g(is_["net_income"], yr1)
    ni0  = g(is_["net_income"], yr0)
    ocf1 = g(cf["operating_cf"], yr1)
    roa1 = ni1 / ta1
    roa0 = ni0 / ta0

    f1 = 1 if roa1 > 0 else 0            # ROA positive
    f2 = 1 if roa1 > roa0 else 0         # ROA improving
    f3 = 1 if ocf1 > 0 else 0            # Positive OCF
    f4 = 1 if ocf1 / ta1 > roa1 else 0  # Quality: OCF/Assets > ROA (low accruals)

    # ── Leverage & Liquidity ──────────────────────────────────────────────────
    lev1 = g(bs["total_debt"], yr1) / ta1
    lev0 = g(bs["total_debt"], yr0) / ta0
    cl1  = g(bs["current_liabilities"], yr1) or 1
    cl0  = g(bs["current_liabilities"], yr0) or 1
    liq1 = g(bs["current_assets"], yr1) / cl1
    liq0 = g(bs["current_assets"], yr0) / cl0
    sh1  = g(is_["shares_outstanding"], yr1)
    sh0  = g(is_["shares_outstanding"], yr0)

    f5 = 1 if lev1 < lev0 else 0                          # Leverage falling
    f6 = 1 if liq1 > liq0 else 0                          # Liquidity improving
    f7 = 1 if (sh0 == 0 or sh1 <= sh0 * 1.01) else 0     # No meaningful dilution

    # ── Operating Efficiency ──────────────────────────────────────────────────
    rev1 = g(is_["revenue"], yr1) or 1
    rev0 = g(is_["revenue"], yr0) or 1
    gp1  = g(is_["gross_profit"], yr1)
    gp0  = g(is_["gross_profit"], yr0)
    gm1  = gp1 / rev1
    gm0  = gp0 / rev0
    at1  = rev1 / ta1
    at0  = rev0 / ta0

    f8 = 1 if gm1 > gm0 else 0   # Gross margin expanding
    f9 = 1 if at1 > at0 else 0   # Asset turnover improving

    total = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    strength = "Strong" if total >= 8 else "Neutral" if total >= 5 else "Weak"

    return {
        "metric":          "piotroski_f_score",
        "score":           total,
        "out_of":          9,
        "strength":        strength,
        "years_compared":  f"{yr0} → {yr1}",
        "breakdown": {
            "profitability": {
                "F1_roa_positive":          f1,
                "F2_roa_improving":         f2,
                "F3_positive_ocf":          f3,
                "F4_quality_earnings":      f4,
                "subtotal":                 f1 + f2 + f3 + f4,
            },
            "leverage_and_liquidity": {
                "F5_leverage_falling":      f5,
                "F6_liquidity_improving":   f6,
                "F7_no_share_dilution":     f7,
                "subtotal":                 f5 + f6 + f7,
            },
            "operating_efficiency": {
                "F8_gross_margin_expanding": f8,
                "F9_asset_turnover_rising":  f9,
                "subtotal":                  f8 + f9,
            },
        },
    }


# ── Profitability Tools ───────────────────────────────────────────────────────

@tool
def compute_profitability_ratios(cik: str, num_years: int = 3) -> dict:
    """
    Compute profitability and returns metrics over num_years fiscal years.

    Metrics:
      - Gross / Operating / Net / EBITDA Margins
      - FCF Margin = FCF / Revenue
      - FCF Conversion = FCF / Net Income  (earnings quality; <0.8 is a warning)
      - ROE = Net Income / Equity
      - ROA = Net Income / Total Assets
      - ROIC = NOPAT / Invested Capital  (benchmark against ~10% WACC)

    Args:
        cik:       SEC Central Index Key
        num_years: Fiscal years to analyse (default 3)
    """
    is_ = _fetch_income_stmt(cik, num_years)
    bs  = _fetch_balance_sheet(cik, num_years)
    cf  = _fetch_cash_flow(cik, num_years)

    years = sorted(is_["years_covered"], reverse=True)
    values = {}

    for yr in years:
        rev    = is_["revenue"].get(yr) or 1
        gp     = is_["gross_profit"].get(yr)
        op_inc = is_["operating_income"].get(yr)
        ni     = is_["net_income"].get(yr)
        ebitda = is_["ebitda"].get(yr)
        fcf    = cf["fcf"].get(yr)
        eq     = bs["total_equity"].get(yr) or 1
        ta     = bs["total_assets"].get(yr) or 1
        debt   = bs["total_debt"].get(yr, 0) or 0
        cash   = bs["cash"].get(yr, 0) or 0

        # ROIC = NOPAT / Invested Capital
        # NOPAT ≈ Operating Income × (1 − effective tax rate); use 21% statutory
        nopat = (op_inc or 0) * 0.79
        invested_capital = eq + debt - cash

        values[yr] = {
            "gross_margin_pct":     _pct(_div(gp, rev)),
            "operating_margin_pct": _pct(_div(op_inc, rev)),
            "net_margin_pct":       _pct(_div(ni, rev)),
            "ebitda_margin_pct":    _pct(_div(ebitda, rev)),
            "fcf_margin_pct":       _pct(_div(fcf, rev)),
            "fcf_conversion":       _div(fcf, ni),   # <0.8 = earnings quality warning
            "roe_pct":              _pct(_div(ni, eq)),
            "roa_pct":              _pct(_div(ni, ta)),
            "roic_pct":             _pct(_div(nopat, invested_capital)) if invested_capital > 0 else None,
        }

    op_margins = {yr: v["operating_margin_pct"] for yr, v in values.items() if v["operating_margin_pct"]}

    return {
        "metric": "profitability_ratios",
        "years":  years,
        "values": values,
        "margin_trend": _trend(op_margins),
        "notes": {
            "fcf_conversion": "FCF/Net Income: >1.0 excellent | 0.8–1.0 good | <0.8 quality-of-earnings concern",
            "roic":           "ROIC > 10% (approx WACC) indicates value creation",
        },
    }


@tool
def compute_growth_metrics(cik: str, num_years: int = 5) -> dict:
    """
    Compute revenue, net income, EPS, and FCF growth — YoY and CAGR.

    Metrics:
      - YoY % growth for Revenue, Net Income, EPS Diluted, FCF
      - CAGR over the full available period for Revenue, Net Income, FCF

    Args:
        cik:       SEC Central Index Key
        num_years: Fiscal years to analyse (default 5 for CAGR meaningfulness)
    """
    is_ = _fetch_income_stmt(cik, num_years)
    cf  = _fetch_cash_flow(cik, num_years)

    years = sorted(is_["years_covered"])   # ascending for growth calc
    yoy = {}

    metric_sources = [
        ("revenue",    is_["revenue"]),
        ("net_income", is_["net_income"]),
        ("eps_diluted", is_["eps_diluted"]),
        ("fcf",        cf["fcf"]),
    ]

    for i in range(1, len(years)):
        yr, prev = years[i], years[i - 1]
        yoy[yr] = {}
        for name, data in metric_sources:
            curr = data.get(yr)
            prev_val = data.get(prev)
            if curr is not None and prev_val and prev_val != 0:
                yoy[yr][f"{name}_yoy_pct"] = round(((curr - prev_val) / abs(prev_val)) * 100, 2)

    # CAGR over full period
    cagr = {}
    if len(years) >= 2:
        n = len(years) - 1
        first, last = years[0], years[-1]
        for name, data in metric_sources:
            v0, vn = data.get(first), data.get(last)
            if v0 and vn and v0 > 0 and vn > 0:
                cagr[f"{name}_{n}y_cagr_pct"] = round(((vn / v0) ** (1 / n) - 1) * 100, 2)

    # Growth trajectory: is the most recent YoY higher or lower than the prior?
    trajectory = "insufficient data"
    rev_yoy = [(yr, v.get("revenue_yoy_pct")) for yr, v in yoy.items() if v.get("revenue_yoy_pct") is not None]
    if len(rev_yoy) >= 2:
        rev_yoy.sort()
        trajectory = "accelerating" if rev_yoy[-1][1] > rev_yoy[-2][1] else "decelerating"

    return {
        "metric":              "growth_metrics",
        "years_covered":       years,
        "yoy_growth":          yoy,
        "cagr":                cagr,
        "revenue_trajectory":  trajectory,
    }


@tool
def compute_dupont_analysis(cik: str, num_years: int = 3) -> dict:
    """
    Decompose ROE using the 3-factor DuPont model over num_years fiscal years.

    Formula: ROE = Net Profit Margin × Asset Turnover × Equity Multiplier
      Net Profit Margin  = Net Income / Revenue           (profitability)
      Asset Turnover     = Revenue / Total Assets         (efficiency)
      Equity Multiplier  = Total Assets / Total Equity    (leverage)

    Identifies whether ROE is driven by quality (margin/efficiency) or
    financial leverage — leverage-driven ROE is lower quality.

    Args:
        cik:       SEC Central Index Key
        num_years: Fiscal years to analyse (default 3)
    """
    is_ = _fetch_income_stmt(cik, num_years)
    bs  = _fetch_balance_sheet(cik, num_years)

    years = sorted(is_["years_covered"], reverse=True)
    values = {}

    for yr in years:
        rev = is_["revenue"].get(yr) or 1
        ni  = is_["net_income"].get(yr)
        ta  = bs["total_assets"].get(yr) or 1
        eq  = bs["total_equity"].get(yr) or 1

        net_margin       = _div(ni, rev)
        asset_turnover   = _div(rev, ta)
        equity_multiplier = _div(ta, eq)
        roe              = _div(ni, eq)

        # Reconstructed ROE should equal direct ROE (sanity check)
        reconstructed = (
            (net_margin or 0) * (asset_turnover or 0) * (equity_multiplier or 0)
            if all([net_margin, asset_turnover, equity_multiplier]) else None
        )

        # Identify dominant ROE driver
        if net_margin and asset_turnover and equity_multiplier:
            scores = {
                "profitability (net margin)": abs(net_margin * 10),
                "efficiency (asset turnover)": abs(asset_turnover),
                "leverage (equity multiplier)": abs(equity_multiplier - 1),
            }
            primary_driver = max(scores, key=scores.get)
        else:
            primary_driver = "insufficient data"

        values[yr] = {
            "net_profit_margin_pct":   _pct(net_margin),
            "asset_turnover":          round(asset_turnover, 4) if asset_turnover else None,
            "equity_multiplier":       round(equity_multiplier, 4) if equity_multiplier else None,
            "roe_pct":                 _pct(roe),
            "reconstructed_roe_pct":   _pct(reconstructed),
            "primary_roe_driver":      primary_driver,
        }

    return {
        "metric":         "dupont_analysis",
        "formula":        "ROE = Net Margin × Asset Turnover × Equity Multiplier",
        "years":          years,
        "values":         values,
        "interpretation": (
            "Equity multiplier >2.0 and rising suggests leverage is inflating ROE. "
            "Quality ROE comes from margin and turnover improvement, not leverage."
        ),
    }
