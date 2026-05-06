"""
LangChain tools for the Valuation Analysis agent.

Five tools, called in sequence by the orchestrator:
  detect_financial_stage   → classify company stage; always called first
  run_earnings_analysis    → P/E, Forward P/E, PEG, P/B
  run_cashflow_analysis    → P/FCF, EV/EBITDA, EV/FCF, FCF yield
  run_revenue_analysis     → P/S, EV/Revenue, Forward P/S
  fetch_analyst_estimates  → Wall St consensus, price targets, growth rates

Raw data is fetched via helpers in data_helpers.py (not exposed as tools).
"""

import yfinance as yf
from langchain_core.tools import tool

from tools.valuation.data_helpers import (
    get_market_snapshot,
    get_income_data,
    get_cashflow_data,
)


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_div(a, b):
    """Return round(a/b, 2) or None when division is undefined."""
    return round(a / b, 2) if a and b and b > 0 else None


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def detect_financial_stage(ticker: str) -> dict:
    """
    Fetch core financials and classify the company into one of four stages.
    Always call this first — the stage determines which analysis tools apply.

    Returns stage, net_income, fcf, revenue so the LLM can reason
    about which tools to call next without redundant fetches.

    Stages
    ------
    profitable_cash_generative  → run_earnings_analysis, run_cashflow_analysis, run_revenue_analysis
    profitable_cash_consuming   → run_earnings_analysis, run_revenue_analysis
    unprofitable_revenue_stage  → run_revenue_analysis
    pre_revenue                 → no quantitative valuation possible
    """
    income   = get_income_data(ticker)
    cashflow = get_cashflow_data(ticker)

    net_income = income.get("net_income") or 0
    fcf        = cashflow.get("fcf") or 0
    revenue    = income.get("revenue_ttm") or 0

    if net_income > 0 and fcf > 0:
        stage             = "profitable_cash_generative"
        recommended_tools = ["run_earnings_analysis", "run_cashflow_analysis", "run_revenue_analysis"]
    elif net_income > 0 and fcf <= 0:
        stage             = "profitable_cash_consuming"
        recommended_tools = ["run_earnings_analysis", "run_revenue_analysis"]
    elif net_income <= 0 and revenue > 0:
        stage             = "unprofitable_revenue_stage"
        recommended_tools = ["run_revenue_analysis"]
    else:
        stage             = "pre_revenue"
        recommended_tools = []

    return {
        "ticker":            ticker,
        "stage":             stage,
        "recommended_tools": recommended_tools,
        "net_income":        net_income,
        "fcf":               fcf,
        "revenue_ttm":       revenue,
        "note": (
            f"Call these tools next: {recommended_tools}"
            if recommended_tools
            else "Quantitative valuation not applicable — company has no revenue."
        ),
    }


@tool
def run_earnings_analysis(ticker: str) -> dict:
    """
    Calculate all earnings-based valuation multiples.
    Use for: profitable_cash_generative, profitable_cash_consuming stages.

    Computes: P/E (TTM), Forward P/E, PEG Ratio, P/B (Price-to-Book).

    Benchmarks
    ----------
    PEG < 1.0  → growth is underpriced (potential buy signal)
    PEG > 2.0  → growth is overpriced
    P/E < 15   → cheap | 15–30 → moderate | > 30 → premium
    P/B < 1    → trading below book value
    """
    market      = get_market_snapshot(ticker)
    income      = get_income_data(ticker)
    info        = yf.Ticker(ticker).info

    price       = market["price"]
    eps_ttm     = income.get("eps_ttm")
    forward_eps = info.get("forwardEps")
    growth_rate = info.get("earningsGrowth")   # decimal, e.g. 0.25 = 25%
    book_value  = income.get("book_value_per_share")

    pe         = _safe_div(price, eps_ttm)
    forward_pe = _safe_div(price, forward_eps)
    peg        = _safe_div(pe, (growth_rate or 0) * 100) if pe else None
    pb         = _safe_div(price, book_value)

    return {
        "ticker":        ticker,
        "analysis_type": "earnings_multiples",
        "price":         price,
        "pe_ttm":        pe,
        "forward_pe":    forward_pe,
        "peg_ratio":     peg,
        "price_to_book": pb,
        "interpretation": {
            "pe":  "N/A" if pe  is None else ("cheap"       if pe  < 15 else "premium"      if pe  > 30 else "moderate"),
            "peg": "N/A" if peg is None else ("undervalued" if peg < 1  else "overvalued"   if peg > 2  else "fair"),
            "pb":  "N/A" if pb  is None else ("below book"  if pb  < 1  else "premium to book"),
        },
    }


@tool
def run_cashflow_analysis(ticker: str) -> dict:
    """
    Calculate cash-flow-based valuation multiples.
    Use for: profitable_cash_generative stage only (requires positive FCF).

    Computes: P/FCF, EV/EBITDA, EV/FCF, FCF yield.
    Cash-flow multiples are harder to manipulate than earnings-based metrics.

    Benchmarks
    ----------
    EV/EBITDA < 10 → cheap | 10–25 → moderate | > 25 → expensive
    P/FCF     < 20 → attractive | 20–50 → moderate | > 50 → expensive
    FCF yield > 5% → strong cash return
    """
    market   = get_market_snapshot(ticker)
    income   = get_income_data(ticker)
    cashflow = get_cashflow_data(ticker)

    mc     = market["market_cap"]
    ev     = market["enterprise_value"]
    fcf    = cashflow["fcf"]
    da     = cashflow["depreciation_amortization"] or 0
    op_inc = income.get("operating_income") or 0
    ebitda = op_inc + da

    pfcf      = _safe_div(mc, fcf)
    ev_ebitda = _safe_div(ev, ebitda)
    ev_fcf    = _safe_div(ev, fcf)

    return {
        "ticker":        ticker,
        "analysis_type": "cashflow_multiples",
        "fcf":           fcf,
        "ebitda":        ebitda,
        "p_fcf":         pfcf,
        "ev_ebitda":     ev_ebitda,
        "ev_fcf":        ev_fcf,
        "fcf_yield_pct": round((fcf / mc) * 100, 2) if mc and fcf and fcf > 0 else None,
        "interpretation": {
            "ev_ebitda": "N/A" if ev_ebitda is None else (
                "cheap" if ev_ebitda < 10 else "expensive" if ev_ebitda > 25 else "moderate"
            ),
            "p_fcf": "N/A" if pfcf is None else (
                "attractive" if pfcf < 20 else "expensive" if pfcf > 50 else "moderate"
            ),
        },
    }


@tool
def run_revenue_analysis(ticker: str) -> dict:
    """
    Calculate revenue-based valuation multiples.
    Use for: all stages that have revenue (not pre_revenue).
    Essential for unprofitable companies where earnings metrics don't apply.

    Computes: P/S (Price-to-Sales), EV/Revenue, Forward P/S.

    Benchmarks
    ----------
    P/S < 2  → cheap | 2–10 → moderate | > 10 → expensive (most sectors)
    """
    market = get_market_snapshot(ticker)
    income = get_income_data(ticker)
    info   = yf.Ticker(ticker).info

    mc          = market["market_cap"]
    ev          = market["enterprise_value"]
    revenue_ttm = income.get("revenue_ttm")
    forward_rev = info.get("revenueEstimate")   # may be None in yfinance

    ps         = _safe_div(mc, revenue_ttm)
    ev_revenue = _safe_div(ev, revenue_ttm)
    forward_ps = _safe_div(mc, forward_rev)

    return {
        "ticker":         ticker,
        "analysis_type":  "revenue_multiples",
        "revenue_ttm":    revenue_ttm,
        "price_to_sales": ps,
        "ev_to_revenue":  ev_revenue,
        "forward_ps":     forward_ps,
        "interpretation": {
            "ps": "N/A" if ps is None else (
                "cheap" if ps < 2 else "expensive" if ps > 10 else "moderate"
            ),
        },
    }


@tool
def fetch_analyst_estimates(ticker: str) -> dict:
    """
    Fetch Wall Street analyst consensus: price targets, recommendation,
    and forward growth estimates.

    Call this to add analyst sentiment context to the valuation verdict.

    Keys returned
    -------------
    recommendation       : strong_buy / buy / hold / sell / strong_sell
    analyst_count        : number of analysts covering the stock
    price_target_mean    : consensus 12-month price target
    price_target_high    : most bullish analyst target
    price_target_low     : most bearish analyst target
    implied_upside_pct   : % upside from current price to mean target
    forward_eps          : next-year consensus EPS estimate
    forward_pe           : forward P/E implied by consensus EPS
    earnings_growth_rate : forward earnings growth rate (decimal)
    """
    info        = yf.Ticker(ticker).info
    price       = info.get("currentPrice")
    target_mean = info.get("targetMeanPrice")

    upside = None
    if price and target_mean:
        upside = round(((target_mean - price) / price) * 100, 1)

    return {
        "ticker":               ticker,
        "recommendation":       info.get("recommendationKey"),
        "analyst_count":        info.get("numberOfAnalystOpinions"),
        "price_target_mean":    target_mean,
        "price_target_high":    info.get("targetHighPrice"),
        "price_target_low":     info.get("targetLowPrice"),
        "implied_upside_pct":   upside,
        "forward_eps":          info.get("forwardEps"),
        "forward_pe":           info.get("forwardPE"),
        "earnings_growth_rate": info.get("earningsGrowth"),
    }
