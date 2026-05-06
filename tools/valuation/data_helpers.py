"""
Raw data helpers for the valuation agent.

These functions are NOT exposed as LangChain tools — they are internal
utilities called by the tool functions in valuation_tools.py.

Functions
---------
get_market_snapshot(ticker)  → price, market cap, enterprise value
get_income_data(ticker)      → revenue, net income, operating income, EPS, book value
get_cashflow_data(ticker)    → operating CF, capex, FCF, D&A
"""

import yfinance as yf


def get_market_snapshot(ticker: str) -> dict:
    """
    Return the three core market-price figures needed for multiple calculations.

    Keys
    ----
    price            : Current share price
    market_cap       : Total market capitalisation
    enterprise_value : EV (market cap + net debt)
    """
    info = yf.Ticker(ticker).info
    return {
        "price":            info.get("currentPrice"),
        "market_cap":       info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
    }


def get_income_data(ticker: str) -> dict:
    """
    Return the most-recent annual income statement figures plus per-share data.

    Keys
    ----
    revenue_ttm          : Trailing twelve-month total revenue
    net_income           : Latest annual net income
    operating_income     : Latest annual operating income (EBIT)
    eps_ttm              : Trailing twelve-month diluted EPS
    book_value_per_share : Shareholders' equity per share
    """
    stock      = yf.Ticker(ticker)
    info       = stock.info
    financials = stock.financials
    latest     = financials.iloc[:, 0] if not financials.empty else {}

    return {
        "revenue_ttm":          info.get("totalRevenue"),
        "net_income":           latest.get("Net Income"),
        "operating_income":     latest.get("Operating Income"),
        "eps_ttm":              info.get("trailingEps"),
        "book_value_per_share": info.get("bookValue"),
    }


def get_cashflow_data(ticker: str) -> dict:
    """
    Return the most-recent annual cash flow statement figures.

    Keys
    ----
    operating_cash_flow       : Cash from operations
    capex                     : Capital expenditure (absolute value)
    fcf                       : Free cash flow = operating CF − capex
    depreciation_amortization : D&A (used to compute EBITDA)
    """
    stock    = yf.Ticker(ticker)
    cashflow = stock.cashflow
    cf       = cashflow.iloc[:, 0] if not cashflow.empty else {}

    operating_cf = cf.get("Operating Cash Flow", 0) or 0
    capex        = abs(cf.get("Capital Expenditure", 0) or 0)

    return {
        "operating_cash_flow":       operating_cf,
        "capex":                     capex,
        "fcf":                       operating_cf - capex,
        "depreciation_amortization": cf.get("Depreciation And Amortization", 0),
    }
