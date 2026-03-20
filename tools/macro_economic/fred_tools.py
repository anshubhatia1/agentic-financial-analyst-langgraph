from langchain_core.tools import tool
from tools.macro_economic.fred_fetcher import fetch_category


@tool
def fetch_economic_growth(start_date: str, end_date: str) -> str:
    """
    Fetch U.S. economic growth indicators from FRED for a given date range.

    Indicators returned:
      - Real GDP (GDPC1, quarterly, chained 2017$)
      - Industrial Production Index (INDPRO, monthly)
      - Real Personal Consumption Expenditures (PCEC96, monthly)
      - ISM Manufacturing PMI (NAPM, monthly)
      - Housing Starts (HOUST, monthly, thousands)

    Each indicator includes: latest value, MoM change, YoY change, 3M and 12M
    rolling averages, 5-year historical mean, z-score, and trend direction.

    Use for queries about GDP, economic output, manufacturing activity,
    consumer spending, housing market, or overall growth momentum.

    Args:
        start_date: Period start in YYYY-MM-DD format (e.g. "2024-01-01")
        end_date:   Period end   in YYYY-MM-DD format (e.g. "2024-12-31")
    """
    return fetch_category("economic_growth", start_date, end_date)


@tool
def fetch_labor_market(start_date: str, end_date: str) -> str:
    """
    Fetch U.S. labor market indicators from FRED for a given date range.

    Indicators returned:
      - Unemployment Rate (UNRATE, monthly, %)
      - Nonfarm Payrolls (PAYEMS, monthly, MoM change in thousands)
      - Labor Force Participation Rate (CIVPART, monthly, %)
      - Average Hourly Earnings (CES0500000003, monthly, $)
      - Initial Jobless Claims (ICSA, weekly, thousands)

    Each indicator includes: latest value, MoM change, YoY change, 3M and 12M
    rolling averages, 5-year historical mean, z-score, and trend direction.

    Use for queries about employment, jobs, wages, unemployment, labor force,
    payrolls, layoffs, or worker earnings.

    Args:
        start_date: Period start in YYYY-MM-DD format
        end_date:   Period end   in YYYY-MM-DD format
    """
    return fetch_category("labor_market", start_date, end_date)


@tool
def fetch_inflation(start_date: str, end_date: str) -> str:
    """
    Fetch U.S. inflation indicators from FRED for a given date range.

    Indicators returned:
      - Core PCE Price Index (PCEPILFE, monthly) — Fed's preferred gauge
      - Consumer Price Index — All Items (CPIAUCSL, monthly)
      - Producer Price Index — All Commodities (PPIACO, monthly)
      - Unit Labor Costs — Nonfarm Business (ULCNFB, quarterly)
      - 5-Year Breakeven Inflation Rate (T5YIE, daily)

    Each indicator includes: latest value, MoM change, YoY change, 3M and 12M
    rolling averages, 5-year historical mean, z-score, and trend direction.

    Use for queries about inflation, CPI, PCE, PPI, price levels, cost
    pressures, supply-side pricing, or inflation expectations.

    Args:
        start_date: Period start in YYYY-MM-DD format
        end_date:   Period end   in YYYY-MM-DD format
    """
    return fetch_category("inflation", start_date, end_date)


@tool
def fetch_financial_policy(start_date: str, end_date: str) -> str:
    """
    Fetch U.S. financial policy and monetary condition indicators from FRED.

    Indicators returned:
      - Federal Funds Rate (FEDFUNDS, monthly, %)
      - 10Y–2Y Treasury Spread (T10Y2Y, daily, %) — yield curve inversion signal
      - M2 Money Supply (M2SL, monthly, $B) — liquidity proxy
      - Moody's BAA–10Y Corporate Credit Spread (BAA10YM, monthly, %)
      - St. Louis Financial Stress Index (STLFSI4, weekly)

    Each indicator includes: latest value, MoM change, YoY change, 3M and 12M
    rolling averages, 5-year historical mean, z-score, and trend direction.

    Use for queries about interest rates, Fed policy, monetary tightening or
    easing, yield curve, credit spreads, liquidity, or financial stability.

    Args:
        start_date: Period start in YYYY-MM-DD format
        end_date:   Period end   in YYYY-MM-DD format
    """
    return fetch_category("financial_policy", start_date, end_date)


# Convenience list — imported by the subgraph and the ToolNode
MACRO_TOOLS = [
    fetch_economic_growth,
    fetch_labor_market,
    fetch_inflation,
    fetch_financial_policy,
]