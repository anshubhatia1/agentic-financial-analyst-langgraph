"""
EDGAR Data Fetching Tools
=========================
Fetches structured financial statement data from SEC EDGAR via the XBRL API.

Public @tools (called by LLM subagents):
  resolve_ticker_to_cik   -> ticker -> CIK + company name
  get_income_statement    -> multi-year income statement from XBRL
  get_balance_sheet       -> multi-year balance sheet from XBRL
  get_cash_flow_statement -> multi-year cash flow statement from XBRL

Internal helpers (prefixed _):
  Used by ratio_tools.py to avoid redundant @tool invocations.
  _fetch_income_stmt, _fetch_balance_sheet, _fetch_cash_flow
"""

import time
import requests
from langchain_core.tools import tool

# SEC requires a descriptive User-Agent (app name + contact email)
_SEC_HEADERS = {
    "User-Agent": "fundamental-analyst-agent research@financial-agent.com",
    "Accept-Encoding": "gzip, deflate",
}


# -- Low-level helpers ---------------------------------------------------------

def _padded_cik(cik: str) -> str:
    """Zero-pad CIK to 10 digits as required by SEC APIs."""
    return str(cik).lstrip("0").zfill(10)


def _rate_limit():
    """Respect SEC's 10 req/sec rate limit."""
    time.sleep(0.15)


def _get_xbrl_facts(cik: str) -> dict:
    """
    Fetch all XBRL facts for a company from SEC.
    Returns the us-gaap facts dict (keyed by concept name).
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{_padded_cik(cik)}.json"
    resp = requests.get(url, headers=_SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    _rate_limit()
    return resp.json().get("facts", {}).get("us-gaap", {})


def _extract_annual(facts: dict, tag_aliases: list, num_years: int, unit: str = "USD") -> dict:
    """
    Extract the most recent num_years annual (FY) values for a financial concept.

    Collects entries from ALL aliases, merges them, then deduplicates by fiscal
    year (keeping the most recently *filed* entry per year).  This correctly
    handles companies that switched XBRL tags between reporting periods -- e.g.
    Apple moved from SalesRevenueNet (pre-2019) to
    RevenueFromContractWithCustomerExcludingAssessedTax (post-2019).  The old
    approach returned on the first tag that had *any* data, which could return
    stale data from 5-8 years ago if a legacy tag was tried first.

    Returns: {year_str: value, ...}  e.g. {"2024": 391035000000, "2023": 383285000000}
    """
    all_entries: list = []

    for tag in tag_aliases:
        if tag not in facts:
            continue
        unit_data = facts[tag].get("units", {}).get(unit, [])
        annual_entries = [
            v for v in unit_data
            if v.get("form") in ("10-K", "10-K/A")
            and v.get("fp") == "FY"
            and v.get("val") is not None
        ]
        all_entries.extend(annual_entries)

    if not all_entries:
        return {}

    # Sort by (fiscal end-date, filed date) descending -> most recent year first,
    # and within the same year the most recent *amendment* wins.
    all_entries.sort(
        key=lambda x: (x.get("end", ""), x.get("filed", "")),
        reverse=True,
    )

    # Deduplicate: one value per fiscal year (first seen = most recent filing)
    seen_years: set = set()
    result: dict = {}
    for v in all_entries:
        year = v["end"][:4]
        if year not in seen_years:
            seen_years.add(year)
            result[year] = v["val"]
        if len(result) == num_years:
            break

    return result


# -- Internal data-fetch helpers (used by ratio_tools) ------------------------

def _fetch_income_stmt(cik: str, num_years: int) -> dict:
    facts = _get_xbrl_facts(cik)
    ny = num_years

    revenue = _extract_annual(facts, [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ], ny)

    gross_profit = _extract_annual(facts, ["GrossProfit"], ny)

    cogs = _extract_annual(facts, [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
    ], ny)

    operating_income = _extract_annual(facts, ["OperatingIncomeLoss"], ny)

    net_income = _extract_annual(facts, ["NetIncomeLoss", "ProfitLoss"], ny)

    interest_expense = _extract_annual(facts, [
        "InterestExpense",
        "InterestAndDebtExpense",
    ], ny)

    da = _extract_annual(facts, [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
    ], ny)

    eps_basic = _extract_annual(facts, ["EarningsPerShareBasic"], ny, unit="USD/shares")
    eps_diluted = _extract_annual(facts, ["EarningsPerShareDiluted"], ny, unit="USD/shares")

    shares = _extract_annual(facts, [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ], ny, unit="shares")

    # Compute EBITDA = Operating Income + D&A
    ebitda = {}
    for yr in operating_income:
        op = operating_income.get(yr, 0) or 0
        d  = da.get(yr, 0) or 0
        if op:
            ebitda[yr] = op + d

    years_covered = sorted(revenue.keys(), reverse=True)

    return {
        "revenue":            revenue,
        "gross_profit":       gross_profit,
        "cogs":               cogs,
        "operating_income":   operating_income,
        "net_income":         net_income,
        "interest_expense":   interest_expense,
        "da":                 da,
        "ebitda":             ebitda,
        "eps_basic":          eps_basic,
        "eps_diluted":        eps_diluted,
        "shares_outstanding": shares,
        "years_covered":      years_covered,
        "cik":                cik,
    }


def _fetch_balance_sheet(cik: str, num_years: int) -> dict:
    facts = _get_xbrl_facts(cik)
    ny = num_years

    cash = _extract_annual(facts, [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "Cash",
    ], ny)

    current_assets      = _extract_annual(facts, ["AssetsCurrent"], ny)
    total_assets        = _extract_annual(facts, ["Assets"], ny)
    current_liabilities = _extract_annual(facts, ["LiabilitiesCurrent"], ny)
    total_liabilities   = _extract_annual(facts, ["Liabilities"], ny)

    long_term_debt = _extract_annual(facts, [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
    ], ny)

    short_term_debt = _extract_annual(facts, [
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
        "DebtCurrent",
    ], ny)

    total_equity = _extract_annual(facts, [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ], ny)

    retained_earnings   = _extract_annual(facts, ["RetainedEarningsAccumulatedDeficit"], ny)
    goodwill            = _extract_annual(facts, ["Goodwill"], ny)
    accounts_receivable = _extract_annual(facts, ["AccountsReceivableNetCurrent"], ny)
    inventory           = _extract_annual(facts, ["InventoryNet"], ny)
    accounts_payable    = _extract_annual(facts, ["AccountsPayableCurrent"], ny)
    ppe_net             = _extract_annual(facts, ["PropertyPlantAndEquipmentNet"], ny)

    # Compute total debt = LT + ST
    total_debt = {}
    for yr in total_assets:
        lt = long_term_debt.get(yr, 0) or 0
        st = short_term_debt.get(yr, 0) or 0
        total_debt[yr] = lt + st

    years_covered = sorted(total_assets.keys(), reverse=True)

    return {
        "cash":                cash,
        "current_assets":      current_assets,
        "total_assets":        total_assets,
        "current_liabilities": current_liabilities,
        "total_liabilities":   total_liabilities,
        "long_term_debt":      long_term_debt,
        "short_term_debt":     short_term_debt,
        "total_debt":          total_debt,
        "total_equity":        total_equity,
        "retained_earnings":   retained_earnings,
        "goodwill":            goodwill,
        "accounts_receivable": accounts_receivable,
        "inventory":           inventory,
        "accounts_payable":    accounts_payable,
        "ppe_net":             ppe_net,
        "years_covered":       years_covered,
        "cik":                 cik,
    }


def _fetch_cash_flow(cik: str, num_years: int) -> dict:
    facts = _get_xbrl_facts(cik)
    ny = num_years

    operating_cf = _extract_annual(facts, [
        "NetCashProvidedByUsedInOperatingActivities",
    ], ny)

    capex = _extract_annual(facts, [
        "PaymentsToAcquirePropertyPlantAndEquipment",
    ], ny)

    da = _extract_annual(facts, [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
    ], ny)

    dividends = _extract_annual(facts, [
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
    ], ny)

    buybacks = _extract_annual(facts, [
        "PaymentsForRepurchaseOfCommonStock",
    ], ny)

    acquisitions = _extract_annual(facts, [
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
    ], ny)

    # FCF = Operating CF - |CapEx|
    fcf = {}
    for yr in operating_cf:
        ocf  = operating_cf.get(yr, 0) or 0
        capx = abs(capex.get(yr, 0) or 0)
        if ocf:
            fcf[yr] = ocf - capx

    years_covered = sorted(operating_cf.keys(), reverse=True)

    return {
        "operating_cf":  operating_cf,
        "capex":         capex,
        "fcf":           fcf,
        "da":            da,
        "dividends":     dividends,
        "buybacks":      buybacks,
        "acquisitions":  acquisitions,
        "years_covered": years_covered,
        "cik":           cik,
    }


# -- Public @tools (called by LLM subagents) ----------------------------------

@tool
def resolve_ticker_to_cik(ticker: str) -> dict:
    """
    Convert a stock ticker to its SEC Central Index Key (CIK) and company name.
    Always call this first -- the CIK is required by all other EDGAR tools.

    Args:
        ticker: Stock ticker symbol (e.g., "MSFT", "AAPL")

    Returns:
        dict with keys: ticker, cik (zero-padded to 10 digits), company_name
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=_SEC_HEADERS, timeout=15)
    resp.raise_for_status()
    _rate_limit()

    ticker_upper = ticker.upper().strip()
    for entry in resp.json().values():
        if entry.get("ticker", "").upper() == ticker_upper:
            cik = str(entry["cik_str"]).zfill(10)
            return {
                "ticker":       ticker_upper,
                "cik":          cik,
                "company_name": entry.get("title", "Unknown"),
            }

    return {"error": f"Ticker '{ticker}' not found in SEC EDGAR."}


@tool
def get_income_statement(cik: str, num_years: int = 3) -> dict:
    """
    Extract a multi-year income statement from SEC EDGAR XBRL data.

    Returns Revenue, Gross Profit, COGS, Operating Income, EBITDA, Net Income,
    D&A, Interest Expense, EPS (basic & diluted), Shares Outstanding.

    Args:
        cik:       SEC Central Index Key (10-digit string)
        num_years: Number of fiscal years to retrieve (default 3)
    """
    return _fetch_income_stmt(cik, num_years)


@tool
def get_balance_sheet(cik: str, num_years: int = 3) -> dict:
    """
    Extract a multi-year balance sheet from SEC EDGAR XBRL data.

    Returns Cash, Current Assets/Liabilities, Total Assets/Liabilities,
    Short/Long-term Debt, Total Equity, Retained Earnings, Goodwill,
    Accounts Receivable, Inventory, Accounts Payable, PP&E.

    Args:
        cik:       SEC Central Index Key (10-digit string)
        num_years: Number of fiscal years to retrieve (default 3)
    """
    return _fetch_balance_sheet(cik, num_years)


@tool
def get_cash_flow_statement(cik: str, num_years: int = 3) -> dict:
    """
    Extract a multi-year cash flow statement from SEC EDGAR XBRL data.

    Returns Operating CF, CapEx, Free Cash Flow, D&A, Dividends, Buybacks, Acquisitions.
    FCF is computed as Operating CF minus CapEx.

    Args:
        cik:       SEC Central Index Key (10-digit string)
        num_years: Number of fiscal years to retrieve (default 3)
    """
    return _fetch_cash_flow(cik, num_years)
