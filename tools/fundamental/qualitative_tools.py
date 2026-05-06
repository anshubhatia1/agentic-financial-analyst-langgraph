"""
Qualitative Text Extraction Tools
==================================
Fetches and parses narrative sections from SEC EDGAR 10-K filings.

Public @tools:
  search_company_filings      → list available 10-K filings
  extract_business_description → Item 1  (products, strategy, competitive position)
  extract_risk_factors         → Item 1A (material risk disclosures)
  extract_mda_section          → Item 7  (management narrative — most analytically rich)
  get_insider_ownership        → insider % and top institutional holders via yfinance

Note: Text extraction is best-effort. Modern 10-K HTML filings parse well.
Older filings (pre-2010) or unusual formats may return partial text.
"""

import re
import time
import requests
import yfinance as yf
from typing import Optional
from bs4 import BeautifulSoup
from langchain_core.tools import tool

_SEC_HEADERS = {
    "User-Agent": "fundamental-analyst-agent research@financial-agent.com",
    "Accept-Encoding": "gzip, deflate",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _padded_cik(cik: str) -> str:
    return str(cik).lstrip("0").zfill(10)


def _rate_limit():
    time.sleep(0.15)


def _get_submissions(cik: str) -> dict:
    """Fetch the company's full filing history from SEC submissions API."""
    url = f"https://data.sec.gov/submissions/CIK{_padded_cik(cik)}.json"
    resp = requests.get(url, headers=_SEC_HEADERS, timeout=20)
    resp.raise_for_status()
    _rate_limit()
    return resp.json()


def _get_latest_10k_info(cik: str) -> Optional[dict]:
    """Return accession number and filing date of the most recent 10-K."""
    subs = _get_submissions(cik)
    recent = subs.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates      = recent.get("filingDate", [])

    for form, acc, date in zip(forms, accessions, dates):
        if form in ("10-K", "10-K/A"):
            return {"accession": acc.replace("-", ""), "date": date, "form": form}
    return None


def _get_primary_doc_url(cik: str, accession: str) -> Optional[str]:
    """
    Find the URL of the primary 10-K HTML document from the filing index.
    Returns the first .htm file listed as the primary document.
    """
    bare_cik = str(int(_padded_cik(cik)))  # remove leading zeros for path
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{bare_cik}/"
        f"{accession}/{accession}-index.htm"
    )

    try:
        resp = requests.get(index_url, headers=_SEC_HEADERS, timeout=20)
        _rate_limit()
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Scan table rows for the primary document
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 3:
                # The primary 10-K row typically has type "10-K" in the last cell
                doc_type = cells[-1].get_text(strip=True) if cells else ""
                link_tag = cells[2].find("a") if len(cells) > 2 else None
                href = link_tag.get("href", "") if link_tag else ""

                if "10-K" in doc_type and href.endswith(".htm"):
                    return f"https://www.sec.gov{href}"

        # Fallback: find first large .htm file that contains the accession number
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith(".htm") and bare_cik in href and accession[:10] in href.replace("-", ""):
                return f"https://www.sec.gov{href}"

    except Exception:
        pass

    return None


def _fetch_10k_text(cik: str) -> Optional[str]:
    """Download and convert the primary 10-K HTML document to plain text."""
    info = _get_latest_10k_info(cik)
    if not info:
        return None

    doc_url = _get_primary_doc_url(cik, info["accession"])
    if not doc_url:
        return None

    try:
        resp = requests.get(doc_url, headers=_SEC_HEADERS, timeout=60)
        resp.raise_for_status()
        _rate_limit()
        soup = BeautifulSoup(resp.content, "html.parser")
        # Remove script/style noise
        for tag in soup(["script", "style", "ix:header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        return None


def _extract_section(
    text: str,
    start_patterns: list[str],
    end_patterns: list[str],
    max_chars: int = 8000,
) -> str:
    """
    Extract a section of text between start and end regex patterns.
    Searches case-insensitively. Returns up to max_chars of content.
    """
    text_upper = text.upper()
    start_pos = None

    for pat in start_patterns:
        m = re.search(pat, text_upper)
        if m:
            start_pos = m.start()
            break

    if start_pos is None:
        return "Section not found in this filing."

    end_pos = len(text)
    search_from = start_pos + 50   # skip past the header itself

    for pat in end_patterns:
        m = re.search(pat, text_upper[search_from:])
        if m:
            candidate = search_from + m.start()
            if candidate < end_pos:
                end_pos = candidate

    section = text[start_pos:end_pos].strip()
    return section[:max_chars]


# ── Public @tools ─────────────────────────────────────────────────────────────

@tool
def search_company_filings(cik: str, form_type: str = "10-K", num_filings: int = 5) -> dict:
    """
    List the most recent SEC filings of a given type for a company.

    Useful to confirm what data is available before extracting sections.

    Args:
        cik:        SEC Central Index Key
        form_type:  Filing type to search (default "10-K")
        num_filings: Number of filings to return (default 5)
    """
    subs    = _get_submissions(cik)
    recent  = subs.get("filings", {}).get("recent", {})
    forms   = recent.get("form", [])
    accs    = recent.get("accessionNumber", [])
    dates   = recent.get("filingDate", [])
    bare_cik = str(int(_padded_cik(cik)))

    results = []
    for form, acc, date in zip(forms, accs, dates):
        if form == form_type:
            acc_nodash = acc.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{bare_cik}/{acc_nodash}/"
            results.append({
                "form":             form,
                "filing_date":      date,
                "accession_number": acc,
                "index_url":        url,
            })
            if len(results) >= num_filings:
                break

    return {
        "cik":     cik,
        "company": subs.get("name", ""),
        "filings": results,
    }


@tool
def extract_business_description(cik: str) -> str:
    """
    Extract Item 1 (Business Description) from the most recent 10-K filing.

    Returns management's description of the company's products, services,
    competitive position, strategy, and market environment.
    Focus on identifying the core competitive moat (if any).

    Args:
        cik: SEC Central Index Key
    """
    text = _fetch_10k_text(cik)
    if not text:
        return "Could not fetch 10-K filing. Check that the CIK is correct."

    return _extract_section(
        text,
        start_patterns=[
            r"ITEM\s*1[\.\s]+BUSINESS\b",
            r"ITEM\s+1\b(?!\s*A)",
        ],
        end_patterns=[
            r"ITEM\s*1A[\.\s]+RISK",
            r"ITEM\s*2[\.\s]",
        ],
        max_chars=6000,
    )


@tool
def extract_risk_factors(cik: str) -> str:
    """
    Extract Item 1A (Risk Factors) from the most recent 10-K filing.

    Returns the material risks the company has formally disclosed to investors.
    Focus on identifying the 3 most impactful and non-boilerplate risks.

    Args:
        cik: SEC Central Index Key
    """
    text = _fetch_10k_text(cik)
    if not text:
        return "Could not fetch 10-K filing. Check that the CIK is correct."

    return _extract_section(
        text,
        start_patterns=[
            r"ITEM\s*1A[\.\s]+RISK\s*FACTORS",
            r"RISK\s*FACTORS",
        ],
        end_patterns=[
            r"ITEM\s*1B[\.\s]",
            r"ITEM\s*2[\.\s]",
        ],
        max_chars=8000,
    )


@tool
def extract_mda_section(cik: str) -> str:
    """
    Extract Item 7 (MD&A) from the most recent 10-K filing.

    Returns management's own narrative explaining revenue drivers, margin dynamics,
    liquidity position, and forward-looking guidance language.
    This is the most analytically important text section in any 10-K.

    Args:
        cik: SEC Central Index Key
    """
    text = _fetch_10k_text(cik)
    if not text:
        return "Could not fetch 10-K filing. Check that the CIK is correct."

    return _extract_section(
        text,
        start_patterns=[
            r"ITEM\s*7[\.\s]+MANAGEMENT.S\s+DISCUSSION\s+AND\s+ANALYSIS",
            r"MANAGEMENT.S\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL",
        ],
        end_patterns=[
            r"ITEM\s*7A[\.\s]+QUANTITATIVE",
            r"ITEM\s*8[\.\s]+FINANCIAL\s+STATEMENTS",
        ],
        max_chars=10000,
    )


@tool
def get_insider_ownership(ticker: str) -> dict:
    """
    Fetch insider and institutional ownership data for a company via yfinance.

    Insider ownership > 10% = strong management alignment.
    Net insider selling by multiple insiders (not RSU vests) is a red flag.
    Institutional concentration > 80% can signal crowding risk.

    Args:
        ticker: Stock ticker symbol (e.g., "MSFT")
    """
    stock = yf.Ticker(ticker)
    info  = stock.info

    # Institutional holders table
    try:
        inst_df = stock.institutional_holders
        top_institutions = (
            inst_df[["Holder", "Shares", "% Out"]].head(5).to_dict("records")
            if inst_df is not None and not inst_df.empty else []
        )
    except Exception:
        top_institutions = []

    # Ownership percentages
    insider_pct      = info.get("heldPercentInsiders", 0) or 0
    institution_pct  = info.get("heldPercentInstitutions", 0) or 0

    # Qualitative signal
    if insider_pct > 0.10:
        insider_signal = "High insider ownership (>10%) — strong management alignment"
    elif insider_pct > 0.03:
        insider_signal = "Moderate insider ownership (3–10%)"
    else:
        insider_signal = "Low insider ownership (<3%) — limited management skin in the game"

    return {
        "ticker":                    ticker,
        "insider_ownership_pct":     round(insider_pct * 100, 2),
        "institutional_ownership_pct": round(institution_pct * 100, 2),
        "top_5_institutional_holders": top_institutions,
        "insider_signal":            insider_signal,
        "note": (
            "Insider selling may reflect personal liquidity needs (RSU vests) "
            "rather than lack of conviction. Focus on the pattern, not individual transactions."
        ),
    }
