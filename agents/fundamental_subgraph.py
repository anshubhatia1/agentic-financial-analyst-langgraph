"""
Fundamental Analysis Agent -- Multi-Subagent Graph with Reflection

Architecture:

    User Query
        |
    resolve_company          (ticker -> CIK, company name)
        |
    [financial_health || profitability || qualitative]  <- parallel
        |
    synthesize               (orchestrator LLM combines all 3 reports)
        |
    critic                   (checks completeness + analytical depth)
        |
    should_revise?
        |-- "revise" --> synthesize  (with critique in context, max 2 iterations)
        `-- "end"    --> END

Subagents (each is a create_react_agent ReAct loop):
  - Financial Health  -- liquidity, leverage, Altman Z, Piotroski F
  - Profitability     -- margins, ROIC, growth, DuPont
  - Qualitative       -- MD&A, risk factors, business description, insider ownership

Reflection:
  - Critic evaluates completeness (8 required sections) + analytical depth
  - Max 2 reflection iterations before forcing final output
  - Critique is injected into the synthesizer prompt on revision passes
"""

import re
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

from config import llm
from state import FundamentalAgentState

# -- Tool imports --------------------------------------------------------------
from tools.fundamental.edgar_tools import (
    resolve_ticker_to_cik,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow_statement,
)
from tools.fundamental.ratio_tools import (
    compute_liquidity_ratios,
    compute_leverage_ratios,
    compute_altman_zscore,
    compute_piotroski_fscore,
    compute_profitability_ratios,
    compute_growth_metrics,
    compute_dupont_analysis,
)
from tools.fundamental.qualitative_tools import (
    search_company_filings,
    extract_business_description,
    extract_risk_factors,
    extract_mda_section,
    get_insider_ownership,
)
from tools.fundamental.constants import (
    FINANCIAL_HEALTH_PROMPT,
    PROFITABILITY_PROMPT,
    QUALITATIVE_PROMPT,
    SYNTHESIZER_PROMPT,
    CRITIC_PROMPT,
)

# -- Tool sets per subagent ----------------------------------------------------

FINANCIAL_HEALTH_TOOLS = [
    compute_liquidity_ratios,
    compute_leverage_ratios,
    compute_altman_zscore,
    compute_piotroski_fscore,
]

PROFITABILITY_TOOLS = [
    compute_profitability_ratios,
    compute_growth_metrics,
    compute_dupont_analysis,
]

QUALITATIVE_TOOLS = [
    search_company_filings,
    extract_business_description,
    extract_risk_factors,
    extract_mda_section,
    get_insider_ownership,
]

# -- Subagent graphs (each is a self-contained ReAct loop) ---------------------

financial_health_agent = create_agent(
    model=llm,
    tools=FINANCIAL_HEALTH_TOOLS,
    system_prompt=FINANCIAL_HEALTH_PROMPT,
)

profitability_agent = create_agent(
    model=llm,
    tools=PROFITABILITY_TOOLS,
    system_prompt=PROFITABILITY_PROMPT,
)

qualitative_agent = create_agent(
    model=llm,
    tools=QUALITATIVE_TOOLS,
    system_prompt=QUALITATIVE_PROMPT,
)


# -- Ticker extraction helper --------------------------------------------------

# Common stopwords to exclude from simple word-match candidates
_STOPWORDS = {
    "A", "AN", "THE", "IS", "IN", "OF", "FOR", "AND", "OR", "MY", "ME",
    "BY", "AS", "AT", "TO", "ON", "UP", "DO", "HOW", "WHAT", "GIVE",
    "SHOW", "LAST", "OVER", "YEAR", "YEARS", "FULL", "DEEP", "GOOD",
    "LONG", "TERM", "GIVE", "HELP", "WITH", "FROM", "THAT", "THIS",
    "ARE", "ITS", "HAS", "BEEN", "HAVE", "DOES", "THEIR", "ABOUT",
    "OVER", "PAST", "NEXT", "TELL", "FIND", "GET", "BASE", "BASED",
}


def _extract_ticker_with_llm(query: str) -> str:
    """
    Use the LLM to extract a stock ticker from a natural language query.

    Handles all forms:
      - Explicit ticker : "Analyze AAPL"         -> "AAPL"
      - Company name    : "Analyze Apple"         -> "AAPL"
      - Informal name   : "How is Google doing?"  -> "GOOGL"
      - Mixed           : "Is Microsoft cheap?"   -> "MSFT"

    Returns the ticker string (uppercase) or "" if none found.
    """
    response = llm.invoke([
        SystemMessage(content=(
            "You are a financial data assistant. "
            "Extract the stock ticker symbol of the company mentioned in the user query. "
            "Return ONLY the ticker symbol in uppercase letters -- nothing else. "
            "Examples: AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, TSLA. "
            "If you cannot identify any company, return exactly: UNKNOWN"
        )),
        HumanMessage(content=query),
    ])

    raw = response.content.strip().upper()

    # Accept a clean 1-5 letter ticker
    if re.fullmatch(r"[A-Z]{1,5}", raw) and raw != "UNKNOWN":
        return raw

    # Fallback: grab the first plausible ticker token from the response
    tokens = re.findall(r"\b[A-Z]{1,5}\b", raw)
    filtered = [t for t in tokens if t not in _STOPWORDS and t != "UNKNOWN"]
    return filtered[0] if filtered else ""


# -- Node: Company Resolution --------------------------------------------------

def resolve_company_node(state: FundamentalAgentState) -> dict:
    """
    Resolve ticker -> CIK and company name via SEC EDGAR.

    Resolution order:
      1. Use ticker already in state (if set by caller)
      2. Simple regex word-match on query  (catches explicit tickers: "AAPL")
      3. LLM extraction                   (catches company names: "Apple", "Google")
      4. If SEC lookup fails on step-2 result, retry with LLM extraction
    """
    ticker = state.get("ticker", "").strip().upper()

    # If CIK is already populated, nothing to resolve
    if state.get("cik"):
        return {}

    # Extract query from state or messages
    query = state.get("query", "")
    if not query and state.get("messages"):
        # Extract from the first HumanMessage
        for msg in state["messages"]:
            if hasattr(msg, "content") and msg.__class__.__name__ == "HumanMessage":
                query = msg.content
                break

    # Step 1: simple word-match -- fast, no LLM call
    if not ticker:
        words = re.findall(r"\b[A-Za-z]{1,5}\b", query)
        candidates = [w.upper() for w in words if w.upper() not in _STOPWORDS]
        ticker = candidates[0] if candidates else ""

    # Step 2: LLM extraction -- handles company names and informal references
    llm_ticker = ""
    if not ticker:
        llm_ticker = _extract_ticker_with_llm(query)
        ticker = llm_ticker

    if not ticker:
        return {"error": "Could not identify a company or ticker in the query."}

    # Attempt SEC resolution
    result = resolve_ticker_to_cik.invoke({"ticker": ticker})

    # If simple match returned wrong word, retry with LLM extraction
    if "error" in result:
        if not llm_ticker:
            llm_ticker = _extract_ticker_with_llm(query)
        if llm_ticker and llm_ticker != ticker:
            result = resolve_ticker_to_cik.invoke({"ticker": llm_ticker})
            ticker = llm_ticker

    if "error" in result:
        return {
            "error": (
                f"Could not resolve '{ticker}' in SEC EDGAR. "
                "Please verify the company name or ticker symbol."
            )
        }

    return {
        "ticker":       result["ticker"],
        "cik":          result["cik"],
        "company_name": result["company_name"],
    }


# -- Node: Financial Health Subagent -------------------------------------------

def financial_health_node(state: FundamentalAgentState) -> dict:
    """
    Run the Financial Health subagent (ReAct loop).
    Computes liquidity, leverage, Altman Z-Score, Piotroski F-Score.
    """
    if state.get("error"):
        return {}

    company = state.get("company_name", state.get("ticker", "Unknown"))
    ticker  = state.get("ticker", "")
    cik     = state.get("cik", "")
    years   = state.get("years", 3)

    prompt = (
        f"Analyze the financial health of {company} "
        f"(Ticker: {ticker}, CIK: {cik}) for the past {years} fiscal years.\n\n"
        f"Use all available tools: compute liquidity ratios, leverage ratios, "
        f"Altman Z-Score, and Piotroski F-Score. "
        f"Interpret every metric with trend context and flag any red flags clearly."
    )

    try:
        result = financial_health_agent.invoke({
            "messages": [HumanMessage(content=prompt)]
        })
        report = result["messages"][-1].content
    except Exception as e:
        report = f"Financial Health analysis failed: {str(e)}"

    return {"financial_health_report": report}


# -- Node: Profitability Subagent ----------------------------------------------

def profitability_node(state: FundamentalAgentState) -> dict:
    """
    Run the Profitability & Growth subagent (ReAct loop).
    Computes margins, ROIC, growth rates, DuPont decomposition.
    """
    if state.get("error"):
        return {}

    company = state.get("company_name", state.get("ticker", "Unknown"))
    cik     = state.get("cik", "")
    years   = state.get("years", 3)

    prompt = (
        f"Analyze the profitability and growth of {company} (CIK: {cik}) "
        f"for the past {years} fiscal years.\n\n"
        f"Use all available tools: profitability ratios, growth metrics, and DuPont analysis. "
        f"Identify margin trends, ROIC vs. cost of capital (~10% WACC benchmark), "
        f"FCF quality, and whether revenue growth is accelerating or decelerating."
    )

    try:
        result = profitability_agent.invoke({
            "messages": [HumanMessage(content=prompt)]
        })
        report = result["messages"][-1].content
    except Exception as e:
        report = f"Profitability analysis failed: {str(e)}"

    return {"profitability_report": report}


# -- Node: Qualitative Subagent ------------------------------------------------

def qualitative_node(state: FundamentalAgentState) -> dict:
    """
    Run the Qualitative subagent (ReAct loop).
    Extracts MD&A, risk factors, business description, insider ownership from 10-K.
    """
    if state.get("error"):
        return {}

    company = state.get("company_name", state.get("ticker", "Unknown"))
    ticker  = state.get("ticker", "")
    cik     = state.get("cik", "")

    prompt = (
        f"Provide qualitative analysis for {company} (Ticker: {ticker}, CIK: {cik}).\n\n"
        f"Use all available tools in this order:\n"
        f"1. search_company_filings -- confirm 10-K availability\n"
        f"2. extract_business_description -- identify the core competitive moat\n"
        f"3. extract_mda_section -- extract key management narrative and guidance signals\n"
        f"4. extract_risk_factors -- identify the top 3 material (non-boilerplate) risks\n"
        f"5. get_insider_ownership -- assess management alignment\n\n"
        f"Cross-reference: does management's MD&A narrative align with what the financial "
        f"numbers show? Flag any disconnect."
    )

    try:
        result = qualitative_agent.invoke({
            "messages": [HumanMessage(content=prompt)]
        })
        report = result["messages"][-1].content
    except Exception as e:
        report = f"Qualitative analysis failed: {str(e)}"

    return {"qualitative_report": report}


# -- Node: Synthesizer ---------------------------------------------------------

def synthesize_node(state: FundamentalAgentState) -> dict:
    """
    Generate a concise synthesized investment response
    tailored to the user's query.
    """

    query            = state.get("query", "")
    company          = state.get("company_name", state.get("ticker", "Unknown"))
    ticker           = state.get("ticker", "")
    reflection_count = state.get("reflection_count", 0)
    critique         = state.get("critique", "")

    subagent_content = (
        f"=== FINANCIAL HEALTH ANALYSIS ===\n"
        f"{state.get('financial_health_report') or 'Not available.'}\n\n"

        f"=== PROFITABILITY & GROWTH ANALYSIS ===\n"
        f"{state.get('profitability_report') or 'Not available.'}\n\n"

        f"=== QUALITATIVE ANALYSIS ===\n"
        f"{state.get('qualitative_report') or 'Not available.'}"
    )

    revision_instruction = ""
    if reflection_count > 0 and critique:
        revision_instruction = (
            f"\n\nRevise the response by addressing the following critique:\n"
            f"{critique}"
        )

    user_content = (
        f"USER QUERY:\n{query}\n\n"
        f"COMPANY: {company} ({ticker})\n\n"
        f"{subagent_content}"
        f"{revision_instruction}"
    )

    messages = [
        SystemMessage(content=SYNTHESIZER_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)

    return {
        "final_report": response.content,
        "messages": [response],
    }


# -- Node: Critic (Reflection) -------------------------------------------------

def critic_node(state: FundamentalAgentState) -> dict:
    """
    Evaluate the synthesized report for completeness and analytical depth.
    Returns is_approved (bool) and critique (str).
    """
    report           = state.get("final_report", "")
    company          = state.get("company_name", state.get("ticker", "Unknown"))
    reflection_count = state.get("reflection_count", 0)

    user_content = (
        f"Evaluate this fundamental analysis report for {company}:\n\n"
        f"{report}\n\n"
        'Return valid JSON only: {"is_approved": true|false, "critique": "..."}'
    )

    messages = [
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        parsed      = json.loads(raw)
        is_approved = bool(parsed.get("is_approved", True))
        critique    = parsed.get("critique", "")
    except Exception:
        is_approved = True
        critique    = ""

    return {
        "is_approved":      is_approved,
        "critique":         critique,
        "reflection_count": reflection_count + 1,
        "messages":         [response],
    }


# -- Router --------------------------------------------------------------------

def should_revise(state: FundamentalAgentState) -> str:
    """Route back to synthesizer or end based on critic approval and iteration count."""
    MAX_REFLECTIONS = 2

    if state.get("is_approved"):
        return "end"

    if state.get("reflection_count", 0) >= MAX_REFLECTIONS:
        return "end"

    return "revise"


# -- Graph assembly ------------------------------------------------------------

def build_fundamental_graph():
    """
    Build and return the compiled fundamental analysis graph.

    Execution order:
      1. resolve_company  (serial)
      2. financial_health, profitability, qualitative  (parallel fan-out)
      3. synthesize  (fan-in, reflection entry point)
      4. critic  (reflection evaluator)
      5. should_revise -> "revise" back to synthesize | "end" -> END
    """
    workflow = StateGraph(FundamentalAgentState)

    workflow.add_node("resolve_company",  resolve_company_node)
    workflow.add_node("financial_health", financial_health_node)
    workflow.add_node("profitability",    profitability_node)
    workflow.add_node("qualitative",      qualitative_node)
    workflow.add_node("synthesize",       synthesize_node)
    workflow.add_node("critic",           critic_node)

    workflow.set_entry_point("resolve_company")

    # Fan-out: resolve -> all 3 subagents in parallel
    workflow.add_edge("resolve_company", "financial_health")
    workflow.add_edge("resolve_company", "profitability")
    workflow.add_edge("resolve_company", "qualitative")

    # Fan-in: all 3 subagents -> synthesize
    workflow.add_edge("financial_health", "synthesize")
    workflow.add_edge("profitability",    "synthesize")
    workflow.add_edge("qualitative",      "synthesize")

    # Reflection loop
    workflow.add_edge("synthesize", "critic")
    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {"revise": "synthesize", "end": END},
    )

    return workflow.compile()
