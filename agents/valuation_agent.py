"""
valuation_agent.py
------------------
Fundamental stock valuation agent built with LangGraph.

The agent follows a structured analysis loop:
  1. detect_financial_stage  – classify company maturity & pick relevant tools
  2. run_earnings_analysis   – P/E, Forward P/E, PEG, P/B  (profitable companies)
  3. run_cashflow_analysis   – P/FCF, EV/EBITDA, EV/FCF    (cash-generative)
  4. run_revenue_analysis    – P/S, EV/Revenue              (all revenue-stage cos)
  5. fetch_analyst_estimates – Wall St consensus & price targets

Final output: Undervalued / Fairly Valued / Overvalued verdict with confidence %.

Usage
-----
    from agents.valuation_agent import run

    result = run("Is Apple undervalued right now?")
    print(result)
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from state import ValuationAgentState


from tools.valuation.constants import VALUATION_SYSTEM_PROMPT
from tools.valuation.valuation_tools import (
    detect_financial_stage,
    run_earnings_analysis,
    run_cashflow_analysis,
    run_revenue_analysis,
    fetch_analyst_estimates,
)

load_dotenv()

# ── LLM + tools ───────────────────────────────────────────────────────────────
TOOLS = [
    detect_financial_stage,
    run_earnings_analysis,
    run_cashflow_analysis,
    run_revenue_analysis,
    fetch_analyst_estimates,
]

_llm_v = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
    max_retries=4,
).bind_tools(TOOLS)


# ── Graph nodes ───────────────────────────────────────────────────────────────

def orchestrator(state: ValuationAgentState) -> ValuationAgentState:
    """
    The single agent node. The LLM:
      - Reads the conversation (user query + any previous tool results)
      - Decides whether to call a tool or produce a final answer
      - If tool_calls are present → graph routes to tool node
      - If no tool_calls → graph routes to END
    """
    system   = SystemMessage(content=VALUATION_SYSTEM_PROMPT)
    response = _llm_v.invoke([system] + state["messages"])
    return {
        "messages":  [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def should_continue(state: ValuationAgentState) -> str:
    """Route to tools if the LLM issued tool calls, else end."""
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END


# ── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(ValuationAgentState)
    g.add_node("orchestrator", orchestrator)
    g.add_node("tools", ToolNode(TOOLS))
    g.set_entry_point("orchestrator")
    g.add_conditional_edges(
        "orchestrator",
        should_continue,
        {"tools": "tools", END: END},
    )
    g.add_edge("tools", "orchestrator")
    return g.compile()
