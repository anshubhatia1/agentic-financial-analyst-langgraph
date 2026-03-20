"""
Macro Economic Analysis Subgraph

This module defines the ReAct (Reasoning + Acting) agent graph for macroeconomic analysis.
It leverages FRED tools to fetch economic indicators and uses an LLM to synthesize institutional-grade reports.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from config import llm
from state import MacroEconomicAgentState
from tools.macro_economic.date_parser import parse_date_range
from tools.macro_economic.fred_tools import MACRO_TOOLS
from tools.macro_economic.constants import MACRO_SYSTEM_PROMPT

# Bind the LLM with tools and create tool executor node
macro_llm = llm.bind_tools(MACRO_TOOLS)
tool_node = ToolNode(MACRO_TOOLS)



def date_parse_node(state: MacroEconomicAgentState) -> dict:
    """
    Extract start_date and end_date from the user query using NLP.

    This node parses natural language date references (e.g., "last 6 months")
    and enriches the user message with explicit date parameters for tool calls.

    Args:
        state: MacroEconomicAgentState containing messages

    Returns:
        Updated state with enriched HumanMessage containing parsed dates
    """
    query = state["query"]

    if not query:
        return {"messages": []}

    # Extract dates from the query
    try:
        start_date, end_date = parse_date_range(query)
        # Append parsed dates to the message content for tool invocation
        enriched_content = (
            f"{query}\n\n"
            f"[Parsed date range: start_date='{start_date}', end_date='{end_date}']"
        )
        enriched_message = HumanMessage(content=enriched_content)
        return {"messages": [enriched_message]}
    except Exception:
        # If date parsing fails, just return the original message
        return {"messages": []}


def analyst_node(state: MacroEconomicAgentState) -> dict:
    """
    Macro analyst LLM node.

    This node calls the LLM with MACRO_SYSTEM_PROMPT injected as context.
    On the first turn, it prepends the system prompt; on subsequent turns,
    it uses the existing conversation with tool results.

    The LLM is bound with MACRO_TOOLS, so it can decide which tools to call
    based on the user query and previous results.

    Args:
        state: MacroEconomicAgentState containing messages

    Returns:
        Updated state with LLM response appended to messages
    """
    messages = state["messages"]
    messages = [SystemMessage(content=MACRO_SYSTEM_PROMPT)] + messages
    # Invoke the LLM (it will decide which tools to call, if any)
    response = macro_llm.invoke(messages)
    # Append response to messages (add_messages reducer handles deduplication)
    return {"messages": [response]}


def should_continue(state: MacroEconomicAgentState) -> str:
    """
    Router function: decides whether to continue with tool execution or end.

    If the LLM's last message contains tool_calls, route to the tools node.
    Otherwise, the analysis is complete, so route to END.

    Args:
        state: MacroEconomicAgentState containing messages

    Returns:
        "tools" to execute tools, "end" to terminate the agent
    """
    messages = state["messages"]
    last_message = messages[-1]
    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return "end"


def build_macro_graph():
    """
    Build and return the macro economic analysis subgraph.

    The graph implements a ReAct loop:
    1. date_parser: Extract dates from user query
    2. analyst: LLM decides which tools to call
    3. tools: Execute the selected tools in parallel
    4. Loop back to analyst if more tools needed, else END

    Returns:
        Uncompiled StateGraph (not yet compiled)
    """
    workflow = StateGraph(MacroEconomicAgentState)

    # Add nodes
    workflow.add_node("date_parser", date_parse_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("date_parser")

    # Add edges
    workflow.add_edge("date_parser", "analyst")
    workflow.add_conditional_edges(
        "analyst",
        should_continue,
        {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "analyst")
    graph = workflow.compile()
    return graph
