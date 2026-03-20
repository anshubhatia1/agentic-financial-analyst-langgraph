from typing_extensions import TypedDict
from typing import Annotated, List, Optional
from langgraph.graph import StateGraph, add_messages

class MacroEconomicAgentState(TypedDict):
    """
    Comprehensive state for the macro economic analysis agent.

    Tracks the full execution context:
    - messages: Conversation history with add_messages reducer
    - query: Original user query (immutable reference)
    """
    messages: Annotated[list, add_messages]
    query: str