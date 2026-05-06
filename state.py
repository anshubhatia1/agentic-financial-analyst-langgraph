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


class ValuationAgentState(TypedDict):
    messages:  Annotated[list, add_messages]
    llm_calls: int


class FundamentalAgentState(TypedDict):
    """
    State for the fundamental analysis multi-agent system.

    Flow: resolve_company → [financial_health ‖ profitability ‖ qualitative]
          → synthesize → critic → (revise loop, max 2) → END

    Subagent outputs feed the synthesizer. The critic drives the reflection loop.
    """
    messages: Annotated[list, add_messages]
    query: str                          # Original user query (e.g., "Analyze MSFT")
    ticker: str                         # Stock ticker (e.g., "MSFT")
    company_name: str                   # Full legal company name
    cik: str                            # SEC Central Index Key
    years: int                          # Lookback period (default 3)

    # ── Subagent outputs ──────────────────────────────────────────────────────
    financial_health_report: str        # Output of Financial Health subagent
    profitability_report: str           # Output of Profitability & Growth subagent
    qualitative_report: str             # Output of Qualitative subagent

    # ── Synthesis & Reflection ────────────────────────────────────────────────
    final_report: str                   # Synthesized report (may be revised)
    critique: str                       # Critic's feedback on completeness/depth
    reflection_count: int               # Number of reflection iterations so far
    is_approved: bool                   # True when critic approves the report

    # ── Error handling ────────────────────────────────────────────────────────
    error: Optional[str]                # Error message if any step fails