# Agentic Financial Analyst with LangGraph

An intelligent multi-agent system for comprehensive financial and macroeconomic analysis powered by LangGraph and GPT-4o. This system leverages a ReAct (Reasoning + Acting) pattern to autonomously fetch real-time economic data, synthesize insights, and generate institutional-grade financial reports.

## 🎯 Overview

The **Agentic Financial Analyst** is a LangGraph-based system that enables AI agents to:
- **Analyze macroeconomic indicators** using FRED (Federal Reserve Economic Data)
- **Process financial market data** from Yahoo Finance
- **Conduct web research** via Tavily API for contextual insights
- **Generate comprehensive reports** with LLM-powered synthesis
- **Parse natural language queries** with automatic date extraction

The system follows the ReAct (Reasoning + Acting) loop, where the LLM agent iteratively:
1. Reasons about what data is needed
2. Selects and executes appropriate tools
3. Synthesizes results into coherent analysis

## ✨ Key Features

- **Macroeconomic Analysis**: Access 500K+ economic time series from FRED
  - GDP and economic growth indicators (real GDP, industrial production, PMI)
  - Labor market data (unemployment, payrolls, wages, jobless claims)
  - Inflation indicators (CPI, PCE, PPI, inflation expectations)
  - Financial policy data (Fed Funds Rate, yield curve, credit spreads, money supply)
  - Advanced metrics: MoM change, YoY change, z-scores, rolling averages, trend analysis

- **Natural Language Interface**: Query the system in plain English
  - Automatic date range parsing (e.g., "last 6 months", "since 2020")
  - Context-aware tool selection by the LLM

- **Future Capabilities** (prepared but not yet active):
  - Yahoo Finance market data (equity, bonds, FX)
  - Tavily web search for contextual insights

- **Institutional-Grade Reporting**: LLM synthesis of multi-source data
  - Structured analysis with proper context
  - Risk considerations and caveats
  - Professional formatting

- **Extensible Agent Architecture**: Built with LangGraph for easy customization
  - Modular subgraph design
  - Tool binding and execution
  - State management via TypedDict

## 🛠️ Tech Stack

- **LLM Framework**: LangChain & LangGraph
- **Language Model**: OpenAI GPT-4o
- **Active Data Sources**:
  - FRED API (macroeconomic indicators)
- **Prepared but Inactive**:
  - Yahoo Finance (market data) — imported in requirements.txt
  - Tavily (web search) — imported in config.py
  - FAISS (vector store) — optional document retrieval
- **Data Processing**: Pandas, NumPy
- **Configuration**: Pydantic, python-dotenv

## 📋 Prerequisites

- Python 3.10+
- API Keys (required):
  - **OpenAI API key** (for GPT-4o)
  - **FRED API key** (free tier at [FRED](https://fredaccount.stlouisfed.org/login/secure/))
- API Keys (optional, for future features):
  - Tavily API key (for web search feature when enabled)

## 🚀 Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd agentic-financial-analyst-langgraph
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-...
FRED_API_KEY=your_fred_api_key
TAVILY_API_KEY=your_tavily_api_key  # Optional
```

## 📁 Project Structure

```
agentic-financial-analyst-langgraph/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.py                          # LLM and tool configuration
├── state.py                           # Agent state definition (TypedDict)
├── agents/
│   └── macro_economic_subgraph.py    # ReAct loop for macro analysis
├── tools/
│   └── macro_economic/
│       ├── fred_tools.py             # FRED data fetching tools
│       ├── date_parser.py            # Natural language date parsing
│       └── constants.py              # System prompts and configurations
├── test.ipynb                        # Example usage notebook
└── .env                              # Environment variables (local)
```

## 🏗️ Architecture

### System Overview

```
User Query
    ↓
date_parser (NLP date extraction)
    ↓
analyst (LLM reasoning with tools)
    ↓
should_continue (Router: tools needed?)
    ├─→ tools (Execute FRED, yfinance, Tavily)
    │   ↓
    └─→ analyst (Synthesize results)
    ↓
END (Return final report)
```

### Core Components

**1. MacroEconomicAgentState** (`state.py`)
- Manages conversation history with message deduplication
- Tracks original user query throughout execution
- Uses LangGraph's `add_messages` reducer

**2. Macro Economic Subgraph** (`agents/macro_economic_subgraph.py`)
- **date_parser**: Extracts start/end dates from natural language
- **analyst**: LLM node bound with available tools
- **tools**: Parallel execution of selected data-fetching tools
- **Router**: Conditional logic to continue or end the loop

**3. FRED Tools** (`tools/macro_economic/fred_tools.py`)
- Fetch economic indicators by series ID
- Handle date ranges and data aggregation
- Integrate with LangChain's tool interface

**4. LLM Configuration** (`config.py`)
- GPT-4o with low temperature (0.2) for consistency
- Tool binding for structured outputs
- Automatic retry logic

## 💻 Usage

### Basic Example (Jupyter Notebook)

```python
from agents.macro_economic_subgraph import build_macro_graph
from state import MacroEconomicAgentState

# Build the agent graph
graph = build_macro_graph()

# Prepare input
initial_state = {
    "messages": [],
    "query": "What has GDP growth been like over the past 2 years?"
}

# Run the agent
result = graph.invoke(initial_state)

# Access final analysis
final_response = result["messages"][-1].content
print(final_response)
```


## 📊 Example Queries

The system currently handles macroeconomic queries:

- **Trend Analysis**: "Show me CPI inflation trends over the past 5 years"
- **Labor Markets**: "How has unemployment changed in the last 2 years?"
- **Growth Indicators**: "What's the latest GDP data and industrial production?"
- **Monetary Policy**: "Show me the Federal Funds Rate and yield curve trends"
- **Multi-indicator**: "Are we heading into recession? Check GDP, unemployment, and yield curve"
- **Date Ranges**: "What was inflation like from 2020 to 2022?" (automatic date parsing)

## 🧪 Testing

Run the included Jupyter notebook for interactive testing:

```bash
jupyter notebook test.ipynb
```
