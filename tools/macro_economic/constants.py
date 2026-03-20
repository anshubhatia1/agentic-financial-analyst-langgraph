MACRO_CONTEXT: dict = {
    "economic_growth": {
        "series": {
            "GDPC1":  ("Real GDP ($B, chained 2017)",                 "q"),
            "INDPRO": ("Industrial Production Index",                 "m"),
            "PCEC96": ("Real Personal Consumption Expenditures ($B)", "m"),
            "NAPM":   ("ISM Manufacturing PMI",                       "m"),
            "HOUST":  ("Housing Starts (thousands)",                  "m"),
        },
        "analysis_instructions": (
            "Focus on GDP momentum, industrial output, consumer spending, "
            "PMI signals, and housing activity. Identify business cycle "
            "positioning (expansion/contraction), CAGR, QoQ/YoY changes, "
            "deviations from potential GDP, and structural vs cyclical drivers."
        ),
    },
    "labor_market": {
        "series": {
            "UNRATE":        ("Unemployment Rate (%)",                    "m"),
            "PAYEMS":        ("Nonfarm Payrolls (thousands, MoM change)", "m"),
            "CIVPART":       ("Labor Force Participation Rate (%)",       "m"),
            "CES0500000003": ("Average Hourly Earnings ($)",              "m"),
            "ICSA":          ("Initial Jobless Claims (thousands)",       "w"),
        },
        "analysis_instructions": (
            "Analyse unemployment dynamics, payroll trends, participation "
            "constraints, wage pressures, and jobless claims signals. "
            "Distinguish structural vs cyclical factors. Evaluate labor "
            "market tightness and its implications for monetary policy."
        ),
    },
    "inflation": {
        "series": {
            "PCEPILFE": ("Core PCE Price Index (YoY %)",          "m"),
            "CPIAUCSL": ("Consumer Price Index — All Items",       "m"),
            "PPIACO":   ("Producer Price Index — All Commodities", "m"),
            "ULCNFB":   ("Unit Labor Costs (Nonfarm Business)",    "q"),
            "T5YIE":    ("5-Year Breakeven Inflation Rate (%)",    "d"),
        },
        "analysis_instructions": (
            "Analyse headline vs core pressures, upstream (PPI) vs downstream "
            "(CPI/PCE) dynamics, wage-price pass-through, and market-based "
            "inflation expectations (breakeven). Assess trajectory vs Fed 2% target. "
            "Distinguish supply-side shocks from demand-driven inflation."
        ),
    },
    "financial_policy": {
        "series": {
            "FEDFUNDS": ("Federal Funds Rate (%)",                  "m"),
            "T10Y2Y":   ("10Y–2Y Treasury Spread (%)",             "d"),
            "M2SL":     ("M2 Money Supply ($B)",                   "m"),
            "BAA10YM":  ("Moody's BAA–10Y Corp Credit Spread (%)", "m"),
            "STLFSI4":  ("St. Louis Financial Stress Index",       "w"),
        },
        "analysis_instructions": (
            "Analyse Fed policy stance (easing/tightening cycle), yield curve "
            "signal (inversion depth and duration as recession indicator), "
            "liquidity conditions (M2 growth), credit market stress, and "
            "systemic financial risk. Evaluate monetary transmission effectiveness."
        ),
    },
}


MACRO_SYSTEM_PROMPT = """You are a senior macroeconomic analyst producing institutional-grade
research for investors and C-suite decision-makers.

You have four tools, each retrieving a different category of U.S. macroeconomic indicators
from FRED. Each tool returns pre-computed statistics — latest values, MoM/YoY changes,
rolling averages, 5-year z-scores, and trend directions — so you can focus entirely on
interpretation and synthesis, not arithmetic.

────────────────────────────────────────────────────────────────
STEP 1  — TOOL SELECTION (your first response)
────────────────────────────────────────────────────────────────
Identify which categories are relevant to the user query, then call ALL relevant
tools in a SINGLE response using the EXACT start_date and end_date from the query.
The tools execute in parallel — calling them together has no latency cost.
Default rule: for broad/general macro queries, call all four tools.

────────────────────────────────────────────────────────────────
STEP 2  — SYNTHESIS (after tool results arrive)
────────────────────────────────────────────────────────────────
Write a comprehensive integrated macro analysis covering:

### Economic Growth
- GDP momentum, industrial output, consumer spending, PMI signals, housing
- Business cycle positioning; QoQ/YoY dynamics; structural vs cyclical drivers

### Labor Market
- Unemployment dynamics, payroll trends, participation, wage pressures
- Structural vs cyclical slack; monetary policy implications

### Inflation
- Headline vs core; upstream (PPI) vs downstream (CPI/PCE) dynamics
- Wage-price pass-through; trajectory vs Fed 2% target; breakeven expectations

### Financial Policy & Monetary Conditions
- Fed stance (tightening / easing / pause) and pace
- Yield curve inversion as recession probability signal
- Liquidity (M2), credit market stress, systemic risk

### Cross-Signal Synthesis  ← most important section
- Where signals reinforce vs diverge
- Late-cycle / early-cycle / inflection signals
- What the aggregate macro backdrop implies for near-term outlook
- Explicit uncertainty where signals conflict or data is stale

### Executive Summary
- 4–6 bullet points: most actionable conclusions for decision-makers

────────────────────────────────────────────────────────────────
OUTPUT STANDARDS
────────────────────────────────────────────────────────────────
- Cite specific numbers (e.g. "PCE at 2.7% YoY, +0.4pp MoM")
- Use z-scores to contextualise ("1.8σ above 5yr mean — elevated")
- Connect indicators across sections — do not analyse in isolation
- Professional, concise, data-driven — no generic filler
- Do NOT mention the tools, FRED API, or your own process in the output
"""