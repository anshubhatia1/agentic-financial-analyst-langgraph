"""
FRED data fetcher + statistics computer
=========================================
Fetches all series for a given macro category in parallel and returns
pre-computed, LLM-ready statistics — not raw tabular data.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import fredapi as fa
import pandas as pd
from tools.macro_economic.constants import MACRO_CONTEXT

logger = logging.getLogger(__name__)


def compute_series_stats(series: pd.Series, name: str, series_id: str) -> str:
    series = series.dropna()
    # series = series.sort_index()
    if series.empty:
        return f"  {name} ({series_id}): No data available for this period.\n"
    
    latest_val = series.iloc[-1]
    latest_date = str(series.index[-1].date())

    lines = [f"\n  {name} ({series_id}):"]
    lines.append(f"    Latest Value:   {latest_val:.4g}  [{latest_date}]")

    # MoM
    if len(series) >= 2:
        prev    = series.iloc[-2]
        mom     = latest_val - prev
        mom_pct = (mom / abs(prev) * 100) if prev != 0 else 0.0
        lines.append(f"    MoM Change:     {mom:+.4g}  ({mom_pct:+.2f}%)")

    # YoY — try 13 lags (monthly), fall back to 5 (quarterly)
    for lag, label in [(13, "YoY"), (5, "YoY(q)")]:
        if len(series) >= lag:
            prior   = series.iloc[-lag]
            yoy     = latest_val - prior
            yoy_pct = (yoy / abs(prior) * 100) if prior != 0 else 0.0
            lines.append(f"    {label} Change:     {yoy:+.4g}  ({yoy_pct:+.2f}%)")
            break
    # Rolling averages
    if len(series) >= 3:
        lines.append(f"    3M Average:     {series.iloc[-3:].mean():.4g}")
    if len(series) >= 12:
        lines.append(f"    12M Average:    {series.iloc[-12:].mean():.4g}")

    # Z-score vs 5-year history (up to 60 observations)
    lookback = min(60, len(series))
    if lookback >= 12:
        hist = series.iloc[-lookback:]
        mean = hist.mean()
        std  = hist.std()
        if std > 0:
            z     = (latest_val - mean) / std
            label = "above" if z > 0 else "below"
            lines.append(f"    5yr Mean:       {mean:.4g}")
            lines.append(f"    Z-Score (5yr):  {z:+.2f}σ  ({label} historical avg)")

    # Trend from last 4 observations
    if len(series) >= 4:
        diffs = series.iloc[-4:].diff().dropna()
        pos   = (diffs > 0).sum()
        neg   = (diffs < 0).sum()
        if   pos >= 3: trend = "Rising consistently"
        elif neg >= 3: trend = "Falling consistently"
        elif pos > neg: trend = "Mostly rising"
        elif neg > pos: trend = "Mostly falling"
        else:           trend = "Mixed / Sideways"
        lines.append(f"    Trend (4 obs):  {trend}")

    return "\n".join(lines) + "\n"


def fetch_category(category: str, start_date: str, end_date: str) -> str:
    """
    Fetch all series for *category* from FRED in parallel threads, then
    return a compact pre-computed statistics block for the LLM to interpret.

    Concurrency model:
      One thread per series → all HTTP calls fire simultaneously.
      With 5 series per category this cuts latency from 5× to 1× fetch time.
    """
    fred        = fa.Fred(os.getenv("FRED_API_KEY", ""))
    series_meta = MACRO_CONTEXT[category]["series"]
    instructions = MACRO_CONTEXT[category]["analysis_instructions"]

    def _fetch_one(sid: str, name: str, freq: str):
        return sid, name, fred.get_series(
            series_id=sid,
            frequency=freq,
            observation_start=start_date,
            observation_end=end_date,
        )

    results: dict[str, tuple[str, pd.Series]] = {}
    failed:  list[str] = []

    with ThreadPoolExecutor(max_workers=len(series_meta)) as executor:
        futures = {
            executor.submit(_fetch_one, sid, name, freq): sid
            for sid, (name, freq) in series_meta.items()
        }
        for future in as_completed(futures):
            sid = futures[future]
            try:
                sid, name, data = future.result()
                results[sid] = (name, data)
            except Exception as exc:
                logger.warning("FRED fetch failed — %s: %s", sid, exc)
                failed.append(sid)

    cat_label = category.replace("_", " ").upper()
    header = (
        f"{'=' * 60}\n"
        f"  {cat_label} INDICATORS\n"
        f"  Period: {start_date}  →  {end_date}\n"
        f"{'=' * 60}\n"
    )
    stats_block   = "".join(
        compute_series_stats(data, name, sid)
        for sid, (name, data) in results.items()
    )
    analysis_note = f"\n  [Analysis focus: {instructions}]\n"
    error_note    = (
        f"\n  [⚠ Failed to fetch: {', '.join(failed)}]\n" if failed else ""
    )

    return header + stats_block + analysis_note + error_note