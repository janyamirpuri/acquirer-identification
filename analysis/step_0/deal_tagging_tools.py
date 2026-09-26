"""
deal_tagging_tools.py

Tool library for an LLM agent that fits free-text deal descriptions into a
structured schema, using a CSV of prior deals as the reference set for
calibrating relative descriptors (e.g. "strong EBITDA margin").

Design notes
------------
- Percentile thresholds for "strong / standard / weak" are computed live
  from the reference CSV, not hardcoded -- they update automatically as
  more deals get added to the book.
- Relative descriptors (margin, growth) are usually sector-dependent: a
  "strong" EBITDA margin in Health IT looks nothing like one in Healthcare
  Services. `group_by` lets you compute percentiles within a peer group
  (e.g. sector) instead of the whole population, and falls back to the
  full dataset automatically if the peer group is too thin to be
  statistically meaningful (< min_group_size rows).
- Every public method on DealComps is meant to be exposed to the LLM as a
  callable tool. TOOL_SCHEMAS at the bottom gives Claude/OpenAI-style
  JSON tool definitions you can pass straight into a tool-use API call.
- Deterministic arithmetic (implied multiples) is done in code, not left
  for the model to compute -- cheaper and removes a class of hallucinated
  numbers.
"""

from __future__ import annotations
import json
from typing import Optional, Literal
import pandas as pd
from pydantic import BaseModel, Field

try:
    from langchain_core.tools import StructuredTool
except ImportError:  # pragma: no cover
    StructuredTool = None


# --------------------------------------------------------------------------
# 1. Target output schema -- what the agent is ultimately filling in
# --------------------------------------------------------------------------

class DealRecord(BaseModel):
    sector: str
    sub_sector: Optional[str] = None
    deal_size_mm: Optional[float] = None
    deal_type: Optional[str] = None
    target_revenue_mm: Optional[float] = None
    target_ebitda_mm: Optional[float] = None
    ev_ebitda_multiple: Optional[float] = None
    ev_revenue_multiple: Optional[float] = None
    ebitda_margin_pct: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    geography: Optional[str] = None
    target_ownership_pre: Optional[Literal["Public", "Private"]] = None

# --------------------------------------------------------------------------
# 2. Reference-deal comps engine
# --------------------------------------------------------------------------

class DealComps:
    def __init__(self, csv_path: str, min_group_size: int = 15):
        self.df = pd.read_csv(csv_path)
        self.min_group_size = min_group_size

    # ---- tool: relative stat thresholds ----
    def relative_stat_thresholds(
        self,
        column: str,
        group_by: Optional[str] = None,
        group_value: Optional[str] = None,
    ) -> dict:
        """
        Compute the 25th/50th/75th percentile break points for `column`,
        optionally within a peer group (group_by == group_value), and
        return the strong/standard/weak bucket definitions.
        """
        if column not in self.df.columns:
            raise ValueError(f"Unknown column: {column}")

        subset = self.df
        used_group = None
        if group_by and group_value and group_by in self.df.columns:
            candidate = self.df[self.df[group_by] == group_value]
            if len(candidate) >= self.min_group_size:
                subset = candidate
                used_group = group_value

        series = subset[column].dropna()
        if len(series) < 5:
            # peer group (or whole file) too thin -- fall back to full population
            series = self.df[column].dropna()
            used_group = None

        p25 = float(series.quantile(0.25))
        p50 = float(series.quantile(0.50))
        p75 = float(series.quantile(0.75))

        return {
            "column": column,
            "peer_group": {"field": group_by, "value": used_group} if used_group else None,
            "n_deals": int(len(series)),
            "p25": round(p25, 3),
            "p50": round(p50, 3),
            "p75": round(p75, 3),
            "tiers": {
                "weak": f"< {round(p25, 3)}",
                "standard": f"{round(p25, 3)} - {round(p75, 3)}",
                "strong": f"> {round(p75, 3)}",
            },
        }


    # ---- tool: implied multiple (deterministic, no LLM arithmetic) ----
    @staticmethod
    def compute_implied_multiple(deal_size_mm: float, metric_mm: float) -> Optional[float]:
        """EV / metric, e.g. deal_size_mm / target_ebitda_mm -> ev_ebitda_multiple."""
        if not metric_mm:
            return None
        return round(deal_size_mm / metric_mm, 2)


# --------------------------------------------------------------------------
# 3. Tool schemas for an LLM tool-use loop
# --------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "relative_stat_thresholds",
            "description": (
                "Get the strong/standard/weak percentile breakpoints for a numeric "
                "column in the prior-deals CSV (25th/50th/75th percentile cuts), "
                "optionally computed within a peer group such as sector."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "e.g. 'ebitda_margin_pct'"},
                    "group_by": {"type": "string", "description": "e.g. 'sector' (optional)"},
                    "group_value": {"type": "string", "description": "e.g. 'Healthcare Services' (optional)"},
                },
                "required": ["column"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_implied_multiple",
            "description": "Deterministically compute EV / metric (e.g. EV/EBITDA or EV/Revenue). Always use this instead of doing the division yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "deal_size_mm": {"type": "number"},
                    "metric_mm": {"type": "number"},
                },
                "required": ["deal_size_mm", "metric_mm"],
                "additionalProperties": False,
            },
        },
    },
]


# --------------------------------------------------------------------------
# 4. Dispatcher -- wire tool_use blocks from the API straight into DealComps
# --------------------------------------------------------------------------

def dispatch_tool_call(comps: DealComps, tool_name: str, tool_input: dict):
    fn = getattr(comps, tool_name, None)
    if fn is None:
        raise ValueError(f"No such tool: {tool_name}")
    return fn(**tool_input)


def make_langchain_tools(comps: DealComps):
    """Expose DealComps as LangChain tools while preserving the same tool names and behavior."""
    if StructuredTool is None:
        raise ImportError("Install langchain-core to use the LangChain tool wrappers.")

    return [
        StructuredTool.from_function(
            func=comps.relative_stat_thresholds,
            name="relative_stat_thresholds",
            description=(
                "Get the strong/standard/weak percentile breakpoints for a numeric column "
                "in the prior-deals CSV (33rd/66th percentile cuts), optionally computed within "
                "a peer group such as sector."
            ),
        ),
        StructuredTool.from_function(
            func=comps.compute_implied_multiple,
            name="compute_implied_multiple",
            description=(
                "Deterministically compute EV / metric (e.g. EV/EBITDA or EV/Revenue). "
                "Always use this instead of doing the division yourself."
            ),
        ),
    ]


if __name__ == "__main__":
    # quick smoke test -- point this at your real CSV path
    comps = DealComps("external_docs/ma_transactions_500.csv")
    print(json.dumps(comps.relative_stat_thresholds("ebitda_margin_pct"), indent=2))
    print(json.dumps(
        comps.relative_stat_thresholds("ebitda_margin_pct", group_by="sector", group_value="Healthcare Services"),
        indent=2,
    ))
