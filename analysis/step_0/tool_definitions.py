from __future__ import annotations
import json
from typing import Optional, Literal
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class DealRecord(BaseModel):
    sector: Optional[str] = Field(default=None, description="Sector of the target company")
    sub_sector: Optional[str] = Field(default=None, description="Sub-sector of the target company")
    deal_size_mm: Optional[float] = Field(default=None, description="Enterprise value of the transaction in USD millions")
    target_revenue_mm: Optional[float] = Field(default=None, description="Target company's revenue (market) in USD millions")
    target_ebitda_mm: Optional[float] = Field(default=None, description="Target company's EBITDA in USD millions")
    ev_ebitda_multiple: Optional[float] = Field(default=None, description="Enterprise value / target EBITDA multiple")
    ev_revenue_multiple: Optional[float] = Field(default=None, description="Enterprise value / target revenue multiple")
    ebitda_margin_pct: Optional[float] = Field(default=None, description="Target company's EBITDA margin in percentage")
    revenue_growth_pct: Optional[float] = Field(default=None, description="Target company's revenue growth in percentage")
    geography: Optional[str] = Field(default=None, description="Geography of the target company")
    target_ownership_pre: Optional[str] = Field(default=None, description="Target company's ownership status before the deal")

class DealComps:
    def __init__(self, csv_path: str, min_group_size: int = 15):
        self.df = pd.read_csv(csv_path)
        self.min_group_size = min_group_size

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
        (other relative terms accepted, such as low/mid/high)
        """
        if column not in self.df.columns:
            raise ValueError(f"Unknown column: {column}")

        subset = self.df
        used_group = None
        if group_by in self.df.columns:
            if group_value in self.df[group_by].values:
                candidate = self.df[self.df[group_by] == group_value]
                if len(candidate) >= self.min_group_size:
                    subset = candidate
                    used_group = group_value

        series = subset[column].dropna()
        if len(series) < 5:
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
        }


    @staticmethod
    def compute_implied_multiple(deal_size_mm: float, metric_mm: float) -> Optional[float]:
        """EV / metric, e.g. deal_size_mm / target_ebitda_mm -> ev_ebitda_multiple."""
        if not metric_mm or deal_size_mm is None:
            return None
        return round(deal_size_mm / metric_mm, 2)
    

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
                "Get the strong/standard/weak (other relative terms accepted, such as low/mid/high) percentile breakpoints for a numeric column "
                "in the prior-deals CSV (25th/50th/75th percentile cuts), optionally computed within "
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
