from __future__ import annotations
import json
from typing import Optional, Literal, Any, Union
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from .subsection_descriptions import acquirer_overview, strategic_fit, precedent_activity, valuation_context, risk_flags, conviction_level

class Rationale(BaseModel):
    acquirer_name: str = Field(..., description="Name of the acquirer")
    acquirer_overview: Optional[str] = Field(None, description=acquirer_overview)
    strategic_fit: Optional[str] = Field(None, description=strategic_fit)
    precedent_activity: Optional[str] = Field(None, description=precedent_activity)
    valuation_context: Optional[str] = Field(None, description=valuation_context)
    risk_flags: Optional[str] = Field(None, description=risk_flags)
    conviction_level: Optional[str] = Field(None, description=conviction_level)


class CSV_Analytics:
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

    @staticmethod
    def compute_implied_multiple(deal_size_mm: float, metric_mm: float) -> Optional[float]:
        """EV / metric, e.g. deal_size_mm / target_ebitda_mm -> ev_ebitda_multiple."""
        if not metric_mm:
            return None
        return round(deal_size_mm / metric_mm, 2)

    def prior_transactions(self, acquirer: str) -> list[dict]:
        """Return a list of prior transactions for the given acquirer."""
        return self.df[self.df["acquirer"] == acquirer].to_dict(orient="records")

    def subset_transactions_by_peer_group(
        self,
        group_by: Union[str, list[str]],
        group_value: Union[Any, list[Any]],
    ) -> list[dict]:
        """Return a list of transactions matching the given peer group.

        Single column:
            subset_transactions_by_peer_group("sector", "Technology")

        Multiple columns (AND across columns):
            subset_transactions_by_peer_group(
                ["sector", "region"], ["Technology", "APAC"]
            )

        Multiple values within a column (OR within that column, via list/tuple/set):
            subset_transactions_by_peer_group(
                ["sector", "region"], [["Technology", "Healthcare"], "APAC"]
            )
        """
        if isinstance(group_by, str):
            group_by = [group_by]
            group_value = [group_value]

        if len(group_by) != len(group_value):
            raise ValueError("group_by and group_value must have the same length")

        mask = pd.Series(True, index=self.df.index)
        for col, val in zip(group_by, group_value):
            if isinstance(val, (list, tuple, set)):
                mask &= self.df[col].isin(val)
            else:
                mask &= self.df[col] == val

        return self.df[mask].to_dict(orient="records")

    def get_sector_benchmark(self, column: str, sector: str) -> str:
        """Return median deal size and EV/EBITDA multiple for a sector, for comparison."""
        rows = self.df[self.df["sector"] == sector]
        if rows.empty:
            return f"No transactions found for sector '{sector}'."
        return (
            f"{sector}: {len(rows)} deals, "
            f"median deal size ${rows.deal_size_mm.median():.0f}M, "
            f"median EV/EBITDA {rows.ev_ebitda_multiple.median():.1f}x"
        )

    def get_peer_group_benchmark(
        self,
        group_by: Union[str, list[str]],
        group_value: Union[Any, list[Any]],
        quantitative_columns: Union[str, list[str]],
    ) -> dict:
        """Return summary stats (count, mean, median) for one or more quantitative
        columns within a peer group defined by one or more group_by/group_value pairs.

        Returns a dict, e.g.:
            {
                "match_found": True,
                "filters": {"sector": "Technology", "region": "APAC"},
                "deal_count": 42,
                "stats": {
                    "deal_size": {"mean": 512.3, "median": 400.0},
                    "ev_ebitda_multiple": {"mean": 11.2, "median": 10.8},
                },
            }
        On no matches, returns match_found: False with an explanatory message
        instead of stats, so an agent can branch on it reliably.
        """
        if isinstance(group_by, str):
            group_by = [group_by]
            group_value = [group_value]
        if isinstance(quantitative_columns, str):
            quantitative_columns = [quantitative_columns]

        if len(group_by) != len(group_value):
            raise ValueError("group_by and group_value must have the same length")

        filters = dict(zip(group_by, group_value))

        mask = pd.Series(True, index=self.df.index)
        for col, val in filters.items():
            if isinstance(val, (list, tuple, set)):
                mask &= self.df[col].isin(val)
            else:
                mask &= self.df[col] == val

        rows = self.df[mask]

        if rows.empty:
            return {
                "match_found": False,
                "filters": filters,
                "message": f"No transactions found for filters {filters}.",
            }

        stats = {}
        for col in quantitative_columns:
            series = pd.to_numeric(rows[col], errors="coerce").dropna()
            if series.empty:
                stats[col] = {"mean": None, "median": None, "note": "no numeric data"}
            else:
                stats[col] = {
                    "mean": round(float(series.mean()), 1),
                    "median": round(float(series.median()), 1),
                }

        return {
            "match_found": True,
            "filters": filters,
            "deal_count": int(len(rows)),
            "stats": stats,
        }

    def get_similar_transactions(self, min_size: float, max_size: float) -> Union[str, list[dict]]:
        """Return transactions within a deal-size range ($MM), to cite as comps."""
        rows = self.df[self.df.deal_size_mm.between(min_size, max_size)]
        if rows.empty:
            return f"No transactions found between ${min_size}MM-${max_size}MM."
        return rows.to_dict(orient="records")


def dispatch_tool_call(comps: CSV_Analytics, tool_name: str, tool_input: dict):
    fn = getattr(comps, tool_name, None)
    if fn is None:
        raise ValueError(f"No such tool: {tool_name}")
    return fn(**tool_input)


# ---------------------------------------------------------------------------
# Explicit args schemas.
#
# StructuredTool.from_function() can auto-infer a schema from type hints, but
# it chokes on Optional[...]/Union[...]/Any/list[Any] signatures (several of
# these methods use exactly that, for the "single value or list of values"
# peer-group filtering pattern). Declaring the schemas explicitly also lets
# us give each *field* its own description, which is what the calling LLM
# actually reads when deciding how to fill in arguments.
# ---------------------------------------------------------------------------

class RelativeStatThresholdsInput(BaseModel):
    column: str = Field(
        ..., description="Numeric column name from the prior-deals CSV to compute percentile breakpoints for, e.g. 'ev_ebitda_multiple' or 'revenue_growth_pct'."
    )
    group_by: Optional[str] = Field(
        None, description="Optional column to restrict the comparison set to a peer group, e.g. 'sector'. Must be paired with group_value."
    )
    group_value: Optional[str] = Field(
        None, description="Value of group_by to filter on, e.g. 'Technology'. If the resulting peer group has fewer deals than the configured minimum, falls back to the full dataset."
    )


class ComputeImpliedMultipleInput(BaseModel):
    deal_size_mm: float = Field(..., description="Enterprise/deal value in $MM (the numerator).")
    metric_mm: float = Field(..., description="Financial metric in $MM to divide into, e.g. target EBITDA or revenue (the denominator).")


class PriorTransactionsInput(BaseModel):
    acquirer: str = Field(..., description="Exact acquirer name to look up prior transactions for, as it appears in the 'acquirer' column of the prior-deals CSV.")


class SubsetTransactionsByPeerGroupInput(BaseModel):
    group_by: Union[str, list[str]] = Field(
        ..., description="One column name, or a list of column names, to filter the prior-deals CSV on, e.g. 'sector' or ['sector', 'region']."
    )
    group_value: Union[Any, list[Any]] = Field(
        ...,
        description=(
            "Value(s) to match against group_by, positionally aligned with it. "
            "For a single group_by column, pass a single value. For multiple group_by "
            "columns, pass a list of the same length (AND across columns). Within any "
            "one column, pass a list/tuple of values to OR them together, e.g. "
            "group_by=['sector','region'], group_value=[['Technology','Healthcare'], 'APAC']."
        ),
    )


class GetSectorBenchmarkInput(BaseModel):
    column: str = Field(..., description="Reserved for future use; the current implementation always reports deal size and EV/EBITDA regardless of this value.")
    sector: str = Field(..., description="Sector name to benchmark, matched exactly against the 'sector' column, e.g. 'Technology'.")


class GetPeerGroupBenchmarkInput(BaseModel):
    group_by: Union[str, list[str]] = Field(
        ..., description="One column name, or a list of column names, defining the peer group, e.g. 'sector' or ['sector', 'region']."
    )
    group_value: Union[Any, list[Any]] = Field(
        ...,
        description=(
            "Value(s) to match against group_by, positionally aligned with it. "
            "Use a list/tuple within a single column's value to OR multiple values "
            "together, e.g. group_by=['sector'], group_value=[['Technology','Healthcare']]."
        ),
    )
    quantitative_columns: Union[str, list[str]] = Field(
        ..., description="One or more numeric column names to summarize with mean/median for the matched peer group, e.g. ['deal_size_mm', 'ev_ebitda_multiple']."
    )


class GetSimilarTransactionsInput(BaseModel):
    min_size: float = Field(..., description="Lower bound of deal size range in $MM, inclusive.")
    max_size: float = Field(..., description="Upper bound of deal size range in $MM, inclusive.")


def make_langchain_tools(comps: CSV_Analytics):
    """Expose CSV_Analytics as LangChain tools while preserving the same tool names and behavior."""
    if StructuredTool is None:
        raise ImportError("Install langchain-core to use the LangChain tool wrappers.")

    return [
        StructuredTool.from_function(
            func=comps.relative_stat_thresholds,
            name="relative_stat_thresholds",
            description=(
                "Get the strong/standard/weak percentile breakpoints (25th/50th/75th) for a "
                "numeric column in the prior-deals CSV, optionally computed within a peer "
                "group such as sector. Use this before classifying a deal's metric as "
                "strong, standard, or weak relative to comparable prior deals."
            ),
            args_schema=RelativeStatThresholdsInput,
        ),
        StructuredTool.from_function(
            func=comps.compute_implied_multiple,
            name="compute_implied_multiple",
            description=(
                "Deterministically compute an implied valuation multiple as EV / metric "
                "(e.g. EV/EBITDA or EV/Revenue) from deal size and a financial metric, both "
                "in $MM. Always use this instead of doing the division yourself."
            ),
            args_schema=ComputeImpliedMultipleInput,
        ),
        StructuredTool.from_function(
            func=comps.prior_transactions,
            name="prior_transactions",
            description=(
                "Look up every prior transaction in the CSV where the given company was "
                "the acquirer. Use this to check an acquirer's deal history or M&A track record."
            ),
            args_schema=PriorTransactionsInput,
        ),
        StructuredTool.from_function(
            func=comps.subset_transactions_by_peer_group,
            name="subset_transactions_by_peer_group",
            description=(
                "Return every prior transaction matching one or more column/value filters "
                "(e.g. sector, region, deal_type), ANDed across columns and ORed within a "
                "column's value list. Use this to pull the full set of comparable deals "
                "before summarizing or citing specific comps."
            ),
            args_schema=SubsetTransactionsByPeerGroupInput,
        ),
        StructuredTool.from_function(
            func=comps.get_sector_benchmark,
            name="get_sector_benchmark",
            description=(
                "Get a quick one-line summary (deal count, median deal size, median "
                "EV/EBITDA) for a single sector. Use for a fast sector-level sanity check; "
                "use get_peer_group_benchmark for multi-column filters or other metrics."
            ),
            args_schema=GetSectorBenchmarkInput,
        ),
        StructuredTool.from_function(
            func=comps.get_peer_group_benchmark,
            name="get_peer_group_benchmark",
            description=(
                "Compute mean/median for one or more numeric columns (e.g. deal_size_mm, "
                "ev_ebitda_multiple, revenue_growth_pct) within a peer group defined by one "
                "or more column/value filters. Returns match_found: False with an "
                "explanatory message if no transactions match, so branch on that field "
                "rather than assuming stats are present."
            ),
            args_schema=GetPeerGroupBenchmarkInput,
        ),
        StructuredTool.from_function(
            func=comps.get_similar_transactions,
            name="get_similar_transactions",
            description=(
                "Return prior transactions whose deal size falls within a given $MM range, "
                "to cite as comparable transactions. Returns an explanatory string instead "
                "of a list if no transactions fall in range."
            ),
            args_schema=GetSimilarTransactionsInput,
        ),
    ]