"""
agentic_deal_extractor.py

The agent loop: takes a free-text deal description and drives an Ollama
model (through the OpenAI-compatible client) through the tools defined in
deal_tagging_tools.py until it produces a final JSON DealRecord.

Setup:
    pip install openai pydantic pandas
    ollama pull llama3.1          # or qwen2.5, mistral-nemo, firefunction-v2 --
                                   # any Ollama model that supports tool calling
    ollama serve                  # if not already running

Usage:
    python agentic_deal_extractor.py \
        --csv external_docs/ma_transactions_500.csv \
        --description "Healthcare Services | Deal Size: ~$200M EV | Profile: Mid-market, private, regional, strong EBITDA margins"


"""

from __future__ import annotations
import json
import argparse
import sys
from pathlib import Path
from pydantic import ValidationError
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from llm_service import build_tool_call_chain

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from deal_tagging_tools import DealComps, DealRecord, TOOL_SCHEMAS, dispatch_tool_call, make_langchain_tools
except ImportError:
    from .deal_tagging_tools import DealComps, DealRecord, TOOL_SCHEMAS, dispatch_tool_call, make_langchain_tools

try:
    from llm_service import client_instance
except ImportError:
    from ...llm_service import client_instance

dotenv.load_dotenv()

SYSTEM_PROMPT = """You are an M&A deal-tagging analyst. You will be given a free-text
description of a target company / deal. Fit it into this exact JSON schema:

{schema}

Rules:
- Use the provided tools to ground every relative or derived judgment. Never guess a
strong/standard/weak label, a category name, or a multiple -- call the matching tool instead.
- For sector, sub_sector, deal_type, geography and acquirer_type, category names do not need to
match exactly any reference list; just use the names as given in the description.
- If the description gives you enough of deal_size_mm / target_ebitda_mm / target_revenue_mm
to back out a multiple, call `compute_implied_multiple` rather than dividing yourself.
- If the description uses a relative term (e.g. "strong EBITDA margins") without a number,
call `relative_stat_thresholds` (grouped by sector where relevant) to find the number range
that word maps to in this market, then pick a value inside that range for the field.
- For labels, assume that M or MM refers to millions and adjust the numeric values accordingly.
- Note that deal size is akin to enterprise value (EV) and should be treated as such.
- For general qualitative information, capture any relevant context or details provided
in the description that is not already covered by the other fields.
- Leave a field null if there truly isn't enough information to estimate it -- do not invent    
a number with no basis.
- Once you have everything you need, respond with ONLY the final JSON object: no prose, no
markdown fences, and no further tool calls in that message.
"""


def extract_deal_record(
    description: str,
    comps: DealComps,
    client=None,
    max_turns: int = 8,
) -> DealRecord:
    if client is None:
        client = client_instance()

    tools = make_langchain_tools(comps)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(schema=DealRecord.model_json_schema())),
        HumanMessage(content=description),
    ]

    model = build_tool_call_chain(client, tools)

    for turn in range(max_turns):
        response = model.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            return _parse_final_json(response.content)

        for call in response.tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            try:
                result = dispatch_tool_call(comps, name, args)
            except Exception as e:  # bad args, unknown tool, etc. -- feed the error back to the model
                result = {"error": str(e)}

            messages.append(
                ToolMessage(
                    content=json.dumps(result, default=str),
                    tool_call_id=call.get("id"),
                )
            )

    raise RuntimeError(f"Agent did not converge on a final answer within {max_turns} turns")


def _parse_final_json(content: str) -> DealRecord:
    """Strip common formatting quirks (markdown fences, a leading 'json' tag) and validate."""
    cleaned = (content or "").strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON:\n{content}") from e

    try:
        return DealRecord.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Model JSON didn't match the schema:\n{e}\n\nRaw: {data}") from e

