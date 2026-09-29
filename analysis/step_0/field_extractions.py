from __future__ import annotations

import json
import logging

import dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ValidationError

from ..llm_service import client_instance
from .tool_definitions import DealComps, DealRecord, make_langchain_tools




dotenv.load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an M&A deal-tagging analyst. You will be given a free-text
description of a target company / deal. Fit it into this exact JSON schema:

{schema}

Rules:
- Use the provided tools to ground every relative or derived judgment. Never guess a
strong/standard/weak label, a category name, or a multiple -- call the matching tool instead.
- For sector, sub_sector, geography, category names do not need to
match exactly any reference list; just use the names as given in the description.
- If the description gives you enough of deal_size_mm / target_ebitda_mm / target_revenue_mm
to back out a multiple, call `compute_implied_multiple` rather than dividing yourself.
- If the description uses a relative term (e.g. "strong EBITDA margins") without a number,
call `relative_stat_thresholds` (grouped by sector where relevant) to find the number range
that word maps to in this market, then pick a value inside that range for the field.
- For labels, assume that M or MM refers to millions and adjust the numeric values accordingly.
- Note that deal size is akin to enterprise value (EV) and should be treated as such.
- the term 'market' refers to company revenue
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
    max_turns: int = 4,
) -> DealRecord:
    logger.info("Starting deal extraction for description length=%s", len(description))
    if client is None:
        client = client_instance()

    tools = make_langchain_tools(comps)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(schema=DealRecord.model_json_schema())),
            ("human", "{description}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(client, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=max_turns,
        early_stopping_method="force",
    )

    logger.info("Executing deal extraction agent with max_iterations=%s", max_turns)
    try:
        result = executor.invoke({"description": description})
    except Exception as exc:
        logger.exception("Deal extraction agent execution failed")
        raise RuntimeError(f"LLM call failed while extracting deal record: {exc}") from exc

    try:
        info = _parse_final_json(result["output"])
        logger.info(info)
        return info
    except (KeyError, ValueError) as exc:
        logger.exception("Final JSON from deal extraction was invalid")
        raise RuntimeError(f"Failed to parse final deal extraction JSON: {exc}") from exc


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

