from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from ..llm_service import client_instance
from .tool_definitions import CSV_Analytics, Rationale, dispatch_tool_call, make_langchain_tools


dotenv.load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an M&A analyst writing a one-page acquirer rationale for a banker.
You will be given a target company profile and a candidate acquirer. 

The goal is not to provide generic commentary or make unsupported claims. When considering potential M&A activity, don't place all value on a single metric or assumption.
Matching in sector exactly is not the primary determinant of a good strategic fit; consider other factors such as sub-sector, geography, and financial metrics.

Fit your analysis into
this exact JSON schema:

{schema}

Rules:
- Use the provided tools to ground every claim in real data. Never guess a multiple, a deal
count, or a percentile-based label (strong/standard/weak) -- call the matching tool instead.
- Cite specific numbers (deal counts, multiples, deal sizes) pulled from the tools wherever
a section makes a claim.
- Never write generic filler like "leading company with a strong track record" -- if you
can't back a claim with a tool result, leave it out or note the gap explicitly.
- conviction_level should follow from what you wrote in strategic_fit, precedent_activity,
valuation_context, and risk_flags -- don't decide it in isolation before writing those.
- Leave a field null if there truly isn't enough information to support it.

- Once you have everything you need, respond with ONLY the final JSON object: no prose, no
markdown fences, and no further tool calls in that message.
"""


def _tool_result_message(call, comps: CSV_Analytics) -> ToolMessage:
    name = call.get("name")
    args = call.get("args") or {}
    try:
        result = dispatch_tool_call(comps, name, args)
    except Exception as exc:  # bad args, unknown tool, etc. -- feed the error back to the model
        result = {"error": str(exc)}

    return ToolMessage(
        content=json.dumps(result, default=str),
        tool_call_id=call.get("id"),
    )


def generate_rationale(
    acquirer: str,
    target_description: str,
    comps: CSV_Analytics,
    client=None,
    max_turns: int = 6,
) -> Rationale:
    logger.info("Generating rationale for acquirer=%s", acquirer)
    if client is None:
        client = client_instance()

    tools = make_langchain_tools(comps)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(schema=Rationale.model_json_schema())),
        HumanMessage(
            content=f"Target profile: {target_description}\n\nCandidate acquirer: {acquirer}"
        ),
    ]

    model = client.bind_tools(tools) if tools else client

    for turn in range(max_turns):
        logger.info("Rationale generation turn %s/%s for %s", turn + 1, max_turns, acquirer)
        try:
            response = model.invoke(messages)
        except Exception as exc:
            logger.exception("Rationale generation LLM call failed for %s during turn %s/%s", acquirer, turn + 1, max_turns)
            raise RuntimeError(f"LLM call failed while generating rationale for {acquirer}: {exc}") from exc

        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            logger.info("Rationale for %s completed without additional tool calls", acquirer)
            try:
                return _parse_final_json(response.content)
            except ValueError as exc:
                logger.exception("Final JSON from rationale generation was invalid for %s", acquirer)
                raise RuntimeError(f"Failed to parse final rationale JSON for {acquirer}: {exc}") from exc

        logger.info("Rationale for %s requested %s tool calls", acquirer, len(tool_calls))
        messages.extend(_tool_result_message(call, comps) for call in tool_calls)

    raise RuntimeError(f"Agent did not converge on a rationale within {max_turns} turns")


def _parse_final_json(content: str) -> Rationale:
    """Strip common formatting quirks (markdown fences, a leading 'json' tag) and validate."""
    cleaned = (content or "").strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON:\n{content}") from e

    try:
        return Rationale.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Model JSON didn't match the schema:\n{e}\n\nRaw: {data}") from e


def run_pipeline(
    acquirers: list[str],
    target_description: str,
    comps: CSV_Analytics,
    max_workers: int = 5,
) -> list[Rationale]:
    logger.info("Starting rationale pipeline for %s acquirers with max_workers=%s", len(acquirers), max_workers)
    client = client_instance()  # one client, shared across threads

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(generate_rationale, a, target_description, comps, client)
            for a in acquirers
        ]
        results = [f.result() for f in futures]
        logger.info("Completed rationale pipeline for %s acquirers", len(results))
        return results
