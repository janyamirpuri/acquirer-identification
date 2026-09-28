import json
import logging
import re

import numpy as np
import pandas as pd

from analysis.llm_service import chat, client_instance

logger = logging.getLogger(__name__)

NUMERIC_SCORE_COLUMNS = [
    "deal_size_mm",
    "target_revenue_mm",
    "target_ebitda_mm",
]

ABSOLUTE_DISTANCE_COLUMNS = [
    "ev_ebitda_multiple",
    "ev_revenue_multiple",
    "ebitda_margin_pct",
    "revenue_growth_pct",
]

CATEGORICAL_SCORE_COLUMNS = [
    "sector",
    "sub_sector",
    "geography",
    "target_ownership_pre",
]

VALID_SIMILARITY_SCORES = {0, 0.5, 1}
RELIABILITY_WEIGHT = 7

UNIQUE_VALUE_SCORING_PROMPT = """
You are a strict terminology similarity grader. You will score the similarity between ONE target term and a LIST of distinct comparison terms, one score per comparison term.

Scoring rubric (apply to each pair independently):
- 1   -> The terms are exact matches, true synonyms, or refer to the same underlying concept (minor formatting differences, abbreviations, or reordered words don't count against this). Example: "Healthcare" vs "Health Care" -> 1. "TMT" vs "Technology, Media & Telecom" -> 1.
- 0.5 -> The terms are related or overlapping but not equivalent - one is a broader/narrower category of the other, they share a parent category, or they're commonly confused/adjacent concepts. Example: "Fintech" vs "Financial Services" -> 0.5. "SaaS" vs "Enterprise Software" -> 0.5.
- 0   -> The terms are unrelated or refer to distinct concepts with no meaningful overlap. Example: "Healthcare" vs "Industrials" -> 0.

Instructions:
1. Normalize each term mentally (case, punctuation, whitespace) before comparing meaning - don't penalize purely cosmetic differences.
2. Base your judgment on semantic/domain meaning, not surface string similarity (e.g., "Bank" and "Tank" look similar but are unrelated -> 0).
3. When uncertain between two adjacent scores, choose the lower one.
4. Score every comparison term independently against the target term - do not let one pair's score influence another's.
5. Return ONLY a JSON object mapping each comparison term (as a string key, copied EXACTLY as given) to its score (0, 0.5, or 1). No explanation, no markdown, no code fences - just the object, e.g. {{"Fintech": 0.5, "Healthcare": 0}}.

Target term: {term_2}

Comparison terms (score every one, using each exactly as written as the key):
{term_list}

JSON object mapping each comparison term to its score:
"""


def _normalize_score_column(scored_df: pd.DataFrame, source_column: str, score_name: str) -> pd.DataFrame:
    """Compute a normalized score from a feature distance column."""
    distance_col = f"distance_{source_column}"
    score_col = f"score_{score_name}"

    scored_df[distance_col] = np.abs(scored_df[source_column] - scored_df[source_column].min())
    scored_df[score_col] = (
        scored_df[distance_col] - scored_df[distance_col].min()
    ) / (scored_df[distance_col].max() - scored_df[distance_col].min())
    return scored_df


def _score_log_distance_feature(transactions_df: pd.DataFrame, target_info: dict) -> pd.DataFrame:
    """Score numeric features by log-distance from the target value."""
    scored_df = transactions_df.copy()

    for column in NUMERIC_SCORE_COLUMNS:
        if column not in scored_df.columns or target_info.get(column) is None:
            continue

        log_values = np.log10(scored_df[column].astype(float))
        log_target = np.log10(float(target_info[column]))
        distance_col = f"log_distance_{column}"
        score_col = f"score_log_{column}"

        scored_df[distance_col] = np.abs(log_values - log_target)
        min_distance = scored_df[distance_col].min()
        max_distance = scored_df[distance_col].max()
        scored_df[score_col] = (scored_df[distance_col] - min_distance) / (max_distance - min_distance)

    return scored_df


def _score_absolute_distance_feature(transactions_df: pd.DataFrame, target_info: dict) -> pd.DataFrame:
    """Score absolute-magnitude features after normalizing distance from the target."""
    scored_df = transactions_df.copy()

    for column in ABSOLUTE_DISTANCE_COLUMNS:
        if column not in scored_df.columns or target_info.get(column) is None:
            continue

        scored_df[f"distance_{column}"] = np.abs(scored_df[column] - target_info[column])
        min_distance = scored_df[f"distance_{column}"].min()
        max_distance = scored_df[f"distance_{column}"].max()
        scored_df[f"score_abs_{column}"] = (
            scored_df[f"distance_{column}"] - min_distance
        ) / (max_distance - min_distance)

    return scored_df


def _parse_score_dict(response_text: str, expected_terms: list) -> dict:
    """Strip markdown fences and parse the model's JSON dict of term -> score."""
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    scores = json.loads(cleaned)

    if not isinstance(scores, dict):
        raise ValueError(f"Expected a JSON object, got: {scores!r}")

    parsed = {}
    for term in expected_terms:
        if term not in scores:
            raise ValueError(f"Missing score for term: {term!r}")

        value = float(scores[term])
        if value not in VALID_SIMILARITY_SCORES:
            raise ValueError(f"Score {value} for {term!r} is not one of {sorted(VALID_SIMILARITY_SCORES)}")
        parsed[term] = value

    return parsed


def score_unique_values(unique_terms: list, target_term: str, client=None, max_retries: int = 1) -> dict:
    """Ask an LLM to score the semantic similarity of each category term to the target."""
    term_list = "\n".join(f"- {term}" for term in unique_terms)
    formatted_prompt = UNIQUE_VALUE_SCORING_PROMPT.format(term_2=target_term, term_list=term_list)

    last_error = None
    for attempt in range(max_retries + 1):
        response = chat(formatted_prompt, client=client)
        try:
            return _parse_score_dict(response, unique_terms)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    print(f"[score_unique_values] failed to parse response after {max_retries + 1} attempt(s): {last_error}")
    return {term: float("nan") for term in unique_terms}


def categorical_scoring(transactions_df: pd.DataFrame, target_info: dict, client=None) -> pd.DataFrame:
    """Score each categorical field by semantic similarity to the target description."""
    if client is None:
        client = client_instance()

    scored_df = transactions_df.copy()
    logger.info("Scoring %s categorical columns against target", len(CATEGORICAL_SCORE_COLUMNS))

    for column in CATEGORICAL_SCORE_COLUMNS:
        if column not in scored_df.columns or target_info.get(column) is None:
            continue

        unique_terms = scored_df[column].dropna().unique().tolist()
        logger.info("Scoring categorical field '%s' with %s unique values", column, len(unique_terms))
        score_map = score_unique_values(unique_terms, target_info[column], client=client)
        scored_df[f"categorical_score_{column}"] = scored_df[column].map(score_map)

    return scored_df


def full_scoring_method(transactions_df: pd.DataFrame, target_info: dict, client=None) -> pd.DataFrame:

    original_columns = transactions_df.columns
    scored_df = categorical_scoring(transactions_df, target_info, client)
    scored_df = _score_absolute_distance_feature(scored_df, target_info)
    scored_df = _score_log_distance_feature(scored_df, target_info)

    categorical_columns = [
        f"categorical_score_{column}"
        for column in CATEGORICAL_SCORE_COLUMNS
        if f"categorical_score_{column}" in scored_df.columns
    ]
    non_categorical_columns = [
        column for column in scored_df.columns
        if column.startswith("score") and column not in categorical_columns
    ]

    if categorical_columns and non_categorical_columns:
        scored_df["similarity_score"] = (
            0.4 * scored_df[categorical_columns].mean(axis=1)
            + 0.6 * scored_df[non_categorical_columns].mean(axis=1)
        )
    elif categorical_columns:
        scored_df["similarity_score"] = scored_df[categorical_columns].mean(axis=1)
    elif non_categorical_columns:
        scored_df["similarity_score"] = scored_df[non_categorical_columns].mean(axis=1)
    else:
        scored_df["similarity_score"] = 0.0

    return scored_df[original_columns.tolist() + ["similarity_score"]]


def acquirer_identification(transactions_df: pd.DataFrame, target_info: dict, client=None) -> pd.DataFrame:
    """Rank acquirers by weighted similarity of their historical deals to the target deal."""
    scored_df = full_scoring_method(transactions_df, target_info, client)
    global_mean = scored_df["similarity_score"].mean()

    grouped = (
        scored_df.groupby("acquirer")["similarity_score"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_score", "count": "n_deals"})
    )

    grouped["reliability_score"] = (
        grouped["n_deals"] * grouped["avg_score"] + RELIABILITY_WEIGHT * global_mean
    ) / (grouped["n_deals"] + RELIABILITY_WEIGHT)

    return grouped.sort_values("reliability_score", ascending=False).reset_index()[:10]["acquirer"].tolist()
