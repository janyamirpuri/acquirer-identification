import pandas as pd
import numpy as np
import time
import json
import re
try:
    from llm_service import chat, client_instance
except ImportError:
    from ...llm_service import chat, client_instance

#'sector', 'sub_sector', 'deal_size_mm', 'deal_type', 'target_revenue_mm', 'target_ebitda_mm', 'ev_ebitda_multiple', 'ev_revenue_multiple', 'ebitda_margin_pct', 'revenue_growth_pct', 'geography', 'target_ownership_pre'

#Quantitative Comparisons
# 
# 

def log_based_scoring(transactions_df: pd.DataFrame, target_info: dict) -> pd.DataFrame:
    """
    Perform log-based scoring of transactions against the target information.

    Args:
        transactions_df (pd.DataFrame): DataFrame containing prior transactions.
        target_info (dict): Dictionary containing target deal information.

    Returns:
        pd.DataFrame: DataFrame with an additional column for log-based scores.
    """
    scored_df = transactions_df.copy()
    # Example scoring logic (replace with actual implementation)
    for column in ['deal_size_mm', 'target_revenue_mm', 'target_ebitda_mm']:
        if column in scored_df.columns and target_info[column] is not None:
            scored_df[f'log_{column}'] = np.log10(scored_df[[column]])
            log_target = np.log10(target_info[column])
            scored_df[f'diff_log_{column}'] = np.abs(scored_df[f'log_{column}'] - log_target)
            scored_df[f'score_diff_log_{column}'] = (scored_df[f'diff_log_{column}'] - scored_df[f'diff_log_{column}'].min()) / (scored_df[f'diff_log_{column}'].max() - scored_df[f'diff_log_{column}'].min())
    return scored_df

def abs_normalized_scoring(transactions_df: pd.DataFrame, target_info: dict) -> pd.DataFrame:
    scored_df = transactions_df.copy()
    for column in ['ev_ebitda_multiple', 'ev_revenue_multiple', 'ebitda_margin_pct', 'revenue_growth_pct']:
        if column in scored_df.columns and target_info[column] is not None:
            scored_df[f'abs_distance_{column}'] = np.abs(scored_df[column] - target_info[column])
            scored_df[f'score_abs_normalized_{column}'] = (scored_df[f'abs_distance_{column}'] - scored_df[f'abs_distance_{column}'].min()) / (scored_df[f'abs_distance_{column}'].max() - scored_df[f'abs_distance_{column}'].min())
    return scored_df

UNIQUE_VALUE_SCORING_PROMPT = """You are a strict terminology similarity grader. You will score the similarity between ONE target term and a LIST of distinct comparison terms, one score per comparison term.

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


def _parse_score_dict(response_text: str, expected_terms: list) -> dict:
    """Strip any markdown fencing and parse the model's JSON dict of term -> score."""
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    scores = json.loads(cleaned)

    if not isinstance(scores, dict):
        raise ValueError(f"Expected a JSON object, got: {scores!r}")

    valid = {0, 0.5, 1}
    out = {}
    for term in expected_terms:
        if term not in scores:
            raise ValueError(f"Missing score for term: {term!r}")
        f = float(scores[term])
        if f not in valid:
            raise ValueError(f"Score {f} for {term!r} is not one of {valid}")
        out[term] = f
    return out


def score_unique_values(unique_terms: list, target_term: str, client=None, max_retries: int = 1) -> dict:
    """Score every distinct term in `unique_terms` against one target term in a single call.

    Returns a dict mapping each input term -> score (0, 0.5, or 1).
    """
    term_list = "\n".join(f"- {t}" for t in unique_terms)
    formatted_prompt = UNIQUE_VALUE_SCORING_PROMPT.format(term_2=target_term, term_list=term_list)

    attempt = 0
    last_err = None
    while attempt <= max_retries:
        response = chat(formatted_prompt, client=client)
        try:
            return _parse_score_dict(response, unique_terms)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            attempt += 1

    # If parsing still fails after retries, fall back to NaNs rather than crashing the whole run
    print(f"[score_unique_values] failed to parse response after {max_retries + 1} attempt(s): {last_err}")
    return {t: float("nan") for t in unique_terms}


def categorical_scoring(
    transactions_df: pd.DataFrame,
    target_info: dict,
    client=None,
) -> pd.DataFrame:
    """
    Scores each categorical column by its unique values only (one LLM call per column),
    then maps those scores back onto every row via the value.
    """
    if client is None:
        client = client_instance()

    scored_df = transactions_df.copy()

    for column in ["sector", "sub_sector", "geography", "target_ownership_pre"]:
        if column not in scored_df.columns or target_info.get(column) is None:
            continue

        target_term = target_info[column]
        unique_terms = scored_df[column].dropna().unique().tolist()

        score_map = score_unique_values(unique_terms, target_term, client=client)
        scored_df[f"categorical_score_{column}"] = scored_df[column].map(score_map)

    return scored_df

def full_scoring_method(transactions_df: pd.DataFrame, target_info: dict) -> pd.DataFrame:
    original = transactions_df.columns
    scored_df = categorical_scoring(transactions_df, target_info)
    scored_df = abs_normalized_scoring(scored_df, target_info)
    scored_df = log_based_scoring(scored_df, target_info)
    main_cols = [x for x in scored_df.columns if x[:5] == 'score']
    scored_df["final_score"] = scored_df[main_cols].mean(axis=1)
    return scored_df[original.tolist() + ["final_score"]]

def acquirer_identification(transactions_df: pd.DataFrame, target_info: dict) -> pd.DataFrame:
    scored_df = full_scoring_method(transactions_df, target_info)
    k = 7
    global_mean = scored_df["final_score"].mean()
    grouped = scored_df.groupby("acquirer")["final_score"].agg(["mean", "count"]).rename(
        columns={"mean": "avg_score", "count": "n_deals"}
    )
    print(grouped["n_deals"].mean())
    grouped["reliability_score"] = (
        grouped["n_deals"] * grouped["avg_score"] + k * global_mean
    ) / (grouped["n_deals"] + k)
    return grouped.sort_values("reliability_score", ascending=False).reset_index()