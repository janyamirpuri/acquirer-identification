import argparse

from langchain_core.runnables import RunnableLambda

from analysis.llm_service import client_instance
from analysis.step_0.field_extractions import DealComps, extract_deal_record
from analysis.step_1.transaction_scoring import acquirer_identification


def _load_comps(inputs):
    return {**inputs, "comps": DealComps(inputs["csv"])}


def _extract_target_record(inputs):
    record = extract_deal_record(inputs["description"], inputs["comps"], client=inputs["client"])
    return {**inputs, "target_json": record.model_dump()}


def _score_transactions(inputs):
    return acquirer_identification(inputs["comps"].df, inputs["target_json"], client=inputs["client"])


def build_pipeline(client=None):
    if client is None:
        client = client_instance()

    return (
        RunnableLambda(lambda x: {"csv": x["csv"], "description": x["description"], "client": client})
        | RunnableLambda(_load_comps)
        | RunnableLambda(_extract_target_record)
        | RunnableLambda(_score_transactions)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the prior deals CSV")
    parser.add_argument("--description", required=True, help="Free-text deal description")
    args = parser.parse_args()

    pipeline = build_pipeline()
    scored_transactions = pipeline.invoke({"csv": args.csv, "description": args.description})
    print(scored_transactions)


if __name__ == "__main__":
    main()