import argparse
from analysis.step_0.agentic_deal_extractor import DealComps, extract_deal_record
from analysis.step_1.transaction_scoring import acquirer_identification


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the prior deals CSV")
    parser.add_argument("--description", required=True, help="Free-text deal description")
    args = parser.parse_args()

    comps = DealComps(args.csv)
    record = extract_deal_record(args.description, comps)
    target_json = record.model_dump()
    print(target_json)
    scored_transactions = acquirer_identification(comps.df, target_json)
    print(scored_transactions)