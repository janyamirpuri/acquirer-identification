import argparse
import logging
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from analysis.llm_service import client_instance
from analysis.step_0.field_extractions import DealComps, extract_deal_record
from analysis.step_1.transaction_scoring import acquirer_identification
from analysis.step_2.rationale_building import run_pipeline
from analysis.step_2.tool_definitions import CSV_Analytics
from analysis.step_3.output_formatting import build_html

import streamlit as st
import time

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _save_uploaded_csv(csv_file):
    if csv_file is None:
        raise ValueError("A CSV file is required.")

    folder = Path(__file__).resolve().parent / "external_docs"
    folder.mkdir(exist_ok=True)

    if hasattr(csv_file, "seek"):
        csv_file.seek(0)

    if hasattr(csv_file, "getvalue"):
        data = csv_file.getvalue()
    elif hasattr(csv_file, "read"):
        data = csv_file.read()
    else:
        raise TypeError("Unsupported CSV upload object.")

    if isinstance(data, str):
        payload = data.encode("utf-8")
    else:
        payload = data

    destination = folder / csv_file.name
    logger.info("Saving uploaded CSV to %s", destination)
    with open(destination, "wb") as out:
        out.write(payload)

    return str(destination)

def _load_deal_comps(inputs):
    return {**inputs, "deal_comps": DealComps(inputs["csv"])}


def _load_csv_analytics(inputs):
    return {**inputs, "csv_analytics": CSV_Analytics(inputs["csv"])}


def _extract_target_record(inputs):
    record = extract_deal_record(inputs["description"], inputs["deal_comps"], client=inputs["client"])
    return {**inputs, "target_json": record.model_dump()}


def _score_transactions(inputs):
    scored = acquirer_identification(inputs["deal_comps"].df, inputs["target_json"], client=inputs["client"])
    return {**inputs, "scored_acquirers": scored}


def _generate_rationales(inputs):
    rationales = run_pipeline(
        inputs["scored_acquirers"],
        inputs["description"],
        inputs["csv_analytics"],
    )
    return {**inputs, "rationales": rationales}


def build_pipeline(client=None):
    if client is None:
        client = client_instance()

    return (
        RunnableLambda(lambda x: {"csv": x["csv"], "description": x["description"], "client": client})
        | RunnableLambda(_load_deal_comps)
        | RunnableLambda(_load_csv_analytics)
        | RunnableLambda(_extract_target_record)
        | RunnableLambda(_score_transactions)
        | RunnableLambda(_generate_rationales)
    )


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv", required=True, help="Path to the prior deals CSV")
#     parser.add_argument("--description", required=True, help="Free-text deal description")
#     args = parser.parse_args()

#     time0 = time.time()

#     pipeline = build_pipeline()
#     result = pipeline.invoke({"csv": args.csv, "description": args.description})
#     sub_analyses = [rationale.model_dump() for rationale in result["rationales"]]
#     html_out = build_html(sub_analyses)
#     with open("user_output.html", "w", encoding="utf-8") as f:
#         f.write(html_out)
#     print(f"Wrote {len(sub_analyses)} entries to user_output.html")
#     print(f"Pipeline execution time: {time.time() - time0:.2f} seconds")


# if __name__ == "__main__":
#     main()

def main():
    st.title("Acquirer Analysis Pipeline")

    with st.form("Input File and Target Description"):
        csv_file = st.file_uploader("Upload prior deals CSV", type=["csv"])
        description = st.text_area("Free-text deal description", value="Sector: Healthcare Services | Deal Size: ~$200M EV | Profile: Mid-market, private, regional, strong EBITDA margins")
        submitted = st.form_submit_button("Run Analysis")

    if submitted:
        if csv_file is None:
            st.warning("Please upload a CSV file before running the analysis.")
            return

        time0 = time.time()
        saved_csv_path = _save_uploaded_csv(csv_file)

        st.info("Running analysis...")
        pipeline = build_pipeline()
        result = pipeline.invoke({"csv": saved_csv_path, "description": description})
        sub_analyses = [rationale.model_dump() for rationale in result["rationales"]]
        html_out = build_html(sub_analyses)
        st.success(f"Wrote {len(sub_analyses)} entries to user_output.html")
        st.info(f"Pipeline execution time: {time.time() - time0:.2f} seconds")
        st.download_button("Download Analysis", file_name="user_output.html", data=html_out, mime="text/html")


if __name__ == "__main__":
    main()