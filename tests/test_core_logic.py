import io
from pathlib import Path

import pandas as pd

import app
from analysis.llm_service import chat
from analysis.step_0 import field_extractions
from analysis.step_0.field_extractions import _parse_final_json
from analysis.step_1.transaction_scoring import full_scoring_method
from analysis.step_2 import rationale_building
from analysis.step_2.tool_definitions import CSV_Analytics


def test_parse_final_json_strips_markdown_fence():
    content = '''```json
{"sector": "Healthcare Services", "deal_size_mm": 200.0, "target_ebitda_mm": 40.0}
```'''

    record = _parse_final_json(content)

    assert record.sector == "Healthcare Services"
    assert record.deal_size_mm == 200.0
    assert record.target_ebitda_mm == 40.0


def test_chat_retries_transient_failures():
    attempts = {"count": 0}

    class FakeResponse:
        content = "ok"

    class FakeClient:
        def invoke(self, prompt):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise TimeoutError("temporary LLM timeout")
            return FakeResponse()

    assert chat("hello", client=FakeClient()) == "ok"
    assert attempts["count"] == 2


def test_generate_rationale_uses_agent_executor(monkeypatch):
    executor_options = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            executor_options.update(kwargs)

        def invoke(self, inputs):
            assert inputs == {"target_description": "Target profile", "acquirer": "Apex"}
            return {"output": '{"acquirer_name": "Apex"}'}

    monkeypatch.setattr(rationale_building, "make_langchain_tools", lambda comps: ["tool"])
    monkeypatch.setattr(rationale_building, "create_tool_calling_agent", lambda client, tools, prompt: "agent")
    monkeypatch.setattr(rationale_building, "AgentExecutor", FakeExecutor)

    rationale = rationale_building.generate_rationale(
        "Apex", "Target profile", comps=None, client=object(), max_turns=3
    )

    assert rationale.acquirer_name == "Apex"
    assert executor_options["agent"] == "agent"
    assert executor_options["max_iterations"] == 3


def test_extract_deal_record_uses_agent_executor(monkeypatch):
    executor_options = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            executor_options.update(kwargs)

        def invoke(self, inputs):
            assert inputs == {"description": "Acme, a healthcare company"}
            return {"output": '{"sector": "Healthcare"}'}

    monkeypatch.setattr(field_extractions, "make_langchain_tools", lambda comps: ["tool"])
    monkeypatch.setattr(field_extractions, "create_tool_calling_agent", lambda client, tools, prompt: "agent")
    monkeypatch.setattr(field_extractions, "AgentExecutor", FakeExecutor)

    record = field_extractions.extract_deal_record(
        "Acme, a healthcare company", comps=None, client=object(), max_turns=2
    )

    assert record.sector == "Healthcare"
    assert executor_options["agent"] == "agent"
    assert executor_options["max_iterations"] == 2


def test_full_scoring_method_returns_similarity_scores(monkeypatch):
    monkeypatch.setattr(
        "analysis.step_1.transaction_scoring.score_unique_values",
        lambda unique_terms, target_term, client=None, max_retries=1: {term: 1.0 for term in unique_terms},
    )

    df = pd.DataFrame(
        {
            "acquirer": ["Apex", "Apex", "NorthStar", "NorthStar"],
            "sector": ["Healthcare", "Healthcare", "Healthcare", "Healthcare"],
            "sub_sector": ["Services", "Services", "Services", "Services"],
            "geography": ["US", "US", "US", "US"],
            "target_ownership_pre": ["Private", "Private", "Private", "Private"],
            "deal_size_mm": [80.0, 100.0, 120.0, 140.0],
            "target_revenue_mm": [60.0, 70.0, 80.0, 90.0],
            "target_ebitda_mm": [15.0, 20.0, 25.0, 30.0],
            "ev_ebitda_multiple": [4.0, 5.0, 6.0, 7.0],
            "ev_revenue_multiple": [1.0, 1.25, 1.5, 1.75],
            "ebitda_margin_pct": [18.0, 25.0, 30.0, 35.0],
            "revenue_growth_pct": [8.0, 10.0, 12.0, 15.0],
        }
    )

    target_info = {
        "deal_size_mm": 100.0,
        "target_revenue_mm": 70.0,
        "target_ebitda_mm": 20.0,
        "ev_ebitda_multiple": 5.0,
        "ev_revenue_multiple": 1.25,
        "ebitda_margin_pct": 25.0,
        "revenue_growth_pct": 10.0,
        "sector": "Healthcare",
        "sub_sector": "Services",
        "geography": "US",
        "target_ownership_pre": "Private",
    }

    scored = full_scoring_method(df, target_info)

    assert "similarity_score" in scored.columns
    assert scored["similarity_score"].notna().all()
    assert scored["similarity_score"].between(0.0, 1.0).all()
    assert scored.loc[1, "similarity_score"] == scored["similarity_score"].max()


def test_csv_analytics_relative_thresholds_and_peer_benchmark(tmp_path):
    csv_path = tmp_path / "sample_deals.csv"
    pd.DataFrame(
        {
            "acquirer": [f"Acquirer {i}" for i in range(1, 21)],
            "sector": ["Healthcare"] * 20,
            "region": ["US"] * 20,
            "deal_size_mm": list(range(50, 70)),
            "ev_ebitda_multiple": [6.0, 7.0, 8.0, 9.0, 10.0] * 4,
        }
    ).to_csv(csv_path, index=False)

    analytics = CSV_Analytics(str(csv_path))
    thresholds = analytics.relative_stat_thresholds("deal_size_mm")

    assert thresholds["column"] == "deal_size_mm"
    assert thresholds["p25"] == 54.75
    assert thresholds["p50"] == 59.5
    assert thresholds["p75"] == 64.25
    assert thresholds["n_deals"] == 20

    benchmark = analytics.get_peer_group_benchmark("sector", "Healthcare", ["deal_size_mm", "ev_ebitda_multiple"])

    assert benchmark["match_found"] is True
    assert benchmark["deal_count"] == 20
    assert "stats" in benchmark
    assert "deal_size_mm" in benchmark["stats"]
    assert "ev_ebitda_multiple" in benchmark["stats"]


def test_build_html_formats_acquirer_sections():
    objects = [
        {
            "acquirer_name": "Apex Health",
            "strategic_fit": "Strong healthcare adjacency and scale",
            "risk_flags": "Integration risk remains moderate",
        }
    ]

    html = app.build_html(objects)

    assert "Apex Health" in html
    assert "Strategic Fit" in html
    assert "Risk Flags" in html
    assert "Strong healthcare adjacency and scale" in html


def test_save_uploaded_csv_writes_file(monkeypatch, tmp_path):
    repo_root = tmp_path / "project"
    docs_dir = repo_root / "external_docs"
    docs_dir.mkdir(parents=True)
    monkeypatch.setattr(app, "__file__", str(repo_root / "app.py"))

    payload = b"deal_id,acquirer\n1,Example\n"
    upload = io.BytesIO(payload)
    upload.name = "sample.csv"

    saved_path = app._save_uploaded_csv(upload)

    assert Path(saved_path).exists()
    assert Path(saved_path).read_bytes() == payload
