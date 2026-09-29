import html
import json
import logging

logger = logging.getLogger(__name__)

FIELD_LABELS = {
    "acquirer_overview": "Acquirer Overview",
    "strategic_fit": "Strategic Fit",
    "precedent_activity": "Precedent Activity",
    "valuation_context": "Valuation Context",
    "risk_flags": "Risk Flags",
    "conviction_level": "Conviction Level",
}
# acquirer_name is used for the card heading, not printed again as a section
SKIP_FIELDS = {"acquirer_name"}

def build_html(objects):
    logger.info("Building HTML output for %s acquirer summaries", len(objects))
    cards = []
    for obj in objects:
        name = obj.get("acquirer_name", "Unknown Acquirer")
        sections = "".join(
            f'<h3>{html.escape(FIELD_LABELS.get(k, k))}</h3><p>{html.escape(v)}</p>'
            for k, v in obj.items()
            if k not in SKIP_FIELDS
        )
        cards.append(f'<div class="card"><h2>{html.escape(name)}</h2>{sections}</div>')

    logger.info("Finished HTML generation for %s acquirer summaries", len(cards))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Acquirer Analysis</title>
<style>
  body {{ font-family: "Times New Roman", Times, serif; color: #000; background: #f5f6f8; margin: 0; padding: 40px; }}
  h1 {{ text-align: center; color: #000; }}
  .card {{ background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);
           padding: 24px 30px; margin: 0 auto 24px; max-width: 800px; color: #000; }}
  .card h2 {{ color: #000; border-bottom: 2px solid #000; padding-bottom: 8px; }}
  .card h3 {{ color: #000; margin-bottom: 4px; margin-top: 18px; }}
  .card p {{ line-height: 1.55; color: #000; margin-top: 0; }}
</style>
</head>
<body>
<h1>Acquirer Analysis Summary</h1>
{''.join(cards)}
</body>
</html>"""
