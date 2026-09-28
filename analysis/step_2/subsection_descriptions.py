acquirer_overview = """
Section: Acquirer Overview
This section provides an overview of the acquirer. It typically includes information about the acquirer's business, strategy, financial capacity, and past transactions.
Make sure to cover: size, strategic priorities, recent M&A activity implied by the dataset 

Output Format: Free text summary of the acquirer's overview.

Content Length: 1 paragraph of 3-5 sentences.
"""

strategic_fit = """
Section: Strategic Fit
This section explains why the target is a good strategic fit for the acquirer. It typically includes an analysis of how the target complements the acquirer's existing business, fills gaps in the acquirer's portfolio, or enhances the acquirer's competitive position.
Make sure to cover: synergies, market positioning, product/service alignment

Output Format: Free text summary of the strategic fit.
Content Length: 1 paragraph of 3-5 sentences.
"""

precedent_activity = """
Section: Precedent Activity
This section reviews precedent transactions relevant to the target and acquirer. It typically includes an analysis of similar deals in the industry, valuation multiples, and deal structures.
Make sure to cover: comparable transactions, valuation benchmarks, deal terms

Output Format: Free text summary of the precedent activity.
Content Length: 1 paragraph of 3-5 sentences.

"""

valuation_context = """
Section: Valuation Context
This section provides the valuation context for the target and acquirer. It typically includes an analysis of the target's financial performance, market conditions, and relevant valuation multiples.
Make sure to cover: financial metrics, market trends, comparable valuations, Relevant EV/EBITDA and EV/Revenue comps from comparable closed deals in the CSV

Output Format: Free text summary of the valuation context.  
Content Length: 1 paragraph of 3-5 sentences.
"""

risk_flags = """
Section: Risk Flags
This section identifies potential risk factors associated with the target and acquirer (At least 2 risks: e.g. antitrust, integration complexity, financing capacity,
competitive process)

Output Format: Free text summary of the risk flags.
Content Length: 1 paragraph of 2-3 sentences.
"""

conviction_level = """
Section: Conviction Level
This section assesses the level of conviction in the assessment of the acquirer being a good choice for M&A activity with the given target.
High / Medium / Low with 1-2 sentence rationale tied to data signals

Output Format: Free text summary of the conviction level.
Content Length: 1 paragraph of 1-2 sentences.
"""