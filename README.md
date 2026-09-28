# acquirer-identification

## Demo Link Due To Paid API

https://www.loom.com/share/89cfe1f5aea844c9817dcb04645f4de1

## How to Run the Prototype

In a bash terminal:

```bash
# initialize a virtual environment
python -m venv .venv

# activate the venv
source .venv/Scripts/activate

# install requirements
pip install -r requirements.txt
```

Add a `.env` file and set:

- `OPENAI_MODEL`
- `OPENAI_API_KEY`

Then run:

```bash
streamlit run app.py
```

Then, upload a CSV and optionally change the target description.

---

## Methodology + Architecture

This is a multi-step process:

### Step 0

Given a target description, fit the data into the available target relevent columns.

```text
sector
sub_sector
deal_size_mm
target_revenue_mm
target_ebitda_mm
ev_ebitda_multiple
ev_revenue_multiple
ebitda_margin_pct
revenue_growth_pct
geography
target_ownership_pre
```

This uses an LLM supplemented with two tools to go from textual description to fields. See `analysis/step_0/tool_definitions`.
The tools are meant to help with calculations for multiples and to help identify theoretical values for statements of relative descriptions (weak, strong, etc.).

### Step 1

Based on the identified fields from the target description, calculate a similarity score for each target in the transaction.

Scored by Abs Log Normalized:

```text
deal_size_mm
target_revenue_mm
target_ebitda_mm
```

Scored by Abs Normalized:

```text
ev_ebitda_multiple
ev_revenue_multiple
ebitda_margin_pct
revenue_growth_pct
```

Scored by LLM on a rank of {0, 0.5, 1}:

```text
sector
sub_sector
geography
target_ownership_pre
```

Note that a score is only made for the columns that are actually available from the target description.

Then, the individual column scores are brought into one score per transaction - a weighted mean. 0.4 for the available categorical (i.e. LLM scored) columns, 0.6 for the rest.

From per-transaction scores, we want acquirer scores. I calculated Bayesian shrinkage with k = 7 so that I could try to place more value on acquirers with more transactions.

Retrieved the acquirers of top 10 scores.

### Step 2

For each of the acquirers identified in the previous step, I generate rationales.

This uses an LLM supplemented with tools (in addition to those earlier, some on obtaining subsets of the dataframes and calculating mean/median of quantitative columns), and run concurrently multiple rationales.  See `analysis/step_2/tool_definitions` for tools.

### Step 3

Format output into HTML.

---

## Assumptions

- I assumed the provided dataset would always follow the same format with the same columns and that there would be no null values, ie fully populated.
- I assumed the target description would have no conflicting values. I assumed that relative descriptors would be strong, weak, etc.

---

## Output Non-Determinism

1. Parameters of OpenAI calls: Set seed = 42, temperature = 0, low reasoning (to be addressed below)
2. Step 0/1: By creating a relatively consistent scoring method for acquirers, there is low variation in the output of 10 acquirers that are chosen. Relatively because there is still an LLM involved in parsing the text description and scoring the categorical match. Most of the variation would be observed in the outputs of step 2 (i.e. the rationale)

---

## How I Would Improve It

1. Spend more time on Prompt Engineering. Right now they cover the bare minimum but I could have provided more contextual knowledge which would lead to more consistent outputs, and also that would help with the quality of rationale generated. Right now the rationale for the acquirers chosen for my methodology dock pretty heavily on mismatching sectors, which I need to reflect on if it is a bad or good thing, and I would need to adjust the prompts accordingly.

2. In alignment with the statement on mismatching sectors. I might take another look at the similarity score methodology. Again, need to reflect on the sector mismatch - right now I have the score set to care more about a reliability of an acquirer (so more transactions) than the actual similarity match (set by the value k). But I may want to try some other reliability scores instead of the bayesian shrinkage, I want to try different values for k, and then I want to check different weights for the categorical vs numerical similarity score (which was per transaction)

3. I would've liked to use a higher level of reasoning, but it was causing the time taken to be over 60 seconds. Similarly, I would've adjusted the parameters for the OpenAI call based on more testing. I would've kept separate parameters for step 0 and step 2. Step 0 is data extraction, reasoning doesn't need to be high. Step 2 could use a higher level of reasoning effort.
