# Dataset Explorer

A Streamlit app for exploring a CSV file: dataset preview, column summary,
data quality checks, descriptive statistics, correlations, distributions,
and a pandas assistant for ad-hoc questions in plain language.

## Features

- **Dataset** — browse and filter the raw table.
- **Overview** — row/column counts, memory footprint, column types, and a
  per-column summary (dtype, non-null count, missing %, unique values).
- **Data Quality** — missing value counts and share of the dataset affected.
- **Statistics** — descriptive statistics for numeric or all columns.
- **Correlations** — correlation heatmap across numeric columns.
- **Distributions** — histogram with adjustable bins and summary stats for a
  chosen numeric column.
- **AI Assistant** — describe an analysis in plain language and get pandas
  code generated against the dataset; review and edit it before running.

No sample data required to try it out — use the "Load sample dataset" button
in the sidebar.

## Setup

```bash
python -m venv myenv
myenv/Scripts/activate   # On macOS/Linux: source myenv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with an OpenAI API key (required
only for the AI Assistant tab):

```
OPEN_AI_API_KEY=your_key_here
```

If you're deploying this somewhere public, also set an access code so
random visitors can't drain your API credits. The rest of the app (dataset
browsing, stats, charts) stays open either way — only the AI Assistant is
gated:

```
ASSISTANT_ACCESS_CODE=pick_something_only_you_know
```

## Run

```bash
streamlit run streamlit_app.py
```

## Notes

The AI Assistant executes generated code with `exec()` against a copy of
the loaded dataframe. A keyword filter blocks obvious escapes (`import os`,
`subprocess`, `open(`, `eval(`, etc.), but this is not a sandbox — review
generated code before running it, especially on data you didn't create
yourself.
