import contextlib
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.6-flash"

# Keywords that are not allowed in AI-generated code before we run it.
# This is a simple safety net, not a real sandbox -- exec() is still exec().
BLOCKED_KEYWORDS = [
    "import os", "import sys", "subprocess", "shutil",
    "open(", "eval(", "exec(", "__import__", "input(",
]

PALETTE = ["#1F6F5C", "#C17A2E", "#5B6E8C", "#A6403A", "#7C8A56", "#8C5B7C"]
ACCENT = PALETTE[0]

NAV_PAGES = [
    "Dataset",
    "Overview",
    "Data Quality",
    "Statistics",
    "Correlations",
    "Distributions",
    "AI Assistant",
]


# --------------------------------------------------------------------------
# Page shell
# --------------------------------------------------------------------------

st.set_page_config(page_title="Dataset Explorer", page_icon="▦", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.3rem; max-width: 1180px; }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        font-weight: 600;
        color: #1F6F5C;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: #6b675c;
        font-size: 0.95rem;
        margin-top: -0.6rem;
        margin-bottom: 1.6rem;
    }
    hr.header-rule {
        border: none;
        border-top: 1px solid #DDD8CB;
        margin: 0 0 1.8rem 0;
    }

    .sidebar-brand {
        padding-bottom: 1rem;
        border-bottom: 1px solid #2C3B35;
        margin-bottom: 1.1rem;
    }
    .sidebar-brand .eyebrow { color: #4FB99B; margin-bottom: 0.1rem; }
    .sidebar-brand h2 { color: #EDEAE2; margin: 0; font-size: 1.25rem; }

    .stat-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        padding: 0.32rem 0;
        color: #ADA99C;
        border-bottom: 1px dashed #2C3B35;
    }
    .stat-row span:last-child {
        color: #EDEAE2;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    .footer-note {
        color: #9a9687;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #DDD8CB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(eyebrow: str, title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if caption:
        st.caption(caption)


def make_sample_dataset(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Home", "Apparel", "Sports", "Grocery"]
    dates = pd.date_range("2024-01-01", periods=365, freq="D")

    units_sold = rng.integers(1, 40, size=n)
    unit_price = np.round(rng.uniform(5, 300, size=n), 2)
    discount = np.round(rng.uniform(0, 0.3, size=n), 2)
    revenue = np.round(units_sold * unit_price * (1 - discount), 2)
    rating = np.clip(np.round(rng.normal(4.1, 0.6, size=n), 1), 1, 5)

    df = pd.DataFrame({
        "order_id": np.arange(10001, 10001 + n),
        "order_date": rng.choice(dates, size=n),
        "region": rng.choice(regions, size=n, p=[0.3, 0.25, 0.25, 0.2]),
        "category": rng.choice(categories, size=n),
        "units_sold": units_sold,
        "unit_price": unit_price,
        "discount": discount,
        "revenue": revenue,
        "customer_rating": rating,
        "is_returned": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
    })

    df.loc[rng.choice(df.index, size=int(n * 0.08), replace=False), "customer_rating"] = np.nan
    df.loc[rng.choice(df.index, size=int(n * 0.05), replace=False), "discount"] = np.nan

    return df.sort_values("order_date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Sidebar: data source + navigation
# --------------------------------------------------------------------------

st.sidebar.markdown(
    '<div class="sidebar-brand"><div class="eyebrow">Workspace</div>'
    "<h2>Dataset Explorer</h2></div>",
    unsafe_allow_html=True,
)

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

if uploaded_file is not None:
    st.session_state.use_sample = False
elif st.sidebar.button("Load sample dataset", width="stretch"):
    st.session_state.use_sample = True

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif st.session_state.use_sample:
    df = make_sample_dataset()

if df is not None:
    missing_pct = df.isna().to_numpy().mean() * 100
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    st.sidebar.markdown(
        f"""
        <div class="stat-row"><span>Rows</span><span>{df.shape[0]:,}</span></div>
        <div class="stat-row"><span>Columns</span><span>{df.shape[1]:,}</span></div>
        <div class="stat-row"><span>Missing</span><span>{missing_pct:.1f}%</span></div>
        <div class="stat-row"><span>Memory</span><span>{memory_mb:.2f} MB</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    page = st.sidebar.radio("Navigate", NAV_PAGES, label_visibility="collapsed")


# --------------------------------------------------------------------------
# Main content
# --------------------------------------------------------------------------

if df is None:
    section_header("Getting started", "Dataset Explorer")
    st.markdown(
        '<p class="app-subtitle">Upload a CSV to get a data quality report, '
        "summary statistics, correlations, distributions, and a pandas "
        "assistant for ad-hoc questions.</p>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**No file handy?**")
            st.write(
                "Use the sample retail dataset from the sidebar to see every "
                "view populated before uploading your own data."
            )
        with right:
            if st.button("Load sample dataset", width="stretch"):
                st.session_state.use_sample = True
                st.rerun()

else:
    section_header("Dataset Explorer", page)
    st.markdown('<hr class="header-rule">', unsafe_allow_html=True)

    if page == "Dataset":
        columns = st.multiselect("Columns", df.columns.tolist(), default=df.columns.tolist())
        st.caption(f"{df.shape[0]:,} rows × {df.shape[1]:,} columns")
        st.dataframe(df[columns] if columns else df, width="stretch", height=520)

    elif page == "Overview":
        c1, c2, c3, c4 = st.columns(4)
        with c1, st.container(border=True):
            st.metric("Rows", f"{df.shape[0]:,}")
        with c2, st.container(border=True):
            st.metric("Columns", f"{df.shape[1]:,}")
        with c3, st.container(border=True):
            st.metric("Duplicate rows", f"{df.duplicated().sum():,}")
        with c4, st.container(border=True):
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

        st.write("")
        left, right = st.columns([2, 3])

        with left:
            st.markdown("**Column types**")
            dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
            dtype_counts.columns = ["dtype", "count"]
            fig = px.bar(
                dtype_counts, x="count", y="dtype", orientation="h",
                color="dtype", color_discrete_sequence=PALETTE,
            )
            fig.update_layout(
                showlegend=False, height=280,
                margin=dict(l=0, r=10, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title=None, xaxis_title=None,
            )
            st.plotly_chart(fig, width="stretch")

        with right:
            st.markdown("**Column summary**")
            summary = pd.DataFrame({
                "column": df.columns,
                "dtype": df.dtypes.astype(str).values,
                "non_null": df.notna().sum().values,
                "missing_pct": (df.isna().mean() * 100).round(1).values,
                "unique": df.nunique().values,
            })
            st.dataframe(
                summary,
                hide_index=True,
                width="stretch",
                height=280,
                column_config={
                    "missing_pct": st.column_config.ProgressColumn(
                        "Missing %", min_value=0, max_value=100, format="%.1f%%"
                    ),
                },
            )

    elif page == "Data Quality":
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)

        if len(missing) == 0:
            st.success("No missing values found.")
        else:
            m1, m2, m3 = st.columns(3)
            with m1, st.container(border=True):
                st.metric("Affected columns", len(missing))
            with m2, st.container(border=True):
                st.metric("Missing cells", f"{missing.sum():,}")
            with m3, st.container(border=True):
                st.metric("Share of dataset", f"{df.isna().to_numpy().mean() * 100:.1f}%")

            st.write("")
            missing_df = missing.reset_index()
            missing_df.columns = ["column", "missing"]
            fig = px.bar(
                missing_df, x="missing", y="column", orientation="h",
                color="missing", color_continuous_scale=["#E7EFEB", ACCENT],
            )
            fig.update_layout(
                height=max(280, 32 * len(missing_df)),
                margin=dict(l=0, r=10, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, yaxis_title=None, xaxis_title="Missing values",
            )
            st.plotly_chart(fig, width="stretch")

    elif page == "Statistics":
        include_all = st.toggle("Include non-numeric columns")

        if include_all:
            desc = df.describe(include="all").T
            # Mixed dtypes (e.g. numeric stats next to datetime/categorical
            # ones) break Arrow serialization, so render this view as text.
            desc = desc.where(desc.notna(), "").astype(str)
        else:
            # pandas' default describe() folds datetime columns in alongside
            # numeric ones, which produces the same Arrow issue -- restrict
            # explicitly to numeric dtypes here.
            desc = df.describe(include=[np.number]).T

        if desc.empty:
            with st.container(border=True):
                st.warning("No numeric columns available to summarize.")
        else:
            st.dataframe(desc, width="stretch", height=460)

    elif page == "Correlations":
        numeric = df.select_dtypes(include="number")

        if numeric.shape[1] < 2:
            with st.container(border=True):
                st.warning("Need at least two numeric columns to compute correlations.")
        else:
            corr = numeric.corr()
            fig = px.imshow(
                corr,
                color_continuous_scale=[(0, "#A6403A"), (0.5, "#F7F5F0"), (1, ACCENT)],
                zmin=-1, zmax=1, aspect="auto",
                text_auto=".2f" if corr.shape[1] <= 14 else False,
            )
            fig.update_layout(
                height=520, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch")

    elif page == "Distributions":
        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) == 0:
            with st.container(border=True):
                st.warning("No numeric columns available to plot.")
        else:
            left, right = st.columns([3, 1])
            with left:
                column = st.selectbox("Column", numeric_cols)
            with right:
                bins = st.slider("Bins", 5, 80, 30)

            series = df[column].dropna()

            s1, s2, s3, s4 = st.columns(4)
            with s1, st.container(border=True):
                st.metric("Mean", f"{series.mean():.2f}")
            with s2, st.container(border=True):
                st.metric("Median", f"{series.median():.2f}")
            with s3, st.container(border=True):
                st.metric("Std. dev", f"{series.std():.2f}")
            with s4, st.container(border=True):
                st.metric("Skew", f"{series.skew():.2f}")

            st.write("")
            fig = px.histogram(
                df, x=column, nbins=bins, marginal="box",
                color_discrete_sequence=[ACCENT],
            )
            fig.update_layout(
                height=440, margin=dict(l=0, r=10, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                bargap=0.05,
            )
            st.plotly_chart(fig, width="stretch")

    elif page == "AI Assistant":
        st.caption(
            "Describe the analysis you want in plain language. The model writes "
            "pandas code against `df`; review it, then run it."
        )

        user_query = st.text_input(
            "Analysis request",
            placeholder="e.g. Show average revenue by region, sorted descending",
            label_visibility="collapsed",
        )

        if "generated_code" not in st.session_state:
            st.session_state.generated_code = ""

        if st.button("Generate code"):
            if user_query.strip() == "":
                st.warning("Enter a request first.")
            else:
                prompt = f"""You are an expert Python data analyst.

Dataset columns:
{list(df.columns)}

Data types:
{df.dtypes.to_string()}

User request:
{user_query}

Rules:
1. Generate ONLY executable pandas code.
2. Assume the dataframe name is df.
3. Store the final answer (a DataFrame, Series, number, or matplotlib plot) in a variable named result.
4. Do not explain anything.
5. Do not use markdown or code fences.
6. Return only Python code."""

                with st.spinner("Generating code..."):
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=prompt,
                    )

                # The model sometimes adds ```python fences even when told not to.
                code = response.text.strip().strip("`")
                if code.startswith("python"):
                    code = code[len("python"):].strip()

                st.session_state.generated_code = code

        if st.session_state.generated_code:
            st.write("")
            st.markdown('<div class="eyebrow">Generated code</div>', unsafe_allow_html=True)
            edited_code = st.text_area(
                "Generated code",
                value=st.session_state.generated_code,
                height=200,
                label_visibility="collapsed",
            )

            if st.button("Run code"):
                if any(word in edited_code for word in BLOCKED_KEYWORDS):
                    st.error("This code uses operations that aren't allowed to run automatically.")
                else:
                    local_vars = {"df": df.copy(), "pd": pd, "np": np}
                    import matplotlib.pyplot as plt
                    local_vars["plt"] = plt
                    output_buffer = io.StringIO()

                    st.markdown('<div class="eyebrow">Result</div>', unsafe_allow_html=True)
                    try:
                        with contextlib.redirect_stdout(output_buffer):
                            exec(edited_code, {}, local_vars)

                        printed_output = output_buffer.getvalue()
                        if printed_output:
                            st.text(printed_output)

                        result = local_vars.get("result")

                        if isinstance(result, (pd.DataFrame, pd.Series)):
                            st.dataframe(result, width="stretch")
                        elif plt.get_fignums():
                            st.pyplot(plt.gcf())
                        elif result is not None:
                            st.write(result)
                        elif not printed_output:
                            st.info(
                                "Code ran but produced no output. Make sure the "
                                "answer is stored in a variable named `result`."
                            )

                    except Exception as e:
                        st.error(f"Error while running generated code: {e}")

                    finally:
                        plt.close("all")

    st.markdown(
        '<div class="footer-note">Dataset Explorer &middot; local session, '
        "nothing is stored server-side.</div>",
        unsafe_allow_html=True,
    )
