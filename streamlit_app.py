import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import contextlib

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY, transport="rest")


model = genai.GenerativeModel("gemini-2.5-flash")

# Keywords that are not allowed in AI-generated code before we run it.
# This is a simple safety net, not a real sandbox -- exec() is still exec().
BLOCKED_KEYWORDS = [
    "import os", "import sys", "subprocess", "shutil",
    "open(", "eval(", "exec(", "__import__", "input(",
]


st.set_page_config(page_title="Mini EDA App", layout="wide")


st.title("📊 Mini AI-Powered EDA Report")
st.write("Upload a CSV file and explore your dataset.")


uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])


if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    # Sidebar
    st.sidebar.title("Navigation")
    option = st.sidebar.radio(
        "Go To",
        [
            "Dataset",
            "Overview",
            "Missing Values",
            "Statistics",
            "Correlation",
            "Distribution",
            "AI Pandas Assistant",
        ],
    )

    if option == "Dataset":
        st.subheader("Dataset")
        st.dataframe(df)

    elif option == "Overview":
        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Duplicates", df.duplicated().sum())

        st.write("### Data Types")
        st.dataframe(df.dtypes.astype(str))

    elif option == "Missing Values":
        st.subheader("Missing Values")

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) == 0:
            st.success("No Missing Values Found 🎉")
        else:
            st.bar_chart(missing)

    elif option == "Statistics":
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

    elif option == "Correlation":

        numeric = df.select_dtypes(include="number")

        if numeric.shape[1] < 2:
            st.warning("Need at least two numeric columns.")
        else:

            corr = numeric.corr()

            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.imshow(corr, cmap="coolwarm")

            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45)

            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns)

            plt.colorbar(cax)

            st.pyplot(fig)

    elif option == "Distribution":

        numeric = df.select_dtypes(include="number").columns

        if len(numeric) == 0:
            st.warning("No Numeric Columns")
        else:

            column = st.selectbox("Select Column", numeric)

            fig, ax = plt.subplots()

            ax.hist(df[column].dropna(), bins=20)

            ax.set_title(column)

            st.pyplot(fig)

    # -------------------------------
    # AI Pandas Assistant
    # -------------------------------
    elif option == "AI Pandas Assistant":

        st.subheader("🤖 AI Pandas Code Generator")

        user_query = st.text_input(
            "What analysis do you want to perform?",
            placeholder="Example: Show average salary department-wise",
        )

        if "generated_code" not in st.session_state:
            st.session_state.generated_code = ""

        if st.button("Generate Pandas Code"):

            if user_query.strip() == "":
                st.warning("Please enter a query.")

            else:

                prompt = f"""
                You are an expert Python Data Analyst.




                Dataset Columns:
                {list(df.columns)}




                Data Types:
                {df.dtypes.to_string()}




                User Request:
                {user_query}




                Rules:
                1. Generate ONLY executable pandas code.
                2. Assume the dataframe name is df.
                3. Store the final answer (a DataFrame, Series, number, or matplotlib plot) in a variable named result.
                4. Do not explain anything.
                5. Do not use markdown or code fences.
                6. Return only Python code.
                """

                with st.spinner("Generating Code..."):

                    response = model.generate_content(prompt)

                # Gemini sometimes adds ```python fences even when told not to
                code = response.text.strip().strip("`")
                if code.startswith("python"):
                    code = code[len("python"):].strip()

                st.session_state.generated_code = code

        if st.session_state.generated_code:

            st.write("### Generated Code")
            edited_code = st.text_area(
                "You can review or edit the code before running it",
                value=st.session_state.generated_code,
                height=200,
            )

            if st.button("▶ Run Code"):

                if any(word in edited_code for word in BLOCKED_KEYWORDS):
                    st.error("This code uses operations that aren't allowed to run automatically.")

                else:
                    local_vars = {"df": df.copy(), "pd": pd, "np": np, "plt": plt}
                    output_buffer = io.StringIO()

                    try:
                        with contextlib.redirect_stdout(output_buffer):
                            exec(edited_code, {}, local_vars)

                        printed_output = output_buffer.getvalue()
                        if printed_output:
                            st.text(printed_output)

                        result = local_vars.get("result")

                        if isinstance(result, (pd.DataFrame, pd.Series)):
                            st.dataframe(result)
                        elif plt.get_fignums():
                            st.pyplot(plt.gcf())
                        elif result is not None:
                            st.write(result)
                        elif not printed_output:
                            st.info(
                                "Code ran but produced no output. "
                                "Make sure the answer is stored in a variable named `result`."
                            )

                    except Exception as e:
                        st.error(f"Error while running generated code: {e}")

                    finally:
                        plt.close("all")


else:
    st.info("Upload a CSV file to begin.")
