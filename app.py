
from pathlib import Path
import time
import pandas as pd
import streamlit as st

from data_loader import load_papers
from recommender import PaperRecommender


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).parent

DEFAULT_DATA = (
    ROOT / "data" / "research_papers_combined_P001_format.csv"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(
    page_title="Research Paper Recommender",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper Recommender")


# --------------------------------------------------
# 2. DATA LOADING AND CACHING
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def read_default_data(path, modified_time):
    return load_papers(path)


@st.cache_resource(show_spinner="Building recommendation index...")
def build_recommender(df, model_name):
    return PaperRecommender(df, model_name=model_name)


# --------------------------------------------------
# 3. RECOMMENDATION SETTINGS
# --------------------------------------------------


with st.sidebar:
    st.header("Recommendation Settings")

    method_label = st.selectbox(
        "Method",
        ["Hybrid", "TF-IDF", "Semantic embeddings"],
        index=0
    )

    method = {
        "Hybrid": "hybrid",
        "TF-IDF": "tfidf",
        "Semantic embeddings": "semantic"
    }[method_label]

    top_k = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    if method == "hybrid":
        alpha = st.slider(
            "TF-IDF weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Controls the contribution of TF-IDF compared with semantic similarity."
        )
    else:
        alpha = 0.5

    if method == "semantic":
        st.caption(
            "Uses a pretrained Sentence Transformer model."
        )


# --------------------------------------------------
# 4. LOAD DATASET
# --------------------------------------------------

try:
    papers = read_default_data(
        str(DEFAULT_DATA),
        DEFAULT_DATA.stat().st_mtime
    )

except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()


if len(papers) < 2:
    st.error("Please provide at least two valid research papers.")
    st.stop()


# --------------------------------------------------
# 5. PUBLICATION YEAR FILTER
# --------------------------------------------------

if "year" in papers.columns:

    year_values = pd.to_numeric(
        papers["year"],
        errors="coerce"
    )

    available_years = sorted(
        year_values.dropna().astype(int).unique().tolist()
    )

    with st.sidebar:

        st.divider()
        st.header("Publication Year")

        year_mode = st.selectbox(
            "Select year filter",
            ["All years", "Single year", "Year range"],
            index=0
        )

        if available_years:

            if year_mode == "Single year":

                selected_year = st.selectbox(
                    "Select publication year",
                    available_years,
                    index=len(available_years) - 1
                )

                papers = papers.loc[
                    year_values == selected_year
                ].copy()

            elif year_mode == "Year range":

                if len(available_years) >= 2:

                    selected_range = st.select_slider(
                        "Select publication year range",
                        options=available_years,
                        value=(
                            available_years[0],
                            available_years[-1]
                        )
                    )

                    start_year, end_year = selected_range

                else:
                    start_year = available_years[0]
                    end_year = available_years[0]

                    st.info(
                        "Only one publication year is available."
                    )

                papers = papers.loc[
                    year_values.between(start_year, end_year)
                ].copy()

        else:
            st.warning("No valid publication years found.")

else:
    st.warning("The dataset does not contain a year column.")


papers = papers.reset_index(drop=True)


# --------------------------------------------------
# 6. VALIDATE FILTERED DATA
# --------------------------------------------------

if len(papers) < 2:
    st.error(
        "Fewer than two papers are available for the selected year filter. "
        "Please select another year or All years."
    )
    st.stop()


# --------------------------------------------------
# 7. DATASET INFORMATION
# --------------------------------------------------

col1, col2 = st.columns(2)

col1.metric(
    "Papers Indexed",
    f"{len(papers):,}"
)

col2.metric(
    "Available Methods",
    "3"
)


# --------------------------------------------------
# 8. BUILD RECOMMENDATION MODEL
# --------------------------------------------------

try:
    recommender = build_recommender(
        papers,
        MODEL_NAME
    )

except Exception as exc:
    st.error(
        f"Unable to prepare recommendation index: {exc}"
    )
    st.stop()


# --------------------------------------------------
# 9. WEBSITE TABS
# --------------------------------------------------

tab_search, tab_similar = st.tabs(
    [
        "Search by topic",
        "Find similar papers"
    ]
)


# --------------------------------------------------
# 10. DISPLAY RECOMMENDATION RESULTS
# --------------------------------------------------

def display_results(results):

    if results.empty:
        st.info("No recommendations found.")
        return

    export_cols = [
        "paper_id",
        "title",
        "authors",
        "year",
        "score",
        "tfidf_score",
        "semantic_score",
        "url"
    ]

    available = [
        col for col in export_cols
        if col in results.columns
    ]

    export = results[available].copy()

    export["score"] = export["score"].round(4)

    export = export.rename(
        columns={
            "paper_id": "Paper ID",
            "title": "Title",
            "authors": "Authors",
            "year": "Year",
            "score": "Ranking score",
            "tfidf_score": "TF-IDF score",
            "semantic_score": "Semantic score",
            "url": "Source URL"
        }
    )

    st.dataframe(
        export,
        use_container_width=True,
        hide_index=True
    )

    for i, (_, row) in enumerate(results.iterrows()):

        with st.expander(
            f"{i + 1}. {row['title']}"
        ):

            meta = " · ".join(
                x for x in [
                    str(row.get("authors", "")).strip(),
                    str(row.get("year", "")).strip()
                ]
                if x
            )

            if meta:
                st.caption(meta)

            st.write(
                str(row.get("abstract", "")).strip()
                or "No abstract available."
            )

            st.caption(
                f"Ranking score: {row['score']:.4f} "
                "(not a probability)"
            )

            url = str(row.get("url", "")).strip()

            if url.startswith(("https://", "http://")):
                st.markdown(
                    f"[Open source paper]({url})"
                )

    st.download_button(
        "Download recommendations as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="paper_recommendations.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 11. SEARCH BY TOPIC
# --------------------------------------------------

with tab_search:

    st.subheader("Search by research topic")

    query = st.text_input(
        "Enter a topic or research question",
        placeholder="e.g., transformer models for sentiment analysis"
    )

    if st.button(
        "Recommend papers",
        type="primary",
        key="search_button"
    ):

        if not query.strip():

            st.warning("Enter a topic first.")

        else:

            try:

                start = time.perf_counter()

                with st.spinner("Finding relevant papers..."):

                    results = recommender.recommend(
                        query,
                        method,
                        top_k,
                        alpha
                    )

                elapsed = (
                    time.perf_counter() - start
                ) * 1000

                st.success(
                    f"Found recommendations in {elapsed:.0f} ms."
                )

                display_results(results)

            except Exception as exc:
                st.error(str(exc))


# --------------------------------------------------
# 12. FIND SIMILAR PAPERS
# --------------------------------------------------

with tab_similar:

    st.subheader(
        "Find papers similar to a selected paper"
    )

    options = {
        f"{row['title']} [{row['paper_id']}]":
        str(row["paper_id"])
        for _, row in papers.iterrows()
    }

    selected_label = st.selectbox(
        "Select a paper",
        list(options.keys())
    )

    if st.button(
        "Find similar papers",
        type="primary"
    ):

        try:

            start = time.perf_counter()

            results = recommender.recommend_for_paper(
                options[selected_label],
                method=method,
                top_k=top_k,
                alpha=alpha
            )

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            st.success(
                f"Recommendations ready in {elapsed:.0f} ms."
            )

            display_results(results)

        except Exception as exc:
            st.error(str(exc))





