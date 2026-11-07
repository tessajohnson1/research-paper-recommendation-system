from pathlib import Path
import pandas as pd
from data_loader import load_papers
from recommender import PaperRecommender

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "research_papers_combined_P001_format.csv"
JUDGMENTS = ROOT / "evaluation_pool_to_judge.csv"
K = 5
ALPHA = 0.5
METHODS = ("tfidf", "semantic", "hybrid")


def main():
    papers = load_papers(DATA)
    judgments = pd.read_csv(JUDGMENTS, keep_default_na=False)
    required = {"query", "paper_id", "relevance"}
    if not required.issubset(judgments.columns):
        raise ValueError(f"Judgment CSV must contain columns: {sorted(required)}")
    if (judgments["relevance"].astype(str).str.strip() == "").any():
        raise ValueError("Some relevance cells are blank. Judge every candidate first.")
    judgments["relevance"] = pd.to_numeric(judgments["relevance"], errors="raise").astype(int)
    if not judgments["relevance"].isin([0, 1]).all():
        raise ValueError("Relevance values must be 1 or 0.")
    paper_ids = set(papers["paper_id"].astype(str))
    unknown = set(judgments["paper_id"].astype(str)) - paper_ids
    if unknown:
        raise ValueError(f"Unknown paper IDs: {sorted(unknown)[:10]}")

    rec = PaperRecommender(papers)
    summary_rows, detail_rows = [], []
    for method in METHODS:
        p_values, r_values, rr_values = [], [], []
        for query, group in judgments.groupby("query", sort=False):
            relevant = set(group.loc[group["relevance"] == 1, "paper_id"].astype(str))
            if not relevant:
                raise ValueError(f"No relevant papers marked for query: {query}")
            ranked = rec.recommend(query, method=method, top_k=len(papers), alpha=ALPHA)["paper_id"].astype(str).tolist()
            top = ranked[:K]
            hits = sum(pid in relevant for pid in top)
            precision = hits / K
            recall = hits / len(relevant)
            first_rank = next((i + 1 for i, pid in enumerate(ranked) if pid in relevant), None)
            rr = 1 / first_rank if first_rank else 0.0
            p_values.append(precision)
            r_values.append(recall)
            rr_values.append(rr)
            detail_rows.append({"method": method, "query": query,
                                f"precision@{K}": precision, f"recall@{K}": recall,
                                "reciprocal_rank": rr,
                                "first_relevant_rank": first_rank or "Not found"})
        summary_rows.append({"method": method, f"precision@{K}": sum(p_values)/len(p_values),
                             f"recall@{K}": sum(r_values)/len(r_values),
                             "MRR": sum(rr_values)/len(rr_values),
                             "number_of_queries": len(p_values)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ROOT / "evaluation_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(ROOT / "evaluation_per_query.csv", index=False)
    print("\nMEAN RESULTS")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Saved evaluation_summary.csv and evaluation_per_query.csv")

if __name__ == "__main__":
    main()
