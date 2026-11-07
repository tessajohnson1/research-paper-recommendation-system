from pathlib import Path
import pandas as pd
from data_loader import load_papers
from recommender import PaperRecommender

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "research_papers_combined_P001_format.csv"
QUERIES = ROOT / "queries.csv"
OUT = ROOT / "evaluation_pool_to_judge.csv"
K_POOL = 10
ALPHA = 0.5


def main():
    papers = load_papers(DATA)
    queries = pd.read_csv(QUERIES)["query"].dropna().astype(str).tolist()
    rec = PaperRecommender(papers)
    rows = []
    print("Building all three rankings. First semantic run may download the model.")
    lookup = papers.copy()
    lookup["paper_id"] = lookup["paper_id"].astype(str)
    lookup = lookup.drop_duplicates("paper_id").set_index("paper_id")
    for query in queries:
        per_method = {}
        for method in ("tfidf", "semantic", "hybrid"):
            per_method[method] = rec.recommend(
                query, method=method, top_k=K_POOL, alpha=ALPHA
            ).reset_index(drop=True)
        ids = []
        for method in ("tfidf", "semantic", "hybrid"):
            ids.extend(per_method[method]["paper_id"].astype(str).tolist())
        ids = list(dict.fromkeys(ids))
        for paper_id in ids:
            paper = lookup.loc[paper_id]
            row = {
                "query": query, "paper_id": paper_id,
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "year": paper.get("year", ""),
                "abstract": paper.get("abstract", ""),
                "tfidf_rank": "", "semantic_rank": "", "hybrid_rank": "",
                "relevance": ""
            }
            for method in ("tfidf", "semantic", "hybrid"):
                ranked_ids = per_method[method]["paper_id"].astype(str).tolist()
                if paper_id in ranked_ids:
                    row[f"{method}_rank"] = ranked_ids.index(paper_id) + 1
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Created {OUT} with {len(rows)} query-paper candidates.")
    print("Fill relevance for every row: 1 = relevant, 0 = not relevant.")

if __name__ == "__main__":
    main()
