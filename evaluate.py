import argparse
import pandas as pd
from data_loader import load_papers
from recommender import PaperRecommender

def parse_ids(value):
    return [x.strip() for x in str(value).split(";") if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="Evaluate paper recommendations using relevance judgments.")
    parser.add_argument("--data", default="data/papers.csv")
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--method", choices=["tfidf", "semantic", "hybrid"], default="tfidf")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    papers = load_papers(args.data)
    judgments = pd.read_csv(args.judgments).fillna("")
    if not {"query", "relevant_paper_ids"}.issubset(judgments.columns):
        raise ValueError("Judgments file needs query and relevant_paper_ids columns.")
    rec = PaperRecommender(papers)
    p_values, r_values, rr_values = [], [], []
    for _, row in judgments.iterrows():
        relevant = set(parse_ids(row["relevant_paper_ids"]))
        if not relevant:
            continue
        results = rec.recommend(row["query"], method=args.method, top_k=args.k, alpha=args.alpha)
        ranked = results["paper_id"].astype(str).tolist()
        hits = sum(pid in relevant for pid in ranked)
        p_values.append(hits / args.k)
        r_values.append(hits / len(relevant))
        first_rank = next((i + 1 for i, pid in enumerate(ranked) if pid in relevant), None)
        rr_values.append(1 / first_rank if first_rank else 0.0)
    if not p_values:
        raise ValueError("No non-empty relevance judgments found.")
    print(f"Method: {args.method}; K={args.k}; queries={len(p_values)}")
    print(f"Mean Precision@{args.k}: {sum(p_values)/len(p_values):.4f}")
    print(f"Mean Recall@{args.k}:    {sum(r_values)/len(r_values):.4f}")
    print(f"MRR:                    {sum(rr_values)/len(rr_values):.4f}")
    print("Use human-checked relevance labels; scores are only meaningful for your judged test set.")

if __name__ == "__main__":
    main()
