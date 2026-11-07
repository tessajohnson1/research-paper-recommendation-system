# Research Paper Recommendation System Using NLP

A complete M.Tech Computational Linguistics mini-project starter. It recommends research papers from their title, abstract, and keywords using:
1. TF-IDF + cosine similarity (lexical/statistical baseline)
2. Sentence Transformer embeddings + cosine similarity (semantic approach)
3. Hybrid ranking (weighted combination of normalized TF-IDF and semantic scores)

The Streamlit UI supports topic queries, selecting a seed paper, CSV upload, result filtering, and exporting recommendations.

## Project structure

```text
research_paper_recommender/
├── app.py
├── recommender.py
├── data_loader.py
├── evaluate.py
├── fetch_arxiv.py
├── requirements.txt
├── data/
│   └── papers.csv
├── tests/
│   └── test_recommender.py
└── README.md
```

## 1. Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Sentence Transformers downloads its model the first time and needs internet. If that is unavailable, choose TF-IDF in the app; the lexical method works without the embedding model after packages are installed.

## 2. Dataset

`data/papers.csv` is a small illustrative dataset to verify the application. It is not large enough for research-grade conclusions. For your final experiment, use a larger, documented collection of paper metadata and keep its source/license details in your report.

Expected columns:
- `paper_id` (optional)
- `title` (required)
- `abstract` (recommended)
- `authors` (optional)
- `year` (optional)
- `keywords` (optional)
- `url` (optional)

You can upload a CSV through the app. Missing optional columns are filled automatically.

### Fetch public metadata from arXiv

Use the helper to fetch metadata for a query. Respect the API's current terms and rate limits, and record the date/query in your report.

```powershell
python fetch_arxiv.py --query "computational linguistics" --max-results 100 --output data/arxiv_papers.csv
```

You can also use queries such as `natural language processing`, `information retrieval`, or `text classification`.

## 3. How the recommendation methods work

### TF-IDF
The title, abstract and keywords are combined, then vectorized using TF-IDF. Cosine similarity ranks papers by lexical overlap. It is a strong, interpretable baseline but may miss related papers that use different wording.

### Semantic embeddings
A pretrained Sentence Transformer converts each paper into a dense vector. Cosine similarity compares meaning, so semantically related wording may match even with less literal overlap. The first run downloads the model.

### Hybrid
A weighted score combines the two methods:
`hybrid = alpha * TF-IDF score + (1-alpha) * semantic score`.
The app lets you choose alpha. Scores are ranking signals, not calibrated probabilities.

## 4. Evaluation

The included `evaluate.py` calculates Precision@K, Recall@K and reciprocal rank for a small set of query-to-relevant-paper labels. Replace the example judgments with human-checked relevance judgments for your dataset.

```powershell
python evaluate.py --data data/papers.csv --judgments data/relevance_judgments.csv --k 5 --method tfidf
python evaluate.py --data data/papers.csv --judgments data/relevance_judgments.csv --k 5 --method semantic
python evaluate.py --data data/papers.csv --judgments data/relevance_judgments.csv --k 5 --method hybrid
```

Judgment CSV format:
```csv
query,relevant_paper_ids
"fake news detection","P003;P004;P012"
"semantic search","P006;P008"
```
Relevant IDs must match `paper_id` values in your dataset. The provided example judgments are only a format demo; do not report them as valid research results without checking relevance.

## 5. Research methodology

1. Collect research-paper metadata (title, abstract, keywords, year, authors, URL).
2. Clean missing values and normalize fields; preserve context for the semantic model.
3. Build a TF-IDF matrix over titles, abstracts and keywords.
4. Encode the same paper text with a Sentence Transformer.
5. For a user query or selected paper, compute cosine similarities.
6. Rank results, exclude the seed paper, and display the top K.
7. Evaluate using manually judged query-paper relevance and Precision@K, Recall@K, MRR and response time.
8. Compare lexical, semantic and hybrid methods on the same queries and same dataset.
## System Architecture

![System Architecture](imagessystem_architecture.png)
## Suggested experiments

- Vary `alpha` for hybrid ranking (0.25, 0.50, 0.75).
- Compare title-only versus title + abstract + keywords.
- Test short broad queries and detailed natural-language queries.
- Report mean Precision@5, Recall@5, MRR and query latency.
- Include examples where TF-IDF and semantic rankings differ.

##  Limitations and ethics

- Recommendations depend on dataset coverage and metadata quality.
- Semantic similarity is not a guarantee of scientific quality or correctness.
- Similarity scores are not probabilities.
- A system trained/evaluated on a small or automatically generated sample cannot support strong performance claims.
- Avoid implying that a paper is open access unless its URL/access status has been verified.


## Pooled human-judged evaluation

The project also includes `create_eval_pool.py` and `evaluate_all.py`. Run `python create_eval_pool.py` to create `evaluation_pool_to_judge.csv`. Review every query-paper pair and fill the `relevance` column with `1` (relevant) or `0` (not relevant), then save the CSV. Run `python evaluate_all.py` to compare TF-IDF, Semantic, and Hybrid using Precision@5, Recall@5, and MRR. The evaluation scripts use `data/research_papers_combined_P001_format.csv` and expect the query and judgment CSV files in the project root. The candidate pool is not ground truth; relevance labels must be assigned by a human.

The merged archive intentionally excludes both `.venv` directories and Python cache files. Create one fresh virtual environment after extracting the project.
