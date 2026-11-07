RESEARCH PAPER RECOMMENDER - EVALUATION GUIDE

The package contains your 1,098-paper CSV dataset, recommender.py, data_loader.py, ten starting queries, and two evaluation scripts.

IMPORTANT: The evaluation pool is not ground truth. A person must inspect each query-paper pair using its title and abstract and enter relevance: 1 = relevant, 0 = not relevant. Fill every row. Every query must have at least one relevant paper. Do not invent labels.

WINDOWS / POWERSHELL
1. Extract the ZIP and open the folder in VS Code.
2. Create and activate a virtual environment:
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
3. Install dependencies:
   python -m pip install --upgrade pip
   pip install -r requirements.txt
4. Generate the candidate pool (first semantic run may download the model):
   python create_eval_pool.py
5. Open evaluation_pool_to_judge.csv in Excel. Review every query-paper pair and enter 1 or 0 in the relevance column. Save as CSV.
6. Calculate metrics:
   python evaluate_all.py

OUTPUTS
- evaluation_summary.csv: mean Precision@5, Recall@5 and MRR for TF-IDF, Semantic and Hybrid.
- evaluation_per_query.csv: per-query results for each method.

EXPERIMENT SETTINGS
K = 5; hybrid alpha = 0.5; model = sentence-transformers/all-MiniLM-L6-v2. The candidate pool is the union of each method's top 10 results for each query. Describe this as a pooled human-judged evaluation in your report.

METRICS
Precision@5 = relevant items in top 5 / 5.
Recall@5 = relevant items in top 5 / all judged relevant items for that query.
Reciprocal rank = 1 / rank of first relevant item. MRR is the mean reciprocal rank across queries.
These are retrieval metrics, not classification accuracy. Results depend on the queries and human relevance judgments.
