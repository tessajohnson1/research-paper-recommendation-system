from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import paper_text

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class PaperRecommender:
    def __init__(self, papers, model_name=DEFAULT_MODEL):
        self.papers = papers.reset_index(drop=True).copy()
        self.model_name = model_name
        self.texts = [paper_text(row) for _, row in self.papers.iterrows()]
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            max_features=100000, sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        self._model = None
        self._embeddings = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Semantic mode requires sentence-transformers. Install it with "
                    "`pip install sentence-transformers`, or select TF-IDF."
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    "Could not load the embedding model. Check internet access/model "
                    "availability, or use TF-IDF mode."
                ) from exc
        return self._model

    def build_embeddings(self, batch_size=32):
        if self._embeddings is None:
            model = self._load_model()
            self._embeddings = model.encode(
                self.texts, batch_size=batch_size, show_progress_bar=False,
                normalize_embeddings=True
            )
        return self._embeddings

    def recommend(self, query, method="hybrid", top_k=5, alpha=0.5,
                  exclude_paper_id=None):
        query = str(query).strip()
        if not query:
            raise ValueError("Enter a search query or choose a paper.")
        top_k = max(1, int(top_k))
        tfidf_scores = cosine_similarity(
            self.vectorizer.transform([query]), self.tfidf_matrix
        ).ravel()

        semantic_scores = None
        if method in ("semantic", "hybrid"):
            model = self._load_model()
            self.build_embeddings()
            query_embedding = model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0]
            semantic_scores = np.asarray(self._embeddings @ query_embedding).ravel()
            # Cosine similarity can be negative. Convert both methods to [0, 1]
            # for a simple interpretable weighted ranking; this is a ranking
            # transformation, not probability calibration.
            semantic_scores = np.clip((semantic_scores + 1.0) / 2.0, 0.0, 1.0)

        tfidf_scores = np.clip(tfidf_scores, 0.0, 1.0)
        if method == "tfidf":
            final_scores = tfidf_scores
        elif method == "semantic":
            final_scores = semantic_scores
        elif method == "hybrid":
            alpha = float(np.clip(alpha, 0.0, 1.0))
            final_scores = alpha * tfidf_scores + (1.0 - alpha) * semantic_scores
        else:
            raise ValueError("method must be 'tfidf', 'semantic', or 'hybrid'.")

        results = self.papers.copy()
        results["tfidf_score"] = tfidf_scores
        results["semantic_score"] = semantic_scores if semantic_scores is not None else np.nan
        results["score"] = final_scores
        if exclude_paper_id is not None:
            results = results[results["paper_id"].astype(str) != str(exclude_paper_id)]
        results = results.sort_values(
            ["score", "title"], ascending=[False, True], kind="mergesort"
        ).head(top_k).reset_index(drop=True)
        return results

    def recommend_for_paper(self, paper_id, method="hybrid", top_k=5, alpha=0.5):
        matches = self.papers.index[self.papers["paper_id"].astype(str) == str(paper_id)].tolist()
        if not matches:
            raise ValueError(f"Paper ID {paper_id!r} was not found.")
        idx = matches[0]
        query = self.texts[idx]
        return self.recommend(
            query, method=method, top_k=top_k, alpha=alpha,
            exclude_paper_id=paper_id
        )
