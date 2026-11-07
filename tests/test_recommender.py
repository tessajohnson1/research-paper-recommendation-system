import pandas as pd
from recommender import PaperRecommender

def test_tfidf_recommender_returns_results():
    papers = pd.DataFrame([
        {"paper_id":"A", "title":"Fake news detection", "abstract":"Detect misinformation using NLP"},
        {"paper_id":"B", "title":"Sentiment analysis", "abstract":"Classify positive and negative text"},
        {"paper_id":"C", "title":"Misinformation detection", "abstract":"Detect fake news in social media"},
    ])
    model = PaperRecommender(papers)
    result = model.recommend("fake news detection", method="tfidf", top_k=2)
    assert len(result) == 2
    assert result.iloc[0]["paper_id"] in {"A", "C"}

def test_similar_paper_excludes_seed():
    papers = pd.DataFrame([
        {"paper_id":"A", "title":"Fake news detection", "abstract":"Detect misinformation"},
        {"paper_id":"B", "title":"Misinformation detection", "abstract":"Detect fake news"},
        {"paper_id":"C", "title":"Sentiment analysis", "abstract":"Analyze opinion"},
    ])
    model = PaperRecommender(papers)
    result = model.recommend_for_paper("A", method="tfidf", top_k=2)
    assert "A" not in result["paper_id"].tolist()
