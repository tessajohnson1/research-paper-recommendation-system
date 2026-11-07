from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["title"]
OPTIONAL_DEFAULTS = {
    "paper_id": "",
    "abstract": "",
    "authors": "",
    "year": "",
    "keywords": "",
    "url": "",
}

def load_papers(source):
    """Load a CSV path or file-like object and normalize its schema."""
    df = pd.read_csv(source)
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        "paper title": "title", "name": "title",
        "summary": "abstract", "author": "authors",
        "publication_year": "year", "doi_url": "url",
        "link": "url", "keyword": "keywords",
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("CSV must contain a 'title' column.")
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    df = df.fillna("")
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"] != ""].copy()
    if df["paper_id"].astype(str).str.strip().eq("").any():
        generated = [f"P{i:05d}" for i in range(1, len(df) + 1)]
        df["paper_id"] = [
            str(value).strip() if str(value).strip() else generated[i]
            for i, value in enumerate(df["paper_id"].tolist())
        ]
    df["paper_id"] = df["paper_id"].astype(str)
    df = df.drop_duplicates(subset=["paper_id"], keep="first").reset_index(drop=True)
    for col in ["abstract", "authors", "year", "keywords", "url"]:
        df[col] = df[col].astype(str)
    return df

def paper_text(row):
    """Create text for ranking. Field labels give title/keywords useful emphasis."""
    parts = []
    title = str(row.get("title", "")).strip()
    keywords = str(row.get("keywords", "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    if title:
        parts.extend([title, title])  # modest title emphasis for lexical matching
    if keywords:
        parts.extend([keywords, keywords])
    if abstract:
        parts.append(abstract)
    return " ".join(parts)
