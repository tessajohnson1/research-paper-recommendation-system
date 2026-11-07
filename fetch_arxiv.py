import argparse
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET

API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}

def main():
    parser = argparse.ArgumentParser(description="Fetch arXiv paper metadata for a project dataset.")
    parser.add_argument("--query", default="computational linguistics")
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--output", default="data/arxiv_papers.csv")
    args = parser.parse_args()
    params = {
        "search_query": f'all:"{args.query}"',
        "start": 0,
        "max_results": min(max(args.max_results, 1), 1000),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = requests.get(API, params=params, timeout=45,
                            headers={"User-Agent": "MTechPaperRecommender/1.0"})
    response.raise_for_status()
    root = ET.fromstring(response.text)
    rows = []
    for entry in root.findall("atom:entry", NS):
        paper_url = (entry.findtext("atom:id", default="", namespaces=NS) or "").strip()
        title = " ".join((entry.findtext("atom:title", default="", namespaces=NS) or "").split())
        abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=NS) or "").split())
        authors = "; ".join(a.findtext("atom:name", default="", namespaces=NS) or ""
                            for a in entry.findall("atom:author", NS))
        published = entry.findtext("atom:published", default="", namespaces=NS) or ""
        year = published[:4]
        categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", NS)]
        rows.append({
            "paper_id": paper_url.rsplit("/", 1)[-1],
            "title": title, "abstract": abstract, "authors": authors,
            "year": year, "keywords": "; ".join(categories), "url": paper_url
        })
        time.sleep(0.05)
    if not rows:
        raise SystemExit("No results found. Try a broader query.")
    out = pd.DataFrame(rows)
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} papers to {args.output}")
    print("Record the query/date and verify API terms and metadata before using it in your report.")

if __name__ == "__main__":
    main()
