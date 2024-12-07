# filename: collect_gpt_papers.py

import requests
import feedparser
from typing import List, Dict

def search_arxiv(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=all:{query}"
    start = 0
    max_results = f"max_results={max_results}"
    url = f"{base_url}{search_query}&start={start}&{max_results}"
    response = requests.get(url)
    feed = feedparser.parse(response.content)

    papers = [{"title": entry.title, "link": entry.link, "summary": entry.summary} for entry in feed.entries]
    return papers

# Collect papers related to GPT models
gpt_papers = search_arxiv("GPT", max_results=50)
for paper in gpt_papers:
    print(f"Title: {paper['title']}\nSummary: {paper['summary']}\n")