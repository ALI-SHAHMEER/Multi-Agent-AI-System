"""
════════════════════════════════════════════════════════════════════════
PART H  –  External API Integration
API 1: Semantic Scholar Graph API – academic paper search (free, no key)
API 2: News API – recent news articles (free tier with key)

Each API is wrapped in a class with:
  • Rate-limit awareness
  • Structured output (dict / list)
  • Error handling
  • LangChain tool wrapper for agent use
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import time
import requests
from typing import Any, Dict, List, Optional
from langchain.tools import tool
from config import NEWS_API_KEY, SEMANTIC_SCHOLAR_BASE


# ═══════════════════════════════════════════════════════════════════════════════
# API 1  –  Semantic Scholar Graph API
# Docs: https://api.semanticscholar.org/graph/v1
# ═══════════════════════════════════════════════════════════════════════════════
class SemanticScholarAPI:
    """
    Wraps the Semantic Scholar Graph API.
    Rate limit: 100 requests/5 min on the free unauthenticated tier.
    """

    BASE = SEMANTIC_SCHOLAR_BASE
    FIELDS = "title,year,abstract,url,authors,citationCount,publicationTypes,venue"

    def __init__(self, requests_per_second: float = 0.5):
        self._last_call = 0.0
        self._min_interval = 1.0 / requests_per_second

    def _wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    # ── 1a. Paper search  ────────────────────────────────────────────────────
    def search_papers(
        self,
        query: str,
        limit: int = 10,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for academic papers by keyword/phrase.
        Returns a list of structured paper dicts.
        """
        self._wait()
        params: Dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fields": self.FIELDS,
        }
        if year_from:
            params["publicationDateOrYear"] = f"{year_from}-"
        if year_to:
            params["publicationDateOrYear"] = f"-{year_to}"
        if year_from and year_to:
            params["publicationDateOrYear"] = f"{year_from}:{year_to}"

        try:
            resp = requests.get(
                f"{self.BASE}/paper/search",
                params=params,
                timeout=15,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            return [self._format_paper(p) for p in data.get("data", [])]

        except requests.exceptions.RequestException as e:
            return [{"error": str(e), "query": query}]

    def _format_paper(self, p: Dict) -> Dict:
        return {
            "title":          p.get("title", ""),
            "year":           p.get("year"),
            "abstract":       (p.get("abstract") or "")[:500],
            "url":            p.get("url", ""),
            "authors":        [a.get("name", "") for a in p.get("authors", [])[:5]],
            "citation_count": p.get("citationCount", 0),
            "venue":          p.get("venue", ""),
            "pub_types":      p.get("publicationTypes", []),
        }

    # ── 1b. Paper recommendations  ──────────────────────────────────────────
    def get_recommendations(self, paper_id: str, limit: int = 5) -> List[Dict]:
        """Get papers related to a specific Semantic Scholar paper ID."""
        self._wait()
        try:
            resp = requests.get(
                f"{self.BASE}/paper/{paper_id}/recommendations",
                params={"limit": limit, "fields": "title,year,abstract,url"},
                timeout=10,
            )
            resp.raise_for_status()
            return [self._format_paper(p) for p in resp.json().get("recommendedPapers", [])]
        except Exception as e:
            return [{"error": str(e)}]

    # ── 1c. Author lookup  ──────────────────────────────────────────────────
    def search_author(self, author_name: str) -> List[Dict]:
        """Find an author and their top papers."""
        self._wait()
        try:
            resp = requests.get(
                f"{self.BASE}/author/search",
                params={"query": author_name, "limit": 3, "fields": "name,hIndex,paperCount"},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            return [{"error": str(e)}]


# ═══════════════════════════════════════════════════════════════════════════════
# API 2  –  News API
# Docs: https://newsapi.org/docs
# ═══════════════════════════════════════════════════════════════════════════════
class NewsAPI:
    """
    Wraps the News API for surfacing recent news about a research topic.
    Requires NEWS_API_KEY (free tier allows 100 requests/day).
    """

    BASE = "https://newsapi.org/v2"

    def __init__(self, api_key: str = NEWS_API_KEY):
        self.api_key = api_key

    def _is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_newsapi_key_here")

    def search_everything(
        self,
        query: str,
        language: str = "en",
        sort_by: str = "relevancy",
        page_size: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search all indexed articles for a query string.
        sort_by: relevancy | popularity | publishedAt
        """
        if not self._is_configured():
            return self._mock_news(query)

        try:
            resp = requests.get(
                f"{self.BASE}/everything",
                params={
                    "q":        query,
                    "language": language,
                    "sortBy":   sort_by,
                    "pageSize": page_size,
                    "apiKey":   self.api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            return [self._format_article(a) for a in articles]

        except requests.exceptions.RequestException as e:
            return [{"error": str(e)}]

    def top_headlines(self, query: str, page_size: int = 5) -> List[Dict]:
        """Fetch breaking news headlines matching the topic."""
        if not self._is_configured():
            return self._mock_news(query)

        try:
            resp = requests.get(
                f"{self.BASE}/top-headlines",
                params={
                    "q":        query,
                    "pageSize": page_size,
                    "apiKey":   self.api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            return [self._format_article(a) for a in resp.json().get("articles", [])]
        except Exception as e:
            return [{"error": str(e)}]

    def _format_article(self, a: Dict) -> Dict:
        return {
            "title":       a.get("title", ""),
            "source":      a.get("source", {}).get("name", ""),
            "author":      a.get("author", ""),
            "description": (a.get("description") or "")[:300],
            "url":         a.get("url", ""),
            "published":   a.get("publishedAt", ""),
        }

    def _mock_news(self, query: str) -> List[Dict]:
        """Returned when no API key is configured, so demos still work."""
        return [
            {
                "title":       f"New developments in {query}",
                "source":      "TechCrunch (simulated)",
                "description": f"Researchers have made significant progress in {query}, "
                               "with multiple papers published this month.",
                "url":         "https://example.com",
                "published":   "2024-11-15T10:00:00Z",
            }
        ]


# ─── LangChain @tool wrappers (agents can call these as tools)  ───────────────
_ss_api = SemanticScholarAPI()
_news_api = NewsAPI()


@tool
def search_academic_papers(query: str) -> str:
    """
    Search Semantic Scholar for academic papers on a research topic.
    Input: a research query string.
    Returns: formatted list of paper titles, authors, year and abstract snippet.
    """
    papers = _ss_api.search_papers(query, limit=5)
    lines = []
    for i, p in enumerate(papers, 1):
        if "error" in p:
            lines.append(f"{i}. Error: {p['error']}")
        else:
            authors = ", ".join(p["authors"][:3])
            lines.append(
                f"{i}. {p['title']} ({p['year']})\n"
                f"   Authors: {authors}\n"
                f"   Citations: {p['citation_count']}\n"
                f"   Abstract: {p['abstract'][:200]}…\n"
                f"   URL: {p['url']}"
            )
    return "\n\n".join(lines) if lines else "No papers found."


@tool
def search_recent_news(query: str) -> str:
    """
    Search for recent news articles about a research topic.
    Input: a topic or keyword string.
    Returns: formatted list of article titles, sources and descriptions.
    """
    articles = _news_api.search_everything(query, page_size=5)
    lines = []
    for i, a in enumerate(articles, 1):
        if "error" in a:
            lines.append(f"{i}. Error: {a['error']}")
        else:
            lines.append(
                f"{i}. {a['title']}\n"
                f"   Source: {a['source']} | {a['published'][:10]}\n"
                f"   {a['description']}\n"
                f"   URL: {a['url']}"
            )
    return "\n\n".join(lines) if lines else "No articles found."


# ─── APILayer: unified interface used by agents  ──────────────────────────────
class APILayer:
    """
    Single entry point for all external APIs.
    Wraps SemanticScholar and NewsAPI with error handling.
    """

    def __init__(self):
        self.scholar = SemanticScholarAPI()
        self.news = NewsAPI()

    def search_papers(self, query: str, limit: int = 5) -> Dict[str, Any]:
        papers = self.scholar.search_papers(query, limit=limit)
        return {
            "source": "Semantic Scholar",
            "query":  query,
            "papers": papers,
            "count":  len(papers),
        }

    def search_news(self, query: str, limit: int = 5) -> Dict[str, Any]:
        articles = self.news.search_everything(query, page_size=limit)
        return {
            "source":   "News API",
            "query":    query,
            "articles": articles,
            "count":    len(articles),
        }

    def full_search(self, query: str) -> Dict[str, Any]:
        """Run both APIs and return a combined structured result."""
        return {
            "papers": self.search_papers(query)["papers"],
            "news":   self.search_news(query)["articles"],
        }

    def get_langchain_tools(self) -> List:
        """Return the @tool-wrapped functions for use in LangChain agents."""
        return [search_academic_papers, search_recent_news]


# ─── Demo  ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    api = APILayer()
    topic = "large language models retrieval augmented generation"

    print("=== Semantic Scholar ===")
    papers = api.search_papers(topic, limit=3)
    print(json.dumps(papers, indent=2))

    print("\n=== News API ===")
    news = api.search_news("LLM hallucination research", limit=3)
    print(json.dumps(news, indent=2))

    print("\n=== LangChain Tool: search_academic_papers ===")
    print(search_academic_papers.run("attention mechanism transformer"))
