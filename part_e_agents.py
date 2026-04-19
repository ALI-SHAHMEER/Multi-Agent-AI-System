"""
════════════════════════════════════════════════════════════════════════
PART E  –  Multi-Agent Design
Four specialist agents, each with a focused role and system prompt.
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from config import OPENAI_API_KEY, LLM_MODEL


# ─── Shared LLM factory  ─────────────────────────────────────────────────────
def _llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 – Planner
# Decomposes the user query into a structured research plan.
# ═══════════════════════════════════════════════════════════════════════════════
PLANNER_SYSTEM = """You are the Planner agent in a multi-agent research system.
Your job is to decompose the user's research query into clear, actionable subtasks.

Output a JSON object with exactly this structure:
{
  "main_topic": "<topic in ≤10 words>",
  "subtasks": [
    {"id": 1, "agent": "retrieval", "task": "<what to search in local docs>"},
    {"id": 2, "agent": "web_research", "task": "<what to search online/APIs>"},
    {"id": 3, "agent": "synthesis", "task": "<how to combine the findings>"}
  ],
  "query_type": "factual|exploratory|comparative",
  "skip_web": false
}

Rules:
- Set skip_web=true only for purely definitional queries that need no recent data.
- Be specific in each task description.
- Output ONLY valid JSON, no markdown fences."""


class PlannerAgent:
    """Breaks a research query into a structured multi-agent plan."""

    def __init__(self):
        self.llm = _llm(temperature=0.1)

    def plan(self, query: str) -> Dict[str, Any]:
        messages = [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=f"Research query: {query}"),
        ]
        response = self.llm(messages)
        try:
            plan = json.loads(response.content)
        except json.JSONDecodeError:
            # Graceful fallback if LLM adds prose around the JSON
            import re
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            plan = json.loads(match.group()) if match else {
                "main_topic": query[:40],
                "subtasks": [],
                "query_type": "exploratory",
                "skip_web": False,
            }
        return plan


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2 – Retrieval Agent
# Searches the local FAISS vector store for relevant document chunks.
# ═══════════════════════════════════════════════════════════════════════════════
RETRIEVAL_SYSTEM = """You are the Retrieval agent in a multi-agent research system.
You receive a search task and a set of retrieved document chunks.
Your job is to:
1. Identify the most relevant information from the chunks.
2. Summarise it clearly, citing source filenames.
3. Rate your confidence (0-10) that the local knowledge base has good coverage.

Output format:
{
  "summary": "<concise summary>",
  "key_findings": ["<finding 1>", "<finding 2>", ...],
  "sources": ["<filename>", ...],
  "confidence": <0-10>
}
Output ONLY valid JSON."""


class RetrievalAgent:
    """Searches the vector store and summarises the top results."""

    def __init__(self, vectorstore=None):
        self.llm = _llm(temperature=0.2)
        self.vectorstore = vectorstore

    def retrieve(self, task: str, query: str) -> Dict[str, Any]:
        # If a vectorstore is available, do real similarity search
        raw_chunks = []
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(query, k=5)
            raw_chunks = [
                {"content": d.page_content, "source": d.metadata.get("source", "?")}
                for d in docs
            ]
        else:
            raw_chunks = [{"content": "No vector store connected.", "source": "n/a"}]

        context = "\n\n".join(
            f"[Source: {c['source']}]\n{c['content']}" for c in raw_chunks
        )

        messages = [
            SystemMessage(content=RETRIEVAL_SYSTEM),
            HumanMessage(content=f"Task: {task}\n\nRetrieved chunks:\n{context}"),
        ]
        response = self.llm(messages)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"summary": response.content, "key_findings": [], "sources": [], "confidence": 5}


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3 – Web Research Agent
# Calls external APIs (Semantic Scholar + News) and returns structured results.
# ═══════════════════════════════════════════════════════════════════════════════
WEB_RESEARCH_SYSTEM = """You are the Web Research agent in a multi-agent research system.
You receive external search results (academic papers and news articles).
Your job is to extract the most relevant information and produce a structured summary.

Output format:
{
  "papers": [
    {"title": "...", "year": 2024, "abstract_snippet": "...", "url": "..."}
  ],
  "news": [
    {"title": "...", "source": "...", "published": "...", "snippet": "..."}
  ],
  "web_summary": "<2-3 sentence synthesis of what you found>"
}
Output ONLY valid JSON."""


class WebResearchAgent:
    """Fetches live results from Semantic Scholar and News API."""

    def __init__(self, news_api_key: str = ""):
        self.llm = _llm(temperature=0.2)
        self.news_api_key = news_api_key

    def search_semantic_scholar(self, query: str, limit: int = 5) -> List[Dict]:
        """Free academic paper search – no API key required."""
        import requests
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,year,abstract,url,authors",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            papers = resp.json().get("data", [])
            return [
                {
                    "title": p.get("title", ""),
                    "year": p.get("year"),
                    "abstract_snippet": (p.get("abstract") or "")[:300],
                    "url": p.get("url", ""),
                    "authors": [a["name"] for a in p.get("authors", [])[:3]],
                }
                for p in papers
            ]
        except Exception as e:
            return [{"error": str(e)}]

    def search_news(self, query: str, limit: int = 5) -> List[Dict]:
        """News API search – requires NEWS_API_KEY."""
        import requests
        if not self.news_api_key:
            return [{"info": "NEWS_API_KEY not set. Set it in .env to enable news search."}]
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "pageSize": limit,
            "sortBy": "relevancy",
            "apiKey": self.news_api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "published": a.get("publishedAt", ""),
                    "snippet": (a.get("description") or "")[:200],
                    "url": a.get("url", ""),
                }
                for a in articles
            ]
        except Exception as e:
            return [{"error": str(e)}]

    def research(self, task: str, query: str) -> Dict[str, Any]:
        papers = self.search_semantic_scholar(query)
        news = self.search_news(query)

        raw_results = json.dumps({"papers": papers, "news": news}, indent=2)[:3000]

        messages = [
            SystemMessage(content=WEB_RESEARCH_SYSTEM),
            HumanMessage(content=f"Task: {task}\n\nRaw results:\n{raw_results}"),
        ]
        response = self.llm(messages)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"papers": papers, "news": news, "web_summary": response.content}


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 4 – Synthesis Agent
# Combines outputs from all agents into a single, cited response.
# ═══════════════════════════════════════════════════════════════════════════════
SYNTHESIS_SYSTEM = """You are the Synthesis agent in a multi-agent research system.
You receive:
- A user's original research query
- Summaries from the Retrieval agent (local documents)
- Summaries from the Web Research agent (papers + news)

Your job is to produce a final, comprehensive, well-structured answer that:
1. Directly addresses the user's query
2. Integrates findings from both local and web sources
3. Uses inline citations like [Local: filename] or [Paper: title]
4. Ends with 3 follow-up questions the user might want to explore

Be scholarly but accessible. Aim for 3-5 paragraphs."""


class SynthesisAgent:
    """Merges all agent outputs into the final answer delivered to the user."""

    def __init__(self):
        self.llm = _llm(temperature=0.4)

    def synthesise(
        self,
        query: str,
        retrieval_result: Dict[str, Any],
        web_result: Dict[str, Any],
    ) -> str:
        context = f"""QUERY: {query}

LOCAL KNOWLEDGE (Retrieval Agent):
{json.dumps(retrieval_result, indent=2)}

WEB / API KNOWLEDGE (Web Research Agent):
{json.dumps(web_result, indent=2)}"""

        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM),
            HumanMessage(content=context),
        ]
        response = self.llm(messages)
        return response.content


# ─── Coordinator: runs all agents in sequence  ────────────────────────────────
class ResearchCoordinator:
    """Manually orchestrates the four agents (no LangGraph yet)."""

    def __init__(self, vectorstore=None, news_api_key: str = ""):
        self.planner = PlannerAgent()
        self.retriever = RetrievalAgent(vectorstore=vectorstore)
        self.web_researcher = WebResearchAgent(news_api_key=news_api_key)
        self.synthesiser = SynthesisAgent()

    def run(self, query: str, verbose: bool = True) -> str:
        if verbose:
            print(f"\n{'═'*60}\nQUERY: {query}\n{'═'*60}")

        # Step 1: Plan
        plan = self.planner.plan(query)
        if verbose:
            print(f"\n[Planner] Topic: {plan.get('main_topic')}")
            print(f"  Query type: {plan.get('query_type')}")
            print(f"  Skip web: {plan.get('skip_web')}")

        # Step 2: Retrieve from local docs
        retrieval_task = next(
            (t["task"] for t in plan.get("subtasks", []) if t["agent"] == "retrieval"),
            query,
        )
        retrieval_result = self.retriever.retrieve(retrieval_task, query)
        if verbose:
            print(f"\n[Retrieval] Confidence: {retrieval_result.get('confidence')}/10")
            print(f"  Sources: {retrieval_result.get('sources', [])}")

        # Step 3: Web research (unless planner said to skip)
        web_result: Dict[str, Any] = {}
        if not plan.get("skip_web", False):
            web_task = next(
                (t["task"] for t in plan.get("subtasks", []) if t["agent"] == "web_research"),
                query,
            )
            web_result = self.web_researcher.research(web_task, query)
            if verbose:
                papers = web_result.get("papers", [])
                print(f"\n[Web Research] Found {len(papers)} papers")

        # Step 4: Synthesise
        final_answer = self.synthesiser.synthesise(query, retrieval_result, web_result)
        if verbose:
            print(f"\n{'─'*60}\nFINAL ANSWER:\n{final_answer}\n{'─'*60}\n")

        return final_answer


if __name__ == "__main__":
    from config import NEWS_API_KEY
    coordinator = ResearchCoordinator(news_api_key=NEWS_API_KEY)
    coordinator.run("What are the latest advances in transformer architectures for NLP?")
