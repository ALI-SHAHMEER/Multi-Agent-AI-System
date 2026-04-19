"""
════════════════════════════════════════════════════════════════════════
PART F  –  LangGraph Workflow
State-based graph with conditional edges and retry logic.

Graph nodes:
  START → plan_node → retrieve_node → [web_node?] → synthesise_node → END

Conditional edges:
  • After retrieve: if confidence < 6 → force web_node
  • After retrieve: if skip_web flag → skip web_node
  • After synthesise: if quality too low → loop back to web_node (once)
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import json
from typing import TypedDict, Optional, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory

from part_e_agents import (
    PlannerAgent,
    RetrievalAgent,
    WebResearchAgent,
    SynthesisAgent,
)
from config import NEWS_API_KEY


# ─── Shared state schema (passed between every node) ─────────────────────────
class ResearchState(TypedDict):
    """
    The central state object that flows through the LangGraph.
    Every node reads from and writes to this dict.
    """
    # Input
    query: str

    # Planner output
    plan: Optional[Dict[str, Any]]

    # Retrieval agent output
    retrieval_result: Optional[Dict[str, Any]]

    # Web research agent output
    web_result: Optional[Dict[str, Any]]

    # Synthesis agent output
    final_answer: Optional[str]

    # Control flags
    skip_web: bool
    web_retry_count: int          # prevents infinite loops
    conversation_history: List[str]   # for multi-turn memory


# ─── Node functions  ─────────────────────────────────────────────────────────
# Each node receives the full state dict and returns a partial update.

planner = PlannerAgent()
retriever = RetrievalAgent()          # attach vectorstore after building it
web_researcher = WebResearchAgent(news_api_key=NEWS_API_KEY)
synthesiser = SynthesisAgent()


def plan_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node 1: Planner.
    Decomposes the query and sets the skip_web flag.
    """
    print("[Node: plan_node]")
    plan = planner.plan(state["query"])
    return {
        "plan": plan,
        "skip_web": plan.get("skip_web", False),
    }


def retrieve_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node 2: Local knowledge retrieval.
    Searches the FAISS vector store and returns findings + confidence score.
    """
    print("[Node: retrieve_node]")
    plan = state.get("plan", {})
    task = next(
        (t["task"] for t in plan.get("subtasks", []) if t["agent"] == "retrieval"),
        state["query"],
    )
    result = retriever.retrieve(task, state["query"])
    return {"retrieval_result": result}


def web_research_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node 3: Web / API research.
    Runs only when routed here by the conditional edge.
    """
    print("[Node: web_research_node]")
    plan = state.get("plan", {})
    task = next(
        (t["task"] for t in plan.get("subtasks", []) if t["agent"] == "web_research"),
        state["query"],
    )
    result = web_researcher.research(task, state["query"])
    return {
        "web_result": result,
        "web_retry_count": state.get("web_retry_count", 0) + 1,
    }


def synthesise_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node 4: Synthesis.
    Merges retrieval + web results into a final cited answer.
    """
    print("[Node: synthesise_node]")
    answer = synthesiser.synthesise(
        query=state["query"],
        retrieval_result=state.get("retrieval_result") or {},
        web_result=state.get("web_result") or {},
    )
    history = state.get("conversation_history", [])
    history.append(f"Q: {state['query']}\nA: {answer[:200]}…")
    return {
        "final_answer": answer,
        "conversation_history": history,
    }


# ─── Conditional edge functions  ─────────────────────────────────────────────
def route_after_retrieve(state: ResearchState) -> str:
    """
    Decision after retrieve_node:
    - If planner flagged skip_web → go straight to synthesis
    - If retrieval confidence is low (< 6) → go to web research for enrichment
    - Otherwise → go to web research normally
    """
    if state.get("skip_web", False):
        print("[Router] skip_web=True → synthesise_node")
        return "synthesise"

    confidence = (state.get("retrieval_result") or {}).get("confidence", 5)
    if confidence < 6:
        print(f"[Router] confidence={confidence} < 6 → web_research_node (forced)")
    else:
        print(f"[Router] confidence={confidence} ≥ 6 → web_research_node (normal)")
    return "web_research"


def route_after_synthesis(state: ResearchState) -> str:
    """
    Decision after synthesise_node:
    - If the answer seems very short and we haven't retried web yet → retry
    - Otherwise → END
    """
    answer = state.get("final_answer", "")
    retries = state.get("web_retry_count", 0)

    if len(answer) < 200 and retries < 1:
        print("[Router] Answer too short → web_research_node retry")
        return "web_research"

    print("[Router] Answer acceptable → END")
    return "end"


# ─── Build the LangGraph  ─────────────────────────────────────────────────────
def build_research_graph() -> StateGraph:
    """
    Assembles the state machine:

    START → plan → retrieve ─┬─(skip_web or high confidence + skip)──→ synthesise
                             └─(low confidence or normal)──→ web_research → synthesise
    synthesise ─┬─(short answer, first retry)──→ web_research
               └─(acceptable)──────────────────→ END
    """
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("plan_node", plan_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("web_research_node", web_research_node)
    graph.add_node("synthesise_node", synthesise_node)

    # Entry point
    graph.set_entry_point("plan_node")

    # Fixed edges
    graph.add_edge("plan_node", "retrieve_node")
    graph.add_edge("web_research_node", "synthesise_node")

    # Conditional edge after retrieve
    graph.add_conditional_edges(
        "retrieve_node",
        route_after_retrieve,
        {
            "synthesise": "synthesise_node",
            "web_research": "web_research_node",
        },
    )

    # Conditional edge after synthesis
    graph.add_conditional_edges(
        "synthesise_node",
        route_after_synthesis,
        {
            "web_research": "web_research_node",
            "end": END,
        },
    )

    return graph


# ─── Runner  ──────────────────────────────────────────────────────────────────
class ResearchGraphRunner:
    """
    Wraps the compiled LangGraph with multi-turn conversation support.
    The conversation_history is threaded through every graph invocation.
    """

    def __init__(self, vectorstore=None):
        # Attach the vectorstore to the retrieval node's agent
        retriever.vectorstore = vectorstore
        self.app = build_research_graph().compile()
        self.history: List[str] = []
        self.memory = ConversationBufferMemory(return_messages=True)

    def run(self, query: str) -> str:
        initial_state: ResearchState = {
            "query": query,
            "plan": None,
            "retrieval_result": None,
            "web_result": None,
            "final_answer": None,
            "skip_web": False,
            "web_retry_count": 0,
            "conversation_history": self.history.copy(),
        }
        final_state = self.app.invoke(initial_state)

        answer = final_state.get("final_answer", "No answer generated.")
        self.history = final_state.get("conversation_history", [])

        # Save to LangChain memory as well (for integration with Part C)
        self.memory.save_context({"input": query}, {"output": answer})
        return answer

    def display_graph(self):
        """Print the Mermaid diagram of the compiled graph."""
        try:
            print(self.app.get_graph().draw_mermaid())
        except Exception as e:
            print(f"Could not draw graph: {e}")


if __name__ == "__main__":
    runner = ResearchGraphRunner()

    print("=== ResearchMind LangGraph Workflow (Part F) ===\n")
    runner.display_graph()

    queries = [
        "Explain the attention mechanism in transformers",
        "What is RAG and how is it used in production?",
    ]
    for q in queries:
        answer = runner.run(q)
        print(f"\nQ: {q}\nA: {answer[:400]}…\n{'─'*60}\n")
