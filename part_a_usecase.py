"""
════════════════════════════════════════════════════════════════════════
PART A  –  Use Case Definition
════════════════════════════════════════════════════════════════════════

PROJECT TITLE
─────────────
ResearchMind – Multi-Agent Academic Research Assistant

PROBLEM STATEMENT
─────────────────
Researchers (graduate students, academics, analysts) spend 60-80 % of
their time on literature search, paper summarisation, and cross-source
synthesis rather than on their own original thinking.  ResearchMind
automates the low-value parts of that workflow.

END USER
────────
• Graduate students writing theses / literature reviews
• Academics exploring a new research area
• Corporate R&D analysts doing technology scouting

WHAT EACH AGENT DOES
────────────────────
1. Planner Agent
   • Receives the raw user query
   • Decomposes it into 2-4 concrete sub-tasks
   • Decides which specialist agents to invoke and in what order
   • Maintains the overall research plan in the LangGraph state

2. Retrieval / Knowledge Agent
   • Searches the local vector store (FAISS) built from uploaded PDFs
   • Retrieves the top-k most relevant document chunks
   • Enriches results with metadata (title, author, page)

3. Web Research Agent
   • Calls Semantic Scholar API to find recent academic papers
   • Calls News API to surface real-world news around the topic
   • Returns structured results (title, abstract, URL, published date)

4. Synthesis / Response Agent
   • Receives all sub-results from the other agents
   • Produces a single coherent answer with inline citations
   • Offers follow-up question suggestions

MCP SERVERS
───────────
• Filesystem MCP  – reads/writes local PDF files and research notes
• Google Drive MCP – accesses cloud-stored papers and collaborative docs

EXTERNAL APIs
─────────────
• Semantic Scholar Graph API  – free academic paper search (no key needed)
• News API                    – recent news articles on the research topic

DATA SOURCES
────────────
• Local  : PDFs stored in ./data/ (loaded into FAISS vector DB)
• Remote : Semantic Scholar + Google Drive + News API live results

HOW LANGGRAPH CONTROLS THE WORKFLOW
────────────────────────────────────
State machine nodes:

  START → plan → retrieve → web_research → synthesise → END

  Branching:
  • After "plan": if query is purely factual → skip web_research
  • After "retrieve": if local docs insufficient → force web_research
  • After "synthesise": if confidence is low → loop back to web_research
════════════════════════════════════════════════════════════════════════
"""

USE_CASE = {
    "title": "ResearchMind – Multi-Agent Academic Research Assistant",
    "problem": "Automate literature search, summarisation, and synthesis",
    "end_user": "Graduate students, academics, corporate R&D analysts",
    "agents": ["Planner", "Retrieval", "WebResearch", "Synthesis"],
    "mcp_servers": ["Filesystem", "Google Drive"],
    "apis": ["Semantic Scholar", "News API"],
    "data_sources": ["Local PDFs (FAISS)", "Google Drive / Remote APIs"],
}

if __name__ == "__main__":
    import json
    print(json.dumps(USE_CASE, indent=2))
