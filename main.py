"""
════════════════════════════════════════════════════════════════════════
main.py  –  Full Integrated Research Assistant
Combines all 8 parts into a single, runnable application.

Run:
    python main.py

Or run individual parts:
    python part_b_basic_chatbot.py
    python part_c_memory.py
    ...
════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_openai import ChatOpenAI

# ── Internal modules ──────────────────────────────────────────────────────────
from config import OPENAI_API_KEY, LLM_MODEL, NEWS_API_KEY, FILESYSTEM_BASE_PATH
from part_d_retrieval import setup_rag_pipeline
from part_e_agents import RetrievalAgent, WebResearchAgent, SynthesisAgent, PlannerAgent
from part_f_langgraph import ResearchGraphRunner, build_research_graph
from part_g_mcp import MCPLayer, FilesystemMCPServer, GoogleDriveMCPServer
from part_h_apis import APILayer


# ─── Full System  ─────────────────────────────────────────────────────────────
class ResearchAssistant:
    """
    The complete Research Assistant wiring all components:

    ┌─────────────────────────────────────────────────────────┐
    │  User query                                             │
    │       │                                                 │
    │  ┌────▼─────┐    LangGraph orchestrates:               │
    │  │  Planner │ → Retrieval → [Web Research] → Synthesis │
    │  └──────────┘                                          │
    │       │                                                 │
    │  Memory (Buffer + Summary)                              │
    │  MCP (Filesystem + Google Drive)                        │
    │  APIs (Semantic Scholar + News)                         │
    │  RAG  (FAISS vector store)                              │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, data_dir: str = FILESYSTEM_BASE_PATH):
        print("Initialising ResearchMind…")

        # Part D: RAG pipeline (FAISS vector store)
        print("  [1/5] Building RAG pipeline…")
        self.rag_chain, self.vectorstore = setup_rag_pipeline(data_dir)

        # Part C: Dual memory strategy
        print("  [2/5] Setting up memory…")
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
        self.buffer_memory = ConversationBufferMemory(return_messages=True)
        self.summary_memory = ConversationSummaryMemory(llm=llm, return_messages=True)

        # Part G: MCP servers
        print("  [3/5] Connecting MCP servers…")
        self.mcp = MCPLayer()

        # Part H: External APIs
        print("  [4/5] Connecting external APIs…")
        self.apis = APILayer()

        # Part F: LangGraph runner (includes Part E agents internally)
        print("  [5/5] Compiling LangGraph workflow…")
        self.graph_runner = ResearchGraphRunner(vectorstore=self.vectorstore)

        print("Ready!\n")

    # ── Primary entry point for every user query ──────────────────────────────
    def ask(self, query: str) -> str:
        """
        Full pipeline:
        1. Save query to conversation memory
        2. Check MCP filesystem for cached answers
        3. Run the LangGraph workflow (plan → retrieve → web → synthesise)
        4. Cache the answer via MCP filesystem
        5. Save answer to both memory types
        6. Return the final answer
        """
        print(f"\nProcessing: {query}")

        # Step 1: Memory context – load previous conversation
        mem_vars = self.buffer_memory.load_memory_variables({})
        history = mem_vars.get("history", [])
        history_text = "\n".join(str(m) for m in history[-4:])  # last 2 turns
        if history_text:
            enriched_query = f"{query}\n\n[Conversation context:\n{history_text}]"
        else:
            enriched_query = query

        # Step 2: Check local MCP cache
        cache_result = self.mcp.filesystem.search_files(query[:30])
        if cache_result.get("total_files_matched", 0) > 0:
            print("  [Cache hit in local MCP filesystem]")

        # Step 3: Run LangGraph
        answer = self.graph_runner.run(enriched_query)

        # Step 4: Persist answer to MCP filesystem
        self.mcp.filesystem.write_file(
            "./data/research_log.txt",
            f"Q: {query}\nA: {answer[:500]}\n",
            mode="append",
        )

        # Step 5: Save to both memory types
        self.buffer_memory.save_context({"input": query}, {"output": answer})
        self.summary_memory.save_context({"input": query}, {"output": answer})

        return answer

    # ── Direct RAG query (bypasses multi-agent, faster for simple lookups)  ──
    def ask_local(self, question: str) -> str:
        """Query only the local FAISS knowledge base (fast, no API calls)."""
        result = self.rag_chain({"query": question})
        return result.get("result", "")

    # ── Pure API search (no local docs)  ─────────────────────────────────────
    def search_papers(self, topic: str):
        return self.apis.search_papers(topic)

    def search_news(self, topic: str):
        return self.apis.search_news(topic)

    # ── MCP helpers  ─────────────────────────────────────────────────────────
    def list_local_files(self):
        return self.mcp.filesystem.list_files()

    def list_drive_files(self, query: str = ""):
        return self.mcp.google_drive.list_files(query=query)

    def save_note(self, content: str, filename: str = "notes.txt"):
        path = str(Path(FILESYSTEM_BASE_PATH) / filename)
        return self.mcp.filesystem.write_file(path, content, mode="append")

    # ── Session summary  ─────────────────────────────────────────────────────
    def get_conversation_summary(self) -> str:
        mem_vars = self.summary_memory.load_memory_variables({})
        return str(mem_vars.get("history", "No conversation yet."))


# ─── CLI  ─────────────────────────────────────────────────────────────────────
HELP_TEXT = """
Commands:
  ask <question>        – Full multi-agent research pipeline
  local <question>      – Local RAG only (fast, no APIs)
  papers <topic>        – Semantic Scholar paper search
  news <topic>          – News API search
  files                 – List local research files
  drive [query]         – List Google Drive files
  save <text>           – Save a note to local storage
  summary               – Show conversation summary
  help                  – Show this message
  exit                  – Quit
"""


def interactive_cli(assistant: ResearchAssistant):
    print("\n" + "═" * 60)
    print("  ResearchMind – Multi-Agent Research Assistant")
    print("═" * 60)
    print(HELP_TEXT)

    while True:
        try:
            raw = input("ResearchMind> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)

        if not raw:
            continue

        parts = raw.split(" ", 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        elif cmd == "ask":
            if not arg:
                print("Usage: ask <your research question>")
            else:
                answer = assistant.ask(arg)
                print(f"\n{answer}\n")

        elif cmd == "local":
            if not arg:
                print("Usage: local <question>")
            else:
                answer = assistant.ask_local(arg)
                print(f"\n{answer}\n")

        elif cmd == "papers":
            if not arg:
                print("Usage: papers <topic>")
            else:
                result = assistant.search_papers(arg)
                for i, p in enumerate(result.get("papers", []), 1):
                    print(f"{i}. {p.get('title')} ({p.get('year')})")
                    print(f"   {p.get('abstract', '')[:150]}…")
                print()

        elif cmd == "news":
            if not arg:
                print("Usage: news <topic>")
            else:
                result = assistant.search_news(arg)
                for i, a in enumerate(result.get("articles", []), 1):
                    print(f"{i}. [{a.get('source')}] {a.get('title')}")
                    print(f"   {a.get('description', '')[:120]}…")
                print()

        elif cmd == "files":
            result = assistant.list_local_files()
            files = result.get("files", [])
            if files:
                for f in files:
                    print(f"  • {f['name']} ({f['size_kb']} KB)")
            else:
                print("  No local files found.")

        elif cmd == "drive":
            result = assistant.list_drive_files(query=arg)
            for f in result.get("files", []):
                print(f"  • [{f['type']}] {f['name']}")

        elif cmd == "save":
            if not arg:
                print("Usage: save <your note>")
            else:
                result = assistant.save_note(arg)
                print(f"Note saved: {result}")

        elif cmd == "summary":
            print(assistant.get_conversation_summary())

        elif cmd == "help":
            print(HELP_TEXT)

        else:
            print(f"Unknown command: {cmd}. Type 'help' for options.")


# ─── Entry point  ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Path(FILESYSTEM_BASE_PATH).mkdir(parents=True, exist_ok=True)

    assistant = ResearchAssistant()
    interactive_cli(assistant)
