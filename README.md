# ResearchMind – Multi-Agent Academic Research Assistant

A full capstone project implementing every requirement from the bootcamp:
LangChain, Memory, RAG, Multi-Agent Design, LangGraph, MCP Servers,
External APIs, and Multiple Data Sources.

---

## System Architecture

```
User Query
    │
    ▼
LangGraph Orchestrator  ◄──── Conversation Memory (Buffer + Summary)
    │
    ├─► Planner Agent       – breaks query into subtasks
    │
    ├─► Retrieval Agent     – FAISS vector search over local PDFs
    │       └── Data: ./data/ (PDFs, TXT) + Google Drive MCP
    │
    ├─► Web Research Agent  – Semantic Scholar + News API
    │       └── APIs: semanticscholar.org + newsapi.org
    │
    └─► Synthesis Agent     – combines all findings → cited answer
            └── MCP: Filesystem (save notes, log results)
```

---

## Quick Start

### 1. Clone / create project folder

```bash
mkdir research_assistant && cd research_assistant
# copy all the provided .py files here
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
# Optionally add NEWS_API_KEY (free at newsapi.org)
```

### 4. Add research documents (optional but recommended)

Drop your own PDF files into `./data/`. The system auto-indexes them.
If the folder is empty, sample documents are created automatically.

### 5. Run the full system

```bash
python main.py
```

---

## Running Individual Parts

| File | What it demonstrates |
|------|---------------------|
| `part_b_basic_chatbot.py` | ChatOpenAI + PromptTemplate + LLMChain |
| `part_c_memory.py` | 4 memory strategies compared |
| `part_d_retrieval.py` | Full RAG pipeline (load → chunk → embed → FAISS) |
| `part_e_agents.py` | 4 specialist agents with manual coordination |
| `part_f_langgraph.py` | LangGraph state machine with conditional routing |
| `part_g_mcp.py` | Filesystem + Google Drive MCP servers |
| `part_h_apis.py` | Semantic Scholar + News API with LangChain @tool |
| `main.py` | Everything combined into an interactive CLI |

---

## CLI Commands (main.py)

```
ask <question>        Full multi-agent research pipeline
local <question>      Local RAG only (fast, no API calls)
papers <topic>        Semantic Scholar academic paper search
news <topic>          News API recent articles
files                 List local research files (via Filesystem MCP)
drive [query]         List Google Drive documents (via Drive MCP)
save <text>           Save a note to local storage (via Filesystem MCP)
summary               Show a summary of the conversation so far
help                  Show all commands
exit                  Quit
```

---

## Technical Requirements Checklist

### Part B – LangChain Foundations
- [x] `ChatOpenAI` – all agents use this
- [x] `PromptTemplate` – per-agent system prompts
- [x] `LLMChain` – Part B basic chatbot; agents use `ChatPromptTemplate`

### Part C – Memory
- [x] `ConversationBufferMemory` – primary memory for the full system
- [x] `ConversationSummaryMemory` – secondary; condenses long sessions
- [x] `ConversationBufferWindowMemory` – sliding window strategy (Part C demo)
- [x] `VectorStoreRetrieverMemory` – semantic memory over past turns (Part C demo)

### Part D – Retrieval (RAG)
- [x] OpenAI `text-embedding-3-small` embeddings
- [x] `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50)
- [x] FAISS vector database with persistence

### Part E – Multi-Agent (4 agents)
- [x] Planner Agent – JSON-structured task decomposition
- [x] Retrieval Agent – FAISS search + summarisation
- [x] Web Research Agent – Semantic Scholar + News API
- [x] Synthesis Agent – multi-source cited answer generation

### Part F – LangGraph
- [x] `StateGraph` with typed `ResearchState`
- [x] 4 nodes: plan_node, retrieve_node, web_research_node, synthesise_node
- [x] Conditional edges based on: confidence score, skip_web flag, answer quality
- [x] Retry loop: if answer too short → loop back to web research

### Part G – MCP Servers (2 servers)
- [x] Filesystem MCP – list, read, write, search local files
- [x] Google Drive MCP – list, read, create cloud documents

### Part H – External APIs (2 APIs)
- [x] Semantic Scholar Graph API (free, no key needed)
  - paper search, author lookup, recommendations
- [x] News API (free tier, key required)
  - keyword search, top headlines
- [x] Both wrapped as LangChain `@tool` functions

### Data Sources (2+ sources)
- [x] Local: PDFs, TXT files in `./data/` (indexed into FAISS)
- [x] Remote: Semantic Scholar, News API, Google Drive (simulated)

---

## Project Structure

```
research_assistant/
├── config.py               # Environment variables & constants
├── part_a_usecase.py       # Use case definition
├── part_b_basic_chatbot.py # LangChain basic chatbot
├── part_c_memory.py        # 4 memory strategies
├── part_d_retrieval.py     # RAG pipeline
├── part_e_agents.py        # 4 specialist agents
├── part_f_langgraph.py     # LangGraph workflow
├── part_g_mcp.py           # MCP server integrations
├── part_h_apis.py          # External API integrations
├── main.py                 # Full integrated application
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── data/                   # Your research PDFs go here
    ├── vector_db/          # FAISS index (auto-generated)
    └── research_log.txt    # Conversation log (auto-generated)
```

---

## LangGraph Workflow Diagram

```
START
  │
  ▼
plan_node ──────────────────────────────────────────────┐
  │                                                      │
  ▼                                                      │
retrieve_node                                            │
  │                                                      │
  ├── skip_web=True ──────────────────────────────────► synthesise_node
  │                                                      │
  └── (confidence < 6) or (normal) ──► web_research_node─┘
                                              │
                                              ▼
                                       synthesise_node
                                              │
                                   ┌──────────┴──────────┐
                                   │                      │
                             answer short             answer ok
                             + retry < 1                  │
                                   │                      ▼
                                   └──► web_research_node  END
                                              │
                                              ▼
                                       synthesise_node
                                              │
                                              ▼
                                             END
```

---

## Adding Your Own Research Papers

1. Place PDF files in the `./data/` directory
2. Delete `./data/vector_db/` (forces rebuild of the FAISS index)
3. Run `python main.py` — the index rebuilds automatically on startup

---

## Extending the System

- **Add a new agent**: create a class in `part_e_agents.py`, add a node in `part_f_langgraph.py`
- **Add a new API**: follow the pattern in `part_h_apis.py`, add a `@tool` function
- **Add a new MCP server**: follow the pattern in `part_g_mcp.py`, register in `MCPLayer`
- **Change the LLM**: update `LLM_MODEL` in `config.py`
