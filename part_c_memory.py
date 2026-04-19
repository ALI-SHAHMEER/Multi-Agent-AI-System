"""
════════════════════════════════════════════════════════════════════════
PART C  –  Conversation Memory
Implements four strategies and lets you compare them side-by-side.

Strategy 1 (required)  : ConversationBufferMemory
Strategy 2             : ConversationSummaryMemory
Strategy 3             : ConversationBufferWindowMemory  (sliding window)
Strategy 4             : VectorStoreRetrieverMemory      (semantic memory)
════════════════════════════════════════════════════════════════════════
"""

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    VectorStoreRetrieverMemory,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import OPENAI_API_KEY, LLM_MODEL, SUMMARY_MAX_TOKENS


# ─── Helper: one ConversationChain for each memory type ───────────────────────
def _make_chain(memory) -> ConversationChain:
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY,
    )
    return ConversationChain(llm=llm, memory=memory, verbose=False)


# ─── Strategy 1 : Buffer memory  ─────────────────────────────────────────────
def strategy_buffer() -> ConversationChain:
    """
    Stores every message verbatim.
    PRO : perfect recall for recent turns
    CON : grows without bound → expensive for long sessions
    """
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
    )
    return _make_chain(memory)


# ─── Strategy 2 : Summary memory  ────────────────────────────────────────────
def strategy_summary() -> ConversationChain:
    """
    Summarises earlier conversation so the context stays compact.
    PRO : bounded token usage
    CON : summarisation may lose precise details
    """
    summariser_llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    memory = ConversationSummaryMemory(
        llm=summariser_llm,
        memory_key="history",
        return_messages=True,
        max_token_limit=SUMMARY_MAX_TOKENS,
    )
    return _make_chain(memory)


# ─── Strategy 3 : Sliding-window memory  ─────────────────────────────────────
def strategy_window(k: int = 5) -> ConversationChain:
    """
    Keeps only the last k exchanges.
    PRO : very cheap, predictable context size
    CON : no memory beyond the window
    """
    memory = ConversationBufferWindowMemory(
        k=k,
        memory_key="history",
        return_messages=True,
    )
    return _make_chain(memory)


# ─── Strategy 4 : Vector / semantic memory  ──────────────────────────────────
def strategy_vector() -> ConversationChain:
    """
    Stores messages as embeddings; retrieves by semantic similarity.
    PRO : 'remembers' relevant facts even from long-ago turns
    CON : retrieval is approximate; requires embedding calls
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )
    # In-memory FAISS store (swap for a persisted one in production)
    vectorstore = FAISS.from_texts(
        ["ResearchMind session started."],
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = VectorStoreRetrieverMemory(
        retriever=retriever,
        memory_key="history",
    )
    return _make_chain(memory)


# ─── Side-by-side comparison  ─────────────────────────────────────────────────
def compare_memory_strategies():
    """Run two rounds of Q&A on each strategy and print context sizes."""
    strategies = {
        "Buffer  ": strategy_buffer(),
        "Summary ": strategy_summary(),
        "Window  ": strategy_window(),
        "Vector  ": strategy_vector(),
    }

    turns = [
        "What is transformer architecture in deep learning?",
        "Can you compare it to recurrent neural networks?",
    ]

    print("\n=== Memory Strategy Comparison (Part C) ===\n")
    for name, chain in strategies.items():
        print(f"\n{'─'*56}\nStrategy: {name}\n{'─'*56}")
        for turn in turns:
            response = chain.predict(input=turn)
            print(f"Q: {turn}\nA: {response[:200]}...\n")

        # Inspect how the memory stores the conversation
        mem_vars = chain.memory.load_memory_variables({})
        history = mem_vars.get("history", "")
        token_estimate = len(str(history).split())
        print(f"[Memory snapshot ~{token_estimate} words]")


# ─── Interactive chatbot with buffer memory  ─────────────────────────────────
def run_memory_chatbot():
    """Full interactive chatbot using buffer + summary combination."""
    chain = strategy_buffer()
    print("\n=== ResearchMind with Memory (Part C) ===")
    print("The assistant now remembers your conversation history.")
    print("Type a message. Press Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            response = chain.predict(input=user_input)
            print(f"Assistant: {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    compare_memory_strategies()
