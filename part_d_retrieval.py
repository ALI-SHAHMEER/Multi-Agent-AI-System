"""
════════════════════════════════════════════════════════════════════════
PART D  –  Retrieval-Augmented Generation (RAG)
Pipeline: PDF loader → chunker → embeddings → FAISS → retriever → chain
════════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    DirectoryLoader,
)
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FILESYSTEM_BASE_PATH,
)


# ─── Step 1 : Load documents from the local data directory ───────────────────
def load_documents(data_dir: str = FILESYSTEM_BASE_PATH) -> List[Document]:
    """
    Loads PDFs, plain text, and CSV files from the data directory.
    Each document keeps its source path as metadata.
    """
    docs: List[Document] = []
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # PDF files
    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    # Text / Markdown files
    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )

    for loader in [pdf_loader, txt_loader]:
        try:
            loaded = loader.load()
            docs.extend(loaded)
            print(f"  Loaded {len(loaded)} pages from {loader.__class__.__name__}")
        except Exception as e:
            print(f"  Warning: {e}")

    # Fallback: create a sample document if the folder is empty
    if not docs:
        print("  No files found — creating sample research document.")
        docs = _create_sample_documents()

    print(f"Total documents loaded: {len(docs)}")
    return docs


def _create_sample_documents() -> List[Document]:
    """Create demo documents so the pipeline works without any uploaded files."""
    return [
        Document(
            page_content="""Transformer Architecture in Deep Learning

The Transformer model, introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017),
revolutionised natural language processing. Unlike recurrent neural networks (RNNs) and long
short-term memory (LSTM) networks, Transformers rely entirely on attention mechanisms to model
relationships between tokens, enabling highly parallelised training.

Key components:
- Self-attention: Each token attends to all other tokens in the sequence.
- Multi-head attention: Multiple attention heads capture different relationship types.
- Positional encoding: Since Transformers lack recurrence, positional embeddings encode order.
- Feed-forward layers: Applied per-token after each attention block.

Transformers form the backbone of BERT, GPT, T5, and virtually all modern LLMs.""",
            metadata={"source": "sample_transformers.txt", "page": 1},
        ),
        Document(
            page_content="""Retrieval-Augmented Generation (RAG)

RAG is a technique that augments an LLM's knowledge by retrieving relevant documents
at inference time and including them in the prompt context.

Workflow:
1. Index: Chunk documents → embed each chunk → store in a vector database.
2. Retrieve: For an incoming query, embed the query → similarity search → top-k chunks.
3. Generate: Prepend retrieved chunks to the prompt → LLM generates a grounded answer.

RAG reduces hallucination because the model grounds its answer in retrieved text.
Common vector stores: FAISS (local, fast), Chroma (open-source), Pinecone (cloud).""",
            metadata={"source": "sample_rag.txt", "page": 1},
        ),
        Document(
            page_content="""Multi-Agent AI Systems

A multi-agent system consists of several autonomous LLM-powered agents, each with a
specific role. Agents communicate by passing structured messages or sharing a global state.

Roles commonly used:
- Planner: Decomposes the user request into subtasks.
- Retriever: Searches knowledge bases for relevant context.
- Researcher: Uses external tools (APIs, web search) to gather information.
- Synthesiser: Combines information from other agents into a final answer.

Frameworks: LangGraph enables state-machine-based orchestration with conditional edges,
making it easy to implement retries, loops, and human-in-the-loop checkpoints.""",
            metadata={"source": "sample_multiagent.txt", "page": 1},
        ),
    ]


# ─── Step 2 : Chunk the documents  ───────────────────────────────────────────
def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Splits large documents into overlapping chunks so that:
    - Each chunk fits in the embedding model's token limit.
    - Overlap ensures context is not lost at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunked {len(docs)} documents into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ─── Step 3 : Embed and store in FAISS  ──────────────────────────────────────
def build_vectorstore(chunks: List[Document]) -> FAISS:
    """
    Creates OpenAI embeddings for every chunk and stores them in FAISS.
    If a persisted index already exists it is loaded instead of rebuilt.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    index_path = Path(VECTOR_DB_PATH)

    if index_path.exists():
        print(f"Loading existing FAISS index from {VECTOR_DB_PATH}")
        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        print("Building new FAISS index (this calls the embedding API)…")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        index_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(VECTOR_DB_PATH)
        print(f"FAISS index saved to {VECTOR_DB_PATH}")

    return vectorstore


# ─── Step 4 : Build the RAG chain  ───────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are ResearchMind, an expert academic research assistant.
Use the following retrieved research excerpts to answer the user's question.
If the answer is not in the context, say so honestly.

Retrieved context:
{context}

Question: {question}

Provide a clear, well-structured answer with references to the source material:""",
)


def build_rag_chain(vectorstore: FAISS) -> RetrievalQA:
    """
    Combines:
    - A retriever (top-5 similar chunks from FAISS)
    - A ChatOpenAI LLM
    - A custom RAG prompt
    into a RetrievalQA chain.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",              # stuff = concatenate all chunks into one prompt
        retriever=retriever,
        return_source_documents=True,    # include chunk sources in the response
        chain_type_kwargs={"prompt": RAG_PROMPT},
        verbose=True,
    )
    return rag_chain


# ─── Convenience: build everything in one call  ───────────────────────────────
def setup_rag_pipeline(data_dir: str = FILESYSTEM_BASE_PATH):
    """Full pipeline: load → chunk → embed → build chain."""
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)
    chain = build_rag_chain(vectorstore)
    return chain, vectorstore


# ─── Interactive RAG chatbot  ─────────────────────────────────────────────────
def run_rag_chatbot():
    print("\n=== ResearchMind RAG Chatbot (Part D) ===")
    print("Building the knowledge base…\n")
    chain, _ = setup_rag_pipeline()

    print("\nKnowledge base ready. Ask a research question.")
    print("Press Ctrl+C to quit.\n")

    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue

            result = chain({"query": question})
            print(f"\nAnswer:\n{result['result']}")

            # Show source documents used
            if result.get("source_documents"):
                print("\nSources:")
                seen = set()
                for doc in result["source_documents"]:
                    src = doc.metadata.get("source", "unknown")
                    if src not in seen:
                        print(f"  • {src}")
                        seen.add(src)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    run_rag_chatbot()
