"""
════════════════════════════════════════════════════════════════════════
PART B  –  Basic Chatbot
Uses: ChatOpenAI · PromptTemplate · LLMChain
════════════════════════════════════════════════════════════════════════
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import OPENAI_API_KEY, LLM_MODEL


# ─── 1. Initialise the LLM ────────────────────────────────────────────────────
def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Return a configured ChatOpenAI instance."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
    )


# ─── 2. Build a reusable research prompt ─────────────────────────────────────
RESEARCH_PROMPT_TEMPLATE = """You are ResearchMind, an expert academic research assistant.
Your job is to help researchers understand complex topics clearly and concisely.

Topic: {topic}

Please provide:
1. A clear, concise explanation
2. Key concepts to understand
3. Suggested areas for deeper study

Answer:"""

research_prompt = PromptTemplate(
    input_variables=["topic"],
    template=RESEARCH_PROMPT_TEMPLATE,
)


# ─── 3. Compose the chain  ────────────────────────────────────────────────────
def build_basic_chain() -> LLMChain:
    """
    LLMChain = PromptTemplate → LLM → output parser (str by default).
    This is the minimal building block used in every subsequent part.
    """
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=research_prompt, verbose=True)
    return chain


# ─── 4. Interactive chatbot loop  ─────────────────────────────────────────────
def run_basic_chatbot():
    """Simple REPL that wraps the LLMChain."""
    chain = build_basic_chain()
    print("\n=== ResearchMind Basic Chatbot (Part B) ===")
    print("Type a research topic. Press Ctrl+C to quit.\n")

    while True:
        try:
            topic = input("Research topic: ").strip()
            if not topic:
                continue
            response = chain.run(topic=topic)   # invoke the prompt → LLM pipeline
            print(f"\nAssistant:\n{response}\n{'─'*60}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    run_basic_chatbot()
