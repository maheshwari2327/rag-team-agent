import os
from typing import TypedDict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from tools import DOCS, search_docs, format_context, keywords

# --- SYSTEM PROMPTS ---

REWRITER_PROMPT = """You are a SEARCH QUERY WRITER. Turn the user's question into 3-5 keywords for a document search.
Return ONLY the keywords separated by spaces."""

ANSWERER_PROMPT = """You are a HANDBOOK ASSISTANT. Answer the question using ONLY the CONTEXT passages.
Be short (max 2 sentences) and end with the source ids in brackets, like [HR-1].
If the context does not contain the answer, reply exactly: I could not find that in the handbook."""

CHECKER_PROMPT = """You are a FACT CHECKER. Check ONLY the statements made in the ANSWER (ignore other passages).
Look for every fact and number of the ANSWER in the CONTEXT.
- If every fact in the answer appears in the CONTEXT, reply with the single word: GROUNDED
- If the answer says something the CONTEXT does not say, reply: UNSUPPORTED: <the exact claim that is missing>"""

# --- CONFIGURATION ---

MAX_ROUNDS = 2
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

def get_llm(model: Optional[str] = None):
    """Instantiate ChatOllama with temperature 0."""
    return ChatOllama(
        model=model or DEFAULT_MODEL,
        temperature=0,
        base_url=DEFAULT_BASE_URL
    )

# --- STATE DEFINITION ---

class AgentState(TypedDict):
    question: str
    search_query: str
    docs: list[dict]
    answer: str
    verdict: str
    rounds: int

# --- NODES ---

def rewriter(state: AgentState, llm=None) -> dict:
    """Turn the user's question into 3-5 keywords for document search."""
    model = llm or get_llm()
    messages = [
        SystemMessage(content=REWRITER_PROMPT),
        HumanMessage(content=state["question"])
    ]
    response = model.invoke(messages)
    raw_output = response.content.strip()
    
    # Clean output to obtain space-separated keywords
    cleaned_query = raw_output.replace("\n", " ").strip("\"'`")
    return {"search_query": cleaned_query}

def retriever(state: AgentState) -> dict:
    """Retrieve passages (NO LLM) using keyword overlap."""
    search_query = state.get("search_query", "")
    question = state.get("question", "")
    
    # search_docs(search_query + the question's keywords)
    q_keywords = " ".join(keywords(question))
    combined_query = f"{search_query} {q_keywords}".strip()
    
    docs = search_docs(combined_query, k=2)
    return {"docs": docs}

def no_answer(state: AgentState) -> dict:
    """Fallback when no source documents are found."""
    return {
        "answer": "I could not find that in the handbook.",
        "verdict": "NO SOURCES"
    }

def answerer(state: AgentState, llm=None) -> dict:
    """Answer question using only context passages with citations."""
    model = llm or get_llm()
    docs = state.get("docs", [])
    context_str = format_context(docs)
    
    user_content = f"CONTEXT:\n{context_str}\n\nQUESTION: {state['question']}"
    
    # If the checker rejected a previous answer, include its complaint
    verdict = state.get("verdict", "")
    if verdict.startswith("UNSUPPORTED:"):
        complaint = verdict.split("UNSUPPORTED:", 1)[1].strip()
        user_content += (
            f"\n\nNOTE: The previous answer was rejected because: {complaint}. "
            "Please rewrite the answer strictly citing facts in the CONTEXT."
        )
        
    messages = [
        SystemMessage(content=ANSWERER_PROMPT),
        HumanMessage(content=user_content)
    ]
    response = model.invoke(messages)
    return {"answer": response.content.strip()}

def checker(state: AgentState, llm=None) -> dict:
    """Fact check answer against context, with verify-the-verifier code check."""
    model = llm or get_llm()
    docs = state.get("docs", [])
    context_str = format_context(docs)
    answer = state.get("answer", "")
    
    checker_input = f"CONTEXT:\n{context_str}\n\nANSWER:\n{answer}"
    messages = [
        SystemMessage(content=CHECKER_PROMPT),
        HumanMessage(content=checker_input)
    ]
    response = model.invoke(messages)
    raw_verdict = response.content.strip()
    
    current_rounds = state.get("rounds", 0) + 1
    
    # VERIFY THE VERIFIER in code:
    # If the claim text (or anything under 4 chars) appears in the context, treat as GROUNDED
    verdict = raw_verdict
    raw_upper = raw_verdict.upper()
    
    if raw_upper == "GROUNDED" or raw_upper.startswith("GROUNDED"):
        verdict = "GROUNDED"
    elif "UNSUPPORTED" in raw_upper:
        if ":" in raw_verdict:
            claim = raw_verdict.split(":", 1)[1].strip()
        else:
            claim = raw_verdict.replace("UNSUPPORTED", "").strip()
            
        clean_claim = claim.strip("\"'").rstrip(".").strip()
        
        # Verify the verifier logic:
        if len(clean_claim) < 4 or clean_claim.lower() in context_str.lower():
            verdict = "GROUNDED"
        else:
            verdict = f"UNSUPPORTED: {clean_claim}"
    else:
        # If the LLM returned text without clear label, default to GROUNDED if no explicit unsupported claim
        verdict = "GROUNDED"
        
    return {"verdict": verdict, "rounds": current_rounds}

# --- CONDITIONAL ROUTING ---

def route_after_retriever(state: AgentState) -> str:
    """Route to answerer if docs are found, otherwise no_answer."""
    if state.get("docs") and len(state["docs"]) > 0:
        return "answerer"
    return "no_answer"

def route_after_checker(state: AgentState) -> str:
    """Route to END if GROUNDED or rounds >= MAX_ROUNDS, else loop back to answerer."""
    verdict = state.get("verdict", "")
    rounds = state.get("rounds", 0)
    if verdict == "GROUNDED" or rounds >= MAX_ROUNDS:
        return END
    return "answerer"

# --- GRAPH BUILDER ---

def build_graph(llm=None):
    """Build and compile the RAG Team StateGraph."""
    workflow = StateGraph(AgentState)
    
    # Bind custom LLM if provided (useful for tests and dependency injection)
    def _rewriter_node(state): return rewriter(state, llm=llm)
    def _answerer_node(state): return answerer(state, llm=llm)
    def _checker_node(state): return checker(state, llm=llm)
    
    workflow.add_node("rewriter", _rewriter_node)
    workflow.add_node("retriever", retriever)
    workflow.add_node("no_answer", no_answer)
    workflow.add_node("answerer", _answerer_node)
    workflow.add_node("checker", _checker_node)
    
    # Edges
    workflow.add_edge(START, "rewriter")
    workflow.add_edge("rewriter", "retriever")
    
    workflow.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {
            "answerer": "answerer",
            "no_answer": "no_answer"
        }
    )
    workflow.add_edge("no_answer", END)
    
    workflow.add_edge("answerer", "checker")
    
    workflow.add_conditional_edges(
        "checker",
        route_after_checker,
        {
            END: END,
            "answerer": "answerer"
        }
    )
    
    return workflow.compile()

# Default compiled application instance
app = build_graph()
