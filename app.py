import streamlit as st
from agent import build_graph
from tools import DOCS

st.set_page_config(
    page_title="RAG Team Handbook Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "custom_model" not in st.session_state:
    st.session_state.custom_model = "llama3.2"

# Sidebar: Configuration & Document Explorer
with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.text_input("Ollama Model", value="llama3.2", help="Local model name in Ollama")
    st.caption("Ensure Ollama is running locally: `ollama serve`")
    
    st.markdown("---")
    st.header("📚 Company Handbook")
    st.caption(f"8 indexed passages available in memory:")
    for doc in DOCS:
        with st.expander(f"[{doc['id']}] {doc['title']}"):
            st.write(doc["text"])
            
    st.markdown("---")
    st.subheader("💡 Example Questions")
    example_prompts = [
        "How many vacation days do full-time employees get?",
        "What is the policy for sick leave over 3 days?",
        "Can I work remotely up to 3 days a week?",
        "What are the password requirements and expiration?",
        "Can I get reimbursement for an international flight?",
    ]
    for prompt in example_prompts:
        if st.button(prompt, key=f"ex_{prompt}", use_container_width=True):
            st.session_state["pending_input"] = prompt

# Main Content
st.title("🏢 RAG Team Multi-Agent Assistant")
st.markdown("""
This multi-agent team uses **LangGraph**, **LangChain**, and local **Ollama (`llama3.2`)** to answer questions 
strictly from the company handbook with an automatic fact-checking verification loop.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Architecture", value="LangGraph Multi-Agent")
with col2:
    st.metric(label="Verifier Loop", value="Fact-Checker + Guardrail")
with col3:
    st.metric(label="LLM", value=model_choice)

st.divider()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Show stage details if available
            if "stages" in msg and msg["stages"]:
                with st.expander("🔍 View Agent Pipeline Steps", expanded=False):
                    for stage in msg["stages"]:
                        st.markdown(stage)
            
            verdict = msg.get("verdict", "")
            if verdict == "GROUNDED":
                st.success("✅ **Answer Checked & Grounded**")
            elif verdict == "NO SOURCES":
                st.warning("⚠️ **No Sources Found**")
            else:
                st.warning(f"⚠️ **Review Notice**: {verdict}")
                
            st.markdown(msg["content"])

# Chat Input Handling
user_input = st.chat_input("Ask a question about handbook policies...")
if "pending_input" in st.session_state and st.session_state["pending_input"]:
    user_input = st.session_state.pop("pending_input")

if user_input:
    # Append and show User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run Agent Workflow
    with st.chat_message("assistant"):
        stage_logs = []
        
        with st.status("🤖 RAG Team at work...", expanded=True) as status_box:
            initial_state = {
                "question": user_input,
                "search_query": "",
                "docs": [],
                "answer": "",
                "verdict": "",
                "rounds": 0
            }
            
            app = build_graph()
            final_state = dict(initial_state)
            
            for step in app.stream(initial_state):
                for node_name, updates in step.items():
                    final_state.update(updates)
                    
                    if node_name == "rewriter":
                        kws = updates.get("search_query", "")
                        msg = f"🔍 **Rewriter Node**: Generated keywords `{kws}`"
                        st.markdown(msg)
                        stage_logs.append(msg)
                        
                    elif node_name == "retriever":
                        docs = updates.get("docs", [])
                        if docs:
                            refs = ", ".join(f"`[{d['id']}] {d['title']}`" for d in docs)
                            msg = f"📚 **Retriever Node** (No LLM): Retrieved {len(docs)} passage(s): {refs}"
                        else:
                            msg = "📚 **Retriever Node**: No matching passages found."
                        st.markdown(msg)
                        stage_logs.append(msg)
                        
                    elif node_name == "no_answer":
                        msg = "⚠️ **Fallback Node**: Route triggered due to zero relevant sources."
                        st.markdown(msg)
                        stage_logs.append(msg)
                        
                    elif node_name == "answerer":
                        r = final_state.get("rounds", 0) + 1
                        ans_draft = updates.get("answer", "")
                        msg = f"✍️ **Answerer Node** (Round {r}): Drafted response with citations."
                        st.markdown(msg)
                        stage_logs.append(f"{msg}\n> *{ans_draft}*")
                        
                    elif node_name == "checker":
                        v = updates.get("verdict", "")
                        r = updates.get("rounds", 0)
                        msg = f"🧐 **Checker Node** (Round {r}): Verdict `{v}`"
                        st.markdown(msg)
                        stage_logs.append(msg)
                        if v != "GROUNDED" and r < 2:
                            retry_msg = "🔄 *Unsupported claim detected. Triggering correction loop back to Answerer...*"
                            st.markdown(retry_msg)
                            stage_logs.append(retry_msg)
                            
            status_box.update(label="Team completed analysis!", state="complete", expanded=False)
            
        verdict = final_state.get("verdict", "")
        answer = final_state.get("answer", "")
        
        if verdict == "GROUNDED":
            st.success("✅ **Answer Checked & Grounded**")
        elif verdict == "NO SOURCES":
            st.warning("⚠️ **No Sources Found**")
        else:
            st.warning(f"⚠️ **Review Notice**: {verdict}")
            
        st.markdown(answer)
        
        # Save to session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "verdict": verdict,
            "stages": stage_logs
        })
