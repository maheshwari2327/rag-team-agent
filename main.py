import sys
from agent import app

BANNER = """
============================================================
              RAG TEAM MULTI-AGENT CLI
  Ollama llama3.2 + LangChain + LangGraph Verification Loop
============================================================
Ask questions about company handbook policies (vacation,
sick leave, remote work, expenses, laptops, passwords, etc.).
Type 'exit' or 'quit' to quit.
"""

def print_separator():
    print("-" * 60)

def run_cli():
    print(BANNER)
    
    while True:
        try:
            print_separator()
            question = input("\nQuestion: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break
                
            initial_state = {
                "question": question,
                "search_query": "",
                "docs": [],
                "answer": "",
                "verdict": "",
                "rounds": 0
            }
            
            print("\n--- Running Multi-Agent Pipeline ---")
            final_state = dict(initial_state)
            
            for step in app.stream(initial_state):
                for node_name, updates in step.items():
                    final_state.update(updates)
                    
                    if node_name == "rewriter":
                        print(f"🔍 [Rewriter] Keywords: {updates.get('search_query', '')}")
                        
                    elif node_name == "retriever":
                        docs = updates.get("docs", [])
                        if docs:
                            doc_refs = ", ".join(f"[{d['id']}] {d['title']}" for d in docs)
                            print(f"📚 [Retriever] Retrieved {len(docs)} passage(s): {doc_refs}")
                        else:
                            print("📚 [Retriever] No relevant passages found.")
                            
                    elif node_name == "no_answer":
                        print(f"⚠️ [No Answer] {updates.get('answer', '')}")
                        
                    elif node_name == "answerer":
                        round_num = final_state.get("rounds", 0) + 1
                        ans = updates.get("answer", "")
                        print(f"✍️  [Answerer] (Round {round_num}) Answer:\n    {ans}")
                        
                    elif node_name == "checker":
                        verdict = updates.get("verdict", "")
                        rounds = updates.get("rounds", 0)
                        if verdict == "GROUNDED":
                            print(f"🧐 [Checker] Round {rounds} Verdict: GROUNDED")
                        else:
                            print(f"🧐 [Checker] Round {rounds} Verdict: {verdict}")
                            if rounds < 2:
                                print("    ↪️ Refined feedback sent back to Answerer for correction...")
            
            # Print Final Result with Badge
            print("\n" + "=" * 60)
            verdict = final_state.get("verdict", "")
            ans = final_state.get("answer", "")
            
            if verdict == "GROUNDED":
                print("Result: ✅ checked")
            elif verdict == "NO SOURCES":
                print("Result: ⚠️ (No sources found)")
            else:
                print("Result: ⚠️ (Max review rounds reached)")
                
            print(f"\n{ans}")
            print("=" * 60)
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}")
            print("Please ensure Ollama is running ('ollama serve') and llama3.2 is pulled ('ollama pull llama3.2').")

if __name__ == "__main__":
    run_cli()
