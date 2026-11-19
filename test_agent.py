import unittest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langgraph.graph import END

import tools
import agent

class TestTools(unittest.TestCase):
    def test_docs_content(self):
        self.assertEqual(len(tools.DOCS), 8)
        ids = [d["id"] for d in tools.DOCS]
        expected_ids = ["HR-1", "HR-2", "HR-3", "FIN-1", "IT-1", "IT-2", "OPS-1", "HR-4"]
        self.assertEqual(ids, expected_ids)
        
        # Verify specific topics mentioned in prompt
        self.assertIn("vacation", tools.DOCS[0]["title"].lower())
        self.assertIn("sick leave", tools.DOCS[1]["title"].lower())
        self.assertIn("remote work", tools.DOCS[2]["title"].lower())
        self.assertIn("expense", tools.DOCS[3]["title"].lower())
        self.assertIn("laptops", tools.DOCS[4]["title"].lower())
        self.assertIn("passwords", tools.DOCS[5]["title"].lower())
        self.assertIn("office hours", tools.DOCS[6]["title"].lower())
        self.assertIn("training", tools.DOCS[7]["title"].lower())

    def test_keywords_filter_stopwords(self):
        kws = tools.keywords("How many vacation days can I take in a calendar year?")
        self.assertNotIn("how", kws)
        self.assertNotIn("can", kws)
        self.assertNotIn("i", kws)
        self.assertNotIn("in", kws)
        self.assertNotIn("a", kws)
        self.assertIn("many", kws)
        self.assertIn("vacation", kws)
        self.assertIn("days", kws)
        self.assertIn("year", kws)

    def test_search_docs_matching(self):
        results = tools.search_docs("vacation days 24", k=2)
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["id"], "HR-1")

    def test_search_docs_no_match(self):
        results = tools.search_docs("quantum spacecraft warp engine", k=2)
        self.assertEqual(results, [])

    def test_search_docs_top_k_and_threshold(self):
        results = tools.search_docs("policy manager approval", k=2)
        self.assertLessEqual(len(results), 2)

    def test_format_context(self):
        sample_docs = [
            {"id": "HR-1", "title": "Vacation", "text": "24 days vacation."}
        ]
        formatted = tools.format_context(sample_docs)
        self.assertEqual(formatted, "[HR-1] Vacation: 24 days vacation.")


class TestAgentLogic(unittest.TestCase):
    def test_verify_the_verifier_grounded(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="GROUNDED")
        
        state = {
            "question": "How many days?",
            "search_query": "vacation days",
            "docs": [{"id": "HR-1", "title": "Vacation Policy", "text": "Employees receive 24 days of paid vacation per calendar year."}],
            "answer": "Employees get 24 days of paid vacation. [HR-1]",
            "verdict": "",
            "rounds": 0
        }
        result = agent.checker(state, llm=mock_llm)
        self.assertEqual(result["verdict"], "GROUNDED")
        self.assertEqual(result["rounds"], 1)

    def test_verify_the_verifier_hallucinated_unsupported_found_in_context(self):
        # Checker claims '24 days' is missing, but '24 days' is in context!
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="UNSUPPORTED: 24 days")
        
        state = {
            "question": "How many days?",
            "search_query": "vacation days",
            "docs": [{"id": "HR-1", "title": "Vacation Policy", "text": "Employees receive 24 days of paid vacation per calendar year."}],
            "answer": "Employees get 24 days of paid vacation. [HR-1]",
            "verdict": "",
            "rounds": 0
        }
        result = agent.checker(state, llm=mock_llm)
        # Verify the verifier must override it to GROUNDED
        self.assertEqual(result["verdict"], "GROUNDED")

    def test_verify_the_verifier_under_4_chars(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="UNSUPPORTED: no")
        
        state = {
            "question": "test",
            "search_query": "test",
            "docs": [{"id": "HR-1", "title": "Vacation Policy", "text": "Employees receive 24 days."}],
            "answer": "Test answer [HR-1]",
            "verdict": "",
            "rounds": 0
        }
        result = agent.checker(state, llm=mock_llm)
        self.assertEqual(result["verdict"], "GROUNDED")

    def test_verify_the_verifier_genuinely_unsupported(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="UNSUPPORTED: 60 days of vacation")
        
        state = {
            "question": "How many days?",
            "search_query": "vacation days",
            "docs": [{"id": "HR-1", "title": "Vacation Policy", "text": "Employees receive 24 days of paid vacation."}],
            "answer": "Employees get 60 days of vacation. [HR-1]",
            "verdict": "",
            "rounds": 0
        }
        result = agent.checker(state, llm=mock_llm)
        self.assertEqual(result["verdict"], "UNSUPPORTED: 60 days of vacation")
        self.assertEqual(result["rounds"], 1)

    def test_conditional_routing(self):
        # route_after_retriever
        self.assertEqual(agent.route_after_retriever({"docs": [{"id": "HR-1"}]}), "answerer")
        self.assertEqual(agent.route_after_retriever({"docs": []}), "no_answer")

        # route_after_checker
        self.assertEqual(agent.route_after_checker({"verdict": "GROUNDED", "rounds": 1}), END)
        self.assertEqual(agent.route_after_checker({"verdict": "UNSUPPORTED: something", "rounds": 1}), "answerer")
        self.assertEqual(agent.route_after_checker({"verdict": "UNSUPPORTED: something", "rounds": 2}), END)


class TestFullGraphExecution(unittest.TestCase):
    def test_graph_flow_success(self):
        mock_llm = MagicMock()
        # 1st call: rewriter
        # 2nd call: answerer
        # 3rd call: checker
        mock_llm.invoke.side_effect = [
            AIMessage(content="vacation days accrued"),
            AIMessage(content="Full-time employees receive 24 days of paid vacation per calendar year. [HR-1]"),
            AIMessage(content="GROUNDED")
        ]
        
        custom_graph = agent.build_graph(llm=mock_llm)
        initial_state = {
            "question": "How many vacation days do I get?",
            "search_query": "",
            "docs": [],
            "answer": "",
            "verdict": "",
            "rounds": 0
        }
        result = custom_graph.invoke(initial_state)
        
        self.assertEqual(result["verdict"], "GROUNDED")
        self.assertEqual(result["rounds"], 1)
        self.assertIn("HR-1", result["answer"])
        self.assertTrue(len(result["docs"]) >= 1)

    def test_graph_flow_no_sources(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="alien spaceship saucer Mars")
        ]
        
        custom_graph = agent.build_graph(llm=mock_llm)
        initial_state = {
            "question": "Can I pilot an alien spaceship to Mars?",
            "search_query": "",
            "docs": [],
            "answer": "",
            "verdict": "",
            "rounds": 0
        }
        result = custom_graph.invoke(initial_state)
        
        self.assertEqual(result["verdict"], "NO SOURCES")
        self.assertEqual(result["answer"], "I could not find that in the handbook.")

    def test_graph_flow_retry_loop_then_grounded(self):
        mock_llm = MagicMock()
        # 1. Rewriter
        # 2. Answerer round 1
        # 3. Checker round 1 (rejects with unsupported claim)
        # 4. Answerer round 2 (corrects answer)
        # 5. Checker round 2 (accepts as grounded)
        mock_llm.invoke.side_effect = [
            AIMessage(content="sick leave days doctor"),
            AIMessage(content="Employees get 10 days sick leave and unlimited free massages. [HR-2]"),
            AIMessage(content="UNSUPPORTED: unlimited free massages"),
            AIMessage(content="Employees are entitled to 10 days of paid sick leave annually. [HR-2]"),
            AIMessage(content="GROUNDED")
        ]
        
        custom_graph = agent.build_graph(llm=mock_llm)
        initial_state = {
            "question": "What is the sick leave policy?",
            "search_query": "",
            "docs": [],
            "answer": "",
            "verdict": "",
            "rounds": 0
        }
        result = custom_graph.invoke(initial_state)
        
        self.assertEqual(result["verdict"], "GROUNDED")
        self.assertEqual(result["rounds"], 2)
        self.assertIn("[HR-2]", result["answer"])


if __name__ == "__main__":
    unittest.main()
