"""Offline tests with fake models: python -m unittest -v"""
import unittest

from analyze import summarise

from wwh_rag.benchmarks import Item
from wwh_rag.parsing import parse_answer, parse_hints, parse_queries
from wwh_rag.pipeline import baseline, run_arms, why_what_how
from wwh_rag.stats import mcnemar_exact

ITEM = Item("q1", "Which gas do plants take in for photosynthesis?",
            ["Oxygen", "Carbon dioxide", "Nitrogen", "Helium"], gold=1)


class FakeSLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_new_tokens=192):
        self.prompts.append(prompt)
        if "WHY:" in prompt and "Sub-queries" not in prompt:
            return ("WHY: Why do plants need a gas for photosynthesis?\n"
                    "WHAT: What gas is consumed in photosynthesis?\n"
                    "HOW: How do leaves take in gases?")
        return "- Plants absorb carbon dioxide.\n- Oxygen is released.\nnoise line"


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply


class FakeSearch:
    def __init__(self):
        self.queries = []

    def search(self, q, k=2):
        self.queries.append(q)
        return [{"snippet": f"snippet about {q}", "url": "http://x"}][:k]


class TestParsing(unittest.TestCase):
    def test_queries_ignore_template_placeholders(self):
        # The original bug: the prompt was echoed, so the template line was parsed.
        echoed = "Return:\nWHY: <why_question>\nWHAT: <what_question>\nHOW: <how_question>"
        self.assertEqual(parse_queries(echoed, "fallback"),
                         {"WHY": "fallback", "WHAT": "fallback", "HOW": "fallback"})

    def test_queries_read(self):
        q = parse_queries("**WHY:** a\nWHAT: b\nhow: c", "f")
        self.assertEqual(q, {"WHY": "a", "WHAT": "b", "HOW": "c"})

    def test_answer_uses_final_line_not_first_capital(self):
        text = "Analyzing each option. A is wrong because...\nFinal Answer: C"
        self.assertEqual(parse_answer(text, 4), 2)

    def test_answer_last_final_wins_and_formats(self):
        self.assertEqual(parse_answer("Final answer: A ... wait. Final Answer: **(D)**", 4), 3)
        self.assertEqual(parse_answer(" (B). ", 4), 1)

    def test_answer_unreadable_or_out_of_range(self):
        self.assertIsNone(parse_answer("I think it's the second one", 4))
        self.assertIsNone(parse_answer("Final Answer: E", 4))
        self.assertEqual(parse_answer("E", 5), 4)

    def test_hints(self):
        self.assertEqual(parse_hints("- a\n* b\nc\n-\n• d"), ["a", "b", "d"])


class TestPipeline(unittest.TestCase):
    def test_full_flow_searches_real_subqueries(self):
        slm, llm, search = FakeSLM(), FakeLLM("reasoning...\nFinal Answer: B"), FakeSearch()
        out = why_what_how(ITEM, "arc", "cot", slm, llm, search)
        self.assertEqual(search.queries[1], "What gas is consumed in photosynthesis?")
        self.assertEqual(len(out["snippets"]), 3)
        self.assertEqual(out["hints"], ["Plants absorb carbon dioxide.", "Oxygen is released."])
        self.assertIn("- Plants absorb carbon dioxide.", llm.prompts[0])   # hints reach Gemini
        self.assertEqual(out["pred"], 1)

    def test_baseline_has_no_hints_and_same_question(self):
        llm = FakeLLM("B")
        out = baseline(ITEM, "arc", "instruction", llm)
        self.assertNotIn("Hints", llm.prompts[0])
        self.assertIn(ITEM.question, llm.prompts[0])
        self.assertEqual(out["pred"], 1)

    def test_search_failure_still_answers(self):
        class Broken(FakeSearch):
            def search(self, q, k=2):
                raise TimeoutError

        out = why_what_how(ITEM, "arc", "cot", FakeSLM(), FakeLLM("Final Answer: A"), Broken())
        self.assertEqual((out["hints"], out["pred"]), ([], 0))


    def test_nosearch_arm_never_searches(self):
        slm, llm = FakeSLM(), FakeLLM("Final Answer: B")
        out = why_what_how(ITEM, "arc", "cot", slm, llm, search=None)
        self.assertEqual(out["snippets"], [])
        self.assertIn("From your own knowledge", slm.prompts[1])
        self.assertIn("- Plants absorb carbon dioxide.", llm.prompts[0])
        self.assertEqual(out["pred"], 1)

    def test_run_arms_same_item_all_arms(self):
        search = FakeSearch()
        out = run_arms(ITEM, "arc", "cot", ["base", "nosearch", "search"],
                       FakeLLM("Final Answer: B"), FakeSLM(), search)
        self.assertEqual(list(out), ["base", "nosearch", "search"])
        self.assertTrue(all(r["correct"] for r in out.values()))
        self.assertEqual(len(search.queries), 3)          # only the search arm searched
        with self.assertRaises(ValueError):
            run_arms(ITEM, "arc", "cot", ["oracle"], FakeLLM("B"))


class TestAnalyze(unittest.TestCase):
    def test_pairs_only_for_arms_present(self):
        def rec(base, search):
            return {"bench": "arc", "mode": "cot", "arms": {
                "base": {"correct": base, "pred": "A"}, "search": {"correct": search, "pred": "B"}}}
        rows = summarise([rec(True, False), rec(False, True), rec(False, True)])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["comparison"], r["only_a_right"], r["only_b_right"]), ("search vs base", 1, 2))
        self.assertAlmostEqual(r["delta_pts"], 100 / 3)


class TestStats(unittest.TestCase):
    def test_mcnemar(self):
        base = [1, 1] + [0] * 8 + [1] * 5
        ours = [0, 0] + [1] * 8 + [1] * 5
        b, c, p = mcnemar_exact(base, ours)
        self.assertEqual((b, c), (2, 8))
        self.assertAlmostEqual(p, 2 * (1 + 10 + 45) / 1024)
        self.assertEqual(mcnemar_exact([1, 0], [1, 0])[2], 1.0)


if __name__ == "__main__":
    unittest.main()
