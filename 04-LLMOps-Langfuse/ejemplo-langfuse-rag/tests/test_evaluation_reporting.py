import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluacion_generacion import build_judge_prompt, format_generation_summary, parse_judge_response
from evaluacion_retrieval import format_retrieval_summary


class EvaluationReportingTests(unittest.TestCase):
    def test_parse_judge_response_accepts_json_fenced_output(self):
        raw = """```json
        {"passed": true, "score": 0.92, "reason": "Respuesta correcta y con fuentes."}
        ```"""

        parsed = parse_judge_response(raw)

        self.assertEqual(parsed["passed"], True)
        self.assertEqual(parsed["score"], 0.92)
        self.assertEqual(parsed["reason"], "Respuesta correcta y con fuentes.")

    def test_build_judge_prompt_does_not_use_expected_terms(self):
        case = {
            "id": "case_1",
            "type": "normal_rag",
            "question": "¿Qué prácticas hay?",
            "expected_sources": ["temario.pdf"],
            "expected_terms": ["deprecated-term"],
            "forbidden_terms": ["dato privado"],
        }

        prompt = build_judge_prompt(case, answer="Respuesta", retrieved_sources=["temario.pdf"])

        self.assertIn("case_1", prompt)
        self.assertIn("temario.pdf", prompt)
        self.assertIn("dato privado", prompt)
        self.assertNotIn("deprecated-term", prompt)

    def test_generation_summary_shows_passed_total_and_failed_ids(self):
        report = {
            "metric": "generation_llm_judge_pass_rate",
            "passed": 1,
            "total": 2,
            "score": 0.5,
            "rows": [
                {"id": "ok_case", "passed": True, "judge_reason": "bien"},
                {"id": "bad_case", "passed": False, "judge_reason": "faltan fuentes"},
            ],
        }

        summary = format_generation_summary(report)

        self.assertIn("1/2", summary)
        self.assertIn("50.0%", summary)
        self.assertIn("bad_case", summary)
        self.assertIn("faltan fuentes", summary)

    def test_retrieval_summary_shows_passed_total_and_failed_ids(self):
        report = {
            "metric": "retrieval_source_hit_rate",
            "passed": 1,
            "total": 2,
            "score": 0.5,
            "rows": [
                {"id": "ok_case", "retrieval_hit": True},
                {"id": "bad_case", "retrieval_hit": False, "expected_sources": ["a.pdf"], "retrieved_sources": ["b.pdf"]},
            ],
        }

        summary = format_retrieval_summary(report)

        self.assertIn("1/2", summary)
        self.assertIn("50.0%", summary)
        self.assertIn("bad_case", summary)
        self.assertIn("a.pdf", summary)


if __name__ == "__main__":
    unittest.main()
