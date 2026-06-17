import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_common import expected_files, expected_pairs, format_expected, pct, render_table
from evaluacion_generacion import (
    build_judge_prompt,
    format_generation_table,
    parse_judge_response,
)
from evaluacion_retrieval import format_retrieval_table


class EvalCommonTests(unittest.TestCase):
    def test_expected_pairs_and_files(self):
        case = {
            "id": "c1",
            "question": "¿?",
            "expected_sources": {"temario.pdf": [2, 3], "plan.json": [1]},
        }

        self.assertEqual(
            expected_pairs(case),
            {("temario.pdf", 2), ("temario.pdf", 3), ("plan.json", 1)},
        )
        self.assertEqual(expected_files(case), {"temario.pdf", "plan.json"})

    def test_format_expected_uses_file_hash_fragment(self):
        case = {"expected_sources": {"temario.pdf": [2, 3]}}

        self.assertEqual(format_expected(case), "temario.pdf#2,3")

    def test_pct_tolerates_zero_denominator(self):
        self.assertEqual(pct(0, 0), "0.0%")
        self.assertEqual(pct(1, 2), "50.0%")

    def test_render_table_includes_headers_and_rows(self):
        table = render_table(["A", "B"], [["1", "2"], ["3", "4"]])

        self.assertIn("A", table)
        self.assertIn("3", table)


class JudgeReportingTests(unittest.TestCase):
    def test_parse_judge_response_accepts_json_fenced_output(self):
        raw = """```json
        {"passed": true, "score": 0.92, "reason": "Respuesta correcta y con fuentes."}
        ```"""

        parsed = parse_judge_response(raw)

        self.assertEqual(parsed["passed"], True)
        self.assertEqual(parsed["score"], 0.92)
        self.assertEqual(parsed["reason"], "Respuesta correcta y con fuentes.")

    def test_build_judge_prompt_hides_expected_sources(self):
        case = {
            "id": "case_1",
            "question": "¿Qué prácticas hay?",
            "expected_sources": {"temario.pdf": [2, 3]},
        }

        prompt = build_judge_prompt(case, answer="Respuesta", retrieved_context=[])

        self.assertIn("case_1", prompt)
        self.assertIn("¿Qué prácticas hay?", prompt)
        # El juez no debe ver el fragmento esperado: evita mezclar retrieval y generación.
        self.assertNotIn("temario.pdf", prompt)
        self.assertNotIn("expected_sources", prompt)

    def test_parse_judge_response_salvages_partial_json(self):
        raw = '{"passed": true, "score": 0.8, "reason": "Correcta pero el cierre se '

        parsed = parse_judge_response(raw)

        self.assertTrue(parsed["passed"])
        self.assertEqual(parsed["score"], 0.8)

    def test_parse_judge_response_flags_unsalvageable_output(self):
        parsed = parse_judge_response('{\n  "passed')

        self.assertFalse(parsed["passed"])
        self.assertTrue(parsed["reason"].startswith("El juez no devolvió JSON válido"))

    def test_generation_table_shows_rates_and_case_ids(self):
        report = {
            "metric": "generation_llm_judge_pass_rate",
            "prompt_name": "rag-basico",
            "model": "demo",
            "passed": 1,
            "total": 2,
            "fragment_hits": 2,
            "avg_judge_score": 0.5,
            "avg_latency_ms": 120.0,
            "rows": [
                {"id": "ok_case", "passed": True, "fragment_hit": True, "judge_score": 1.0, "latency_ms": 100, "judge_reason": "bien"},
                {"id": "bad_case", "passed": False, "fragment_hit": True, "judge_score": 0.0, "latency_ms": 140, "judge_reason": "respuesta incompleta"},
            ],
        }

        table = format_generation_table(report)

        self.assertIn("1/2", table)
        self.assertIn("50.0%", table)
        self.assertIn("bad_case", table)
        self.assertIn("respuesta incompleta", table)

    def test_retrieval_table_shows_rates_and_case_ids(self):
        report = {
            "metric": "retrieval_fragment_hit_rate",
            "k": 4,
            "total": 2,
            "fragment_hits": 1,
            "file_hits": 2,
            "rows": [
                {"id": "ok_case", "fragment_hit": True, "file_hit": True, "expected": "a.pdf#1", "retrieved": [("a.pdf", 1)]},
                {"id": "bad_case", "fragment_hit": False, "file_hit": True, "expected": "a.pdf#2", "retrieved": [("a.pdf", 5)]},
            ],
        }

        table = format_retrieval_table(report)

        self.assertIn("1/2", table)
        self.assertIn("50.0%", table)
        self.assertIn("bad_case", table)
        self.assertIn("a.pdf", table)


if __name__ == "__main__":
    unittest.main()
