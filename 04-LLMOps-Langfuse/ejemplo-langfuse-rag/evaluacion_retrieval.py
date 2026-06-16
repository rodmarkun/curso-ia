"""Evalúa solo la recuperación: ¿Chroma devuelve las fuentes esperadas?

Uso:
    uv run python evaluacion_retrieval.py

No llama al LLM. Sirve para separar errores de retrieval de errores de generación.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import SimpleRAG

load_dotenv()

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "eval_cases.json"


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def format_retrieval_summary(report: dict) -> str:
    total = int(report.get("total") or 0)
    passed = int(report.get("passed") or 0)
    score = float(report.get("score") or 0.0)
    lines = [f"Evaluación retrieval: {passed}/{total} casos pasados ({score * 100:.1f}%)."]
    failed_rows = [row for row in report.get("rows", []) if not row.get("retrieval_hit")]
    if not failed_rows:
        lines.append("Todos los casos recuperaron una fuente esperada.")
    else:
        lines.append("Casos fallidos:")
        for row in failed_rows:
            expected = ", ".join(row.get("expected_sources") or []) or "sin fuente esperada"
            retrieved = ", ".join(row.get("retrieved_sources") or []) or "sin resultados"
            lines.append(f"- {row.get('id')}: esperaba [{expected}], recuperó [{retrieved}]")
    return "\n".join(lines)


def evaluate_retrieval(k: int = 4) -> dict:
    rag = SimpleRAG(auto_ingest=True)
    rows = []
    passed = 0

    for case in load_cases():
        expected_sources = set(case.get("expected_sources") or [])
        results = rag.search(case["question"], k=k)
        retrieved_sources = [r.source for r in results]
        retrieved_set = set(retrieved_sources)
        hit = not expected_sources or bool(expected_sources & retrieved_set)
        max_similarity = max((r.score for r in results), default=0.0)
        if hit:
            passed += 1
        rows.append(
            {
                "id": case["id"],
                "type": case["type"],
                "question": case["question"],
                "expected_sources": sorted(expected_sources),
                "retrieved_sources": retrieved_sources,
                "retrieval_hit": hit,
                "max_similarity": round(max_similarity, 4),
            }
        )

    return {
        "metric": "retrieval_source_hit_rate",
        "k": k,
        "passed": passed,
        "total": len(rows),
        "score": round(passed / len(rows), 3) if rows else 0.0,
        "rows": rows,
    }


if __name__ == "__main__":
    report = evaluate_retrieval()
    print(format_retrieval_summary(report))
    print("\nJSON completo:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
