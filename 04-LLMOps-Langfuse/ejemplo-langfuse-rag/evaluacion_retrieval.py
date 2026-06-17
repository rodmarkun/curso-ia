"""Evalúa solo la recuperación: ¿Chroma devuelve los fragmentos esperados?

Uso:
    uv run python evaluacion_retrieval.py

No llama al LLM. Sirve para separar errores de retrieval de errores de generación.

Cada caso indica en `expected_sources` no solo el fichero, sino el/los
fragmento(s) correctos (`fichero -> [frag, ...]`). Distinguimos así dos cosas:
- acierto de fragmento: recuperamos el fragmento exacto que contiene la respuesta.
- acierto de fichero: recuperamos el fichero correcto aunque sea otro fragmento.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (
    expected_files,
    expected_pairs,
    format_expected,
    format_pairs,
    load_cases,
    pct,
    render_table,
)
from rag_core import SimpleRAG

load_dotenv()


def evaluate_retrieval(k: int = 4) -> dict:
    rag = SimpleRAG(auto_ingest=True)
    rows = []
    fragment_hits = 0
    file_hits = 0

    for case in load_cases():
        exp_pairs = expected_pairs(case)
        exp_files = expected_files(case)

        results = rag.search(case["question"], k=k)
        retrieved_pairs = [(r.source, r.chunk_id) for r in results]
        retrieved_files = {source for source, _ in retrieved_pairs}

        matched_pairs = sorted(exp_pairs & set(retrieved_pairs))
        fragment_hit = bool(matched_pairs)
        file_hit = bool(exp_files & retrieved_files)
        max_similarity = max((r.score for r in results), default=0.0)

        if fragment_hit:
            fragment_hits += 1
        if file_hit:
            file_hits += 1

        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected": format_expected(case),
                "expected_sources": case.get("expected_sources") or {},
                "retrieved": retrieved_pairs,
                "matched_fragments": matched_pairs,
                "fragment_hit": fragment_hit,
                "file_hit": file_hit,
                "max_similarity": round(max_similarity, 4),
            }
        )

    total = len(rows)
    return {
        "metric": "retrieval_fragment_hit_rate",
        "k": k,
        "total": total,
        "fragment_hits": fragment_hits,
        "file_hits": file_hits,
        "fragment_hit_rate": round(fragment_hits / total, 3) if total else 0.0,
        "file_hit_rate": round(file_hits / total, 3) if total else 0.0,
        "rows": rows,
    }


def format_retrieval_table(report: dict) -> str:
    headers = ["Caso", "Frag OK", "File OK", "Esperado", "Recuperado (top-k)"]
    rows = []
    for row in report["rows"]:
        rows.append(
            [
                row["id"],
                "✓" if row["fragment_hit"] else "✗",
                "✓" if row["file_hit"] else "✗",
                row["expected"],
                format_pairs(row["retrieved"]),
            ]
        )

    total = report["total"]
    summary = render_table(
        ["Métrica", "Valor"],
        [
            ["Casos", str(total)],
            ["Aciertos de fragmento", f"{report['fragment_hits']}/{total} ({pct(report['fragment_hits'], total)})"],
            ["Aciertos de fichero", f"{report['file_hits']}/{total} ({pct(report['file_hits'], total)})"],
            ["k (fragmentos recuperados)", str(report["k"])],
        ],
    )

    return (
        f"Evaluación de retrieval (k={report['k']})\n\n"
        + render_table(headers, rows)
        + "\n\nResumen\n\n"
        + summary
    )


if __name__ == "__main__":
    report = evaluate_retrieval()
    print(format_retrieval_table(report))
    print("\nJSON completo:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
