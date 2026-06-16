"""Compara versiones de prompt sobre el mismo conjunto de casos.

Uso:
    uv run python evaluacion_comparativa.py

Genera eval_comparativa.json. La evaluación de generación usa un LLM juez.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluacion_generacion import evaluate_generation, format_generation_summary

ROOT = Path(__file__).resolve().parent
PROMPTS_TO_COMPARE = ["rag-basico"]


if __name__ == "__main__":
    reports = []
    for prompt_name in PROMPTS_TO_COMPARE:
        print(f"\n=== Evaluando {prompt_name} ===")
        report = evaluate_generation(prompt_name=prompt_name, model="gpt-oss:120b", k=4)
        reports.append(report)
        print(format_generation_summary(report))

    summary = {
        "comparison": [
            {
                "prompt_name": r["prompt_name"],
                "score": r["score"],
                "passed": r["passed"],
                "total": r["total"],
                "avg_latency_ms": round(sum(row["latency_ms"] for row in r["rows"]) / len(r["rows"]), 2),
                "avg_tokens_estimated": round(sum(row["tokens_estimated"] for row in r["rows"]) / len(r["rows"]), 2),
            }
            for r in reports
        ],
        "reports": reports,
    }
    out = ROOT / "eval_comparativa.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["comparison"], ensure_ascii=False, indent=2))
    print(f"\nResultado completo escrito en {out}")
