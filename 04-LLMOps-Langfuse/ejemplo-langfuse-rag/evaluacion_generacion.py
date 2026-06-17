"""Evalúa la generación con un LLM juez.

Uso:
    uv run python evaluacion_generacion.py --prompt-name rag-basico

Primero ejecuta el RAG y después entrega pregunta, respuesta y contexto recuperado
al LLM juez, que devuelve JSON con `passed`, `score` y `reason`. No busca términos
exactos: juzga semánticamente si la respuesta es correcta y está soportada por el
contexto recuperado.

Además del veredicto del juez (¿respuesta correcta?), la tabla final muestra el
acierto de recuperación a nivel de fragmento, para distinguir fallos de retrieval
de fallos de generación.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (
    expected_pairs,
    format_expected,
    load_cases,
    pct,
    render_table,
)
from rag_core import DEFAULT_MODEL, DEFAULT_PROMPT_NAME, DEFAULT_PROVIDER, SimpleRAG, get_chat_llm

load_dotenv()

ROOT = Path(__file__).resolve().parent


def _json_for_prompt(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_judge_prompt(case: dict, *, answer: str, retrieved_context: list[dict] | None = None) -> str:
    """Construye el prompt del juez evaluando SOLO contra el contexto recuperado.

    Deliberadamente NO le pasamos `expected_sources` (el fragmento "correcto"). El
    juez solo ve la pregunta, el contexto recuperado y la respuesta. Así medimos
    calidad de generación sin mezclarla con fallos de recuperación: si el fragmento
    correcto no se recuperó, eso lo capta la métrica de retrieval, no el juez.
    """

    context_for_judge = []
    for item in retrieved_context or []:
        context_for_judge.append(
            {
                "source": item.get("source"),
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score"),
                "text": str(item.get("text", ""))[:1800],
            }
        )

    judge_payload = {
        "case_id": case.get("id"),
        "question": case.get("question"),
        "retrieved_context": context_for_judge,
        "answer_to_evaluate": answer,
    }

    return f"""Eres un juez estricto para una evaluación RAG de una clase de LLMOps.

Evalúa la respuesta del sistema USANDO ÚNICAMENTE el contexto recuperado que se te entrega.
No conoces cuál era el fragmento "correcto" esperado: juzga solo si la respuesta es coherente
con el contexto disponible. No busques palabras exactas: juzga semánticamente.

Criterios:
1. Si el contexto recuperado contiene la información, la respuesta debe contestar la pregunta apoyándose en él.
2. La respuesta no debe inventar datos que no estén en el contexto recuperado.
3. Si el contexto recuperado NO contiene la información necesaria, la respuesta CORRECTA es reconocer que no está disponible. En ese caso apruébala (`passed=true`): no penalices a la generación por un fallo de recuperación.

Datos del caso:
{_json_for_prompt(judge_payload)}

Devuelve SOLO JSON válido con esta forma exacta:
{{
  "passed": true,
  "score": 0.0,
  "reason": "explicación breve en español"
}}

`score` debe estar entre 0 y 1. Usa `passed=true` solo si la respuesta sería aceptable para enseñar en clase.
"""


def _salvage_judge_fields(raw: str) -> dict | None:
    """Rescata passed/score/reason de una salida JSON parcial o mal formada.

    Sirve para cuando el modelo devuelve JSON truncado o con texto alrededor. Solo
    da algo por válido si al menos encuentra el campo `passed`.
    """

    passed_match = re.search(r'"passed"\s*:\s*(true|false)', raw, flags=re.IGNORECASE)
    if not passed_match:
        return None
    payload: dict = {"passed": passed_match.group(1).lower() == "true"}
    score_match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', raw)
    if score_match:
        payload["score"] = float(score_match.group(1))
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
    if reason_match:
        payload["reason"] = reason_match.group(1)
    return payload


def parse_judge_response(raw: str) -> dict:
    """Parsea la salida JSON del juez, tolerando fences de Markdown y JSON parcial."""

    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    else:
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match:
            text = object_match.group(0)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Antes de rendirnos, intentamos rescatar los campos de un JSON parcial.
        payload = _salvage_judge_fields(raw)
        if payload is None:
            return {
                "passed": False,
                "score": 0.0,
                "reason": f"El juez no devolvió JSON válido. Respuesta cruda: {raw[:300]}",
            }

    passed = bool(payload.get("passed", False))
    try:
        score = float(payload.get("score", 1.0 if passed else 0.0))
    except (TypeError, ValueError):
        score = 1.0 if passed else 0.0
    score = max(0.0, min(1.0, score))
    reason = str(payload.get("reason") or "Sin explicación del juez.").strip()
    return {"passed": passed, "score": score, "reason": reason}


# Instrucciones cada vez más explícitas para forzar JSON válido si el juez falla.
_JUDGE_RETRY_SUFFIXES = (
    "",
    "\n\nRecuerda: devuelve únicamente JSON válido en una sola línea, sin texto ni razonamiento adicional.",
    '\n\nIMPORTANTE: responde EXCLUSIVAMENTE con el objeto JSON {"passed": bool, "score": número, "reason": "texto"}. Nada antes ni después.',
)


def judge_answer(judge_llm, case: dict, *, answer: str, sources: list[dict]) -> dict:
    prompt = build_judge_prompt(case, answer=answer, retrieved_context=sources)
    last_raw = ""
    last_parsed: dict | None = None
    for retry_suffix in _JUDGE_RETRY_SUFFIXES:
        response = judge_llm.invoke(prompt + retry_suffix)
        raw = getattr(response, "content", str(response))
        last_raw = raw
        parsed = parse_judge_response(raw)
        parsed["raw_judge_output"] = raw
        last_parsed = parsed
        if raw.strip() and not parsed["reason"].startswith("El juez no devolvió JSON válido"):
            return parsed
    assert last_parsed is not None
    last_parsed["raw_judge_output"] = last_raw
    return last_parsed


def evaluate_generation(
    prompt_name: str,
    model: str,
    k: int = 4,
    send_to_langfuse: bool = False,
    *,
    provider: str = DEFAULT_PROVIDER,
    base_url: str | None = None,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    judge_base_url: str | None = None,
) -> dict:
    cases = load_cases()
    rag = SimpleRAG(auto_ingest=True)
    judge_llm = get_chat_llm(
        provider=judge_provider or provider,
        model=judge_model or os.getenv("JUDGE_MODEL") or model,
        base_url=judge_base_url or base_url,
        temperature=0,
        num_ctx=16384,
        # Generoso a propósito: gpt-oss y otros modelos "reasoning" pueden gastar
        # tokens en pensamiento interno y devolver el JSON truncado si el límite es
        # bajo. El veredicto es corto, así que sobra presupuesto sin coste relevante.
        num_predict=2048,
        reasoning=False,
        output_format="json" if (judge_provider or provider) == "ollama" else None,
    )
    rows = []
    passed = 0
    fragment_hits = 0

    for case in cases:
        result = rag.answer(
            case["question"],
            k=k,
            model=model,
            provider=provider,
            base_url=base_url,
            prompt_name=prompt_name,
            user_id="eval-script",
            session_id=f"eval-{prompt_name}",
            send_to_langfuse=send_to_langfuse,
            reasoning=False,
        )
        answer = result["answer"]
        sources = result["sources"]

        # Acierto de recuperación a nivel de fragmento (mismo criterio que evaluacion_retrieval).
        retrieved_pairs = {(s["source"], int(s["chunk_id"])) for s in sources}
        fragment_hit = bool(expected_pairs(case) & retrieved_pairs)
        if fragment_hit:
            fragment_hits += 1

        judge = judge_answer(judge_llm, case, answer=answer, sources=sources)
        ok = bool(judge["passed"])
        if ok:
            passed += 1

        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "passed": ok,
                "judge_score": judge["score"],
                "judge_reason": judge["reason"],
                "fragment_hit": fragment_hit,
                "expected": format_expected(case),
                "expected_sources": case.get("expected_sources") or {},
                "retrieved_sources": sorted({s["source"] for s in sources}),
                "trace_id": result["trace"]["trace_id"],
                "latency_ms": result["trace"]["latency_ms"],
                "tokens_estimated": result["trace"]["usage"]["total_tokens_estimated"],
                "langfuse": result["langfuse"],
                "answer_preview": answer[:500],
                "judge_raw_preview": judge.get("raw_judge_output", "")[:500],
            }
        )

    total = len(rows)
    report = {
        "metric": "generation_llm_judge_pass_rate",
        "prompt_name": prompt_name,
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "judge_model": judge_model or os.getenv("JUDGE_MODEL") or model,
        "judge_provider": judge_provider or provider,
        "judge_base_url": judge_base_url or base_url,
        "k": k,
        "passed": passed,
        "total": total,
        "fragment_hits": fragment_hits,
        "answer_pass_rate": round(passed / total, 3) if total else 0.0,
        "fragment_hit_rate": round(fragment_hits / total, 3) if total else 0.0,
        "avg_judge_score": round(sum(row["judge_score"] for row in rows) / total, 3) if total else 0.0,
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / total, 1) if total else 0.0,
        "rows": rows,
    }
    out = ROOT / f"eval_results_{prompt_name}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def format_generation_table(report: dict) -> str:
    headers = ["Caso", "Frag OK", "Resp OK", "Score", "Latencia(ms)", "Motivo del juez"]
    rows = []
    for row in report["rows"]:
        reason = row.get("judge_reason") or ""
        if len(reason) > 70:
            reason = reason[:67] + "..."
        rows.append(
            [
                row["id"],
                "✓" if row["fragment_hit"] else "✗",
                "✓" if row["passed"] else "✗",
                f"{row['judge_score']:.2f}",
                f"{row['latency_ms']:.0f}",
                reason,
            ]
        )

    total = report["total"]
    summary = render_table(
        ["Métrica", "Valor"],
        [
            ["Casos", str(total)],
            ["Respuestas correctas (juez)", f"{report['passed']}/{total} ({pct(report['passed'], total)})"],
            ["Aciertos de recuperación (fragmento)", f"{report['fragment_hits']}/{total} ({pct(report['fragment_hits'], total)})"],
            ["Score medio del juez", f"{report['avg_judge_score']:.3f}"],
            ["Latencia media", f"{report['avg_latency_ms']:.0f} ms"],
        ],
    )

    return (
        f"Evaluación de generación · LLM juez (prompt={report['prompt_name']}, modelo={report['model']})\n\n"
        + render_table(headers, rows)
        + "\n\nResumen\n\n"
        + summary
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-name", default=DEFAULT_PROMPT_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["ollama", "openai-compatible"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-provider", default=None, choices=["ollama", "openai-compatible"])
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--langfuse", action="store_true", help="Enviar trazas de evaluación a Langfuse")
    args = parser.parse_args()

    report = evaluate_generation(
        args.prompt_name,
        args.model,
        args.k,
        args.langfuse,
        provider=args.provider,
        base_url=args.base_url,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        judge_base_url=args.judge_base_url,
    )
    print(format_generation_table(report))
    print("\nJSON completo:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
