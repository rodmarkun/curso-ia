"""Toda la integración con Langfuse en un solo archivo.

Idea docente: `rag_core.py` es un RAG normal. Este módulo es *lo único* que hay
que añadir para tener observabilidad y Prompt Management. La prompt vive en
Langfuse y no hay copia en el código: si no se puede leer, `compile_prompt` falla
con un error claro. La observabilidad sí degrada en silencio: si Langfuse está
apagado, simplemente no se envía la traza y la app sigue.

Dos bloques:
1. PROMPT MANAGEMENT: leer/compilar la prompt desde Langfuse.
2. OBSERVABILIDAD: enviar la traza (spans, generation, scores) a Langfuse.
"""

from __future__ import annotations

import os


DEFAULT_PROMPT_NAME = os.getenv("LANGFUSE_PROMPT_NAME", "rag-basico")
DEFAULT_PROMPT_LABEL = os.getenv("LANGFUSE_PROMPT_LABEL", "production")


def is_enabled() -> bool:
    """¿Está activada la integración con Langfuse vía LANGFUSE_ENABLED?"""

    return os.getenv("LANGFUSE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


def get_client():
    """Crea cliente Langfuse si hay credenciales; devuelve None si no está configurado."""

    if not is_enabled():
        return None
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    from langfuse import Langfuse

    return Langfuse()


# ---------------------------------------------------------------------------
# 1. PROMPT MANAGEMENT
# ---------------------------------------------------------------------------


_CREATE_PROMPT_HINT = (
    "Créala manualmente en Langfuse Prompt Management (copia el contenido de "
    "../PROMPT.txt) y pulsa «Refrescar prompts desde Langfuse»."
)


def list_available_prompts() -> list[dict]:
    """Lista las prompts disponibles en Langfuse.

    Si Langfuse no está configurado o no hay prompts todavía, devuelve una lista
    vacía: ya no hay copia local de la prompt en el código.
    """

    client = get_client()
    if client is None:
        return []

    try:
        response = client.api.prompts.list(limit=100)
        prompts = []
        for item in response.data:
            prompt = client.get_prompt(item.name, label=DEFAULT_PROMPT_LABEL, cache_ttl_seconds=0)
            prompts.append(
                {
                    "name": item.name,
                    "version": getattr(prompt, "version", None),
                    "labels": list(getattr(prompt, "labels", []) or []),
                    "source": "langfuse",
                    "config": getattr(prompt, "config", {}) or {},
                }
            )
        return prompts
    except Exception:
        return []


def compile_prompt(prompt_name: str, *, documents: str, context: str, question: str) -> tuple[str, dict]:
    """Obtiene y compila una prompt desde Langfuse.

    La prompt vive en Langfuse, no en el código. Si no se puede leer, falla con un
    error claro en lugar de inventar una prompt local.
    """

    client = get_client()
    if client is None:
        raise RuntimeError(
            f"Langfuse no está configurado, así que no existe la prompt `{prompt_name}`. "
            f"Activa LANGFUSE_ENABLED y configura las claves. {_CREATE_PROMPT_HINT}"
        )

    try:
        prompt_client = client.get_prompt(prompt_name, label=DEFAULT_PROMPT_LABEL, cache_ttl_seconds=0)
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo leer la prompt `{prompt_name}` desde Langfuse. {_CREATE_PROMPT_HINT} "
            f"Detalle: {exc}"
        ) from exc

    prompt = prompt_client.compile(documents=documents, context=context, question=question)
    return prompt, {
        "name": prompt_client.name,
        "source": "langfuse",
        "version": prompt_client.version,
        "labels": list(prompt_client.labels or []),
        "config": prompt_client.config or {},
    }


# ---------------------------------------------------------------------------
# 2. OBSERVABILIDAD (tracing)
# ---------------------------------------------------------------------------


def send_trace(trace: dict, *, enabled: bool | None = None) -> dict:
    """Envía una traza a Langfuse si está configurado; nunca bloquea la demo local."""

    if enabled is None:
        enabled = is_enabled()
    if not enabled:
        return {"sent": False, "reason": "LANGFUSE_ENABLED no está activo; no se envía observabilidad."}
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return {"sent": False, "reason": "Faltan LANGFUSE_PUBLIC_KEY o LANGFUSE_SECRET_KEY."}

    try:
        from langfuse import Langfuse, propagate_attributes

        client = Langfuse()
        release = os.getenv("LANGFUSE_RELEASE", "sesion-04-demo")
        version = str(trace.get("prompt_version") or trace.get("prompt_name") or "unknown")
        tags = ["curso-ia", "rag", "llmops"]

        # In Langfuse SDK v4, user_id/session_id are trace-level attributes, not
        # regular metadata. They must be propagated before creating observations;
        # otherwise they only appear buried in metadata and the Langfuse trace list
        # shows empty User/Session columns.
        with propagate_attributes(
            user_id=str(trace.get("user_id") or "test"),
            session_id=str(trace.get("session_id") or "sesion-04-demo"),
            version=version,
            tags=tags,
            trace_name="rag_answer",
        ):
            with client.start_as_current_observation(
                name="rag_answer",
                as_type="chain",
                input=trace["question"],
                output=trace["answer"],
                metadata={
                    "prompt_name": trace.get("prompt_name"),
                    "prompt_version": trace.get("prompt_version"),
                    "prompt_source": trace.get("prompt_source"),
                    "model_config": trace.get("model_config"),
                    "retrieved_sources": [s["source"] for s in trace["sources"]],
                    "k": trace["k"],
                    "release": release,
                },
                version=version,
                trace_context={"trace_id": trace["trace_id"]},
            ) as rag_span:
                rag_span.set_trace_io(input=trace["question"], output=trace["answer"])
                for tool_call in trace.get("tool_calls", []):
                    with rag_span.start_as_current_observation(
                        name=tool_call.get("name", "tool_call"),
                        as_type="tool",
                        input=tool_call.get("input"),
                        output=tool_call.get("output"),
                        metadata={
                            "description": tool_call.get("description"),
                            "duration_ms": tool_call.get("duration_ms"),
                            "lesson_reference": "Sesión 02: modelos usando herramientas",
                        },
                    ):
                        pass
                with rag_span.start_as_current_observation(
                    name="retrieval",
                    as_type="retriever",
                    input=trace["question"],
                    output=trace["sources"],
                    metadata={"k": trace["k"], "max_similarity": trace["scores"]["max_similarity"]},
                ):
                    pass
                with rag_span.start_as_current_observation(
                    name="ollama_generation",
                    as_type="generation",
                    input=trace["prompt"],
                    output=trace["answer"],
                    model=trace["model"],
                    usage_details={
                        "input": trace["usage"]["prompt_tokens_estimated"],
                        "output": trace["usage"]["completion_tokens_estimated"],
                        "total": trace["usage"]["total_tokens_estimated"],
                    },
                ):
                    pass
                rag_span.score_trace(name="has_sources", value=trace["scores"]["has_sources"])
                rag_span.score_trace(name="max_similarity", value=trace["scores"]["max_similarity"])
                rag_span.score_trace(name="latency_ms", value=trace["latency_ms"])

        client.flush()
        trace_url = None
        if hasattr(client, "get_trace_url"):
            try:
                trace_url = client.get_trace_url(trace_id=trace["trace_id"])
            except Exception:
                # Some self-hosted/local setups protect the helper endpoint used
                # by get_trace_url. The trace itself has already been flushed;
                # avoid turning a missing convenience URL into a failed demo.
                trace_url = None
        return {"sent": True, "trace_id": trace["trace_id"], "trace_url": trace_url}
    except Exception as exc:  # pragma: no cover - ruta de resiliencia para clase
        return {"sent": False, "reason": f"Error enviando a Langfuse: {exc}"}
