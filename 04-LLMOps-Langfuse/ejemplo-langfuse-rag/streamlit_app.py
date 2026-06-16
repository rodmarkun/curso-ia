"""App Streamlit para enseñar RAG + Langfuse + prompt management + evaluación.

Ejecutar:
    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_DOCS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_PROVIDER,
    OLLAMA_BASE_URL,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_PROMPT_NAME,
    DEFAULT_REASONING,
    DEFAULT_TEMPERATURE,
    OLLAMA_CLOUD_BASE_URL,
    OLLAMA_LOCAL_BASE_URL,
    SimpleRAG,
    get_ollama_base_url_for_mode,
    list_available_prompts,
)

load_dotenv(override=True)

st.set_page_config(page_title="Demo LLMOps: RAG + Langfuse", page_icon="📊", layout="wide")

st.title("📊 Demo LLMOps: RAG + Langfuse + evaluación")
st.caption(
    "Partimos de la app RAG de la sesión 03 y añadimos trazas, Prompt Management, métricas y evaluación de comportamiento."
)

with st.sidebar:
    st.header("Configuración")
    docs_dir = Path(st.text_input("Carpeta de documentos", value=os.getenv("DOCS_DIR", str(DEFAULT_DOCS_DIR))))
    chroma_dir = Path(st.text_input("Carpeta Chroma persistente", value=os.getenv("CHROMA_DIR", str(DEFAULT_CHROMA_DIR))))
    embedding_model = st.text_input(
        "Modelo de embeddings",
        value=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    k = st.slider("Fragmentos recuperados", min_value=1, max_value=12, value=4)

    st.markdown("### Prompt Management")
    langfuse_env_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    send_to_langfuse = st.toggle("Enviar trazas a Langfuse", value=langfuse_env_enabled)
    if send_to_langfuse:
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            st.success("Langfuse configurado · crea la prompt manualmente en Prompt Management")
        else:
            st.warning("Faltan claves de Langfuse; se seguirá mostrando la traza local.")
    else:
        st.info("Langfuse desactivado: se usa fallback local y se muestra traza JSON.")

    if st.button("Refrescar prompts desde Langfuse"):
        st.cache_data.clear()
        st.rerun()


@st.cache_data(ttl=30)
def cached_available_prompts() -> list[dict]:
    return list_available_prompts()


with st.sidebar:
    available_prompts = cached_available_prompts()
    prompt_names = [p["name"] for p in available_prompts] or [DEFAULT_PROMPT_NAME]
    default_index = prompt_names.index(DEFAULT_PROMPT_NAME) if DEFAULT_PROMPT_NAME in prompt_names else 0
    prompt_name = st.selectbox("Prompt", options=prompt_names, index=default_index)
    selected_prompt = next((p for p in available_prompts if p["name"] == prompt_name), available_prompts[0])
    prompt_config = selected_prompt.get("config") or {}
    st.caption(
        f"Fuente: {selected_prompt.get('source')} · versión: {selected_prompt.get('version') or 'n/a'} · labels: {', '.join(selected_prompt.get('labels') or []) or 'n/a'}"
    )
    with st.expander("Config de la prompt"):
        st.json(prompt_config)

    st.markdown("### Configuración del modelo")
    provider_options = ["ollama", "openai-compatible"]
    default_provider = DEFAULT_PROVIDER if DEFAULT_PROVIDER in provider_options else "ollama"
    provider = st.selectbox("Proveedor", options=provider_options, index=provider_options.index(default_provider))
    ollama_mode = None
    if provider == "ollama":
        configured_ollama_base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
        default_mode = "cloud"
        if configured_ollama_base_url.rstrip("/") == OLLAMA_LOCAL_BASE_URL:
            default_mode = "local"
        elif configured_ollama_base_url.rstrip("/") != OLLAMA_CLOUD_BASE_URL:
            default_mode = "custom"
        ollama_modes = {
            "cloud": "Ollama Cloud",
            "local": "Ollama local",
            "custom": "Ollama remoto/custom",
        }
        ollama_mode_labels = list(ollama_modes.values())
        ollama_mode_by_label = {label: value for value, label in ollama_modes.items()}
        selected_ollama_label = st.selectbox(
            "Tipo de Ollama",
            options=ollama_mode_labels,
            index=list(ollama_modes).index(default_mode),
        )
        ollama_mode = ollama_mode_by_label[selected_ollama_label]
        if ollama_mode == "custom":
            custom_base_url = st.text_input(
                "Base URL del modelo",
                value=configured_ollama_base_url,
                help="Servidor Ollama remoto que expone la API de Ollama, por ejemplo http://192.168.1.50:11434.",
            )
            base_url = get_ollama_base_url_for_mode("custom", current_base_url=custom_base_url)
        else:
            base_url = get_ollama_base_url_for_mode(ollama_mode)
            st.text_input(
                "Base URL del modelo",
                value=base_url,
                disabled=True,
                help="Se elige automáticamente según el tipo de Ollama.",
            )
    else:
        default_base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
        base_url_help = "Runpod/vLLM: pega la URL pública del endpoint; la app añade /v1 si hace falta."
        base_url = st.text_input("Base URL del modelo", value=default_base_url, help=base_url_help)
    model = st.text_input("Modelo", value=str(prompt_config.get("model") or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)))
    api_key = None
    if provider == "openai-compatible":
        api_key = st.text_input(
            "API key OpenAI-compatible (si el endpoint la requiere)",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
    # These generation parameters belong to Langfuse Prompt Management.
    # The Streamlit UI reads them from the selected prompt config but does not
    # expose duplicate controls, so the live demo has a single source of truth.
    temperature = float(prompt_config.get("temperature", DEFAULT_TEMPERATURE))
    num_ctx = int(prompt_config.get("num_ctx", DEFAULT_NUM_CTX))
    num_predict = int(prompt_config.get("num_predict", DEFAULT_NUM_PREDICT))
    reasoning_config = prompt_config.get("reasoning", DEFAULT_REASONING)
    if isinstance(reasoning_config, str):
        reasoning = reasoning_config.lower() in {"1", "true", "yes", "on"}
    else:
        reasoning = bool(reasoning_config)
    if provider == "ollama" and ollama_mode == "custom":
        # Remote Ollama endpoints used in class, especially RunPod with Qwen
        # thinking models, can return empty content if reasoning/thinking is
        # enabled. Force normal answer content for remote Ollama demos.
        reasoning = False

    user_id = st.text_input("user_id demo", value="test")
    session_id = st.text_input("session_id demo", value="sesion-04-live")

    if provider == "ollama" and "ollama.com" in base_url and os.getenv("OLLAMA_API_KEY"):
        st.success("OLLAMA_API_KEY encontrada para Ollama Cloud")
    elif provider == "ollama" and "ollama.com" in base_url:
        st.warning("Falta OLLAMA_API_KEY para Ollama Cloud. Para local usa http://localhost:11434.")
    elif provider == "ollama":
        st.info("Modo Ollama local: asegúrate de tener `ollama serve` y el modelo descargado.")
    else:
        st.info("Modo OpenAI-compatible: útil para Runpod/vLLM.")

    if st.button("Sincronizar documentos con Chroma"):
        st.cache_resource.clear()
        st.rerun()


@st.cache_resource(show_spinner="Sincronizando documentos con Chroma...")
def get_rag(docs_dir_str: str, chroma_dir_str: str, embedding_model_name: str) -> SimpleRAG:
    return SimpleRAG(
        Path(docs_dir_str),
        chroma_dir=Path(chroma_dir_str),
        embedding_model=embedding_model_name,
        auto_ingest=True,
    )


rag = get_rag(str(docs_dir), str(chroma_dir), embedding_model)

metrics_left, metrics_mid, metrics_right, metrics_fourth = st.columns(4)
with metrics_left:
    st.metric("Chunks en Chroma", rag.chunk_count)
with metrics_mid:
    st.metric("Dimensión embedding", rag.embedding_dimension)
with metrics_right:
    st.metric("Prompt", prompt_name)
with metrics_fourth:
    st.metric("Modelo", model)

st.subheader("Pregunta a tus documentos")
question = st.text_input(
    "Pregunta",
    value="¿Qué prácticas de laboratorio tiene Sistemas Digitales?",
    placeholder="¿Qué prácticas de laboratorio tiene Sistemas Digitales?",
)

if st.button("Responder", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Buscando en Chroma, generando respuesta y creando traza..."):
            result = rag.answer(
                question,
                k=k,
                model=model,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                num_ctx=int(num_ctx),
                num_predict=int(num_predict),
                reasoning=reasoning,
                prompt_name=prompt_name,
                user_id=user_id,
                session_id=session_id,
                send_to_langfuse=send_to_langfuse,
            )
    except Exception as exc:
        st.exception(exc)
    else:
        st.markdown("### Respuesta")
        st.write(result["answer"])
        answer_text = str(result["answer"]).strip()
        if answer_text and len(answer_text) < 120 and not answer_text.endswith((".", "!", "?", ":")):
            st.warning(
                "La respuesta parece cortada o de baja calidad. Para la demo usa un modelo instruct fiable "
                "como `gpt-oss:120b`; si estás en Ollama local, prueba el nombre completo del modelo "
                "o cambia a otro modelo. La traza de Langfuse permite comparar este fallo con una respuesta buena."
            )

        trace = result["trace"]
        st.markdown("### Métricas de traza")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Latencia", f"{trace['latency_ms']} ms")
        c2.metric("Tokens estimados", trace["usage"]["total_tokens_estimated"])
        c3.metric("Fuentes", len(trace["sources"]))
        c4.metric("Max similitud", trace["scores"]["max_similarity"])
        c5.metric("User", trace["user_id"])

        langfuse_status = result["langfuse"]
        if langfuse_status.get("sent"):
            st.success(f"Traza enviada a Langfuse: {langfuse_status.get('trace_id')}")
            if langfuse_status.get("trace_url"):
                st.link_button("Abrir traza en Langfuse", langfuse_status["trace_url"])
        else:
            st.info(f"Langfuse: {langfuse_status.get('reason')}")

        st.markdown("### Fragmentos recuperados desde Chroma")
        for i, source in enumerate(result["sources"], start=1):
            with st.expander(
                f"{i}. {source['source']} · fragmento {source['chunk_id']} · similitud {source['score']:.3f}"
            ):
                st.text(source["text"][:2000])

        st.markdown("### Herramienta llamada por el agente")
        for tool_call in trace.get("tool_calls", []):
            with st.expander(f"{tool_call['name']} · {tool_call['duration_ms']} ms", expanded=True):
                st.caption(tool_call["description"])
                st.json({"input": tool_call["input"], "output": tool_call["output"]})

        with st.expander("Ver prompt enviado al modelo"):
            st.code(result["prompt"], language="text")

        with st.expander("Ver traza local completa JSON"):
            st.json(trace)

st.divider()
st.markdown(
    """
### Qué enseña esta app

1. **Retrieval**: qué chunks recupera Chroma y con qué similitud.
2. **Tool use**: el agente llama una herramienta (`tool_inventario_documentos`) antes de generar.
3. **Generation**: qué prompt ve el modelo y qué respuesta produce.
4. **Prompt Management**: el dropdown sale de las prompts disponibles en Langfuse.
5. **Model config**: modelo, temperatura, contexto y salida quedan trazados.
6. **Observabilidad**: una traza local explica qué pasó incluso sin Langfuse.
7. **Evaluación**: los scripts de evaluación separan fallos de recuperación y fallos de generación.
"""
)
