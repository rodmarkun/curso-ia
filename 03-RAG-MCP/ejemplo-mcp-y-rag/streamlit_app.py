"""App Streamlit mínima para enseñar RAG con Chroma + Ollama Cloud.

Ejecutar:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_core import DEFAULT_CHROMA_DIR, DEFAULT_DOCS_DIR, DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL, SimpleRAG


load_dotenv()

st.set_page_config(page_title="Demo RAG + Chroma + MCP", page_icon="📚", layout="wide")

st.title("📚 Demo RAG con Chroma persistente + Ollama Cloud")
st.caption(
    "Carga documentos locales, los guarda como embeddings en Chroma, recupera vectores similares y usa un LLM para responder con contexto."
)

with st.sidebar:
    st.header("Configuración")
    docs_dir = Path(st.text_input("Carpeta de documentos", value=os.getenv("DOCS_DIR", str(DEFAULT_DOCS_DIR))))
    chroma_dir = Path(st.text_input("Carpeta Chroma persistente", value=os.getenv("CHROMA_DIR", str(DEFAULT_CHROMA_DIR))))
    model = st.text_input("Modelo Ollama Cloud", value=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    embedding_model = st.text_input(
        "Modelo de embeddings",
        value=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    k = st.slider("Fragmentos recuperados", min_value=1, max_value=8, value=4)

    if os.getenv("OLLAMA_API_KEY"):
        st.success("OLLAMA_API_KEY encontrada")
    else:
        st.warning("Falta OLLAMA_API_KEY. Copia .env.example a .env y añade tu clave.")

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

left, right = st.columns([1, 3])

with left:
    st.metric("Chunks en Chroma", rag.chunk_count)
    st.metric("Dimensión embedding", rag.embedding_dimension)

with right:
    st.subheader("Pregunta a tus documentos")
    question = st.text_input(
        "Pregunta",
        value="¿Qué prácticas de laboratorio tiene Sistemas Digitales?",
        placeholder="¿Qué prácticas de laboratorio tiene Sistemas Digitales?",
    )

    if st.button("Responder", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Buscando en Chroma y consultando Ollama Cloud..."):
                result = rag.answer(question, k=k, model=model)
        except Exception as exc:
            st.exception(exc)
        else:
            st.markdown("### Respuesta")
            st.write(result["answer"])

            st.markdown("### Fragmentos recuperados desde Chroma")
            for i, source in enumerate(result["sources"], start=1):
                with st.expander(
                    f"{i}. {source['source']} · fragmento {source['chunk_id']} · similitud {source['score']:.3f}"
                ):
                    st.text(source["text"][:2000])

            with st.expander("Ver prompt enviado al modelo"):
                st.code(result["prompt"], language="text")

st.divider()
st.markdown(
    """
**Qué está pasando:**

1. La app lee PDFs, CSV, Excel y JSON de la carpeta de documentos.
2. Calcula un hash SHA-256 de cada documento.
3. Si el documento no está en Chroma, o cambió, lo divide en chunks y lo ingiere.
4. Chroma calcula/guarda embeddings persistentes para cada chunk.
5. La pregunta se convierte en embedding y Chroma recupera los chunks más cercanos.
6. La app construye un prompt con esos chunks.
7. Ollama Cloud genera la respuesta final mediante `ChatOllama`.
"""
)
