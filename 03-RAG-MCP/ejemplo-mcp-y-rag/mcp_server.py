"""Servidor MCP mínimo que expone el RAG con Chroma como herramientas.

Ejecutar en modo stdio:
    python mcp_server.py

Un cliente MCP verá herramientas como:
- listar_documentos
- estado_ingesta
- sincronizar_documentos
- buscar_en_documentos
- responder_con_rag

Este servidor reutiliza `rag_core.py`, por lo que usa los mismos documentos y la misma base
Chroma persistente que la app Streamlit.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from rag_core import DEFAULT_CHROMA_DIR, DEFAULT_DOCS_DIR, DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL, SimpleRAG


load_dotenv()

DOCS_DIR = Path(os.getenv("DOCS_DIR", str(DEFAULT_DOCS_DIR)))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(DEFAULT_CHROMA_DIR)))
MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

mcp = FastMCP("campus-rag-demo")
rag = SimpleRAG(DOCS_DIR, chroma_dir=CHROMA_DIR, embedding_model=EMBEDDING_MODEL, auto_ingest=True)


@mcp.tool()
def listar_documentos() -> list[str]:
    """Lista los documentos disponibles en la carpeta de la práctica."""

    return rag.document_names


@mcp.tool()
def estado_ingesta() -> dict:
    """Muestra el estado de la base Chroma persistente y la última sincronización."""

    return {
        "docs_dir": str(DOCS_DIR),
        "chroma_dir": str(CHROMA_DIR),
        "document_count": len(rag.document_names),
        "chunk_count": rag.chunk_count,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": rag.embedding_dimension,
        "last_ingest_report": rag.last_ingest_report,
    }


@mcp.tool()
def sincronizar_documentos() -> dict:
    """Sincroniza la carpeta de documentos con Chroma: añade nuevos y actualiza modificados."""

    rag.last_ingest_report = rag.ingest_if_needed()
    return estado_ingesta()


@mcp.tool()
def buscar_en_documentos(pregunta: str, k: int = 4) -> list[dict]:
    """Busca fragmentos relevantes por similitud vectorial en Chroma, sin llamar al LLM."""

    results = rag.search(pregunta, k=k)
    return [
        {
            "source": result.source,
            "chunk_id": result.chunk_id,
            "score": round(result.score, 4),
            "text": result.text[:1200],
        }
        for result in results
    ]


@mcp.tool()
def responder_con_rag(pregunta: str, k: int = 4) -> dict:
    """Responde una pregunta usando búsqueda vectorial en Chroma + Ollama Cloud."""

    result = rag.answer(pregunta, k=k, model=MODEL)
    # Recortamos el texto de las fuentes para que la respuesta MCP sea manejable.
    return {
        "answer": result["answer"],
        "sources": [
            {
                "source": source["source"],
                "chunk_id": source["chunk_id"],
                "score": round(source["score"], 4),
            }
            for source in result["sources"]
        ],
    }


@mcp.resource("documentos://lista")
def recurso_lista_documentos() -> str:
    """Recurso MCP de solo lectura con los nombres de documentos disponibles."""

    return "\n".join(rag.document_names)


if __name__ == "__main__":
    mcp.run()
