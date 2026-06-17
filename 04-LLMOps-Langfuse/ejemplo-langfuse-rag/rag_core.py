"""Núcleo RAG mínimo con Chroma persistente + Ollama Cloud.

Flujo docente:
1. Leemos documentos de una carpeta.
2. Los partimos en chunks.
3. Calculamos embeddings para cada chunk.
4. Guardamos esos vectores en Chroma, una base de datos vectorial persistente.
5. En cada pregunta, Chroma recupera los chunks más cercanos semánticamente.
6. Enviamos esos chunks al LLM en Ollama Cloud para generar la respuesta.

Ingesta incremental:
- Si el documento ya existe en Chroma con el mismo hash, no se vuelve a ingerir.
- Si el documento es nuevo, se ingiere.
- Si el documento cambió, se borran sus chunks antiguos y se reingiere.
- Si un documento desapareció de la carpeta, se eliminan sus chunks antiguos.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Union
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import chromadb
import pandas as pd
from chromadb.api.models.Collection import Collection
from pypdf import PdfReader

# Important for MCP stdio: avoid progress bars/log spam on stdout while loading embeddings.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)

from sentence_transformers import SentenceTransformer


OLLAMA_CLOUD_BASE_URL = "https://ollama.com"
OLLAMA_LOCAL_BASE_URL = "http://localhost:11434"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", OLLAMA_CLOUD_BASE_URL)
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = BASE_DIR / "documentos-ejemplo"
DEFAULT_CHROMA_DIR = BASE_DIR / "chroma_db"
DEFAULT_COLLECTION_NAME = "campus_rag_demo"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
DEFAULT_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "900"))
DEFAULT_REASONING = os.getenv("OLLAMA_REASONING", "false").lower() in {"1", "true", "yes", "on"}

# Toda la integración con Langfuse (Prompt Management + observabilidad) vive en
# langfuse_integration.py. El RAG solo le pide compilar la prompt y enviar la traza.
from langfuse_integration import DEFAULT_PROMPT_NAME, compile_prompt, send_trace


def get_ollama_base_url_for_mode(mode: str, *, current_base_url: str | None = None) -> str:
    """Resolve la URL de Ollama desde el modo elegido en la UI."""

    normalized = (mode or "cloud").strip().lower()
    if normalized == "cloud":
        return OLLAMA_CLOUD_BASE_URL
    if normalized == "local":
        return OLLAMA_LOCAL_BASE_URL
    if normalized == "custom":
        return current_base_url or OLLAMA_BASE_URL
    raise ValueError("Modo Ollama no soportado. Usa 'cloud', 'local' o 'custom'.")


def is_local_ollama_base_url(base_url: str) -> bool:
    """Detecta si la URL apunta a un Ollama local que podemos validar con /api/tags."""

    host = (urlparse(base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def validate_local_ollama_model(model: str, base_url: str, *, timeout: float = 2.0) -> None:
    """Comprueba que Ollama local esté levantado y que el modelo exista.

    ChatOllama falla más tarde con un error menos claro si el modelo no está
    descargado. Esta validación convierte ese caso en un mensaje accionable para
    la demo: arrancar `ollama serve` o ejecutar `ollama pull <modelo>`.
    """

    tags_url = base_url.rstrip("/") + "/api/tags"
    request = Request(tags_url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"No se pudo conectar con Ollama local en {base_url}. "
            "Arranca Ollama con `ollama serve` y vuelve a intentarlo."
        ) from exc

    installed_models = {
        str(item.get("name") or item.get("model") or "")
        for item in payload.get("models", [])
        if isinstance(item, dict)
    }
    model_aliases = {model}
    if ":" not in model:
        # Ollama accepts `gemma4` as shorthand for `gemma4:latest`, while
        # /api/tags reports the installed model as `gemma4:latest`. Accept
        # that normal Ollama shorthand so the preflight check does not block
        # a model that ChatOllama can actually run.
        model_aliases.add(f"{model}:latest")
    if installed_models.isdisjoint(model_aliases):
        available = ", ".join(sorted(m for m in installed_models if m)) or "ninguno"
        raise RuntimeError(
            f"Modelo Ollama local no instalado: {model}. "
            f"Instálalo con `ollama pull {model}`. Modelos disponibles: {available}."
        )


DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


@dataclass
class Chunk:
    """Fragmento de documento que se almacenará como vector en Chroma."""

    text: str
    source: str
    chunk_id: int
    file_hash: str


@dataclass
class SearchResult:
    """Resultado de búsqueda vectorial desde Chroma."""

    text: str
    source: str
    chunk_id: int
    score: float


def list_document_files(docs_dir: Union[Path, str] = DEFAULT_DOCS_DIR) -> list[Path]:
    """Lista los documentos soportados de una carpeta."""

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return []

    supported = {".pdf", ".txt", ".md", ".csv", ".json", ".xlsx"}
    return sorted(p for p in docs_path.iterdir() if p.is_file() and p.suffix.lower() in supported)


def file_sha256(path: Path) -> str:
    """Hash estable para detectar si un documento ya se ingirió o cambió."""

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Página {i}]\n{text}")
    return "\n\n".join(pages)


def _read_spreadsheet(path: Path) -> str:
    # Para una demo RAG, convertimos cada hoja a texto tabular simple.
    sheets = pd.read_excel(path, sheet_name=None)
    parts: list[str] = []
    for sheet_name, df in sheets.items():
        parts.append(f"Hoja: {sheet_name}\n{df.to_csv(index=False)}")
    return "\n\n".join(parts)


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".csv":
        return pd.read_csv(path).to_csv(index=False)
    if suffix == ".xlsx":
        return _read_spreadsheet(path)
    if suffix == ".json":
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
    raise ValueError(f"Formato no soportado: {path.suffix}")


def split_text(text: str, *, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Parte texto en chunks con solape, usando caracteres para mantenerlo muy simple."""

    clean = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = start + chunk_size
        chunks.append(clean[start:end])
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def load_chunks_for_file(path: Path) -> list[Chunk]:
    """Lee un documento y lo convierte en chunks con metadatos."""

    file_hash = file_sha256(path)
    text = _read_file(path)
    return [
        Chunk(text=chunk_text, source=path.name, chunk_id=idx, file_hash=file_hash)
        for idx, chunk_text in enumerate(split_text(text), start=1)
    ]


class SentenceTransformerEmbeddingFunction:
    """Embedding function compatible con Chroma usando SentenceTransformers.

    Lo mantenemos explícito para que los estudiantes vean que Chroma almacena vectores,
    no texto mágico. Chroma llama a esta función al insertar documentos y al consultar.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:  # Chroma espera este nombre de parámetro.
        return self.embed_documents(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        vectors = self.model.encode(input, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)

    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    @property
    def dimension(self) -> int:
        vector = self.model.encode(["test"], normalize_embeddings=True, show_progress_bar=False)[0]
        return int(len(vector))


class SimpleRAG:
    """RAG con Chroma persistente e ingesta incremental."""

    def __init__(
        self,
        docs_dir: Union[Path, str] = DEFAULT_DOCS_DIR,
        *,
        chroma_dir: Union[Path, str] = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        auto_ingest: bool = True,
    ):
        self.docs_dir = Path(docs_dir)
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.embedding_function = SentenceTransformerEmbeddingFunction(embedding_model)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self.last_ingest_report: dict = {}
        if auto_ingest:
            self.last_ingest_report = self.ingest_if_needed()

    @property
    def document_names(self) -> list[str]:
        return [p.name for p in list_document_files(self.docs_dir)]

    @property
    def chunk_count(self) -> int:
        return int(self.collection.count())

    @property
    def embedding_dimension(self) -> int:
        return self.embedding_function.dimension

    def _existing_file_state(self, source: str) -> tuple[str | None, str | None]:
        existing = self.collection.get(where={"source": source}, limit=1, include=["metadatas"])
        metadatas = existing.get("metadatas") or []
        if not metadatas:
            return None, None
        metadata = metadatas[0]
        return str(metadata.get("file_hash")), str(metadata.get("embedding_model"))

    def _delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def _ingest_file(self, path: Path) -> int:
        chunks = load_chunks_for_file(path)
        if not chunks:
            return 0

        ids = [f"{chunk.file_hash}:{chunk.source}:{chunk.chunk_id}" for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "file_hash": chunk.file_hash,
                "embedding_model": self.embedding_model_name,
                "path": str(path),
            }
            for chunk in chunks
        ]
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return len(chunks)

    def ingest_if_needed(self) -> dict:
        """Sincroniza la carpeta de documentos con la colección persistente de Chroma."""

        files = list_document_files(self.docs_dir)
        current_sources = {path.name for path in files}
        report = {
            "new_documents": [],
            "updated_documents": [],
            "unchanged_documents": [],
            "removed_documents": [],
            "chunks_added": 0,
        }

        # Eliminar chunks de documentos que ya no existen en la carpeta.
        existing = self.collection.get(include=["metadatas"])
        existing_sources = {
            metadata.get("source")
            for metadata in (existing.get("metadatas") or [])
            if metadata and metadata.get("source")
        }
        for source in sorted(existing_sources - current_sources):
            self._delete_source(str(source))
            report["removed_documents"].append(source)

        # Ingerir solo documentos nuevos o modificados.
        for path in files:
            current_hash = file_sha256(path)
            stored_hash, stored_embedding_model = self._existing_file_state(path.name)

            if stored_hash == current_hash and stored_embedding_model == self.embedding_model_name:
                report["unchanged_documents"].append(path.name)
                continue

            if stored_hash is None:
                report["new_documents"].append(path.name)
            else:
                report["updated_documents"].append(path.name)
                self._delete_source(path.name)

            report["chunks_added"] += self._ingest_file(path)

        return report

    def search(self, question: str, *, k: int = 4) -> list[SearchResult]:
        """Busca los k chunks más cercanos usando Chroma."""

        if self.collection.count() == 0:
            return []

        result = self.collection.query(
            query_texts=[question],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        output: list[SearchResult] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            # Con hnsw:space=cosine, Chroma devuelve distancia coseno. Convertimos a similitud.
            similarity = 1.0 - float(distance)
            output.append(
                SearchResult(
                    text=text,
                    source=str(metadata.get("source", "desconocido")),
                    chunk_id=int(metadata.get("chunk_id", 0)),
                    score=similarity,
                )
            )
        return output

    def build_prompt(
        self,
        question: str,
        results: Iterable[SearchResult],
        *,
        prompt_name: str = DEFAULT_PROMPT_NAME,
        document_inventory: dict | None = None,
    ) -> tuple[str, dict]:
        """Construye el prompt final desde Langfuse Prompt Management."""

        context_blocks = []
        for result in results:
            context_blocks.append(
                f"[Fuente: {result.source} | fragmento {result.chunk_id} | similitud {result.score:.3f}]\n{result.text}"
            )
        context = "\n\n---\n\n".join(context_blocks) or "No se recuperó contexto."
        document_names = None
        if document_inventory:
            document_names = (document_inventory.get("output") or {}).get("documents")
        document_names = document_names or self.document_names
        documents = "\n".join(f"- {name}" for name in document_names) or "- No hay documentos disponibles."
        return compile_prompt(prompt_name, documents=documents, context=context, question=question)

    def tool_inventario_documentos(self) -> dict:
        """Tool didáctica: lista los documentos disponibles para responder.

        En la sesión 02 vimos que un modelo/agente puede usar herramientas. En
        esta sesión la herramienta es deliberadamente simple y determinista para
        que en Langfuse se vea con claridad cuándo el agente la llama y qué
        devuelve antes de construir el prompt final.
        """

        started = time.perf_counter()
        files = list_document_files(self.docs_dir)
        output = {
            "document_count": len(files),
            "documents": [path.name for path in files],
        }
        return {
            "name": "tool_inventario_documentos",
            "description": "Lista los documentos de la carpeta RAG antes de responder.",
            "input": {"docs_dir": str(self.docs_dir)},
            "output": output,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    @staticmethod
    def _estimate_usage(prompt: str, answer: str) -> dict:
        """Estimación simple para clase. No sustituye al usage real del proveedor."""

        prompt_tokens = max(1, round(len(prompt) / 4))
        completion_tokens = max(1, round(len(answer) / 4))
        return {
            "prompt_tokens_estimated": prompt_tokens,
            "completion_tokens_estimated": completion_tokens,
            "total_tokens_estimated": prompt_tokens + completion_tokens,
            "note": "Estimación aproximada por caracteres; usa usage real si el proveedor lo devuelve.",
        }

    @staticmethod
    def _source_payload(results: Iterable[SearchResult]) -> list[dict]:
        return [
            {
                "source": result.source,
                "chunk_id": result.chunk_id,
                "score": result.score,
                "text": result.text,
            }
            for result in results
        ]

    def answer(
        self,
        question: str,
        *,
        k: int = 4,
        model: str = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
        reasoning: bool = DEFAULT_REASONING,
        prompt_name: str = DEFAULT_PROMPT_NAME,
        user_id: str = "test",
        session_id: str | None = None,
        send_to_langfuse: bool | None = None,
    ) -> dict:
        """Recupera contexto, genera respuesta y opcionalmente envía observabilidad a Langfuse."""

        started = time.perf_counter()
        session_id = session_id or "sesion-04-demo"
        document_inventory = self.tool_inventario_documentos()
        results = self.search(question, k=k)
        prompt, prompt_meta = self.build_prompt(
            question,
            results,
            prompt_name=prompt_name,
            document_inventory=document_inventory,
        )
        llm = get_chat_llm(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            reasoning=reasoning,
        )
        response = llm.invoke(prompt)
        answer_text = getattr(response, "content", str(response))
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        sources = self._source_payload(results)
        max_similarity = max((s["score"] for s in sources), default=0.0)
        model_config = {
            "provider": provider,
            "model": model,
            "base_url": base_url or _default_base_url_for_provider(provider),
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "reasoning": reasoning,
        }
        trace = {
            "trace_id": uuid.uuid4().hex,
            "session_id": session_id,
            "user_id": user_id,
            "question": question,
            "answer": answer_text,
            "model": model,
            "model_config": model_config,
            "prompt_name": prompt_meta.get("name", prompt_name),
            "prompt_version": prompt_meta.get("version"),
            "prompt_source": prompt_meta.get("source"),
            "prompt_labels": prompt_meta.get("labels", []),
            "prompt_config": prompt_meta.get("config", {}),
            "prompt": prompt,
            "k": k,
            "tool_calls": [document_inventory],
            "sources": sources,
            "latency_ms": latency_ms,
            "usage": self._estimate_usage(prompt, answer_text),
            "scores": {
                "has_sources": 1.0 if sources else 0.0,
                "max_similarity": round(max_similarity, 4),
                "retrieval_confidence_demo": round(max_similarity, 4),
            },
        }
        langfuse_status = send_trace(trace, enabled=send_to_langfuse)
        return {
            "answer": answer_text,
            "sources": sources,
            "prompt": prompt,
            "trace": trace,
            "langfuse": langfuse_status,
        }


def _default_base_url_for_provider(provider: str) -> str:
    """URL por defecto según proveedor para que quede trazada aunque venga de .env."""

    if provider == "openai-compatible":
        return DEFAULT_OPENAI_BASE_URL
    return OLLAMA_BASE_URL


def get_chat_llm(
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    reasoning: bool = DEFAULT_REASONING,
    output_format: Literal["json"] | None = None,
):
    """Crea el cliente LLM para la demo.

    - ``provider='ollama'``: Ollama Cloud o servidor Ollama local. Para local,
      usa ``base_url='http://localhost:11434'`` y un modelo descargado con
      ``ollama pull``.
    - ``provider='openai-compatible'``: endpoints vLLM/Runpod que exponen una
      API compatible con ``/v1/chat/completions``.
    """

    provider = (provider or DEFAULT_PROVIDER).lower()
    base_url = base_url or _default_base_url_for_provider(provider)

    if provider == "openai-compatible":
        if not base_url:
            raise RuntimeError("Falta OPENAI_BASE_URL para provider=openai-compatible.")
        from langchain_openai import ChatOpenAI

        normalized_base_url = base_url.rstrip("/")
        if not normalized_base_url.endswith("/v1"):
            normalized_base_url = normalized_base_url + "/v1"
        return ChatOpenAI(
            model=model,
            base_url=normalized_base_url,
            api_key=api_key or DEFAULT_OPENAI_API_KEY or "not-needed",
            temperature=temperature,
            max_tokens=num_predict,
        )

    if provider != "ollama":
        raise RuntimeError(f"Proveedor no soportado: {provider}. Usa 'ollama' u 'openai-compatible'.")

    # Ollama local no necesita API key. Ollama Cloud sí.
    if "ollama.com" in base_url and not os.getenv("OLLAMA_API_KEY"):
        raise RuntimeError(
            "Falta OLLAMA_API_KEY. Para Ollama Cloud crea un .env desde .env.example; "
            "para Ollama local cambia OLLAMA_BASE_URL a http://localhost:11434."
        )
    if is_local_ollama_base_url(base_url):
        validate_local_ollama_model(model, base_url)

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        reasoning=reasoning,
        format=output_format,
    )
