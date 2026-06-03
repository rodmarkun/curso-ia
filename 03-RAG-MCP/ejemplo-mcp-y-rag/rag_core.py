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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

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


OLLAMA_BASE_URL = "https://ollama.com"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = BASE_DIR / "documentos-ejemplo"
DEFAULT_CHROMA_DIR = BASE_DIR / "chroma_db"
DEFAULT_COLLECTION_NAME = "campus_rag_demo"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
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


class ChromaRAG:
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

    def build_prompt(self, question: str, results: Iterable[SearchResult]) -> str:
        context_blocks = []
        for result in results:
            context_blocks.append(
                f"[Fuente: {result.source} | fragmento {result.chunk_id} | similitud {result.score:.3f}]\n{result.text}"
            )
        context = "\n\n---\n\n".join(context_blocks) or "No se recuperó contexto."
        documents = "\n".join(f"- {name}" for name in self.document_names) or "- No hay documentos disponibles."

        return f"""Eres un asistente docente experto para una práctica de RAG.
Tu objetivo es dar respuestas útiles, claras y bien justificadas usando el contexto recuperado.

REGLAS IMPORTANTES:
1. Responde en el mismo idioma de la pregunta del usuario.
2. Usa SOLO la información del contexto recuperado y la lista de documentos disponibles.
3. Si falta información para responder con seguridad, dilo claramente: "No lo sé con los documentos disponibles".
4. No inventes datos, cifras, nombres de prácticas, criterios ni conclusiones que no aparezcan en el contexto.
5. Sintetiza: no copies fragmentos largos; explica con tus palabras y conecta ideas relacionadas.
6. Cita las fuentes usadas al final de las frases o viñetas con el nombre del archivo, por ejemplo: (inventario.csv).
7. Si la pregunta pide una visión general de los documentos, organiza la respuesta por documento cuando el contexto lo permita.
8. Si la pregunta pide pasos, criterios o recomendaciones, usa viñetas breves y accionables.

DOCUMENTOS DISPONIBLES:
{documents}

CONTEXTO RECUPERADO DESDE CHROMA POR BÚSQUEDA VECTORIAL:
{context}

PREGUNTA DEL USUARIO:
{question}

RESPUESTA:"""

    def answer(self, question: str, *, k: int = 4, model: str = DEFAULT_MODEL) -> dict:
        """Recupera contexto con Chroma y llama a Ollama Cloud para generar una respuesta."""

        results = self.search(question, k=k)
        prompt = self.build_prompt(question, results)
        llm = get_ollama_llm(model=model)
        response = llm.invoke(prompt)
        answer_text = getattr(response, "content", str(response))
        return {
            "answer": answer_text,
            "sources": [
                {
                    "source": result.source,
                    "chunk_id": result.chunk_id,
                    "score": result.score,
                    "text": result.text,
                }
                for result in results
            ],
            "prompt": prompt,
        }


# Alias corto para mantener imports sencillos en Streamlit y MCP.
SimpleRAG = ChromaRAG


def get_ollama_llm(*, model: str = DEFAULT_MODEL):
    """Crea el cliente LLM igual que en la práctica anterior: Ollama Cloud + LangChain."""

    if not os.getenv("OLLAMA_API_KEY"):
        raise RuntimeError(
            "Falta OLLAMA_API_KEY. Crea un .env a partir de .env.example o exporta la variable."
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        # Damos margen al modelo 120B para leer varios chunks y responder con detalle.
        num_ctx=8192,
        num_predict=900,
    )
