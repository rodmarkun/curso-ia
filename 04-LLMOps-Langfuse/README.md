# Sesión 04 — LLMOps + Langfuse

Objetivo de la sesión: convertir el RAG de la sesión 03 en una aplicación de IA **observable, evaluable y operable**.

La sesión sigue `GUION_DEMO_EN_VIVO.md`. No hay documentos auxiliares obligatorios: el guion, la presentación y la app son la fuente de verdad.

## Materiales que sí se usan

- `sesion-04-llmops-langfuse.html` — presentación visual de apoyo.
- `INSTRUCCIONES-LANGFUSE` — guía para estudiantes: levantar Langfuse desde el repo con `docker compose up -d`.
- `PROMPT.txt` — prompt `rag-basico` que hay que crear en Langfuse después de instalarlo desde cero.
- `ejemplo-langfuse-rag/` — aplicación Streamlit con RAG, Chroma, Langfuse, Prompt Management y evaluación.

## Arranque rápido de la app

```bash
cd 04-LLMOps-Langfuse/ejemplo-langfuse-rag
cp .env.example .env
uv sync
uv run streamlit run streamlit_app.py
```

Edita `.env` antes de preguntar al modelo:

- Para **Ollama Cloud**: añade `OLLAMA_API_KEY` y deja `OLLAMA_BASE_URL=https://ollama.com`.
- Para **Ollama local**: usa `OLLAMA_BASE_URL=http://localhost:11434` y un modelo descargado con `ollama pull`.
- Para **Runpod/vLLM**: usa `LLM_PROVIDER=openai-compatible` y `OPENAI_BASE_URL=<endpoint>`.

## Langfuse local

Langfuse ya está incluido en esta carpeta. Para levantarlo:

```bash
cd 04-LLMOps-Langfuse
docker compose up -d
open http://localhost:3000
```

El compose crea automáticamente un proyecto de demo:

```text
Proyecto: sesion-04-rag
Usuario:  demo@example.com
Password: demo1234
Public key: pk-lf-sesion-04-demo
Secret key: lf-secret-sesion-04-demo
```

La app RAG ya trae estos valores en `.env.example`. Copia `.env.example` a `.env`, arranca Streamlit y activa **Enviar trazas a Langfuse**.

## Evals de clase

```bash
cd 04-LLMOps-Langfuse/ejemplo-langfuse-rag
uv run python evaluacion_retrieval.py
uv run python evaluacion_generacion.py --prompt-name rag-basico
uv run python evaluacion_comparativa.py
```

La idea central: **un cambio de prompt, documentos, retrieval o modelo es un cambio de producto; trazas + evals son cómo dejamos de adivinar.**
