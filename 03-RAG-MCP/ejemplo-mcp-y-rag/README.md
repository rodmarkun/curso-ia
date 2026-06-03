# Ejemplo MCP y RAG

Ejemplo mínimo para clase: una aplicación RAG con **Streamlit**, **Chroma persistente** y un servidor **MCP** muy simple.

El objetivo principal es que los estudiantes vean el flujo real:

```text
documentos → chunks → embeddings → Chroma vector DB → retrieval → prompt → Ollama Cloud
```

Incluye:

- `streamlit_app.py`: interfaz web sencilla para preguntar a los documentos.
- `rag_core.py`: carga documentos, calcula hashes, ingiere incrementalmente en Chroma, busca por similitud vectorial y llama al LLM.
- `mcp_server.py`: servidor MCP mínimo que expone el RAG como herramientas.
- `documentos-ejemplo/`: PDFs, CSV, Excel y JSON usados como base documental.
- `chroma_db/`: se crea automáticamente como base de datos vectorial persistente. Está ignorada por git.

## 1. Instalar con uv

> Recomendado: Python 3.11. El SDK oficial de MCP no está disponible para Python 3.9.

```bash
cd ejemplo-mcp-y-rag
uv sync
```

No hace falta activar el entorno virtual ni tener `streamlit` instalado globalmente. Usa siempre `uv run ...`.

## 2. Configurar Ollama Cloud

Crea una clave en:

<https://ollama.com/settings/keys>

Después:

```bash
cp .env.example .env
```

Edita `.env` y añade tu clave:

```bash
OLLAMA_API_KEY=ollama_xxx
OLLAMA_MODEL=gpt-oss:120b
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
DOCS_DIR=documentos-ejemplo
CHROMA_DIR=chroma_db
```

El ejemplo usa Ollama Cloud igual que la práctica anterior, mediante LangChain:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gpt-oss:120b",
    base_url="https://ollama.com",
    temperature=0,
)
```

Los embeddings se calculan localmente con SentenceTransformers para que la recuperación vectorial sea visible y barata. Chroma guarda esos vectores en disco.

## 3. Ejecutar la app Streamlit

La primera ejecución descargará el modelo de embeddings y creará `chroma_db/`.

```bash
uv run streamlit run streamlit_app.py
```

Prueba preguntas como:

- ¿Qué documentos hay disponibles y de qué tratan?
- ¿Qué prácticas de laboratorio tiene Sistemas Digitales?
- ¿Qué criterios se usan en la rúbrica del proyecto de robótica?
- ¿Qué equipos del inventario necesitan mantenimiento?

## 4. Ingesta incremental en Chroma

La ingesta ocurre automáticamente al arrancar `SimpleRAG` desde Streamlit o MCP.

Para cada documento:

1. Se calcula un hash SHA-256 del archivo.
2. Se comprueba si Chroma ya tiene chunks de ese documento con el mismo hash.
3. Si el hash coincide y se está usando el mismo modelo de embeddings, no se reingiere.
4. Si el documento es nuevo, se añade.
5. Si el documento cambió, o cambió el modelo de embeddings, se eliminan sus chunks antiguos y se añaden los nuevos.
6. Si un documento fue eliminado de la carpeta, se eliminan sus chunks de Chroma.

Así puedes añadir un nuevo PDF/CSV/XLSX/JSON a `documentos-ejemplo/` y al reiniciar o pulsar **Sincronizar documentos con Chroma** solo se ingiere ese nuevo documento.

## 5. Ejecutar el servidor MCP

En otra terminal:

```bash
cd ejemplo-mcp-y-rag
uv run python mcp_server.py
```

Este servidor usa transporte `stdio`, que es el formato típico para conectarlo a un cliente MCP.

Expone estas herramientas:

| Herramienta | Qué hace |
|---|---|
| `listar_documentos` | Devuelve los nombres de documentos cargados. |
| `estado_ingesta` | Devuelve ruta de documentos, ruta de Chroma, número de chunks y última sincronización. |
| `sincronizar_documentos` | Revisa la carpeta y añade/actualiza/elimina documentos en Chroma según haga falta. |
| `buscar_en_documentos` | Recupera fragmentos relevantes desde Chroma sin llamar al LLM. |
| `responder_con_rag` | Recupera contexto desde Chroma y responde con Ollama Cloud. |

También expone un recurso:

| Recurso | Qué contiene |
|---|---|
| `documentos://lista` | Lista de documentos disponibles. |

## 6. Añadir el servidor MCP a Claude Code

Desde la carpeta del proyecto, puedes registrar este servidor MCP en Claude Code así:

```bash
cd /Users/pablorodriguez/Projects/curso-ia/03-RAG-MCP/ejemplo-mcp-y-rag
claude mcp add -s local campus-rag-demo -- uv --directory "$PWD" run python mcp_server.py
```

Comprueba que Claude Code lo ve con:

```bash
claude mcp list
```

Después abre Claude Code en este proyecto:

```bash
claude
```

Dentro de Claude Code puedes revisar los servidores MCP con:

```text
/mcp
```

La opción `-s local` guarda esta configuración solo para este proyecto, normalmente en `.claude/settings.local.json`, que es personal y no debería subirse a git.

## 7. Ejemplo de configuración MCP

Un cliente MCP podría configurarlo así, ajustando la ruta al proyecto. Con `uv`, puedes usar `uv` como comando para no depender de activar un virtualenv:

```json
{
  "mcpServers": {
    "campus-rag-demo": {
      "command": "uv",
      "args": ["--directory", "/ruta/al/proyecto/ejemplo-mcp-y-rag", "run", "python", "mcp_server.py"],
      "env": {
        "OLLAMA_API_KEY": "ollama_xxx",
        "OLLAMA_MODEL": "gpt-oss:120b",
        "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "DOCS_DIR": "/ruta/al/proyecto/ejemplo-mcp-y-rag/documentos-ejemplo",
        "CHROMA_DIR": "/ruta/al/proyecto/ejemplo-mcp-y-rag/chroma_db"
      }
    }
  }
}
```

## 8. Qué conceptos enseña

### Vector DB / Chroma

Chroma es la base de datos vectorial. Guarda:

- el texto de cada chunk,
- el vector de embedding,
- metadatos como `source`, `chunk_id` y `file_hash`,
- todo persistido en `chroma_db/`.

### Vector retrieval

La búsqueda no es por palabras clave. Chroma convierte la pregunta en un vector y compara ese vector con los vectores guardados:

```text
query → embedding → búsqueda por similitud → chunks más cercanos
```

### RAG

1. **Carga**: leer documentos locales.
2. **Chunking**: dividir documentos en fragmentos.
3. **Embeddings**: convertir cada fragmento en un vector numérico.
4. **Indexación**: guardar los vectores en Chroma.
5. **Retrieval**: buscar chunks similares a la pregunta.
6. **Augmentation**: construir un prompt con el contexto recuperado.
7. **Generation**: pedir al LLM que responda usando ese contexto.

### MCP

MCP convierte capacidades de una aplicación en herramientas estándar que un agente puede descubrir y llamar.

En este ejemplo:

- Streamlit es la interfaz humana.
- `rag_core.py` es la lógica reutilizable.
- Chroma es la base de datos vectorial persistente.
- MCP es otra interfaz sobre la misma lógica, pensada para clientes/agentes.

## 9. Estructura

```text
ejemplo-mcp-y-rag/
├── documentos-ejemplo/
├── rag_core.py
├── streamlit_app.py
├── mcp_server.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
