# Guion de demo en vivo — Sesión 04 LLMOps + Langfuse

"Del prototipo al servicio de IA".

Este guion mantiene el orden simple de la demo. Debajo de cada paso dejo los comandos que puedo necesitar en clase.

---

## 0. Preparación antes de empezar

Entrar en la carpeta de la práctica y comprobar que todo instala:

```bash
cd 04-LLMOps-Langfuse/ejemplo-langfuse-rag
cp .env.example .env
uv sync
```

Comprobar que no tengo nada raro ocupando puertos:

```bash
lsof -i :3000
lsof -i :8501
```

Si necesito matar algo en un puerto concreto:

```bash
kill -9 <PID>
```

Comprobar Docker:

```bash
docker --version
docker compose version
```

Comprobar Ollama local si lo voy a usar luego:

```bash
ollama --version
ollama list
```

---

## 1. Levantamos el RAG del otro día

Primero, levantamos el RAG del otro día y vemos que teniendo cientos de usuarios sería complicado saber qué falla, cómo estamos respondiendo, qué documentos se recuperan, etc. Aquí entra Langfuse.

Arrancar Streamlit:

```bash
cd 04-LLMOps-Langfuse/ejemplo-langfuse-rag
uv run streamlit run streamlit_app.py
```

Si quiero fijar puerto:

```bash
uv run streamlit run streamlit_app.py --server.port 8501
```

Abrir la app:

```bash
open http://localhost:8501
```

Preguntas para enseñar el RAG:

```text
¿Qué prácticas de laboratorio tiene Sistemas Digitales?
```

```text
¿Qué criterios se usan en la rúbrica del proyecto de robótica?
```

```text
¿Qué equipos del inventario necesitan mantenimiento?
```

Cosas que enseño en la UI:

- respuesta;
- fragmentos recuperados desde Chroma;
- prompt enviado al modelo;
- traza local JSON;
- latencia, tokens estimados, fuentes y similitud.

---

## 2. Desplegamos Langfuse con Docker en el puerto 3000

Enseñamos CÓMO se despliega Langfuse con Docker. Todo está ya dentro de la carpeta de la sesión, así que no descargamos nada durante la clase.

Ir a la carpeta de la sesión:

```bash
cd 04-LLMOps-Langfuse
```

Ver que el compose está dentro del proyecto:

```bash
ls docker-compose.yml
```

Levantar Langfuse:

```bash
docker compose up -d
```

Ver servicios:

```bash
docker compose ps
```

Ver logs si tarda o falla:

```bash
docker compose logs -f langfuse-web
```

Comprobar por HTTP:

```bash
curl -s -L -o /tmp/langfuse_home.html \
  -w 'code=%{http_code}\nurl=%{url_effective}\n' \
  http://localhost:3000
```

Abrir Langfuse:

```bash
open http://localhost:3000
```

El `docker-compose.yml` crea un proyecto de demo automáticamente:

```text
Proyecto: sesion-04-rag
Usuario:  demo@example.com
Password: demo1234
Public key: pk-lf-sesion-04-demo
Secret key: lf-secret-sesion-04-demo
```

Conectar la app RAG con Langfuse:

```bash
cd ejemplo-langfuse-rag
cp .env.example .env
```

`.env.example` ya trae estas variables:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-sesion-04-demo
LANGFUSE_SECRET_KEY=lf-secret-sesion-04-demo
LANGFUSE_RELEASE=sesion-04-local
```

Si `.env` ya existía de otra demo, comprobarlo:

```bash
grep LANGFUSE .env
```

Reiniciar Streamlit después de cambiar `.env`.

---

## 3. Enviamos trazas reales a Langfuse

En Streamlit:

1. activar **Enviar trazas a Langfuse**;
2. hacer una pregunta;
3. abrir la traza local;
4. enseñar la sección **Herramienta llamada por el agente**;
5. entrar en Langfuse y enseñar la traza.

Conexión con la sesión 02: allí vimos que un modelo puede usar herramientas. Aquí el agente llama una herramienta sencilla antes de generar:

```text
tool_inventario_documentos
```

En Langfuse debe verse como una observación de tipo `tool`, antes de `retrieval` y `ollama_generation`. Su salida lista los documentos disponibles para responder.

Pregunta útil:

```text
¿Qué prácticas de laboratorio tiene Sistemas Digitales?
```

Abrir Langfuse:

```bash
open http://localhost:3000
```

Si no aparece nada, revisar:

```bash
cd 04-LLMOps-Langfuse
docker compose ps
docker compose logs --tail=100 langfuse-web
```

Y revisar `.env` de la app:

```bash
cd 04-LLMOps-Langfuse/ejemplo-langfuse-rag
grep LANGFUSE .env
```

---

## 4. Cambiar prompts con Langfuse

Cómo cambiar de prompts con Langfuse: al principio tengo un solo prompt de RAG básico. Después, añadimos una segunda versión que añade:

```text
Responde en el idioma del usuario.
Al principio de la respuesta añade: IDIOMA DETECTADO: <idioma del usuario>
```

En Langfuse:

1. ir a Prompt Management;
2. crear el prompt `rag-basico` pegando el contenido de `PROMPT.txt`;
3. guardarlo con el label `production`;
4. crear una nueva versión;
5. añadir esas dos líneas al principio;
6. dejar el label `production` en la versión que quiero usar.

En Streamlit:

- pulsar **Refrescar prompts desde Langfuse**;
- preguntar otra vez.

Preguntas para probar idioma:

```text
¿Qué criterios se usan en la rúbrica del proyecto de robótica?
```

```text
What lab equipment needs maintenance?
```

Si quiero comprobar desde terminal que la app ve Langfuse configurado:

```bash
grep 'LANGFUSE_ENABLED\|LANGFUSE_HOST\|LANGFUSE_PROMPT' .env
```

---

## 5. Cambiar config en los prompts de Langfuse

Cómo cambiar de config en los prompts de Langfuse: cambiamos el modelo que utilizamos para responder. También podemos cambiar temperatura, max tokens, contexto, `k`, etc.

Config ejemplo en Langfuse:

```json
{
  "model": "gpt-oss:120b",
  "temperature": 0,
  "num_ctx": 8192,
  "num_predict": 900,
  "reasoning": false,
  "k": 4
}
```

En Streamlit enseño:

- config de la prompt;
- modelo leído desde Langfuse;
- base URL;
- proveedor.

Punto importante para decir en clase: `temperature`, `num_ctx` y `num_predict` ya no se editan en Streamlit. Viven en Langfuse para que haya una sola fuente de verdad de la configuración del prompt.

Preguntar de nuevo y enseñar que la traza incluye config/modelo:

```text
¿Qué equipos del inventario necesitan mantenimiento?
```

---

## 6. Evaluación: retrieval

Pasamos a evaluación. Primero creamos una evaluación para ver cómo de correctos son los fragmentos que recuperamos de los documentos que tenemos.

Abrir casos de evaluación:

```bash
cd 04-LLMOps-Langfuse/ejemplo-langfuse-rag
code eval_cases.json
```

Si no quiero abrir editor, mostrarlo en terminal:

```bash
python -m json.tool eval_cases.json
```

Ejecutar evaluación de retrieval:

```bash
uv run python evaluacion_retrieval.py
```

Qué explico:

- no llama al LLM;
- es rápida y barata;
- comprueba si la fuente esperada aparece en top-k;
- si falla retrieval, cambiar el prompt no suele arreglarlo.

Si quiero guardar el resultado en un fichero:

```bash
uv run python evaluacion_retrieval.py > eval_retrieval_result.json
```

---

## 7. Evaluación: generación

Lo mismo para la calidad de la generación.

Ejecutar evaluación de generación:

```bash
uv run python evaluacion_generacion.py --prompt-name rag-basico
```

Si quiero enviar estas trazas a Langfuse:

```bash
uv run python evaluacion_generacion.py --prompt-name rag-basico --langfuse
```

Resultado que genera:

```bash
ls eval_results_*.json
```

Ver resultado:

```bash
python -m json.tool eval_results_rag-basico.json | less
```

Qué explico:

- aquí sí llama al modelo;
- comprueba términos esperados;
- comprueba términos prohibidos;
- comprueba que la fuente esperada haya llegado;
- mide latencia;
- estima tokens;
- crea trazas.

Casos que enseño:

```text
missing_info_profesor_telefono
security_change_grade
prompt_injection_documents
```

---

## 8. Comparar prompts / regresión

Después de cambiar el prompt, comparamos si el sistema mejora o empeora.

Ejecutar comparativa:

```bash
uv run python evaluacion_comparativa.py
```

Ver salida completa:

```bash
python -m json.tool eval_comparativa.json | less
```

Si quiero editar qué prompts comparar:

```bash
code evaluacion_comparativa.py
```

Ahí cambiar:

```python
PROMPTS_TO_COMPARE = ["rag-basico"]
```

por algo como:

```python
PROMPTS_TO_COMPARE = ["rag-basico", "rag-basico-idioma"]
```

Mensaje clave:

```text
Prompt nuevo ≠ producción automática. Primero pasa evals.
```

---

## 9. Modelos locales

Cómo desplegar modelos locales.

Primero, enseñamos la página LLMFitCheck.

Abrir LLMFitCheck:

```bash
open https://llmfitcheck.com
```

Explicar qué es la cuantización:

```text
FP16 / BF16: más calidad, más VRAM.
INT8: menos VRAM.
Q4: mucha menos VRAM, algo menos de calidad.
KV cache: memoria extra por contexto y concurrencia.
```

Vamos a Ollama y descargamos modelo. Hablamos con él en local y sin internet.

Comprobar Ollama:

```bash
ollama --version
ollama list
```

Descargar modelo:

```bash
ollama pull qwen2.5:7b-instruct
```

Hablar con él:

```bash
ollama run qwen2.5:7b-instruct
```

Probar desde API local:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b-instruct",
  "prompt": "Explícame qué es cuantización en dos frases.",
  "stream": false
}'
```

---

## 10. Conectar la aplicación RAG con el modelo local

Conectamos la aplicación de RAG con nuestro modelo recién descargado.

Editar `.env`:

```bash
cd 04-LLMOps-Langfuse/ejemplo-langfuse-rag
```

Variables:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

O hacerlo directamente en la UI:

```text
Proveedor: ollama
Base URL: http://localhost:11434
Modelo: qwen2.5:7b-instruct
```

Reiniciar Streamlit si cambié `.env`:

```bash
uv run streamlit run streamlit_app.py
```

Pregunta de prueba:

```text
¿Qué prácticas de laboratorio tiene Sistemas Digitales?
```

Comparar con Ollama Cloud:

```text
Calidad, latencia, privacidad, coste, mantenimiento.
```

---

## 11. OpenCode con modelo local

Mostramos cómo usar OpenCode con nuestro modelo recién descargado.

Comprobar si está instalado:

```bash
opencode --version
```

Si está configurado para Ollama/local, abrir en el repo:

```bash
cd 04-LLMOps-Langfuse/ejemplo-langfuse-rag
opencode
```

Prompt de demo:

```text
Explícame la arquitectura de rag_core.py y dónde se envían las trazas a Langfuse.
```

Si OpenCode no está preparado, no perder tiempo: explicar que la idea es que el mismo modelo local puede usarse también como asistente de coding.

---

## 12. Runpod

Vemos Runpod y cómo se utiliza.

Abrir Runpod:

```bash
open https://www.runpod.io/console
```

Qué enseño:

- elegir GPU;
- elegir template vLLM/OpenAI-compatible;
- elegir modelo;
- mirar coste/hora;
- exponer endpoint;
- apagar cuando terminemos.

Comando conceptual de vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype auto \
  --max-model-len 8192
```

Probar endpoint OpenAI-compatible cuando Runpod esté levantado:

```bash
export OPENAI_BASE_URL="https://<endpoint-runpod>"
export OPENAI_API_KEY="<token-si-aplica>"

curl "$OPENAI_BASE_URL/v1/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

---

## 13. Utilizar modelo de Runpod en la app

Desplegamos modelo en Runpod y utilizamos el modelo de Runpod para nuestras cosas.

Editar `.env`:

```bash
LLM_PROVIDER=openai-compatible
OPENAI_BASE_URL=https://<endpoint-runpod>
OPENAI_API_KEY=<token-si-aplica>
OLLAMA_MODEL=Qwen/Qwen2.5-7B-Instruct
```

O en la UI:

```text
Proveedor: openai-compatible
Base URL: https://<endpoint-runpod>
Modelo: Qwen/Qwen2.5-7B-Instruct
API key: <token-si-aplica>
```

Preguntar otra vez:

```text
¿Qué equipos del inventario necesitan mantenimiento?
```

Enseñar en Langfuse que cambia:

- modelo;
- proveedor/base_url en metadata;
- latencia;
- tokens estimados;
- respuesta.

Apagar Runpod al final para no gastar:

```text
Runpod console → Stop / Terminate pod
```

---

## 14. Cierre

Mensaje final:

```text
Prompt/model/document changes are production changes;
traces + evals are how we stop guessing.
```

Parar Langfuse conservando datos:

```bash
cd 04-LLMOps-Langfuse
docker compose down
```

Borrar Langfuse y empezar de cero otro día:

```bash
cd 04-LLMOps-Langfuse
docker compose down -v
```
