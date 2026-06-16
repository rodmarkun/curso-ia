# INSTRUCCIONES-LANGFUSE

Guía rápida para levantar **Langfuse local** desde este repo para la Sesión 04.

La idea es que no tengas que descargar ni preparar nada aparte: el `docker-compose.yml` ya está dentro de esta carpeta.

---

## 1. Requisitos

Necesitas tener instalado y funcionando:

- Docker Desktop o Docker Engine.
- Docker Compose, normalmente incluido como `docker compose`.

Compruébalo:

```bash
docker --version
docker compose version
```

Si Docker Desktop está instalado pero los comandos fallan, abre Docker Desktop y espera a que termine de arrancar.

---

## 2. Levantar Langfuse

Desde la raíz del repo del curso:

```bash
cd 04-LLMOps-Langfuse
docker compose up -d
```

Eso levanta:

- Langfuse web;
- Langfuse worker;
- Postgres;
- ClickHouse;
- Redis;
- MinIO.

Comprueba que los servicios están arrancando:

```bash
docker compose ps
```

Abre Langfuse:

```bash
open http://localhost:3000
```

En Linux:

```bash
xdg-open http://localhost:3000
```

También puedes abrirlo manualmente en el navegador:

```text
http://localhost:3000
```

---

## 3. Usuario, proyecto y claves de demo

El `docker-compose.yml` crea automáticamente un proyecto de demo.

Datos de acceso:

```text
URL:      http://localhost:3000
Usuario:  demo@example.com
Password: demo1234
Proyecto: sesion-04-rag
```

Claves del proyecto demo:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-sesion-04-demo
LANGFUSE_SECRET_KEY=lf-secret-sesion-04-demo
```

Estas claves son solo para clase/local. No las uses en producción.

---

## 4. Conectar la app RAG de la sesión

En otra terminal, desde la raíz del repo:

```bash
cd 04-LLMOps-Langfuse/ejemplo-langfuse-rag
cp .env.example .env
uv sync
uv run streamlit run streamlit_app.py
```

Abre Streamlit si no se abre solo:

```bash
open http://localhost:8501
```

En Linux:

```bash
xdg-open http://localhost:8501
```

En la barra lateral de Streamlit, activa:

```text
Enviar trazas a Langfuse
```

Haz una pregunta de prueba:

```text
¿Qué prácticas de laboratorio tiene Sistemas Digitales?
```

Después entra en Langfuse:

```text
http://localhost:3000
```

Y busca la traza en el proyecto `sesion-04-rag`.

---

## 5. Qué debe tener el `.env` de la app

El fichero `ejemplo-langfuse-rag/.env.example` ya viene preparado. Si tienes que revisarlo, estas son las líneas importantes:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-sesion-04-demo
LANGFUSE_SECRET_KEY=lf-secret-sesion-04-demo
LANGFUSE_RELEASE=sesion-04-local
```

Si ya tenías un `.env` antiguo de otra prueba, comprueba las variables:

```bash
cd 04-LLMOps-Langfuse/ejemplo-langfuse-rag
grep LANGFUSE .env
```

Si no coinciden, edítalas o vuelve a copiar el ejemplo:

```bash
cp .env.example .env
```

Luego reinicia Streamlit.

---

## 6. Comprobar que Langfuse responde

Desde `04-LLMOps-Langfuse`:

```bash
curl -s -L -o /tmp/langfuse_home.html \
  -w 'code=%{http_code}\nurl=%{url_effective}\n' \
  http://localhost:3000
```

Un resultado correcto suele ser `code=200` o una redirección hacia login/onboarding.

Ver logs si algo va mal:

```bash
cd 04-LLMOps-Langfuse
docker compose logs -f langfuse-web
```

Ver todos los servicios:

```bash
docker compose ps
```

---

## 7. Problemas frecuentes

### El puerto 3000 está ocupado

Comprueba qué lo usa:

```bash
lsof -i :3000
```

Para cerrar el proceso:

```bash
kill -9 <PID>
```

Después vuelve a levantar Langfuse:

```bash
cd 04-LLMOps-Langfuse
docker compose up -d
```

### Docker dice que no puede conectar con el daemon

Abre Docker Desktop y espera a que esté listo. Luego prueba:

```bash
docker ps
```

### Langfuse tarda mucho en arrancar

Es normal la primera vez porque descarga imágenes y arranca varias bases de datos. Mira el estado:

```bash
cd 04-LLMOps-Langfuse
docker compose ps
docker compose logs -f langfuse-web
```

### No aparecen trazas en Langfuse

1. Revisa que Langfuse esté levantado:

```bash
cd 04-LLMOps-Langfuse
docker compose ps
```

2. Revisa `.env` de la app:

```bash
cd 04-LLMOps-Langfuse/ejemplo-langfuse-rag
grep LANGFUSE .env
```

3. Comprueba que `LANGFUSE_ENABLED=true`.
4. Reinicia Streamlit después de cambiar `.env`.
5. En Streamlit, activa **Enviar trazas a Langfuse**.
6. Haz otra pregunta.

### Cambié claves/proyecto y ahora está raro

Los datos de Langfuse se guardan en volúmenes Docker. Si quieres empezar de cero:

```bash
cd 04-LLMOps-Langfuse
docker compose down -v
docker compose up -d
```

Esto borra los datos locales de Langfuse de esta demo.

---

## 8. Parar Langfuse

Parar conservando datos:

```bash
cd 04-LLMOps-Langfuse
docker compose down
```

Parar y borrar datos:

```bash
cd 04-LLMOps-Langfuse
docker compose down -v
```

---

## 9. Resumen ultrarrápido

Terminal 1 — Langfuse:

```bash
cd 04-LLMOps-Langfuse
docker compose up -d
open http://localhost:3000
```

Terminal 2 — app RAG:

```bash
cd 04-LLMOps-Langfuse/ejemplo-langfuse-rag
cp .env.example .env
uv sync
uv run streamlit run streamlit_app.py
```

Pregunta de prueba:

```text
¿Qué prácticas de laboratorio tiene Sistemas Digitales?
```
