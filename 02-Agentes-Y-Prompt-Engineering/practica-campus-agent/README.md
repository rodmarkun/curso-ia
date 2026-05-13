# Practica Campus Agent

Codebase base para practicar agentes con LangChain, Ollama Cloud, tools, permisos y memoria.

La idea es empezar pequeno y evolucionar con Claude Code. Esta carpeta ya trae un chat minimo con Ollama Cloud y un prompt de sistema. No trae tools, login, roles ni memoria: eso se anade por iteraciones.

## Preparacion

```bash
cd practica-campus-agent
cp .env.example .env
uv sync
```

Edita `.env` y anade tu clave de Ollama Cloud.

## Comandos

```bash
uv run pytest
uv run python -m campus_agent.cli
```

El CLI inicial es un chat normal. Si preguntas por notas o expedientes, el agente debe reconocer que todavia no tiene tools para consultar datos reales.

## Iteraciones obligatorias

1. Tools de lectura: define tools para consultar `campus_data.json` y conectalas al agente inicial.
2. Roles y login: anade usuarios con contrasena, distingue `alumno` y `admin`, y filtra las tools disponibles por rol.
3. Memoria in-memory: anade memoria de conversacion con un checkpointer en memoria.
4. Tests de seguridad: cubre login, permisos por rol, restricciones de tools y prompt injection basico.

## Roles esperados

- `alumno`: solo puede leer sus propios datos.
- `admin`: puede leer datos de cualquier alumno.

El LLM no decide permisos. Los permisos se aplican en codigo.
