# CLAUDE.md

Instrucciones locales para trabajar en esta practica con Claude Code.

## Objetivo del proyecto

Construir iterativamente un agente de Campus con LangChain y Ollama Cloud. La base ya tiene un chat minimo con Ollama y prompt de sistema; el primer trabajo de la practica es anadir tools sobre `campus_data.json`.

El sistema usa `campus_data.json` como almacenamiento inicial y debe demostrar:

- lectura segura de datos academicos
- login con usuario y contrasena de demo
- permisos por rol y por alumno
- tools visibles segun rol
- memoria in-memory por conversacion
- tests que cubran casos permitidos y bloqueados

## Reglas de seguridad

- No leas, muestres, copies ni resumas `.env`.
- Usa `.env.example` para documentar variables de entorno.
- No incluyas claves de API, tokens, cookies ni secretos en codigo, tests, README o logs.
- El LLM nunca concede permisos. La sesion del servidor manda.
- Las tools deben comprobar permisos en codigo antes de leer datos.
- Un alumno solo puede leer su propio `student_id`.
- Un admin puede leer cualquier expediente.
- Si una accion no esta clara, falla de forma cerrada: no mostrar datos.

## Forma de trabajar

- Haz cambios pequenos y ejecuta tests despues de cada iteracion.
- Mantén la codebase simple; no anadas frameworks hasta que una iteracion lo pida.
- Prefiere funciones pequenas y modelos Pydantic explicitos.
- No reescribas todo el proyecto para resolver una iteracion.
- Cuando cambies tools, login, permisos o memoria, anade o actualiza tests.

## Comandos utiles

```bash
uv run pytest
uv run python -m campus_agent.cli
```

## Iteraciones esperadas

1. Tools de lectura.
2. Roles, login y filtrado de tools.
3. Memoria in-memory.
4. Tests de seguridad minimos.
