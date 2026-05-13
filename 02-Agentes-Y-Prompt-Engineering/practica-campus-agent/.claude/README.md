# Configuracion local de Claude Code

Esta carpeta contiene reglas locales para trabajar con la practica.

Puntos importantes:

- `.env` no debe leerse ni mostrarse.
- `.env.example` si puede usarse para documentar configuracion.
- Las iteraciones deben ser pequenas y verificables con tests.
- La seguridad vive en codigo: login, permisos, filtrado de tools y validaciones no dependen del prompt.
