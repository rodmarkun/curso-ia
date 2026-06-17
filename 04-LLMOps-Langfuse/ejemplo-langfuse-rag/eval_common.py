"""Utilidades compartidas por las dos evaluaciones (retrieval y generación).

Mantiene en un solo sitio:
- la carga de `eval_cases.json`,
- cómo se interpreta `expected_sources` (fichero -> fragmento(s)),
- el cálculo de aciertos de fragmento y de fichero,
- el render de tablas en texto plano para stdout.

Cada caso tiene la forma mínima:
    {"id": ..., "question": ..., "expected_sources": {"fichero": [frag, ...]}}
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "eval_cases.json"


def load_cases() -> list[dict]:
    """Carga los casos de evaluación desde eval_cases.json."""

    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def expected_pairs(case: dict) -> set[tuple[str, int]]:
    """Conjunto de pares (fichero, fragmento) esperados para un caso."""

    pairs: set[tuple[str, int]] = set()
    for source, fragments in (case.get("expected_sources") or {}).items():
        for fragment in fragments:
            pairs.add((source, int(fragment)))
    return pairs


def expected_files(case: dict) -> set[str]:
    """Conjunto de ficheros esperados (sin tener en cuenta el fragmento)."""

    return set((case.get("expected_sources") or {}).keys())


def format_expected(case: dict) -> str:
    """Representación legible `fichero#frag1,frag2` para mostrar en tablas."""

    parts = []
    for source, fragments in (case.get("expected_sources") or {}).items():
        frags = ",".join(str(f) for f in fragments)
        parts.append(f"{source}#{frags}")
    return " | ".join(parts) or "—"


def format_pairs(pairs: list[tuple[str, int]]) -> str:
    """Representación legible de una lista de pares (fichero, fragmento)."""

    return " | ".join(f"{source}#{frag}" for source, frag in pairs) or "—"


def pct(numerator: int, denominator: int) -> str:
    """Porcentaje formateado, tolerante a denominador 0."""

    if not denominator:
        return "0.0%"
    return f"{numerator / denominator * 100:.1f}%"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render minimalista de una tabla en texto plano (sin dependencias)."""

    columns = [headers] + [[str(cell) for cell in row] for row in rows]
    widths = [max(len(col[i]) for col in columns) for i in range(len(headers))]

    def render_row(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    separator = "-+-".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines += [render_row([str(cell) for cell in row]) for row in rows]
    return "\n".join(lines)
