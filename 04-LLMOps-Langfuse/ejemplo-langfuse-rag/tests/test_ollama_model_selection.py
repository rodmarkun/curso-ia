import json
import sys
from pathlib import Path
from urllib.error import URLError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core import (
    OLLAMA_CLOUD_BASE_URL,
    OLLAMA_LOCAL_BASE_URL,
    get_ollama_base_url_for_mode,
    validate_local_ollama_model,
)


def test_ollama_mode_resolves_cloud_and_local_base_urls():
    assert get_ollama_base_url_for_mode("cloud", current_base_url="http://ignored:11434") == OLLAMA_CLOUD_BASE_URL
    assert get_ollama_base_url_for_mode("local", current_base_url="https://ollama.com") == OLLAMA_LOCAL_BASE_URL


def test_ollama_mode_preserves_custom_remote_base_url():
    assert (
        get_ollama_base_url_for_mode("custom", current_base_url="http://192.168.1.50:11434")
        == "http://192.168.1.50:11434"
    )


def test_validate_local_ollama_model_accepts_installed_model(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"models": [{"name": "qwen2.5:7b-instruct"}]}).encode()

    monkeypatch.setattr("rag_core.urlopen", lambda request, timeout: FakeResponse())

    validate_local_ollama_model("qwen2.5:7b-instruct", "http://localhost:11434")


def test_validate_local_ollama_model_accepts_latest_shorthand(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"models": [{"name": "gemma4:latest"}]}).encode()

    monkeypatch.setattr("rag_core.urlopen", lambda request, timeout: FakeResponse())

    validate_local_ollama_model("gemma4", "http://localhost:11434")


def test_validate_local_ollama_model_raises_clear_error_when_missing(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"models": [{"name": "llama3.1:8b"}]}).encode()

    monkeypatch.setattr("rag_core.urlopen", lambda request, timeout: FakeResponse())

    with pytest.raises(RuntimeError, match="Modelo Ollama local no instalado.*ollama pull qwen2.5:7b-instruct"):
        validate_local_ollama_model("qwen2.5:7b-instruct", "http://localhost:11434")


def test_validate_local_ollama_model_raises_clear_error_when_server_unreachable(monkeypatch):
    def raise_url_error(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("rag_core.urlopen", raise_url_error)

    with pytest.raises(RuntimeError, match="No se pudo conectar con Ollama local.*ollama serve"):
        validate_local_ollama_model("qwen2.5:7b-instruct", "http://localhost:11434")
