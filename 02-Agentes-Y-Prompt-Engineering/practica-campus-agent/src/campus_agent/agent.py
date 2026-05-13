import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from campus_agent.tools import build_tools


SYSTEM_PROMPT = """
Eres un asistente del Campus Virtual.
Responde de forma clara y breve.
Todavia no tienes herramientas conectadas, asi que no inventes notas,
matriculas ni datos personales. Si te piden datos academicos concretos,
explica que necesitas una herramienta de consulta para verificarlos.
"""


def build_llm() -> ChatOllama:
    load_dotenv()
    model = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
    base_url = os.environ.get("OLLAMA_HOST", "https://ollama.com")
    api_key = os.environ.get("OLLAMA_API_KEY")

    client_kwargs = {}
    if api_key:
        client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

    return ChatOllama(
        model=model,
        base_url=base_url,
        client_kwargs=client_kwargs,
    )


def build_agent():
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=build_tools(),
        system_prompt=SYSTEM_PROMPT,
    )
