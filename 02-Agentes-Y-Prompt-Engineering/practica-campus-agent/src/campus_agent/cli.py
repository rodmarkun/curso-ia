from campus_agent.agent import build_agent


def main() -> None:
    agent = build_agent()
    print("Chat del Campus Virtual")
    print("Escribe 'salir' para terminar.")
    print("Nota: la codebase inicial aun no tiene tools ni memoria.")

    while True:
        message = input("Tu: ").strip()
        if message.lower() in {"salir", "exit", "quit"}:
            break
        if not message:
            continue

        result = agent.invoke({"messages": [{"role": "user", "content": message}]})
        print("Agente:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
