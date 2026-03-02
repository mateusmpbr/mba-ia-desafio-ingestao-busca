from search import search_prompt


def main():
    print("Faça sua pergunta (digite 'sair' para encerrar):\n")
    while True:
        try:
            pergunta = input("PERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando chat.")
            break

        if not pergunta:
            continue

        if pergunta.lower() in ("sair", "exit", "quit"):
            print("Encerrando chat.")
            break

        try:
            resposta = search_prompt(pergunta)
        except Exception as e:
            print("Erro ao processar a pergunta:", e)
            continue

        print("\nRESPOSTA:")
        print(resposta)
        print("\n---\n")


if __name__ == "__main__":
    main()